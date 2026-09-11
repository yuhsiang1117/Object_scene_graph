"""NavAgent: the control loop behind a single act(frame) -> action call.

What one step does, in order:

    floors.observe        which storey, and what height to band the costmap at
    costmap.update        depth -> occupancy for the storey the agent is on
    on_keyframe           detector -> object layer -> presence beliefs, and a
                          glance at every surface in plain view
    candidates.check      does any mapped track now clear the gates and become
                          a goal (presence, identity, evidence, reachability)
    dispatch              the handler for whichever state the agent is in

The state machine and the two literal actions live in `state.py`. Everything a
state does lives beside it:

    exploration/strategy.py   where to go next -- frontiers and mapped surfaces
                              competing under one index
    agent/candidate.py        what becomes a goal, and the pre-approach verify
    agent/approach.py         the terminal walk and the stop decision
    verification/absence.py   "I got there and it was not there", as evidence
    agent/floor_policy.py     storeys, portals and stairs (inert unless enabled)

What is left here is the wiring, the FSM state itself, and the few things every
state needs: the costmap, a plan, a path to follow, and a detection of the
target in the current frame. NavAgent is the only owner of FSM state -- `state`,
`_goal_xy`, `_current_path`, `_candidate_id`, `_target_obj_xy`, the deadline --
which is why the handlers read and write it through their `nav` back-reference
rather than each keeping a copy.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..core.profiler import Profiler
from ..core.config import resolve_navigation
from ..core.types import Detection, FrameData
from ..exploration.async_scorer import AsyncScorer
from ..exploration.ascent_selector import FrontierCommitState
from ..exploration.selector import frontier_goal_xy
from ..exploration.strategy import ExplorationStrategy, WorldView
from ..graph.scene_graph import ROOM_IDS_PER_FLOOR, SceneGraph
from ..mapping.costmap import PLANE, Costmap2D
from ..objects.object_layer import ObjectLayer
from ..perception.detector import Detector
from ..perception.vocabulary import target_vocabulary
from ..pipeline.beliefs import build_affinity_prior, build_presence_filter
from ..perception.keyframe import KeyframeSelector, KeyframeStore
from ..planning.controller import WaypointController, agent_heading
from ..planning.escape import ActionHistoryEscape
from ..planning.planner import PlanResult, StraightLinePlanner
from ..planning.voronoi_planner import HybridVoronoiPlanner
from ..verification.absence import AbsenceSensor
from ..verification.viewpoint import ViewpointPlanner
from .approach import ApproachPolicy
from .candidate import CandidatePolicy
from .floor_policy import FloorPolicy
from ..core.labels import normalize_label
from ..graph.containers import CONTAINER_CATEGORIES
from .state import FORWARD_ACTION, STOP_ACTION, TURN_ACTION, State

class NavAgent:
    def __init__(
        self,
        cfg,
        detector: Detector,
        scorer: AsyncScorer,
        verifier,  # always None in old-algorithm mode (no VLM verifier); kept for call-site compat
        target_category: str,
        keyframe_dir: Optional[str] = None,
        profiler: Optional[Profiler] = None,
        nav_fn=None,
        reachable_fn=None,
        pointnav=None,
        ranker=None,
        floor_planner=None,
        room_classifier=None,
        image_text=None,
        gate_itm=None,
        stair_segmenter=None,
    ) -> None:
        self.cfg = cfg
        self.detector = detector
        self.scorer = scorer
        self.verifier = verifier
        # Habitat-navmesh driving (old-stack alignment): nav_fn(goal_xy) returns
        # the next discrete action toward goal_xy on Habitat's navmesh, or None
        # when arrived/unreachable. When set, it replaces the costmap planner +
        # controller for all goal-following. The costmap is still built (for
        # frontier extraction / scene graph), only navigation switches.
        self._nav_fn = nav_fn
        self._reachable_fn = reachable_fn
        self.navigation = resolve_navigation(cfg.agent)
        self._use_navmesh = nav_fn is not None and self.navigation == "navmesh"
        if pointnav is None and self.navigation == "pointnav":
            from ..planning.pointnav_driver import build_pointnav

            pointnav = build_pointnav(cfg)
        self.pointnav = pointnav
        self._driver = nav_fn if self._use_navmesh else self.pointnav
        self._direct_approach = self._driver is not None
        self.ranker = ranker
        self.floor_planner = floor_planner
        self._floor_goal_dir = 0
        self.room_classifier = room_classifier
        self.image_text = image_text
        self.gate_itm = gate_itm
        self.stair_segmenter = stair_segmenter
        self.stair_detector = None
        self._down_look_every = int(getattr(cfg.agent, "down_look_every", 0))
        ascent_selector = str(getattr(cfg.exploration, "selector", "utility")) == "ascent"
        self.commit_state = (
            FrontierCommitState(quantise_m=cfg.exploration.frontier_dedup_m)
            if ascent_selector or bool(getattr(cfg.exploration, "frontier_commit", False))
            else None
        )
        self._approach_recheck = image_text is not None and bool(
            getattr(cfg.verification, "approach_recheck", False)
        )
        self._approach_recheck_thresh = float(
            getattr(cfg.verification, "approach_recheck_thresh", 0.0)
        )
        self._frontier_desc = str(getattr(cfg.exploration, "frontier_desc", "graph"))
        self.frontier_semantics = None
        if self._frontier_desc in ("frame", "frame_objects"):
            from ..exploration.frontier_semantics import FrontierSemantics

            self.frontier_semantics = FrontierSemantics(
                match_radius_m=float(
                    getattr(cfg.exploration, "frontier_desc_match_m", 1.0)
                ),
                fov_rad=np.radians(float(getattr(cfg.eval, "hfov_deg", 79.0))),
                max_range_m=float(cfg.mapping.max_range_m),
            )
        # Terminal-view verification mode: skip the pre-approach best_crop VLM
        # call and instead verify the live close-up frame at the STOP decision
        # (see _do_approach). Requires a verifier; no-op when verifier is None.
        self._terminal_verify = (
            verifier is not None and bool(cfg.verification.terminal)
        )
        self.profiler = profiler or Profiler()
        # Debug hook: if set, called with (frame, dets) every keyframe right
        # after the detections that feed object_layer.update() are computed
        # -- lets diagnostics observe exactly what the scene graph is built
        # from without duplicating the keyframe-timing logic. None by default
        # (zero cost, never called).
        self.on_keyframe_detections = None

        # Counters the whole stack writes into; FloorPolicy shares the dict, so
        # reset() clears it in place rather than rebinding it.
        self.stats: dict = {}
        # Which storey the agent is on, and one costmap per storey. Inert
        # unless floor.enabled -- see agent/floor_policy.py.
        value_map_factory = None
        if image_text is not None:
            from ..mapping.value_map import ValueMap2D

            value_map_factory = lambda costmap: ValueMap2D(
                costmap, max_depth_m=cfg.mapping.max_range_m
            )
        self.floors = FloorPolicy(
            cfg, self.stats, value_map_factory=value_map_factory
        )
        if bool(getattr(cfg.mapping, "multi_floor", False)):
            from ..mapping.stairs import StairDetector

            self.stair_detector = StairDetector(
                resolution_m=cfg.mapping.resolution_m,
                max_range_m=cfg.mapping.max_range_m,
                min_hits=int(getattr(cfg.exploration, "stair_min_hits", 1)),
                min_cells=int(getattr(cfg.exploration, "stair_min_cells", 25)),
                up_mode=str(getattr(cfg.agent, "stair_up_mode", "detector")),
            )
        self.selection_planner = (
            StraightLinePlanner()
            if self._driver is not None
            and not bool(cfg.agent.frontier_reachability_gate)
            else self.planner
        )
        self.object_layer = ObjectLayer(
            assoc_score_thresh=cfg.scene_graph.assoc_score_thresh,
            assoc_depth_gate_m=cfg.scene_graph.assoc_depth_gate_m,
            assoc_category_gate=cfg.scene_graph.assoc_category_gate,
            min_obs_for_refine=cfg.scene_graph.min_obs_for_refine,
            refine_every=cfg.scene_graph.refine_every,
            refine_max_center_move_m=cfg.scene_graph.refine_max_center_move_m,
            link_dist_m=cfg.scene_graph.link_dist_m,
            link_max_frame_gap=cfg.scene_graph.link_max_frame_gap,
            min_det_score=cfg.scene_graph.min_det_score,
            min_det_bbox_px=cfg.scene_graph.min_det_bbox_px,
            confirm_baseline_m=cfg.scene_graph.confirm_baseline_m,
            repeat_view_discount=cfg.scene_graph.repeat_view_discount,
            presence_filter=build_presence_filter(cfg),
            target_bypasses_gates=cfg.scene_graph.target_bypasses_gates,
            max_range_m=cfg.mapping.max_range_m,
            fp_disable_radius_m=cfg.scene_graph.fp_disable_radius_m,
            cloud_stride=cfg.scene_graph.cloud_stride,
            cloud_cap=cfg.scene_graph.cloud_cap,
        )
        self.scene_graph = SceneGraph(
            container_top_h_m=tuple(cfg.scene_graph.container_top_h_m),
            container_min_area_m2=cfg.scene_graph.container_min_area_m2,
            container_support_tol_m=cfg.scene_graph.container_support_tol_m,
            container_min_obs=cfg.scene_graph.container_min_obs,
            container_min_score=cfg.scene_graph.container_min_score,
            container_merge_m=cfg.scene_graph.container_merge_m,
        )
        self.keyframes = KeyframeStore(save_dir=keyframe_dir)
        self.kf_selector = KeyframeSelector(
            cfg.scene_graph.keyframe_trans_m, cfg.scene_graph.keyframe_rot_deg
        )
        self.controller = WaypointController(forward_m=cfg.agent.forward_m)
        self.viewpoint_planner = ViewpointPlanner(list(cfg.verification.ring_radii_m))
        # "I walked there and it was not there" as evidence, from two sensors
        # with different error rates. See verification/absence.py.
        self.absence = AbsenceSensor(cfg, verifier, self.profiler, self.stats)
        # The two state handlers. Each owns what only it uses; NavAgent stays
        # the single owner of FSM state.
        self.approach = ApproachPolicy(self)
        self.candidates = CandidatePolicy(self)
        # Where to go next -- frontiers and mapped surfaces under one index.
        # It owns everything an exploration round remembers; see
        # exploration/strategy.py.
        self.exploration = ExplorationStrategy(
            cfg, self.selection_planner, scorer, self.viewpoint_planner,
            build_affinity_prior(cfg), self.stats, self.profiler,
        )

        self.reset(target_category)

    # ------------------------------------------------------------------ floors

    @property
    def costmap(self) -> Costmap2D:
        """The occupancy map of the storey the agent is on.

        This property IS the multi-floor seam. Every consumer -- planner,
        frontier extractor, room segmenter, viewpoint planner, controller, the
        debug/top-down visualizers -- still receives a plain 2D `Costmap2D` and
        needs no knowledge that other floors exist.
        """
        return self.floors.costmap

    @property
    def planner(self) -> HybridVoronoiPlanner:
        """Planner scoped to the active storey."""
        return self.floor_layer.planner

    @property
    def floor_layer(self):
        return self.floors.layer

    @property
    def _floor_stack(self):  # graph/map_store.py snapshots the stack
        return self.floors.stack

    @property
    def _room_labels(self):
        return self.floors.room_labels

    @_room_labels.setter
    def _room_labels(self, labels) -> None:
        self.floors.room_labels = labels

    # Per-episode floor telemetry, recorded by eval/runner.py.
    @property
    def floor_log(self) -> list:
        return self.floors.floor_log

    @property
    def portal_log(self) -> list:
        return self.floors.portal_log

    @property
    def floor_y_drift(self) -> float:
        return self.floors.floor_y_drift

    @property
    def stair_regions(self) -> list:
        return self.floors.stair_regions

    # Per-episode exploration telemetry, recorded by eval/runner.py.
    @property
    def search_log_events(self) -> list:
        return self.exploration.search_log_events

    @property
    def frontier_select_log(self) -> list:
        return self.exploration.frontier_select_log

    @property
    def giveup_log(self) -> list:
        return self.exploration.giveup_log

    @property
    def _current_frontier(self):  # eval/runner.py's debug video draws it
        return self.exploration.current_frontier

    @_current_frontier.setter
    def _current_frontier(self, frontier) -> None:
        self.exploration.current_frontier = frontier

    @property
    def _frontier_reach_m(self) -> float:
        base = float(self.exploration.frontier_reach_m)
        return max(base, float(getattr(self.pointnav, "stop_radius", 0.0)))

    # Per-episode approach telemetry, recorded by eval/runner.py.
    @property
    def approach_bbox_log(self) -> list:
        return self.approach.bbox_log

    @property
    def approach_stop_reason(self):
        return self.approach.stop_reason

    @property
    def approach_diag(self) -> dict:
        return self.approach.diag

    @property
    def approach_retarget_log(self) -> list:
        return self.approach.retarget_log

    @property
    def goal_commit_log(self) -> list:
        return self.candidates.goal_commit_log

    @property
    def candidate_reject_log(self) -> list:
        return self.candidates.reject_log

    # ------------------------------------------------------------------ reset

    def reset(self, target_category: str) -> None:
        self.target = target_category
        self.state = State.INIT
        self.step_count = 0
        self._scan_steps_left = (
            int(round(360.0 / self.cfg.agent.turn_deg)) if self.cfg.agent.initial_scan else 0
        )
        self.floors.reset()
        # ``FloorStack.reset`` constructs a fresh layer (and therefore a fresh
        # planner).  Keep the ordinary selection path attached to that planner;
        # the straight-line selector is intentionally independent.
        if not isinstance(self.selection_planner, StraightLinePlanner):
            self.selection_planner = self.planner
        self._kf_count = 0
        self._current_path: Optional[np.ndarray] = None
        self._candidate_id: Optional[int] = None
        self._goal_xy: Optional[np.ndarray] = None
        self._last_action: Optional[str] = None
        self._goto_deadline = 10**9
        self._target_obj_xy: Optional[np.ndarray] = None
        self._goal_floor_y_cache: Optional[float] = None
        self._agent_xy: Optional[np.ndarray] = None
        self._target_cloud_xy: Optional[np.ndarray] = None
        self._room_votes: list = []
        self.stats.clear()
        self.stats.update({"plan_ok": 0, "plan_fail": 0, "select_none": 0, "select_ok": 0})
        # Phase 2 instrumentation (docs/DYNAMIC_SCENES.md): when the map STOPPED
        # believing in something, and what it believed at the moment it
        # committed to a goal. Belief latency and stale-goal rate are computed
        # from these two logs plus the relocation step the env records.
        self.presence_events: List[dict] = []
        self._disbelieved: set = set()
        self.state_log = []
        self.approach.reset()
        self.candidates.reset()
        self.kf_selector.reset()
        self.controller.reset()
        self.exploration.reset()
        self._escape = ActionHistoryEscape(int(self.cfg.agent.escape_window))
        self._progress_ref_step = 0
        self._progress_ref_xy = np.zeros(2)
        self._frontier_ref_dist = None
        self._approach_start_step = 0
        self._terminal_last_xy = None
        self._terminal_min_d = float("inf")
        self._terminal_stalls = 0
        self._approach_itm_max = 0.0
        self._approach_itm_n = 0
        self.approach_recheck_max = None
        self._last_itm = 0.0
        self._climb_carrot = bool(getattr(self.cfg.agent, "climb_carrot", False))
        self._carrot_xy = None
        self._carrot_disable_end = False
        self._climb_last_dist = None
        self._climb_paused_steps = 0
        self._pitch_ticks = 0
        self._last_down_look_step = -(10 ** 9)
        if self.commit_state is not None:
            self.commit_state.reset()
        if self.frontier_semantics is not None:
            self.frontier_semantics.reset()
        self.detector.set_vocabulary(
            target_vocabulary(self.target, self.cfg.detector.vocabulary)
        )
        self.object_layer.set_target(self.target)
        self.object_layer.keep_cloud_labels = {self.target}
        if self.pointnav is not None:
            self.pointnav.reset()
        for component in (
            self.ranker, self.floor_planner, self.room_classifier, self.image_text
        ):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()

    def rearm(self, max_steps: int) -> None:
        """Give the agent another attempt without giving it a new map.

        Everything learned survives -- presence beliefs, searched surfaces,
        objects mapped along the way -- because that carry-over is the whole
        point of retrying, and it is what separates this from resetting. Only
        the navigation state goes back: the committed candidate is released and
        the agent returns to EXPLORE with its remaining step budget.

        The belief work belongs to the caller (eval/attempts.py), because what a
        failed attempt is WORTH is a protocol question, not an agent one.
        """
        self._candidate_id = None
        self._target_obj_xy = None
        self._goal_xy = None
        self._current_path = None
        self.approach.at_viewpoint = False
        self.approach.scan_turns_left = 0
        self.approach.scan_expected = 0
        self.approach.stop_reason = None
        self.state = State.EXPLORE
        self._goto_deadline = self.step_count + int(max_steps)
        self.stats["attempts"] = self.stats.get("attempts", 1) + 1

    # ------------------------------------------------------------------- act

    def act(self, frame: FrameData) -> str:
        self.step_count += 1
        prev_state = self.state
        with self.profiler.timeit("control_loop"):
            action = self._act_inner(frame)
        if self._escape.window > 0 and not (
            action == STOP_ACTION and self.state is State.DONE
        ):
            action = self._escape(action)
        if self.state != prev_state:
            self.state_log.append((self.step_count, self.state.value))
        self._last_action = action
        return action

    def _act_inner(self, frame: FrameData) -> str:
        self._agent_xy = frame.camera_position[list(PLANE)].copy()
        if self.pointnav is not None:
            self.pointnav.observe(frame)
        floor_y = self.floors.observe(frame, self.step_count)
        # ExplorationStrategy is deliberately floor-agnostic; repoint its seam
        # whenever the active FloorLayer changes.
        self.exploration.planner = self.planner
        if not isinstance(self.selection_planner, StraightLinePlanner):
            self.selection_planner = self.planner
        self.exploration.planner = self.selection_planner
        standing_y = float(frame.camera_position[1] - self.cfg.agent.camera_height)
        self._floor_y = float(floor_y)
        self._off_plane_m = abs(standing_y - self._floor_y)
        reject_m = float(getattr(self.cfg.mapping, "floor_reject_m", 0.0))
        multi_floor = bool(
            getattr(self.cfg.mapping, "multi_floor", False)
            or getattr(self.cfg.floor, "per_floor_costmap", False)
        )
        off_map = (
            self.floors.on_stairs if multi_floor
            else reject_m > 0.0 and self._off_plane_m > reject_m
        )

        if off_map:
            self.stats["frames_off_plane"] = self.stats.get("frames_off_plane", 0) + 1
        else:
            with self.profiler.timeit("costmap"):
                self.costmap.update(
                    frame,
                    floor_y=floor_y,
                    obstacle_low=self.cfg.mapping.obstacle_low_m,
                    obstacle_high=self.cfg.mapping.obstacle_high_m,
                    max_range=self.cfg.mapping.max_range_m,
                    stride=self.cfg.mapping.depth_stride,
                )
            if self.image_text is not None:
                self._update_value_map(frame, self.floor_layer)
            # Tests and external callers may inject an image-text scorer
            # without enabling the semantic value map. Preserve that legacy
            # approach-recheck path in that case; when a value map exists the
            # single score above already feeds both mechanisms.
            if (
                self.state is State.APPROACH
                and self.image_text is not None
                and self.floor_layer.value_map is None
            ):
                scores = self.image_text.score(
                    frame.rgb, [self.target.replace("_", " ")]
                )
                if len(scores):
                    self._last_itm = float(scores[0])
                    self._approach_itm_max = max(
                        self._approach_itm_max, self._last_itm
                    )
                    self._approach_itm_n += 1
        self.controller.observe_progress(
            frame.T_wc, self._last_action, self.costmap, self.step_count
        )
        if self.controller.stuck:
            self.controller.stuck = False
            self._current_path = None  # force replan

        if self.kf_selector.is_keyframe(frame.T_wc):
            self._on_keyframe(frame)
            if self.cfg.exploration.search_posterior:
                self.exploration.glance(self._world(frame))

        down_look = self._down_look(frame, self.floor_layer, off_map)
        if down_look is not None:
            return down_look

        # Candidate target check happens in every state except terminal ones
        if self.state in (State.INIT, State.EXPLORE, State.GOTO_FRONTIER):
            self.candidates.check(frame.camera_position[list(PLANE)])

        if self.state == State.INIT:
            if self._scan_steps_left > 0:
                self._scan_steps_left -= 1
                return TURN_ACTION
            self.state = State.EXPLORE

        if self.state == State.EXPLORE:
            pursuing = self.floors.pursuit_ok(
                frame, self.step_count, self._goto_deadline
            )
            if pursuing and self._goal_xy is not None:
                self.state = State.GOTO_FRONTIER  # resume the climb
            else:
                # Look at the surface before the selection round scores it
                # searched -- the selection round retires it on arrival, and
                # the belief update should rest on a frame that shows it.
                facing = self.exploration.face_surface(self._world(frame))
                if facing is not None:
                    return facing
                self._explore(frame)
            if self.state == State.EXPLORE:  # nothing selectable
                return TURN_ACTION  # keep looking around; map will grow

        if self.state == State.GOTO_FRONTIER:
            reselect = int(getattr(self.cfg.exploration, "reselect_every", 0))
            if reselect > 0 and self.step_count % reselect == 0:
                self._select_new_frontier(frame)
            if self.exploration.maybe_give_up(
                self._world(frame),
                portal_ok=self.floors.pursuing
                and self.floors.pursuit_ok(frame, self.step_count, self._goto_deadline),
            ):
                self._current_path = None
                self.state = State.EXPLORE
                return self._act_inner_post_transition(frame)
            action = self._follow_path(frame)
            if action is not None:
                return action
            self.exploration.current_frontier = None
            self.state = State.EXPLORE
            return self._act_inner_post_transition(frame)

        if self.state == State.GOTO_VERIFY_VIEW:
            # Same terminal semantics as GOTO_TARGET: with discrete actions
            # the agent rarely lands exactly on the viewpoint — verify once
            # we are near it, the path is consumed, or the deadline passes.
            agent_xy = frame.camera_position[list(PLANE)]
            near_view = (
                self._goal_xy is not None
                and np.linalg.norm(agent_xy - self._goal_xy) < 0.35
            )
            if near_view or self.step_count > self._goto_deadline:
                self.state = State.VERIFYING
            else:
                action = self._follow_path(frame)
                if action is not None:
                    return action
                self.state = State.VERIFYING

        if self.state == State.VERIFYING:
            return self.candidates.verify(frame)

        if self.state == State.APPROACH:
            return self._do_approach(frame)

        return STOP_ACTION

    def _act_inner_post_transition(self, frame: FrameData) -> str:
        """Re-enter EXPLORE logic once after a state transition (no recursion
        beyond one level: EXPLORE either picks a path or turns in place)."""
        self._explore(frame)
        if self.state == State.GOTO_FRONTIER:
            action = self._follow_path(frame)
            if action is not None:
                return action
        return TURN_ACTION

    # -------------------------------------------------------------- keyframes

    def _active_container_tracks(self):
        """The tracks making up the surface the search is currently inspecting.

        A container is a view over tracks rather than an entity of its own, and
        an L-shaped sofa is two ellipsoids under one anchor id, so the answer is
        a list -- projecting only the representative would crop half the sofa.
        """
        cid = getattr(self.exploration, "search_container", None)
        if cid is None:
            return None
        node = self.scene_graph.containers.get(int(cid))
        return list(node.track_ids) if node is not None else None

    def _foveate(self, frame: FrameData, dets: list) -> list:
        """A second detector pass over the container surfaces in view.

        Reported as the DECISION, not the state: `foveate_added` counts only
        detections the whole-frame pass did not already have, and
        `foveate_added_target` only those of the episode's target -- the arm's
        entire claim. An arm that fires constantly and adds no target is a null,
        and has to be legible as one.
        """
        from ..perception.foveate import container_regions, foveated_detect, merge

        sg = self.cfg.scene_graph
        only_ids = None
        if sg.foveate_active_only:
            only_ids = self._active_container_tracks()
            if not only_ids:
                return dets
        regions = container_regions(
            self.object_layer, frame, CONTAINER_CATEGORIES,
            max_range_m=float(sg.foveate_max_range_m),
            min_px=float(sg.foveate_min_bbox_px),
            max_regions=int(sg.foveate_max_regions),
            only_ids=only_ids,
        )
        if not regions:
            return dets
        self.stats["foveate_regions"] = self.stats.get("foveate_regions", 0) + len(regions)
        extra = foveated_detect(self.detector, frame.rgb, regions,
                                pad=float(sg.foveate_pad))
        merged, n_added = merge(dets, extra)
        if n_added:
            self.stats["foveate_added"] = self.stats.get("foveate_added", 0) + n_added
            want = normalize_label(self.target)
            hits = sum(1 for d in merged[len(dets):]
                       if normalize_label(d.label) == want)
            if hits:
                self.stats["foveate_added_target"] = (
                    self.stats.get("foveate_added_target", 0) + hits
                )
        return merged

    def _on_keyframe(self, frame: FrameData) -> None:
        self._kf_count += 1
        with self.profiler.timeit("detector"):
            dets = self.detector.detect(frame.rgb)
            if self.cfg.scene_graph.foveate_containers:
                dets = self._foveate(frame, dets)
        if self.frontier_semantics is not None:
            room = self.room_classifier.classify(frame.rgb) if self.room_classifier else None
            heading = agent_heading(frame.T_wc)
            self.frontier_semantics.observe(
                self.step_count,
                room,
                [d.label for d in dets],
                camera_xy=frame.camera_position[list(PLANE)],
                heading_xy=np.array([np.cos(heading), np.sin(heading)]),
            )
        if self.room_classifier is not None:
            self._room_votes.append((
                frame.camera_position[list(PLANE)].copy(),
                self.room_classifier.classify(frame.rgb),
            ))
        if self.on_keyframe_detections is not None:
            self.on_keyframe_detections(frame, dets)
        with self.profiler.timeit("object_layer"):
            self.object_layer.update(frame, dets, floor_key=self.floors.current_id)
        self.keyframes.add(frame)

        pf = self.object_layer.presence_filter
        if pf is not None:
            # Surfaced per episode so the mechanism is measurable on real runs:
            # how much evidence the beliefs rest on, and how many objects the
            # agent has actually looked for and failed to find.
            self.stats["presence_expected"] = pf.n_expected
            self.stats["presence_negative"] = pf.n_negative
            self.stats["presence_positive"] = pf.n_positive
            self.stats["presence_disbelieved"] = sum(
                1 for t in self.object_layer.tracks() if t.presence.p < 0.1
            )
            for track in self.object_layer.tracks():
                if track.presence.p >= 0.1 or track.id in self._disbelieved:
                    continue
                # First crossing only: the step here is what "belief latency"
                # is measured against, so a belief that dips, recovers and dips
                # again must not reset the clock.
                self._disbelieved.add(track.id)
                self.presence_events.append(
                    {
                        "step": int(self.step_count),
                        "track_id": int(track.id),
                        "label": str(track.label),
                        "center": [float(v) for v in self.object_layer.center_of(track)],
                        "p": round(float(track.presence.p), 4),
                        "n_missed": int(track.presence.n_missed),
                    }
                )

        if self.floors.stairs_due(self._kf_count):
            with self.profiler.timeit("stairs"):
                self.floors.mark_stairs_traversable(self.object_layer)

        if self._kf_count % self.cfg.scene_graph.room_seg_every_kf == 1:
            with self.profiler.timeit("room_seg"):
                self._room_labels = self.floor_layer.segmenter.segment(self.costmap)
        if self._room_labels is not None:
            if self._room_labels.shape != self.costmap.grid.shape:
                self._room_labels = self.floor_layer.segmenter.segment(self.costmap)
            with self.profiler.timeit("scene_graph"):
                self.scene_graph.rebuild_floor(
                    self._room_labels, self.costmap, self.object_layer,
                    floor_key=self.floors.current_id,
                    floor_height=self.floors.height_of(self.floors.current_id),
                )
                self._label_rooms(self.floor_layer)

    def _label_rooms(self, layer) -> None:
        from collections import Counter

        if not self._room_votes or layer.room_labels is None:
            return
        base = layer.key * ROOM_IDS_PER_FLOOR
        votes = {}
        for xy, name in self._room_votes:
            rc = layer.costmap.world_to_grid(xy)
            if not layer.costmap.in_bounds(rc):
                continue
            local = int(layer.room_labels[rc[0], rc[1]])
            if local > 0:
                votes.setdefault(base + local, Counter())[name] += 1
        self.stats["rooms_total"] = len(self.scene_graph.rooms)
        for room_id, counter in votes.items():
            room = self.scene_graph.rooms.get(room_id)
            if room is not None:
                room.label = counter.most_common(1)[0][0]
        self.stats["rooms_labelled"] = sum(
            1 for room in self.scene_graph.rooms.values() if room.label
        )

    def _check_candidates(self) -> None:
        agent_xy = self._agent_xy if self._agent_xy is not None else np.zeros(2)
        # ASCENT's compatibility facade can request the direct candidate gate
        # without installing a mover (unit construction and policy A/Bs).  In
        # that case preserve its verify/cooldown contract explicitly.
        if self._direct_approach and not self._use_navmesh and self.pointnav is None:
            candidates = self.object_layer.candidates(
                self.target,
                min_obs=self.cfg.verification.min_obs,
                min_score=self.cfg.verification.min_score,
                min_bbox_px=self.cfg.verification.min_bbox_px,
                min_evidence=self.cfg.verification.min_evidence,
                floor_key=self.floors.current_id,
                step=self.step_count,
            )
            if not candidates:
                return
            track = candidates[0]
            self._candidate_id = track.id
            if self.verifier is not None and not self.cfg.verification.absence_only:
                with self.profiler.timeit("verification"):
                    accepted = self.verifier.verify(track, self.target)
                if not accepted:
                    cooldown = int(self.cfg.verification.reject_cooldown_steps)
                    if cooldown > 0:
                        self.object_layer.suppress(track.id, self.step_count + cooldown)
                    else:
                        self.object_layer.blacklist(track.id)
                    self._candidate_id = None
                    self.stats["verify_reject"] = self.stats.get("verify_reject", 0) + 1
                    return
            obj = self.object_layer.center_of(track)
            self._start_approach(obj[list(PLANE)], floor_y=float(obj[1]))
            return
        self.candidates.check(agent_xy)

    def _world(self, frame: FrameData) -> WorldView:
        """What the exploration strategy is allowed to see this round.

        Built per call rather than held: `costmap` is a different object once
        the agent changes storey, and `goal_xy` belongs to the FSM.
        """
        return WorldView(
            frame=frame,
            step=self.step_count,
            agent_xy=frame.camera_position[list(PLANE)],
            costmap=self.costmap,
            scene_graph=self.scene_graph,
            object_layer=self.object_layer,
            keyframes=self.keyframes,
            target=self.target,
            goal_xy=self._goal_xy,
            floor_id=self.floors.current_id,
            value_map=self.floor_layer.value_map,
        )

    def _explore(self, frame: FrameData) -> None:
        """Run one selection round and act on what it chose.

        The strategy decides where; this applies it. Both kinds of choice enter
        GOTO_FRONTIER, because that is the only state that follows `_goal_xy` --
        a surface chosen but left in EXPLORE is a surface never visited, which
        is the defect that invalidated every C3 result before it was found.
        """
        world = self._world(frame)
        choice = self.exploration.select(
            world,
            floor_switch=lambda cost, target_floor=None: self._try_floor_switch(
                frame, cost, target_floor=target_floor
            ),
        )
        if choice is None:
            return
        self._goal_xy = choice.goal_xy
        self._current_path = choice.path
        self.exploration.note_progress(world)
        self.state = State.GOTO_FRONTIER

    def _try_floor_switch(
        self, frame: FrameData, best_path_cost, target_floor: Optional[int] = None
    ) -> bool:
        """Ask the floor policy whether to leave this storey, and go if so.

        The policy decides; the FSM moves. Returns True when a portal is now
        being driven to, which the caller reads as "this selection round is
        settled". Checked BEFORE committing to a far frontier, because "the best
        thing here is 12 m away" is exactly ASCENT's condition for reasoning
        about storeys.
        """
        portal = self.floors.try_switch(
            frame, self.step_count, best_path_cost,
            self.scene_graph, self.target, self._reachable_fn,
            target_floor=target_floor,
        )
        if portal is None:
            return False
        self._goal_xy = portal.goal_xy
        self._goal_floor_y_cache = portal.target_y
        self._current_path = None
        self.exploration.current_frontier = None
        self.exploration.note_progress(self._world(frame))
        self.state = State.GOTO_FRONTIER
        self._goto_deadline = self.step_count + portal.deadline_steps
        return True

    # ------------------------------------------------------------- candidates

    def _absence_at_arrival(self, frame: FrameData, reason: str) -> Optional[str]:
        """The approach is ending and the target was never seen. Say so.

        Returns an action when the candidate is abandoned (the caller must not
        STOP), or None to let the normal termination proceed. A track that has
        been seen at some point during this approach is left alone -- the target
        was there, so this is a geometry or timing problem, not absence.

        The reading itself is verification/absence.py; what is here is applying
        its verdict to the FSM.
        """
        track = (
            self.object_layer.get(self._candidate_id)
            if self._candidate_id is not None else None
        )
        if track is None or self.approach.last_good_xy is not None:
            return None
        verdict = self.absence.observe(
            track, self.target, frame, self.object_layer.presence_filter,
            self.approach.scan_expected, reason,
        )
        if verdict is None or not verdict.abandon:
            return None
        self.presence_events.append(
            {
                "step": int(self.step_count),
                "track_id": int(track.id),
                "label": str(track.label),
                "center": [float(v) for v in self.object_layer.center_of(track)],
                "p": round(float(verdict.p), 4),
                "n_missed": int(track.presence.n_missed),
                **verdict.event,
            }
        )
        # Deliberately NOT blacklisted -- see verification/absence.py.
        self._candidate_id = None
        self._target_obj_xy = None
        self.state = State.EXPLORE
        return TURN_ACTION

    # ---------------------------------------------------------------- helpers

    def _best_target_detection(self, frame: FrameData) -> Optional[Detection]:
        """Runs the detector on the current frame and returns its highest-
        confidence detection matching the target category, or None."""
        target = normalize_label(self.target)
        with self.profiler.timeit("detector"):
            dets = self.detector.detect(frame.rgb)
        matches = [
            d for d in dets
            if normalize_label(d.label) == target and d.score > 0.25
        ]
        return max(matches, key=lambda d: d.score) if matches else None

    def _target_visible(self, frame: FrameData) -> bool:
        """Does the detector see the target category in the current view?"""
        return self._best_target_detection(frame) is not None

    @staticmethod
    def _detection_depth(det: Detection, frame: FrameData) -> Optional[float]:
        """Median metric depth (m) over the detection's mask, using only valid
        depth pixels; None if too few valid samples (mask off the depth range)."""
        ys, xs = np.nonzero(det.mask)
        if ys.size == 0:
            return None
        d = frame.depth[ys, xs]
        valid = d > 1e-3
        if int(valid.sum()) < 8:
            return None
        return float(np.median(d[valid]))

    def _plan_to(
        self, frame: FrameData, goal_xy: np.ndarray, goal_tolerance_m: Optional[float] = None
    ) -> None:
        agent_xy = frame.camera_position[list(PLANE)]
        with self.profiler.timeit("planner"):
            result: PlanResult = self.planner.plan(
                self.costmap, agent_xy, goal_xy, goal_tolerance_m
            )
        self._current_path = result.path if result.success else None
        self.stats["plan_ok" if result.success else "plan_fail"] += 1

    def _follow_path(self, frame: FrameData) -> Optional[str]:
        goal = (
            frontier_goal_xy(
                self.exploration.current_frontier, self.costmap,
                self.exploration.goal_prefer_free,
            )
            if self.state == State.GOTO_FRONTIER
            and self.exploration.current_frontier is not None
            else self._goal_xy
        )
        if goal is None:
            return None
        if self._use_navmesh:
            # Drive on Habitat's navmesh. None = arrived-or-unreachable; if we're
            # still far from a frontier goal, block it (as the stub-block does).
            #
            # The height must be keyed on whether the GOAL is cross-floor, not
            # on the state. An ordinary frontier goal is on the agent's own
            # floor and takes the default; a portal pursuit runs in this same
            # GOTO_FRONTIER state but targets another storey, and keying on the
            # state discarded its height -- snapping the portal's (x, z) onto
            # the floor BELOW it. The agent then walked to a point under the
            # mezzanine, arrived, never gained height, and the pursuit was
            # abandoned as "no vertical progress" (26 of 35 endings on full v1).
            cross_floor_goal = self.floors.pursuing or self.state != State.GOTO_FRONTIER
            action = self._nav_fn(goal, self._goal_floor_y_cache if cross_floor_goal else None)
            if action is None and self.state == State.GOTO_FRONTIER:
                self.exploration.retire_pursued(self._world(frame), goal)
            return action
        if self.pointnav is not None:
            step = self.pointnav.step(goal)
            if step.action is None and self.state == State.GOTO_FRONTIER:
                if step.reason == "policy_stop" and not bool(
                    self.cfg.agent.pointnav_stop_means_blocked
                ):
                    self.stats["pointnav_stop_forced_forward"] = (
                        self.stats.get("pointnav_stop_forced_forward", 0) + 1
                    )
                    return "move_forward"
                if float(
                    np.linalg.norm(frame.camera_position[list(PLANE)] - goal)
                ) > self._frontier_reach_m:
                    # ``retire_pursued`` owns both the blacklist update and the
                    # associated counter.  Doing either here would double count
                    # a PointNav policy stop.
                    pass
                self.exploration.retire_pursued(self._world(frame), goal)
            return step.action
        if self._current_path is None:
            self._plan_to(frame, goal)
            if self._current_path is None:
                if self.state == State.GOTO_FRONTIER:
                    self.exploration.block(
                        self.exploration.current_frontier, 50, self.step_count
                    )
                return None
        action = self.controller.act(frame.T_wc, self._current_path)
        if action is None:
            self._current_path = None
            if self.state == State.GOTO_FRONTIER:
                self.exploration.retire_pursued(self._world(frame), goal)
        return action

    # ------------------------------------------------ ASCENT compatibility API

    @property
    def frontier_extractor(self):
        return self.exploration.frontier_extractor

    @property
    def _progress_ref_step(self) -> int:
        return int(self.exploration.progress_ref_step)

    @_progress_ref_step.setter
    def _progress_ref_step(self, value: int) -> None:
        self.exploration.progress_ref_step = int(value)

    @property
    def _progress_ref_xy(self) -> np.ndarray:
        return self.exploration.progress_ref_xy

    @_progress_ref_xy.setter
    def _progress_ref_xy(self, value: np.ndarray) -> None:
        self.exploration.progress_ref_xy = np.asarray(value, dtype=float)

    @property
    def _select_every(self) -> int:
        return int(getattr(self.cfg.exploration, "select_every", 5))

    @property
    def _reselect_every(self) -> int:
        return int(getattr(self.cfg.exploration, "reselect_every", 0))

    @property
    def _last_select_step(self) -> int:
        return self.exploration._last_select_step

    @_last_select_step.setter
    def _last_select_step(self, value: int) -> None:
        self.exploration._last_select_step = int(value)

    def _select_new_frontier(self, frame: FrameData) -> None:
        prev = self._current_frontier
        before = self.exploration._last_select_step
        progress_step = self._progress_ref_step
        progress_xy = self._progress_ref_xy.copy()
        frontier_dist = self._frontier_ref_dist
        self._explore(frame)
        if self.exploration._last_select_step == before:
            return
        current = self._current_frontier
        if current is None:
            return
        same = prev is not None and float(
            np.linalg.norm(prev.centroid_xy - current.centroid_xy)
        ) < 0.5
        if same:
            self._progress_ref_step = progress_step
            self._progress_ref_xy = progress_xy
            self._frontier_ref_dist = frontier_dist
        else:
            self.stats["frontier_switch"] = self.stats.get("frontier_switch", 0) + 1
            self._progress_ref_step = self.step_count
            self._progress_ref_xy = frame.camera_position[list(PLANE)].copy()
            self._frontier_ref_dist = None

    def _frontier_consumed(self, frontier) -> bool:
        if frontier is None:
            return False
        goal = frontier_goal_xy(frontier, self.costmap)
        rc = self.costmap.world_to_grid(goal)
        radius = max(1, int(round(self.cfg.agent.agent_radius / self.costmap.resolution)))
        h, w = self.costmap.grid.shape
        r0, r1 = max(0, rc[0] - radius), min(h, rc[0] + radius + 1)
        c0, c1 = max(0, rc[1] - radius), min(w, rc[1] + radius + 1)
        if r0 >= r1 or c0 >= c1:
            return False
        from ..mapping.costmap import UNKNOWN

        return not bool((self.costmap.grid[r0:r1, c0:c1] == UNKNOWN).any())

    def _frontier_stalled(
        self, agent_xy: np.ndarray, frontier, stick_m: float, stick_steps: int
    ) -> bool:
        if stick_steps <= 0:
            return False
        if self.cfg.agent.frontier_stick_rule == "closing":
            goal = frontier_goal_xy(frontier, self.costmap) if frontier else self._goal_xy
            if goal is None:
                return False
            distance = float(np.linalg.norm(agent_xy - goal))
            if self._frontier_ref_dist is None:
                self._frontier_ref_dist = distance
                self._progress_ref_step = self.step_count
                return False
            if abs(self._frontier_ref_dist - distance) > stick_m:
                self._frontier_ref_dist = distance
                self._progress_ref_step = self.step_count
                return False
            return self.step_count - self._progress_ref_step >= stick_steps
        if self.step_count - self._progress_ref_step < stick_steps:
            return False
        moved = float(np.linalg.norm(agent_xy - self._progress_ref_xy))
        self._progress_ref_step = self.step_count
        self._progress_ref_xy = agent_xy.copy()
        return moved < stick_m

    def _nearest_point_stop(self, agent_xy: np.ndarray) -> Optional[str]:
        track = self.object_layer.get(self._candidate_id) if self._candidate_id is not None else None
        if track is None:
            return None
        distance = self.object_layer.nearest_point_dist_xy(
            track, agent_xy, float(self.cfg.agent.terminal_percentile)
        )
        if distance is None or distance >= float(self.cfg.agent.terminal_engage_m):
            return None
        if distance <= float(self.cfg.agent.terminal_stop_m):
            return "nearest_point"
        moved = self._terminal_last_xy is not None and float(
            np.linalg.norm(agent_xy - self._terminal_last_xy)
        ) > 0.05
        self._terminal_last_xy = agent_xy.copy()
        if not moved:
            return None
        if abs(distance - self._terminal_min_d) < float(self.cfg.agent.terminal_progress_eps):
            self._terminal_stalls += 1
            if self._terminal_stalls >= int(self.cfg.agent.terminal_stall_steps):
                return "nearest_point_stalled"
        else:
            self._terminal_stalls = 0
            self._terminal_min_d = min(self._terminal_min_d, distance)
        return None

    def _carrot_goal(
        self, frame: FrameData, agent_xy: np.ndarray
    ) -> Optional[np.ndarray]:
        """Place ASCENT's short stair waypoint along the farthest depth ray."""
        depth = frame.depth
        if depth.size == 0:
            return None
        max_value = float(np.max(depth))
        if not np.isfinite(max_value):
            return None
        rows_cols = np.argwhere(depth == max_value)
        if rows_cols.size == 0:
            return None
        u = float(np.mean(rows_cols[:, 1]))
        intr = frame.intrinsics
        hfov = 2.0 * float(np.arctan(intr.width / (2.0 * intr.fx)))
        normalized_u = float(np.clip((u - float(intr.cx)) / float(intr.cx), -1.0, 1.0))
        heading = agent_heading(frame.T_wc) + normalized_u * hfov / 2.0
        distance = float(getattr(self.cfg.agent, "climb_carrot_m", 0.8))
        return agent_xy + distance * np.array([np.cos(heading), np.sin(heading)])

    def _update_carrot(
        self, frame: FrameData, agent_xy: np.ndarray
    ) -> Optional[np.ndarray]:
        fresh = self._carrot_goal(frame, agent_xy)
        if fresh is None:
            return self._carrot_xy
        end = getattr(self, "_climb_goal_xy", None)
        near_end = end is not None and float(np.linalg.norm(end - agent_xy)) <= 0.5
        if self._carrot_xy is None or end is None or near_end or self._carrot_disable_end:
            self._carrot_xy = fresh
        elif np.linalg.norm(fresh - end) < np.linalg.norm(self._carrot_xy - end):
            self._carrot_xy = fresh
        return self._carrot_xy

    def _carrot_action(self, frame: FrameData, agent_xy: np.ndarray) -> str:
        goal = self._update_carrot(frame, agent_xy)
        if goal is None:
            return FORWARD_ACTION
        if self.pointnav is not None:
            nav = self.pointnav.step(goal)
            if nav.action is None:
                self.stats["climb_forced_forward"] = (
                    self.stats.get("climb_forced_forward", 0) + 1
                )
                return FORWARD_ACTION
            return nav.action
        action = self._follow_to(frame, goal)
        return action if action is not None else FORWARD_ACTION

    def _carrot_stalled(self, agent_xy: np.ndarray) -> bool:
        ref = getattr(self, "_climb_centroid_xy", None)
        if ref is None:
            return False
        distance = float(np.linalg.norm(agent_xy - ref))
        if self._climb_last_dist is None or abs(self._climb_last_dist - distance) > 0.2:
            self._climb_last_dist = distance
            self._climb_paused_steps = 0
        else:
            self._climb_paused_steps += 1
        if self._climb_paused_steps > 15:
            self._carrot_disable_end = True
        return self._climb_paused_steps > 30

    def _update_value_map(self, frame: FrameData, layer) -> None:
        """Score the current view and fuse it into the active floor map.

        The image-text component is constructed only for explicit
        ``exploration.value_map`` configurations. Keeping the update here,
        after the floor policy has selected the layer and the costmap has
        grown, guarantees that value/confidence arrays remain aligned with the
        floor-specific occupancy grid. A stride avoids repeatedly scoring
        effectively identical frames and preserves the approach re-check
        telemetry from the same scalar.
        """
        value_map = getattr(layer, "value_map", None)
        if self.image_text is None or value_map is None:
            return
        stride = max(1, int(getattr(self.cfg.exploration, "value_stride", 1)))
        if self.step_count % stride:
            return
        prompt = str(
            getattr(
                self.cfg.exploration,
                "value_prompt",
                "Seems like there is a {target} ahead.",
            )
        ).format(target=self.target.replace("_", " "))
        with self.profiler.timeit("value_map"):
            scores = self.image_text.score(frame.rgb, [prompt])
            if len(scores) == 0:
                return
            value = float(scores[0])
            value_map.update(frame, value)
        self.stats["value_calls"] = self.stats.get("value_calls", 0) + 1
        self._last_itm = value
        if self.state is State.APPROACH:
            self._approach_itm_max = max(self._approach_itm_max, value)
            self._approach_itm_n += 1

    def _seg_stair_mask(self, frame: FrameData) -> Optional[np.ndarray]:
        if self.stair_segmenter is None:
            return None
        with self.profiler.timeit("stair_seg"):
            return self.stair_segmenter.stair_mask(frame)

    def _accumulate_down_stairs(self, frame: FrameData, layer) -> None:
        if self.stair_detector is None:
            return
        with self.profiler.timeit("stairs"):
            self.stair_detector.accumulate(
                frame, layer, None, self._seg_stair_mask(frame)
            )

    def _down_look(self, frame: FrameData, layer, off_map: bool) -> Optional[str]:
        if self._pitch_ticks > 0:
            if not off_map:
                self._accumulate_down_stairs(frame, layer)
            self._pitch_ticks -= 1
            return "look_up"
        if self._down_look_every <= 0:
            return None
        if self.state not in (State.INIT, State.EXPLORE, State.GOTO_FRONTIER):
            return None
        if self.step_count - self._last_down_look_step < self._down_look_every:
            return None
        self._last_down_look_step = self.step_count
        self._pitch_ticks += 1
        self.stats["down_look"] = self.stats.get("down_look", 0) + 1
        return "look_down"

    def _floor_direction_boost(self, kind: str) -> float:
        if not self._floor_goal_dir:
            return 1.0
        boost = float(getattr(self.cfg.exploration, "floor_llm_boost", 5.0))
        wanted = "up" if self._floor_goal_dir > 0 else "down"
        return boost if kind == wanted else 1.0 / boost

    def _mark_floor_explored(self, n_explore: int) -> None:
        layer = self.floor_layer
        rule = str(
            getattr(self.cfg.exploration, "stair_explored_rule", "no_frontiers")
        )
        if rule == "no_frontiers":
            if n_explore == 0:
                layer.explored = True
        elif not layer.explored:
            layer.explored = layer.steps_on_floor >= int(
                getattr(self.cfg.exploration, "floor_exp_steps", 100)
            )

    def _left_the_stairs(self, agent_xy: np.ndarray) -> bool:
        cells = getattr(self, "_climb_cells_xy", None)
        if cells is None or not len(cells):
            return True
        distance = float(np.linalg.norm(cells - agent_xy, axis=1).min())
        return distance > float(getattr(self.cfg.agent, "stair_exit_m", 0.5))

    def _on_a_staircase(self, agent_xy: np.ndarray) -> bool:
        detector = self.stair_detector
        layer = self.floor_layer
        if detector is None or layer.up_stair_hits is None:
            return False
        mask = (
            (layer.up_stair_hits >= detector.min_hits)
            | (layer.down_stair_hits >= detector.min_hits)
        )
        if layer.disabled_stair is not None:
            mask &= ~layer.disabled_stair
        rc = layer.costmap.world_to_grid(agent_xy)
        radius = max(
            1,
            int(round(float(getattr(self.cfg.agent, "stair_exit_m", 0.5)) /
                      layer.costmap.resolution)),
        )
        r0, r1 = max(0, rc[0] - radius), min(mask.shape[0], rc[0] + radius + 1)
        c0, c1 = max(0, rc[1] - radius), min(mask.shape[1], rc[1] + radius + 1)
        if r0 >= r1 or c0 >= c1:
            return False
        yy, xx = np.ogrid[r0:r1, c0:c1]
        disk = (yy - rc[0]) ** 2 + (xx - rc[1]) ** 2 <= radius ** 2
        return bool((mask[r0:r1, c0:c1] & disk).any())

    def _floor_frozen(self, frame: FrameData) -> bool:
        if self.state is State.CLIMB and bool(
            getattr(self.cfg.mapping, "freeze_floor_in_climb", False)
        ):
            return True
        if bool(getattr(self.cfg.mapping, "freeze_floor_on_stairs", False)):
            return self._on_a_staircase(frame.camera_position[list(PLANE)])
        return False

    def _recheck_rejects(self, stop_reason: str) -> bool:
        """Apply ASCENT's latched image-text gate before committing a stop."""
        self.approach_recheck_max = float(self._approach_itm_max)
        if not self._approach_recheck:
            return False
        self.stats["recheck_calls"] = self.stats.get("recheck_calls", 0) + 1
        if self._approach_itm_n == 0:
            self.stats["recheck_no_obs"] = self.stats.get("recheck_no_obs", 0) + 1
            return False
        if self._approach_itm_max >= self._approach_recheck_thresh:
            self.stats["recheck_pass"] = self.stats.get("recheck_pass", 0) + 1
            return False
        self.stats["recheck_reject"] = self.stats.get("recheck_reject", 0) + 1
        self.stats[f"recheck_reject_{stop_reason}"] = (
            self.stats.get(f"recheck_reject_{stop_reason}", 0) + 1
        )
        if self._candidate_id is not None:
            self.object_layer.blacklist(self._candidate_id)
        self._candidate_id = None
        self._target_obj_xy = None
        self._goal_xy = None
        self._current_path = None
        self.state = State.EXPLORE
        return True

    def _commit_terminal_stop(self, stop_reason: str) -> str:
        if self._recheck_rejects(stop_reason):
            return TURN_ACTION
        self.state = State.DONE
        self.approach.stop_reason = stop_reason
        return STOP_ACTION

    def _start_approach(
        self,
        obj_xy: np.ndarray,
        agent_xy: Optional[np.ndarray] = None,
        floor_y: Optional[float] = None,
    ) -> None:
        """Compatibility entry point shared by the FSM and ASCENT facade."""
        self._approach_itm_max = 0.0
        self._approach_itm_n = 0
        self.approach_recheck_max = None
        self._approach_start_step = self.step_count
        here = agent_xy if agent_xy is not None else self._agent_xy
        if self._candidate_id is not None and here is not None:
            track = self.object_layer.get(self._candidate_id)
            if track is not None:
                nearest = self.object_layer.nearest_point_xy(track, here)
                self._target_cloud_xy = (
                    None if nearest is None else np.asarray(nearest, dtype=float).copy()
                )
        self.approach.start(obj_xy, agent_xy, floor_y)

    def _abandon_approach(self) -> str:
        disabled = self._candidate_id is not None and self.object_layer.disable_target(
            self._candidate_id
        )
        if not disabled and self._target_obj_xy is not None:
            self.object_layer.disable_place(self._target_obj_xy, self.target)
        self.stats["approach_abandon"] = self.stats.get("approach_abandon", 0) + 1
        self._candidate_id = self._target_obj_xy = self._goal_xy = None
        self._current_path = None
        self.state = State.EXPLORE
        return TURN_ACTION

    def _do_approach(self, frame: FrameData) -> str:
        budget = int(getattr(self.cfg.agent, "approach_abandon_steps", 0))
        if (
            self._reachable_fn is None and budget > 0
            and self.step_count - self._approach_start_step >= budget
        ):
            return self._abandon_approach()
        if self.cfg.agent.terminal_rule == "nearest_point":
            reason = self._nearest_point_stop(frame.camera_position[list(PLANE)])
            if reason is not None and (
                not self.cfg.agent.terminal_requires_detection
                or self._target_visible(frame)
            ):
                return self._commit_terminal_stop(reason)
        if hasattr(self, "_approach_steps_left"):
            self.approach.steps_left = self._approach_steps_left
        action = self.approach.step(frame)
        self._approach_steps_left = self.approach.steps_left
        if action == STOP_ACTION:
            return self._commit_terminal_stop(self.approach.stop_reason or "approach")
        return action

    def _follow_to(self, frame: FrameData, goal_xy: np.ndarray) -> Optional[str]:
        if self.pointnav is not None:
            creep = (
                float(self.cfg.agent.pointnav_approach_creep_m)
                if self.state is State.APPROACH else 0.0
            )
            return self.pointnav(
                goal_xy, creep_below=creep,
                stop_radius=float(self.cfg.agent.pointnav_arrival_m),
            )
        return self.approach.follow_to(frame, goal_xy)
