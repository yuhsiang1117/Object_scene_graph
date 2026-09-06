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
from ..core.types import Detection, FrameData
from ..exploration.async_scorer import AsyncScorer
from ..exploration.selector import frontier_goal_xy
from ..exploration.strategy import ExplorationStrategy, WorldView
from ..graph.scene_graph import SceneGraph
from ..mapping.costmap import PLANE, Costmap2D
from ..objects.object_layer import ObjectLayer
from ..perception.detector import Detector
from ..perception.vocabulary import target_vocabulary
from ..pipeline.beliefs import build_affinity_prior, build_presence_filter
from ..perception.keyframe import KeyframeSelector, KeyframeStore
from ..planning.controller import WaypointController
from ..planning.planner import PlanResult
from ..planning.voronoi_planner import HybridVoronoiPlanner
from ..verification.absence import AbsenceSensor
from ..verification.viewpoint import ViewpointPlanner
from .approach import ApproachPolicy
from .candidate import CandidatePolicy
from .floor_policy import FloorPolicy
from ..core.labels import normalize_label
from ..graph.containers import CONTAINER_CATEGORIES
from .state import STOP_ACTION, TURN_ACTION, State

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
        self._use_navmesh = (
            nav_fn is not None and bool(cfg.agent.use_habitat_navmesh)
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
        self.floors = FloorPolicy(cfg, self.stats)
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
        # GVG Voronoi (medial-axis) navigation ported from ObjectSceneGraph_old,
        # with a grid-A* fallback for early/tiny maps (single planner so frontier
        # selection and path planning both get the fallback).
        self.planner = HybridVoronoiPlanner(
            collision_m=cfg.agent.agent_radius + cfg.mapping.inflate_margin_m,
            goal_near_m=cfg.exploration.voronoi_goal_near_m,
            inflate_radius_m=cfg.agent.agent_radius + cfg.mapping.inflate_margin_m,
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
            cfg, self.planner, scorer, self.viewpoint_planner,
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
        self._kf_count = 0
        self._current_path: Optional[np.ndarray] = None
        self._candidate_id: Optional[int] = None
        self._goal_xy: Optional[np.ndarray] = None
        self._last_action: Optional[str] = None
        self._goto_deadline = 10**9
        self._target_obj_xy: Optional[np.ndarray] = None
        self._goal_floor_y_cache: Optional[float] = None
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
        self.detector.set_vocabulary(
            target_vocabulary(self.target, self.cfg.detector.vocabulary)
        )
        self.object_layer.set_target(self.target)

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
        if self.state != prev_state:
            self.state_log.append((self.step_count, self.state.value))
        self._last_action = action
        return action

    def _act_inner(self, frame: FrameData) -> str:
        floor_y = self.floors.observe(frame, self.step_count)

        with self.profiler.timeit("costmap"):
            self.costmap.update(
                frame,
                floor_y=floor_y,
                obstacle_low=self.cfg.mapping.obstacle_low_m,
                obstacle_high=self.cfg.mapping.obstacle_high_m,
                max_range=self.cfg.mapping.max_range_m,
                stride=self.cfg.mapping.depth_stride,
            )
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

        # Candidate target check happens in every state except terminal ones
        if self.state in (State.INIT, State.EXPLORE, State.GOTO_FRONTIER):
            self.candidates.check()

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
            return self.approach.step(frame)

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
        regions = container_regions(
            self.object_layer, frame, CONTAINER_CATEGORIES,
            max_range_m=float(sg.foveate_max_range_m),
            min_px=float(sg.foveate_min_bbox_px),
            max_regions=int(sg.foveate_max_regions),
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
        if self.on_keyframe_detections is not None:
            self.on_keyframe_detections(frame, dets)
        with self.profiler.timeit("object_layer"):
            self.object_layer.update(frame, dets)
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
                self.scene_graph.rebuild(
                    self._room_labels, self.costmap, self.object_layer,
                    floors=self.floors.estimator if self.cfg.floor.enabled else None,
                )

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
            world, floor_switch=lambda cost: self._try_floor_switch(frame, cost)
        )
        if choice is None:
            return
        self._goal_xy = choice.goal_xy
        self._current_path = choice.path
        self.exploration.note_progress(world)
        self.state = State.GOTO_FRONTIER

    def _try_floor_switch(self, frame: FrameData, best_path_cost) -> bool:
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

