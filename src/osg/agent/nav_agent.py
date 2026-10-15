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
    agent/climb.py            State.CLIMB: getting up or down the staircase

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
from .climb import ClimbPolicy
from .close_look import CloseLookPolicy
from .stop_gate import StopGate
from .floor_policy import FloorPolicy
from .stair_sense import StairSense
from ..core.labels import normalize_label
from ..graph.containers import CONTAINER_CATEGORIES
from .state import FORWARD_ACTION, STOP_ACTION, TURN_ACTION, WAIT_ACTION, State

def _horizontal_radius_m(track) -> float:
    """The track's ground-plane extent, from its fitted ellipsoid.

    The semi-axes are in the ellipsoid's own frame, so the honest ground-plane
    radius is the largest of them: a box seen edge-on has its long axis
    horizontal whichever way R points, and over-estimating the radius pushes the
    viewpoint ring outwards, which is the safe direction -- a ring that is too
    big costs a little distance, one that is too small lands in an occupied cell
    and loses the viewpoint entirely.
    """
    ellipsoid = getattr(track, "ellipsoid", None)
    axes = getattr(ellipsoid, "axes", None)
    if axes is None:
        return 0.0
    try:
        radius = float(np.max(np.abs(np.asarray(axes, dtype=float))))
    except (TypeError, ValueError):
        return 0.0
    return radius if np.isfinite(radius) else 0.0


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
        nearest_navigable_fn=None,
        pointnav=None,
        ranker=None,
        floor_planner=None,
        room_classifier=None,
        image_text=None,
        feature_memory=None,
        region_proposer=None,
        gate_itm=None,
        stair_segmenter=None,
        stair_detector=None,
        ram=None,
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
        # nearest_navigable_fn(xy, floor_y) -> the navmesh point nearest to xy
        # on the agent's island, for closing the last metre of an approach.
        self._nearest_navigable_fn = nearest_navigable_fn
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
        # `build_agent` hands every policy the same components. The served
        # GroundingDINO stair detector is built only for ascentnav, so this is
        # None on the nav_agent line and the cfg-driven StairDetector below is
        # what runs; an injected one wins because it is the stronger sensor.
        self.stair_detector = stair_detector
        # RAM++ tags are an ascentnav prompt input; accepted so the common
        # construction contract holds, unused here.
        self.ram = ram
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
        # Appearance matching (objects/feature_memory.py), or None. Kept apart
        # from `image_text`: any non-None value THERE turns the semantic value
        # map on for every frame, which is a different mechanism entirely.
        self.feature_memory = feature_memory
        # Class-agnostic proposals for the objects the detector never names
        # (perception/region_proposer.py). None unless region_proposal.enabled.
        self.region_proposer = region_proposer
        self._region_admits = 0
        self._region_cache = None
        self._region_kf = 0          # keyframes seen this episode
        self._region_named = False   # has the LABEL path ever named the target
        # One entry per appearance commit: which surface, which track, how
        # much it looked like the query, and how many it beat.
        self.feature_pick_log: list = []
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
        if self.stair_detector is None and bool(getattr(cfg.mapping, "multi_floor", False)):
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
            feature_memory=feature_memory,
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
            containers_floor_relative=bool(getattr(
                cfg.scene_graph, "containers_floor_relative", False)),
            container_merge_sigma=float(
                getattr(cfg.scene_graph, "container_merge_sigma", 0.0) or 0.0
            ),
        )
        self.keyframes = KeyframeStore(save_dir=keyframe_dir)
        self.kf_selector = KeyframeSelector(
            cfg.scene_graph.keyframe_trans_m, cfg.scene_graph.keyframe_rot_deg
        )
        self.controller = WaypointController(forward_m=cfg.agent.forward_m)
        self.viewpoint_planner = ViewpointPlanner(
            list(cfg.verification.ring_radii_m),
            min_clearance_m=float(getattr(cfg.verification, "viewpoint_min_clearance_m", 0.0) or 0.0),
        )
        # "I walked there and it was not there" as evidence, from two sensors
        # with different error rates. See verification/absence.py.
        self.absence = AbsenceSensor(cfg, verifier, self.profiler, self.stats)
        # The two state handlers. Each owns what only it uses; NavAgent stays
        # the single owner of FSM state.
        self.approach = ApproachPolicy(self)
        self.candidates = CandidatePolicy(self)
        self.close_look = CloseLookPolicy(self)
        self.stop_gate = StopGate(self)
        self.climb = ClimbPolicy(self)
        self.stairs = StairSense(self)
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

    @property
    def climb_trace(self) -> list:  # eval/record.py records it per episode
        return self.climb.trace

    @property
    def close_look_log(self) -> list:
        return self.close_look.log

    @property
    def glance_ranges(self) -> dict:
        return {str(k): round(float(v), 3) for k, v in self.exploration.glance_ranges.items()}

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
        # Phase 2 instrumentation (docs/history/DYNAMIC_SCENES.md): when the map STOPPED
        # believing in something, and what it believed at the moment it
        # committed to a goal. Belief latency and stale-goal rate are computed
        # from these two logs plus the relocation step the env records.
        self.presence_events: List[dict] = []
        self.floor_llm_log: list = []
        self._disbelieved: set = set()
        self._stale_stop_used = False
        self._stale_stop_pending = False
        self._relook = None  # (key, centre_xy, radius_m, exclude_xy, label)
        self._explore_hold_until = 0  # see `rearm`
        self._failed_switches_by_floor = {}
        self._switch_banned_at = {}
        self._failed_attempts_by_floor: dict = {}   # see `rearm`
        self._disproved_floors: set = set()
        self.state_log = []
        self.approach.reset()
        self.candidates.reset()
        self.close_look.reset()
        self.stop_gate.reset()
        self.kf_selector.reset()
        self.controller.reset()
        self.exploration.reset()
        # After the strategy's reset, which rebuilds its per-episode fields:
        # the verdict is the agent's, and the strategy reads the same set.
        self.exploration.disproved_floors = self._disproved_floors
        # The SAME set, so a storey disproved here also stops the floor policy
        # walking back to it. It gated three channels and not the one that
        # chooses where to go (docs/history/CROSS_ANCHOR_STATUS.md).
        self.floors.disproved_floors = self._disproved_floors
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
        self.climb.reset()
        self._failed_switches_by_floor: dict = {}
        self._switch_banned_at: dict = {}
        self.stairs.reset()
        if self.commit_state is not None:
            self.commit_state.reset()
        if self.frontier_semantics is not None:
            self.frontier_semantics.reset()
        self.detector.set_vocabulary(
            target_vocabulary(self.target, self.cfg.detector.vocabulary)
        )
        if self.region_proposer is not None:
            self.region_proposer.set_target(self.target)
            self._region_admits = 0
            self._region_cache = None
            self._region_kf = 0
            self._region_named = False
            self.object_layer.set_proposal_text(self.region_proposer.text)
        if self.feature_memory is not None:
            # The prompt changes once an episode, so the text encoder runs once
            # an episode. Centres come from the object layer, which resolves a
            # linked component rather than one ellipsoid.
            self.feature_memory.set_target(self.target)
            self.feature_memory.bind_centres(
                lambda t: self.object_layer.center_of(t)[list(PLANE)]
            )
            self.feature_pick_log = []
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

    def release_candidate(self) -> None:
        """Drop the committed candidate and go back to EXPLORE, map intact.

        The navigation half of `rearm`, shared with the STOP gate, which
        releases a candidate without an attempt having been spent on it."""
        self._candidate_id = None
        self._target_obj_xy = None
        self._goal_xy = None
        self._current_path = None
        self.approach.at_viewpoint = False
        self.approach.scan_turns_left = 0
        self.approach.scan_expected = 0
        self.approach.stop_reason = None
        self.close_look.abort()
        self._stale_stop_pending = False
        self.state = State.EXPLORE

    def hold_explore(self) -> None:
        """Keep the commit open for one exploration round (see `rearm`)."""
        hold = int(getattr(self.cfg.agent, "explore_after_failed_attempt_steps", 0))
        if hold > 0:
            self._explore_hold_until = self.step_count + hold
            force = getattr(self.exploration, "force_select_next", None)
            if callable(force):
                force()

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
        self._schedule_relook()
        self.release_candidate()
        self._goto_deadline = self.step_count + int(max_steps)
        self.stats["attempts"] = self.stats.get("attempts", 1) + 1
        # EXPLORE is not the same thing as an exploration ROUND. `_act_inner`
        # runs `candidates.check` before the EXPLORE branch on every step, so
        # the next same-label track commits immediately and the round -- rate
        # limited to one per `exploration.select_every` steps -- never happens.
        # Hold the commit open long enough for one, and make that one run now
        # rather than at the rate limit's convenience.
        self._note_failed_attempt_on_storey()
        self.hold_explore()

    # ------------------------------------------------------------------- act

    # ------------------------------------------------- storeys disproved

    def _note_failed_attempt_on_storey(self) -> None:
        """A failed attempt is evidence about the STOREY, not just the track.

        Counted per storey; at `floor_disproved_after_failed_attempts` the
        storey is disproved: its target-labelled tracks stop carrying the
        _TARGET_PRESENT veto (`_presence_for_floor_evidence`) and the nearest
        other known storey is requested outright, ahead of a posterior that
        abstains by design. See the config comment for the measurement.
        """
        threshold = int(getattr(self.cfg.agent, "floor_disproved_after_failed_attempts", 0))
        if threshold <= 0:
            return
        floor = int(self.floors.current_id)
        n = self._failed_attempts_by_floor.get(floor, 0) + 1
        self._failed_attempts_by_floor[floor] = n
        if n < threshold or floor in self._disproved_floors:
            return
        self._disproved_floors.add(floor)
        self.stats["floors_disproved"] = self.stats.get("floors_disproved", 0) + 1
        levels = {int(k): float(v) for k, v in self.floors.estimator.levels.items()}
        here = levels.get(floor)
        others = [k for k in levels if k != floor and k not in self._disproved_floors]
        if here is None or not others:
            if self._request_new_storey("storey disproved, none other known"):
                return
            self.stats["floor_disproved_no_other"] = (
                self.stats.get("floor_disproved_no_other", 0) + 1)
            return
        other = min(others, key=lambda k: abs(levels[k] - here))
        self.exploration.forced_floor = int(other)
        self.exploration.requested_floor = int(other)
        self.stats["floor_disproved_requests"] = (
            self.stats.get("floor_disproved_requests", 0) + 1)
        self.stats["floor_disproved_to"] = int(other)

    def _request_new_storey(self, reason: str) -> bool:
        """Ask the operator for a storey this agent has never stood on.

        The robot cannot climb, and the map it restored may hold one storey,
        so when this one is finished the two floor requests above have
        nothing to name: the posterior scores storeys that have surfaces, and
        the disproved rule asks for the nearest KNOWN level. This names the
        lowest key nobody has stood on; the operator carries the robot and
        declares it on /osg/floor (docs/THOR.md), which `select` reads as
        arrival and clears the request. Only on the external floor source and
        only with `agent.request_new_storey_when_exhausted`; never a
        simulator behaviour.
        """
        if not bool(getattr(self.cfg.agent, "request_new_storey_when_exhausted", False)):
            return False
        if not bool(getattr(self.floors, "external", False)):
            return False
        if self.exploration.requested_floor is not None:
            return False  # already asking
        known = {int(k) for k in self.floors.known_levels()}
        key = 0
        while key in known or key in self._disproved_floors:
            key += 1
        self.exploration.forced_floor = int(key)
        self.exploration.requested_floor = int(key)
        self.stats["new_storey_requests"] = self.stats.get("new_storey_requests", 0) + 1
        self.stats["new_storey_request_reason"] = reason
        self._new_storey_reason = reason
        self._wait_at_stairs()
        return True

    def _retract_new_storey_request(self) -> None:
        """A frontier turned up after all: the storey was not finished.

        Only a request raised for exhaustion is withdrawn; a spent step budget
        stands. Without this the wish stood for the rest of the run, and
        because a standing wish holds the base still instead of turning, the
        map stopped growing and the frontier that had appeared was never
        looked at (outputs/20260922_231133, steps 88-127).
        """
        if getattr(self, "_new_storey_reason", None) != "no frontier left":
            return
        self.exploration.requested_floor = None
        self.exploration.forced_floor = None
        self._new_storey_reason = None
        self.stats.pop("stairs_wait_goal_ros", None)
        self.stats["new_storey_requests_retracted"] = (
            self.stats.get("new_storey_requests_retracted", 0) + 1)

    def _wait_at_stairs(self) -> bool:
        """Drive to the staircase the operator declared and wait there.

        `ros2.stairs_xy` is (x, y) in the robot's `map` frame, given at launch.
        The request above is the decision; this is where the robot goes to be
        carried: GOTO_FRONTIER is the one state that follows `_goal_xy`, and
        on arrival the FSM drops to EXPLORE, where `_waiting_for_carry` holds
        the base until /osg/floor arrives. Unset, the robot waits where it is.
        """
        xy = getattr(getattr(self.cfg, "ros2", None), "stairs_xy", None)
        if not xy:
            return False
        from ..ros2.frames import ros_xy_to_pipeline

        self._goal_xy = ros_xy_to_pipeline(float(xy[0]), float(xy[1]))
        self._current_path = None
        self.exploration.current_frontier = None
        self._candidate_id = None
        self.state = State.GOTO_FRONTIER
        self.stats["stairs_wait_goto"] = self.stats.get("stairs_wait_goto", 0) + 1
        self.stats["stairs_wait_goal_ros"] = [float(xy[0]), float(xy[1])]
        return True

    def _check_storey_budget(self) -> None:
        """`agent.request_new_storey_after_steps`, checked every step.

        Counted from the first selection round on this storey -- i.e. after
        the restored anchor has been walked to and tested, which is the part
        of the search pass a budget must not cut short -- and checked here
        rather than in `_explore` so it fires while a frontier is being driven
        to, not at the next EXPLORE round. Measured on the robot (budget 40):
        spent at step 96, acted on at 111, three frontiers later.
        """
        budget = int(getattr(self.cfg.agent, "request_new_storey_after_steps", 0) or 0)
        if budget <= 0 or self.state not in (State.EXPLORE, State.GOTO_FRONTIER):
            return
        here = int(self.floors.current_id)
        if self.exploration.steps_on_storey(self.step_count, here) >= budget:
            self._request_new_storey("storey step budget spent")

    def _take_operator_waypoint(self, world) -> bool:
        """`ros2.waypoint_xy`: the first exploration goal on the declared
        storey is where the operator said, facing `waypoint_yaw_deg`; the
        search resumes from there. Taken once per run."""
        ros = getattr(self.cfg, "ros2", None)
        xy = getattr(ros, "waypoint_xy", None)
        if not xy or getattr(self, "_waypoint_taken", False):
            return False
        want_floor = int(getattr(ros, "waypoint_floor", -1))
        if want_floor >= 0 and int(self.floors.current_id) != want_floor:
            return False
        from ..ros2.frames import ros_xy_to_pipeline

        goal = ros_xy_to_pipeline(float(xy[0]), float(xy[1]))
        yaw_deg = getattr(ros, "waypoint_yaw_deg", None)
        backend = getattr(getattr(self, "pointnav", None), "backend", None)
        if yaw_deg is not None and hasattr(backend, "set_goal_yaw"):
            backend.set_goal_yaw(goal, float(np.radians(float(yaw_deg))))
        self._waypoint_taken = True
        self._goal_xy = goal
        self._current_path = None
        self._candidate_id = None
        self.exploration.current_frontier = None
        self.exploration.note_progress(world)
        self.state = State.GOTO_FRONTIER
        self.stats["operator_waypoint"] = [float(xy[0]), float(xy[1]),
                                           None if yaw_deg is None else float(yaw_deg)]
        return True

    def _waiting_for_carry(self) -> bool:
        """Has the agent asked the operator for a storey it cannot reach?"""
        if not bool(getattr(self.floors, "external", False)):
            return False
        want = self.exploration.requested_floor
        return want is not None and int(want) != int(self.floors.current_id)

    def _storey_disproved(self, floor_id) -> bool:
        return int(floor_id) in self._disproved_floors

    def _presence_for_floor_evidence(self):
        """The `presence_of` handed to `floor_target_evidence` for the current
        storey: None or the belief test as configured -- unless this storey is
        disproved, in which case every target-labelled track on it is treated
        as not believed, so the _TARGET_PRESENT veto is lifted."""
        believed = (
            self._track_still_believed
            if bool(getattr(self.cfg.exploration, "floor_evidence_by_presence", False))
            else None
        )
        if not self._storey_disproved(self.floors.current_id):
            return believed
        self.stats["floor_veto_lifted"] = self.stats.get("floor_veto_lifted", 0) + 1
        return lambda _track_id: False

    def _may_preempt_pursuit(self, track) -> bool:
        """While a floor switch is being walked, only a candidate seen live,
        close, and confidently may interrupt it (config: protect_floor_switch)."""
        if self._storey_disproved(self.floors.current_id):
            # Candidates are filtered to the CURRENT storey, and this storey
            # has been disproved by the agent's own failed attempts. Measured
            # (outputs/mf5_pass2_v7 ep1): with only the live/close/confident
            # test, the ceiling-fixture false positive at score 0.82 passed it
            # the moment the agent walked under it on the way to the stairs,
            # and took the last attempt. A verdict from two walks beats one
            # more confident look at the same storey.
            self.stats["pursuit_preempt_blocked_disproved"] = (
                self.stats.get("pursuit_preempt_blocked_disproved", 0) + 1)
            return False
        centre = np.asarray(self.object_layer.center_of(track), dtype=float)[list(PLANE)]
        here = self._agent_xy if self._agent_xy is not None else centre
        near = float(np.linalg.norm(centre - here)) <= float(
            getattr(self.cfg.agent, "protect_floor_switch_range_m", 1.5))
        strong = float(getattr(track, "best_score", 0.0)) >= float(
            getattr(self.cfg.agent, "protect_floor_switch_min_score", 0.6))
        live = bool(getattr(track, "seen_live", False))
        ok = near and strong and live
        key = "pursuit_preempt_allowed" if ok else "pursuit_preempt_blocked"
        self.stats[key] = self.stats.get(key, 0) + 1
        return ok

    def _schedule_relook(self) -> None:
        """A stale stop just failed: the object is probably still on that
        surface. Remember the surface so the next attempt looks at it first,
        from somewhere other than where the agent stood."""
        if not bool(self.cfg.verification.relook_after_stale_stop):
            return
        if self._relook is not None or not self._stale_stop_used:
            return
        track = self.object_layer.get(self._candidate_id) if self._candidate_id is not None else None
        if track is None or not getattr(track, "from_prior", False) or track.seen_live:
            return
        here = None if self._agent_xy is None else np.asarray(self._agent_xy, dtype=float).copy()
        centre = np.asarray(self.object_layer.center_of(track), dtype=float)[list(PLANE)]
        key, radius, label = -int(track.id), _horizontal_radius_m(track), str(track.label)
        view = next((o for o in getattr(self.scene_graph, "objects", []) if int(o.track_id) == int(track.id)), None)
        cid = getattr(view, "container_id", None) if view is not None else None
        node = self.scene_graph.containers.get(int(cid)) if cid is not None else None
        if node is not None:
            from .close_look import _container_radius_m

            key = int(cid)
            centre = np.asarray(node.center, dtype=float)[list(PLANE)]
            radius = _container_radius_m(self, node)
            label = str(node.label)
        self._relook = (key, centre, radius, here, label)
        self.stats["relook_scheduled"] = self.stats.get("relook_scheduled", 0) + 1

    def _start_relook(self, frame: FrameData) -> Optional[str]:
        if self._relook is None or self.close_look.active:
            return None
        key, centre, radius, exclude_xy, label = self._relook
        self._relook = None
        self.stats["relook_started"] = self.stats.get("relook_started", 0) + 1
        self.close_look.start(key, centre, radius, "explore", reason="relook",
                              label=label, exclude_xy=exclude_xy)
        return self.close_look.step(frame)

    def act(self, frame: FrameData) -> str:
        self.step_count += 1
        prev_state = self.state
        with self.profiler.timeit("control_loop"):
            action = self._act_inner(frame)
            if action == STOP_ACTION and self.state is State.DONE:
                # Every STOP passes here: the approach, the close look and the
                # stale-anchor stop all end in DONE. Off by default.
                instead = self.stop_gate.check(frame)
                if instead is not None:
                    action = instead
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
        with self.profiler.timeit("floors"):
            floor_y = self.floors.observe(frame, self.step_count)
        self._check_storey_budget()
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
                clear_m = float(getattr(self.cfg.mapping, "footprint_clear_m", 0.0) or 0.0)
                if clear_m > 0.0:
                    n_cleared = self.costmap.clear_footprint(self._agent_xy, clear_m)
                    self.stats["footprint_cells_cleared"] = (
                        self.stats.get("footprint_cells_cleared", 0) + n_cleared)
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
            if self.feature_memory is not None:
                # Mechanism counters ride out with the rest; on this benchmark
                # SR cannot resolve anything under about three trials, so an arm
                # that cannot be seen in a counter cannot be judged at all.
                self.stats.update(self.feature_memory.counters)
            if self.cfg.exploration.search_posterior:
                with self.profiler.timeit("glance"):
                    self.exploration.glance(self._world(frame))
            with self.profiler.timeit("close_look"):
                self.close_look.maybe_opportunistic(frame)

        down_look = self.stairs._down_look(frame, self.floor_layer, off_map)
        if down_look is not None:
            return down_look

        # Candidate target check happens in every state except terminal ones
        if self.state in (State.INIT, State.EXPLORE, State.GOTO_FRONTIER) or (
            self.state is State.CLOSE_LOOK and self.close_look.resume == "explore"
        ):
            if self.step_count <= self._explore_hold_until:
                # A failed attempt is holding the commit open so one real
                # exploration round can happen (`rearm`). A hold, not a ban.
                self.stats["explore_hold_steps"] = (
                    self.stats.get("explore_hold_steps", 0) + 1)
            elif self._storey_disproved(self.floors.current_id):
                # Candidates are filtered to the current storey, and this
                # storey's own failed attempts have disproved it -- pursuit or
                # no pursuit. Measured (outputs/mf5_pass2_v11 ep1): the climb
                # ended on its step budget at -1.51 m, `pursuing` went False,
                # and three steps later the agent committed to an upper-storey
                # fake at y=4.0 while standing on the stairs, spending its last
                # attempt on the storey it had just left.
                self.stats["candidates_skipped_disproved"] = (
                    self.stats.get("candidates_skipped_disproved", 0) + 1)
            else:
                admit = None
                if (
                    bool(getattr(self.cfg.agent, "protect_floor_switch", False))
                    and bool(getattr(self.floors, "pursuing", False))
                ):
                    admit = self._may_preempt_pursuit
                with self.profiler.timeit("candidates"):
                    self.candidates.check(frame.camera_position[list(PLANE)], admit=admit)
        if self.close_look.active and self.state is not State.CLOSE_LOOK:
            self.close_look.interrupted()  # a commit pre-empted the look
        if self.state == State.CLOSE_LOOK:
            return self.close_look.step(frame)

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
                relook = self._start_relook(frame)
                if relook is not None:
                    return relook
                # Look at the surface before the selection round scores it
                # searched -- the selection round retires it on arrival, and
                # the belief update should rest on a frame that shows it.
                facing = self.exploration.face_surface(self._world(frame))
                if facing is not None:
                    return facing
                self._explore(frame)
            if self.state == State.EXPLORE:  # nothing selectable
                if self._waiting_for_carry():
                    return WAIT_ACTION  # at the stairs; the operator's move
                return TURN_ACTION  # keep looking around; map will grow

        if self.state == State.CLIMB:
            return self.climb._do_climb(frame)

        if self.state == State.GOTO_FRONTIER:
            if (
                bool(getattr(self.cfg.agent, "climb_enabled", False))
                and self.floors.pursuing
                and self._goal_xy is not None
                and self.climb._at_the_stairs(frame)
            ):
                self.climb._start_climb(frame)
                return self.climb._do_climb(frame)
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
        named_dets = list(dets or [])
        dets = self._propose_regions(frame, dets)
        if self.on_keyframe_detections is not None:
            # The ground-truth view counts what the DETECTOR saw; proposals
            # have their own counters.
            self.on_keyframe_detections(frame, named_dets)
        with self.profiler.timeit("object_layer"):
            self.object_layer.update(frame, dets, floor_key=self.floors.current_id)
        self.keyframes.add(frame)
        # Stair evidence used to accumulate ONLY on the one frame of a periodic
        # look-down, and `down_look_every` defaults to 0 -- so on every ycb
        # preset `StairDetector.accumulate` never ran, `up_stair_hits` and
        # `down_stair_hits` stayed empty, `_on_a_staircase` was always False
        # and `StairDetector.extract` had no caller at all. The detector emits
        # `stairs` masks on ordinary keyframes; stamp them here.
        if (
            bool(getattr(self.cfg.agent, "stair_evidence_every_kf", False))
            and self.stair_detector is not None
            and not self.floors.on_stairs
        ):
            with self.profiler.timeit("stairs"):
                self.stair_detector.accumulate(
                    frame, self.floor_layer, dets, self.stairs._seg_stair_mask(frame)
                )

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
        if self._take_operator_waypoint(world):
            return
        choice = self.exploration.select(
            world,
            floor_switch=lambda cost, target_floor=None: self._try_floor_switch(
                frame, cost, target_floor=target_floor
            ),
        )
        here = int(self.floors.current_id)
        on_storey = self.exploration.steps_on_storey(self.step_count, here)
        # Not on the first round after a carry: the new storey's grid holds
        # only what the camera has seen since the switch, and "no frontier" on
        # a near-empty grid is "unmapped", not "finished".
        settled = on_storey >= int(getattr(self.cfg.exploration, "select_every", 5))
        if choice is None and settled and self.exploration.storey_exhausted(here):
            self._request_new_storey("no frontier left")
        elif choice is not None and self._waiting_for_carry():
            self._retract_new_storey_request()
        if self.state == State.GOTO_FRONTIER:
            return  # the request sent it to the stairs; the frontier can wait
        if choice is None:
            return
        self._goal_xy = choice.goal_xy
        self._current_path = choice.path
        self.exploration.note_progress(world)
        self.state = State.GOTO_FRONTIER

    def _track_still_believed(self, track_id: int) -> bool:
        """Is this mapped instance still believed to be where the map put it?

        The floor gate's question, answered by the presence filter rather than
        by the label alone. A track the agent has already walked to and not
        found (`absence_arrivals`) is not believed however high its log-odds
        happens to sit, because that arrival is the direct test.
        """
        track = self.object_layer.get(int(track_id))
        if track is None:
            return False
        if int(getattr(track, "absence_arrivals", 0)) > 0:
            return False
        presence = getattr(track, "presence", None)
        if presence is None:
            return True
        return float(presence.p) >= float(self.cfg.scene_graph.presence.min_presence)

    def _llm_floor_choice(self, target_floor: Optional[int]):
        """Let the text model choose the storey the posterior just asked for.

        Only for a DIRECTED request, which in the fused pipeline is the point at
        which the agent has tested the stale anchor and found the object gone.
        That is the one moment in an episode when "which floor is it on now?" is
        both open and worth a call -- as opposed to ASCENT's every-60-steps
        cadence, which asks it while the agent is still walking to a pose the
        map is confident about.

        Returns the floor key to head for, None to leave the request alone, or
        False for "the model says stay here".

        Until now `FloorDecisionPlanner.decide` had no caller outside its tests,
        `_floor_goal_dir` was assigned 0 at construction and never again, and
        `_floor_direction_boost` was never called -- so every preset claiming
        `floor_llm: true` measured nothing. This is that wiring.
        """
        if target_floor is None or self.floor_planner is None:
            return target_floor
        if not bool(getattr(self.cfg.exploration, "floor_llm", False)):
            return target_floor
        if self._storey_disproved(self.floors.current_id):
            # The model reads the scene graph, and on a disproved storey the
            # graph's target entries are the very false positives the attempts
            # just disproved. Measured (outputs/mf5_pass2_v7 ep1, step 158):
            # asked, it answered "stay -- Floor 2 explicitly lists 'toy
            # airplane' among its contained objects", vetoing a request the
            # agent had earned by walking there twice.
            self.stats["floor_llm_skipped_disproved"] = (
                self.stats.get("floor_llm_skipped_disproved", 0) + 1)
            return target_floor
        stack = self.floors.stack
        asked_before = int(getattr(self.floor_planner, "asks", 0))
        direction = self.floor_planner.decide(
            self.target, stack, self.scene_graph, self.step_count,
        )
        self.stats["floor_llm_asks"] = int(getattr(self.floor_planner, "asks", 0))
        self.stats["floor_llm_moves"] = int(getattr(self.floor_planner, "moves", 0))
        self.stats["floor_llm_blocked_throttle"] = int(
            getattr(self.floor_planner, "blocked_throttle", 0))
        self.stats["floor_llm_blocked_too_soon"] = int(
            getattr(self.floor_planner, "blocked_too_soon", 0))
        self.stats["floor_llm_blocked_one_floor"] = int(
            getattr(self.floor_planner, "blocked_one_floor", 0))
        if direction is None:
            # Two very different things: the gate refused to spend a call, or a
            # call was made and its answer was unusable. `asks` only moves in
            # the second case, so the difference is recoverable.
            key = ("floor_llm_no_answer" if asked_before < self.stats["floor_llm_asks"]
                   else "floor_llm_not_asked")
            self.stats[key] = self.stats.get(key, 0) + 1
            return target_floor
        self._floor_goal_dir = int(direction)
        if direction == 0:
            self.stats["floor_llm_stay"] = self.stats.get("floor_llm_stay", 0) + 1
            # Log the refusal too. "Stay" is the interesting answer: it is the
            # one that overrides the posterior by doing nothing, and without it
            # in the record an arm that talked the agent out of every correct
            # floor change looks identical to one that was never asked.
            self.floor_llm_log.append({
                "step": int(self.step_count),
                "from_floor": int(stack.current_id),
                "posterior": int(target_floor),
                "direction": 0,
                "chosen": int(stack.current_id),
                "reason": str(getattr(self.floor_planner, "last_reason", ""))[:200],
            })
            return False
        chosen = stack.up() if direction > 0 else stack.down()
        if chosen is None:
            # The model wants a storey the stack has never allocated. Nothing to
            # aim at, so the posterior's own answer stands.
            self.stats["floor_llm_no_such_floor"] = (
                self.stats.get("floor_llm_no_such_floor", 0) + 1
            )
            return target_floor
        self.floor_llm_log.append({
            "step": int(self.step_count),
            "from_floor": int(stack.current_id),
            "posterior": int(target_floor),
            "direction": int(direction),
            "chosen": int(chosen.key),
            "reason": str(getattr(self.floor_planner, "last_reason", ""))[:200],
        })
        if int(chosen.key) != int(target_floor):
            self.stats["floor_llm_override"] = (
                self.stats.get("floor_llm_override", 0) + 1
            )
        else:
            self.stats["floor_llm_agreed"] = self.stats.get("floor_llm_agreed", 0) + 1
        return int(chosen.key)

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
        if (
            bool(getattr(self.cfg.floor, "hold_pursuit", False))
            and self.floors.pursuing
            and self.floors.pursuit_ok(frame, self.step_count, self._goto_deadline)
        ):
            # A directed request comes back every selection round, and until
            # now each one re-ran `try_switch`, which restarted the pursuit:
            # `_portal_step` and `_portal_start_y` reset every 5 steps, so the
            # grace window and the deadline never elapsed, `end_pursuit` never
            # ran, no failure was ever remembered, and the same target was
            # chosen again -- 55 times on base, 82 on v9. A pursuit that is
            # still making its case is left to make it.
            self.stats["floor_switch_reissue_suppressed"] = (
                self.stats.get("floor_switch_reissue_suppressed", 0) + 1
            )
            return True
        target_floor = self._llm_floor_choice(target_floor)
        if target_floor is False:
            return False  # the model said stay; this round is settled
        stair_xyz = None
        if str(getattr(self.cfg.floor, "climb_targets", "portals")) == "stairs_first":
            from ..mapping.stairs import stair_tracks
            here_y = float(self.floors.height_of(self.floors.current_id))
            # (xy, kind): kind is "up", "down", or None when unknown. A semantic
            # `stairs` track's 3D centre sits mid-flight, so its height relative
            # to this floor says which way the flight goes.
            stair_xyz = []
            for c in stair_tracks(
                self.object_layer,
                min_obs=int(self.cfg.floor.stair_min_obs),
                min_evidence=float(self.cfg.floor.stair_min_evidence),
            ):
                c = np.asarray(c, dtype=float)
                if abs(float(c[1]) - here_y) >= float(self.cfg.floor.new_level_m):
                    continue  # a staircase on another storey
                rel = float(c[1]) - here_y
                kind = "up" if rel > 0.3 else ("down" if rel < -0.3 else None)
                stair_xyz.append((c[list(PLANE)], kind))
            n_sem = len(stair_xyz)
            # The detector's own accumulated evidence, which until now was
            # stamped and never read back.
            if self.stair_detector is not None and self.floor_layer.up_stair_hits is not None:
                for det in self.stair_detector.extract(self.floor_layer):
                    stair_xyz.append((np.asarray(det.centroid_xy, dtype=float), det.kind))
            self.stats["stair_tracks_offered"] = max(
                self.stats.get("stair_tracks_offered", 0), n_sem)
            self.stats["stair_regions_offered"] = max(
                self.stats.get("stair_regions_offered", 0), len(stair_xyz) - n_sem)
        if self._switches_exhausted_here():
            return False
        with self.profiler.timeit("floor_switch"):
            portal = self.floors.try_switch(
                frame, self.step_count, best_path_cost,
                self.scene_graph, self.target, self._reachable_fn,
                target_floor=target_floor,
                presence_of=self._presence_for_floor_evidence(),
                stair_xyz=stair_xyz,
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

    def _seen_this_approach(self) -> bool:
        """Was the target confirmed visible during this approach, closely
        enough for that to protect the stop from the absence reading?

        `verification.seen_counts_within_m` is the range beyond which a
        sighting does not count; 0 is the shipped rule, any range. Measured
        on the robot (outputs/20260922_191444): one 0.36-score "sports ball"
        at 4.9 m, 34 steps before arrival, and the run STOPped at an empty
        spot while both ball tracks' beliefs had already fallen to 0.44 --
        the reading that would have abandoned was never taken because of
        that one far frame. Detector recall beyond 3 m is 0.24-0.26
        (`absence_max_range_m`), so a far sighting is a weak claim either way.
        """
        if self.approach.last_good_xy is None:
            return False
        lim = float(getattr(self.cfg.verification, "seen_counts_within_m", 0.0) or 0.0)
        rng = getattr(self.approach, "last_good_range_m", None)
        if lim > 0.0 and rng is not None and float(rng) > lim:
            self.stats["far_sighting_ignored"] = self.stats.get("far_sighting_ignored", 0) + 1
            return False
        return True

    def _absence_at_arrival(self, frame: FrameData, reason: str,
                            from_look: bool = False) -> Optional[str]:
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
        if track is None or self._seen_this_approach():
            return None
        if self._stale_stop_pending and not from_look:
            # Granted at the end of the close look; the approach walked back to
            # its ring and this is the arrival the STOP was promised for.
            self._stale_stop_pending = False
            return None
        if (
            bool(self.cfg.verification.stop_at_stale_anchor_once)
            and not self._stale_stop_used
            and bool(getattr(track, "from_prior", False))
            and not track.seen_live
        ):
            # The stale anchor is the answer more often than not, and a stop
            # here costs one attempt of three. Returning None lets the
            # approach STOP; the protocol scores it and, if it fails, applies
            # the negative reading (eval/attempts.py). From the close look the
            # agent stands on the 1.5 m ring, too far to score, so the look's
            # own path re-approaches the tight ring and the STOP is taken on
            # that arrival instead (`_stale_stop_pending`).
            self._stale_stop_used = True
            self.stats["stale_anchor_stop"] = self.stats.get("stale_anchor_stop", 0) + 1
            if (bool(self.cfg.verification.stale_stop_at_nearest_free)
                    and not bool(getattr(self.approach, "closed", False))):
                # (skipped when the approach already closed the last metre on
                # the navmesh, which gets nearer than the costmap's free cell)
                # Not here: as close to the old position as the furniture allows.
                from ..mapping.costmap import nearest_free_xy

                centre = np.asarray(self.object_layer.center_of(track), dtype=float)[list(PLANE)]
                goal = np.asarray(nearest_free_xy(self.costmap, centre), dtype=float)
                here = frame.camera_position[list(PLANE)]
                if float(np.linalg.norm(goal - here)) > 0.2:
                    self._goal_xy = goal
                    self._current_path = None
                    self.approach.path_goal = None
                    self.approach.at_viewpoint = False
                    self.approach.steps_left = max(int(self.approach.steps_left), 60)
                    self._goto_deadline = max(int(self._goto_deadline), self.step_count + 60)
                    self._stale_stop_pending = True
                    self.stats["stale_stop_reaimed"] = self.stats.get("stale_stop_reaimed", 0) + 1
                    self.state = State.APPROACH
                    return TURN_ACTION
            self._stale_stop_pending = bool(from_look)
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
        """The detector's best detection of the target on this frame, or None.

        This is the choke point the approach stop, the close look and the
        exploration check all ask, and -- through the close look -- it is what
        decides that a committed track is ABSENT. Asking only the label path
        there is circular for exactly the objects the proposal stage exists
        for: the detector that could not name the object is re-asked whether
        the object is present, says no, and the track is retired.

        Measured on in_anchor__0117__banana: the agent walked to the banana,
        looked from 1.5 m, was told "not detected", dropped the track's belief
        from 0.95 to 0.433 and committed elsewhere -- in an episode where the
        proposal stage had already admitted 16 regions.
        """
        target = normalize_label(self.target)
        with self.profiler.timeit("detector"):
            dets = self.detector.detect(frame.rgb)
        matches = [
            d for d in dets
            if normalize_label(d.label) == target and d.score > 0.25
        ]
        if matches:
            return max(matches, key=lambda d: d.score)
        # The fallthrough is for the track whose identity rests on appearance.
        # This function also stops and steers every approach, ends the close
        # look and gates the terminal stop, so a proposal answering it while
        # the agent works a track the DETECTOR named lets a region on the
        # wrong object do all of that. Measured with the stage on every
        # keyframe: a named red plate approach stopped at 1.52 m on a region,
        # a tin can scored a false stop at step 46 on its prior-map track.
        # A proposal-only track gets the proposal sensor; a named one gets the
        # detector, exactly as before the stage existed.
        if not self._working_a_proposal_track():
            return None
        return self._region_detection(frame)

    def _working_a_proposal_track(self) -> bool:
        cid = getattr(self, "_candidate_id", None)
        if cid is None:
            return False
        track = self.object_layer.get(int(cid))
        return bool(track is not None and getattr(track, "proposal_only", False))

    def _region_active(self) -> bool:
        """Is the proposal stage a fallback for THIS episode?

        Per-frame gating was the defect: on a trial where the detector works the
        target is absent from most individual frames, so a per-frame test fires
        on nearly all of them. Measured on the full 107, that took the 21
        perception trials from 0 to 6 and the other 86 from 57 to 41, with 13 of
        the 18 lost trials exhausting all three attempts on regions the stage
        had admitted.

        Episode-level instead: once the label path has named the target even
        once, the detector can see this object and the stage stays off.
        """
        rp = self.region_proposer
        if rp is None:
            return False
        cfg = rp.cfg
        if self._region_admits >= int(cfg.max_per_episode):
            return False
        if bool(getattr(cfg, "every_keyframe", False)):
            return True
        if bool(getattr(cfg, "require_never_named", False)) and self._region_named:
            return False
        return self._region_kf >= int(getattr(cfg, "unnamed_keyframes", 0) or 0)

    def _region_detection(self, frame: FrameData) -> Optional[Detection]:
        """The proposal stage's answer for this frame, computed at most once.

        Cached on the frame id because this is called several times per step --
        the approach stop, the close look and the absence sensor each ask -- and
        a segment-plus-encode per call would multiply the stage's cost by the
        number of askers rather than by the number of frames.
        """
        rp = self.region_proposer
        if rp is None or self.target is None or not bool(rp.cfg.use_for_absence):
            return None
        if not self._region_active():
            return None
        key = int(getattr(frame, "frame_id", -1))
        cached = getattr(self, "_region_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        with self.profiler.timeit("region_proposal"):
            det = rp.propose(frame.rgb)
        self._region_cache = (key, det)
        self.stats.update(rp.counters)
        return det

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

    # ------------------------------------------------------------------ climb
    #
    # ASCENT's staircase behaviour (`ascent_policy.py:1069-1189`), fitted to this
    # FSM. Three phases there: get close to the stair frontier, drive at its
    # centroid, then steer at the farthest depth ray -- the "carrot" -- which on
    # a flight of stairs points up it. Arrival is "off the stairs having been on
    # the centroid"; here it is the floor estimator committing a new storey,
    # which `FloorPolicy.observe` already turns into `end_pursuit("arrived")`.
    #
    # Why this exists: `portals.py` says a portal is "a place to walk toward,
    # after which the navmesh handles the climb", and commit 866ab0e removed
    # the navmesh. Measured on 00821's cracker box, base made 55 switch
    # attempts and rose 0.17 m; with the navmesh restored, one attempt and the
    # whole 3.6 m storey. This is the sensor-only replacement for that step.



    def _switches_exhausted_here(self) -> bool:
        """Has this storey refused to be left often enough to stop asking?

        Banning the switch is not enough on its own: the search posterior and
        the storey LLM keep raising the same request every round, so the
        standing request is cleared with it (config:
        agent.max_failed_switches_per_storey).
        """
        limit = int(getattr(self.cfg.agent, "max_failed_switches_per_storey", 0) or 0)
        if limit <= 0:
            return False
        here = int(self.floors.current_id)
        if self._failed_switches_by_floor.get(here, 0) < limit:
            return False
        ban_steps = int(getattr(self.cfg.agent, "switch_ban_steps", 0) or 0)
        since = self.step_count - int(self._switch_banned_at.get(here, self.step_count))
        if ban_steps > 0 and since >= ban_steps:
            # Served. One more go, and the counter starts again from here.
            self._failed_switches_by_floor[here] = 0
            self._switch_banned_at.pop(here, None)
            self.stats["floor_switch_ban_lifted"] = (
                self.stats.get("floor_switch_ban_lifted", 0) + 1)
            return False
        self._switch_banned_at.setdefault(here, self.step_count)
        self.stats["floor_switch_banned_after_failures"] = (
            self.stats.get("floor_switch_banned_after_failures", 0) + 1)
        self.exploration.forced_floor = None
        self.exploration.requested_floor = None
        return True

    def _note_failed_switch(self) -> None:
        """One more storey exit that did not happen."""
        here = int(self.floors.current_id)
        self._failed_switches_by_floor[here] = self._failed_switches_by_floor.get(here, 0) + 1
        self.stats["failed_switches_here"] = self._failed_switches_by_floor[here]














    def _propose_regions(self, frame: FrameData, dets):
        """Add a class-agnostic proposal for the target, when one is warranted.

        The stage returns an ordinary `Detection` under the target label, so
        admission, the presence filter, the identity channel, the candidate
        gate, the VLM and the attempt protocol all judge it exactly as they
        judge the detector's own output. Nothing here can stop an approach or
        score a trial by itself.

        It runs only when the detector produced nothing for the target on this
        keyframe -- the case it exists for -- and at most
        `max_per_episode` times, because at the measured 85% precision an
        unbounded stage would write a lot of wrong tracks.
        """
        rp = self.region_proposer
        if rp is None or self.target is None:
            return dets
        cfg = rp.cfg
        self._region_kf += 1
        want = normalize_label(self.target)
        named_now = any(normalize_label(d.label) == want for d in dets or [])
        if named_now:
            # The detector can see this object. Whatever else is true of the
            # episode, it does not need a fallback -- and a fallback that runs
            # anyway competes with a detector that was about to succeed.
            self._region_named = True
        if not self._region_active():
            return dets
        if bool(cfg.only_when_unnamed) and named_now:
            return dets
        key = int(getattr(frame, "frame_id", -1))
        cached = getattr(self, "_region_cache", None)
        if cached is not None and cached[0] == key:
            det = cached[1]
        else:
            with self.profiler.timeit("region_proposal"):
                det = rp.propose(frame.rgb)
            self._region_cache = (key, det)
        self.stats.update(rp.counters)
        if det is None:
            return dets
        self._region_admits += 1
        self.stats["region_admits"] = self._region_admits
        return list(dets or []) + [det]

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



    def _floor_frozen(self, frame: FrameData) -> bool:
        if self.state is State.CLIMB and bool(
            getattr(self.cfg.mapping, "freeze_floor_in_climb", False)
        ):
            return True
        if bool(getattr(self.cfg.mapping, "freeze_floor_on_stairs", False)):
            return self.stairs._on_a_staircase(frame.camera_position[list(PLANE)])
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
        obj_radius_m = 0.0
        if self._candidate_id is not None:
            track = self.object_layer.get(self._candidate_id)
            if track is not None:
                if here is not None:
                    nearest = self.object_layer.nearest_point_xy(track, here)
                    self._target_cloud_xy = (
                        None if nearest is None else np.asarray(nearest, dtype=float).copy()
                    )
                obj_radius_m = _horizontal_radius_m(track)
        self.approach.start(obj_xy, agent_xy, floor_y, obj_radius_m=obj_radius_m)

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
            # `.step` rather than `__call__`: the action is the same, but the
            # REASON is the only way to tell "I am there" from "the network gave
            # up" from "I stopped closing", and an arm whose whole question is
            # why approaches do not terminate cannot be read without it.
            with self.profiler.timeit("mover"):
                step = self.pointnav.step(
                    goal_xy, creep_below=creep,
                    stop_radius=float(self.cfg.agent.pointnav_arrival_m),
                )
            self.stats[f"pointnav_{step.reason}"] = (
                self.stats.get(f"pointnav_{step.reason}", 0) + 1
            )
            # How often the goal moved far enough to wipe the policy's
            # recurrent state. Tracked by the driver since it was written and
            # never written down, which is why the climb's slowness could only
            # be guessed at (config: climb_carrot_hold_m).
            self.stats["pointnav_resets"] = int(
                getattr(self.pointnav, "n_resets", 0) or 0)
            return step.action
        return self.approach.follow_to(frame, goal_xy)
