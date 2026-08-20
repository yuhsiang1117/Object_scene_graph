"""NavAgent: the full pipeline behind a single act(frame) -> action call.

State machine:
    INIT (360 scan) -> EXPLORE <-> GOTO_FRONTIER
                           |  candidate found
                           v
                  GOTO_VERIFY_VIEW -> VERIFYING --accept--> APPROACH -> STOP
                           ^                |                  |
                           |                +--reject--> blacklist, EXPLORE
                           +---------------------(retreat if visibility lost)

APPROACH walks toward the verified object one step at a time, checking with
the detector on every step: stop when the detection's bbox is large enough
(close and clearly visible) or, if a step carries the agent out of view,
retreat to the last pose that was confirmed visible and stop there. This
targets HM3D's success metric directly — distance-to-goal is measured
against the episode's view_points set (poses from which the object is
actually visible), not against the object's raw 3D position, so a
distance-only stopping rule can land just outside that set even when the
agent is standing right next to the object (see docs/DESIGN_AND_ROADMAP.md
P0->P1 history: three distance-based strategies all stalled at dtg
0.107-0.147 m).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from ..core.profiler import Profiler
from ..core.types import Detection, FrameData
from ..exploration.async_scorer import AsyncScorer
from ..exploration.search_belief import (
    InspectionLog,
    build_container_candidates,
    select_candidate,
)
from ..exploration.selector import frontier_goal_xy, select_frontier
from ..graph.priors import floor_target_evidence
from ..graph.scene_graph import SceneGraph
from ..mapping.costmap import PLANE, Costmap2D
from ..mapping.floor_stack import FloorStack
from ..mapping.floors import FloorEstimator
from ..mapping.frontier import Frontier, FrontierExtractor
from ..mapping.portals import FloorSwitchPolicy, find_portals
from ..mapping.stairs import apply_stair_mask, detect_stairs, stair_tracks
from ..objects.object_layer import ObjectLayer
from ..perception.detector import Detector
from ..perception.keyframe import KeyframeSelector, KeyframeStore
from ..planning.controller import WaypointController
from ..planning.planner import PlanResult
from ..planning.voronoi_planner import HybridVoronoiPlanner
from ..verification.viewpoint import ViewpointPlanner

STOP_ACTION = "stop"
TURN_ACTION = "turn_left"


def target_vocabulary(target: str, vocabulary) -> List[str]:
    """Target first, then the generic list with anything that COLLIDES removed.

    Measured on the YCB benchmark: with the target "cracker box" the vocabulary
    also offered the generic "box", and YOLOE labelled every sighting "box" --
    263 mapped tracks, 3 of them the target, none of them proposable, because
    candidates() matches on the target category. The specific class was in the
    vocabulary and still never won.

    So drop a generic entry that is a whole-word part of the target ("box" for
    "cracker box"), and drop an exact duplicate of the target. Anything that is
    not a sub-phrase of the target is left alone -- this narrows the vocabulary
    only where it was actively competing with the goal.
    """
    target = str(target).replace("_", " ").strip()
    words = target.lower().split()
    out = [target]
    for entry in vocabulary:
        text = str(entry).replace("_", " ").strip()
        low = text.lower()
        if low == target.lower():
            continue
        parts = low.split()
        n = len(parts)
        if n < len(words) and any(words[i:i + n] == parts for i in range(len(words) - n + 1)):
            continue
        out.append(text)
    return out


def _make_affinity(cfg):
    """Where does this class of object get put down? None unless asked for.

    graph/priors.py has a hand-written table for the categories this project
    has cared about; every YCB target is missing from it, and a prior of "no
    idea" makes the search posterior fall back to a flat weight over every
    surface in the house -- the undirected wandering C3 exists to replace.
    """
    ec = getattr(cfg, "exploration", None)
    if ec is None or not getattr(ec, "affinity_llm", False):
        return None
    from ..graph.containers import CONTAINER_CATEGORIES
    from ..llm.affinity import AffinityProvider
    from ..llm.client import ChatClient

    client = None
    if getattr(cfg.llm, "api_key", ""):
        client = ChatClient(
            cfg.llm.base_url, cfg.llm.text_model, cfg.llm.api_key,
            cfg.llm.timeout_s, cfg.llm.max_image_px, cfg.llm.send_response_format,
        )
    return AffinityProvider(
        client, sorted(CONTAINER_CATEGORIES),
        cache_path=str(getattr(ec, "affinity_cache", "") or "") or None,
    )


def _make_presence_filter(cfg):
    """None unless scene_graph.presence.enabled -- the filter must be an opt-in
    A/B, not a silent default (docs/DYNAMIC_SCENES.md, Phase 1)."""
    pc = getattr(cfg.scene_graph, "presence", None)
    if pc is None or not getattr(pc, "enabled", False):
        return None
    from ..objects.presence import PresenceFilter, RecallModel

    recall = (
        RecallModel.load(pc.recall_model_path, constant=pc.recall_constant)
        if pc.recall_model_path
        else RecallModel(constant=pc.recall_constant)
    )
    return PresenceFilter(
        recall=recall,
        q_false_alarm=pc.q_false_alarm,
        l_clamp=pc.l_clamp,
        l_clamp_pos=getattr(pc, "l_clamp_pos", 3.0),
        occ_ratio_max=pc.occ_ratio_max,
        depth_tol_m=pc.depth_tol_m,
        # Expectation shares the ADMISSION threshold by construction: expecting
        # detections at a size the layer would have discarded biases the filter.
        min_area_px=cfg.scene_graph.min_det_bbox_px,
        range_m=tuple(pc.range_m),
        img_inside_frac=pc.img_inside_frac,
        min_depth_samples=pc.min_depth_samples,
        max_samples=pc.max_samples,
        max_tracks=pc.max_tracks,
        z_overlap_iou=pc.z_overlap_iou,
        log_path=pc.log_path,
    )


class State(Enum):
    INIT = "init"
    EXPLORE = "explore"
    GOTO_FRONTIER = "goto_frontier"
    GOTO_VERIFY_VIEW = "goto_verify_view"
    VERIFYING = "verifying"
    APPROACH = "approach"
    DONE = "done"


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
            nav_fn is not None and bool(getattr(cfg.agent, "use_habitat_navmesh", False))
        )
        # Terminal-view verification mode: skip the pre-approach best_crop VLM
        # call and instead verify the live close-up frame at the STOP decision
        # (see _do_approach). Requires a verifier; no-op when verifier is None.
        self._terminal_verify = (
            verifier is not None and bool(getattr(cfg.verification, "terminal", False))
        )
        self.profiler = profiler or Profiler()
        # Debug hook: if set, called with (frame, dets) every keyframe right
        # after the detections that feed object_layer.update() are computed
        # -- lets diagnostics observe exactly what the scene graph is built
        # from without duplicating the keyframe-timing logic. None by default
        # (zero cost, never called).
        self.on_keyframe_detections = None

        # One costmap per storey. With floor.per_floor_costmap off the stack
        # holds exactly one layer forever and `self.costmap` is that single map,
        # so the single-floor code path is byte-identical.
        _f = getattr(cfg, "floor", None)
        self._stairs_on = bool(getattr(_f, "stairs", False))
        # Drive frontier goals to the free-snapped centroid rather than an
        # UNKNOWN cell. Only meaningful on the navmesh, where unknown space
        # gets snapped unpredictably; the costmap planner treats unknown as
        # traversable and is unaffected either way.
        self._frontier_goal_free = bool(
            getattr(cfg.exploration, "frontier_goal_free_cell", False)
        )
        self._frontier_cost_free = bool(
            getattr(cfg.exploration, "frontier_cost_free_cell", False)
        )
        self._cross_floor_on = bool(getattr(_f, "cross_floor", False))
        self._floor_stack = FloorStack(
            resolution_m=cfg.mapping.resolution_m,
            room_seg_kwargs=dict(
                min_room_radius_m=cfg.scene_graph.room_min_radius_m,
                door_width_m=cfg.scene_graph.room_door_width_m,
            ),
            # The height layer is 4x the grid, so only pay for it when the
            # stair detector will actually read it.
            track_height=self._stairs_on or self._cross_floor_on,
        )
        # Which storey the agent is on (docs/MULTI_FLOOR.md). Constructed
        # unconditionally so the estimate is always logged; whether it FEEDS
        # the costmap is gated by floor.enabled / floor.estimate_only.
        fcfg = getattr(cfg, "floor", None)
        self.floors = FloorEstimator(
            camera_height=cfg.agent.camera_height,
            level_tol_m=getattr(fcfg, "level_tol_m", 0.35),
            merge_m=getattr(fcfg, "merge_m", 0.6),
            new_level_m=getattr(fcfg, "new_level_m", 1.8),
            min_dwell_steps=getattr(fcfg, "min_dwell_steps", 6),
            min_horizontal_run_m=getattr(fcfg, "min_horizontal_run_m", 2.5),
        )
        # Cross-floor exploration needs the height layer to see portals, so it
        # implies track_height even when stair detection is off.
        self._switch_policy = (
            FloorSwitchPolicy(
                max_steps=cfg.agent.max_steps,
                near_frontier_m=getattr(fcfg, "near_frontier_m", 4.0),
                min_interval_steps=getattr(fcfg, "switch_min_interval", 50),
                no_switch_before=getattr(fcfg, "no_switch_before", 50),
                no_switch_after_frac=getattr(fcfg, "no_switch_after_frac", 0.7),
                use_target_evidence=getattr(fcfg, "use_target_evidence", True),
                early_switch_step=getattr(fcfg, "early_switch_step", 30),
                min_objects_to_judge=getattr(fcfg, "min_objects_to_judge", 8),
                strong_evidence=getattr(fcfg, "strong_evidence", 2),
                evidence_patience_steps=getattr(fcfg, "evidence_patience_steps", 120),
            )
            if getattr(fcfg, "cross_floor", False) else None
        )
        self.frontier_extractor = FrontierExtractor(
            min_cells=cfg.exploration.frontier_min_cells,
            dedup_m=cfg.exploration.frontier_dedup_m,
        )
        self.object_layer = ObjectLayer(
            assoc_score_thresh=cfg.scene_graph.assoc_score_thresh,
            assoc_depth_gate_m=cfg.scene_graph.assoc_depth_gate_m,
            assoc_category_gate=cfg.scene_graph.assoc_category_gate,
            min_obs_for_refine=cfg.scene_graph.min_obs_for_refine,
            refine_every=cfg.scene_graph.refine_every,
            refine_max_center_move_m=cfg.scene_graph.refine_max_center_move_m,
            link_dist_m=cfg.scene_graph.link_dist_m,
            link_max_frame_gap=getattr(cfg.scene_graph, "link_max_frame_gap", None),
            min_det_score=cfg.scene_graph.min_det_score,
            min_det_bbox_px=cfg.scene_graph.min_det_bbox_px,
            confirm_baseline_m=cfg.scene_graph.confirm_baseline_m,
            repeat_view_discount=cfg.scene_graph.repeat_view_discount,
            presence_filter=_make_presence_filter(cfg),
        )
        self.scene_graph = SceneGraph(
            container_top_h_m=tuple(cfg.scene_graph.container_top_h_m),
            container_min_area_m2=cfg.scene_graph.container_min_area_m2,
            container_support_tol_m=cfg.scene_graph.container_support_tol_m,
            container_min_obs=getattr(cfg.scene_graph, "container_min_obs", 1),
            container_min_score=getattr(cfg.scene_graph, "container_min_score", 0.0),
            container_merge_m=getattr(cfg.scene_graph, "container_merge_m", 0.0),
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
            goal_near_m=getattr(cfg.exploration, "voronoi_goal_near_m", 0.7),
            inflate_radius_m=cfg.agent.agent_radius + cfg.mapping.inflate_margin_m,
        )
        self.controller = WaypointController(forward_m=cfg.agent.forward_m)
        self.viewpoint_planner = ViewpointPlanner(list(cfg.verification.ring_radii_m))
        self._affinity = _make_affinity(cfg)

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
        return self._floor_stack.costmap

    @property
    def floor_layer(self):
        return self._floor_stack.current

    @property
    def _room_labels(self) -> Optional[np.ndarray]:
        return self._floor_stack.current.room_labels

    @_room_labels.setter
    def _room_labels(self, labels: Optional[np.ndarray]) -> None:
        self._floor_stack.current.room_labels = labels

    # ------------------------------------------------------------------ reset

    def reset(self, target_category: str) -> None:
        self.target = target_category
        self.state = State.INIT
        self.step_count = 0
        self._scan_steps_left = (
            int(round(360.0 / self.cfg.agent.turn_deg)) if self.cfg.agent.initial_scan else 0
        )
        self._floor_y: Optional[float] = None
        # (step, floor_id, floor_height) on every committed floor change, plus
        # the first step. Surfaced per episode by eval/runner.py.
        self.floor_log: list = []
        self.floor_y_drift = 0.0
        self.stair_regions: list = []
        self.portal_log: list = []
        self._portal_active = False
        self._portal_start_y = 0.0
        self._portal_step = 0
        self.floors.reset()
        self._floor_stack.reset()
        self._kf_count = 0
        self._current_path: Optional[np.ndarray] = None
        self._current_frontier: Optional[Frontier] = None
        # Location-keyed blacklist: frontier ids are reassigned on every
        # extraction, so blocking must be spatial to persist. [(xy, until)]
        self._blocked_frontier_pts: list = []
        # Centroid of the frontier the agent most recently gave up on: excluded
        # from the "all frontiers blocked" fallback so the agent doesn't
        # immediately re-pursue the dead-end it just abandoned.
        # (xy, floor) -- floor-scoped for the same reason as the blacklist.
        self._last_giveup_pt: Optional[tuple] = None
        # A frontier is only genuinely "reached" if we end up within this of its
        # goal. The controller reports None (arrived) whenever the planned path
        # terminates within its arrival tolerance of the agent -- which also
        # happens for a degenerate stub path when the frontier is unreachable
        # (goal snapped to a nearby node). This threshold separates the two.
        self._frontier_reach_m = 0.5
        self._candidate_id: Optional[int] = None
        self._goal_xy: Optional[np.ndarray] = None
        self._last_action: Optional[str] = None
        self._last_select_step = -100
        self._goto_deadline = 10**9
        self._progress_ref_step = 0
        self._progress_ref_xy = np.zeros(2)
        self._target_obj_xy: Optional[np.ndarray] = None
        self._went_to_best_cam = False
        self._center_turns = 0  # centering turns spent on the current candidate
        # APPROACH state: path-goal cache is separate from _goal_xy/_current_path
        # used by GOTO_FRONTIER/GOTO_VERIFY_VIEW because APPROACH switches
        # between an "advance toward the object" goal and a "retreat to the
        # last visible pose" goal within the same episode phase.
        self._path_goal: Optional[np.ndarray] = None
        self._approach_last_good_xy: Optional[np.ndarray] = None
        self._approach_steps_left = 0
        self._goal_floor_y_cache: Optional[float] = None
        self.stats = {"plan_ok": 0, "plan_fail": 0, "select_none": 0, "select_ok": 0}
        # Phase 2 instrumentation (docs/DYNAMIC_SCENES.md): when the map STOPPED
        # believing in something, and what it believed at the moment it
        # committed to a goal. Belief latency and stale-goal rate are computed
        # from these two logs plus the relocation step the env records.
        self.presence_events: List[dict] = []
        # What has already been searched, and how well (C3). A visit multiplies
        # a surface's belief by (1 - d) rather than zeroing it, so a place
        # glanced at from four metres stays plausible and one inspected closely
        # mostly stops being -- the distinction an ignore list cannot make.
        self._search_log = InspectionLog()
        self._search_container: Optional[int] = None
        self._search_started_step = 0
        self._approach_at_viewpoint = False
        self._scan_turns_left = 0
        self._scan_expected = 0
        # Set once the agent has been to the last known place and found nothing.
        self._target_confirmed_moved = False
        self.search_log_events: List[dict] = []
        self.goal_commit_log: List[dict] = []
        self._disbelieved: set = set()
        self.state_log = []
        self.frontier_select_log: list = []
        self.giveup_log: list = []
        # Calibration data for approach_stop_bbox_px (P1c): every bbox_px
        # observed during APPROACH, plus why the episode's approach ended.
        self.approach_bbox_log: list = []
        self.approach_stop_reason: Optional[str] = None
        # Approach-navigation diagnostics (for the terminal approach that ends
        # the episode): why the agent stopped short of a correctly-mapped
        # target. Split path_consumed into planner-no-path vs controller-arrived
        # and record the approach geometry. See scripts/analyze_approach.py.
        self.approach_diag: dict = {}
        self._last_follow_none_reason: Optional[str] = None
        self.kf_selector.reset()
        self.controller.reset()
        self.detector.set_vocabulary(
            target_vocabulary(self.target, self.cfg.detector.vocabulary)
        )

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
        if self._floor_y is None:
            self._floor_y = float(frame.camera_position[1] - self.cfg.agent.camera_height)

        # Track the storey every step. With floor.estimate_only (the default)
        # this only LOGS -- the costmap keeps using the latched _floor_y, so
        # the estimator can be validated against the per-scene navmesh ground
        # truth (scripts/scene_floors.py) before behaviour depends on it.
        prev_floor = self.floors.current
        floor_id = self.floors.update(
            float(frame.camera_position[1]), self.step_count,
            xy=frame.camera_position[list(PLANE)],
        )
        if floor_id != prev_floor:
            self._end_portal_pursuit("arrived")
        if floor_id != prev_floor or not self.floor_log:
            self.floor_log.append(
                (self.step_count, int(floor_id),
                 round(float(frame.camera_position[1]) - self.cfg.agent.camera_height, 3))
            )
        fcfg = getattr(self.cfg, "floor", None)
        floor_y = self._floor_y
        live_floor = getattr(fcfg, "enabled", False) and not getattr(fcfg, "estimate_only", True)
        if live_floor:
            floor_y = self.floors.height_of(floor_id)
            if getattr(fcfg, "per_floor_costmap", False):
                # Point the stack at the agent's storey BEFORE mapping, so this
                # frame lands in that floor's own grid. While on stairs the
                # estimator freezes floor_id, so the treads keep going to the
                # floor being left rather than opening a phantom layer.
                self._floor_stack.set_current(
                    floor_id, step=self.step_count,
                    agent_xy=frame.camera_position[list(PLANE)],
                )
        # How far the estimated floor height ever strays from the value the old
        # code latched on frame 1. On a single storey this should be ~0; larger
        # means the obstacle band is silently shifting and perturbing
        # trajectories that have nothing to do with multi-floor.
        self.floor_y_drift = max(
            self.floor_y_drift, abs(self.floors.height_of(floor_id) - self._floor_y)
        )

        with self.profiler.timeit("costmap"):
            self.costmap.update(
                frame,
                floor_y=floor_y,
                obstacle_low=self.cfg.mapping.obstacle_low_m,
                obstacle_high=self.cfg.mapping.obstacle_high_m,
                max_range=self.cfg.mapping.max_range_m,
                stride=self.cfg.mapping.depth_stride,
            )
        self.controller.observe_progress(frame.T_wc, self._last_action, self.costmap, self.step_count)
        if self.controller.stuck:
            self.controller.stuck = False
            self._current_path = None  # force replan

        if self.kf_selector.is_keyframe(frame.T_wc):
            self._on_keyframe(frame)
            if getattr(self.cfg.exploration, "search_posterior", False):
                self._glance_at_surfaces(frame)

        # Candidate target check happens in every state except terminal ones
        if self.state in (State.INIT, State.EXPLORE, State.GOTO_FRONTIER):
            self._check_candidates()

        if self.state == State.INIT:
            if self._scan_steps_left > 0:
                self._scan_steps_left -= 1
                return TURN_ACTION
            self.state = State.EXPLORE

        if self.state == State.EXPLORE:
            if self._portal_pursuit_ok(frame) and self._goal_xy is not None:
                self.state = State.GOTO_FRONTIER  # resume the climb
            else:
                self._select_new_frontier(frame)
            if self.state == State.EXPLORE:  # nothing selectable
                return TURN_ACTION  # keep looking around; map will grow

        if self.state == State.GOTO_FRONTIER:
            # Give-up net: no displacement for a while means an obstacle the
            # map cannot see (below the obstacle band, glass, sim collision).
            # Abandon this frontier instead of pushing against it forever.
            agent_xy = frame.camera_position[list(PLANE)]
            if self.step_count - self._progress_ref_step >= 15:
                # A portal pursuit is judged on vertical progress; a switchback
                # staircase barely moves in (x, z) while climbing fine.
                if self._portal_active and self._portal_pursuit_ok(frame):
                    self._progress_ref_step = self.step_count
                    self._progress_ref_xy = agent_xy.copy()
                elif np.linalg.norm(agent_xy - self._progress_ref_xy) < 0.2:
                    self.giveup_log.append((
                        self.step_count,
                        [round(float(x), 2) for x in self._current_frontier.centroid_xy]
                        if self._current_frontier is not None else None,
                        [round(float(x), 2) for x in agent_xy],
                    ))
                    self._block_frontier(self._current_frontier, 100)
                    self.stats["frontier_give_up"] = self.stats.get("frontier_give_up", 0) + 1
                    if self._current_frontier is not None:
                        self._last_giveup_pt = (self._current_frontier.centroid_xy.copy(),
                                        self._current_frontier.floor)
                    self._current_frontier = None
                    self._current_path = None
                    self.state = State.EXPLORE
                    self._progress_ref_step = self.step_count
                    self._progress_ref_xy = agent_xy.copy()
                    return self._act_inner_post_transition(frame)
                self._progress_ref_step = self.step_count
                self._progress_ref_xy = agent_xy.copy()
            action = self._follow_path(frame)
            if action is not None:
                return action
            self._current_frontier = None
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
            return self._do_verification(frame)

        if self.state == State.APPROACH:
            return self._do_approach(frame)

        return STOP_ACTION

    def _do_approach(self, frame: FrameData) -> str:
        """Walk toward the verified object while it stays visible.

        Every step re-runs the detector on the current pose:
        - visible and within the target metric range (median mask depth <=
          approach_stop_depth_m) -> close and in clear view, stop. Depth is the
          primary signal (bbox area is object-size-dependent); bbox is a
          fallback for when the mask carries no valid depth.
        - visible but still too far -> record this pose as good, advance one
          more step toward the object.
        - not visible -> if a previous pose was confirmed visible, retreat
          there (a step just carried us behind an occluder the 2D costmap
          LOS check cannot see, e.g. a desk edge) and stop; otherwise the
          object was never visible from this approach at all, so keep
          advancing toward it (there is nothing better to retreat to) until
          the deadline.
        """
        agent_xy = frame.camera_position[list(PLANE)]
        # Track how close the agent gets to its approach goal this episode.
        if self.approach_diag and self._goal_xy is not None:
            dg = float(np.linalg.norm(agent_xy - self._goal_xy))
            cur = self.approach_diag.get("min_dist_to_goal_m")
            if cur is None or dg < cur:
                self.approach_diag["min_dist_to_goal_m"] = dg
        det = self._best_target_detection(frame)

        if det is not None:
            self._approach_last_good_xy = agent_xy.copy()
            x1, y1, x2, y2 = det.bbox_xyxy
            bbox_px = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            depth = self._detection_depth(det, frame)
            # Log (step, bbox_px, depth) for terminal calibration.
            self.approach_bbox_log.append(
                (self.step_count, round(float(bbox_px), 1),
                 round(float(depth), 3) if depth is not None else None)
            )
            stop_reason: Optional[str] = None
            # When the goal IS a viewpoint, arriving at it is the stop
            # condition. A depth stop would fire en route -- the viewpoint sits
            # at 0.8-1.2 m and the depth threshold is 1.0 m -- and leave the
            # agent short of the pose success is actually measured at.
            if self._approach_at_viewpoint:
                pass
            elif getattr(self.cfg.agent, "approach_depth_stop", True):
                if depth is not None:
                    if depth <= self.cfg.agent.approach_stop_depth_m:
                        stop_reason = "depth"
                elif bbox_px >= self.cfg.agent.approach_stop_bbox_px:  # fallback: no valid depth
                    stop_reason = "bbox"
            if stop_reason is not None:
                # Terminal-view verification: the agent is close and the target
                # fills the view -- this live close-up is the decisive frame.
                # Ask the VLM before committing STOP; a rejection means the
                # detector locked onto a false positive, so blacklist it and
                # resume exploring rather than stopping on empty/wrong space.
                if self._terminal_verify:
                    self.stats["terminal_verify"] = self.stats.get("terminal_verify", 0) + 1
                    # Full live frame with the target boxed (scene context).
                    if not self.verifier.verify_bbox(frame.rgb, det.bbox_xyxy, self.target):
                        self.stats["terminal_reject"] = self.stats.get("terminal_reject", 0) + 1
                        if self._candidate_id is not None:
                            self.object_layer.blacklist(self._candidate_id)
                        self._candidate_id = None
                        self._target_obj_xy = None
                        self.state = State.EXPLORE
                        return TURN_ACTION
                self.state = State.DONE
                self.approach_stop_reason = stop_reason
                return STOP_ACTION
        elif (
            not self._use_navmesh  # navmesh knows the path; a momentary FOV loss
            # while turning along it must NOT trigger a retreat, or the agent
            # oscillates (approach -> lose detection -> retreat -> re-detect ...)
            # until the deadline. Costmap mode keeps the LOS-occlusion retreat.
            and self._approach_last_good_xy is not None
            and np.linalg.norm(agent_xy - self._approach_last_good_xy) > 0.1
        ):
            action = self._follow_to(frame, self._approach_last_good_xy)
            if action is not None:
                return action
            abandon = self._absence_at_arrival(frame, "retreat")
            if abandon is not None:
                return abandon
            self.state = State.DONE  # retreat path consumed/unreachable: stop here
            self.approach_stop_reason = "retreat"
            return STOP_ACTION

        if self.step_count > self._goto_deadline or self._approach_steps_left <= 0:
            abandon = self._absence_at_arrival(frame, "deadline")
            if abandon is not None:
                return abandon
            self.state = State.DONE
            self.approach_stop_reason = "deadline"
            return STOP_ACTION
        self._approach_steps_left -= 1
        self._last_follow_none_reason = None
        action = self._follow_to(frame, self._goal_xy)
        if action is None:  # path consumed or unreachable: as close as it gets
            turn = self._scan_at_viewpoint(det, frame)
            if turn is not None:
                return turn
            abandon = self._absence_at_arrival(frame, "path_consumed")
            if abandon is not None:
                return abandon
            self.state = State.DONE
            self.approach_stop_reason = "path_consumed"
            if self.approach_diag is not None:
                self.approach_diag["path_consumed_cause"] = self._last_follow_none_reason
                if self._last_follow_none_reason == "planner_no_path":
                    self.approach_diag["plan_fail"] = self.approach_diag.get("plan_fail", 0) + 1
            return STOP_ACTION
        return action

    def _act_inner_post_transition(self, frame: FrameData) -> str:
        """Re-enter EXPLORE logic once after a state transition (no recursion
        beyond one level: EXPLORE either picks a path or turns in place)."""
        self._select_new_frontier(frame)
        if self.state == State.GOTO_FRONTIER:
            action = self._follow_path(frame)
            if action is not None:
                return action
        return TURN_ACTION

    # -------------------------------------------------------------- keyframes

    def _on_keyframe(self, frame: FrameData) -> None:
        self._kf_count += 1
        with self.profiler.timeit("detector"):
            dets = self.detector.detect(frame.rgb)
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

        if self._stairs_on:
            fc = self.cfg.floor
            if self._kf_count % max(1, int(fc.stair_detect_every_kf)) == 1:
                with self.profiler.timeit("stairs"):
                    self._detect_stairs()

        if self._kf_count % self.cfg.scene_graph.room_seg_every_kf == 1:
            with self.profiler.timeit("room_seg"):
                self._room_labels = self.floor_layer.segmenter.segment(self.costmap)
        if self._room_labels is not None:
            if self._room_labels.shape != self.costmap.grid.shape:
                self._room_labels = self.floor_layer.segmenter.segment(self.costmap)
            with self.profiler.timeit("scene_graph"):
                self.scene_graph.rebuild(
                    self._room_labels, self.costmap, self.object_layer,
                    floors=self.floors
                    if getattr(getattr(self.cfg, "floor", None), "enabled", False) else None,
                )

    def _detect_stairs(self) -> None:
        """Find steppable regions on the current storey and mark them
        traversable, so the staircase stops reading as a wall."""
        fc = self.cfg.floor
        regions = detect_stairs(
            self.costmap,
            climb_limit_m=fc.climb_limit_m,
            min_dh_m=fc.stair_min_dh_m,
            cell_m=fc.stair_cell_m,
            min_cells=fc.stair_min_cells,
            min_rise_m=fc.stair_min_rise_m,
            semantic_centers=stair_tracks(
                self.object_layer,
                min_obs=fc.stair_min_obs,
                min_evidence=fc.stair_min_evidence,
            ),
            require_semantic=fc.stair_require_semantic,
        )
        if not regions:
            return
        n = apply_stair_mask(self.costmap, regions, max_area_frac=fc.stair_max_area_frac)
        self.stair_regions = regions
        self.stats["stair_cells"] = self.stats.get("stair_cells", 0) + n
        self.stats["stair_regions"] = len(regions)
        self.stats["stair_regions_semantic"] = sum(1 for r in regions if r.semantic)
        self.stats["stair_max_rise_m"] = round(
            max(self.stats.get("stair_max_rise_m", 0.0), max(r.rise_m for r in regions)), 2
        )

    # ------------------------------------------------------------ exploration

    def _block_frontier(self, f: Optional[Frontier], duration: int) -> None:
        # Blocks carry their storey. Stored unconditionally: on a single floor
        # every entry is floor 0, so the floor test below is a tautology and
        # behaviour is unchanged -- no second code path to keep in sync.
        if f is not None:
            self._blocked_frontier_pts.append(
                (f.centroid_xy.copy(), self.step_count + duration, f.floor)
            )

    def _blocked_ids(self, frontiers) -> set:
        self._blocked_frontier_pts = [
            b for b in self._blocked_frontier_pts if b[1] > self.step_count
        ]
        active = [(xy, floor) for xy, _, floor in self._blocked_frontier_pts]
        return {
            f.id
            for f in frontiers
            if any(
                floor == f.floor and np.linalg.norm(f.centroid_xy - xy) < 0.6
                for xy, floor in active
            )
        }

    @staticmethod
    def _heading_xy(frame: FrameData) -> np.ndarray:
        """Agent forward direction on the ground plane (unit). Camera looks along
        +z (OpenCV), so world-forward = R @ [0,0,1], projected to (x, z)."""
        fwd = frame.T_wc[:3, :3] @ np.array([0.0, 0.0, 1.0])
        v = fwd[list(PLANE)]
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else np.array([1.0, 0.0])

    def _select_new_frontier(self, frame: FrameData) -> None:
        # Extraction + top-N path planning is expensive; while waiting the
        # agent turns in place, which grows the map anyway.
        if self.step_count - self._last_select_step < 5:
            return
        self._last_select_step = self.step_count
        # A surface is only searched once the agent has actually got there.
        # Marking it on the next selection round instead -- which fires every 5
        # steps -- spent belief on places the agent had merely set off towards,
        # so it visited seven surfaces in 500 steps and inspected none of them.
        if self._search_container is not None:
            agent_xy = frame.camera_position[list(PLANE)]
            arrived = (
                self._goal_xy is not None
                and float(np.linalg.norm(agent_xy - self._goal_xy))
                <= float(getattr(self.cfg.exploration, "search_arrival_m", 1.2))
            )
            spent = self.step_count - self._search_started_step
            if not arrived and spent < int(getattr(self.cfg.exploration, "search_max_steps", 60)):
                return  # still on the way: stay committed to this surface
            self._mark_surface_searched(arrived=arrived)
        with self.profiler.timeit("frontier_extract"):
            frontiers = self.frontier_extractor.extract(
                self.costmap, frame.camera_position[list(PLANE)],
                floor=self._floor_stack.current_id,
            )
        if not frontiers:
            # Nothing left on this floor is the strongest possible "no near
            # frontier", so the portal gate still gets its chance.
            if self._try_floor_switch(frame, None):
                return
            return
        # Async scoring request (never blocks); use whatever scores exist now
        self.scorer.request(frontiers, self.scene_graph, self.target, self.keyframes)
        blocked = self._blocked_ids(frontiers)
        agent_xy = frame.camera_position[list(PLANE)]
        heading_xy = self._heading_xy(frame)
        failed: set = set()
        with self.profiler.timeit("frontier_select"):
            best = select_frontier(
                frontiers,
                self.scorer.latest(),
                self.planner,
                self.costmap,
                agent_xy,
                unscored_prior=self.cfg.exploration.unscored_prior,
                min_path_cost_m=self.cfg.exploration.min_path_cost_m,
                top_n=self.cfg.exploration.top_n_frontiers,
                blocked=blocked,
                failed_out=failed,
                info_gain_weight=self.cfg.exploration.info_gain_weight,
                info_gain_radius_m=self.cfg.exploration.info_gain_radius_m,
                los_visibility_penalty=self.cfg.exploration.los_visibility_penalty,
                heading_xy=heading_xy,
                continuity_weight=self.cfg.exploration.continuity_weight,
                goal_prefer_free=self._frontier_goal_free,
                cost_prefer_free=self._frontier_cost_free,
            )
        by_id = {f.id: f for f in frontiers}
        for fid in failed:  # block only the candidates that actually failed
            self._block_frontier(by_id.get(fid), 50)

        surface = self._select_surface(agent_xy, best)
        if surface is not None:
            self._goal_xy = surface.goal_xy
            self._search_container = int(surface.ref_id)
            self._search_started_step = self.step_count
            # Actually GO there. Only GOTO_FRONTIER follows _goal_xy; setting the
            # goal while the state stayed EXPLORE meant the agent never moved,
            # re-selected the same surface five steps later, and scored it
            # "never reached" each time. Every C3 result before this was
            # measuring selections that were never acted on: eight inspections
            # of one desk, an unchanged 2.3 m path cost, arrived=False
            # throughout. The give-up net handles a null frontier already.
            self._current_frontier = None
            self._progress_ref_step = self.step_count
            self._progress_ref_xy = agent_xy.copy()
            self.state = State.GOTO_FRONTIER
            self._current_path = None
            self._goal_frontier = None
            self.stats["search_surface"] = self.stats.get("search_surface", 0) + 1
            self.search_log_events.append(
                {
                    "step": int(self.step_count),
                    "container_id": int(surface.ref_id),
                    "label": surface.label,
                    "prior": round(float(surface.prior), 4),
                    "path_cost": round(float(surface.path_cost or 0.0), 2),
                    "utility": round(float(surface.utility or 0.0), 5),
                }
            )
            return
        if best is None or best.path_cost is None:
            # Every frontier was blocked (a give-up/plan-fail cascade in
            # cluttered scenes leaves nothing selectable) -- rather than turn in
            # place burning the step budget until blocks expire, fall back to the
            # best path-reachable frontier ignoring blocks, excluding only the
            # one just given up on. A frontier blocked from an earlier pose is
            # often reachable now; if it re-stalls, give-up catches it again.
            relaxed_blocked = set()
            if self._last_giveup_pt is not None:
                relaxed_blocked = {
                    f.id for f in frontiers
                    if f.floor == self._last_giveup_pt[1]
                    and np.linalg.norm(f.centroid_xy - self._last_giveup_pt[0]) < 0.6
                }
            if len(relaxed_blocked) < len(frontiers):
                best = select_frontier(
                    frontiers, self.scorer.latest(), self.planner, self.costmap,
                    agent_xy, unscored_prior=self.cfg.exploration.unscored_prior,
                    min_path_cost_m=self.cfg.exploration.min_path_cost_m,
                    top_n=self.cfg.exploration.top_n_frontiers, blocked=relaxed_blocked,
                    info_gain_weight=self.cfg.exploration.info_gain_weight,
                    info_gain_radius_m=self.cfg.exploration.info_gain_radius_m,
                    los_visibility_penalty=self.cfg.exploration.los_visibility_penalty,
                    heading_xy=heading_xy,
                    continuity_weight=self.cfg.exploration.continuity_weight,
                    goal_prefer_free=self._frontier_goal_free,
                    cost_prefer_free=self._frontier_cost_free,
                )
        # Nothing near left on this floor? Consider leaving it. Checked BEFORE
        # committing to a far frontier, because "the best thing here is 12 m
        # away" is exactly ASCENT's condition for reasoning about storeys.
        if self._try_floor_switch(frame, None if best is None else best.path_cost):
            return

        if best is None or best.path_cost is None:
            self.stats["select_none"] += 1
            return
        self.stats["select_ok"] += 1
        # Per-selection trace (step, agent xy, chosen frontier xy, path cost,
        # #frontiers) for exploration-efficiency debugging. See scripts.
        self.frontier_select_log.append((
            self.step_count,
            [round(float(x), 2) for x in agent_xy],
            [round(float(x), 2) for x in best.centroid_xy],
            round(float(best.path_cost), 2) if best.path_cost is not None else None,
            len(frontiers),
        ))
        self._current_frontier = best
        self._plan_to(frame, frontier_goal_xy(best, self.costmap, self._frontier_goal_free))
        if self._current_path is not None:
            self.state = State.GOTO_FRONTIER
            # A fresh pursuit starts its own 15-step progress window; without
            # this the give-up timer carried over from whatever frontier was
            # pursued (or given up on) before, and could fire on the very
            # first step of the new pursuit based on stale position data.
            self._progress_ref_step = self.step_count
            self._progress_ref_xy = agent_xy.copy()
        else:
            self._block_frontier(best, 50)

    def _glance_at_surfaces(self, frame: FrameData) -> None:
        """A surface in plain view has been searched, without driving to it.

        Measured: a full inspection costs the agent about fifty steps -- approach,
        arrival, commitment budget -- so a 500-step episode manages seven to nine
        of them. Simulating the search order over this scene's 112 surfaces says
        the target is typically reached after 37-45 inspections but only 33-40 m
        of travel, so the budget that binds is inspections, not distance. Most of
        those surfaces are simply in view along the way; looking counts.

        A glance is weaker evidence than standing at the surface, so it retires
        belief at a lower rate -- the search log already expresses that as
        (1 - d), and a passing look gets a smaller d.
        """
        pf = self.object_layer.presence_filter
        containers = getattr(self.scene_graph, "containers", None)
        if pf is None or not containers:
            return
        d = float(getattr(self.cfg.exploration, "search_glance_detect_prob", 0.35))
        rng = float(getattr(self.cfg.exploration, "search_glance_range_m", 4.0))
        K, T_cw = frame.intrinsics.K(), frame.T_cw
        h, w = frame.depth.shape
        for cid, node in containers.items():
            p_cam = T_cw[:3, :3] @ node.center + T_cw[:3, 3]
            z = float(p_cam[2])
            if not (0.3 <= z <= rng):
                continue
            uv = K @ p_cam
            u, v = float(uv[0] / z), float(uv[1] / z)
            if not (0 <= u < w and 0 <= v < h):
                continue
            measured = float(frame.depth[int(v), int(u)])
            if measured > 1e-3 and measured < z - 0.5:
                continue  # something solid between us and the surface
            self._search_log.searched(cid, d)

    def _select_surface(self, agent_xy, best_frontier):
        """The best mapped surface, if it beats the best frontier on b*d/c.

        Both sides are the same index -- `select_frontier` already returns
        score/path_cost -- so the comparison is like for like, with
        search_frontier_weight naming the one judgement call: what unmapped
        space is worth against a plausible surface.
        """
        cfg = self.cfg.exploration
        if not getattr(cfg, "search_posterior", False):
            return None
        if not getattr(self.scene_graph, "containers", None):
            return None
        cands = build_container_candidates(
            self.scene_graph,
            self.target,
            self._search_log,
            detect_prob=float(getattr(cfg, "search_detect_prob", 0.8)),
            last_known_xy=self._last_known_target_xy(),
            proximity_len_m=float(getattr(cfg, "search_proximity_len_m", 4.0)),
            plane=PLANE,
            affinity_source=self._affinity,
        )
        if not cands:
            return None
        # Drive to a pose you can STAND in, not to the middle of the furniture.
        # A container's centre is inside the desk; the follower ends wherever the
        # navmesh allows, arrival is never registered, and the surface is scored
        # as "never reached" -- a quarter credit -- so it stays top of the list
        # and gets chosen again. Measured before this fix, one episode's entire
        # search was: desk, desk, desk, desk, desk, desk, desk, desk, with its
        # prior decaying 3.20, 2.56, 2.05, 1.64 ... and an unchanged 2.5 m path
        # cost every time. Eight inspections, one surface.
        reachable = []
        for c in cands:
            view = self.viewpoint_planner.approach_viewpoint(c.goal_xy, self.costmap)
            c.goal_xy = np.asarray(
                view if view is not None else self._nearest_free_xy(c.goal_xy), dtype=float
            )
            reachable.append(c)
        cands = reachable
        # Prefer surfaces in the room the agent is already in. Simulated over
        # this scene: room-grouped order reaches the target in a median 37
        # inspections and 33 m against 45 and 40 m for a plain global argmax,
        # because crossing the house repeatedly is what the global index does
        # once the nearby surfaces are retired.
        room_bonus = float(getattr(cfg, "search_same_room_bonus", 1.0))
        if room_bonus > 1.0 and getattr(self.scene_graph, "rooms", None):
            here = self.scene_graph.room_of_point(agent_xy)
            if here is not None:
                for c in cands:
                    node = self.scene_graph.containers.get(c.ref_id)
                    if node is not None and node.room_id == here.id:
                        c.prior *= room_bonus
        surface = select_candidate(
            cands, self.planner, self.costmap, agent_xy,
            top_n=int(getattr(cfg, "top_n_frontiers", 5)),
            min_path_cost_m=float(getattr(cfg, "min_path_cost_m", 0.5)),
        )
        if surface is None or surface.utility is None:
            return None
        beta = float(getattr(cfg, "search_frontier_weight", 1.0))
        if best_frontier is not None and best_frontier.path_cost:
            frontier_util = beta * (best_frontier.score or 0.0) / best_frontier.path_cost
            if frontier_util >= surface.utility:
                return None
        return surface

    def _last_known_target_xy(self):
        """Where the target was last believed to be, or None once it is known to
        have left there.

        Proximity encodes "objects are moved by someone doing a task, so short
        displacements dominate". That holds until the agent goes and confirms
        the object is NOT at its old place -- after which the premise the term
        rests on has been refuted, and the surfaces it favours are exactly the
        ones just ruled out. Measured on nine cross-anchor episodes: keeping the
        term ranks the true destination 32nd of 112 surfaces on median, and puts
        it in the top 8 (what one episode inspects) in 0 of 9. Dropping it once
        absence is confirmed gives median 20 and 3 of 9.
        """
        if self._target_confirmed_moved:
            return None
        best = None
        for track in self.object_layer.tracks(include_blacklisted=True):
            if str(track.label).lower().replace("_", " ") != str(self.target).lower().replace("_", " "):
                continue
            if best is None or track.presence.n_expected > best.presence.n_expected:
                best = track
        if best is None:
            return None
        return self.object_layer.center_of(best)[list(PLANE)]

    def _mark_surface_searched(self, arrived: bool = True) -> None:
        """Arriving at a surface without the target is a look that did not find
        it -- worth (1 - d), not worth zero and not worth nothing.

        Giving up on the way there is a much weaker look, and is scored as such:
        a place the agent never reached has barely been ruled out, and spending
        full belief on it would retire the very surfaces it failed to inspect.
        """
        if self._search_container is None:
            return
        d = float(getattr(self.cfg.exploration, "search_detect_prob", 0.8))
        if not arrived:
            d *= float(getattr(self.cfg.exploration, "search_unreached_credit", 0.25))
        remaining = self._search_log.searched(self._search_container, d)
        self.search_log_events.append(
            {
                "step": int(self.step_count),
                "container_id": int(self._search_container),
                "searched": True,
                "arrived": bool(arrived),
                "belief_factor": round(remaining, 4),
            }
        )
        self._search_container = None

    def _portal_pursuit_ok(self, frame: FrameData) -> bool:
        """Should the agent keep driving to its portal instead of re-exploring?

        Held while it is still climbing (or descending) and the deadline has not
        passed. Vertical progress is the test, not horizontal: on a switchback
        staircase the (x, z) displacement over 15 steps can be small while the
        agent is making perfectly good progress, which is also why the ordinary
        give-up net must not judge a portal pursuit.
        """
        if not self._portal_active:
            return False
        if self.step_count > self._goto_deadline:
            self._end_portal_pursuit("deadline")
            return False
        climbed = abs(float(frame.camera_position[1]) - self._portal_start_y)
        if climbed >= self.cfg.floor.portal_progress_m or self.floors.on_stairs:
            return True
        # Not moving vertically and not on stairs: the portal was unreachable or
        # the agent is stuck at the foot of it -- fall back to exploring.
        if self.step_count - self._portal_step > self.cfg.floor.portal_grace_steps:
            self._end_portal_pursuit("no_vertical_progress")
            return False
        return True

    def _end_portal_pursuit(self, reason: str) -> None:
        self._portal_active = False
        self.stats[f"portal_end_{reason}"] = self.stats.get(f"portal_end_{reason}", 0) + 1

    def _try_floor_switch(self, frame: FrameData, best_path_cost) -> bool:
        """Head for another storey when this one has nothing near left.

        Returns True if a portal was selected and the agent is now driving to
        it. The portal is only a heading -- the navmesh walks the actual stairs,
        and the floor estimator commits the new storey once the agent settles
        there, at which point FloorStack swaps in that floor's map.
        """
        if self._switch_policy is None:
            return False
        evidence, n_objects = floor_target_evidence(
            self.scene_graph, self._floor_stack.current_id, self.target
        )
        if not self._switch_policy.may_switch(
            self.step_count, best_path_cost, evidence=evidence, n_objects=n_objects,
            steps_on_floor=self.step_count - self._floor_stack.current.first_step,
        ):
            return False

        floor_y = self.floors.height_of(self._floor_stack.current_id)
        portals = find_portals(
            self.costmap, floor_y,
            min_delta_m=self.cfg.floor.new_level_m,
            max_delta_m=self.cfg.floor.portal_max_delta_m,
            min_cells=self.cfg.floor.portal_min_cells,
        )
        self.stats["portals_seen"] = max(self.stats.get("portals_seen", 0), len(portals))
        if not portals:
            return False

        agent_xy = frame.camera_position[list(PLANE)]
        # Prefer a storey we have NOT searched, then the nearest. Nearest-only
        # let the agent bounce back onto a floor it had already given up on --
        # 4-5 transitions in some episodes, paying the travel cost each time.
        levels = self.floors.levels

        def unvisited(p):
            return not any(
                abs(h - p.target_y) <= self.cfg.floor.level_tol_m for h in levels.values()
            )

        portals.sort(key=lambda p: (not unvisited(p),
                                    float(np.linalg.norm(p.centroid_xy - agent_xy))))
        target = portals[0]
        if self._reachable_fn is not None and not self._reachable_fn(
            target.centroid_xy, target.target_y
        ):
            return False

        self._goal_xy = target.centroid_xy.copy()
        self._goal_floor_y_cache = target.target_y
        self._current_frontier = None
        self._current_path = None
        self.state = State.GOTO_FRONTIER
        # Hold this goal against same-floor frontier re-selection. While the
        # agent is on the stairs its floor id is frozen, so the costmap it sees
        # is still the floor BELOW -- and left alone, exploration picks a
        # frontier down there and walks the agent back down. Measured: three
        # episodes climbed ~1.6 m and turned around exactly this way.
        self._portal_active = True
        self._portal_start_y = float(frame.camera_position[1])
        self._portal_step = self.step_count
        self._goto_deadline = self.step_count + self.cfg.floor.portal_deadline_steps
        self._progress_ref_step = self.step_count
        self._progress_ref_xy = agent_xy.copy()
        self._switch_policy.note_switch(self.step_count)
        self.stats["floor_switch_attempts"] = self.stats.get("floor_switch_attempts", 0) + 1
        self.portal_log.append((
            self.step_count,
            [round(float(x), 2) for x in target.centroid_xy],
            round(float(target.delta_y), 2),
            target.n_cells,
        ))
        return True

    # ------------------------------------------------------------- candidates

    def _goal_floor_y(self, center: np.ndarray) -> Optional[float]:
        """Height to snap a 3D goal at, or None to keep the legacy behaviour of
        substituting the agent's own height.

        Snap at the goal's FLOOR, not at its ellipsoid centre: an object's
        centre sits 0.3-1.0 m above the ground, and near a mezzanine edge that
        offset is enough to snap onto the wrong storey.

        No clearance offset is added. The navmesh sits at floor height, so the
        floor height IS the right query -- and on a single floor it equals the
        agent's own standing height, which makes this a genuine no-op there.
        An earlier +0.1 m "clearance" was enough on its own to change the snap
        result and perturb single-floor trajectories.
        """
        fcfg = getattr(self.cfg, "floor", None)
        if not getattr(self.cfg.agent, "navmesh_3d_goals", False):
            return None
        if not getattr(fcfg, "enabled", False) or not self.floors.levels:
            return None
        return self.floors.height_of(self.floors.floor_of_height(float(center[1])))

    def _check_candidates(self) -> None:
        candidates = self.object_layer.candidates(
            self.target,
            min_obs=self.cfg.verification.min_obs,
            min_score=self.cfg.verification.min_score,
            min_bbox_px=self.cfg.verification.min_bbox_px,
            min_evidence=self.cfg.verification.min_evidence,
            min_presence=getattr(
                getattr(self.cfg.scene_graph, "presence", None), "min_presence", 0.0
            ),
            max_identity_rejections=int(getattr(
                getattr(self.cfg.scene_graph, "presence", None),
                "max_identity_rejections", 0,
            )),
        )
        if not candidates:
            return
        track = candidates[0]
        self._candidate_id = track.id
        self._center_turns = 0  # fresh centering budget for this candidate
        obj_center = self.object_layer.center_of(track)
        obj_xy = obj_center[list(PLANE)]

        # Navmesh alignment (old stack): navigate straight to the object
        # position and let Habitat's navmesh drive there, then STOP on arrival
        # -- like publishing /goal_object. No viewpoint pre-positioning.
        if self._use_navmesh:
            # Don't commit to a target on a disconnected navmesh island (a
            # visible-but-unreachable object, e.g. in a sealed bathroom): the
            # agent can never get there, so blacklist it and keep exploring for
            # a reachable goal instead of stopping and failing the episode.
            if self._reachable_fn is not None and not self._reachable_fn(
                obj_xy, self._goal_floor_y(obj_center)
            ):
                self.object_layer.blacklist(track.id)
                self._candidate_id = None
                self.stats["unreachable_skip"] = self.stats.get("unreachable_skip", 0) + 1
                return
            # VLM verify the candidate before committing (no VERIFYING state in
            # navmesh mode). Reject -> blacklist and keep exploring; this is the
            # only FP gate in the navmesh path.
            if self.verifier is not None and not getattr(
                self.cfg.verification, "absence_only", False
            ):
                with self.profiler.timeit("verification"):
                    ok = self.verifier.verify(track, self.target)
                if not ok:
                    # The VLM looked at this exact object and said it is not the
                    # target. That is an identity verdict and belongs in the
                    # identity channel; blacklisting would make it permanent and
                    # unrecoverable, which is the mistake this file has had to
                    # unlearn three times. Count it at full weight so one clear
                    # "no" retires the candidate.
                    track.identity_rejections += int(getattr(
                        getattr(self.cfg.scene_graph, "presence", None),
                        "max_identity_rejections", 0,
                    )) or 1
                    self._candidate_id = None
                    self.stats["verify_reject"] = self.stats.get("verify_reject", 0) + 1
                    return
            self._log_goal_commit(track)
            self._start_approach(obj_xy, floor_y=self._goal_floor_y(obj_center))
            return

        # Always pre-position at a viewpoint from which the object is visible
        # before approaching -- HM3D success requires stopping at such a pose,
        # not merely near the object's 3D center. When verification is off the
        # VERIFYING state simply skips the VLM call (see _do_verification).
        view_xy = self.viewpoint_planner.approach_viewpoint(obj_xy, self.costmap)
        if view_xy is None:
            return  # not yet observable from mapped space; keep exploring
        self._goal_xy = view_xy
        self.state = State.GOTO_VERIFY_VIEW
        self._current_path = None
        self._goto_deadline = self.step_count + 80

    def _scan_at_viewpoint(self, det, frame: FrameData) -> Optional[str]:
        """Sweep in place on arrival, until the target is seen or the budget ends.

        A viewpoint is a pose the object is visible FROM, but the navmesh
        follower arrives on whatever heading the path happened to end with, and
        one frame from one heading is a thin basis for deciding an object is
        gone. Measured both ways: concluding absence from the arrival frame
        abandoned a bowl that was exactly where the map said, while stopping
        without looking declared success on empty space. A full sweep costs a
        dozen steps and makes the detector's silence mean something.
        """
        if not self._approach_at_viewpoint or det is not None:
            return None
        if self._scan_turns_left <= 0:
            return None
        # Turn TOWARD the object, not blindly. A full blind sweep ends on the
        # heading it started from -- the navmesh follower's arrival heading --
        # so the absence decision was being taken on whatever happened to be in
        # front. Captured at the moment of one such decision on a CORRECT map:
        # a wall and a painting, with the bowl's table off frame to the right.
        # The VLM answered "bare" and was right about the pixels it was shown.
        from ..planning.controller import TURN_LEFT, TURN_RIGHT, _wrap, agent_heading

        if self._target_obj_xy is not None:
            agent_xy = frame.camera_position[list(PLANE)]
            to_obj = self._target_obj_xy - agent_xy
            if float(np.linalg.norm(to_obj)) > 1e-3:
                err = _wrap(
                    float(np.arctan2(to_obj[1], to_obj[0])) - agent_heading(frame.T_wc)
                )
                if abs(err) > np.radians(15.0):
                    self._scan_turns_left -= 1
                    self.stats["approach_face_turns"] = (
                        self.stats.get("approach_face_turns", 0) + 1
                    )
                    return TURN_RIGHT if err > 0 else TURN_LEFT
                # Facing it and still nothing: that is the informative frame, so
                # decide here rather than sweeping on past it.
                self._scan_turns_left = 0
                return None
        # Record whether the object was EXPECTED at any heading of the sweep.
        # The decision below used to test only the frame the sweep ended on --
        # after a full circle, the arrival heading again, which need not face
        # the object -- so the agent arrived at a ghost, swept right past it and
        # concluded nothing. Measured: all nine cross-anchor episodes stopped at
        # 29-58 steps with 440+ unspent, and the re-search never ran once.
        pf = self.object_layer.presence_filter
        track = (
            self.object_layer.get(self._candidate_id)
            if self._candidate_id is not None else None
        )
        if pf is not None and track is not None:
            if pf.expectation(track, frame, center_only=True) is not None:
                self._scan_expected += 1

        # Note what is deliberately NOT done here: applying a negative reading
        # per sweep frame. Twelve looks at the same object from the same pose
        # are not twelve independent observations -- same range, same lighting,
        # same viewing angle on the same geometry -- so multiplying their
        # likelihoods turns one correlated detector failure into overwhelming
        # evidence of absence. Measured: doing it dropped SR from 0.429 to
        # 0.286 by abandoning a bowl that was exactly where the map said. The
        # sweep's job is to give the detector a chance, not to vote.
        self._scan_turns_left -= 1
        self.stats["approach_scan_turns"] = self.stats.get("approach_scan_turns", 0) + 1
        return TURN_ACTION

    def _absence_at_arrival(self, frame: FrameData, reason: str) -> Optional[str]:
        """The approach is ending and the target was never seen. Say so.

        Walking to where the map said an object was, finding nothing, and
        stopping there is how a stale map converts a success into a confident
        failure -- and, worse, it teaches the map nothing, so the next episode
        makes the same trip. Arriving without a sighting IS an observation:
        this applies it as negative evidence, and abandons the candidate when
        the belief no longer supports stopping on it.

        Returns an action when the candidate is abandoned (the caller must not
        STOP), or None to let the normal termination proceed. A track that has
        been seen at some point during this approach is left alone -- the
        target was there, this is a geometry or timing problem, not absence.
        """
        vc = self.cfg.verification
        if not getattr(vc, "absence_on_arrival", True):
            return None
        pf = self.object_layer.presence_filter
        track = (
            self.object_layer.get(self._candidate_id)
            if self._candidate_id is not None else None
        )
        if pf is None or track is None or self._approach_last_good_xy is not None:
            return None
        # The VLM is a second sensor with its own (r, q); when it is available,
        # ask it about the target's own footprint rather than trusting the
        # detector's silence alone. A failed call returns None and is treated as
        # no information, never as absence.
        recall, q = float(getattr(vc, "detector_absence_recall", 0.5)), None
        asked_vlm = False
        if self.verifier is not None and getattr(vc, "absence_use_vlm", True):
            proj = track.ellipsoid.project(frame.intrinsics.K(), frame.T_cw)
            if proj is not None:
                with self.profiler.timeit("absence_vlm"):
                    still = self.verifier.verify_still_there(frame.rgb, proj.bbox(), self.target)
                if still is not None:
                    asked_vlm = True
                    recall = float(getattr(vc, "vlm_recall", 0.9))
                    q = float(getattr(vc, "vlm_q", 0.2))
                    if still:
                        # It IS there and the detector merely missed it. Let the
                        # stop stand -- this is the case that made a correct map
                        # abandon a bowl 0.8 m in front of it.
                        pf.apply_reading(track, True, recall, q)
                        return None
        # The expectation gate is the DETECTOR's precondition: its silence only
        # means something where a detection was likely (frustum, range, apparent
        # size, occlusion -- C1 already answers this). A VLM that answered about
        # the region has already looked, so its answer stands on its own.
        if not asked_vlm and getattr(vc, "absence_requires_expectation", True):
            # A sweep that expected to see it at ANY heading has looked at it.
            if self._scan_expected == 0 and pf.expectation(track, frame, center_only=True) is None:
                self.stats["absence_not_expected"] = self.stats.get("absence_not_expected", 0) + 1
                return None

        p = pf.apply_reading(track, False, recall, q)
        self.stats["absence_checks"] = self.stats.get("absence_checks", 0) + 1
        if asked_vlm:
            self.stats["absence_vlm"] = self.stats.get("absence_vlm", 0) + 1

        if p >= float(getattr(vc, "abandon_below_p", 0.35)):
            return None  # still believed: stop as before, and keep the evidence
        self.stats["absence_abandon"] = self.stats.get("absence_abandon", 0) + 1
        # Walking to a mapped pose and not finding the TARGET says something the
        # belief cannot carry, because a false positive is an object that is
        # genuinely there and will be re-detected on the very next keyframe.
        track.identity_rejections += 1
        self.presence_events.append(
            {
                "step": int(self.step_count),
                "track_id": int(track.id),
                "label": str(track.label),
                "center": [float(v) for v in self.object_layer.center_of(track)],
                "p": round(float(p), 4),
                "n_missed": int(track.presence.n_missed),
                "cause": f"absent_on_arrival:{reason}",
            }
        )
        # Deliberately NOT blacklisted. Blacklisting is permanent, and C1's
        # whole premise is that no state is absorbing: the belief carries the
        # information, and `min_presence` keeps a disbelieved track out of the
        # candidate list until evidence brings it back. Measured cost of getting
        # this wrong: on a CORRECT map the agent abandoned the bowl, wandered,
        # and finished the episode standing 0.088 m from the goal -- inside the
        # success radius -- unable to stop, because the only track that could
        # have been the answer had been struck off for good.
        self._candidate_id = None
        self._target_obj_xy = None
        self._target_confirmed_moved = True
        self.state = State.EXPLORE
        return TURN_ACTION

    def _log_goal_commit(self, track) -> None:
        """What the map believed at the moment it committed. A commit to a
        track the agent has already looked for and failed to find is a stale
        goal -- the failure DualMap's ignore list exists to paper over."""
        self.goal_commit_log.append(
            {
                "step": int(self.step_count),
                "track_id": int(track.id),
                "label": str(track.label),
                "p": round(float(track.presence.p), 4),
                "n_missed": int(track.presence.n_missed),
                "center": [float(v) for v in self.object_layer.center_of(track)],
            }
        )

    def _do_verification(self, frame: FrameData) -> str:
        track = self.object_layer.get(self._candidate_id) if self._candidate_id is not None else None
        if track is None:
            self.state = State.EXPLORE
            return TURN_ACTION
        from ..planning.controller import TURN_LEFT, TURN_RIGHT, _wrap, agent_heading

        # Center-then-verify: if a VLM verifier is active and the target is
        # actually visible in the live view, bring its detection to the middle
        # of the camera before the VLM call, then verify that well-framed frame.
        if (
            self.verifier is not None
            and getattr(self.cfg.verification, "center_before_verify", True)
        ):
            det = self._best_target_detection(frame)
            if det is not None:
                bbox_cx = 0.5 * (float(det.bbox_xyxy[0]) + float(det.bbox_xyxy[2]))
                offset = float(np.arctan2(bbox_cx - frame.intrinsics.cx, frame.intrinsics.fx))
                if (
                    abs(offset) > np.radians(self.cfg.verification.center_tol_deg)
                    and self._center_turns < self.cfg.verification.center_max_turns
                ):
                    self._center_turns += 1
                    self.stats["center_turn"] = self.stats.get("center_turn", 0) + 1
                    # target right of centre (offset>0) -> turn right to centre it
                    return TURN_RIGHT if offset > 0 else TURN_LEFT
                # Centred (or out of centring budget): verify the live framed view.
                with self.profiler.timeit("verification"):
                    accepted = (
                        True if self._terminal_verify
                        else self.verifier.verify_bbox(frame.rgb, det.bbox_xyxy, self.target)
                    )
                if accepted:
                    obj_xy = self.object_layer.center_of(track)[list(PLANE)]
                    self._start_approach(obj_xy, agent_xy=frame.camera_position[list(PLANE)])
                    return self._do_approach(frame)
                self.object_layer.blacklist(track.id)
                self._candidate_id = None
                self.state = State.EXPLORE
                return TURN_ACTION
            # target not visible in the live view -> fall through to the
            # 3D-facing / best-cam recovery below.

        # Face the object first so the live view actually shows it.

        obj_xy = self.object_layer.center_of(track)[list(PLANE)]
        agent_xy = frame.camera_position[list(PLANE)]
        to_obj = obj_xy - agent_xy
        if np.linalg.norm(to_obj) > 0.05:
            err = _wrap(float(np.arctan2(to_obj[1], to_obj[0])) - agent_heading(frame.T_wc))
            if abs(err) > np.radians(20.0):
                return TURN_RIGHT if err > 0 else TURN_LEFT
        # HM3D success viewpoints require the object to actually be VISIBLE
        # from the stop pose; 2D line-of-sight misses desk-height occluders
        # (we stopped 0.12 m outside the viewpoint set). If the detector
        # cannot see the target from here, return to the pose the best
        # detection was made from — proven reachable AND proven visible
        # (ring alternatives proved unreachable and thrashed the deadline).
        if (
            not self._target_visible(frame)
            and not self._went_to_best_cam
            and track.best_cam_xy is not None
            and np.linalg.norm(track.best_cam_xy - agent_xy) > 0.35
        ):
            self._went_to_best_cam = True
            self._goal_xy = track.best_cam_xy.copy()
            self.state = State.GOTO_VERIFY_VIEW
            self._current_path = None
            self._goto_deadline = self.step_count + 60
            return self._follow_path(frame) or TURN_ACTION
        with self.profiler.timeit("verification"):
            # Pre-approach verification. Skipped (accept) when the verifier is
            # off (old-fidelity mode) OR in terminal-view mode, where the
            # decisive VLM check is deferred to the STOP moment in _do_approach.
            accepted = (
                True if (self.verifier is None or self._terminal_verify)
                else self.verifier.verify(track, self.target, live_view=frame.rgb)
            )
        if accepted:
            obj_xy = self.object_layer.center_of(track)[list(PLANE)]
            self._start_approach(obj_xy, agent_xy=frame.camera_position[list(PLANE)])
            return self._do_approach(frame)
        self.object_layer.blacklist(track.id)
        self._candidate_id = None
        self.state = State.EXPLORE
        return TURN_ACTION

    # ---------------------------------------------------------------- helpers

    def _start_approach(
        self,
        obj_xy: np.ndarray,
        agent_xy: Optional[np.ndarray] = None,
        floor_y: Optional[float] = None,
    ) -> None:
        # Height to snap the navmesh goal at for the rest of this approach.
        # None keeps the legacy "use the agent's own height" behaviour.
        self._goal_floor_y_cache = floor_y
        self._approach_at_viewpoint = False
        if self._use_navmesh and getattr(self.cfg.agent, "approach_to_viewpoint", False):
            # HM3D scores success as the distance from the final pose to the
            # nearest GOAL VIEW POINT, and those are sampled on rings at fixed
            # radii around the object. Stopping when the target's depth reaches
            # approach_stop_depth_m puts the agent at 1.0 m -- radially between
            # the 0.8 m and 1.2 m rings, about 0.2 m from the nearest viewpoint
            # either way. Measured: four of seven batch episodes ended at 0.18,
            # 0.19, 0.21 and 0.28 m against a 0.18 m radius, having found the
            # object. ViewpointPlanner samples the SAME radii, so driving to one
            # of its poses puts the agent ON a ring, where the only error left
            # is angular -- at worst half the sampling step, about 0.10 m.
            view_xy = self.viewpoint_planner.approach_viewpoint(obj_xy, self.costmap)
            if view_xy is not None:
                self.stats["approach_viewpoint"] = self.stats.get("approach_viewpoint", 0) + 1
            else:
                # Not observable from mapped FREE space yet. The fallback used to
                # be the object's own centre, and that is unwinnable by
                # construction: a tabletop object's centre is an occupied cell
                # inside the furniture, so the follower stalls against it and the
                # agent ends up INSIDE the innermost 0.8 m viewpoint ring, where
                # HM3D cannot score a success however well the object was found.
                # Measured over 42 episodes: 23 approaches took this branch, and
                # the 10 episodes that ended on such a goal scored SR 0.100
                # against 0.516 for the rest, stalling at 0.54-1.51 m.
                #
                # So relax the viewpoint search instead of abandoning it --
                # unmapped is not unstandable, and a ray that clips the object's
                # own table is not a blocked view. Any pose ON a ring beats any
                # pose off it.
                view_xy = self.viewpoint_planner.approach_viewpoint(
                    obj_xy, self.costmap, require_line_of_sight=False, allow_unknown=True
                )
                self.stats["approach_viewpoint_none"] = (
                    self.stats.get("approach_viewpoint_none", 0) + 1
                )
                if view_xy is not None:
                    self.stats["approach_viewpoint_relaxed"] = (
                        self.stats.get("approach_viewpoint_relaxed", 0) + 1
                    )
            if view_xy is not None:
                self._goal_xy = np.asarray(view_xy, dtype=float).copy()
                self._approach_at_viewpoint = True
            else:
                # Every ring pose is out of bounds. The nearest free cell is
                # still a cell the agent can stand in, which the object's own
                # centre is not.
                self._goal_xy = self._nearest_free_xy(obj_xy)
                self.stats["approach_goal_nearest_free"] = (
                    self.stats.get("approach_goal_nearest_free", 0) + 1
                )
        elif self._use_navmesh:
            # Navigate to the object itself; the navmesh snaps to the nearest
            # standable point (effectively a viewpoint), like old /goal_object.
            self._goal_xy = obj_xy.copy()
        elif getattr(self.cfg.agent, "approach_navigable_goal", False) and agent_xy is not None:
            self._goal_xy = self._approach_goal_xy(obj_xy, agent_xy)
        else:
            self._goal_xy = self._nearest_free_xy(obj_xy)
        self._target_obj_xy = obj_xy.copy()
        self._scan_turns_left = int(getattr(self.cfg.agent, "approach_scan_turns", 12))
        self._scan_expected = 0
        self.state = State.APPROACH
        self._current_path = None
        self._path_goal = None
        if self._use_navmesh:
            # Navmesh drives the FULL distance to the object (no viewpoint
            # pre-positioning), so the short-leg cap (approach_max_steps ~= 3 m)
            # cuts the approach off while the target is still in view. Let it
            # navigate to the object, bounded only by a generous deadline.
            self._goto_deadline = self.step_count + self.cfg.agent.navmesh_approach_steps
            self._approach_steps_left = 10 ** 9
        else:
            self._goto_deadline = self.step_count + 100
            self._approach_steps_left = self.cfg.agent.approach_max_steps
        # Snapshot the terminal approach for navigation diagnostics: the goal
        # cell (nearest-free to the mapped object), how far it sits from the
        # object, the agent's start distance to it, and the costmap status of
        # the goal cell -- an UNKNOWN/OCCUPIED goal cell explains an
        # unreachable-path stop. min_dist_to_goal is filled in per step.
        self.approach_diag = {
            "goal_xy": [float(x) for x in self._goal_xy],
            "obj_xy": [float(x) for x in obj_xy],
            "goal_to_obj_m": float(np.linalg.norm(self._goal_xy - obj_xy)),
            "goal_cell": self._cell_status(self._goal_xy),
            "start_step": self.step_count,
            "min_dist_to_goal_m": None,
            "plan_fail": 0,
            "path_consumed_cause": None,
        }
        self._approach_last_good_xy = None

    def _best_target_detection(self, frame: FrameData) -> Optional[Detection]:
        """Runs the detector on the current frame and returns its highest-
        confidence detection matching the target category, or None."""
        target = self.target.lower().replace("_", " ").strip()
        with self.profiler.timeit("detector"):
            dets = self.detector.detect(frame.rgb)
        matches = [
            d for d in dets
            if d.label.lower().replace("_", " ").strip() == target and d.score > 0.25
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
            result: PlanResult = self.planner.plan(self.costmap, agent_xy, goal_xy, goal_tolerance_m)
        self._current_path = result.path if result.success else None
        self.stats["plan_ok" if result.success else "plan_fail"] += 1

    def _follow_path(self, frame: FrameData) -> Optional[str]:
        goal = (
            frontier_goal_xy(self._current_frontier, self.costmap, self._frontier_goal_free)
            if self.state == State.GOTO_FRONTIER and self._current_frontier is not None
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
            cross_floor_goal = self._portal_active or self.state != State.GOTO_FRONTIER
            action = self._nav_fn(goal, self._goal_floor_y_cache if cross_floor_goal else None)
            if (
                action is None
                and self.state == State.GOTO_FRONTIER
                and self._current_frontier is not None
                and np.linalg.norm(frame.camera_position[list(PLANE)] - goal)
                > self._frontier_reach_m
            ):
                self._block_frontier(self._current_frontier, 100)
                self._last_giveup_pt = (self._current_frontier.centroid_xy.copy(),
                                        self._current_frontier.floor)
                self.stats["frontier_stub_block"] = self.stats.get("frontier_stub_block", 0) + 1
            return action
        if self._current_path is None:
            self._plan_to(frame, goal)
            if self._current_path is None:
                if self.state == State.GOTO_FRONTIER:
                    self._block_frontier(self._current_frontier, 50)
                return None
        action = self.controller.act(frame.T_wc, self._current_path)
        if action is None:
            self._current_path = None
            # Fix A: the controller returns None the moment the planned path
            # ends within its arrival tolerance of the agent. When the planner
            # hands back a degenerate stub path (goal snapped near the start
            # because the real frontier is unreachable), that reads as a false
            # "arrival": the FSM drops GOTO_FRONTIER->EXPLORE without blocking
            # the frontier, so the same one is re-selected every cycle and the
            # agent freezes in place (give-up never fires -- it only counts
            # elapsed steps *inside* GOTO_FRONTIER, and the re-entry resets its
            # timer). If we are still far from the frontier goal, this was not a
            # real arrival: block the frontier so a different one is chosen next.
            if (
                self.state == State.GOTO_FRONTIER
                and self._current_frontier is not None
                and np.linalg.norm(frame.camera_position[list(PLANE)] - goal)
                > self._frontier_reach_m
            ):
                self._block_frontier(self._current_frontier, 100)
                # Also mark it as the last give-up point so the "all frontiers
                # blocked" relaxed fallback (which deliberately ignores the
                # blacklist) doesn't immediately re-pursue this same stub.
                self._last_giveup_pt = (self._current_frontier.centroid_xy.copy(),
                                        self._current_frontier.floor)
                self.stats["frontier_stub_block"] = self.stats.get("frontier_stub_block", 0) + 1
        return action

    def _follow_to(self, frame: FrameData, goal_xy: np.ndarray) -> Optional[str]:
        """Follow a path to an explicit goal, replanning when the goal
        changes (APPROACH alternates between an advance goal and a retreat
        goal within the same state, unlike the other terminal states which
        have one fixed goal for their whole visit).

        Used only by APPROACH, so tightens both the planner's and the
        controller's stopping tolerance beyond the loose defaults used for
        frontier/verify-view travel (P1f): geometry analysis against
        HM3D's actual view_points showed several near-miss episodes
        stopped within 5-8 cm (straight-line) of a real view_point, yet
        habitat's geodesic distance_to_goal still read 0.15-0.17 m --
        the default 0.2-0.3 m tolerances left slack for a short geodesic
        detour around a nearby thin obstacle to blow the 0.13 m success
        radius even when we were geometrically almost there.
        """
        if self._use_navmesh:
            # navmesh drives to the object; None = arrived
            return self._nav_fn(goal_xy, self._goal_floor_y_cache)
        need_replan = (
            self._current_path is None
            or self._path_goal is None
            or np.linalg.norm(self._path_goal - goal_xy) > 0.05
        )
        if need_replan:
            self._plan_to(frame, goal_xy, goal_tolerance_m=self.cfg.agent.approach_goal_tolerance_m)
            self._path_goal = goal_xy.copy() if self._current_path is not None else None
            if self._current_path is None:
                # planner could not reach goal_xy (costmap disconnected /
                # goal in unknown/inflated space) -- distinct from the
                # controller reporting arrival below.
                self._last_follow_none_reason = "planner_no_path"
                return None
        action = self.controller.act(
            frame.T_wc, self._current_path, arrival_tol_m=self.cfg.agent.approach_arrival_tol_m
        )
        if action is None:
            self._current_path = None
            self._path_goal = None
            # controller consumed the path (thinks it arrived at goal_xy) --
            # a false "arrival" here means the planned path was a stub / ended
            # short of the true goal.
            self._last_follow_none_reason = "controller_arrived"
        return action

    def _approach_goal_xy(self, obj_xy: np.ndarray, agent_xy: np.ndarray) -> np.ndarray:
        """Navigable approach goal: a standoff point at approach_standoff_m from
        the object along the ray toward the agent (the side the object was
        observed from -> open, reachable space), snapped to the nearest free
        cell. Avoids the enclosed-pocket goals that _nearest_free_xy produces
        by placing the goal in front of the object rather than hard against it.
        """
        standoff = float(self.cfg.agent.approach_standoff_m)
        to_agent = agent_xy - obj_xy
        dist = float(np.linalg.norm(to_agent))
        if dist < 1e-3:
            return self._nearest_free_xy(obj_xy)
        # If the agent is already closer than the standoff, keep the goal at the
        # standoff (do not push it behind the agent past the object).
        cand = obj_xy + (to_agent / dist) * min(standoff, dist)
        return self._nearest_free_xy(cand)

    def _cell_status(self, xy: np.ndarray) -> str:
        """Costmap classification of a world point: free/occupied/unknown/oob."""
        from ..mapping.costmap import FREE, OCCUPIED, UNKNOWN

        rc = self.costmap.world_to_grid(xy)
        h, w = self.costmap.grid.shape
        if not (0 <= rc[0] < h and 0 <= rc[1] < w):
            return "oob"
        v = self.costmap.grid[rc[0], rc[1]]
        return {FREE: "free", OCCUPIED: "occupied", UNKNOWN: "unknown"}.get(int(v), str(int(v)))

    def _nearest_free_xy(self, xy: np.ndarray) -> np.ndarray:
        """Nearest FREE cell to a (possibly occupied) object position — the
        closest pose the agent can actually stand at."""
        from ..mapping.costmap import FREE

        rc = self.costmap.world_to_grid(xy)
        h, w = self.costmap.grid.shape
        best, best_d = xy, np.inf
        rad = int(1.5 / self.costmap.resolution)
        r0, r1 = max(0, rc[0] - rad), min(h, rc[0] + rad + 1)
        c0, c1 = max(0, rc[1] - rad), min(w, rc[1] + rad + 1)
        free = np.argwhere(self.costmap.grid[r0:r1, c0:c1] == FREE)
        if free.shape[0] == 0:
            return xy
        free_world = self.costmap.grid_to_world(free + np.array([r0, c0]))
        d = np.linalg.norm(free_world - xy, axis=1)
        return free_world[int(np.argmin(d))]
