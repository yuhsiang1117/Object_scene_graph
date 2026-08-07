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
            min_det_score=cfg.scene_graph.min_det_score,
            min_det_bbox_px=cfg.scene_graph.min_det_bbox_px,
            confirm_baseline_m=cfg.scene_graph.confirm_baseline_m,
            repeat_view_discount=cfg.scene_graph.repeat_view_discount,
        )
        self.scene_graph = SceneGraph()
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
            [self.target.replace("_", " ")] + list(self.cfg.detector.vocabulary)
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
            if getattr(self.cfg.agent, "approach_depth_stop", True):
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
            self.state = State.DONE  # retreat path consumed/unreachable: stop here
            self.approach_stop_reason = "retreat"
            return STOP_ACTION

        if self.step_count > self._goto_deadline or self._approach_steps_left <= 0:
            self.state = State.DONE
            self.approach_stop_reason = "deadline"
            return STOP_ACTION
        self._approach_steps_left -= 1
        self._last_follow_none_reason = None
        action = self._follow_to(frame, self._goal_xy)
        if action is None:  # path consumed or unreachable: as close as it gets
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
            )
        by_id = {f.id: f for f in frontiers}
        for fid in failed:  # block only the candidates that actually failed
            self._block_frontier(by_id.get(fid), 50)
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
        self._plan_to(frame, frontier_goal_xy(best, self.costmap))
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
            self.step_count, best_path_cost, evidence=evidence, n_objects=n_objects
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
            if self.verifier is not None:
                with self.profiler.timeit("verification"):
                    ok = self.verifier.verify(track, self.target)
                if not ok:
                    self.object_layer.blacklist(track.id)
                    self._candidate_id = None
                    self.stats["verify_reject"] = self.stats.get("verify_reject", 0) + 1
                    return
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
        if self._use_navmesh:
            # Navigate to the object itself; the navmesh snaps to the nearest
            # standable point (effectively a viewpoint), like old /goal_object.
            self._goal_xy = obj_xy.copy()
        elif getattr(self.cfg.agent, "approach_navigable_goal", False) and agent_xy is not None:
            self._goal_xy = self._approach_goal_xy(obj_xy, agent_xy)
        else:
            self._goal_xy = self._nearest_free_xy(obj_xy)
        self._target_obj_xy = obj_xy.copy()
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
            frontier_goal_xy(self._current_frontier, self.costmap)
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
