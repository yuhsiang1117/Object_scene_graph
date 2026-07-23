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
from ..graph.scene_graph import SceneGraph
from ..mapping.costmap import PLANE, Costmap2D
from ..mapping.frontier import Frontier, FrontierExtractor
from ..mapping.room_seg import VoronoiRoomSegmenter
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
    ) -> None:
        self.cfg = cfg
        self.detector = detector
        self.scorer = scorer
        self.verifier = verifier
        self.profiler = profiler or Profiler()
        # Debug hook: if set, called with (frame, dets) every keyframe right
        # after the detections that feed object_layer.update() are computed
        # -- lets diagnostics observe exactly what the scene graph is built
        # from without duplicating the keyframe-timing logic. None by default
        # (zero cost, never called).
        self.on_keyframe_detections = None

        self.costmap = Costmap2D(resolution=cfg.mapping.resolution_m)
        self.frontier_extractor = FrontierExtractor(
            min_cells=cfg.exploration.frontier_min_cells,
            dedup_m=cfg.exploration.frontier_dedup_m,
        )
        self.room_segmenter = VoronoiRoomSegmenter(
            min_room_radius_m=cfg.scene_graph.room_min_radius_m,
            door_width_m=cfg.scene_graph.room_door_width_m,
        )
        self.object_layer = ObjectLayer(
            assoc_score_thresh=cfg.scene_graph.assoc_score_thresh,
            assoc_depth_gate_m=cfg.scene_graph.assoc_depth_gate_m,
            assoc_category_gate=cfg.scene_graph.assoc_category_gate,
            min_obs_for_refine=cfg.scene_graph.min_obs_for_refine,
            refine_every=cfg.scene_graph.refine_every,
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

    # ------------------------------------------------------------------ reset

    def reset(self, target_category: str) -> None:
        self.target = target_category
        self.state = State.INIT
        self.step_count = 0
        self._scan_steps_left = (
            int(round(360.0 / self.cfg.agent.turn_deg)) if self.cfg.agent.initial_scan else 0
        )
        self._floor_y: Optional[float] = None
        self._kf_count = 0
        self._room_labels: Optional[np.ndarray] = None
        self._current_path: Optional[np.ndarray] = None
        self._current_frontier: Optional[Frontier] = None
        # Location-keyed blacklist: frontier ids are reassigned on every
        # extraction, so blocking must be spatial to persist. [(xy, until)]
        self._blocked_frontier_pts: list = []
        # Centroid of the frontier the agent most recently gave up on: excluded
        # from the "all frontiers blocked" fallback so the agent doesn't
        # immediately re-pursue the dead-end it just abandoned.
        self._last_giveup_pt: Optional[np.ndarray] = None
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
        # APPROACH state: path-goal cache is separate from _goal_xy/_current_path
        # used by GOTO_FRONTIER/GOTO_VERIFY_VIEW because APPROACH switches
        # between an "advance toward the object" goal and a "retreat to the
        # last visible pose" goal within the same episode phase.
        self._path_goal: Optional[np.ndarray] = None
        self._approach_last_good_xy: Optional[np.ndarray] = None
        self._approach_steps_left = 0
        self.stats = {"plan_ok": 0, "plan_fail": 0, "select_none": 0, "select_ok": 0}
        self.state_log = []
        self.giveup_log: list = []
        # Calibration data for approach_stop_bbox_px (P1c): every bbox_px
        # observed during APPROACH, plus why the episode's approach ended.
        self.approach_bbox_log: list = []
        self.approach_stop_reason: Optional[str] = None
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

        with self.profiler.timeit("costmap"):
            self.costmap.update(
                frame,
                floor_y=self._floor_y,
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
            self._select_new_frontier(frame)
            if self.state == State.EXPLORE:  # nothing selectable
                return TURN_ACTION  # keep looking around; map will grow

        if self.state == State.GOTO_FRONTIER:
            # Give-up net: no displacement for a while means an obstacle the
            # map cannot see (below the obstacle band, glass, sim collision).
            # Abandon this frontier instead of pushing against it forever.
            agent_xy = frame.camera_position[list(PLANE)]
            if self.step_count - self._progress_ref_step >= 15:
                if np.linalg.norm(agent_xy - self._progress_ref_xy) < 0.2:
                    self.giveup_log.append((
                        self.step_count,
                        [round(float(x), 2) for x in self._current_frontier.centroid_xy]
                        if self._current_frontier is not None else None,
                        [round(float(x), 2) for x in agent_xy],
                    ))
                    self._block_frontier(self._current_frontier, 100)
                    self.stats["frontier_give_up"] = self.stats.get("frontier_give_up", 0) + 1
                    if self._current_frontier is not None:
                        self._last_giveup_pt = self._current_frontier.centroid_xy.copy()
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
        - visible with a large-enough bbox -> close and in clear view, stop.
        - visible but still small -> record this pose as good, advance one
          more step toward the object.
        - not visible -> if a previous pose was confirmed visible, retreat
          there (a step just carried us behind an occluder the 2D costmap
          LOS check cannot see, e.g. a desk edge) and stop; otherwise the
          object was never visible from this approach at all, so keep
          advancing toward it (there is nothing better to retreat to) until
          the deadline.
        """
        agent_xy = frame.camera_position[list(PLANE)]
        det = self._best_target_detection(frame)

        if det is not None:
            self._approach_last_good_xy = agent_xy.copy()
            x1, y1, x2, y2 = det.bbox_xyxy
            bbox_px = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            # Calibration data (P1c): approach_stop_bbox_px is one global
            # threshold, but "close enough" bbox area plausibly differs a
            # lot by object scale (sofa vs plant). Log every observed value
            # so per-category thresholds can be derived from real episodes
            # instead of guessed -- the project's repeated lesson this
            # session is that guessed thresholds get corrected anyway.
            self.approach_bbox_log.append((self.step_count, round(float(bbox_px), 1)))
            if bbox_px >= self.cfg.agent.approach_stop_bbox_px:
                self.state = State.DONE
                self.approach_stop_reason = "bbox"
                return STOP_ACTION
        elif (
            self._approach_last_good_xy is not None
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
        action = self._follow_to(frame, self._goal_xy)
        if action is None:  # path consumed or unreachable: as close as it gets
            self.state = State.DONE
            self.approach_stop_reason = "path_consumed"
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

        if self._kf_count % self.cfg.scene_graph.room_seg_every_kf == 1:
            with self.profiler.timeit("room_seg"):
                self._room_labels = self.room_segmenter.segment(self.costmap)
        if self._room_labels is not None:
            if self._room_labels.shape != self.costmap.grid.shape:
                self._room_labels = self.room_segmenter.segment(self.costmap)
            with self.profiler.timeit("scene_graph"):
                self.scene_graph.rebuild(self._room_labels, self.costmap, self.object_layer)

    # ------------------------------------------------------------ exploration

    def _block_frontier(self, f: Optional[Frontier], duration: int) -> None:
        if f is not None:
            self._blocked_frontier_pts.append((f.centroid_xy.copy(), self.step_count + duration))

    def _blocked_ids(self, frontiers) -> set:
        active = [xy for xy, until in self._blocked_frontier_pts if until > self.step_count]
        self._blocked_frontier_pts = [
            (xy, until) for xy, until in self._blocked_frontier_pts if until > self.step_count
        ]
        return {
            f.id
            for f in frontiers
            if any(np.linalg.norm(f.centroid_xy - xy) < 0.6 for xy in active)
        }

    def _select_new_frontier(self, frame: FrameData) -> None:
        # Extraction + top-N path planning is expensive; while waiting the
        # agent turns in place, which grows the map anyway.
        if self.step_count - self._last_select_step < 5:
            return
        self._last_select_step = self.step_count
        with self.profiler.timeit("frontier_extract"):
            frontiers = self.frontier_extractor.extract(
                self.costmap, frame.camera_position[list(PLANE)]
            )
        if not frontiers:
            return
        # Async scoring request (never blocks); use whatever scores exist now
        self.scorer.request(frontiers, self.scene_graph, self.target, self.keyframes)
        blocked = self._blocked_ids(frontiers)
        agent_xy = frame.camera_position[list(PLANE)]
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
                    if np.linalg.norm(f.centroid_xy - self._last_giveup_pt) < 0.6
                }
            if len(relaxed_blocked) < len(frontiers):
                best = select_frontier(
                    frontiers, self.scorer.latest(), self.planner, self.costmap,
                    agent_xy, unscored_prior=self.cfg.exploration.unscored_prior,
                    min_path_cost_m=self.cfg.exploration.min_path_cost_m,
                    top_n=self.cfg.exploration.top_n_frontiers, blocked=relaxed_blocked,
                )
        if best is None or best.path_cost is None:
            self.stats["select_none"] += 1
            return
        self.stats["select_ok"] += 1
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

    # ------------------------------------------------------------- candidates

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
        obj_xy = self.object_layer.center_of(track)[list(PLANE)]

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
        # Face the object first so the live view actually shows it.
        from ..planning.controller import TURN_LEFT, TURN_RIGHT, _wrap, agent_heading

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
            # verification off (old-fidelity mode): accept without a VLM call and
            # go straight to the visibility-driven bbox approach.
            accepted = (
                True if self.verifier is None
                else self.verifier.verify(track, self.target, live_view=frame.rgb)
            )
        if accepted:
            obj_xy = self.object_layer.center_of(track)[list(PLANE)]
            self._start_approach(obj_xy)
            return self._do_approach(frame)
        self.object_layer.blacklist(track.id)
        self._candidate_id = None
        self.state = State.EXPLORE
        return TURN_ACTION

    # ---------------------------------------------------------------- helpers

    def _start_approach(self, obj_xy: np.ndarray) -> None:
        self._goal_xy = self._nearest_free_xy(obj_xy)
        self._target_obj_xy = obj_xy.copy()
        self.state = State.APPROACH
        self._current_path = None
        self._path_goal = None
        self._goto_deadline = self.step_count + 100
        self._approach_steps_left = self.cfg.agent.approach_max_steps
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
                self._last_giveup_pt = self._current_frontier.centroid_xy.copy()
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
        need_replan = (
            self._current_path is None
            or self._path_goal is None
            or np.linalg.norm(self._path_goal - goal_xy) > 0.05
        )
        if need_replan:
            self._plan_to(frame, goal_xy, goal_tolerance_m=self.cfg.agent.approach_goal_tolerance_m)
            self._path_goal = goal_xy.copy() if self._current_path is not None else None
            if self._current_path is None:
                return None
        action = self.controller.act(
            frame.T_wc, self._current_path, arrival_tol_m=self.cfg.agent.approach_arrival_tol_m
        )
        if action is None:
            self._current_path = None
            self._path_goal = None
        return action

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
