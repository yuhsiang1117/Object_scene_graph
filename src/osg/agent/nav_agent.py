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
from typing import List, Optional

import numpy as np

from ..core.profiler import Profiler
from ..core.types import Detection, FrameData
from ..exploration.async_scorer import AsyncScorer
from ..exploration.selector import frontier_goal_xy
from ..exploration.strategy import ExplorationStrategy, WorldView
from ..graph.scene_graph import SceneGraph
from ..mapping.costmap import PLANE, Costmap2D, cell_status, nearest_free_xy
from ..objects.object_layer import ObjectLayer
from ..perception.detector import Detector
from ..perception.keyframe import KeyframeSelector, KeyframeStore
from ..planning.controller import WaypointController
from ..planning.planner import PlanResult
from ..planning.voronoi_planner import HybridVoronoiPlanner
from ..verification.viewpoint import ViewpointPlanner
from .floor_policy import FloorPolicy

STOP_ACTION = "stop"
# WaypointController.act's default arrival tolerance. Named here because
# _frontier_reach_m is derived from it: the two must agree or an ordinary
# arrival is classified as an unreachable stub.
FRONTIER_ARRIVAL_TOL_M = 0.2
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
    ec = cfg.exploration
    if ec is None or not ec.affinity_llm:
        return None
    from ..graph.containers import CONTAINER_CATEGORIES
    from ..llm.affinity import AffinityProvider
    from ..llm.client import ChatClient

    client = None
    if cfg.llm.api_key:
        client = ChatClient(
            cfg.llm.base_url, cfg.llm.text_model, cfg.llm.api_key,
            cfg.llm.timeout_s, cfg.llm.max_image_px, cfg.llm.send_response_format,
        )
    return AffinityProvider(
        client, sorted(CONTAINER_CATEGORIES),
        cache_path=str(ec.affinity_cache or "") or None,
    )


def _make_presence_filter(cfg):
    """None unless scene_graph.presence.enabled -- the filter must be an opt-in
    A/B, not a silent default (docs/DYNAMIC_SCENES.md, Phase 1)."""
    pc = cfg.scene_graph.presence
    if pc is None or not pc.enabled:
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
        l_clamp_pos=pc.l_clamp_pos,
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
            presence_filter=_make_presence_filter(cfg),
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
        # Where to go next -- frontiers and mapped surfaces under one index.
        # It owns everything an exploration round remembers; see
        # exploration/strategy.py.
        self.exploration = ExplorationStrategy(
            cfg, self.planner, scorer, self.viewpoint_planner,
            _make_affinity(cfg), self.stats, self.profiler,
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
        self.stats.clear()
        self.stats.update({"plan_ok": 0, "plan_fail": 0, "select_none": 0, "select_ok": 0})
        # Phase 2 instrumentation (docs/DYNAMIC_SCENES.md): when the map STOPPED
        # believing in something, and what it believed at the moment it
        # committed to a goal. Belief latency and stale-goal rate are computed
        # from these two logs plus the relocation step the env records.
        self.presence_events: List[dict] = []
        self._approach_at_viewpoint = False
        self._scan_turns_left = 0
        self._scan_expected = 0
        self.goal_commit_log: List[dict] = []
        self._disbelieved: set = set()
        self.state_log = []
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
        self.exploration.reset()
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
        self.controller.observe_progress(frame.T_wc, self._last_action, self.costmap, self.step_count)
        if self.controller.stuck:
            self.controller.stuck = False
            self._current_path = None  # force replan

        if self.kf_selector.is_keyframe(frame.T_wc):
            self._on_keyframe(frame)
            if self.cfg.exploration.search_posterior:
                self.exploration.glance(self._world(frame))

        # Candidate target check happens in every state except terminal ones
        if self.state in (State.INIT, State.EXPLORE, State.GOTO_FRONTIER):
            self._check_candidates()

        if self.state == State.INIT:
            if self._scan_steps_left > 0:
                self._scan_steps_left -= 1
                return TURN_ACTION
            self.state = State.EXPLORE

        if self.state == State.EXPLORE:
            if self.floors.pursuit_ok(frame, self.step_count, self._goto_deadline) and self._goal_xy is not None:
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
            elif self.cfg.agent.approach_depth_stop:
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
        self._explore(frame)
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
        choice = self.exploration.select(world, floor_switch=lambda cost: self._try_floor_switch(frame, cost))
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

    def _check_candidates(self) -> None:
        candidates = self.object_layer.candidates(
            self.target,
            min_obs=self.cfg.verification.min_obs,
            min_score=self.cfg.verification.min_score,
            min_bbox_px=self.cfg.verification.min_bbox_px,
            min_evidence=self.cfg.verification.min_evidence,
            min_presence=self.cfg.scene_graph.presence.min_presence,
            max_identity_rejections=int(self.cfg.scene_graph.presence.max_identity_rejections),
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
                obj_xy, self.floors.goal_floor_y(obj_center)
            ):
                self.object_layer.blacklist(track.id)
                self._candidate_id = None
                self.stats["unreachable_skip"] = self.stats.get("unreachable_skip", 0) + 1
                return
            # VLM verify the candidate before committing (no VERIFYING state in
            # navmesh mode). Reject -> blacklist and keep exploring; this is the
            # only FP gate in the navmesh path.
            if self.verifier is not None and not self.cfg.verification.absence_only:
                with self.profiler.timeit("verification"):
                    ok = self.verifier.verify(track, self.target)
                if not ok:
                    # The VLM looked at a picture of this object and said it is
                    # not the target. That is identity evidence and belongs in
                    # the identity channel; blacklisting would make it permanent,
                    # which is the mistake this file has had to unlearn three
                    # times.
                    #
                    # It counts as ONE piece of evidence, not a verdict, and that
                    # is a measurement rather than caution. The picture is the
                    # stored crop of the best detection, and for these targets it
                    # is 46-101 px on its longest side -- there is no more image
                    # to be had, the objects are simply small in the frame. Given
                    # decisive weight it cost real successes: of the first three
                    # rejections in a pilot run all three were CORRECT
                    # candidates, and two of them had converted in the run
                    # without the gate. One doubt plus one failed approach
                    # retires a track; one doubt alone does not.
                    track.identity_rejections += 1
                    self._candidate_id = None
                    self.stats["verify_reject"] = self.stats.get("verify_reject", 0) + 1
                    return
            self._log_goal_commit(track)
            self._start_approach(obj_xy, floor_y=self.floors.goal_floor_y(obj_center))
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
        if not vc.absence_on_arrival:
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
        recall, q = float(vc.detector_absence_recall), None
        asked_vlm = False
        if self.verifier is not None and vc.absence_use_vlm:
            proj = track.ellipsoid.project(frame.intrinsics.K(), frame.T_cw)
            if proj is not None:
                with self.profiler.timeit("absence_vlm"):
                    still = self.verifier.verify_still_there(frame.rgb, proj.bbox(), self.target)
                if still is not None:
                    asked_vlm = True
                    recall = float(vc.vlm_recall)
                    q = float(vc.vlm_q)
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
        if not asked_vlm and vc.absence_requires_expectation:
            # A sweep that expected to see it at ANY heading has looked at it.
            if self._scan_expected == 0 and pf.expectation(track, frame, center_only=True) is None:
                self.stats["absence_not_expected"] = self.stats.get("absence_not_expected", 0) + 1
                return None

        p = pf.apply_reading(track, False, recall, q)
        self.stats["absence_checks"] = self.stats.get("absence_checks", 0) + 1
        if asked_vlm:
            self.stats["absence_vlm"] = self.stats.get("absence_vlm", 0) + 1

        if p >= float(vc.abandon_below_p):
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
            and self.cfg.verification.center_before_verify
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
        if self._use_navmesh and self.cfg.agent.approach_to_viewpoint:
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
                self._goal_xy = nearest_free_xy(self.costmap, obj_xy)
                self.stats["approach_goal_nearest_free"] = (
                    self.stats.get("approach_goal_nearest_free", 0) + 1
                )
        elif self._use_navmesh:
            # Navigate to the object itself; the navmesh snaps to the nearest
            # standable point (effectively a viewpoint), like old /goal_object.
            self._goal_xy = obj_xy.copy()
        elif self.cfg.agent.approach_navigable_goal and agent_xy is not None:
            self._goal_xy = self._approach_goal_xy(obj_xy, agent_xy)
        else:
            self._goal_xy = nearest_free_xy(self.costmap, obj_xy)
        self._target_obj_xy = obj_xy.copy()
        self._scan_turns_left = int(self.cfg.agent.approach_scan_turns)
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
            "goal_cell": cell_status(self.costmap, self._goal_xy),
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
            return nearest_free_xy(self.costmap, obj_xy)
        # If the agent is already closer than the standoff, keep the goal at the
        # standoff (do not push it behind the agent past the object).
        cand = obj_xy + (to_agent / dist) * min(standoff, dist)
        return nearest_free_xy(self.costmap, cand)

