"""NavAgent: the full pipeline behind a single act(frame) -> action call.

State machine:
    INIT (360 scan) -> EXPLORE <-> GOTO_FRONTIER
                           |  candidate found
                           v
                  GOTO_VERIFY_VIEW -> VERIFYING --accept--> GOTO_TARGET -> STOP
                           ^                |
                           |                +--reject--> blacklist, EXPLORE
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from ..core.profiler import Profiler
from ..core.types import FrameData
from ..exploration.async_scorer import AsyncScorer
from ..exploration.selector import select_frontier
from ..graph.scene_graph import SceneGraph
from ..mapping.costmap import PLANE, Costmap2D
from ..mapping.frontier import Frontier, FrontierExtractor
from ..mapping.room_seg import VoronoiRoomSegmenter
from ..objects.object_layer import ObjectLayer
from ..perception.detector import Detector
from ..perception.keyframe import KeyframeSelector, KeyframeStore
from ..planning.controller import WaypointController
from ..planning.planner import AStarPlanner, PlanResult
from ..verification.verifier import TargetVerifier
from ..verification.viewpoint import ViewpointPlanner

STOP_ACTION = "stop"
TURN_ACTION = "turn_left"


class State(Enum):
    INIT = "init"
    EXPLORE = "explore"
    GOTO_FRONTIER = "goto_frontier"
    GOTO_VERIFY_VIEW = "goto_verify_view"
    VERIFYING = "verifying"
    GOTO_TARGET = "goto_target"
    DONE = "done"


class NavAgent:
    def __init__(
        self,
        cfg,
        detector: Detector,
        scorer: AsyncScorer,
        verifier: Optional[TargetVerifier],
        target_category: str,
        keyframe_dir: Optional[str] = None,
        profiler: Optional[Profiler] = None,
    ) -> None:
        self.cfg = cfg
        self.detector = detector
        self.scorer = scorer
        self.verifier = verifier
        self.profiler = profiler or Profiler()

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
            min_obs_for_refine=cfg.scene_graph.min_obs_for_refine,
            refine_every=cfg.scene_graph.refine_every,
            link_dist_m=cfg.scene_graph.link_dist_m,
        )
        self.scene_graph = SceneGraph()
        self.keyframes = KeyframeStore(save_dir=keyframe_dir)
        self.kf_selector = KeyframeSelector(
            cfg.scene_graph.keyframe_trans_m, cfg.scene_graph.keyframe_rot_deg
        )
        self.planner = AStarPlanner(
            inflate_radius_m=cfg.agent.agent_radius + cfg.mapping.inflate_margin_m
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
        self._blocked_frontiers: dict = {}  # frontier_id -> unblock_step
        self._candidate_id: Optional[int] = None
        self._goal_xy: Optional[np.ndarray] = None
        self._last_action: Optional[str] = None
        self.kf_selector.reset()
        self.controller.reset()
        self.detector.set_vocabulary(
            [self.target.replace("_", " ")] + list(self.cfg.detector.vocabulary)
        )

    # ------------------------------------------------------------------- act

    def act(self, frame: FrameData) -> str:
        self.step_count += 1
        with self.profiler.timeit("control_loop"):
            action = self._act_inner(frame)
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
        self.controller.observe_progress(frame.T_wc, self._last_action, self.costmap)
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
            action = self._follow_path(frame)
            if action is not None:
                return action
            self._current_frontier = None
            self.state = State.EXPLORE
            return self._act_inner_post_transition(frame)

        if self.state == State.GOTO_VERIFY_VIEW:
            action = self._follow_path(frame)
            if action is not None:
                return action
            self.state = State.VERIFYING

        if self.state == State.VERIFYING:
            return self._do_verification(frame)

        if self.state == State.GOTO_TARGET:
            if self._arrived_at_goal(frame):
                self.state = State.DONE
                return STOP_ACTION
            action = self._follow_path(frame)
            if action is not None:
                return action
            self.state = State.DONE
            return STOP_ACTION

        return STOP_ACTION

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

    def _select_new_frontier(self, frame: FrameData) -> None:
        with self.profiler.timeit("frontier_extract"):
            frontiers = self.frontier_extractor.extract(self.costmap)
        if not frontiers:
            return
        # Async scoring request (never blocks); use whatever scores exist now
        self.scorer.request(frontiers, self.scene_graph, self.target, self.keyframes)
        blocked = {fid for fid, until in self._blocked_frontiers.items() if until > self.step_count}
        agent_xy = frame.camera_position[list(PLANE)]
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
            )
        if best is None or best.path_cost is None:
            for f in frontiers:  # nothing reachable: block them briefly
                self._blocked_frontiers[f.id] = self.step_count + 50
            return
        self._current_frontier = best
        self._plan_to(frame, best.centroid_xy)
        if self._current_path is not None:
            self.state = State.GOTO_FRONTIER
        else:
            self._blocked_frontiers[best.id] = self.step_count + 50

    # ------------------------------------------------------------- candidates

    def _check_candidates(self) -> None:
        candidates = self.object_layer.candidates(
            self.target, min_obs=self.cfg.verification.min_obs
        )
        if not candidates:
            return
        track = candidates[0]
        self._candidate_id = track.id
        obj_xy = self.object_layer.center_of(track)[list(PLANE)]

        if self.verifier is None:
            # Verification disabled (paper baseline): head straight for it
            self._goal_xy = obj_xy
            self.state = State.GOTO_TARGET
            self._current_path = None
            return

        view_xy = self.viewpoint_planner.approach_viewpoint(obj_xy, self.costmap)
        if view_xy is None:
            return  # not yet observable from mapped space; keep exploring
        self._goal_xy = view_xy
        self.state = State.GOTO_VERIFY_VIEW
        self._current_path = None

    def _do_verification(self, frame: FrameData) -> str:
        track = self.object_layer.get(self._candidate_id) if self._candidate_id is not None else None
        if track is None:
            self.state = State.EXPLORE
            return TURN_ACTION
        with self.profiler.timeit("verification"):
            accepted = self.verifier.verify(track, self.target, live_view=frame.rgb)
        if accepted:
            self._goal_xy = self.object_layer.center_of(track)[list(PLANE)]
            self.state = State.GOTO_TARGET
            self._current_path = None
            if self._arrived_at_goal(frame):
                self.state = State.DONE
                return STOP_ACTION
            action = self._follow_path(frame)
            return action if action is not None else STOP_ACTION
        self.object_layer.blacklist(track.id)
        self._candidate_id = None
        self.state = State.EXPLORE
        return TURN_ACTION

    # ---------------------------------------------------------------- helpers

    def _plan_to(self, frame: FrameData, goal_xy: np.ndarray) -> None:
        agent_xy = frame.camera_position[list(PLANE)]
        with self.profiler.timeit("planner"):
            result: PlanResult = self.planner.plan(self.costmap, agent_xy, goal_xy)
        self._current_path = result.path if result.success else None

    def _follow_path(self, frame: FrameData) -> Optional[str]:
        goal = (
            self._current_frontier.centroid_xy
            if self.state == State.GOTO_FRONTIER and self._current_frontier is not None
            else self._goal_xy
        )
        if goal is None:
            return None
        if self._current_path is None:
            self._plan_to(frame, goal)
            if self._current_path is None:
                if self.state == State.GOTO_FRONTIER and self._current_frontier is not None:
                    self._blocked_frontiers[self._current_frontier.id] = self.step_count + 50
                return None
        action = self.controller.act(frame.T_wc, self._current_path)
        if action is None:
            self._current_path = None
        return action

    def _arrived_at_goal(self, frame: FrameData) -> bool:
        if self._goal_xy is None:
            return False
        agent_xy = frame.camera_position[list(PLANE)]
        # Stop just outside the object so the success-distance check passes
        return bool(
            np.linalg.norm(agent_xy - self._goal_xy)
            < max(self.cfg.agent.success_distance * 0.9, 0.35)
        )
