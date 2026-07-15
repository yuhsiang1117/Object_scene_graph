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
from ..exploration.selector import frontier_goal_xy, select_frontier
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
        # Location-keyed blacklist: frontier ids are reassigned on every
        # extraction, so blocking must be spatial to persist. [(xy, until)]
        self._blocked_frontier_pts: list = []
        self._candidate_id: Optional[int] = None
        self._goal_xy: Optional[np.ndarray] = None
        self._last_action: Optional[str] = None
        self._last_select_step = -100
        self._goto_deadline = 10**9
        self._progress_ref_step = 0
        self._progress_ref_xy = np.zeros(2)
        self.stats = {"plan_ok": 0, "plan_fail": 0, "select_none": 0, "select_ok": 0}
        self.state_log = []
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
            # Give-up net: no displacement for a while means an obstacle the
            # map cannot see (below the obstacle band, glass, sim collision).
            # Abandon this frontier instead of pushing against it forever.
            agent_xy = frame.camera_position[list(PLANE)]
            if self.step_count - self._progress_ref_step >= 15:
                if np.linalg.norm(agent_xy - self._progress_ref_xy) < 0.2:
                    self._block_frontier(self._current_frontier, 100)
                    self.stats["frontier_give_up"] = self.stats.get("frontier_give_up", 0) + 1
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
            action = self._follow_path(frame)
            if action is not None:
                return action
            self.state = State.VERIFYING

        if self.state == State.VERIFYING:
            return self._do_verification(frame)

        if self.state == State.GOTO_TARGET:
            # Terminal approach: one planned path, followed to its end. No
            # replanning here — with a 0.25 m step and 30 deg turns the agent
            # otherwise orbits the goal until the budget runs out.
            if self._arrived_at_goal(frame) or self.step_count > self._goto_deadline:
                self.state = State.DONE
                return STOP_ACTION
            if self._current_path is None:
                self._plan_to(frame, self._goal_xy)
                if self._current_path is None:
                    self.state = State.DONE
                    return STOP_ACTION
            action = self.controller.act(frame.T_wc, self._current_path)
            if action is None:  # path consumed: as close as the map allows
                self.state = State.DONE
                return STOP_ACTION
            return action

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
            frontiers = self.frontier_extractor.extract(self.costmap)
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
            self.stats["select_none"] += 1
            return
        self.stats["select_ok"] += 1
        self._current_frontier = best
        self._plan_to(frame, frontier_goal_xy(best, self.costmap))
        if self._current_path is not None:
            self.state = State.GOTO_FRONTIER
        else:
            self._block_frontier(best, 50)

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
            self._goal_xy = self._nearest_free_xy(obj_xy)
            self.state = State.GOTO_TARGET
            self._current_path = None
            self._goto_deadline = self.step_count + 100
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
        # Face the object first so the live view actually shows it.
        from ..planning.controller import TURN_LEFT, TURN_RIGHT, _wrap, agent_heading

        obj_xy = self.object_layer.center_of(track)[list(PLANE)]
        agent_xy = frame.camera_position[list(PLANE)]
        to_obj = obj_xy - agent_xy
        if np.linalg.norm(to_obj) > 0.05:
            err = _wrap(float(np.arctan2(to_obj[1], to_obj[0])) - agent_heading(frame.T_wc))
            if abs(err) > np.radians(20.0):
                return TURN_RIGHT if err > 0 else TURN_LEFT
        with self.profiler.timeit("verification"):
            accepted = self.verifier.verify(track, self.target, live_view=frame.rgb)
        if accepted:
            self._goal_xy = self._nearest_free_xy(
                self.object_layer.center_of(track)[list(PLANE)]
            )
            self.state = State.GOTO_TARGET
            self._current_path = None
            self._goto_deadline = self.step_count + 100
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

    def _arrived_at_goal(self, frame: FrameData) -> bool:
        if self._goal_xy is None:
            return False
        agent_xy = frame.camera_position[list(PLANE)]
        # Success distance is tight (0.1 m geodesic to a goal viewpoint):
        # drive onto the free cell nearest the object before stopping.
        return bool(np.linalg.norm(agent_xy - self._goal_xy) < 0.2)
