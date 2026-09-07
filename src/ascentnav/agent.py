"""ASCENT's navigation pipeline, driving ASCENT's own maps.

What this is, and what it is not
--------------------------------
`osg/agent/ascent_agent.py` (S37) put ASCENT's *control flow* on top of OSG's
maps. This package goes the other way and runs **ASCENT's maps** -- its
`ObstacleMap`, `ValueMap` and `ObjectPointCloudMap`, vendored under
`ascentnav/mapping/` -- under ASCENT's control flow, because those maps differ
from OSG's in kind rather than in parameters:

| | OSG costmap | ASCENT ObstacleMap |
|---|---|---|
| obstacle band | [0.15, 1.5] m | [0.61, 0.88] m |
| free space | depth raycast per frame | fog-of-war reveal over a navigable map |
| navigable | `~inflated(radius)` on demand | dilated by a square agent-radius kernel, maintained |
| frontiers | contour of the explored region | `detect_frontier_waypoints` on explored ^ navigable |
| stairs | a separate hit grid | first-class up/down maps, excluded from dilation |

Substitutions, and why each is honest
-------------------------------------
Three ASCENT components cannot run in this container, and each is replaced by
the OSG equivalent rather than skipped:

* **Detector.** ASCENT runs D-FINE + GroundingDINO + MobileSAM. `transformers`
  is not installed and the weights are HF-format. OSG's YOLOE is used instead --
  and S15 measured the detector as worth **zero** on this split (11s@512 against
  11l@640: 0 at the loose gate, -1 at the tight one), so this is the
  substitution least likely to matter.
* **Value map scorer.** ASCENT scores frames with BLIP-2 ITM; `lavis` is not
  installed. OSG's CLIP `ImageTextScorer` supplies the cosine, through the same
  `ValueMap.update_map` call. Already a documented deviation in
  `ascent_aligned.yaml`.
* **LLM.** ASCENT runs Qwen2.5-7B locally. OSG's hosted client is used. Also
  already a documented deviation.

So this is ASCENT's *navigation* -- maps, frontiers, control flow, mover -- with
OSG's perception plumbing. Given the gap this repo is chasing is navigation
(ASCENT 70% sensor-only against OSG's 63% *with* a navmesh), that is the part
worth reproducing faithfully.

The object scene graph is built alongside, from the same object cloud, and is
read-only with respect to every navigation decision.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from osg.core.types import Detection, FrameData
from osg.graph.scene_graph import SceneGraph
from osg.planning.escape import ActionHistoryEscape

from .constants import STAIR_CLASS_ID
from osg.perception.stair_seg import MIN_STAIR_PIXELS
from .geometry import (
    camera_pitch,
    normalise_depth,
    robot_xy_heading,
    tf_camera_to_episodic,
)
from .mapping.object_point_cloud_map import ObjectPointCloudMap
from .mapping.obstacle_map import ObstacleMap
from .mapping.value_map import ValueMap
from .stairs import (
    CLIMB_PAUSED_ABANDON,
    ClimbState,
    carrot_waypoint,
    ratchet_carrot,
    robot_on_stairs,
)

STOP, FORWARD, LEFT, RIGHT = "stop", "move_forward", "turn_left", "turn_right"
MAP_SIZE = 1600  # ascent/map_controller.py:35


class AscentNavAgent:
    """ASCENT's `Ascent_Policy.act` over ASCENT's maps, in OSG's harness."""

    def __init__(
        self,
        cfg,
        detector,
        scorer,
        verifier,
        target_category: str,
        *,
        image_text=None,
        pointnav=None,
        stair_segmenter=None,
        profiler=None,
        **_ignored,
    ) -> None:
        self.cfg = cfg
        self.detector = detector
        self.verifier = verifier
        self.image_text = image_text
        self.pointnav = pointnav
        self.stair_segmenter = stair_segmenter
        self.profiler = profiler

        a, e = cfg.agent, cfg.eval
        self.camera_height = float(a.camera_height)
        self.min_depth = float(getattr(e, "depth_min_m", 0.5))
        self.max_depth = float(getattr(e, "depth_max_m", 5.0))
        self.hfov = float(np.radians(e.hfov_deg))
        self.fx = self.fy = e.rgb_width / (2 * np.tan(self.hfov / 2))
        self.turn_deg = float(a.turn_deg)
        self.max_steps = int(a.max_steps)

        self.nearby_distance = float(getattr(cfg.exploration, "nearby_distance_m", 3.0))
        self.stop_radius = float(getattr(a, "pointnav_stop_radius", 0.9))
        self.abandon_steps = int(getattr(a, "approach_abandon_steps", 100) or 100)
        self.escape = ActionHistoryEscape(int(getattr(a, "escape_window", 30) or 30))

        self.scene_graph = SceneGraph()
        self.reset(target_category)

    # ------------------------------------------------------------------ reset

    def reset(self, target_category: str) -> None:
        self.target = target_category
        self.step_count = 0
        # ASCENT keeps one obstacle/value/object map PER STOREY and swaps the
        # active triple on a floor change (`map_controller.py:67-92, 253-257`).
        # One map collapses the storeys onto each other, which is what made the
        # first ascentnav run score 0.0% on the 21 cross-floor episodes.
        self._floors = [self._new_floor()]
        self._floor_idx = 0
        self.climb = ClimbState()
        self._stair_disabled: set = set()
        self._pitch_ticks = 0
        self.scene_graph = SceneGraph()

        if self.pointnav is not None:
            self.pointnav.reset()
        # YOLOE is open-vocabulary: without this it has no classes to look for
        # and detects nothing at all. `NavAgent.reset` does the same
        # (nav_agent.py:479); writing this agent from scratch rather than
        # subclassing dropped it, and the symptom was four episodes of
        # `steps_to_first_candidate: None` in a scene the baseline sweeps 5/5.
        self.detector.set_vocabulary(
            [self.target.replace("_", " ")] + list(self.cfg.detector.vocabulary)
        )
        self._init_left = int(round(360.0 / self.turn_deg)) if self.cfg.agent.initial_scan else 0
        self._navigate_steps = 0
        self._min_dist_seen = np.inf
        self._verified = False
        self._verify_after = 0
        self._disabled_frontiers: set = set()
        self._last_frontier: Optional[np.ndarray] = None
        # What this step decided, for viz only (`ascentnav/viz.py`). Kept on the
        # agent rather than passed around because the renderer runs after `act`.
        self._selected_frontier: Optional[np.ndarray] = None
        self._nav_goal: Optional[np.ndarray] = None
        self._last_dets: list = []
        self._pn_goal: Optional[np.ndarray] = None
        self._stick_steps = 0
        self._last_frontier_dist = 0.0
        self.escape.reset()

        # Runner interface
        self.stats: dict = {}
        self.state_log: list = []
        self.frontier_select_log: list = []
        self.giveup_log: list = []
        self.approach_bbox_log: list = []
        self.approach_diag: Optional[dict] = None
        self.approach_stop_reason: Optional[str] = None
        self.approach_recheck_max = 0.0
        self._approach_itm_n = 0
        self.steps_to_first_candidate: Optional[int] = None
        self.step_trace: list = []
        self._state = "init"

    def _new_floor(self) -> dict:
        return {
            "obstacle": ObstacleMap(
                min_height=float(getattr(self.cfg.agent, "ascent_min_obstacle_h", 0.61)),
                max_height=float(getattr(self.cfg.agent, "ascent_max_obstacle_h", 0.88)),
                agent_radius=float(self.cfg.agent.agent_radius),
                area_thresh=float(getattr(self.cfg.exploration, "area_thresh_m2", 1.5)),
                size=MAP_SIZE,
            ),
            "value": ValueMap(value_channels=1, size=MAP_SIZE, obstacle_map=None),
            "object": ObjectPointCloudMap(erosion_size=5, size=MAP_SIZE),
        }

    @property
    def obstacle_map(self):
        return self._floors[self._floor_idx]["obstacle"]

    @property
    def value_map(self):
        return self._floors[self._floor_idx]["value"]

    @property
    def object_map(self):
        return self._floors[self._floor_idx]["object"]

    def _switch_floor(self, direction: int) -> bool:
        """Move to the storey the staircase led to, allocating it if new.

        Up appends, down prepends and shifts the index -- ASCENT's
        `_add_floor_map` (`map_controller.py:229-252`), whose floor list is
        therefore ordered bottom-to-top rather than in visit order.
        """
        is_new = False
        if direction == 1:
            self._floor_idx += 1
            if self._floor_idx >= len(self._floors):
                self._floors.append(self._new_floor())
                is_new = True
        elif self._floor_idx == 0:
            self._floors.insert(0, self._new_floor())
            is_new = True
        else:
            self._floor_idx -= 1
        self.stats["floor_switches"] = self.stats.get("floor_switches", 0) + 1
        self.stats["n_floors"] = len(self._floors)
        # An unmapped storey gets the opening scan again -- ASCENT routes into
        # `_initialize` on arrival (`ascent_policy.py:504-512`) but only when the
        # floor is not `_done_initializing`, so a storey being revisited keeps
        # the map it already built and does not spend 12 steps turning.
        if is_new:
            self._init_left = int(round(360.0 / self.turn_deg))
        return is_new

    def debug_panel(self, frame: FrameData) -> "np.ndarray":
        """One BGR video frame: RGB + obstacle map + value map, with the
        frontier this step actually selected marked. The runner calls this when
        `eval.debug_frames` is on; nothing else in the pipeline reads it."""
        from .viz import debug_panel

        robot_xy, heading = robot_xy_heading(frame)
        return debug_panel(self, frame, self.target, robot_xy, heading, self._last_dets)

    # ------------------------------------------------- runner compatibility

    @property
    def costmap(self):
        """The occupancy view the runner renders. ASCENT's navigable map, in the
        shape OSG's visualiser expects."""
        return _CostmapView(self.obstacle_map)

    @property
    def object_layer(self):
        return _ObjectLayerView(self.object_map, self.target)

    # ------------------------------------------------------------------ act

    def act(self, frame: FrameData) -> str:
        prev = self._state
        action = self._act_inner(frame)
        if self._state != prev:
            self.state_log.append((self.step_count, self._state))
        if self._state != "done":
            action = self.escape(action)
        self.step_count += 1
        return action

    def _act_inner(self, frame: FrameData) -> str:
        if self.step_count >= self.max_steps - 1:
            self._state = "done"
            return STOP

        if self.pointnav is not None:
            # The mover reads this step's depth and pose directly.
            self.pointnav.observe(frame)
        tf = tf_camera_to_episodic(frame, self.camera_height)
        depth_n = normalise_depth(frame.depth, self.min_depth, self.max_depth)
        robot_xy, heading = robot_xy_heading(frame)
        pitch_deg = float(np.degrees(-camera_pitch(frame)))
        zeros = np.zeros(frame.depth.shape[:2], np.uint8)

        # One detector pass per step: the target filter and the stair filter
        # are two views of the same result. YOLOE dominates the step budget, so
        # calling it twice would halve the throughput of every run.
        raw = self.detector.detect(frame.rgb)
        dets = self._filter_target(raw)
        self._last_dets = dets
        self._update_object_map(frame, dets, tf, depth_n)
        # The two stair inputs ASCENT intersects (`obstacle_map.py:520-524`):
        # a detector mask AND RedNet's MPCAT40 stair class. Passing zeros here
        # is what left the first run with no floor-transition capability at all.
        seg = self._stair_seg(frame)
        stair_mask = self._stair_det_mask(raw, seg)
        self.obstacle_map.update_map(
            depth_n, tf, self.min_depth, self.max_depth, self.fx, self.fy, self.hfov,
            {}, zeros,
            stair_mask if stair_mask is not None else zeros,
            seg if seg is not None else zeros,
            pitch_deg, not self.climb.climbing, self.climb.reached, self.climb.direction,
        )
        self._stair_diag(seg, stair_mask)
        self._trace(frame, robot_xy, heading, pitch_deg, seg, stair_mask)
        self.obstacle_map.update_agent_traj(robot_xy, heading)
        # Same base-class bookkeeping, so the value map can draw the trajectory
        # too (`base_map.py:31`). Viz only -- nothing reads it for decisions.
        self.value_map.update_agent_traj(robot_xy, heading)
        self._update_value_map(frame, tf, depth_n)

        goal = self._object_goal(robot_xy)

        # ASCENT dispatches the stair branch FIRST (`ascent_policy.py:439-566`):
        # a floor transition in progress outranks initialising, exploring and
        # even a visible target.
        if self.climb.climbing:
            self._state = "climb"
            return self._do_climb(depth_n, robot_xy, heading, pitch_deg)

        if self._init_left > 0:
            self._init_left -= 1
            self._state = "explore"
            return LEFT
        if goal is None:
            self._state = "explore"
            self._navigate_steps = 0
            return self._explore(robot_xy, heading, depth_n, pitch_deg)
        self._state = "approach"
        return self._navigate(robot_xy, heading, goal)

    # ------------------------------------------------------------ perception

    def _filter_target(self, dets: List[Detection]) -> List[Detection]:
        target = self.target.lower().replace("_", " ").strip()
        return [d for d in dets
                if d.label.lower().replace("_", " ").strip() == target]

    def _trace(self, frame, robot_xy, heading, pitch_deg, seg, stair_mask) -> None:
        """One row per step, for offline diagnosis of a stuck episode.

        Cheap (about 20 numbers) and always collected; the runner only writes it
        out when `eval.debug_frames` is on. The whole point is to separate "the
        stairs were never seen" from "the stairs were seen and not climbed",
        which the aggregate counters cannot do.
        """
        om = self.obstacle_map
        f = None
        if self.climb.climbing:
            fr = self._stair_frontier()
            f = None if fr is None else [round(float(v), 2) for v in fr]
        self.step_trace.append({
            "s": self.step_count,
            "state": self._state,
            "xy": [round(float(v), 2) for v in robot_xy],
            "h": round(float(frame.camera_position[1]), 2),
            "yaw": round(float(heading), 2),
            "pitch": round(pitch_deg, 1),
            "floor": self._floor_idx,
            "nfront": int(np.atleast_2d(np.asarray(om.frontiers)).reshape(-1, 2).shape[0]),
            "seg_px": int(np.count_nonzero(seg == STAIR_CLASS_ID)) if seg is not None else 0,
            "det_px": int(np.count_nonzero(stair_mask)) if stair_mask is not None else 0,
            "up_px": int(om._up_stair_map.sum()),
            "dn_px": int(om._down_stair_map.sum()),
            "up_f": [round(float(v), 2) for v in np.asarray(om._up_stair_frontiers).reshape(-1, 2)[0]]
                    if np.size(om._up_stair_frontiers) else None,
            "dn_f": [round(float(v), 2) for v in np.asarray(om._down_stair_frontiers).reshape(-1, 2)[0]]
                    if np.size(om._down_stair_frontiers) else None,
            "climb": (self.climb.direction if self.climb.climbing else 0),
            "reach": int(self.climb.reached),
            "cent": int(self.climb.reached_centroid),
            "paused": self.climb.paused,
            "stair_f": f,
            "pn_goal": (None if getattr(self, "_pn_goal", None) is None
                        else [round(float(v), 2) for v in self._pn_goal]),
            "pn_resets": int(getattr(self.pointnav, "n_resets", 0)) if self.pointnav else 0,
            "on_stairs": int(robot_on_stairs(
                om._up_stair_map if self.climb.direction != 2 else om._down_stair_map,
                self._robot_px(robot_xy),
                self.cfg.agent.agent_radius * om.pixels_per_meter)),
        })

    def _stair_diag(self, seg, stair_mask) -> None:
        """Counters for the two ways the stair path can be silently dead: the
        fused mask never firing, and the maps never producing a frontier."""
        st = self.stats
        if seg is not None and int(np.count_nonzero(seg == STAIR_CLASS_ID)) > MIN_STAIR_PIXELS:
            st["stair_seg_frames"] = st.get("stair_seg_frames", 0) + 1
        if stair_mask is not None and np.any(stair_mask):
            st["stair_mask_frames"] = st.get("stair_mask_frames", 0) + 1
        om = self.obstacle_map
        st["up_stair_px"] = max(st.get("up_stair_px", 0), int(om._up_stair_map.sum()))
        st["down_stair_px"] = max(st.get("down_stair_px", 0), int(om._down_stair_map.sum()))
        if om._has_up_stair:
            st["has_up_stair_steps"] = st.get("has_up_stair_steps", 0) + 1
        if om._has_down_stair:
            st["has_down_stair_steps"] = st.get("has_down_stair_steps", 0) + 1

    def _stair_det_mask(self, dets: List[Detection], seg):
        """The detector half of ASCENT's stair fusion.

        ASCENT appends `" stair ."` to its GroundingDINO caption every step
        (`map_controller.py:700-704`) and ANDs the result with RedNet
        (`obstacle_map.py:520-524`). YOLOE carries `stairs` in this agent's
        vocabulary and is the natural stand-in -- but S33 measured the two
        candidates over 250 stair poses from this very split:

            YOLOE `stairs`                    10%  recall  [0% control]
            RedNet AND YOLOE (ASCENT's rule)  10%          [0%]
            RedNet alone                      54%          [3%]

        An intersection cannot beat its weaker input, and all 26 YOLOE firings
        already sit inside RedNet's 134. Keeping ASCENT's AND here would import
        GroundingDINO's job without GroundingDINO and cap stair recall at 10%,
        so the default UNIONS the two, which makes the map's own AND a no-op and
        leaves RedNet deciding. `agent.stair_up_mode: ascent` restores the
        strict fusion for an A/B.
        """
        m = None
        for d in dets:
            if d.label.lower().strip() in ("stairs", "stair", "staircase", "steps"):
                m = d.mask.astype(bool) if m is None else (m | d.mask.astype(bool))
        if str(getattr(self.cfg.agent, "stair_up_mode", "rednet")) != "ascent" and seg is not None:
            rednet = seg == STAIR_CLASS_ID
            m = rednet if m is None else (m | rednet)
        return None if m is None else m.astype(np.uint8)

    def _stair_seg(self, frame: FrameData):
        if self.stair_segmenter is None:
            return None
        m = self.stair_segmenter.stair_mask(frame)
        if m is None:
            return None
        return np.where(m, STAIR_CLASS_ID, 0).astype(np.uint8)

    def _update_object_map(self, frame, dets, tf, depth_n) -> None:
        cone = 2 * np.arctan(frame.depth.shape[1] / (2 * self.fx))
        if dets:
            best = max(dets, key=lambda d: d.score)
            if self.steps_to_first_candidate is None:
                self.steps_to_first_candidate = self.step_count
            self._maybe_verify(frame, best)
            # uint8, not bool: `_extract_object_cloud` does `object_mask * 255`
            # and hands the result to cv2.erode, which rejects the int64 a bool
            # array promotes to. ASCENT's masks arrive from ObjectDetections
            # already uint8, so its own path never sees this.
            self.object_map.update_map(
                self.target, depth_n, best.mask.astype(np.uint8), tf,
                self.min_depth, self.max_depth, self.fx, self.fy,
            )
        self.object_map.update_explored(tf, self.max_depth, cone)

    def _update_value_map(self, frame, tf, depth_n) -> None:
        if self.image_text is None:
            value = 0.0
        else:
            prompt = str(getattr(self.cfg.exploration, "value_prompt",
                                 "Seems like there is a {target} ahead."))
            # OSG's interface is `score(rgb, [texts]) -> array`; ASCENT's
            # BLIP-2 client exposes `cosine(rgb, text)`. Same quantity, and this
            # is the seam where the scorer is substituted.
            text = prompt.replace("{target}", self.target.replace("_", " "))
            value = float(np.asarray(self.image_text.score(frame.rgb, [text])).reshape(-1)[0])
            self.stats["value_calls"] = self.stats.get("value_calls", 0) + 1
        self.value_map.update_map(
            np.array([value]), depth_n, tf, self.min_depth, self.max_depth, self.hfov)

    # ------------------------------------------------------------- the goal

    def _object_goal(self, robot_xy) -> Optional[np.ndarray]:
        if not self.object_map.has_object(self.target):
            return None
        # 2D, as ASCENT passes (`ascent_policy.py:441` hands it `robot_xy`).
        # `get_best_object` subtracts this from a 2D point for its hysteresis
        # test, so a 3D position fails to broadcast -- while `_get_closest_point`
        # alone accepts either, which is why this only shows up once an object
        # is actually detected.
        return self.object_map.get_best_object(self.target, np.asarray(robot_xy, dtype=float))

    def _cur_dist_to_goal(self, robot_xy) -> float:
        if not self.object_map.has_object(self.target):
            return float("inf")
        cloud = self.object_map.get_target_cloud(self.target)
        if len(cloud) == 0:
            return float("inf")
        pos = np.array([robot_xy[0], robot_xy[1], self.camera_height])
        closest = self.object_map._get_closest_point(cloud, pos)
        return float(np.linalg.norm(closest[:2] - pos[:2]))

    # ------------------------------------------------------------- exploring

    def _explore(self, robot_xy, heading, depth_n=None, pitch_deg=0.0) -> str:
        """`ascent_policy.py:648-711`, on ASCENT's own frontier waypoints."""
        frontiers = [f for f in self.obstacle_map.frontiers
                     if tuple(np.round(f, 3)) not in self._disabled_frontiers]
        if len(frontiers) == 0:
            # This floor is exhausted -- ASCENT's own definition of done
            # (`ascent_policy.py:655`). Leave it if a staircase is known.
            self.obstacle_map._this_floor_explored = True
            self.stats["no_frontier_steps"] = self.stats.get("no_frontier_steps", 0) + 1
            self._selected_frontier = None
            if depth_n is not None and self._maybe_start_climb():
                self._state = "climb"
                # ASCENT starts driving to the staircase on this very step
                # (`ascent_policy.py:795`), it does not burn one turning.
                return self._do_climb(depth_n, robot_xy, heading, pitch_deg)
            return LEFT  # nothing to go to; keep turning so the map grows

        best = self._best_frontier(np.array(frontiers), robot_xy)
        self._selected_frontier = best
        if best is None:
            return LEFT
        self._sticky(best, robot_xy)
        self.frontier_select_log.append((
            self.step_count, [round(float(v), 2) for v in robot_xy],
            [round(float(v), 2) for v in best], None, len(frontiers)))
        # stop_radius 0, not 0.9. ASCENT passes `stop=False` for every frontier
        # call, which makes its `rho < stop_radius` branch inert
        # (`ascent_policy.py:869-872`) -- there is no such thing as arriving at a
        # frontier. Handing the driver 0.9 instead made it report "arrived" for
        # anything within 0.9 m, and each of those became a blind forced-forward
        # rather than an action the network chose: 189 of 500 steps on the first
        # smoke.
        action = self._pointnav(robot_xy, heading, best, stop_radius=0.0)
        if action is None:
            # ASCENT overrides a network STOP on an explore frontier with a
            # forward step and keeps the target (`:708-710`).
            self.stats["explore_forced_forward"] = self.stats.get("explore_forced_forward", 0) + 1
            return FORWARD
        return action

    def _best_frontier(self, frontiers, robot_xy) -> Optional[np.ndarray]:
        """Value argmax with ASCENT's nearby shortcut (`llm_planner.py:157-179`).

        No division by path cost: ASCENT ranks on value and lets distance in
        only as a hard shortcut for anything within `nearby_distance`.
        """
        if len(frontiers) == 1:
            return frontiers[0]
        dists = np.linalg.norm(frontiers - robot_xy, axis=1)
        near = np.where(dists < self.nearby_distance)[0]
        if len(near):
            return frontiers[near[int(np.argmin(dists[near]))]]
        sorted_pts, sorted_vals = self.value_map.sort_waypoints(frontiers, 0.5)
        return sorted_pts[0] if len(sorted_pts) else frontiers[int(np.argmin(dists))]

    def _sticky(self, best, robot_xy) -> None:
        """`_handle_frontier_stick_and_disable` (llm_planner.py:239-257)."""
        if self._last_frontier is not None and np.allclose(self._last_frontier, best, atol=1e-3):
            cur = float(np.linalg.norm(best - robot_xy))
            if self._stick_steps == 0:
                self._last_frontier_dist, self._stick_steps = cur, 1
            elif abs(self._last_frontier_dist - cur) > 0.3:
                self._stick_steps, self._last_frontier_dist = 0, cur
            elif self._stick_steps >= 20:
                self._disabled_frontiers.add(tuple(np.round(best, 3)))
                self._stick_steps = 0
                self.giveup_log.append((self.step_count,
                                        [round(float(v), 2) for v in best],
                                        [round(float(v), 2) for v in robot_xy]))
                self.stats["frontier_give_up"] = self.stats.get("frontier_give_up", 0) + 1
            else:
                self._stick_steps += 1
        else:
            self._stick_steps, self._last_frontier_dist = 0, 0.0
        self._last_frontier = np.asarray(best, dtype=float).copy()

    # ------------------------------------------------------------ navigating

    def _navigate(self, robot_xy, heading, goal) -> str:
        """`ascent_policy.py:876-938`."""
        self._nav_goal = np.asarray(goal, dtype=float)
        self._navigate_steps += 1
        d = self._cur_dist_to_goal(robot_xy)

        if d < 1.0:
            closing_stalled = abs(d - self._min_dist_seen) < 0.1
            if d <= 0.6 or closing_stalled:
                if self._verified or self.verifier is None:
                    self._state = "done"
                    self.approach_stop_reason = "nearest_point"
                    return STOP
                self._give_up_on_target("unverified")
                return LEFT
            self._min_dist_seen = min(self._min_dist_seen, d)
            return FORWARD

        if self._navigate_steps >= self.abandon_steps:
            self._give_up_on_target("abandon")
            return LEFT

        action = self._pointnav(robot_xy, heading, goal, stop_radius=0.0)
        if action is None:
            self.stats["navigate_forced_forward"] = self.stats.get("navigate_forced_forward", 0) + 1
            return FORWARD
        return action

    def _give_up_on_target(self, why: str) -> None:
        """ASCENT clears the cloud and disables the cells (`:917-922`)."""
        cloud = self.object_map.get_target_cloud(self.target) \
            if self.object_map.has_object(self.target) else None
        if cloud is not None and len(cloud):
            self.object_map._disabled_object_map[self.object_map._map == 1] = 1
        self.object_map.clouds = {}
        self.object_map._map.fill(0)
        self._navigate_steps = 0
        self._nav_goal = None
        self._min_dist_seen = np.inf
        self._verified = False
        self._verify_after = self.step_count
        self._state = "explore"
        self.stats[f"give_up_{why}"] = self.stats.get(f"give_up_{why}", 0) + 1

    # ---------------------------------------------------------------- stairs

    def _robot_px(self, robot_xy):
        return self.obstacle_map._xy_to_px(np.atleast_2d(robot_xy))

    def _stair_frontier(self):
        om = self.obstacle_map
        f = om._up_stair_frontiers if self.climb.direction == 1 else om._down_stair_frontiers
        return None if f is None or np.size(f) == 0 else np.asarray(f)[0]

    def _maybe_start_climb(self) -> bool:
        """Leave this floor when it has nothing left to explore.

        ASCENT's rule, and it is deliberately not a step count: a floor is done
        when it has NO explore frontiers left (`ascent_policy.py:655`), and only
        then does it look for a staircase to an unexplored storey
        (`_navigate_stair_if_unexplored_floor`, `:762-798`). Up is tried before
        down, and a direction already climbed is skipped -- `:670-674`.
        """
        om = self.obstacle_map
        for direction, has, done in ((1, om._has_up_stair, om._explored_up_stair),
                                     (2, om._has_down_stair, om._explored_down_stair)):
            if not has or done:
                continue
            f = (om._up_stair_frontiers if direction == 1 else om._down_stair_frontiers)
            if f is None or np.size(f) == 0:
                continue
            if tuple(np.round(np.asarray(f)[0], 2)) in self._stair_disabled:
                continue
            self.climb.start(direction)
            self.stats["climb_attempt"] = self.stats.get("climb_attempt", 0) + 1
            return True
        return False

    def _do_climb(self, depth_n, robot_xy, heading, pitch_deg) -> str:
        """One step of a floor transition.

        This is `ascent_policy.py:447-520` (the dispatch), `:940-1016`
        (`_get_close_to_stair`) and `:1018-1140` (`_climb_stair`) folded
        together, with the state transitions that ASCENT keeps in
        `map_controller._process_stair_climb_state` (`:259-316`) inlined --
        there is one env here, so the split buys nothing.
        """
        om = self.obstacle_map
        stair_map = om._up_stair_map if self.climb.direction == 1 else om._down_stair_map
        px = self._robot_px(robot_xy)
        on_stairs = robot_on_stairs(stair_map, px, self.cfg.agent.agent_radius * om.pixels_per_meter)
        f = self._stair_frontier()
        if f is None:
            # The staircase stopped being extracted underneath us.
            self.climb.reset()
            self._state = "explore"
            return LEFT

        # --- phase 0: walk to the foot of the stairs --------------------
        if not self.climb.reached:
            if not on_stairs:
                return self._approach_stair(f, robot_xy, heading)
            self.climb.reached = True
            self.climb.get_close_steps = 0
            self.climb.stick_steps = 0
            self.climb.last_dist = 0.0

        # --- the pause counter, `ascent_policy.py:1036-1044` -------------
        # Measured against the staircase centroid, not against the carrot: the
        # carrot moves with the agent, so it can never register a stall.
        dist = float(np.linalg.norm(f - robot_xy))
        if abs(self.climb.last_dist - dist) > 0.2:
            self.climb.paused = 0
            self.climb.last_dist = dist
        else:
            self.climb.paused += 1
        if self.climb.paused > 15:
            self.climb.disable_end = True

        # --- did the transition end? `map_controller.py:279-315` ---------
        # Off the stair map having once been on its centroid == arrived on the
        # next storey. A stall of 30 says the same geometry is no longer moving
        # us, and ASCENT reads that as a bad staircase rather than a slow one.
        if self.climb.reached_centroid and not on_stairs:
            return self._finish_climb(px)
        if self.climb.paused >= CLIMB_PAUSED_ABANDON:
            return self._abandon_stair()

        # --- pitch: a descent has to look at the treads ------------------
        # `ascent_policy.py:448-456` tilts to -60 while closing on the centroid
        # and back to -30 for the flight itself. Up-stairs gets no tilt: S14a
        # measured pitch as harmful there, and ASCENT's own up-stair look_up is
        # a detection aid, not part of the climb.
        if self.climb.direction == 2:
            if not self.climb.reached_centroid and pitch_deg > -55.0:
                return "look_down"
            if self.climb.reached_centroid and pitch_deg < -35.0:
                return "look_up"

        # --- phase 1: reach the staircase centroid -----------------------
        if not self.climb.reached_centroid:
            if dist <= 0.3:                       # map_controller.py:274-277
                self.climb.reached_centroid = True
            else:
                action = self._pointnav(robot_xy, heading, f, stop_radius=0.0)
                if action is None:                # ascent_policy.py:1061-1065
                    self.climb.reached_centroid = True
                    return FORWARD
                return action

        # --- phase 2: the carrot ----------------------------------------
        fresh = carrot_waypoint(depth_n, robot_xy, heading, self.hfov,
                                float(getattr(self.cfg.agent, "climb_carrot_m", 0.8)))
        if fresh is None:
            return FORWARD
        end_px = om._up_stair_end if self.climb.direction == 1 else om._down_stair_end
        self.climb.carrot_xy = ratchet_carrot(
            self.climb.carrot_xy, fresh, end_px, px,
            lambda xy: om._xy_to_px(np.atleast_2d(xy)),
            om.pixels_per_meter, self.climb.disable_end)
        om._carrot_goal_px = om._xy_to_px(np.atleast_2d(self.climb.carrot_xy))
        action = self._pointnav(robot_xy, heading, self.climb.carrot_xy, stop_radius=0.0)
        if action is None:
            # A STOP on a staircase means the treads fill the view, which is the
            # one moment not to stop (`ascent_policy.py:1136-1139`).
            self.stats["climb_forced_forward"] = self.stats.get("climb_forced_forward", 0) + 1
            return FORWARD
        return action

    def _approach_stair(self, f, robot_xy, heading) -> str:
        """`ascent_policy.py:963-1016` -- pointnav to the staircase centroid,
        retiring it if 30 steps pass without closing 0.3 m or the approach runs
        past 60 stalled steps."""
        if self.climb.stuck_on_approach(float(np.linalg.norm(f - robot_xy))):
            return self._abandon_stair()
        action = self._pointnav(robot_xy, heading, f, stop_radius=0.0)
        if action is None:
            return self._abandon_stair()   # ascent_policy.py:1011-1015
        return action

    def _abandon_stair(self) -> str:
        """`map_controller._disable_stair_and_reset_state` (`:328-386`)."""
        f = self._stair_frontier()
        if f is not None:
            self._stair_disabled.add(tuple(np.round(f, 2)))
        om = self.obstacle_map
        if self.climb.direction == 1:
            om._disabled_stair_map[om._up_stair_map == 1] = 1
            om._up_stair_map.fill(0)
            om._has_up_stair = False
        else:
            om._disabled_stair_map[om._down_stair_map == 1] = 1
            om._down_stair_map.fill(0)
            om._has_down_stair = False
            om._look_for_downstair_flag = False
        self.climb.reset()
        self.stats["climb_fail"] = self.stats.get("climb_fail", 0) + 1
        self._state = "explore"
        return LEFT

    def _finish_climb(self, px) -> str:
        """Record where the flight ended and move onto the new storey's maps."""
        old = self.obstacle_map
        direction = self.climb.direction
        if direction == 1:
            old._up_stair_end = px[0].copy()
            old._explored_up_stair = True
        else:
            old._down_stair_end = px[0].copy()
            old._explored_down_stair = True
        self.climb.reset()
        self.stats["climb_ok"] = self.stats.get("climb_ok", 0) + 1
        if self._switch_floor(direction):   # also re-arms the opening scan
            self._link_stair_to_new_floor(old, direction)
        self._state = "explore"
        return LEFT

    def _link_stair_to_new_floor(self, old, direction: int) -> None:
        """Hand the staircase we just climbed to the storey we arrived on, with
        its ends swapped and already marked explored.

        `map_controller._update_linked_stair_map` (`:433-476`) plus the
        `_explored_*_stair = True` at `:570-572`. Without this the new floor
        rediscovers the same flight as a fresh down-staircase and, the moment it
        runs out of frontiers, climbs straight back down -- and then up again.
        """
        new = self.obstacle_map
        if direction == 1:
            new._down_stair_map = old._up_stair_map.copy()
            new._down_stair_frontiers = np.array(old._up_stair_frontiers).copy()
            new._down_stair_start = np.array(old._up_stair_end).copy()
            new._down_stair_end = np.array(old._up_stair_start).copy()
            new._has_down_stair = True
            new._explored_down_stair = True
        else:
            new._up_stair_map = old._down_stair_map.copy()
            new._up_stair_frontiers = np.array(old._down_stair_frontiers).copy()
            new._up_stair_start = np.array(old._down_stair_end).copy()
            new._up_stair_end = np.array(old._down_stair_start).copy()
            new._has_up_stair = True
            new._explored_up_stair = True

    # ---------------------------------------------------------------- mover

    def _pointnav(self, robot_xy, heading, goal, stop_radius) -> Optional[str]:
        if self.pointnav is None:
            return FORWARD
        # The driver works in OSG's plane, so hand it the goal back in that
        # frame -- `to_ccw_frame` is an involution, so one flip undoes the other.
        self._pn_goal = np.asarray(goal, dtype=float)
        nav = self.pointnav.step(np.array([goal[0], -goal[1]]), stop_radius=stop_radius)
        return nav.action

    def _maybe_verify(self, frame, det) -> None:
        """Latch the goal gate, as ASCENT latches `_double_check_goal`.

        ASCENT's gate is a BLIP-2 cosine >= 0.15 re-evaluated every step until it
        passes (`map_controller.py:770-776`). BLIP-2 is not available here, and
        S26 measured CLIP's whole-image cosine as carrying no signal about
        commit correctness (AUC 0.479), so a CLIP threshold would be an
        arbitrary number dressed as ASCENT's. OSG's VLM verifier is used
        instead -- the largest measured win in this log -- kept in ASCENT's
        SHAPE: asked repeatedly until it passes, and never used to retire a
        target.

        Note what is NOT here: OSG's commit gate (`verification.min_score` 0.70,
        `min_obs`, `min_bbox_px`, `min_evidence`, applied in
        `object_layer.candidates`). Those four are set by this preset and are
        inert under this agent, which ingests any detection the detector emits
        above its own `conf` and commits as soon as a cloud exists. S13 measured
        the `min_score` raise alone at net +3 episodes on the navmesh arm, so
        this is untested headroom rather than a considered omission.

        A rejection costs a cooldown rather than the object, which is the S38
        finding: a NO from across a room is evidence about that view.
        """
        if self.verifier is None or self._verified:
            self._verified = True
            return
        if self.step_count < self._verify_after:
            return
        self._verify_after = self.step_count + int(
            getattr(self.cfg.verification, "reject_cooldown_steps", 100) or 100
        )
        self.stats["verify_calls"] = self.stats.get("verify_calls", 0) + 1
        if self.verifier.verify_bbox(frame.rgb, det.bbox_xyxy, self.target):
            self._verified = True
        else:
            self.stats["verify_reject"] = self.stats.get("verify_reject", 0) + 1


class _CostmapView:
    """Enough of `Costmap2D` for the runner's visualiser."""

    def __init__(self, om) -> None:
        self._om = om
        self.resolution = 1.0 / om.pixels_per_meter
        self.grid = np.where(om._map.astype(bool), 100, np.where(om.explored_area, 0, -1)).astype(np.int8)
        self.origin = np.array([-om._episode_pixel_origin[1] * self.resolution,
                                -om._episode_pixel_origin[0] * self.resolution])

    def world_to_grid(self, xy):
        return np.floor((np.asarray(xy) - self.origin) / self.resolution).astype(int)

    def grid_to_world(self, rc):
        return self.origin + (np.asarray(rc, dtype=float) + 0.5) * self.resolution

    def coverage_cells(self) -> int:
        return int((self.grid != -1).sum())


class _ObjectLayerView:
    """Enough of `ObjectLayer` for the runner's per-episode fields."""

    def __init__(self, om, target) -> None:
        self._om, self._target = om, target

    def tracks(self, include_blacklisted: bool = False):
        return []

    def get(self, _tid):
        return None
