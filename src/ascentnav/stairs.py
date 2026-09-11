"""ASCENT's stair machinery, transcribed for one environment.

`Map_Controller` keeps these flags in a dozen parallel per-env lists and
`Ascent_Policy.act` dispatches on them (`ascent_policy.py:447-557`). One object
per agent is the same state with the indexing removed. Every rule below cites
the line it comes from, and where the reference is inconsistent with itself
the reference wins:

- F3: `_disable_stair_and_reset_state` (`map_controller.py:328-380`) zeroes
  `_climb_stair_flag` at :350 BEFORE testing it at :354/:367, so the stair-map
  burn and the neighbour-floor deletion never run. A failed staircase is
  retried. `burn_on_disable` reproduces that as the default (False) and offers
  the intended behaviour as an A/B.
- F5: the pause >= 30 branch (`act` :459-514) does NOT change floor. It copies
  the stair into the neighbour floor's map, levels the camera and re-runs
  `_initialize` on the SAME floor. The floor index changes only in
  `_process_stair_climb_state` (:299-315).
- F4: `act` :517-530 ("unknowingly reached the stairs") tests
  `_climb_stair_over` inside `if not _climb_stair_over` and is dead. Passive
  entry is `_detect_passive_stair_entry` (:626-672) only.

The three sticky-rule counters the stair approach shares with the LLM planner
(`ascent_policy.py:1015-1042`) live on the planner; this module reads them
through the agent.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .constants import PASSIVE_STAIR_DETECTION_THRESHOLD

# ascent/constants.py:234-236, reused for the stair approach
STICK_DISTANCE_M = 0.3
GET_CLOSE_MAX_STEPS = 60
CLIMB_PAUSED_ABANDON = 30


def robot_on_stairs(stair_map: np.ndarray, robot_px: np.ndarray, radius_px: float) -> bool:
    """Is any stair cell inside the agent's footprint?

    `Map_Controller.is_robot_in_stair_map_fast` (`map_controller.py:181-227`),
    returning only the boolean the callers use.
    """
    if stair_map is None or not np.any(stair_map):
        return False
    x, y = float(robot_px[0, 0]), float(robot_px[0, 1])
    rows, cols = stair_map.shape
    x0, x1 = max(0, int(x - radius_px)), min(cols - 1, int(x + radius_px))
    y0, y1 = max(0, int(y - radius_px)), min(rows - 1, int(y + radius_px))
    if x0 > x1 or y0 > y1:
        return False
    sub = stair_map[y0:y1 + 1, x0:x1 + 1]
    yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
    mask = (yy - y) ** 2 + (xx - x) ** 2 <= radius_px ** 2
    return bool(np.any(sub[mask]))


def stairs_in_upper_half(seg_stair_mask) -> bool:
    """`check_stairs_in_upper_50_percent` (`ascent/utils.py:163-183`): more than
    50 stair pixels in the top half of the frame."""
    if seg_stair_mask is None or not np.any(seg_stair_mask):
        return False
    return int(np.sum(seg_stair_mask[: seg_stair_mask.shape[0] // 2])) > 50


def carrot_waypoint(depth_normalised: np.ndarray, robot_xy: np.ndarray, heading: float,
                    hfov: float, carrot_m: float = 0.8) -> Optional[np.ndarray]:
    """Steer at the farthest thing in view (`ascent_policy.py:1128-1140`)."""
    if depth_normalised.size == 0:
        return None
    mx = float(np.max(depth_normalised))
    idx = np.argwhere(depth_normalised == mx)
    if idx.size == 0:
        return None
    u = float(np.mean(idx[:, 1]))
    cx = depth_normalised.shape[1] / 2.0
    offset = float(np.clip((u - cx) / cx, -1.0, 1.0)) * (hfov / 2.0)
    target = (heading - offset) % (2 * np.pi)
    return np.asarray(robot_xy, dtype=float) + carrot_m * np.array([np.cos(target), np.sin(target)])


def ratchet_carrot(current: Optional[np.ndarray], fresh: np.ndarray, stair_end_px,
                   robot_px: np.ndarray, to_px, pixels_per_meter: float,
                   disable_end: bool) -> np.ndarray:
    """Keep whichever carrot is closer to the recorded stair end
    (`ascent_policy.py:1146-1163`), by L1 in map pixels as ASCENT does."""
    if current is None or stair_end_px is None or np.size(stair_end_px) == 0 or disable_end:
        return fresh
    if np.linalg.norm(np.asarray(stair_end_px) - robot_px[0]) <= 0.5 * pixels_per_meter:
        return fresh
    l1 = lambda p: abs(stair_end_px[0] - p[0][0]) + abs(stair_end_px[1] - p[0][1])
    return fresh if l1(to_px(fresh)) < l1(to_px(current)) else current


class StairController:
    """`Map_Controller`'s per-env stair state and its transitions."""

    def __init__(self, agent_radius: float, burn_on_disable: bool = False,
                 passive_entry: bool = True, stats: Optional[dict] = None) -> None:
        self.agent_radius = float(agent_radius)
        self.burn_on_disable = bool(burn_on_disable)
        self.passive_entry = bool(passive_entry)
        self.stats = stats if stats is not None else {}
        self.reset()

    def reset(self) -> None:
        # `Map_Controller.reset` (:133-180)
        self.climb_stair_over = True
        self.climb_stair_flag = 0          # 1 up, 2 down
        self.reach_stair = False
        self.reach_stair_centroid = False
        self.stair_dilate_flag = False
        self.get_close_to_stair_step = 0
        self.stair_frontier: Optional[np.ndarray] = None
        self.temp_stair_map: Optional[np.ndarray] = None
        self.carrot_goal_xy: Optional[np.ndarray] = None
        self.last_carrot_xy: Optional[np.ndarray] = None
        self.last_carrot_px: Optional[np.ndarray] = None
        self.passive_up_steps = 0
        self.passive_down_steps = 0

    # ---------------------------------------------------------- predicates

    @property
    def climbing(self) -> bool:
        return not self.climb_stair_over

    @property
    def direction(self) -> int:
        return self.climb_stair_flag

    def on_stairs(self, om, robot_px, stair_map=None) -> bool:
        m = stair_map
        if m is None:
            m = om._up_stair_map if self.climb_stair_flag == 1 else om._down_stair_map
        return robot_on_stairs(m, robot_px, self.agent_radius * om.pixels_per_meter)

    # ----------------------------------------------------- state resets

    def _reset_climb_state(self, om) -> None:
        """`_reset_stair_climb_state` (:318-326)."""
        self.reach_stair = False
        self.reach_stair_centroid = False
        self.stair_dilate_flag = False
        self.climb_stair_flag = 0
        om._climb_stair_paused_step = 0
        self.last_carrot_xy = None
        self.last_carrot_px = None

    def update_stair_state(self, om) -> None:
        """`_update_stair_state` (:382-389)."""
        om._climb_stair_paused_step = 0
        self.climb_stair_over = True
        self.climb_stair_flag = 0
        self.reach_stair = False
        self.reach_stair_centroid = False
        self.stair_dilate_flag = False

    def disable_stair_and_reset(self, agent, om, disabled_frontier: np.ndarray,
                                is_reverse: bool = False) -> None:
        """`_disable_stair_and_reset_state` (:328-380), F3 preserved.

        The reference zeroes `_climb_stair_flag` before the `== 1 / == 2`
        branches, so those branches never run. `burn_on_disable=True` runs them
        with the flag captured first -- the behaviour the code was written for.
        """
        flag_before = self.climb_stair_flag
        if np.size(disabled_frontier) > 0:
            om._disabled_frontiers.add(tuple(np.asarray(disabled_frontier).reshape(-1)[:2]))
        self.get_close_to_stair_step = 0
        agent.planner.frontier_stick_step = 0
        om._climb_stair_paused_step = 0
        self.last_carrot_xy = None
        self.last_carrot_px = None
        self.reach_stair = False
        self.reach_stair_centroid = False
        self.stair_dilate_flag = False
        self.climb_stair_over = True
        self.climb_stair_flag = 0
        om._disable_end = False
        self.stats["climb_fail"] = self.stats.get("climb_fail", 0) + 1
        flag = flag_before if self.burn_on_disable else self.climb_stair_flag
        if flag == 1:
            om._disabled_stair_map[om._up_stair_map == 1] = 1
            om._up_stair_map.fill(0)
            om._up_stair_frontiers = np.array([])
            om._has_up_stair = False
            om._look_for_downstair_flag = False
            if not is_reverse and agent._floor_idx + 1 < len(agent._floors):
                del agent._floors[agent._floor_idx + 1]
        elif flag == 2:
            om._disabled_stair_map[om._down_stair_map == 1] = 1
            om._down_stair_map.fill(0)
            om._down_stair_frontiers = np.array([])
            om._has_down_stair = False
            om._look_for_downstair_flag = False
            if not is_reverse and agent._floor_idx - 1 >= 0:
                del agent._floors[agent._floor_idx - 1]
                agent._floor_idx -= 1

    # ------------------------------------------------ per-step, pre-map

    def pre_update(self, agent, om, robot_xy: np.ndarray, robot_px: np.ndarray) -> None:
        """`_update_obstacle_map` :481-506: passive entry, then the climb
        state machine on a once-dilated copy of the active stair map."""
        if self.climb_stair_over and self.climb_stair_flag == 0:
            if self.passive_entry:
                self.detect_passive_entry(agent, om, robot_px)
        if not self.climb_stair_over:
            stair_map = None
            if self.climb_stair_flag == 1:
                stair_map = om._up_stair_map
            elif self.climb_stair_flag == 2:
                stair_map = om._down_stair_map
            if stair_map is not None:
                if not self.stair_dilate_flag:
                    self.temp_stair_map = cv2.dilate(stair_map.astype(np.uint8), (7, 7), iterations=1)
                    self.stair_dilate_flag = True
                else:
                    self.temp_stair_map = stair_map
                self.process_climb_state(agent, om, robot_xy, robot_px,
                                         self.temp_stair_map, self.climb_stair_flag)

    def process_climb_state(self, agent, om, robot_xy, robot_px, stair_map, direction: int) -> None:
        """`_process_stair_climb_state` (:259-316)."""
        on = self.on_stairs(om, robot_px, stair_map)
        if not self.reach_stair:
            if on:
                self.reach_stair = True
                self.get_close_to_stair_step = 0
                if direction == 1:
                    om._up_stair_start = robot_px[0].copy()
                else:
                    om._down_stair_start = robot_px[0].copy()
        elif not self.reach_stair_centroid:
            if (self.stair_frontier is not None and np.size(self.stair_frontier)
                    and np.linalg.norm(np.asarray(self.stair_frontier).reshape(-1, 2)[0]
                                       - np.atleast_2d(robot_xy)) <= 0.3):
                self.reach_stair_centroid = True
        else:
            if not on and om._climb_stair_paused_step >= 30:
                # :279-297 -- a staircase that stalled for 30 steps and is no
                # longer under the agent: burn it, drop the neighbour floor.
                sf = np.asarray(self.stair_frontier).reshape(-1, 2)
                self._reset_climb_state(om)
                self.climb_stair_over = True
                if len(sf):
                    om._disabled_frontiers.add(tuple(sf[0]))
                self.stats["climb_fail"] = self.stats.get("climb_fail", 0) + 1
                if direction == 1:
                    om._disabled_stair_map[om._up_stair_map == 1] = 1
                    om._up_stair_map.fill(0)
                    om._has_up_stair = False
                    if agent._floor_idx + 1 < len(agent._floors):
                        del agent._floors[agent._floor_idx + 1]
                else:
                    om._disabled_stair_map[om._down_stair_map == 1] = 1
                    om._down_stair_frontiers = np.zeros_like(np.asarray(om._down_stair_frontiers))
                    om._has_down_stair = False
                    om._look_for_downstair_flag = False
                    if agent._floor_idx - 1 >= 0:
                        del agent._floors[agent._floor_idx - 1]
                        agent._floor_idx -= 1
            elif not on:
                # :299-315 -- off the stairs having been on the centroid: arrived.
                self._reset_climb_state(om)
                self.climb_stair_over = True
                self.stats["climb_ok"] = self.stats.get("climb_ok", 0) + 1
                if direction == 1:
                    om._up_stair_end = robot_px[0].copy()
                    if not agent._floors[agent._floor_idx + 1]["obstacle"]._done_initializing:
                        self.handle_new_floor_initialization(agent, om, 1)
                    else:
                        agent._floor_idx += 1
                else:
                    om._down_stair_end = robot_px[0].copy()
                    if not agent._floors[agent._floor_idx - 1]["obstacle"]._done_initializing:
                        self.handle_new_floor_initialization(agent, om, 2)
                    else:
                        agent._floor_idx -= 1
                agent.stats["floor_switches"] = agent.stats.get("floor_switches", 0) + 1

    def handle_new_floor_initialization(self, agent, om, direction: int) -> None:
        """`_handle_new_floor_initialization` (:563-624)."""
        agent._done_initializing = False
        agent._initialize_step = 0
        if direction == 1:
            om._explored_up_stair = True
            agent._floors[agent._floor_idx + 1]["obstacle"]._explored_down_stair = True
            ori = om._up_stair_map.copy()
            fr = np.asarray(om._up_stair_frontiers)
            prev = om
            agent._floor_idx += 1
            new = agent.obstacle_map
            self._update_linked_stair_map(new, ori, fr, "_down_stair_map", "_down_stair_start",
                                          "_down_stair_end", "_down_stair_frontiers",
                                          prev, "_up_stair_start", "_up_stair_end", "_up_stair_frontiers")
            new._has_down_stair = True
        else:
            om._explored_down_stair = True
            agent._floors[agent._floor_idx - 1]["obstacle"]._explored_up_stair = True
            ori = om._down_stair_map.copy()
            fr = np.asarray(om._down_stair_frontiers)
            prev = om
            agent._floor_idx -= 1
            new = agent.obstacle_map
            self._update_linked_stair_map(new, ori, fr, "_up_stair_map", "_up_stair_start",
                                          "_up_stair_end", "_up_stair_frontiers",
                                          prev, "_down_stair_start", "_down_stair_end", "_down_stair_frontiers")
            new._has_up_stair = True

    @staticmethod
    def _update_linked_stair_map(new, original_stair_map, stair_frontiers, target_map_attr,
                                 start_attr, end_attr, frontier_attr,
                                 prev, prev_start_attr, prev_end_attr, prev_frontier_attr) -> None:
        """`_update_linked_stair_map` (:433-476): the flight just climbed becomes
        the arrival floor's staircase in the other direction, ends swapped."""
        n, labels, _stats, centroids = cv2.connectedComponentsWithStats(
            original_stair_map.astype(np.uint8), connectivity=8)
        sf = np.asarray(stair_frontiers).reshape(-1, 2)
        closest, best = -1, float("inf")
        for i in range(1, n):
            c = new._px_to_xy(np.atleast_2d(centroids[i]))
            d = (abs(sf[0][0] - c[0][0]) + abs(sf[0][1] - c[0][1])) if len(sf) else float("inf")
            if d < best:
                best, closest = d, i
        if closest != -1:
            filtered = original_stair_map.copy()
            filtered[labels != closest] = 0
            setattr(new, target_map_attr, filtered)
            setattr(new, start_attr, np.array(getattr(prev, prev_end_attr)).copy())
            setattr(new, end_attr, np.array(getattr(prev, prev_start_attr)).copy())
            setattr(new, frontier_attr, np.array(getattr(prev, prev_frontier_attr)).copy())

    def detect_passive_entry(self, agent, om, robot_px) -> None:
        """`_detect_passive_stair_entry` (:626-672)."""
        if om._has_up_stair and np.size(om._up_stair_frontiers) > 0:
            if self.on_stairs(om, robot_px, om._up_stair_map):
                self.passive_up_steps += 1
                self.passive_down_steps = 0
                if self.passive_up_steps >= PASSIVE_STAIR_DETECTION_THRESHOLD:
                    nxt = agent._floor_idx + 1
                    if nxt >= len(agent._floors) or not agent._floors[nxt]["obstacle"]._this_floor_explored:
                        self.trigger_stair_climbing(om, 1, robot_px)
                    self.passive_up_steps = 0
            else:
                self.passive_up_steps = 0
        if om._has_down_stair and np.size(om._down_stair_frontiers) > 0:
            if self.on_stairs(om, robot_px, om._down_stair_map):
                self.passive_down_steps += 1
                self.passive_up_steps = 0
                if self.passive_down_steps >= PASSIVE_STAIR_DETECTION_THRESHOLD:
                    prv = agent._floor_idx - 1
                    if prv < 0 or not agent._floors[prv]["obstacle"]._this_floor_explored:
                        self.trigger_stair_climbing(om, 2, robot_px)
                    self.passive_down_steps = 0
            else:
                self.passive_down_steps = 0

    def trigger_stair_climbing(self, om, direction: int, robot_px) -> None:
        """`_trigger_stair_climbing` (:674-690)."""
        self.climb_stair_over = False
        self.climb_stair_flag = direction
        self.reach_stair = True
        self.reach_stair_centroid = False
        self.get_close_to_stair_step = 0
        self.stats["climb_attempt"] = self.stats.get("climb_attempt", 0) + 1
        self.stats["passive_stair_entry"] = self.stats.get("passive_stair_entry", 0) + 1
        if direction == 1:
            om._up_stair_start = robot_px[0].copy()
            self.stair_frontier = np.asarray(om._up_stair_frontiers)
        else:
            om._down_stair_start = robot_px[0].copy()
            self.stair_frontier = np.asarray(om._down_stair_frontiers)

    def start_navigating(self, om, direction: int) -> None:
        """`_navigate_stair_if_unexplored_floor` :838-841."""
        self.climb_stair_over = False
        self.climb_stair_flag = direction
        self.stair_frontier = np.asarray(
            om._up_stair_frontiers if direction == 1 else om._down_stair_frontiers)
        self.stats["climb_attempt"] = self.stats.get("climb_attempt", 0) + 1
