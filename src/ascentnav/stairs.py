"""ASCENT's stair machinery: detection input, climb state, floor stack.

The vendored `ObstacleMap` already does the hard part -- it accumulates
`_up_stair_map` / `_down_stair_map`, closes and filters them into components,
publishes `_up_stair_frontiers` as the largest component's centroid, and excludes
stair cells from the agent-radius dilation so a staircase never inflates into an
obstacle (`obstacle_map.py:574-598, 710-735`). None of that runs unless the map is
handed real stair masks, which is why the first `ascentnav` run scored 0.0% on
the 21 cross-floor episodes: it was passing zeros.

Two inputs are required, and ASCENT intersects them (`obstacle_map.py:520-524`):

    fusion_stair_mask = stair_mask & (seg_mask == STAIR_CLASS_ID)

`stair_mask` is a detector's "stair" mask -- GroundingDINO in ASCENT, YOLOE here,
which already carries `stairs` in its vocabulary. `seg_mask` is RedNet's MPCAT40
segmentation. The map then routes the fused pixels by the SIGN OF THE CAMERA
PITCH: level or up into the up-stair map, tilted down into the down-stair map
(`obstacle_map.py:534-542`). That is what the pitch is for -- not recall. S14a
and S32 both measured tilting as no help to stair DETECTION; its job is
disambiguating which way a staircase goes.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# ascent/constants.py:234-236, reused for the stair approach
STICK_DISTANCE_M = 0.3
GET_CLOSE_MAX_STEPS = 60
CLIMB_PAUSED_ABANDON = 30


def robot_on_stairs(
    stair_map: np.ndarray, robot_px: np.ndarray, radius_px: float
) -> bool:
    """Is any stair cell inside the agent's footprint?

    Port of `Map_Controller.is_robot_in_stair_map_fast`
    (map_controller.py:181-227), returning only the boolean the callers use.
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


class ClimbState:
    """The flags ASCENT keeps per environment for a floor transition.

    `Map_Controller` spreads these across a dozen parallel lists; one object per
    agent is the same state with the indexing removed.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.climbing = False          # _climb_stair_over inverted
        self.direction = 0             # _climb_stair_flag: 1 up, 2 down
        self.reached = False           # _reach_stair
        self.reached_centroid = False  # _reach_stair_centroid
        self.get_close_steps = 0       # _get_close_to_stair_step
        self.stick_steps = 0           # _frontier_stick_step
        self.last_dist = 0.0           # _last_frontier_distance
        self.carrot_xy: Optional[np.ndarray] = None
        self.disable_end = False       # _disable_end
        self.paused = 0                # _climb_stair_paused_step

    def start(self, direction: int) -> None:
        self.reset()
        self.climbing = True
        self.direction = direction

    def stuck_on_approach(self, dist: float) -> bool:
        """ASCENT retires a stair frontier it cannot close on
        (`ascent_policy.py:963-981`): 30 consecutive steps without the distance
        to it moving 0.3 m, or 60 such steps in total over the approach.

        Both counters advance ONLY on a step that failed to close the distance,
        which is why a long but progressing walk to a far staircase never times
        out -- the budget is on stalling, not on walking.
        """
        if abs(self.last_dist - dist) > STICK_DISTANCE_M:
            self.stick_steps = 0
            self.last_dist = dist
            return False
        self.stick_steps += 1
        self.get_close_steps += 1
        return self.stick_steps >= 30 or self.get_close_steps >= GET_CLOSE_MAX_STEPS


def carrot_waypoint(
    depth_normalised: np.ndarray,
    robot_xy: np.ndarray,
    heading: float,
    hfov: float,
    carrot_m: float = 0.8,
) -> Optional[np.ndarray]:
    """Steer at the farthest thing in view (`ascent_policy.py:1075-1112`).

    On a flight the treads and side walls are close and the far end is not, so
    the maximum-depth bearing points along the well. In ASCENT's episodic frame
    the heading is CCW-positive, so the offset is SUBTRACTED -- the opposite of
    the port in `osg/agent/nav_agent.py`, which works in OSG's CW-positive
    plane. Same rotation, and the two signs are not interchangeable.
    """
    if depth_normalised.size == 0:
        return None
    mx = float(np.max(depth_normalised))
    idx = np.argwhere(depth_normalised == mx)
    if idx.size == 0:
        return None
    u = float(np.mean(idx[:, 1]))
    cx = depth_normalised.shape[1] / 2.0
    offset = float(np.clip((u - cx) / cx, -1.0, 1.0)) * (hfov / 2.0)
    target = heading - offset
    return np.asarray(robot_xy, dtype=float) + carrot_m * np.array(
        [np.cos(target), np.sin(target)]
    )


def ratchet_carrot(
    current: Optional[np.ndarray],
    fresh: np.ndarray,
    stair_end_px: np.ndarray,
    robot_px: np.ndarray,
    to_px,
    pixels_per_meter: float,
    disable_end: bool,
) -> np.ndarray:
    """Keep whichever carrot is closer to the recorded stair end
    (`ascent_policy.py:1104-1121`), by L1 in map pixels as ASCENT does.

    Released when there is nothing to ratchet against, when the agent is
    already within half a metre of the end, or when the stall detector has
    given up on the end being reachable.
    """
    if current is None or stair_end_px is None or np.size(stair_end_px) == 0 or disable_end:
        return fresh
    if np.linalg.norm(np.asarray(stair_end_px) - robot_px[0]) <= 0.5 * pixels_per_meter:
        return fresh
    l1 = lambda p: abs(stair_end_px[0] - p[0][0]) + abs(stair_end_px[1] - p[0][1])
    return fresh if l1(to_px(fresh)) < l1(to_px(current)) else current
