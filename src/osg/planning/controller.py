"""Waypoint pursuit with discrete habitat actions + stuck detection."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..mapping.costmap import OCCUPIED, PLANE, Costmap2D

FORWARD, TURN_LEFT, TURN_RIGHT = "move_forward", "turn_left", "turn_right"


def agent_heading(T_wc: np.ndarray) -> float:
    """Heading angle in the ground plane. OpenCV camera forward is +z."""
    fwd = T_wc[:3, 2]
    return float(np.arctan2(fwd[PLANE[1]], fwd[PLANE[0]]))


class WaypointController:
    def __init__(
        self,
        lookahead_m: float = 0.5,
        heading_tol_deg: float = 15.0,
        forward_m: float = 0.25,
        stuck_after: int = 3,
    ) -> None:
        self.lookahead = lookahead_m
        self.heading_tol = np.radians(heading_tol_deg)
        self.forward_m = forward_m
        self.stuck_after = stuck_after
        self._last_pos: Optional[np.ndarray] = None
        self._no_progress = 0
        self.stuck = False

    def reset(self) -> None:
        self._last_pos = None
        self._no_progress = 0
        self.stuck = False

    def act(self, T_wc: np.ndarray, path: np.ndarray) -> Optional[str]:
        """Next discrete action toward the path, None when path is consumed."""
        pos = T_wc[:3, 3][list(PLANE)]
        if path.shape[0] == 0:
            return None
        # Find the lookahead waypoint: first path point beyond lookahead dist
        dists = np.linalg.norm(path - pos, axis=1)
        beyond = np.nonzero(dists > self.lookahead)[0]
        if beyond.size == 0:
            if dists[-1] < 0.2:
                return None  # arrived
            wp = path[-1]
        else:
            # Skip waypoints behind the closest point to avoid backtracking
            closest = int(np.argmin(dists))
            idx = beyond[beyond >= closest]
            wp = path[idx[0]] if idx.size else path[-1]

        to_wp = wp - pos
        desired = np.arctan2(to_wp[1], to_wp[0])
        err = _wrap(desired - agent_heading(T_wc))
        if abs(err) > self.heading_tol:
            # habitat turn_left = +rotation about world +y, which *decreases*
            # atan2(f_z, f_x): positive heading error therefore needs turn_right.
            return TURN_RIGHT if err > 0 else TURN_LEFT
        return FORWARD

    def observe_progress(self, T_wc: np.ndarray, last_action: Optional[str], costmap: Costmap2D) -> None:
        """Call after each executed action; flags stuck and marks the blocked
        cell so the next replan routes around it."""
        pos = T_wc[:3, 3][list(PLANE)]
        if last_action == FORWARD and self._last_pos is not None:
            moved = np.linalg.norm(pos - self._last_pos)
            if moved < 0.5 * self.forward_m:
                self._no_progress += 1
            else:
                self._no_progress = 0
            if self._no_progress >= self.stuck_after:
                heading = agent_heading(T_wc)
                block = pos + np.array([np.cos(heading), np.sin(heading)]) * self.forward_m
                rc = costmap.world_to_grid(block)
                # Mark a small disk, not one cell: an 8-connected planner
                # slips diagonally past a single blocked cell and drives the
                # agent into the same invisible obstacle forever.
                h, w = costmap.grid.shape
                for dr in (-2, -1, 0, 1, 2):
                    for dc in (-2, -1, 0, 1, 2):
                        r, c = rc[0] + dr, rc[1] + dc
                        if dr * dr + dc * dc <= 4 and 0 <= r < h and 0 <= c < w:
                            costmap.grid[r, c] = OCCUPIED
                self.stuck = True
                self._no_progress = 0
        self._last_pos = pos.copy()


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)
