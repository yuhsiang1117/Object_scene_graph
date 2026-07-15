"""Waypoint pursuit with discrete habitat actions + stuck detection."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..mapping.costmap import OCCUPIED, PLANE, UNKNOWN, Costmap2D

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
        stuck_mark_radius_m: float = 0.35,
        stuck_mark_expiry_steps: int = 60,
    ) -> None:
        self.lookahead = lookahead_m
        self.heading_tol = np.radians(heading_tol_deg)
        self.forward_m = forward_m
        self.stuck_after = stuck_after
        self.stuck_mark_radius_m = stuck_mark_radius_m
        self.stuck_mark_expiry_steps = stuck_mark_expiry_steps
        self._last_pos: Optional[np.ndarray] = None
        self._no_progress = 0
        self.stuck = False
        # (row, col, expire_at_step) for every forced-OCCUPIED cell so it can
        # be released later — see observe_progress for why this must decay.
        self._stuck_cells: List[tuple] = []

    def reset(self) -> None:
        self._last_pos = None
        self._no_progress = 0
        self.stuck = False
        self._stuck_cells = []

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

    def observe_progress(
        self, T_wc: np.ndarray, last_action: Optional[str], costmap: Costmap2D, step_count: int = 0
    ) -> None:
        """Call after each executed action; flags stuck and marks the blocked
        area so the next replan routes around it.

        The mark MUST expire. A forced-OCCUPIED cell blocks the raycast in
        costmap.update() from ever reaching past it, so once marked it can
        never be corrected by a fresh depth observation on its own — it is
        self-reinforcing. Combined with the wide halo below (needed so A*
        can't slip through a one-cell sliver next to a real doorway-width
        obstacle), an un-decaying mark can permanently seal off a real
        doorway after just one or two stuck events at slightly different
        angles: an agent was observed circling a ~3 m room for the entire
        500-step episode, never leaving, after two early stuck events near
        its only exit (giveup_log showed both at step <120; final position
        never moved beyond that room for the remaining 380 steps).
        """
        self._decay_stuck_marks(costmap, step_count)
        pos = T_wc[:3, 3][list(PLANE)]
        if last_action == FORWARD and self._last_pos is not None:
            moved = np.linalg.norm(pos - self._last_pos)
            if moved < 0.5 * self.forward_m:
                self._no_progress += 1
            else:
                self._no_progress = 0
            if self._no_progress >= self.stuck_after:
                heading = agent_heading(T_wc)
                direction = np.array([np.cos(heading), np.sin(heading)])
                # A single small disk 0.1 m ahead left enough of a doorway-
                # width bottleneck unblocked that A* kept re-routing three
                # different frontier pursuits through the exact same spot
                # (observed: agent frozen at one position for 45 steps across
                # 3 give-up cycles). Mark a wider halo — several offsets
                # along the heading, each a bigger disk — to actually force
                # a detour instead of leaving a one-cell sliver bypass.
                expire_at = step_count + self.stuck_mark_expiry_steps
                for dist in (0.5 * self.forward_m, self.forward_m, 1.8 * self.forward_m):
                    self._mark_disk(costmap, pos + direction * dist, self.stuck_mark_radius_m, expire_at)
                self.stuck = True
                self._no_progress = 0
        self._last_pos = pos.copy()

    def _mark_disk(
        self, costmap: Costmap2D, center_xy: np.ndarray, radius_m: float, expire_at_step: int
    ) -> None:
        rc = costmap.world_to_grid(center_xy)
        r_cells = max(1, int(round(radius_m / costmap.resolution)))
        h, w = costmap.grid.shape
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                if dr * dr + dc * dc <= r_cells * r_cells:
                    r, c = rc[0] + dr, rc[1] + dc
                    if 0 <= r < h and 0 <= c < w:
                        costmap.grid[r, c] = OCCUPIED
                        self._stuck_cells.append((r, c, expire_at_step))

    def _decay_stuck_marks(self, costmap: Costmap2D, step_count: int) -> None:
        """Release expired forced-OCCUPIED cells back to UNKNOWN (not FREE:
        we have no fresh evidence either way — A* already treats UNKNOWN as
        traversable at a penalty, which is exactly the "try again, but not
        for free" behavior we want)."""
        if not self._stuck_cells:
            return
        still_active = []
        h, w = costmap.grid.shape
        for r, c, expire_at in self._stuck_cells:
            if expire_at <= step_count:
                if 0 <= r < h and 0 <= c < w and costmap.grid[r, c] == OCCUPIED:
                    costmap.grid[r, c] = UNKNOWN
            else:
                still_active.append((r, c, expire_at))
        self._stuck_cells = still_active


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)
