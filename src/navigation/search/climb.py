"""Where to step on a staircase (`agent.search.climb_carrot_free`, `climb_mount_m`).

ASCENT's carrot on the flight (`stairs.carrot_waypoint`, `ascent_policy.py:
1128-1140`) is 0.8 m toward the FARTHEST pixel in the depth image. On the
S83 baseline that pixel was a median 4.1 m away and, half the time, beyond
the 5 m depth clip: a doorway or the stairwell seen over the banister. In the
steps of climbs that stalled, a straight 0.8 m toward the carrot was walkable
29% of the time against 74% in climbs that went through, although some
walkable direction was in view 92% of the time (Habitat's navmesh, used only
to score). The point-goal policy then turns away from the obstacle and the
carrot, redrawn in the new view, comes with it.

`free_carrot_offset` keeps the reference's choice -- the farthest thing in
view -- but only among directions whose first `carrot_m` the depth image
shows clear of anything the body would hit. Sensor-only: the depth frame and
the camera pose the agent already has.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def obstacle_points(depth_m: np.ndarray, T_wc: np.ndarray, fx: float, fy: float,
                    base_y: float, reach_m: float, stride: int = 4,
                    max_depth: float = 5.0, step_rise_m: float = 0.3,
                    slope: float = 0.8, top_m: float = 1.7) -> np.ndarray:
    """(N, 2) camera-relative ground-plane offsets (x right of the world, z)
    of the depth points within `reach_m` that the body would hit.

    A point counts when it stands above the floor a flight could carry the
    agent to: higher than `step_rise_m + slope * range` above the feet
    (a 38-degree flight rises 0.78 m per metre, so its treads stay under the
    line and a banister, a wall or a newel post rises above it), and lower
    than `top_m` (the ceiling over a stairwell is not in the way).
    """
    h, w = depth_m.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    d = depth_m[::stride, ::stride].astype(np.float64)
    ok = (d > 1e-3) & (d < max_depth)
    if not np.any(ok):
        return np.zeros((0, 2))
    u, v, d = us[ok], vs[ok], d[ok]
    cx, cy = w / 2.0, h / 2.0
    pc = np.stack([(u - cx) / fx * d, (v - cy) / fy * d, d], axis=1)
    pw = pc @ T_wc[:3, :3].T + T_wc[:3, 3]
    rel = pw - T_wc[:3, 3]
    rng = np.hypot(rel[:, 0], rel[:, 2])
    hgt = pw[:, 1] - base_y
    hit = (rng < reach_m) & (hgt > step_rise_m + slope * rng) & (hgt < top_m)
    return rel[hit][:, [0, 2]]


def free_carrot_offset(depth_m: np.ndarray, T_wc: np.ndarray, fx: float, fy: float,
                       base_y: float, hfov: float, carrot_m: float = 0.8,
                       radius_m: float = 0.18, max_depth: float = 5.0,
                       n_dirs: int = 41) -> Optional[float]:
    """The reference's bearing offset (`carrot_waypoint`: (u - cx) / cx *
    hfov / 2, positive to the right) of the farthest column whose corridor
    -- `radius_m` either side, `carrot_m` long -- holds no obstacle point.
    None when every direction in view is blocked (the caller falls back to
    the reference carrot)."""
    h, w = depth_m.shape
    obs = obstacle_points(depth_m, T_wc, fx, fy, base_y, carrot_m + radius_m,
                          max_depth=max_depth)
    fwd = np.asarray(T_wc[:3, 2], dtype=float)
    heading = np.arctan2(fwd[2], fwd[0])
    right = np.asarray(T_wc[:3, 0], dtype=float)
    # image-right is +offset; in the ground plane it is the camera x axis
    sign = 1.0 if np.cross([fwd[0], fwd[2]], [right[0], right[2]]) > 0 else -1.0
    clipped = np.clip(depth_m, 0.0, max_depth)
    best, best_far = [], -1.0
    for k in range(n_dirs):
        u = (k + 0.5) * w / n_dirs
        off = float(np.clip((u - w / 2.0) / (w / 2.0), -1.0, 1.0)) * (hfov / 2.0)
        ang = heading + sign * off
        if len(obs):
            along = obs[:, 0] * np.cos(ang) + obs[:, 1] * np.sin(ang)
            across = -obs[:, 0] * np.sin(ang) + obs[:, 1] * np.cos(ang)
            if np.any((along > 0.0) & (along < carrot_m + radius_m) & (np.abs(across) < radius_m)):
                continue
        col = clipped[:, int(min(w - 1, u))]
        far = float(col.max())
        if far > best_far + 1e-6:
            best, best_far = [off], far
        elif abs(far - best_far) <= 1e-6:
            best.append(off)
    if not best:
        return None
    return float(np.mean(best))


class HeadingMemory:
    """`agent.search.climb_heading_memory`: step along the flight by heading.

    The agent turns in fixed steps (30 degrees on HM3D), so from any spot it
    can face one of `n` headings. Each step: if the last action was a forward,
    judge it by what it did to the agent's height --
      refused (did not move): close that heading at that spot, choose again;
      progress (height moved the way the attempt goes): keep the heading;
      wrong way (height moved against it): close the heading at both the spot
        it left and the spot it reached, and turn round to the opposite one;
      level (moved, height unchanged, e.g. a landing): choose again.
    Choosing again takes the open heading nearest the carrot's bearing. Facing
    the goal heading, step; else turn toward it the short way. The goal is
    held while turning, so the carrot (redrawn in every new view) cannot drag
    the agent back and forth, which is what the reference did in the stalled
    climbs.

    S87 round 1 closed a wrong-way heading only at the spot it left and chose
    again after every forward: on the dev probe the agent walked a flight
    back the way it came, one 0.1 m spot at a time, and dropped headings that
    were working for the carrot's bearing into a wall.
    """

    def __init__(self, turn_deg: float = 30.0, cell_m: float = 0.1,
                 moved_m: float = 0.05, wrong_way_m: float = 0.08,
                 progress_m: float = 0.05) -> None:
        self.turn = float(np.radians(turn_deg))
        self.n = int(round(2 * np.pi / self.turn))
        self.cell_m = float(cell_m)
        self.moved_m = float(moved_m)
        self.wrong_way_m = float(wrong_way_m)
        self.progress_m = float(progress_m)
        self.reset()

    def reset(self) -> None:
        self.closed: dict = {}            # cell -> headings refused or wrong-way there
        self.goal: Optional[int] = None   # the heading being turned to
        self.pending = None               # (cell, heading, xy, y) of the last forward
        self.refused = 0
        self.wrong_way = 0
        self.kept = 0

    def _cell(self, xy) -> tuple:
        return tuple(np.floor(np.asarray(xy, dtype=float) / self.cell_m).astype(int))

    def _bin(self, ang: float) -> int:
        return int(round(float(ang) / self.turn)) % self.n

    def step(self, robot_xy, heading: float, base_y: float, direction: int,
             bearing: float) -> str:
        if self.pending is not None:
            cell0, b0, xy0, y0 = self.pending
            self.pending = None
            self.goal = None
            dy = float(base_y) - y0
            sign = 1.0 if direction == 1 else -1.0 if direction == 2 else 0.0
            if np.linalg.norm(np.asarray(robot_xy, dtype=float) - xy0) < self.moved_m:
                self.closed.setdefault(cell0, set()).add(b0)
                self.refused += 1
            elif sign * dy < -self.wrong_way_m:
                self.closed.setdefault(cell0, set()).add(b0)
                self.closed.setdefault(self._cell(robot_xy), set()).add(b0)
                self.goal = (b0 + self.n // 2) % self.n
                self.wrong_way += 1
            elif sign * dy > self.progress_m:
                self.goal = b0
                self.kept += 1
        cell = self._cell(robot_xy)
        closed = self.closed.get(cell, set())
        if len(closed) >= self.n:                  # all refused here: start this spot afresh
            self.closed[cell] = closed = set()
        cur = self._bin(heading)
        if self.goal is None or self.goal in closed:
            want = float(bearing)
            best, best_d = None, None
            for b in range(self.n):
                if b in closed:
                    continue
                d = abs((b * self.turn - want + np.pi) % (2 * np.pi) - np.pi)
                if best_d is None or d < best_d - 1e-9:
                    best, best_d = b, d
            self.goal = best
        if self.goal == cur:
            self.pending = (cell, cur, np.asarray(robot_xy, dtype=float).copy(), float(base_y))
            return "move_forward"
        diff = (self.goal - cur + self.n // 2) % self.n - self.n // 2
        return "turn_left" if diff > 0 else "turn_right"
