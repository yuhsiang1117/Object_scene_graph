"""2D occupancy costmap built from depth frames.

World plane axes are (x, z) with y as height (habitat convention, y-up).
Grid values: -1 unknown, 0 free, 100 occupied. The grid auto-grows.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from ..core.geometry import backproject
from ..core.types import FrameData

UNKNOWN, FREE, OCCUPIED = -1, 0, 100
PLANE = (0, 2)  # world axes forming the ground plane
HEIGHT_AXIS = 1


class Costmap2D:
    def __init__(self, resolution: float = 0.05, size_m: float = 20.0) -> None:
        self.resolution = resolution
        n = int(size_m / resolution)
        self.grid = np.full((n, n), UNKNOWN, dtype=np.int8)
        self.origin = np.array([-size_m / 2.0, -size_m / 2.0])  # world xy of grid[0, 0]
        self._grow_listeners: List[Callable[[int, int, int, int], None]] = []

    def add_grow_listener(self, fn: Callable[[int, int, int, int], None]) -> None:
        """Register a callback fired when the grid is reallocated.

        Called as ``fn(old_h, old_w, off_r, off_c)``: the listener must
        reallocate its own same-shaped array to ``(2*old_h, 2*old_w)`` and copy
        the old contents to ``[off_r:off_r+old_h, off_c:off_c+old_w]``.

        Anything holding a grid-aligned array alongside this one (a semantic
        value map, stair-hit counters, room labels) uses this rather than
        watching for shape changes, so the two share one origin, one
        resolution and one shape by construction -- a world point resolves to
        the same cell in both, always.
        """
        self._grow_listeners.append(fn)

    # ------------------------------------------------------------- transforms

    def world_to_grid(self, xy: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(xy) - self.origin) / self.resolution).astype(int)

    def grid_to_world(self, rc: np.ndarray) -> np.ndarray:
        return self.origin + (np.asarray(rc, dtype=float) + 0.5) * self.resolution

    def in_bounds(self, rc: np.ndarray) -> bool:
        return 0 <= rc[0] < self.grid.shape[0] and 0 <= rc[1] < self.grid.shape[1]

    def ensure_contains(self, xy: np.ndarray, margin_m: float = 2.0) -> None:
        rc = self.world_to_grid(xy)
        m = int(margin_m / self.resolution)
        h, w = self.grid.shape
        if 0 + m <= rc[0] < h - m and 0 + m <= rc[1] < w - m:
            return
        # Double the grid, keeping content centered
        off_r, off_c = h // 2, w // 2
        new = np.full((h * 2, w * 2), UNKNOWN, dtype=np.int8)
        new[off_r : off_r + h, off_c : off_c + w] = self.grid
        self.grid = new
        self.origin = self.origin - np.array([off_r, off_c]) * self.resolution
        for fn in self._grow_listeners:
            fn(h, w, off_r, off_c)
        self.ensure_contains(xy, margin_m)

    # ---------------------------------------------------------------- update

    def update(
        self,
        frame: FrameData,
        floor_y: float,
        obstacle_low: float = 0.2,
        obstacle_high: float = 1.5,
        max_range: float = 5.0,
        stride: int = 4,
    ) -> None:
        cam_xy = frame.camera_position[list(PLANE)]
        self.ensure_contains(cam_xy, margin_m=max_range + 1.0)

        pts = backproject(frame.depth, frame.intrinsics, frame.T_wc, stride=stride, max_depth=max_range)
        if pts.shape[0] == 0:
            return
        rel_h = pts[:, HEIGHT_AXIS] - floor_y
        xy = pts[:, list(PLANE)]

        floor_mask = (rel_h > -0.3) & (rel_h < obstacle_low)
        obst_mask = (rel_h >= obstacle_low) & (rel_h < obstacle_high)

        cam_rc = self.world_to_grid(cam_xy)
        self._raycast_batch(cam_rc, xy[floor_mask], xy[obst_mask])
        # The agent's own cell is free by construction
        if self.in_bounds(cam_rc):
            self.grid[cam_rc[0], cam_rc[1]] = FREE

    def _raycast_batch(
        self, cam_rc: np.ndarray, floor_xy: np.ndarray, obst_xy: np.ndarray
    ) -> None:
        """Vectorized raycasting: all rays sampled as a (M, N) grid of cells.
        Intermediate unknown/free cells become FREE; rays stop at the first
        already-occupied cell (never carve walls); obstacle endpoints are
        stamped OCCUPIED last. The per-point python Bresenham this replaces
        dominated the control loop (~1.6 s/frame)."""
        ends = []
        occ_flags = []
        for pts, occ in ((floor_xy, False), (obst_xy, True)):
            if pts.shape[0] == 0:
                continue
            rc = np.floor((pts - self.origin) / self.resolution).astype(np.int64)
            rc, idx = np.unique(rc, axis=0, return_index=True)
            ends.append(rc)
            occ_flags.append(np.full(rc.shape[0], occ))
        if not ends:
            return
        end_rc = np.concatenate(ends)  # (N, 2)
        end_occ = np.concatenate(occ_flags)  # (N,)
        h, w = self.grid.shape
        inb = (end_rc[:, 0] >= 0) & (end_rc[:, 0] < h) & (end_rc[:, 1] >= 0) & (end_rc[:, 1] < w)
        end_rc, end_occ = end_rc[inb], end_occ[inb]
        if end_rc.shape[0] == 0:
            return

        delta = end_rc - cam_rc[None, :]
        n_steps = int(np.abs(delta).max()) + 1
        # 2x supersampling closes diagonal gaps a true Bresenham would fill
        m = min(2 * n_steps + 1, 4096)
        t = np.linspace(0.0, 1.0, m)[:, None, None]
        samples = np.rint(cam_rc[None, None, :] + t * delta[None, :, :]).astype(np.int64)
        rr = np.clip(samples[..., 0], 0, h - 1)
        cc = np.clip(samples[..., 1], 0, w - 1)

        vals = self.grid[rr, cc]  # (M, N)
        blocked = vals == OCCUPIED
        any_blocked = blocked.any(axis=0)
        first_block = np.where(any_blocked, blocked.argmax(axis=0), m)  # (N,)
        step_idx = np.arange(m)[:, None]
        visible = step_idx < first_block[None, :]  # strictly before the wall

        self.grid[rr[visible], cc[visible]] = FREE
        # Obstacle endpoints whose ray was not blocked earlier
        reached = ~any_blocked | (first_block >= m - 1)
        stamp = end_occ & reached
        self.grid[end_rc[stamp, 0], end_rc[stamp, 1]] = OCCUPIED

    # ------------------------------------------------------------------ views

    def inflated(self, radius_m: float) -> np.ndarray:
        """Boolean obstacle map dilated by radius (for planning)."""
        from scipy import ndimage

        obst = self.grid == OCCUPIED
        r = max(1, int(round(radius_m / self.resolution)))
        struct = _disk(r)
        return ndimage.binary_dilation(obst, structure=struct)

    def free_mask(self) -> np.ndarray:
        return self.grid == FREE

    def unknown_mask(self) -> np.ndarray:
        return self.grid == UNKNOWN

    def coverage_cells(self) -> int:
        return int((self.grid != UNKNOWN).sum())


def grow_aligned(
    arr: np.ndarray, old_h: int, old_w: int, off_r: int, off_c: int, fill=0
) -> np.ndarray:
    """Reallocate a grid-aligned array to match a costmap that just grew.

    The counterpart to Costmap2D.add_grow_listener: pass the callback's four
    arguments straight through and store the result. Keeping the doubling
    arithmetic in one place is the point -- an off-by-one here would silently
    shift a whole map relative to the costmap it is supposed to overlay.
    """
    new = np.full((old_h * 2, old_w * 2), fill, dtype=arr.dtype)
    new[off_r : off_r + old_h, off_c : off_c + old_w] = arr
    return new


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return x * x + y * y <= r * r
