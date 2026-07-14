"""2D occupancy costmap built from depth frames.

World plane axes are (x, z) with y as height (habitat convention, y-up).
Grid values: -1 unknown, 0 free, 100 occupied. The grid auto-grows.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core.geometry import backproject, bresenham
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
        new = np.full((h * 2, w * 2), UNKNOWN, dtype=np.int8)
        new[h // 2 : h // 2 + h, w // 2 : w // 2 + w] = self.grid
        self.grid = new
        self.origin = self.origin - np.array([h // 2, w // 2]) * self.resolution
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
        # Free space: raycast from camera to floor points
        for p in xy[floor_mask]:
            self._ray_free(cam_rc, self.world_to_grid(p), mark_end=FREE)
        # Obstacles: raycast free up to the obstacle cell, then mark occupied
        for p in xy[obst_mask]:
            self._ray_free(cam_rc, self.world_to_grid(p), mark_end=OCCUPIED)
        # The agent's own cell is free by construction
        if self.in_bounds(cam_rc):
            self.grid[cam_rc[0], cam_rc[1]] = FREE

    def _ray_free(self, rc0: np.ndarray, rc1: np.ndarray, mark_end: int) -> None:
        """Marks intermediate unknown/free cells FREE, endpoint mark_end.
        Occupied intermediate cells stop the ray (don't carve through walls)."""
        r1, c1 = int(rc1[0]), int(rc1[1])
        for r, c in bresenham(int(rc0[0]), int(rc0[1]), r1, c1):
            if not (0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]):
                return
            if (r, c) == (r1, c1):
                self.grid[r, c] = mark_end
                return
            if self.grid[r, c] != OCCUPIED:
                self.grid[r, c] = FREE
            else:
                return  # blocked

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


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return x * x + y * y <= r * r
