"""Improvement C, part 1: approach-viewpoint planning for the last-mile
problem — pick an unobstructed, well-placed pose from which to verify (and
declare) the candidate target.
"""
from __future__ import annotations

from typing import List, Optional  # noqa: F401

import numpy as np
from scipy import ndimage

from ..core.geometry import bresenham
from ..mapping.costmap import FREE, OCCUPIED, Costmap2D


class ViewpointPlanner:
    def __init__(self, ring_radii_m: Optional[List[float]] = None, n_samples: int = 16) -> None:
        self.ring_radii = ring_radii_m or [0.8, 1.2, 1.5, 2.0]
        self.n_samples = n_samples

    def approach_viewpoint(
        self,
        obj_xy: np.ndarray,
        costmap: Costmap2D,
        exclude: Optional[List[np.ndarray]] = None,
        exclude_radius_m: float = 0.5,
    ) -> Optional[np.ndarray]:
        """Best world-xy pose to observe the object from, or None if the
        object is not yet observable from mapped free space. `exclude` lists
        previously tried viewpoints (e.g., where the detector could not see
        the object due to 3D occlusion the 2D map misses)."""
        exclude = exclude or []
        clearance = ndimage.distance_transform_edt(costmap.grid != OCCUPIED) * costmap.resolution
        best, best_score = None, -1.0
        for radius in self.ring_radii:
            for k in range(self.n_samples):
                ang = 2.0 * np.pi * k / self.n_samples
                cand = obj_xy + radius * np.array([np.cos(ang), np.sin(ang)])
                if any(np.linalg.norm(cand - e) < exclude_radius_m for e in exclude):
                    continue
                rc = costmap.world_to_grid(cand)
                if not costmap.in_bounds(rc) or costmap.grid[rc[0], rc[1]] != FREE:
                    continue
                if not self._line_of_sight(costmap, cand, obj_xy):
                    continue
                # Prefer clearance and a mid-range viewing distance (~1.2 m)
                score = float(clearance[rc[0], rc[1]]) - 0.3 * abs(radius - 1.2)
                if score > best_score:
                    best, best_score = cand, score
            if best is not None:
                return best  # nearest ring with a valid view wins
        return None

    @staticmethod
    def _line_of_sight(costmap: Costmap2D, from_xy: np.ndarray, to_xy: np.ndarray) -> bool:
        """The ray must not cross occupied cells (cells adjacent to the
        object itself are excluded — the object is an obstacle)."""
        rc0 = costmap.world_to_grid(from_xy)
        rc1 = costmap.world_to_grid(to_xy)
        r1, c1 = int(rc1[0]), int(rc1[1])
        skip_near = max(2, int(0.3 / costmap.resolution))  # cells near the object
        for r, c in bresenham(int(rc0[0]), int(rc0[1]), r1, c1):
            if np.hypot(r1 - r, c1 - c) <= skip_near:
                continue
            if costmap.in_bounds(np.array([r, c])) and costmap.grid[r, c] == OCCUPIED:
                return False
        return True
