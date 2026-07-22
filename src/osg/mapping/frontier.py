"""Frontier extraction, ported from ObjectSceneGraph_old (WFD, Wavefront
Frontier Detection). A frontier cell is an UNKNOWN cell that (a) is adjacent
to free space REACHABLE from the robot, and (b) does not touch an obstacle;
contiguous frontier cells form a cluster, kept if larger than a min size, its
centroid snapped to the nearest free cell and rejected if too close to an
obstacle. This is the vectorized equivalent of the old four-state double-BFS
(same frontier set, far cheaper than a per-cell Python BFS at ~100 calls/ep).

`extract` now takes the robot pose (WFD starts from it / restricts to the
robot's reachable region). The `Frontier` dataclass is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import ndimage

from .costmap import FREE, OCCUPIED, UNKNOWN, Costmap2D


@dataclass
class Frontier:
    id: int
    centroid_xy: np.ndarray  # world coords (ground plane)
    cells: np.ndarray  # (N, 2) grid rows/cols (the unknown frontier cells)
    size: int
    path_cost: Optional[float] = None
    score: Optional[float] = None


class FrontierExtractor:
    def __init__(
        self,
        min_cells: int = 5,             # old min_frontier_cluster_size (strict >)
        dedup_m: float = 1.0,           # kept for construction compat (light dedup)
        # old obstacle-proximity reject radius was 0.4m; that kills doorway
        # frontiers (the frame is inherently within 0.4m), so use a light 0.15m
        # -- the planner already filters truly unreachable targets.
        collision_distance_m: float = 0.15,
    ) -> None:
        self.min_cells = min_cells
        self.dedup_m = dedup_m
        self.collision_distance_m = collision_distance_m
        self._next_id = 0

    def extract(self, costmap: Costmap2D, robot_xy: Optional[np.ndarray] = None) -> List[Frontier]:
        grid = costmap.grid
        free = grid == FREE
        unknown = grid == UNKNOWN
        occ = grid == OCCUPIED
        if not free.any() or not unknown.any():
            return []

        # Robot's reachable free component (WFD explores outward from the robot).
        reach = free
        if robot_xy is not None:
            rc = costmap.world_to_grid(robot_xy)
            if costmap.in_bounds(rc):
                lbl, _ = ndimage.label(free, structure=np.ones((3, 3)))
                rlab = lbl[rc[0], rc[1]]
                if rlab == 0:  # robot cell not free -> snap to nearest free
                    _, (ir, ic) = ndimage.distance_transform_edt(~free, return_indices=True)
                    rlab = lbl[ir[rc[0], rc[1]], ic[rc[0], rc[1]]]
                if rlab > 0:
                    reach = lbl == rlab

        # unknown cell with a reachable-free 4-neighbor
        free_neigh = np.zeros_like(unknown)
        free_neigh[1:, :] |= reach[:-1, :]
        free_neigh[:-1, :] |= reach[1:, :]
        free_neigh[:, 1:] |= reach[:, :-1]
        free_neigh[:, :-1] |= reach[:, 1:]
        # not touching an obstacle (8-neighborhood dilation of occupied)
        occ_dil = ndimage.binary_dilation(occ, structure=np.ones((3, 3)))
        frontier_cells = unknown & free_neigh & ~occ_dil

        if not frontier_cells.any():
            return []

        # nearest-free index map for centroid snapping
        _, (fr, fc) = ndimage.distance_transform_edt(~free, return_indices=True)
        rad = max(1, int(self.collision_distance_m / costmap.resolution))

        labels, n = ndimage.label(frontier_cells, structure=np.ones((3, 3)))
        frontiers: List[Frontier] = []
        h, w = grid.shape
        for lbl in range(1, n + 1):
            rc = np.argwhere(labels == lbl)
            if rc.shape[0] <= self.min_cells:  # old: strict >
                continue
            cr, cc = rc.mean(axis=0)
            sr, sc = int(round(cr)), int(round(cc))
            # snap centroid to nearest free cell
            sr, sc = int(fr[sr, sc]), int(fc[sr, sc])
            # reject if an obstacle sits within the collision radius of the centroid
            r0, r1 = max(0, sr - rad), min(h, sr + rad + 1)
            c0, c1 = max(0, sc - rad), min(w, sc + rad + 1)
            if occ[r0:r1, c0:c1].any():
                continue
            frontiers.append(
                Frontier(
                    id=self._next_id,
                    centroid_xy=costmap.grid_to_world(np.array([sr, sc], dtype=float)),
                    cells=rc,
                    size=rc.shape[0],
                )
            )
            self._next_id += 1

        return self._dedup(frontiers)

    def _dedup(self, frontiers: List[Frontier]) -> List[Frontier]:
        frontiers = sorted(frontiers, key=lambda f: -f.size)
        kept: List[Frontier] = []
        for f in frontiers:
            if all(np.linalg.norm(f.centroid_xy - k.centroid_xy) >= self.dedup_m for k in kept):
                kept.append(f)
        return kept
