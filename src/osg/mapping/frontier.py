"""Frontier extraction: free cells adjacent to unknown, clustered into
connected components, deduplicated by centroid distance (paper: drop
frontiers closer than a threshold to avoid redundant targets).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import ndimage

from .costmap import FREE, UNKNOWN, Costmap2D


@dataclass
class Frontier:
    id: int
    centroid_xy: np.ndarray  # world coords (ground plane)
    cells: np.ndarray  # (N, 2) grid rows/cols
    size: int
    path_cost: Optional[float] = None
    score: Optional[float] = None


class FrontierExtractor:
    def __init__(self, min_cells: int = 8, dedup_m: float = 1.0) -> None:
        self.min_cells = min_cells
        self.dedup_m = dedup_m
        self._next_id = 0

    def extract(self, costmap: Costmap2D) -> List[Frontier]:
        grid = costmap.grid
        free = grid == FREE
        unknown = grid == UNKNOWN
        # Free cell with an unknown 4-neighbor
        neigh_unknown = np.zeros_like(unknown)
        neigh_unknown[1:, :] |= unknown[:-1, :]
        neigh_unknown[:-1, :] |= unknown[1:, :]
        neigh_unknown[:, 1:] |= unknown[:, :-1]
        neigh_unknown[:, :-1] |= unknown[:, 1:]
        frontier_cells = free & neigh_unknown

        labels, n = ndimage.label(frontier_cells, structure=np.ones((3, 3)))
        frontiers: List[Frontier] = []
        for lbl in range(1, n + 1):
            rc = np.argwhere(labels == lbl)
            if rc.shape[0] < self.min_cells:
                continue
            centroid_rc = rc.mean(axis=0)
            frontiers.append(
                Frontier(
                    id=self._next_id,
                    centroid_xy=costmap.grid_to_world(centroid_rc),
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
