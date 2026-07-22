"""Morphological room segmentation, ported from ObjectSceneGraph_old
(room_segmentation.py, default "Morphological" method):

  1. erode the free-space mask (3x3, `erode_iters` times) -> the ~12px erosion
     severs thin doorway connections, leaving one disconnected "core" per room
  2. label the disconnected cores (connected components)
  3. grow every core label outward in lock-step (nearest-core assignment over the
     free region) until the whole free space is relabelled -- recovers the eroded
     shell and fills the doorways, rooms meeting at ~equidistant borders

Class name / `segment()` signature / room-id stabilisation are unchanged so the
rest of the pipeline (scene_graph.rebuild, nav_agent) is unaffected.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage

from .costmap import Costmap2D


class VoronoiRoomSegmenter:
    def __init__(
        self,
        min_room_radius_m: float = 0.9,   # kept for construction compat (unused)
        door_width_m: float = 1.2,        # kept for construction compat (unused)
        min_room_cells: int = 60,
        erode_iters: int = 12,            # old room_segmentation.py erode_iteration
    ) -> None:
        self.min_room_radius_m = min_room_radius_m
        self.door_width_m = door_width_m
        self.min_room_cells = min_room_cells
        self.erode_iters = erode_iters
        self._prev_labels: Optional[np.ndarray] = None
        self._next_room_id = 1

    def segment(self, costmap: Costmap2D) -> np.ndarray:
        """Returns (H, W) int32 room-id map, 0 = no room."""
        import cv2

        free = costmap.free_mask()
        if free.sum() < self.min_room_cells:
            return np.zeros(costmap.grid.shape, dtype=np.int32)

        free_u8 = free.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(free_u8, kernel, iterations=self.erode_iters)
        n_cores, cores = cv2.connectedComponents(eroded)  # 0 = background

        if n_cores <= 1:
            # erosion removed every core (small / thin free space): one room
            labels = free.astype(np.int32)
        else:
            # multi-source wavefront: assign each free cell its nearest core's
            # label (Voronoi of the cores over the free region -> fills doorways)
            _, (ir, ic) = ndimage.distance_transform_edt(cores == 0, return_indices=True)
            grown = cores[ir, ic]
            labels = np.where(free, grown, 0).astype(np.int32)

        labels = self._drop_small(labels)
        labels = self._stabilize_ids(labels)
        self._prev_labels = labels
        return labels

    # ------------------------------------------------------------- internals

    def _drop_small(self, labels: np.ndarray) -> np.ndarray:
        out = labels.copy()
        for lbl in np.unique(labels):
            if lbl > 0 and (labels == lbl).sum() < self.min_room_cells:
                out[labels == lbl] = 0
        return out

    def _stabilize_ids(self, labels: np.ndarray) -> np.ndarray:
        """Remap new room labels to persistent room ids by max overlap."""
        out = np.zeros_like(labels)
        prev = self._prev_labels
        for lbl in np.unique(labels):
            if lbl == 0:
                continue
            mask = labels == lbl
            rid = 0
            if prev is not None and prev.shape == labels.shape:
                overlap = prev[mask]
                overlap = overlap[overlap > 0]
                if overlap.size > 0:
                    vals, counts = np.unique(overlap, return_counts=True)
                    best = vals[np.argmax(counts)]
                    if counts.max() > 0.3 * mask.sum():
                        rid = int(best)
            if rid == 0:
                rid = self._next_room_id
                self._next_room_id += 1
            out[mask] = rid
        self._next_room_id = max(self._next_room_id, int(out.max()) + 1)
        return out
