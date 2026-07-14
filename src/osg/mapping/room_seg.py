"""Voronoi-style room segmentation on the free-space map:
distance transform -> local-maxima seeds -> watershed -> merge fragments
whose shared boundary is wider than a doorway.

Room ids are kept stable across re-runs by matching new labels to previous
labels via maximum cell overlap.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import ndimage

from .costmap import Costmap2D


class VoronoiRoomSegmenter:
    def __init__(
        self,
        min_room_radius_m: float = 0.9,
        door_width_m: float = 1.2,
        min_room_cells: int = 60,
    ) -> None:
        self.min_room_radius_m = min_room_radius_m
        self.door_width_m = door_width_m
        self.min_room_cells = min_room_cells
        self._prev_labels: Optional[np.ndarray] = None
        self._next_room_id = 1

    def segment(self, costmap: Costmap2D) -> np.ndarray:
        """Returns (H, W) int32 room-id map, 0 = no room."""
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed

        free = costmap.free_mask()
        if free.sum() < self.min_room_cells:
            return np.zeros(costmap.grid.shape, dtype=np.int32)

        dist = ndimage.distance_transform_edt(free) * costmap.resolution
        min_dist_px = max(3, int(self.min_room_radius_m / costmap.resolution))
        peaks = peak_local_max(
            dist, min_distance=min_dist_px, threshold_abs=self.min_room_radius_m, labels=free
        )
        if peaks.shape[0] == 0:
            labels = free.astype(np.int32)
        else:
            markers = np.zeros(dist.shape, dtype=np.int32)
            for i, (r, c) in enumerate(peaks, start=1):
                markers[r, c] = i
            labels = watershed(-dist, markers, mask=free).astype(np.int32)
            labels = self._merge_open_boundaries(labels, dist, costmap.resolution)
            labels = self._drop_small(labels)

        labels = self._stabilize_ids(labels)
        self._prev_labels = labels
        return labels

    # ------------------------------------------------------------- internals

    def _merge_open_boundaries(
        self, labels: np.ndarray, dist: np.ndarray, resolution: float
    ) -> np.ndarray:
        """If two rooms share a boundary whose clearance exceeds a doorway
        width, they are one room (watershed over-segmented an open space)."""
        h, w = labels.shape
        pairs: Dict[tuple, float] = {}
        for dr, dc in ((0, 1), (1, 0)):
            a = labels[: h - dr, : w - dc]
            b = labels[dr:, dc:]
            m = (a > 0) & (b > 0) & (a != b)
            if not m.any():
                continue
            d = np.minimum(dist[: h - dr, : w - dc][m], dist[dr:, dc:][m])
            la, lb = a[m], b[m]
            for va, vb, vd in zip(la, lb, d):
                key = (min(va, vb), max(va, vb))
                pairs[key] = max(pairs.get(key, 0.0), float(vd))

        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for (a, b), clearance in pairs.items():
            # Boundary clearance > half door width means no wall pinch there.
            if clearance > self.door_width_m / 2.0:
                parent[find(a)] = find(b)

        out = labels.copy()
        for lbl in np.unique(labels):
            if lbl > 0:
                out[labels == lbl] = find(int(lbl))
        return out

    def _drop_small(self, labels: np.ndarray) -> np.ndarray:
        out = labels.copy()
        for lbl in np.unique(labels):
            if lbl > 0 and (labels == lbl).sum() < self.min_room_cells:
                out[labels == lbl] = 0
        return out

    def _stabilize_ids(self, labels: np.ndarray) -> np.ndarray:
        """Remap new watershed labels to persistent room ids by max overlap."""
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
