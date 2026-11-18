"""Top-down semantic value map: where the target is likely to be.

Each frame contributes one scalar (image-text similarity to "a <target> is
ahead") painted over the ground area that frame actually observed, weighted by
how confidently it observed it. Frontier selection then prefers directions that
look like the target's habitat, instead of ranking purely on distance and
unknown-area -- the mechanism VLFM and ASCENT use.

**Confidence is angular, not radial.** A surface seen down the optical axis is
observed well whether it is 1 m or 4 m away; one at the edge of the frame is
observed poorly at any range. Fusion keeps the value from the most confident
observation of each cell, so a glancing look never overwrites a head-on one.

Coordinates: the map shares a Costmap2D's origin, resolution and shape, and
registers a grow listener so it stays aligned through auto-grow. The observed
region is built directly in WORLD coordinates from the camera's own axes --
deliberately not by rasterising a local patch and rotating it into place, which
is where a sign error produces a mirrored map that looks perfectly plausible on
a heatmap while quietly steering the agent backwards.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..core.types import FrameData
from .costmap import PLANE, Costmap2D, grow_aligned


class ValueMap2D:
    def __init__(
        self,
        costmap: Costmap2D,
        max_depth_m: float = 5.0,
        min_confidence: float = 0.25,
        decision_threshold: float = 0.35,
        use_max_confidence: bool = True,
        n_range_samples: int = 64,
    ) -> None:
        self.costmap = costmap
        self.value = np.zeros(costmap.grid.shape, dtype=np.float32)
        self.conf = np.zeros(costmap.grid.shape, dtype=np.float32)
        costmap.add_grow_listener(self._on_grow)
        self.max_depth_m = max_depth_m
        self.min_confidence = min_confidence
        self.decision_threshold = decision_threshold
        self.use_max_confidence = use_max_confidence
        self.n_range_samples = n_range_samples
        self.n_updates = 0

    def _on_grow(self, h: int, w: int, off_r: int, off_c: int) -> None:
        self.value = grow_aligned(self.value, h, w, off_r, off_c, fill=0.0)
        self.conf = grow_aligned(self.conf, h, w, off_r, off_c, fill=0.0)

    def reset(self) -> None:
        self.value.fill(0.0)
        self.conf.fill(0.0)
        self.n_updates = 0

    # ----------------------------------------------------------------- update

    def update(self, frame: FrameData, value: float) -> None:
        """Paint `value` over the ground area this frame observed."""
        cells, conf = self._observed_cells(frame)
        if cells.shape[0] == 0:
            return
        self._fuse(cells, conf, float(value))
        self.n_updates += 1

    def _observed_cells(self, frame: FrameData) -> Tuple[np.ndarray, np.ndarray]:
        """Grid cells this frame saw, with a confidence for each.

        One ray per depth column: its free range is bounded by the farthest
        valid depth in that column (past the first surface we have no evidence),
        and its confidence by how far off the optical axis it is.
        """
        h, w = frame.depth.shape[:2]
        intr = frame.intrinsics
        u = np.arange(w, dtype=np.float32)
        theta = np.arctan((u - intr.cx) / intr.fx)  # per-column bearing

        d = frame.depth
        valid = d > 1e-3
        # Farthest valid return per column bounds free space along that bearing;
        # columns with no return at all see nothing, not everything.
        col_range = np.where(valid.any(axis=0), np.where(valid, d, 0.0).max(axis=0), 0.0)
        col_range = np.clip(col_range, 0.0, self.max_depth_m)

        # Ray directions in the ground plane, taken from the camera's own axes
        # so no rotation convention has to be re-derived here.
        R = frame.T_wc[:3, :3]
        fwd = R @ np.array([0.0, 0.0, 1.0])
        right = R @ np.array([1.0, 0.0, 0.0])
        f2 = fwd[list(PLANE)]
        r2 = right[list(PLANE)]
        nf, nr = np.linalg.norm(f2), np.linalg.norm(r2)
        if nf < 1e-6 or nr < 1e-6:  # camera pointing straight up/down
            return np.empty((0, 2), dtype=int), np.empty(0, dtype=np.float32)
        f2, r2 = f2 / nf, r2 / nr
        dirs = f2[None, :] + np.tan(theta)[:, None] * r2[None, :]
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        # Sample along each ray out to its free range.
        t = np.linspace(0.0, 1.0, self.n_range_samples, dtype=np.float32)
        rng = col_range[:, None] * t[None, :]  # (W, S)
        cam_xy = frame.camera_position[list(PLANE)]
        pts = cam_xy[None, None, :] + dirs[:, None, :] * rng[..., None]  # (W, S, 2)

        conf = self._confidence(theta, intr, w)
        conf = np.repeat(conf[:, None], self.n_range_samples, axis=1)

        keep = rng > 1e-3
        pts, conf = pts[keep], conf[keep]
        if pts.shape[0] == 0:
            return np.empty((0, 2), dtype=int), np.empty(0, dtype=np.float32)

        rc = self.costmap.world_to_grid(pts)
        gh, gw = self.value.shape
        inb = (rc[:, 0] >= 0) & (rc[:, 0] < gh) & (rc[:, 1] >= 0) & (rc[:, 1] < gw)
        return rc[inb], conf[inb]

    def _confidence(self, theta: np.ndarray, intr, width: int) -> np.ndarray:
        """cos^2 falloff from the optical axis to the frame edge, remapped into
        [min_confidence, 1]. Angular only: range does not reduce confidence,
        because a surface seen down the axis is seen well at any distance."""
        half_fov = float(np.arctan((width / 2.0) / intr.fx))
        a = np.clip(np.abs(theta) / max(half_fov, 1e-6), 0.0, 1.0) * (np.pi / 2.0)
        c = np.cos(a) ** 2
        return (self.min_confidence + (1.0 - self.min_confidence) * c).astype(np.float32)

    def _fuse(self, cells: np.ndarray, conf: np.ndarray, value: float) -> None:
        """Keep each cell's value from its most confident observation.

        Several rays land in one cell, so reduce within the frame first --
        otherwise whichever sample happened to be written last would win.
        """
        flat = cells[:, 0] * self.value.shape[1] + cells[:, 1]
        order = np.argsort(conf)  # ascending, so the highest lands last
        flat, conf = flat[order], conf[order]
        frame_conf = np.zeros(self.value.size, dtype=np.float32)
        frame_conf[flat] = conf

        idx = np.flatnonzero(frame_conf)
        if idx.size == 0:
            return
        new = frame_conf[idx]
        old = self.conf.reshape(-1)[idx]

        # A low-confidence look that is also worse than what we already have
        # says nothing; dropping it stops the fringe of every cone from
        # diluting a good observation.
        keep = ~((new < self.decision_threshold) & (new < old))
        idx, new, old = idx[keep], new[keep], old[keep]
        if idx.size == 0:
            return

        v = self.value.reshape(-1)
        c = self.conf.reshape(-1)
        if self.use_max_confidence:
            better = new > old
            v[idx[better]] = value
            c[idx[better]] = new[better]
        else:
            denom = old + new
            w_old = np.divide(old, denom, out=np.zeros_like(denom), where=denom > 0)
            w_new = np.divide(new, denom, out=np.zeros_like(denom), where=denom > 0)
            v[idx] = v[idx] * w_old + value * w_new
            c[idx] = old * w_old + new * w_new

    # ---------------------------------------------------------------- readout

    def value_at(self, xy: np.ndarray, radius_m: float = 0.5) -> float:
        """Median of the observed values within `radius_m` of a world point.

        Median, not mean: a frontier sits at the edge of the observed region, so
        its disc straddles unobserved cells and a few stray high values should
        not carry it. Returns 0.0 when nothing there has been observed.
        """
        rc = self.costmap.world_to_grid(np.asarray(xy, dtype=float))
        rad = max(1, int(round(radius_m / self.costmap.resolution)))
        h, w = self.value.shape
        r0, r1 = max(0, rc[0] - rad), min(h, rc[0] + rad + 1)
        c0, c1 = max(0, rc[1] - rad), min(w, rc[1] + rad + 1)
        if r0 >= r1 or c0 >= c1:
            return 0.0
        patch_v = self.value[r0:r1, c0:c1]
        patch_c = self.conf[r0:r1, c0:c1]
        seen = patch_c > 0
        return float(np.median(patch_v[seen])) if seen.any() else 0.0
