"""Stair detection: where the agent can leave the current floor.

Two independent signals, because the failure modes are different:

* **Down stairs — pure geometry.** Points that back-project below the floor the
  agent is standing on. No model, no vocabulary, and it works in the dark; a
  hole in the floor plane is unambiguous. (ASCENT inverts a normalised depth
  image to get this; our depth is metric, so the direct test is equivalent and
  simpler.)
* **Up stairs — detector mask AND geometry.** A rising staircase is *above* the
  floor plane and so is indistinguishable from a wall on height alone. YOLOE's
  open-vocabulary "stairs" class supplies the semantics; a geometric check on
  the masked points rejects its two dominant false positives -- a patterned
  flat floor (no rise) and a wall or shelf front (no depth extent).

Both accumulate hit counts across frames before being turned into components:
a single frame is far too noisy to commit a floor transition to.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import ndimage

from ..core.geometry import backproject
from ..core.types import Detection, FrameData
from .costmap import FREE, HEIGHT_AXIS, PLANE, Costmap2D, grow_aligned

STAIR_LABELS = {"stairs", "stair", "staircase", "steps"}


@dataclass
class StairGeometry:
    """Shape of a candidate staircase, in the camera's ground frame."""

    span_m: float  # horizontal extent along the view ray
    rise_m: float  # highest point above the floor
    slope: float  # metres of rise per metre of horizontal distance
    n_points: int

    def is_stair_like(
        self, min_span_m: float, min_rise_m: float, min_slope: float
    ) -> bool:
        return (
            self.n_points >= 20
            and self.span_m >= min_span_m
            and self.rise_m >= min_rise_m
            and self.slope >= min_slope
        )


def stair_geometry(pts_world: np.ndarray, cam_xy: np.ndarray, floor_y: float) -> StairGeometry:
    """Fit rise-vs-distance over a candidate's 3D points.

    A real staircase climbs as it recedes: 30-35 degrees is slope 0.58-0.70. A
    flat patterned floor has slope ~0 whatever its texture; a wall or shelf
    front has essentially no horizontal span, since every point sits at the
    same distance.
    """
    if pts_world.shape[0] < 2:
        return StairGeometry(0.0, 0.0, 0.0, int(pts_world.shape[0]))
    h = pts_world[:, HEIGHT_AXIS] - floor_y
    r = np.linalg.norm(pts_world[:, list(PLANE)] - np.asarray(cam_xy), axis=1)
    span = float(r.max() - r.min())
    slope = float(np.polyfit(r, h, 1)[0]) if span > 1e-3 else 0.0
    return StairGeometry(span, float(h.max()), slope, int(pts_world.shape[0]))


@dataclass
class StairDetection:
    kind: str  # "up" | "down"
    centroid_xy: np.ndarray
    cells: np.ndarray  # (N, 2) grid rows/cols
    n_cells: int
    support: int  # summed hit count, i.e. how many frames agreed


class StairDetector:
    def __init__(
        self,
        resolution_m: float = 0.05,
        down_drop_m: float = 0.35,
        up_rise_m: float = 0.35,
        up_slope_min: float = 0.30,
        up_span_min_m: float = 0.6,
        up_min_score: float = 0.35,
        min_hits: int = 3,
        min_cells: int = 25,
        close_iters: int = 2,
        max_range_m: float = 5.0,
        stride: int = 2,
        disable_margin_m: float = 0.5,
        up_mode: str = "detector",
    ) -> None:
        """
        Args:
            down_drop_m: a point this far below the standing floor is a hole in
                it. Sits above sensor noise and above the 0.15 m the costmap
                already tolerates as floor, and below one stair riser.
            up_rise_m / up_slope_min / up_span_min_m: the geometric gate on a
                detector "stairs" mask. slope 0.30 is a permissive floor under
                a real 30-35 degree staircase (0.58-0.70).
            min_hits / min_cells: a cell must be seen this often, and a
                component be this large, to count -- one frame is far too noisy
                to commit a floor transition to.
        """
        self.resolution_m = resolution_m
        self.down_drop_m = down_drop_m
        self.up_rise_m = up_rise_m
        self.up_slope_min = up_slope_min
        self.up_span_min_m = up_span_min_m
        self.up_min_score = up_min_score
        self.min_hits = min_hits
        self.min_cells = min_cells
        self.close_iters = close_iters
        self.max_range_m = max_range_m
        self.stride = stride
        # Margin around a retired staircase: the same stairwell re-observed
        # grows slightly, and without a margin the fringe forms a new
        # component right next to the one just given up on.
        self.disable_margin_m = disable_margin_m
        # Which up-stair signal to believe: detector | ascent | rednet.
        # See _up_masks for what each measured.
        self.up_mode = str(up_mode)

    # ------------------------------------------------------------ accumulate

    def accumulate(
        self,
        frame: FrameData,
        layer,  # FloorLayer
        dets: Optional[List[Detection]] = None,
        seg_stair_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Fold one frame's stair evidence into the floor's hit grids.

        `seg_stair_mask` is RedNet's MPCAT40 stair mask when the segmenter is
        enabled, already past its own >20-pixel gate (`perception/stair_seg.py`).
        How it combines with the detector is `up_mode` -- see `_up_masks`.
        """
        self._ensure_grids(layer)
        cam_xy = frame.camera_position[list(PLANE)]

        below = self._below_floor_points(frame, layer.floor_y)
        self._stamp(layer.down_stair_hits, layer.costmap, below)

        for mask, gated in self._up_masks(dets, seg_stair_mask):
            pts = backproject(
                frame.depth, frame.intrinsics, frame.T_wc,
                mask=mask, stride=self.stride, max_depth=self.max_range_m,
            )
            if gated:
                geom = stair_geometry(pts, cam_xy, layer.floor_y)
                if not geom.is_stair_like(
                    self.up_span_min_m, self.up_rise_m, self.up_slope_min
                ):
                    continue
            self._stamp(layer.up_stair_hits, layer.costmap, pts)

    def _up_masks(self, dets, seg_stair_mask):
        """The up-stair pixel masks to believe this frame, and whether each
        still has to pass the geometric gate.

        Three sources, because the measurement says the choice matters (S33, 250
        stair poses, control false-positive in brackets):

          `detector`  YOLOE's `stairs` class + geometric gate -- 10% [0%].
                      What every number before S33 was measured on.
          `ascent`    ASCENT's exact rule: the detector mask INTERSECTED with
                      RedNet's, no geometric gate (`obstacle_map.py:520-524`) --
                      10% [0%]. Faithful, and worth nothing here: an
                      intersection cannot beat its weaker input, and ASCENT's
                      second opinion is GroundingDINO, not YOLOE. All 26 YOLOE
                      firings already sit inside RedNet's 134, so the AND
                      discards 81% of what RedNet found.
          `rednet`    RedNet alone through OSG's own geometric gate -- the same
                      "two independent opinions" idea as ASCENT's fusion, with
                      the second opinion being geometry rather than a detector
                      this repo does not have.
        """
        mode = self.up_mode
        det_masks = [
            d.mask for d in (dets or [])
            if d.label.lower().strip() in STAIR_LABELS and d.score >= self.up_min_score
        ]

        if mode == "ascent":
            # No geometric gate: ASCENT projects the fused pixels straight into
            # the stair map, trusting the intersection of two detectors instead.
            if seg_stair_mask is None:
                return []
            out = []
            for m in det_masks:
                fused = m.astype(bool) & seg_stair_mask
                if fused.any():
                    out.append((fused, False))
            return out

        if mode == "rednet":
            if seg_stair_mask is None:
                return []
            return [(seg_stair_mask, True)]

        return [(m, True) for m in det_masks]

    def _below_floor_points(self, frame: FrameData, floor_y: float) -> np.ndarray:
        pts = backproject(
            frame.depth, frame.intrinsics, frame.T_wc,
            stride=self.stride, max_depth=self.max_range_m,
        )
        if pts.shape[0] == 0:
            return pts
        drop = floor_y - pts[:, HEIGHT_AXIS]
        # Upper bound as well as lower: a point 4 m down is seen through a
        # stairwell void or is a depth artefact, not the step in front of us.
        return pts[(drop > self.down_drop_m) & (drop < 4.0)]

    @staticmethod
    def _ensure_grids(layer) -> None:
        """Allocate the hit grids and keep them aligned when the costmap grows."""
        if layer.up_stair_hits is not None:
            return
        shape = layer.costmap.grid.shape
        layer.up_stair_hits = np.zeros(shape, dtype=np.int16)
        layer.down_stair_hits = np.zeros(shape, dtype=np.int16)
        layer.disabled_stair = np.zeros(shape, dtype=bool)

        def on_grow(h, w, off_r, off_c, _layer=layer):
            _layer.up_stair_hits = grow_aligned(_layer.up_stair_hits, h, w, off_r, off_c)
            _layer.down_stair_hits = grow_aligned(_layer.down_stair_hits, h, w, off_r, off_c)
            _layer.disabled_stair = grow_aligned(
                _layer.disabled_stair, h, w, off_r, off_c, fill=False
            )

        layer.costmap.add_grow_listener(on_grow)

    def disable(self, layer, cells: np.ndarray) -> None:
        """Retire a staircase after a failed traversal.

        Marks the component's cells plus a margin, and CLEARS the hit counts
        underneath, so the accumulated evidence cannot immediately re-form the
        same component. Retiring by cells rather than by a centroid matters
        because the hit grids keep accumulating: a component grows, its centroid
        drifts, and the next extraction looks like a brand-new staircase.
        """
        self._ensure_grids(layer)
        if cells.shape[0] == 0:
            return
        mask = np.zeros(layer.disabled_stair.shape, dtype=bool)
        mask[cells[:, 0], cells[:, 1]] = True
        rad = max(1, int(round(self.disable_margin_m / self.resolution_m)))
        mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3)), iterations=rad)
        layer.disabled_stair |= mask
        layer.up_stair_hits[mask] = 0
        layer.down_stair_hits[mask] = 0

    @staticmethod
    def _stamp(hits: np.ndarray, costmap: Costmap2D, pts: np.ndarray) -> None:
        """Add ONE hit per cell per call, so min_hits counts frames.

        Without the dedup it counts points, and a single dense close-range
        surface (hundreds of points per cell) clears any threshold on its own --
        which is exactly the single-frame noise the threshold exists to reject.
        It would also let a near surface outvote a far one purely by sampling
        density.
        """
        if pts.shape[0] == 0:
            return
        rc = costmap.world_to_grid(pts[:, list(PLANE)])
        h, w = hits.shape
        ok = (rc[:, 0] >= 0) & (rc[:, 0] < h) & (rc[:, 1] >= 0) & (rc[:, 1] < w)
        rc = np.unique(rc[ok], axis=0)
        if rc.shape[0]:
            hits[rc[:, 0], rc[:, 1]] += 1

    # --------------------------------------------------------------- extract

    def extract(self, layer) -> List[StairDetection]:
        """Turn accumulated hits into stair components on this floor."""
        out: List[StairDetection] = []
        for kind, hits in (("up", layer.up_stair_hits), ("down", layer.down_stair_hits)):
            if hits is None:
                continue
            out.extend(
                self._components(kind, hits, layer.costmap, layer.disabled_stair)
            )
        return out

    def _components(
        self,
        kind: str,
        hits: np.ndarray,
        costmap: Costmap2D,
        disabled: Optional[np.ndarray] = None,
    ) -> List[StairDetection]:
        import cv2

        mask = hits >= self.min_hits
        if disabled is not None:
            mask = mask & ~disabled
        mask = mask.astype(np.uint8)
        if not mask.any():
            return []
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=self.close_iters
        )
        labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
        if n == 0:
            return []

        # Snap centroids to the nearest standable cell, the same way frontier
        # centroids are snapped -- a staircase's own cells are not free space,
        # so the raw centroid is not a place the agent can be sent to.
        free = costmap.grid == FREE
        if not free.any():
            return []
        _, (fr, fc) = ndimage.distance_transform_edt(~free, return_indices=True)

        out = []
        for lbl in range(1, n + 1):
            rc = np.argwhere(labels == lbl)
            if rc.shape[0] < self.min_cells:
                continue
            cr, cc = rc.mean(axis=0)
            sr, sc = int(round(cr)), int(round(cc))
            sr, sc = int(fr[sr, sc]), int(fc[sr, sc])
            out.append(
                StairDetection(
                    kind=kind,
                    centroid_xy=costmap.grid_to_world(np.array([sr, sc], dtype=float)),
                    cells=rc,
                    n_cells=int(rc.shape[0]),
                    support=int(hits[rc[:, 0], rc[:, 1]].sum()),
                )
            )
        return out
