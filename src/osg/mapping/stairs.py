"""Stair detection: where can the agent change storey?

Two signals, and the ordering between them is set by measurement rather than
preference. On the 100-episode v1 run the YOLOE `stairs` class fired in only
**15% of multi-floor episodes** (12 tracks / 100 episodes) -- high quality when
it fired (median score 0.60, median 26 observations) but far too sparse to
gate a floor transition on. So:

* **Geometry leads.** ZONDA's criterion: on a coarse grid, a cell is steppable
  when the largest height difference to its 8 neighbours is below the agent's
  climb limit. Flat floor gives ~0, a wall gives metres, a staircase sits in
  between at roughly its riser height. No model, no weights, works on any
  observed surface.
* **Semantics confirm.** A `stairs` track in the object layer raises confidence
  for cells near it, but is never required.

The output is deliberately conservative. Confirmed cells get written FREE in
the costmap and exempted from the obstacle stamp, and
`docs/history/INVESTIGATION.md` records that *every* previous attempt to soften the
costmap's obstacle writes (hit-count corroboration, speckle filtering) cost
more real geometry than it gained. So a cell must clear the geometric test AND
belong to a component large enough to be an actual staircase before it is
touched, and the total relabelled area is capped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.labels import normalize_label
from .costmap import FREE, HEIGHT_AXIS, PLANE, Costmap2D, block_min, grow_aligned
from ..core.geometry import backproject
from ..core.types import Detection, FrameData

STAIR_LABELS = ("stairs", "staircase", "stair")


@dataclass
class StairRegion:
    """A connected run of steppable-but-not-flat cells."""

    cells_rc: np.ndarray          # (N, 2) indices into the FINE costmap grid
    centroid_xy: np.ndarray       # world (x, z)
    n_cells: int
    mean_dh: float
    low_y: float
    high_y: float
    semantic: bool = False        # a `stairs` track corroborates it

    @property
    def rise_m(self) -> float:
        return float(self.high_y - self.low_y)


def stair_tracks(object_layer, min_obs: int = 2, min_evidence: float = 1.0) -> List[np.ndarray]:
    """3D centres of `stairs` tracks that clear the existing corroboration bar.

    Reuses the object layer's own multi-frame accumulation (`evidence`, already
    tuned over 1343 tracks) rather than inventing a second one.
    """
    out = []
    for t in object_layer.tracks(include_blacklisted=True):
        if normalize_label(t.label) not in STAIR_LABELS:
            continue
        if t.n_obs >= min_obs and float(getattr(t, "evidence", 0.0)) >= min_evidence:
            out.append(np.asarray(object_layer.center_of(t), dtype=float))
    return out


def detect_stairs(
    costmap: Costmap2D,
    climb_limit_m: float = 0.2,
    min_dh_m: float = 0.03,
    cell_m: float = 0.1,
    min_cells: int = 12,
    min_rise_m: float = 1.0,
    semantic_centers: Optional[Sequence[np.ndarray]] = None,
    semantic_radius_m: float = 1.5,
    require_semantic: bool = False,
) -> List[StairRegion]:
    """Steppable, non-flat regions of the costmap.

    `min_dh_m` excludes flat floor (which would otherwise make the whole map a
    "staircase"); `climb_limit_m` excludes walls and should match Habitat's
    navmesh `max_climb` so the costmap and the navmesh agree on what the
    embodiment can step over.

    `min_rise_m` is the one that does the real work, and 1.0 is measured rather
    than guessed. The per-cell test alone is satisfied by thresholds, ramps,
    sloped floor and low furniture: on the 35-episode SINGLE-floor gate it fired
    in 28/35 episodes, relabelling 43k cells in scenes with no storey to reach.
    Every one of those false regions rose 0.33-0.75 m, while a staircase that
    actually connects HM3D storeys must climb 2.5-3.4 m. Demanding a metre of
    rise separates them cleanly. The cost is that a staircase glimpsed only in
    part stays unconfirmed until enough of its run is observed -- the safe
    direction, since the relabel is permanent.
    """
    from scipy import ndimage

    dh, valid = costmap.height_gradient(cell_m=cell_m)
    steppable = valid & np.isfinite(dh) & (dh > min_dh_m) & (dh < climb_limit_m)
    if not steppable.any():
        return []

    block = max(1, int(round(cell_m / costmap.resolution)))
    labels, n = ndimage.label(steppable, structure=np.ones((3, 3)))
    coarse_h = block_min(costmap.height, block)

    regions: List[StairRegion] = []
    for lbl in range(1, n + 1):
        coarse_rc = np.argwhere(labels == lbl)
        if coarse_rc.shape[0] * (block ** 2) < min_cells:
            continue
        ys = coarse_h[labels == lbl]
        ys = ys[np.isfinite(ys)]
        if ys.size == 0 or float(ys.max() - ys.min()) < min_rise_m:
            continue

        fine_rc = _expand_to_fine(coarse_rc, block, costmap.grid.shape)
        if fine_rc.shape[0] == 0:
            continue
        centroid = costmap.grid_to_world(fine_rc.mean(axis=0))

        semantic = False
        if semantic_centers is not None and len(semantic_centers):
            xy = np.stack([np.asarray(c, float)[list(PLANE)] for c in semantic_centers])
            semantic = bool(
                (np.linalg.norm(xy - centroid[None, :], axis=1) <= semantic_radius_m).any()
            )
        if require_semantic and not semantic:
            continue

        regions.append(
            StairRegion(
                cells_rc=fine_rc,
                centroid_xy=centroid,
                n_cells=int(fine_rc.shape[0]),
                mean_dh=float(np.nanmean(dh[labels == lbl])),
                low_y=float(ys.min()),
                high_y=float(ys.max()),
                semantic=semantic,
            )
        )
    regions.sort(key=lambda r: -r.n_cells)
    return regions


@dataclass
class Flight:
    """A run of treads between this storey and the next, read off the height
    layer rather than a detector mask.

    `cells_rc` index the costmap grid; `heights` are absolute world y per cell.
    `foot_xy` is the tread nearest THIS storey -- the lowest of an ascent, the
    highest of a descent (`find_flights`) -- and `top_xy` the far end. `kind`
    is "up" or "down" relative to `floor_y`.
    """

    kind: str
    cells_rc: np.ndarray
    heights: np.ndarray
    foot_xy: np.ndarray
    top_xy: np.ndarray
    span_m: float
    #: Walking distance in metres from the MOUTH to each cell, measured THROUGH
    #: the flight's own cells rather than straight across it. A staircase that
    #: doubles back has no usable straight axis -- measured on 00873, 28 of its
    #: 57 densified ground-truth steps run BACKWARDS along the foot->top chord
    #: -- so a carrot chosen by height band or by euclidean distance aims
    #: across the banister at the turn. This gives the run an ORDER instead.
    #: None when the ordering was not requested or could not be built.
    path_m: Optional[np.ndarray] = None

    @property
    def n_cells(self) -> int:
        return int(self.cells_rc.shape[0])


DESCENT_GOALS = ("lowest_tread", "mouth", "extrapolated")


def mouth_xy(flight, descent: str = "lowest_tread", floor_y: Optional[float] = None,
             resolution: Optional[float] = None) -> np.ndarray:
    """Where the agent steps ON to this flight, from its OWN storey.

    An ascent's is its lowest tread, `foot_xy`. A descent's is its top, and
    `find_flights` stores the descent's HIGHEST tread in `foot_xy` too -- but
    this function was written (eb263bc) reading `foot_xy` as the lowest tread
    of every flight, so for a descent it returned `top_xy`, the LOWEST tread
    seen: the far end, on the storey below. `descent` (`floor.descent_goal`):

      lowest_tread  that, as every arm measured through 2026-10-17 ran it
      mouth         the highest tread seen (`foot_xy`)
      extrapolated  the lip, where the flight's slope meets this floor
                    (`descent_mouth_xy`), else the highest tread seen

    Measured on Benchmark v2 seed 42 (mfv2_osg_climb): 00862's descending
    pursuits went to (-0.93, -2.22) and (-0.22, -3.82) for a flight whose top is
    at (-0.69, -1.23), 00873's 2-3 m from its top at (-4.4, -4.3), and no
    episode on either that had to go down ever did.
    """
    if flight.kind == "up":
        return np.asarray(flight.foot_xy, dtype=float)
    if descent == "lowest_tread":
        return np.asarray(flight.top_xy, dtype=float)
    if descent == "extrapolated" and floor_y is not None and resolution is not None:
        lip = descent_mouth_xy(flight, float(floor_y), float(resolution))
        if lip is not None:
            return np.asarray(lip, dtype=float)
    return np.asarray(flight.foot_xy, dtype=float)


def descent_mouth_xy(flight, floor_y: float, resolution: float, local_m: float = 1.2,
                     min_slope: float = 0.35, max_slope: float = 1.2,
                     max_reach_m: float = 2.5) -> Optional[np.ndarray]:
    """Where a DESCENDING flight leaves this storey's floor, extrapolated up its
    slope from the highest tread seen; None when the slope is not a stair's.

    From the storey above, a descending flight is hidden behind its own top
    edge: the camera, 0.88 m up with a 31-degree half field of view, sees a
    tread only along a line steeper than the flight itself, so the treads near
    the top appear only when the agent stands at the edge looking down. Even
    the highest cell of the blob `find_flights` builds from farther away can
    be a metre down the flight -- a point under this storey, which the mover
    can only walk up to the banister of. In `tests/unit/test_flights.py` the
    first four treads of twelve are hidden that way and the highest seen is
    1.1 m from the lip.

    The flight's own surface says where its top is: fit a plane to the treads
    within `local_m` of the highest one (one leg of a switchback, not both),
    and walk up its gradient until it reaches the floor's height.
    """
    if flight.kind != "down" or flight.cells_rc.shape[0] < 6:
        return None
    top = np.asarray(flight.foot_xy, dtype=float)     # a descent's highest tread
    res = float(resolution)
    rc = np.asarray(flight.cells_rc, dtype=float)
    h = np.asarray(flight.heights, dtype=float)
    hi = int(np.argmax(h))
    near = np.linalg.norm((rc - rc[hi]) * res, axis=1) <= float(local_m)
    if near.sum() < 6:
        return None
    # Plane h = a*dr + b*dc + c in metres of row/column offset from the top cell.
    d = (rc[near] - rc[hi]) * res
    A = np.c_[d, np.ones(len(d))]
    coef, *_ = np.linalg.lstsq(A, h[near], rcond=None)
    grad = coef[:2]
    slope = float(np.linalg.norm(grad))
    if not (float(min_slope) <= slope <= float(max_slope)):
        return None
    # Start from the middle of the highest tread, not its highest cell, which
    # sits wherever depth noise put it -- often at the tread's side.
    band = h >= float(h[hi]) - 0.1
    # Rows/columns map to world (x, z) the way `grid_to_world` does: one cell
    # is `resolution` along each, in the same order.
    start = top + (rc[band].mean(axis=0) - rc[hi]) * res
    rise = float(floor_y) - float(h[band].mean())
    if rise <= 0.0:
        return start
    reach = min(rise / slope, float(max_reach_m))
    return start + (grad / slope) * reach


def find_flights(
    costmap: Costmap2D,
    floor_y: float,
    new_level_m: float = 1.8,
    min_span_m: float = 1.0,
    min_cells: int = 150,
    max_cells: int = 4000,
    margin_m: float = 0.2,
    wide_span_m: Optional[float] = None,
    wide_mask: Optional[np.ndarray] = None,
    order_path: bool = False,
) -> List[Flight]:
    """Staircases as connected runs of INTERMEDIATE-height cells.

    Stage 4's detector looked for steppable cells by their local height step
    and failed structurally: a tread's interior is flat, fell below its
    `min_dh`, and the flight fragmented into riser edges. This asks a different
    question. The height layer keeps the lowest surface seen per cell within a
    storey of the floor, so every tread of a flight is a cell whose height sits
    strictly BETWEEN this floor and the next -- and those cells are contiguous,
    because treads touch. Connect them, and keep a component only if its
    heights SPAN at least `min_span_m`: a table is a plateau and spans nothing,
    a sloped floor is large and is capped by `max_cells`, and a flight that
    joins two storeys spans a metre or more by construction.

    The detector's `stairs` mask, seen from below, stamps cells under the upper
    flight: on 00821 its target sat 2.7 m beside the true foot. The lowest cell
    of an intermediate-height component IS the foot, whatever the mask thinks.
    """
    from scipy import ndimage

    if costmap.height is None:
        return []
    rel = costmap.height - float(floor_y)
    seen = np.isfinite(rel)
    lo, hi = float(margin_m), float(new_level_m) - float(margin_m)
    # A taller band, for cells the stair map vouches for. `new_level_m` is a
    # constant and a real storey can be taller: on 00800 it is 3.0 m, and the
    # ramp pasted from the ASCENT map spans all of it, so a 1.8 m band cut the
    # flight at 1.6 m and the climb lost its waypoints half-way up. Widening
    # the band for EVERY cell is not the fix -- from the upper storey the
    # agent sees the lower one over the banister, and those cells are
    # intermediate-height too, so they join the staircase into one blob whose
    # mouth is somewhere else entirely (measured, outputs/mf5_pass2_v14 ep1:
    # a 1460-cell "flight_down" whose goal was 3 m from the real top mouth,
    # pursued for 392 steps, never climbed). Only cells the stair map marks
    # may go beyond the constant.
    wide = None
    if wide_span_m is not None and wide_mask is not None:
        hi_wide = float(wide_span_m) - float(margin_m)
        if hi_wide > hi:
            wide = np.asarray(wide_mask, dtype=bool)
    out: List[Flight] = []
    bands = (
        ("up", seen & (rel > lo) & (rel < hi)),
        ("down", seen & (rel < -lo) & (rel > -hi)),
    )
    if wide is not None:
        bands = (
            ("up", seen & (rel > lo) & ((rel < hi) | (wide & (rel < hi_wide)))),
            ("down", seen & (rel < -lo) & ((rel > -hi) | (wide & (rel > -hi_wide)))),
        )
    for kind, mask in bands:
        if not mask.any():
            continue
        labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
        for label in range(1, n + 1):
            rc = np.argwhere(labels == label)
            if rc.shape[0] < min_cells or rc.shape[0] > max_cells:
                continue
            h = costmap.height[rc[:, 0], rc[:, 1]]
            span = float(h.max() - h.min())
            if span < min_span_m:
                continue
            lowest, highest = int(np.argmin(h)), int(np.argmax(h))
            if kind == "up":
                foot, top = rc[lowest], rc[highest]
            else:
                # Descending, the mouth on THIS floor is the highest tread.
                foot, top = rc[highest], rc[lowest]
            out.append(Flight(
                kind=kind, cells_rc=rc, heights=h,
                foot_xy=costmap.grid_to_world(foot.astype(float)),
                top_xy=costmap.grid_to_world(top.astype(float)),
                span_m=span,
                path_m=(_path_from(rc, foot, float(costmap.resolution))
                        if order_path else None),
            ))
    return out


def _path_from(cells_rc: np.ndarray, source_rc: np.ndarray, resolution: float):
    """Walking distance from `source_rc` to every cell, THROUGH the component.

    A plain BFS over the flight's own cells, 8-connected, with a diagonal
    costing sqrt(2). That is all the ordering a staircase needs: the treads are
    contiguous by construction (`find_flights` builds the component by
    connectivity), so distance-through-the-cells runs along the staircase and
    keeps increasing around a switchback, where distance-across it does not.

    Returns metres, or None when the component is empty. Cells the BFS cannot
    reach (it should reach all of them -- they are one connected component)
    keep `inf`, and callers must treat that as "no ordering here".
    """
    from collections import deque

    if cells_rc.shape[0] == 0:
        return None
    lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(cells_rc)}
    start = lookup.get((int(source_rc[0]), int(source_rc[1])))
    if start is None:
        return None
    dist = np.full(cells_rc.shape[0], np.inf, dtype=float)
    dist[start] = 0.0
    # Dial's algorithm would be exact for two edge weights; a deque BFS with a
    # relaxation pass is close enough for an ordering and much shorter. The
    # carrot only needs monotonicity along the run, not a metric.
    queue = deque([start])
    step = float(resolution)
    diag = step * float(np.sqrt(2.0))
    while queue:
        i = queue.popleft()
        r, c = int(cells_rc[i][0]), int(cells_rc[i][1])
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                j = lookup.get((r + dr, c + dc))
                if j is None:
                    continue
                w = diag if (dr and dc) else step
                if dist[i] + w < dist[j] - 1e-9:
                    dist[j] = dist[i] + w
                    queue.append(j)
    return dist


def apply_stair_mask(
    costmap: Costmap2D, regions: Sequence[StairRegion], max_area_frac: float = 0.05
) -> int:
    """Mark confirmed stair cells traversable. Returns the number of cells set.

    `max_area_frac` caps the relabelled area as a fraction of the KNOWN map: a
    runaway mask would carve free space through real obstacles and, because the
    exemption is permanent, could never be undone.
    """
    if not regions:
        return 0
    if costmap.stair_mask is None:
        costmap.stair_mask = np.zeros(costmap.grid.shape, dtype=bool)

    known = int((costmap.grid != -1).sum())
    # A zero budget means "relabel nothing", not "no limit" -- this is a safety
    # cap on a permanent, irreversible write, so it must fail closed.
    budget = int(max_area_frac * known)
    added = 0
    for region in regions:
        rc = region.cells_rc
        if added + rc.shape[0] > budget:
            break
        costmap.stair_mask[rc[:, 0], rc[:, 1]] = True
        added += rc.shape[0]
    if added:
        costmap.grid[costmap.stair_mask] = FREE
    return added


# ---------------------------------------------------------------------- helpers


def _expand_to_fine(coarse_rc: np.ndarray, block: int, shape: Tuple[int, int]) -> np.ndarray:
    """Coarse cell indices -> every fine cell they cover."""
    if coarse_rc.shape[0] == 0:
        return np.zeros((0, 2), dtype=int)
    offs = np.array([(i, j) for i in range(block) for j in range(block)])
    fine = (coarse_rc[:, None, :] * block + offs[None, :, :]).reshape(-1, 2)
    h, w = shape
    keep = (fine[:, 0] >= 0) & (fine[:, 0] < h) & (fine[:, 1] >= 0) & (fine[:, 1] < w)
    return fine[keep]


@dataclass
class StairDetection:
    kind: str
    centroid_xy: np.ndarray
    cells: np.ndarray
    n_cells: int
    support: int


class StairDetector:
    """Accumulated per-floor up/down stair evidence used by ASCENT policies."""

    def __init__(
        self,
        resolution_m: float = 0.05,
        min_hits: int = 3,
        min_cells: int = 25,
        max_range_m: float = 5.0,
        up_mode: str = "detector",
        close_iters: int = 2,
        disable_margin_m: float = 0.5,
        **_kwargs,
    ) -> None:
        self.resolution_m = float(resolution_m)
        self.min_hits = int(min_hits)
        self.min_cells = int(min_cells)
        self.max_range_m = float(max_range_m)
        self.up_mode = str(up_mode)
        self.close_iters = int(close_iters)
        self.disable_margin_m = float(disable_margin_m)

    @staticmethod
    def _ensure_grids(layer) -> None:
        if layer.up_stair_hits is not None:
            return
        shape = layer.costmap.grid.shape
        layer.up_stair_hits = np.zeros(shape, dtype=np.int16)
        layer.down_stair_hits = np.zeros(shape, dtype=np.int16)
        layer.disabled_stair = np.zeros(shape, dtype=bool)

        def on_grow(h, w, off_r, off_c, current=layer):
            current.up_stair_hits = grow_aligned(
                current.up_stair_hits, h, w, off_r, off_c
            )
            current.down_stair_hits = grow_aligned(
                current.down_stair_hits, h, w, off_r, off_c
            )
            current.disabled_stair = grow_aligned(
                current.disabled_stair, h, w, off_r, off_c, fill=False
            )

        layer.costmap.add_grow_listener(on_grow)

    @staticmethod
    def _stamp(hits: np.ndarray, costmap: Costmap2D, pts: np.ndarray) -> None:
        if pts.shape[0] == 0:
            return
        rc = costmap.world_to_grid(pts[:, list(PLANE)])
        h, w = hits.shape
        ok = (rc[:, 0] >= 0) & (rc[:, 0] < h) & (rc[:, 1] >= 0) & (rc[:, 1] < w)
        rc = np.unique(rc[ok], axis=0)
        if len(rc):
            hits[rc[:, 0], rc[:, 1]] += 1

    def accumulate(
        self,
        frame: FrameData,
        layer,
        dets: Optional[List[Detection]] = None,
        seg_stair_mask: Optional[np.ndarray] = None,
    ) -> None:
        self._ensure_grids(layer)
        pts = backproject(
            frame.depth, frame.intrinsics, frame.T_wc,
            stride=2, max_depth=self.max_range_m,
        )
        if len(pts):
            drop = layer.floor_y - pts[:, HEIGHT_AXIS]
            self._stamp(layer.down_stair_hits, layer.costmap, pts[(drop > 0.35) & (drop < 4.0)])
        masks = []
        if self.up_mode == "rednet" and seg_stair_mask is not None:
            masks = [seg_stair_mask]
        else:
            masks = [d.mask for d in (dets or []) if normalize_label(d.label) in STAIR_LABELS]
        for mask in masks:
            up = backproject(
                frame.depth, frame.intrinsics, frame.T_wc,
                mask=mask, stride=2, max_depth=self.max_range_m,
            )
            self._stamp(layer.up_stair_hits, layer.costmap, up)

    def disable(self, layer, cells: np.ndarray) -> None:
        from scipy import ndimage

        self._ensure_grids(layer)
        if not len(cells):
            return
        mask = np.zeros(layer.disabled_stair.shape, dtype=bool)
        mask[cells[:, 0], cells[:, 1]] = True
        radius = max(1, int(round(self.disable_margin_m / self.resolution_m)))
        mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3)), iterations=radius)
        layer.disabled_stair |= mask
        layer.up_stair_hits[mask] = 0
        layer.down_stair_hits[mask] = 0

    def extract(self, layer) -> List[StairDetection]:
        from scipy import ndimage

        out = []
        for kind, hits in (("up", layer.up_stair_hits), ("down", layer.down_stair_hits)):
            if hits is None:
                continue
            mask = hits >= self.min_hits
            if layer.disabled_stair is not None:
                mask &= ~layer.disabled_stair
            labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
            free = layer.costmap.grid == FREE
            if not free.any():
                continue
            _, (fr, fc) = ndimage.distance_transform_edt(~free, return_indices=True)
            for label in range(1, count + 1):
                cells = np.argwhere(labels == label)
                if len(cells) < self.min_cells:
                    continue
                r, c = np.rint(cells.mean(axis=0)).astype(int)
                r, c = int(fr[r, c]), int(fc[r, c])
                out.append(StairDetection(
                    kind, layer.costmap.grid_to_world(np.array([r, c], float)),
                    cells, len(cells), int(hits[cells[:, 0], cells[:, 1]].sum()),
                ))
        return out
