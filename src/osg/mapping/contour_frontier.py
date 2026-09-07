"""Contour-based frontier extraction, ported from ASCENT.

The WFD extractor in `frontier.py` clusters UNKNOWN *cells* and takes their
centroid. ASCENT instead takes the **contour of the explored region**, splits it
wherever it stops bordering unexplored space, and puts a waypoint at each
surviving arc's midpoint. Two consequences make this worth measuring:

* A 2D cluster centroid can fall outside its own frontier (a concave component's
  centroid sits in the notch), which is why `selector.frontier_goal_xy` exists to
  re-pick a real cell. An arc midpoint is on the contour by construction.
* The size filter measures something different. WFD drops frontiers with fewer
  than `min_cells` cells -- a length threshold on the frontier. ASCENT drops
  frontiers whose *adjacent unexplored region* is smaller than `area_thresh`
  (1.5 m² in `experiments/eval_ascent_hm3d.yaml`), so a long frontier onto a
  closet is removed while a narrow doorway into a large room is kept. OSG has
  the same notion as a score multiplier (`info_gain_weight`) but not as a
  filter, so small pockets still consume one of the five path-planning slots
  per round.

Ported from `third_party/frontier_exploration/frontier_exploration/
frontier_detection.py`, function for function. Deliberate deviations:

* `contour_to_frontiers` is `@njit` upstream; numba is not a dependency here, so
  it is plain Python. It runs once per selection round on a few hundred contour
  points.
* Upstream densifies contours with its own `bresenhamline`, which excludes the
  start point and includes the end. `core.geometry.bresenham` includes both, so
  each segment drops its first point to reproduce the same point set.
* No restriction to the robot's reachable component. WFD does that
  (`frontier.py:69-80`); ASCENT does not, and reproducing ASCENT is the point.
  `robot_xy` is accepted for signature compatibility and ignored.
* No dedup pass. Contour splitting makes the arcs disjoint by construction.

All internal work is in OpenCV's (x, y) = (col, row) convention, matching
upstream; the conversion to (row, col) happens once, where `Frontier` is built.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from ..core.geometry import bresenham
from .costmap import UNKNOWN, Costmap2D
from .frontier import Frontier


def _interpolate_contour(contour: np.ndarray) -> np.ndarray:
    """Densify a cv2 contour so adjacent points are 8-connected.

    Upstream builds segments between consecutive points plus one closing the
    loop, then rasterises each. `bresenhamline` there excludes the start point,
    so segments abut without repeats; `bresenham` here includes it, hence the
    `[1:]`.
    """
    pts_xy = contour.reshape(-1, 2)
    if pts_xy.shape[0] < 2:
        return contour.reshape(-1, 1, 2)
    out: List[np.ndarray] = []
    for i in range(pts_xy.shape[0]):
        x0, y0 = pts_xy[i]
        x1, y1 = pts_xy[(i + 1) % pts_xy.shape[0]]  # last segment closes the loop
        line = [(c, r) for r, c in bresenham(int(y0), int(x0), int(y1), int(x1))]
        if len(line) > 1:
            out.append(np.array(line[1:], dtype=np.int32))
    if not out:
        return contour.reshape(-1, 1, 2)
    return np.concatenate(out).reshape(-1, 1, 2)


def _contour_to_frontiers(contour: np.ndarray, unexplored: np.ndarray) -> List[np.ndarray]:
    """Split a contour into the runs of points that border unexplored space.

    Direct port. Three details that are easy to lose and change the output:
    `np.split` leaves the split element at the head of the following piece, so
    every piece but the first drops one point; a run needs more than two points
    to count; and a frontier straddling the contour's start index has to be
    stitched back together, or the seam silently halves it.
    """
    bad_inds: List[int] = []
    n = len(contour)
    for idx in range(n):
        x, y = contour[idx][0]
        if unexplored[y, x] == 0:
            bad_inds.append(idx)

    pieces = np.split(contour, bad_inds)
    front_last_split = (
        0 not in bad_inds and len(bad_inds) > 0 and max(bad_inds) < n - 2
    )
    kept: List[np.ndarray] = []
    for idx, f in enumerate(pieces):
        if len(f) > 2 or (idx == 0 and front_last_split):
            kept.append(f if idx == 0 else f[1:])

    if len(kept) > 1 and front_last_split:
        last = kept.pop()
        kept[0] = np.concatenate((last, kept[0]))
    return kept


def _frontier_midpoint(frontier: np.ndarray) -> Optional[np.ndarray]:
    """The point half-way along the arc, by cumulative segment length."""
    pts = frontier.reshape(-1, 2).astype(np.float64)
    if pts.shape[0] < 2:
        return None
    seg = np.stack([pts[:-1], pts[1:]], axis=1)
    lengths = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
    cum = np.cumsum(lengths)
    total = cum[-1]
    if total <= 0:
        return pts[0]
    half = total / 2.0
    i = int(np.argmax(cum > half))
    up_to = cum[i - 1] if i > 0 else 0.0
    if lengths[i] <= 0:
        return seg[i, 0]
    t = (half - up_to) / lengths[i]
    return seg[i, 0] + t * (seg[i, 1] - seg[i, 0])


def _filter_out_small_unexplored(
    navigable: np.ndarray, explored: np.ndarray, area_thresh_px: float
) -> np.ndarray:
    """Absorb unexplored pockets below `area_thresh_px` into the explored mask.

    This is the filter that has no equivalent in the WFD path. A pocket that is
    entirely unexplored and too small to be worth a detour is marked explored,
    so no frontier is ever generated onto it.

    Both masks must be 0/1, not 0/255: the purity test below compares the set of
    values under the contour against `{1}`. With 0/255 inputs it can never hold,
    every pocket survives, and the filter silently does nothing -- which is the
    entire reason to prefer this extractor.
    """
    if area_thresh_px < 0:
        return explored

    unexplored = navigable.copy()
    unexplored[explored > 0] = 0
    contours, _ = cv2.findContours(unexplored, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    small = []
    for c in contours:
        if cv2.contourArea(c) < area_thresh_px:
            mask = cv2.drawContours(np.zeros_like(explored), [c], 0, 1, -1)
            if set(unexplored[mask.astype(bool)].tolist()) == {1}:
                small.append(c)

    out = explored.copy()
    if small:
        # 255, matching upstream: the mask is only ever tested with `> 0`.
        cv2.drawContours(out, small, -1, 255, -1)
    return out


def _detect_frontiers(
    navigable: np.ndarray, explored: np.ndarray, area_thresh_px: float
) -> List[np.ndarray]:
    explored = explored.copy()
    explored[navigable == 0] = 0
    filtered = _filter_out_small_unexplored(navigable, explored, area_thresh_px)

    contours, _ = cv2.findContours(filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    unexplored = np.where(filtered > 0, 0, navigable)
    # Blur for leeway, so a contour running one pixel off the boundary still
    # registers as bordering unexplored space.
    unexplored = cv2.blur(np.where(unexplored > 0, 255, unexplored).astype(np.uint8), (3, 3))

    out: List[np.ndarray] = []
    for c in contours:
        if len(c) == 0:
            continue
        out.extend(_contour_to_frontiers(_interpolate_contour(c), unexplored))
    return out


class ContourFrontierExtractor:
    """Drop-in alternative to `FrontierExtractor` with the same `extract`."""

    def __init__(
        self,
        area_thresh_m2: float = 1.5,   # eval_ascent_hm3d.yaml:27
        agent_radius_m: float = 0.18,
    ) -> None:
        self.area_thresh_m2 = area_thresh_m2
        self.agent_radius_m = agent_radius_m
        self._next_id = 0

    def masks(self, costmap: Costmap2D) -> tuple:
        """The two binary masks ASCENT's detector consumes, derived from OSG's
        single tri-state grid.

        `navigable` is the complement of the obstacles dilated by the agent
        radius, so UNKNOWN counts as navigable -- that is ASCENT's semantics and
        what makes unexplored space have any extent at all. `explored` is
        dilated 5x5 and re-masked by navigable, mirroring `_get_frontiers` and
        `obstacle_map.py:365`.

        OSG's own `inflated()` is used rather than ASCENT's square kernel (7 px
        square, so 0.15 m of growth, against 0.2 m for the disc here). Matching
        ASCENT exactly would leave extraction disagreeing with the planner that
        has to reach these frontiers, which matters more than 0.05 m.
        """
        navigable = (~costmap.inflated(self.agent_radius_m)).astype(np.uint8)
        explored = (costmap.grid != UNKNOWN).astype(np.uint8)
        explored = cv2.dilate(explored, np.ones((5, 5), np.uint8), iterations=1)
        explored[navigable == 0] = 0
        return navigable, explored

    def extract(
        self,
        costmap: Costmap2D,
        robot_xy: Optional[np.ndarray] = None,  # unused; see module docstring
        floor_key: int = 0,
    ) -> List[Frontier]:
        navigable, explored = self.masks(costmap)
        if not navigable.any() or not explored.any():
            return []

        area_px = self.area_thresh_m2 / (costmap.resolution ** 2)
        segments = _detect_frontiers(navigable, explored, area_px)

        out: List[Frontier] = []
        for seg in segments:
            mid_xy = _frontier_midpoint(seg)
            if mid_xy is None:
                continue
            # (x, y) -> (row, col) for everything downstream.
            centroid_rc = np.array([mid_xy[1], mid_xy[0]], dtype=float)
            cells = seg.reshape(-1, 2)[:, ::-1].copy()
            out.append(
                Frontier(
                    id=self._next_id,
                    centroid_xy=costmap.grid_to_world(centroid_rc),
                    cells=cells,
                    size=int(cells.shape[0]),
                    floor_key=floor_key,
                )
            )
            self._next_id += 1
        return out
