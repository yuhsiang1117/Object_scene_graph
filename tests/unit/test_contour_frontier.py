"""Contour-based frontier extraction (ASCENT's method).

The acceptance gate is orientation. An (x, y) vs (row, col) mix-up produces a
frontier set mirrored across the main diagonal: the right *count* of frontiers
at plausible-looking positions, sending the agent the wrong way. That cannot be
caught by eye on a map render, so it is pinned in four directions.

Every fixture here is a walled room. That is not incidental: a free region
floating in unknown space has no split points at all, so the whole contour is
one frontier and its arc midpoint lands at the far corner. The behaviour is
faithful to ASCENT -- it just never arises there, because fog-of-war explored
regions always abut walls.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.mapping.contour_frontier import ContourFrontierExtractor
from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN, Costmap2D

GAPS = {
    "east": np.array([1.0, 0.0]),
    "west": np.array([-1.0, 0.0]),
    "north": np.array([0.0, 1.0]),
    "south": np.array([0.0, -1.0]),
}


def _rect(cm: Costmap2D, x0, x1, z0, z1, value) -> None:
    """Mark an axis-aligned world rectangle, inclusive of both ends."""
    r0, c0 = cm.world_to_grid(np.array([x0, z0], dtype=float))
    r1, c1 = cm.world_to_grid(np.array([x1, z1], dtype=float))
    cm.grid[min(r0, r1):max(r0, r1) + 1, min(c0, c1):max(c0, c1) + 1] = value


def _room(gap: str | None = None, gap_m: float = 1.0, size_m: float = 12.0) -> Costmap2D:
    """A 4x4 m explored room walled on all sides, optionally with one opening.

    The gap is UNKNOWN rather than FREE: the agent has not seen through it yet,
    which is what makes it a frontier.
    """
    cm = Costmap2D(resolution=0.05, size_m=size_m)
    cm.grid[:, :] = UNKNOWN
    _rect(cm, -2, 2, -2, 2, FREE)
    for a in (-2, 2):
        _rect(cm, a, a, -2, 2, OCCUPIED)
        _rect(cm, -2, 2, a, a, OCCUPIED)
    h = gap_m / 2
    if gap == "east":
        _rect(cm, 2, 2, -h, h, UNKNOWN)
    elif gap == "west":
        _rect(cm, -2, -2, -h, h, UNKNOWN)
    elif gap == "north":
        _rect(cm, -h, h, 2, 2, UNKNOWN)
    elif gap == "south":
        _rect(cm, -h, h, -2, -2, UNKNOWN)
    return cm


def _room_with_pocket(pocket_m: float) -> Costmap2D:
    """The room above, with a sealed unexplored pocket beyond its east door."""
    cm = _room("east", gap_m=1.0, size_m=14.0)
    p = pocket_m
    _rect(cm, 2, 2 + p, -p / 2, p / 2, UNKNOWN)
    _rect(cm, 2 + p, 2 + p, -p / 2, p / 2, OCCUPIED)
    _rect(cm, 2, 2 + p, -p / 2, -p / 2, OCCUPIED)
    _rect(cm, 2, 2 + p, p / 2, p / 2, OCCUPIED)
    return cm


# --------------------------------------------------------------- orientation


@pytest.mark.parametrize("gap", list(GAPS))
def test_frontier_lands_at_the_opening(gap):
    """A mirrored coordinate transform sends north/south to east/west, so it
    fails at least two of these four."""
    fronts = ContourFrontierExtractor().extract(_room(gap))
    assert fronts, f"no frontier found for the {gap} opening"

    direction = GAPS[gap]
    best = max(float(f.centroid_xy @ direction) for f in fronts)
    assert best > 1.5, (
        f"{gap}: no frontier lies toward the opening (best projection "
        f"{best:.2f} m, wall is at 2.0 m) -- coordinate order is mirrored"
    )
    # And nothing should be found off to the sides.
    for other, d in GAPS.items():
        if other == gap or np.allclose(d, -direction):
            continue
        assert max(float(f.centroid_xy @ d) for f in fronts) < 1.5


def test_sealed_room_has_no_frontier():
    assert ContourFrontierExtractor().extract(_room(None)) == []


def test_midpoint_lies_on_its_own_contour():
    """Why arc midpoints beat cluster centroids: a concave component's centroid
    falls outside the frontier, an arc midpoint cannot."""
    cm = _room("east")
    for f in ContourFrontierExtractor().extract(cm):
        cells_xy = np.stack([cm.grid_to_world(rc.astype(float)) for rc in f.cells])
        d = float(np.linalg.norm(cells_xy - f.centroid_xy, axis=1).min())
        assert d <= 2 * cm.resolution, "midpoint drifted off its own contour"


# ------------------------------------------------------------- the size filter


@pytest.mark.parametrize(
    "pocket_m,area_m2,expected",
    [(0.6, 0.36, 0), (1.0, 1.0, 0), (2.0, 4.0, 1), (3.0, 9.0, 1)],
)
def test_area_threshold_filters_by_what_the_frontier_opens(pocket_m, area_m2, expected):
    """The behaviour WFD has no equivalent for.

    WFD filters on the frontier's own cell count -- a length. This filters on
    the area the frontier would reveal, so the same 1 m doorway is kept or
    dropped depending on what is behind it. OSG has the notion as a score
    multiplier (info_gain_weight) but not as a filter, so small pockets still
    consume one of the five path-planning slots each round.
    """
    got = ContourFrontierExtractor(area_thresh_m2=1.5).extract(_room_with_pocket(pocket_m))
    assert len(got) == expected, f"{area_m2} m² pocket against a 1.5 m² threshold"


def test_threshold_disabled_keeps_the_small_pocket():
    """-1 is upstream's sentinel for 'no filtering'; without it the parametrized
    test above could pass because the geometry is broken rather than filtered."""
    assert len(ContourFrontierExtractor(area_thresh_m2=-1.0).extract(_room_with_pocket(0.6))) == 1


def test_masks_are_zero_one_not_zero_255():
    """The purity test in _filter_out_small_unexplored compares the values under
    a contour against {1}. With 0/255 masks it can never hold, every pocket
    survives, and the one behaviour this extractor exists for silently stops
    working."""
    navigable, explored = ContourFrontierExtractor().masks(_room_with_pocket(0.6))
    assert set(np.unique(navigable).tolist()) <= {0, 1}
    assert set(np.unique(explored).tolist()) <= {0, 1}


# ------------------------------------------------------------------ interface


def test_unknown_counts_as_navigable():
    """ASCENT's navigable map is the complement of dilated obstacles, so
    never-observed space is navigable. Without that, unexplored space has no
    extent and nothing can border it."""
    cm = _room("east")
    navigable, _ = ContourFrontierExtractor().masks(cm)
    rc = cm.world_to_grid(np.array([4.0, 0.0]))  # well outside the room
    assert navigable[rc[0], rc[1]] == 1


def test_ids_are_unique_and_monotonic():
    """Kept per-round like the WFD extractor, because `blocked` and `failed_out`
    in the selector are keyed by id within a round."""
    ext = ContourFrontierExtractor()
    cm = _room("east")
    ids = [f.id for f in ext.extract(cm)] + [f.id for f in ext.extract(cm)]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_empty_and_fully_explored_maps_yield_nothing():
    blank = Costmap2D(resolution=0.05, size_m=8.0)
    blank.grid[:, :] = UNKNOWN
    assert ContourFrontierExtractor().extract(blank) == []

    full = Costmap2D(resolution=0.05, size_m=8.0)
    full.grid[:, :] = FREE
    assert ContourFrontierExtractor().extract(full) == []
