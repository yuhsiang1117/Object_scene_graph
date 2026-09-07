"""Stair detection geometry.

Thresholds here are anchored to a real measurement on HM3D
(scripts/measure_stair_recall.py, recorded in docs/AB_RESULTS.md):

    up stairs     YOLOE 'stairs' 24% recall / 0% control FP;
                  when it fires the masked points have slope ~0.87
    down stairs   below-floor geometry 50% recall / 0% control FP
"""
from __future__ import annotations

import numpy as np

from osg.mapping.costmap import FREE, Costmap2D
from osg.mapping.floor_stack import FloorLayer
from osg.mapping.stairs import StairDetector, stair_geometry

CAM = np.array([0.0, 0.0])


def _points(xs, ys, zs):
    return np.stack([np.asarray(xs, float), np.asarray(ys, float), np.asarray(zs, float)], axis=1)


def _staircase(n=200, slope=0.65, r0=1.0, r1=3.0):
    """Points on a flight climbing `slope` metres per metre, receding in +z."""
    r = np.linspace(r0, r1, n)
    return _points(np.zeros(n), (r - r0) * slope, r)


# ------------------------------------------------------------------- geometry


def test_staircase_is_stair_like():
    g = stair_geometry(_staircase(), CAM, floor_y=0.0)
    assert 0.6 < g.slope < 0.7
    assert g.is_stair_like(min_span_m=0.6, min_rise_m=0.35, min_slope=0.30)


def test_flat_floor_is_not_stair_like():
    """A patterned rug or tiled floor: span but no rise."""
    n = 200
    r = np.linspace(1.0, 3.0, n)
    g = stair_geometry(_points(np.zeros(n), np.full(n, 0.02), r), CAM, floor_y=0.0)
    assert abs(g.slope) < 0.05
    assert not g.is_stair_like(0.6, 0.35, 0.30)


def test_wall_is_not_stair_like():
    """A wall or shelf front spans heights but sits at ONE distance, so it has
    no horizontal extent to climb over."""
    n = 200
    g = stair_geometry(_points(np.zeros(n), np.linspace(0.0, 2.0, n), np.full(n, 2.5)),
                       CAM, floor_y=0.0)
    assert not g.is_stair_like(0.6, 0.35, 0.30)


def test_too_few_points_is_not_stair_like():
    g = stair_geometry(_staircase(n=5), CAM, floor_y=0.0)
    assert not g.is_stair_like(0.6, 0.35, 0.30)


# ------------------------------------------------------------------ hit grids


def _layer(size_m=10.0, res=0.1):
    cm = Costmap2D(resolution=res, size_m=size_m)
    cm.grid[:, :] = FREE  # so centroids have somewhere standable to snap to
    return FloorLayer(key=0, floor_y=0.0, costmap=cm, planner=object())


def _stamp_column(det, layer, kind, xy, n_frames, half_m=0.4):
    """Simulate `n_frames` observations of a stair patch centred on `xy`."""
    det._ensure_grids(layer)
    hits = layer.up_stair_hits if kind == "up" else layer.down_stair_hits
    g = np.mgrid[-half_m:half_m:0.05, -half_m:half_m:0.05].reshape(2, -1).T + np.asarray(xy)
    pts = _points(g[:, 0], np.zeros(len(g)), g[:, 1])
    for _ in range(n_frames):
        det._stamp(hits, layer.costmap, pts)


def test_extract_finds_a_repeatedly_seen_patch():
    det = StairDetector(min_hits=3, min_cells=25)
    layer = _layer()
    _stamp_column(det, layer, "down", (1.5, 2.0), n_frames=5)

    found = det.extract(layer)
    assert [d.kind for d in found] == ["down"]
    assert np.linalg.norm(found[0].centroid_xy - np.array([1.5, 2.0])) < 0.5


def test_single_sighting_is_rejected():
    """One frame is far too noisy to commit a floor transition to."""
    det = StairDetector(min_hits=3, min_cells=25)
    layer = _layer()
    _stamp_column(det, layer, "down", (1.5, 2.0), n_frames=1)
    assert det.extract(layer) == []


def test_small_component_is_rejected():
    det = StairDetector(min_hits=1, min_cells=200)
    layer = _layer()
    _stamp_column(det, layer, "down", (1.5, 2.0), n_frames=5, half_m=0.05)
    assert det.extract(layer) == []


def test_hit_grids_follow_the_costmap_when_it_grows():
    det = StairDetector(min_hits=3, min_cells=25)
    layer = _layer()
    _stamp_column(det, layer, "down", (1.5, 2.0), n_frames=5)
    before = det.extract(layer)[0].centroid_xy

    layer.costmap.ensure_contains(np.array([30.0, 30.0]), margin_m=1.0)
    layer.costmap.grid[:, :] = FREE

    assert layer.down_stair_hits.shape == layer.costmap.grid.shape
    after = det.extract(layer)[0].centroid_xy
    assert np.linalg.norm(after - before) < 1e-6, "stair moved when the map grew"


def test_disabled_cells_are_never_re_extracted():
    """Retiring a failed staircase must be by CELLS, not centroid.

    The hit grids keep accumulating while the agent looks around, so a
    component grows and its snapped centroid drifts. A centroid blacklist
    therefore lets the same unusable stairwell reappear as a "new" staircase
    every selection round -- measured at 72 climb attempts in one episode, all
    of the same place.
    """
    det = StairDetector(min_hits=3, min_cells=25)
    layer = _layer()
    _stamp_column(det, layer, "down", (1.5, 2.0), n_frames=5)
    found = det.extract(layer)
    assert len(found) == 1

    det.disable(layer, found[0].cells)
    assert det.extract(layer) == []

    # Retiring must also clear the accumulated evidence, or the very next
    # extraction rebuilds the component from hits that are already banked.
    assert layer.down_stair_hits[found[0].cells[:, 0], found[0].cells[:, 1]].max() == 0


def test_disabling_one_staircase_leaves_others():
    det = StairDetector(min_hits=3, min_cells=25)
    layer = _layer(size_m=20.0)
    _stamp_column(det, layer, "down", (1.5, 2.0), n_frames=5)
    _stamp_column(det, layer, "down", (-4.0, -4.0), n_frames=5)
    found = det.extract(layer)
    assert len(found) == 2

    det.disable(layer, found[0].cells)
    left = det.extract(layer)
    assert len(left) == 1
    assert np.linalg.norm(left[0].centroid_xy - found[1].centroid_xy) < 1e-6


def test_min_hits_default_is_the_swept_value():
    """Swept offline over 38 cross-floor episodes measuring per-STAIRCASE recall
    after accumulation, not per-pose detection:

        min_hits  min_cells   found   false components
               3         25    8/38                 38   (the old default)
               1         25   14/38                 26

    1 dominates 3 on both axes. This is the threshold that gates cross-floor
    behaviour -- a 2.5x larger detector took per-pose recall 0% -> 19% and
    produced no extra climb attempts at all.
    """
    from osg.core.config import ExplorationConfig

    cfg = ExplorationConfig()
    assert cfg.stair_min_hits == 1
    assert cfg.stair_min_cells == 25


def test_a_single_frame_can_now_form_a_component():
    """What min_hits=1 buys: one sighting of a wide enough flight is enough.
    At 3 the same evidence is discarded unless the agent happens to see the
    same cells from three keyframes."""
    import numpy as np

    from osg.mapping.costmap import FREE, Costmap2D
    from osg.mapping.stairs import StairDetector

    cm = Costmap2D(resolution=0.05, size_m=8.0)
    # Component centroids are snapped to the nearest standable cell, so the map
    # needs free space to exist at all.
    cm.grid[:, :] = FREE

    class _L:
        costmap = cm
        floor_y = 0.0
        up_stair_hits = None
        down_stair_hits = None
        disabled_stair = None

    layer = _L()
    det = StairDetector(resolution_m=0.05, min_hits=1, min_cells=25)
    det._ensure_grids(layer)
    r, c = cm.world_to_grid(np.array([1.0, 0.0]))
    layer.up_stair_hits[r - 4:r + 4, c - 4:c + 4] = 1  # 64 cells, seen once

    assert [d.kind for d in det.extract(layer)] == ["up"]
    det.min_hits = 3
    assert det.extract(layer) == [], "at 3 the same single sighting is discarded"
