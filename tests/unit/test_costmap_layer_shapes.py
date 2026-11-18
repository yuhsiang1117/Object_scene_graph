"""A restored snapshot replaces `grid`; the parallel layers must follow.

`ensure_contains` keeps `height`, `stair_mask` and `height_synthetic` with the
grid when the map grows, and says so in a comment: "Every parallel layer must
grow with it or the shapes silently diverge." Restoring a prior map is the
other way the grid changes shape, and nothing was enforcing it there -- a
400x400 `height_synthetic` sat behind a restored 576x576 grid and the first
depth frame raised

    IndexError: index 330800 is out of bounds for axis 0 with size 160000

from `_record_heights`. That is episode 1 of every run that loads a prior map
with its occupancy, which is the whole single-floor DualMap arm.
"""
from __future__ import annotations

import numpy as np

from osg.mapping.costmap import Costmap2D


def _grown(resolution=0.05, size_m=20.0):
    cm = Costmap2D(resolution=resolution, size_m=size_m, track_height=True)
    assert cm.height_synthetic is not None, "track_height should allocate the flags"
    return cm


def test_growing_keeps_every_layer_with_the_grid():
    cm = _grown()
    before = cm.grid.shape
    cm.ensure_contains(np.array([200.0, 200.0]), margin_m=2.0)
    assert cm.grid.shape != before, "the fixture did not actually grow the map"
    for name in ("height", "stair_mask", "height_synthetic"):
        layer = getattr(cm, name)
        if layer is not None:
            assert layer.shape == cm.grid.shape, name


def test_replacing_the_grid_reshapes_the_layers():
    cm = _grown()
    cm.stair_mask = np.zeros(cm.grid.shape, dtype=bool)
    # What `map_store.restore_grid` does: a wholesale swap at the snapshot's
    # extent, which is whatever the mapping pass grew to.
    cm.grid = np.full((576, 576), -1, dtype=cm.grid.dtype)
    cm.height = np.full((576, 576), np.nan, dtype=np.float32)
    cm.conform_layers()
    for name in ("height", "stair_mask", "height_synthetic"):
        layer = getattr(cm, name)
        assert layer is not None and layer.shape == (576, 576), name


def test_recording_heights_survives_a_restored_grid():
    """The failure as it actually arrived: one depth point, straight to
    IndexError."""
    cm = _grown()
    cm.grid = np.full((576, 576), -1, dtype=cm.grid.dtype)
    cm.height = np.full((576, 576), np.nan, dtype=np.float32)
    cm.conform_layers()

    # A point near the far corner, so its flat index exceeds the old 400x400.
    far = cm.origin + np.array([560, 560]) * cm.resolution
    pts = np.zeros((1, 3), dtype=float)
    pts[0, 0], pts[0, 2] = far[0], far[1]
    pts[0, 1] = 1.25
    cm._record_heights(pts)   # must not raise

    rc = np.floor((far - cm.origin) / cm.resolution).astype(int)
    assert cm.height[rc[0], rc[1]] == np.float32(1.25)


def test_a_synthetic_height_is_replaced_by_a_real_reading():
    """The flags still do their job after a conform -- the mechanism they exist
    for is the pasted ASCENT ramp, and blanking them must not disable it."""
    cm = _grown()
    rc = (10, 12)
    xy = cm.origin + np.array(rc) * cm.resolution + cm.resolution / 2.0
    cm.height[rc] = -3.26          # the invented ramp, which runs low
    cm.height_synthetic[rc] = True

    pts = np.zeros((1, 3), dtype=float)
    pts[0, 0], pts[0, 2], pts[0, 1] = xy[0], xy[1], -1.21   # the true surface
    cm._record_heights(pts)

    assert cm.height[rc] == np.float32(-1.21), "a real reading must replace the invention"
    assert not cm.height_synthetic[rc], "and clear the flag"
