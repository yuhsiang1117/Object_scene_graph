from __future__ import annotations

import numpy as np

from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN, Costmap2D
from osg.mapping.frontier import FrontierExtractor


def _map_with_opening() -> Costmap2D:
    """A free room with walls, one opening into unknown space."""
    cm = Costmap2D(resolution=0.1, size_m=10.0)
    cm.grid[:, :] = UNKNOWN
    cm.grid[40:60, 40:60] = FREE
    cm.grid[39, 39:61] = OCCUPIED
    cm.grid[60, 39:61] = OCCUPIED
    cm.grid[39:61, 39] = OCCUPIED
    cm.grid[40:44, 60] = OCCUPIED   # partial right wall ...
    cm.grid[52:61, 60] = OCCUPIED   # ... with a gap at rows 44..51
    return cm


def test_single_frontier_at_opening():
    cm = _map_with_opening()
    frontiers = FrontierExtractor(min_cells=3, dedup_m=0.5).extract(cm)
    assert len(frontiers) == 1
    f = frontiers[0]
    # The frontier sits at the gap (rows 44-51, col 59)
    assert 43 <= f.cells[:, 0].mean() <= 52
    assert 58 <= f.cells[:, 1].mean() <= 60


def test_min_cells_filters_noise():
    cm = _map_with_opening()
    # Tiny unknown pinhole inside the room -> 1-2 frontier cells around it
    cm.grid[50, 50] = UNKNOWN
    frontiers = FrontierExtractor(min_cells=6, dedup_m=0.5).extract(cm)
    assert len(frontiers) == 1  # pinhole ignored


def test_dedup_merges_close_frontiers():
    cm = _map_with_opening()
    # Second small gap adjacent to the first (within dedup distance)
    cm.grid[46:48, 60] = FREE
    near = FrontierExtractor(min_cells=2, dedup_m=3.0).extract(cm)
    far = FrontierExtractor(min_cells=2, dedup_m=0.05).extract(cm)
    assert len(near) <= len(far)
