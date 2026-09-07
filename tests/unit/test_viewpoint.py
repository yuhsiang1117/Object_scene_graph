from __future__ import annotations

import numpy as np

from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN, Costmap2D
from osg.verification.viewpoint import ViewpointPlanner


def test_viewpoint_found_in_open_room():
    cm = Costmap2D(resolution=0.1, size_m=10.0)
    cm.grid[:, :] = FREE
    obj = np.array([0.0, 0.0])
    rc = cm.world_to_grid(obj)
    cm.grid[rc[0] - 2 : rc[0] + 3, rc[1] - 2 : rc[1] + 3] = OCCUPIED  # object blob
    vp = ViewpointPlanner().approach_viewpoint(obj, cm)
    assert vp is not None
    d = np.linalg.norm(vp - obj)
    assert 0.7 <= d <= 2.1


def test_viewpoint_respects_walls():
    """Object fully enclosed by walls with free space only outside -> the
    chosen viewpoint must not be through a wall (no line of sight)."""
    cm = Costmap2D(resolution=0.1, size_m=10.0)
    cm.grid[:, :] = FREE
    obj = np.array([0.0, 0.0])
    rc = cm.world_to_grid(obj)
    # Box wall at ~0.6 m around the object, opening on the +row side
    r0, c0 = rc
    cm.grid[r0 - 6, c0 - 6 : c0 + 7] = OCCUPIED
    cm.grid[r0 - 6 : r0 + 7, c0 - 6] = OCCUPIED
    cm.grid[r0 - 6 : r0 + 7, c0 + 6] = OCCUPIED
    # (+row side open)
    vp = ViewpointPlanner(ring_radii_m=[0.8, 1.2]).approach_viewpoint(obj, cm)
    assert vp is not None
    # Viewpoint must be on the open (+row = +x world) side
    assert vp[0] > 0


def test_no_viewpoint_when_unknown():
    cm = Costmap2D(resolution=0.1, size_m=10.0)
    cm.grid[:, :] = UNKNOWN
    vp = ViewpointPlanner().approach_viewpoint(np.array([0.0, 0.0]), cm)
    assert vp is None
