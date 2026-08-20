from __future__ import annotations

import numpy as np
import pytest

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


def test_the_relaxed_search_still_lands_on_a_ring_when_the_map_is_unknown():
    """The strict search refuses an UNKNOWN cell, and the caller used to fall
    back to the object's own centre -- an occupied cell inside the furniture,
    which puts the agent INSIDE the innermost ring where HM3D cannot score a
    success. Measured over 42 episodes: goals on such a cell scored SR 0.100
    against 0.516 for the rest. Unmapped is not unstandable; the navmesh
    follower finds out for real."""
    cm = Costmap2D(size_m=8.0, resolution=0.05)
    cm.grid[:] = UNKNOWN
    obj = np.array([0.0, 0.0])
    planner = ViewpointPlanner(ring_radii_m=[0.8, 1.2])

    assert planner.approach_viewpoint(obj, cm) is None
    relaxed = planner.approach_viewpoint(
        obj, cm, require_line_of_sight=False, allow_unknown=True
    )
    assert relaxed is not None
    assert np.linalg.norm(relaxed - obj) == pytest.approx(0.8, abs=0.06), (
        "a relaxed viewpoint must still sit ON a ring: success is scored against "
        "goal view points sampled at these radii"
    )


def test_the_relaxed_search_still_refuses_an_occupied_cell():
    """Relaxing 'not yet mapped' must not relax 'known to be a wall'."""
    cm = Costmap2D(size_m=8.0, resolution=0.05)
    cm.grid[:] = OCCUPIED
    vp = ViewpointPlanner(ring_radii_m=[0.8]).approach_viewpoint(
        np.array([0.0, 0.0]), cm, require_line_of_sight=False, allow_unknown=True
    )
    assert vp is None
