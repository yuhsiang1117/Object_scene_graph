"""Viewpoint rings measured from the object's surface rather than its centre.

The rings were calibrated against a benchmark that scores distance to a sampled
viewpoint. Scoring distance to the OBJECT removes the slack, so the ladder has
to be able to shrink -- and it can only shrink safely if it clears the object,
which is what the extent offset is for.
"""
from __future__ import annotations

import numpy as np

from osg.mapping.costmap import FREE, OCCUPIED, Costmap2D
from osg.verification.viewpoint import ViewpointPlanner


def open_costmap(size_m: float = 20.0, resolution: float = 0.05) -> Costmap2D:
    costmap = Costmap2D(resolution=resolution, size_m=size_m)
    costmap.grid[:] = FREE
    return costmap


def test_extent_offset_pushes_the_ring_out_by_the_object_radius():
    """A ring drawn around a big object must land outside it, not inside."""
    planner = ViewpointPlanner(ring_radii_m=[0.5])
    costmap = open_costmap()
    origin = np.zeros(2)

    centred = planner.approach_viewpoint(origin, costmap)
    offset = planner.approach_viewpoint(origin, costmap, obj_radius_m=0.6)
    assert centred is not None and offset is not None

    # The chosen pose sits on its ring, and the ring is radius + extent.
    assert float(np.linalg.norm(centred - origin)) == 0.5
    assert float(np.linalg.norm(offset - origin)) == 1.1


def test_zero_radius_is_the_original_behaviour():
    """The default must reproduce every result measured before this existed."""
    planner = ViewpointPlanner(ring_radii_m=[0.8, 1.2, 1.5, 2.0])
    costmap = open_costmap()
    origin = np.array([0.3, -0.2])
    assert np.allclose(
        planner.approach_viewpoint(origin, costmap),
        planner.approach_viewpoint(origin, costmap, obj_radius_m=0.0),
    )


def test_the_nearest_usable_ring_wins_so_extra_tight_rings_are_free():
    """Prepending tighter rings can only move the goal inwards, never outwards.

    This is why the preset can add 0.5 and 0.65 below the calibrated ladder
    without removing it: when the tight rings are blocked -- by the table the
    object stands on, usually -- the planner falls through to the old radii.
    """
    origin = np.zeros(2)
    costmap = open_costmap()
    old = ViewpointPlanner(ring_radii_m=[0.8, 1.2, 1.5, 2.0])
    new = ViewpointPlanner(ring_radii_m=[0.5, 0.65, 0.8, 1.2, 1.5, 2.0])
    assert float(np.linalg.norm(new.approach_viewpoint(origin, costmap) - origin)) == 0.5
    assert float(np.linalg.norm(old.approach_viewpoint(origin, costmap) - origin)) == 0.8

    # Block everything within 0.7 m of the object, as a table would: the tight
    # rings become unusable and the ladder falls back to the calibrated radius.
    blocked = open_costmap()
    for row in range(blocked.grid.shape[0]):
        for col in range(blocked.grid.shape[1]):
            xy = blocked.grid_to_world(np.array([row, col]))
            if float(np.linalg.norm(xy - origin)) < 0.7:
                blocked.grid[row, col] = OCCUPIED
    chosen = new.approach_viewpoint(origin, blocked, require_line_of_sight=False)
    assert chosen is not None
    assert float(np.linalg.norm(chosen - origin)) == 0.8
