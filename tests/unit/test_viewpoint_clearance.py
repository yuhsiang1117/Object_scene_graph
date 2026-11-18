"""A viewpoint the agent does not fit in is not a viewpoint.

`approach_viewpoint` computes a clearance transform and, before this, used it
only to SCORE candidates -- so a pose one cell from a wall could win a ring and
become the approach goal. Habitat's navmesh excludes such a pose; the navmesh
follower silently snapped the goal onto the mesh, but the PointNav mover takes
the goal as a bearing and presses into the obstacle.
"""
from __future__ import annotations

import numpy as np

from osg.mapping.costmap import FREE, OCCUPIED, Costmap2D
from osg.verification.viewpoint import ViewpointPlanner


def _corridor_costmap(res=0.05, size_m=12.0):
    """Free space with a wall, so clearance varies across the rings."""
    cm = Costmap2D(resolution=res, size_m=size_m)
    cm.grid[:, :] = FREE
    return cm


def _mark_wall(cm, x_from, x_to, z):
    for x in np.arange(x_from, x_to, cm.resolution / 2.0):
        rc = cm.world_to_grid(np.array([x, z]))
        if cm.in_bounds(rc):
            cm.grid[rc[0], rc[1]] = OCCUPIED


def test_off_by_default_reproduces_the_old_choice():
    cm = _corridor_costmap()
    obj = np.array([0.0, 0.0])
    old = ViewpointPlanner([0.8], n_samples=32).approach_viewpoint(obj, cm)
    same = ViewpointPlanner([0.8], n_samples=32,
                            min_clearance_m=0.0).approach_viewpoint(obj, cm)
    assert old is not None
    assert np.allclose(old, same)


def test_a_pose_against_a_wall_is_refused():
    """Wall the whole ring except a slot too narrow to stand in."""
    cm = _corridor_costmap()
    obj = np.array([0.0, 0.0])
    # A wall 0.85 m from the object, i.e. just beyond the 0.8 m ring, so every
    # 0.8 m sample on that side has ~0.05 m of clearance.
    _mark_wall(cm, -1.5, 1.5, 0.85)
    lax = ViewpointPlanner([0.8], n_samples=32).approach_viewpoint(obj, cm)
    strict = ViewpointPlanner([0.8], n_samples=32,
                              min_clearance_m=0.30).approach_viewpoint(obj, cm)
    assert lax is not None
    if strict is not None:
        clearance_ok = abs(float(strict[1]) - 0.85) >= 0.30
        assert clearance_ok, "a strict viewpoint must not hug the wall"


def test_clearance_is_measured_from_occupied_cells():
    """A box just outside the ring: every pose on it is within a few cm of a wall."""
    cm = _corridor_costmap()
    obj = np.array([0.0, 0.0])
    w = 0.85
    for z in (w, -w):
        _mark_wall(cm, -w, w + cm.resolution, z)
    for x in (w, -w):          # the two sides, so the ring is boxed in
        for z in np.arange(-w, w + cm.resolution, cm.resolution / 2.0):
            rc = cm.world_to_grid(np.array([x, z]))
            if cm.in_bounds(rc):
                cm.grid[rc[0], rc[1]] = OCCUPIED
    lax = ViewpointPlanner([0.8], n_samples=32).approach_viewpoint(obj, cm)
    assert lax is not None, "the old planner happily stands 5 cm from a wall"
    strict = ViewpointPlanner([0.8], n_samples=32, min_clearance_m=0.6)
    assert strict.approach_viewpoint(obj, cm) is None, (
        "no pose on this ring has 0.6 m of clearance"
    )


def test_a_wide_open_ring_still_returns_a_pose():
    cm = _corridor_costmap()
    obj = np.array([0.0, 0.0])
    strict = ViewpointPlanner([0.8], n_samples=32, min_clearance_m=0.30)
    got = strict.approach_viewpoint(obj, cm)
    assert got is not None
    assert abs(float(np.linalg.norm(got - obj)) - 0.8) < 1e-6
