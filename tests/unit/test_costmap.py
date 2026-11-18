from __future__ import annotations

import numpy as np

from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN, Costmap2D

from .conftest import make_camera, make_frame


def test_update_marks_free_and_obstacle(intrinsics):
    """Camera at origin looking +z(world) at a wall 3 m away: cells along the
    ray free, wall cells occupied."""
    cm = Costmap2D(resolution=0.05, size_m=20.0)
    # Camera 0.88 m above floor (y up in world, camera y down -> use make_camera)
    T_wc = make_camera([0.0, 0.88, 0.0], [0.0, 0.88, 3.0])
    frame = make_frame(intrinsics, T_wc, depth_value=3.0)
    cm.update(frame, floor_y=0.0)

    # The wall at z=3, x=0 is at camera height 0.88 -> obstacle band
    wall_rc = cm.world_to_grid(np.array([0.0, 3.0]))
    assert cm.grid[wall_rc[0], wall_rc[1]] == OCCUPIED
    # Midway cell should be free
    mid_rc = cm.world_to_grid(np.array([0.0, 1.5]))
    assert cm.grid[mid_rc[0], mid_rc[1]] == FREE
    # Behind the camera stays unknown
    behind_rc = cm.world_to_grid(np.array([0.0, -2.0]))
    assert cm.grid[behind_rc[0], behind_rc[1]] == UNKNOWN


def test_autogrow():
    cm = Costmap2D(resolution=0.05, size_m=4.0)
    before = cm.grid.shape
    cm.ensure_contains(np.array([10.0, 10.0]), margin_m=1.0)
    assert cm.grid.shape[0] > before[0]
    rc = cm.world_to_grid(np.array([10.0, 10.0]))
    assert cm.in_bounds(rc)


def test_inflated_dilation():
    cm = Costmap2D(resolution=0.1, size_m=5.0)
    rc = cm.world_to_grid(np.array([0.0, 0.0]))
    cm.grid[rc[0], rc[1]] = OCCUPIED
    inf = cm.inflated(0.3)
    assert inf[rc[0] + 2, rc[1]]  # 0.2 m away is blocked
    assert not inf[rc[0] + 6, rc[1]]  # 0.6 m away is not


def test_clear_footprint_frees_the_disc_under_the_robot_and_nothing_else():
    """The robot is there, so the cells are free: occupied and unknown alike.
    Measured on the Stretch, obstacle speckle under the base cut the map into
    98 islands and frontier detection found nothing from the robot's own."""
    cm = Costmap2D(resolution=0.1, size_m=4.0)
    cm.grid[:, :] = UNKNOWN
    rc = cm.world_to_grid(np.array([0.0, 0.0]))
    cm.grid[rc[0] - 1: rc[0] + 2, rc[1] - 1: rc[1] + 2] = OCCUPIED   # speckle under the base
    cm.grid[rc[0] + 5, rc[1]] = OCCUPIED                              # a real wall, outside the disc

    changed = cm.clear_footprint(np.array([0.0, 0.0]), radius_m=0.3)

    assert cm.grid[rc[0], rc[1]] == FREE
    assert (cm.grid[rc[0] - 1: rc[0] + 2, rc[1] - 1: rc[1] + 2] == FREE).all()
    assert cm.grid[rc[0] + 3, rc[1]] == FREE and cm.grid[rc[0] - 3, rc[1]] == FREE
    assert cm.grid[rc[0] + 5, rc[1]] == OCCUPIED, "outside the radius is untouched"
    assert cm.grid[rc[0] + 4, rc[1]] == UNKNOWN
    assert changed == int((cm.grid == FREE).sum())


def test_clear_footprint_with_no_radius_changes_nothing():
    cm = Costmap2D(resolution=0.1, size_m=4.0)
    before = cm.grid.copy()
    assert cm.clear_footprint(np.array([0.0, 0.0]), radius_m=0.0) == 0
    assert (cm.grid == before).all()
