"""The OSG -> ASCENT frame conversion.

Same failure class as the mover's heading convention: get a sign wrong and every
map is built from mirrored geometry while every intermediate value still looks
reasonable. Pinned against the same fixture `test_controller.py` uses.
"""
from __future__ import annotations

import numpy as np
import pytest

from ascentnav.geometry import (
    camera_pitch,
    normalise_depth,
    robot_xy_heading,
    tf_camera_to_episodic,
    xyz_yaw_pitch_roll_to_tf_matrix,
)
from osg.core.types import CameraIntrinsics, FrameData

from .conftest import make_camera

INTR = CameraIntrinsics.from_hfov(79.0, 640, 480)


def _frame(pos, look_at):
    T = make_camera(pos, look_at)
    return FrameData(frame_id=0, rgb=np.zeros((480, 640, 3), np.uint8),
                     depth=np.full((480, 640), 2.0, np.float32),
                     T_wc=T, intrinsics=INTR)


def test_position_flips_the_second_plane_axis():
    """ASCENT's episodic frame is CCW-positive; OSG's (x, z) plane is not."""
    xy, _ = robot_xy_heading(_frame([3.0, 0.88, 4.0], [4.0, 0.88, 4.0]))
    assert xy == pytest.approx([3.0, -4.0])


def test_heading_is_negated_into_the_ccw_frame():
    # facing world +x -> yaw 0 in both frames
    _, yaw = robot_xy_heading(_frame([0, 0.88, 0], [1, 0.88, 0]))
    assert yaw == pytest.approx(0.0)
    # facing world +z is a RIGHT turn in OSG (test_controller pins this), so it
    # must be a NEGATIVE yaw in a CCW-positive frame.
    _, yaw = robot_xy_heading(_frame([0, 0.88, 0], [0, 0.88, 1]))
    assert yaw == pytest.approx(-np.pi / 2)


def test_level_camera_has_zero_pitch():
    assert camera_pitch(_frame([0, 0.88, 0], [1, 0.88, 0])) == pytest.approx(0.0, abs=1e-9)


def test_looking_down_is_positive_pitch_in_ascents_convention():
    """ASCENT passes `radians(-pitch_angle)`, so looking UP is negative."""
    down = camera_pitch(_frame([0, 0.88, 0], [1, 0.38, 0]))
    up = camera_pitch(_frame([0, 0.88, 0], [1, 1.38, 0]))
    assert down > 0 and up < 0
    assert down == pytest.approx(-up)


def _centre_pixel_cloud(depth_m):
    """A single depth return dead centre, through VLFM's own projection.

    Deriving the camera convention from `get_point_cloud` rather than asserting
    one: it returns `(z, -x, -y)`, i.e. X forward, Y left, Z up -- which is not
    the (x right, y down, z forward) an OpenCV habit would assume, and guessing
    it wrong is exactly the error this file exists to catch.
    """
    from ascentnav.vendor.vlfm.utils.geometry_utils import get_point_cloud

    d = np.zeros((480, 640), np.float32)
    d[240, 320] = depth_m
    mask = d > 0
    fx = fy = 640 / (2 * np.tan(np.radians(79) / 2))
    return get_point_cloud(d, mask, fx, fy)


def test_camera_frame_is_x_forward_y_left_z_up():
    cloud = _centre_pixel_cloud(3.0)
    assert np.allclose(cloud[0], [3.0, 0.0, 0.0], atol=1e-6)


def test_transform_places_a_forward_point_ahead_of_the_agent():
    """The end-to-end check that matters: a return straight ahead of the camera
    must land ahead of the agent in the episodic frame, with no lateral drift."""
    from ascentnav.vendor.vlfm.utils.geometry_utils import transform_points

    f = _frame([2.0, 0.88, -1.0], [3.0, 0.88, -1.0])  # at (2,-1) facing world +x
    world = transform_points(tf_camera_to_episodic(f, 0.88), _centre_pixel_cloud(3.0))[0]
    xy, _ = robot_xy_heading(f)
    assert world[0] == pytest.approx(xy[0] + 3.0, abs=1e-6)
    assert world[1] == pytest.approx(xy[1], abs=1e-6)
    assert world[2] == pytest.approx(0.88, abs=1e-6), "level camera: same height"


def test_turning_right_moves_the_forward_point_right():
    from ascentnav.vendor.vlfm.utils.geometry_utils import transform_points

    f = _frame([0, 0.88, 0], [0, 0.88, 1])  # facing world +z = OSG's "right"
    world = transform_points(tf_camera_to_episodic(f, 0.88), _centre_pixel_cloud(3.0))[0]
    # world +z maps to -y in the CCW frame, so 3 m ahead must be y = -3
    assert world[1] == pytest.approx(-3.0, abs=1e-6)
    assert world[0] == pytest.approx(0.0, abs=1e-6)


def test_looking_down_puts_the_point_below_the_camera():
    """The sign check the stair probe uses, applied to this transform."""
    from ascentnav.vendor.vlfm.utils.geometry_utils import transform_points

    level = _frame([0, 0.88, 0], [1, 0.88, 0])
    down = _frame([0, 0.88, 0], [1, 0.38, 0])
    zs = [transform_points(tf_camera_to_episodic(f, 0.88), _centre_pixel_cloud(3.0))[0][2]
          for f in (level, down)]
    assert zs[0] == pytest.approx(0.88, abs=1e-6)
    assert zs[1] < zs[0], "tilting down must lower the backprojected point"


def test_depth_normalisation_round_trips():
    d = np.array([[0.5, 2.75, 5.0]], np.float32)
    n = normalise_depth(d, 0.5, 5.0)
    assert np.allclose(n, [[0.0, 0.5, 1.0]])
    assert np.allclose(n * (5.0 - 0.5) + 0.5, d)


def test_transform_matches_the_ascent_helper_exactly():
    tf = xyz_yaw_pitch_roll_to_tf_matrix(np.array([1.0, 2.0, 3.0]), 0.3, 0.0, 0.0)
    assert np.allclose(tf[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(tf[:2, :2], [[np.cos(0.3), -np.sin(0.3)],
                                    [np.sin(0.3), np.cos(0.3)]])
