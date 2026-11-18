"""ROS conventions -> the pipeline's, pinned by construction rather than by eye.

A wrong rotation here does not crash: it builds a mirrored or side-lying map
that looks entirely like a room, and the failure surfaces 400 steps later as an
agent that will not approach anything. So every conversion is tested against
something independent of its own arithmetic -- the repo's existing `W_ROS`, a
round trip, or a projected point.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.core.types import CameraIntrinsics
from osg.ros2 import frames


def _project(intr: CameraIntrinsics, T_wc: np.ndarray, p_w: np.ndarray):
    p_c = np.linalg.inv(T_wc) @ np.append(np.asarray(p_w, float), 1.0)
    return (intr.fx * p_c[0] / p_c[2] + intr.cx, intr.fy * p_c[1] / p_c[2] + intr.cy)


def _pose(quat_xyzw=(0.1, 0.2, 0.3, 0.9), t=(1.0, 2.0, 3.0)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = frames.quat_to_matrix(*quat_xyzw)
    T[:3, 3] = t
    return T


# ------------------------------------------------------------- the world turn


def test_the_world_turn_is_the_one_the_repo_already_uses():
    """`scripts/render_dualmap_sequence.py` replays collector recordings into
    habitat through `W_ROS`. Using a different rotation here would put a map
    built on the robot in a different frame from one built from a recording."""
    W_ROS = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)
    assert np.allclose(frames.T_HAB_ROS, W_ROS)


def test_the_world_turn_is_a_rotation_not_a_reflection():
    """det -1 would mirror the room and still render convincingly."""
    assert frames.looks_like_rotation(frames.T_HAB_ROS)


def test_a_known_point_lands_where_habitat_expects_it():
    """ROS x forward / y left / z up -> habitat x / y up / z."""
    out = (frames.T_HAB_ROS @ np.array([1.0, 2.0, 0.0, 1.0]))[:3]
    assert np.allclose(out, [1.0, 0.0, -2.0])


def test_a_camera_standing_up_reports_its_height_on_the_height_axis():
    """`HEIGHT_AXIS = 1` is where every floor decision reads. A camera 1.3 m up
    in ROS must be 1.3 m up in `camera_position`, or the costmap's obstacle
    band is nonsense."""
    from osg.mapping.costmap import HEIGHT_AXIS

    T_map_cam = np.eye(4)
    T_map_cam[:3, 3] = [2.0, -1.0, 1.3]
    T_wc = frames.ros_pose_to_pipeline(T_map_cam)
    assert T_wc[:3, 3][HEIGHT_AXIS] == pytest.approx(1.3)
    assert frames.looks_like_rotation(T_wc)


def test_the_ground_plane_round_trips_through_ros():
    from osg.mapping.costmap import PLANE

    assert PLANE == (0, 2)  # the axes `pipeline_xy_to_ros` assumes
    xy = np.array([3.0, -4.0])
    assert np.allclose(frames.ros_xy_to_pipeline(*frames.pipeline_xy_to_ros(xy)), xy)


def test_a_goal_and_the_pose_that_reaches_it_agree():
    """The goal conversion and the pose conversion are inverses on the plane:
    drive to the published goal and `camera_position` reads back the goal the
    pipeline asked for. This is the loop the fake-robot check closes."""
    from osg.mapping.costmap import PLANE

    goal_pipeline = np.array([1.5, -2.5])
    gx, gy = frames.pipeline_xy_to_ros(goal_pipeline)
    T_map_cam = np.eye(4)
    T_map_cam[:3, 3] = [gx, gy, 1.3]
    arrived = frames.ros_pose_to_pipeline(T_map_cam)[:3, 3][list(PLANE)]
    assert np.allclose(arrived, goal_pipeline)


def test_goal_yaw_faces_the_direction_of_travel():
    """Nav2's goal checker enforces the orientation, so an arbitrary one makes
    the robot spin on the spot at the end of every leg."""
    # Pipeline +x is ROS +x: a goal straight ahead needs yaw 0.
    assert frames.goal_yaw_ros(np.zeros(2), np.array([2.0, 0.0])) == pytest.approx(0.0)
    # Pipeline -z is ROS +y: yaw +90 degrees.
    assert frames.goal_yaw_ros(np.zeros(2), np.array([0.0, -2.0])) == pytest.approx(np.pi / 2)
    assert frames.goal_yaw_ros(np.zeros(2), np.zeros(2)) == 0.0  # degenerate, not a NaN


def test_yaw_to_quat_and_back():
    for yaw in (0.0, 1.1, -2.4, np.pi):
        R = frames.quat_to_matrix(*frames.yaw_to_quat(yaw))
        assert float(np.arctan2(R[1, 0], R[0, 0])) == pytest.approx(yaw, abs=1e-9)


def test_quat_to_matrix_normalises_and_rejects_nothing_silently():
    R = frames.quat_to_matrix(0.0, 0.0, 2.0, 2.0)  # unnormalised
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.allclose(frames.quat_to_matrix(0.0, 0.0, 0.0, 0.0), np.eye(3))


# ------------------------------------------------------------------ the camera


def test_intrinsics_come_from_the_message_not_from_an_assumed_centre():
    intr = frames.intrinsics_from_camera_info(
        [615.0, 0.0, 331.5, 0.0, 616.0, 241.5, 0.0, 0.0, 1.0], 640, 480)
    assert (intr.fx, intr.fy, intr.cx, intr.cy) == (615.0, 616.0, 331.5, 241.5)
    assert (intr.width, intr.height) == (640, 480)


@pytest.mark.parametrize("rotate_deg", [90, 180, -90, 270])
def test_rotating_the_image_moves_the_camera_model_with_it(rotate_deg):
    """The authority on the rotation's sign is a projected point, not a
    comment: project a world point before and after, and the pixel must move
    exactly the way `np.rot90` moved the image."""
    intr = CameraIntrinsics(fx=320.0, fy=310.0, cx=331.0, cy=239.0, width=640, height=480)
    T_wc = _pose()
    rgb = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 2.0, np.float32)
    p_w = (T_wc @ np.array([0.4, -0.2, 5.0, 1.0]))[:3]

    rgb_r, depth_r, intr_r, T_r = frames.rotate_frame(rgb, depth, intr, T_wc, rotate_deg)

    u, v = _project(intr, T_wc, p_w)
    w, h = intr.width, intr.height
    for _ in range(int(round(rotate_deg / 90.0)) % 4):
        u, v, w, h = v, w - 1 - u, h, w
    assert _project(intr_r, T_r, p_w) == pytest.approx((u, v), abs=1e-6)
    assert (intr_r.width, intr_r.height) == (w, h)
    assert rgb_r.shape[:2] == (h, w) == depth_r.shape
    assert frames.looks_like_rotation(T_r)


def test_no_rotation_leaves_everything_alone():
    intr = CameraIntrinsics(fx=1.0, fy=2.0, cx=3.0, cy=4.0, width=6, height=5)
    rgb = np.zeros((5, 6, 3), np.uint8)
    depth = np.zeros((5, 6), np.float32)
    T = _pose()
    rgb_r, depth_r, intr_r, T_r = frames.rotate_frame(rgb, depth, intr, T, 0.0)
    assert intr_r == intr and np.array_equal(T_r, T)
    assert rgb_r.shape == rgb.shape and depth_r.shape == depth.shape


def test_four_quarter_turns_are_the_identity():
    intr = CameraIntrinsics(fx=1.0, fy=2.0, cx=3.0, cy=4.0, width=5, height=5)
    rgb = np.random.default_rng(1).integers(0, 255, (5, 5, 3), dtype=np.uint8)
    depth = np.zeros((5, 5), np.float32)
    T = _pose()
    r, d, i, t = rgb, depth, intr, T
    for _ in range(4):
        r, d, i, t = frames.rotate_frame(r, d, i, t, 90)
    assert np.array_equal(r, rgb) and i == intr and np.allclose(t, T)


# ------------------------------------------------------------------- the depth


def test_millimetre_depth_becomes_metres_and_keeps_zero_meaning_invalid():
    d = frames.depth_to_metres(np.array([[0, 1000, 2500]], np.uint16), "16UC1", 0.001)
    assert d.dtype == np.float32
    assert np.allclose(d, [[0.0, 1.0, 2.5]])


def test_float_depth_loses_its_nans_because_the_pipeline_tests_greater_than_zero():
    """`depth > 0` is how the costmap and the ellipsoid layer reject invalid
    pixels; a NaN sails past that into the point cloud."""
    d = frames.depth_to_metres(
        np.array([[np.nan, np.inf, 1.5, -1.0]], np.float32), "32FC1")
    assert np.allclose(d, [[0.0, 0.0, 1.5, 0.0]])


def test_depth_beyond_the_sensor_range_is_dropped_not_clipped():
    """Clipping would stamp a wall at max range across the whole view."""
    d = frames.depth_to_metres(np.array([[1.0, 9.0]], np.float32), "32FC1", max_m=5.0)
    assert np.allclose(d, [[1.0, 0.0]])


def test_an_unknown_encoding_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unsupported depth encoding"):
        frames.depth_to_metres(np.zeros((2, 2), np.uint8), "8UC1")
