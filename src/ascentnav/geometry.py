"""OSG's world frame -> ASCENT's episodic frame.

ASCENT works in an episodic frame built from GPS + compass: X forward at
yaw 0, Y left, Z up, origin at the episode start
(`ascent_policy.py:234-238`). OSG's `FrameData.T_wc` is an OpenCV camera-to-world
matrix in habitat's y-up world.

Rather than converting a matrix into another convention -- the kind of step that
produces an agent which mirrors every turn while looking plausible -- this
rebuilds ASCENT's transform from the same primitives ASCENT builds it from:
position, yaw, pitch. The (x, -z) flip and the negated heading are exactly the
`to_ccw_frame` conversion already unit-tested for the PointNav mover
(`osg/planning/pointnav_driver.py`), so both consumers share one convention.
"""
from __future__ import annotations

import numpy as np

from osg.mapping.costmap import PLANE
from osg.planning.controller import agent_heading


def xyz_yaw_pitch_roll_to_tf_matrix(
    xyz: np.ndarray, yaw: float, pitch: float, roll: float
) -> np.ndarray:
    """Verbatim port of `ascent/utils.py:122-162`."""
    x, y, z = xyz
    r_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
    r_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
    r_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
    tf = np.eye(4)
    tf[:3, :3] = r_yaw @ r_pitch @ r_roll
    tf[:3, 3] = [x, y, z]
    return tf


def robot_xy_heading(frame) -> tuple:
    """Agent position and heading in ASCENT's CCW frame."""
    xz = frame.camera_position[list(PLANE)]
    return np.array([xz[0], -xz[1]]), -agent_heading(frame.T_wc)


def camera_pitch(frame) -> float:
    """Camera pitch in ASCENT's sign convention.

    ASCENT tracks `_pitch_angle` by hand in degrees (positive = looking up) and
    passes `radians(-pitch_angle)` (`ascent_policy.py:236`), so its
    `camera_pitch` is NEGATIVE when looking up. OSG reads the sensor pose
    instead, which needs no bookkeeping and stays correct if the camera is ever
    tilted by something other than a counted action.

    `T_wc[:3, 2]` is the OpenCV camera forward in the world, and habitat's world
    is y-up, so a positive y component means looking up -- hence the negation.
    """
    fwd = frame.T_wc[:3, 2]
    return float(-np.arcsin(np.clip(fwd[1], -1.0, 1.0)))


def tf_camera_to_episodic(frame, camera_height: float) -> np.ndarray:
    """ASCENT's `tf_camera_to_episodic` for an OSG frame."""
    xy, yaw = robot_xy_heading(frame)
    return xyz_yaw_pitch_roll_to_tf_matrix(
        np.array([xy[0], xy[1], camera_height]), yaw, camera_pitch(frame), 0.0
    )


def normalise_depth(depth: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    """Metres -> the [0, 1] depth every ASCENT map expects.

    ASCENT reads habitat with `normalize_depth=True`; OSG keeps metres because
    its costmap and object layer want them. Both of ASCENT's map updates undo
    this internally with `depth * (max - min) + min`, so the round trip is exact.
    """
    d = np.clip(np.asarray(depth, dtype=np.float32), min_m, max_m)
    return (d - min_m) / (max_m - min_m)
