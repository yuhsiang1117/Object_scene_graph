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
    """Agent position and heading in ASCENT's CCW frame, WORLD-anchored."""
    xz = frame.camera_position[list(PLANE)]
    return np.array([xz[0], -xz[1]]), -agent_heading(frame.T_wc)


class EpisodeAnchor:
    """The pose the episode started at, so every map is centred on it.

    ASCENT's maps live in a GPS/compass frame -- origin at the episode start,
    heading 0 along the start facing (`ascent_policy.py:233-237`). The port
    fed raw world coordinates into `_xy_to_px`, which has no bounds check and a
    40 m half-extent from the WORLD origin; on this split the furthest start is
    22.9 m out, so the budget was being quietly spent. Anchoring at the start
    is what the reference does and what the map size was chosen for.
    """

    def __init__(self, xy_world: np.ndarray, heading_world: float) -> None:
        self.xy = np.asarray(xy_world, dtype=float).copy()
        self.heading = float(heading_world)
        c, s = np.cos(-self.heading), np.sin(-self.heading)
        self._r_world_to_ep = np.array([[c, -s], [s, c]])
        self._r_ep_to_world = self._r_world_to_ep.T

    @classmethod
    def from_frame(cls, frame) -> "EpisodeAnchor":
        xy, heading = robot_xy_heading(frame)
        return cls(xy, heading)

    def to_episodic(self, xy_world: np.ndarray) -> np.ndarray:
        return self._r_world_to_ep @ (np.asarray(xy_world, dtype=float) - self.xy)

    def to_world(self, xy_ep: np.ndarray) -> np.ndarray:
        return self._r_ep_to_world @ np.asarray(xy_ep, dtype=float) + self.xy

    def heading_to_episodic(self, heading_world: float) -> float:
        h = float(heading_world) - self.heading
        return float(np.arctan2(np.sin(h), np.cos(h)))


def episodic_xy_heading(frame, anchor: EpisodeAnchor) -> tuple:
    """Agent position and heading in ASCENT's CCW frame, EPISODE-anchored."""
    xy, heading = robot_xy_heading(frame)
    return anchor.to_episodic(xy), anchor.heading_to_episodic(heading)


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


def tf_camera_to_episodic(frame, camera_height: float, anchor: "EpisodeAnchor | None" = None,
                          pitch: "float | None" = None) -> np.ndarray:
    """ASCENT's `tf_camera_to_episodic` for an OSG frame.

    `pitch` in ASCENT's convention (radians, negative when looking up) may be
    supplied by a caller that tracks it by hand, as the reference does.
    """
    xy, yaw = (episodic_xy_heading(frame, anchor) if anchor is not None
               else robot_xy_heading(frame))
    p = camera_pitch(frame) if pitch is None else float(pitch)
    return xyz_yaw_pitch_roll_to_tf_matrix(
        np.array([xy[0], xy[1], camera_height]), yaw, p, 0.0
    )


def normalise_depth(depth: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    """Metres -> the [0, 1] depth every ASCENT map expects.

    ASCENT reads habitat with `normalize_depth=True`; OSG keeps metres because
    its costmap and object layer want them. Both of ASCENT's map updates undo
    this internally with `depth * (max - min) + min`, so the round trip is exact.
    """
    d = np.clip(np.asarray(depth, dtype=np.float32), min_m, max_m)
    return (d - min_m) / (max_m - min_m)
