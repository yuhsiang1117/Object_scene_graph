"""ROS conventions -> the pipeline's conventions. All of it, in one file.

Three separate conversions hide behind the word "frame", and mixing them up
produces a run that looks plausible and maps a mirror image of the room:

  world     ROS `map` is z-up, x-forward, y-left. The pipeline's world is
            habitat's: y-up, ground plane `PLANE = (0, 2)`. `T_HAB_ROS` below
            is the repo's existing answer (scripts/render_dualmap_sequence.py's
            `W_ROS`, used to replay the collector's recordings into habitat);
            reusing it means a map built on the robot and a map built from a
            collector recording land in the same frame.

  camera    ROS's `*_optical_frame` is already OpenCV (z forward, x right,
            y down), which is exactly what `FrameData.T_wc` wants. So there is
            no camera-side conversion at all -- `T_wc = T_HAB_ROS @ T_map_cam`
            -- and the only thing that disturbs it is a physically rotated
            sensor (`rotate_frame`).

  goal      back the other way, and only the ground plane: the pipeline's
            (x, z) -> ROS's (x, y).

Nothing here imports ROS, so it is unit-tested in the CPU suite.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core.types import CameraIntrinsics

# habitat-from-ROS: hab = (ros_x, ros_z, -ros_y). A rotation, det +1 --
# reflections here would mirror the map and still look like a room.
# Identical to `W_ROS` in scripts/render_dualmap_sequence.py; the test asserts
# that, so the two cannot drift apart.
T_HAB_ROS = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, -1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=float,
)


def ros_pose_to_pipeline(T_map_cam: np.ndarray) -> np.ndarray:
    """TF's `map -> camera_optical` -> `FrameData.T_wc`.

    The optical frame is OpenCV already, so only the world side turns.
    """
    return T_HAB_ROS @ np.asarray(T_map_cam, dtype=float)


def pipeline_xy_to_ros(xy: np.ndarray) -> Tuple[float, float]:
    """A ground-plane goal, pipeline (x, z) -> ROS map (x, y)."""
    xy = np.asarray(xy, dtype=float)
    return float(xy[0]), float(-xy[1])


def ros_xy_to_pipeline(x: float, y: float) -> np.ndarray:
    """The inverse of `pipeline_xy_to_ros`, for reading a ROS pose back."""
    return np.array([float(x), float(-y)], dtype=float)


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """ROS quaternion order (x, y, z, w) -> a 3x3 rotation.

    `sim/habitat_env.py` has the same function in habitat's (w, x, y, z) order.
    Two orderings, two functions, rather than one function and a flag: the bug
    this prevents is silent.
    """
    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    if n == 0.0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    """Yaw about ROS `map` z -> (x, y, z, w)."""
    return 0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0))


def transform_to_matrix(translation, rotation_xyzw) -> np.ndarray:
    """TF `Transform` parts -> a 4x4. Kept here so the bridge stays thin."""
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(*rotation_xyzw)
    T[:3, 3] = np.asarray(translation, dtype=float)
    return T


def goal_yaw_ros(agent_xy_pipeline: np.ndarray, goal_xy_pipeline: np.ndarray) -> float:
    """Which way to face on arrival, in ROS map yaw.

    `NavigateToPose` takes a full pose, and the orientation is not optional:
    Nav2's goal checker enforces it, so an arbitrary one makes the robot spin
    on the spot at the end of every leg. Facing the direction of travel means
    the camera arrives pointed at whatever the goal was chosen for.
    """
    ax, ay = pipeline_xy_to_ros(agent_xy_pipeline)
    gx, gy = pipeline_xy_to_ros(goal_xy_pipeline)
    if abs(gx - ax) < 1e-9 and abs(gy - ay) < 1e-9:
        return 0.0
    return float(np.arctan2(gy - ay, gx - ax))


def intrinsics_from_camera_info(K_flat, width: int, height: int) -> CameraIntrinsics:
    """`CameraInfo.k` (row-major 3x3) -> the pipeline's intrinsics.

    Read from the message rather than derived from an HFOV: on a real camera
    the principal point is not the image centre, and the depth alignment the
    driver performs is stated in exactly these numbers.
    """
    K = np.asarray(K_flat, dtype=float).reshape(3, 3)
    return CameraIntrinsics(
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
        width=int(width), height=int(height),
    )


def depth_to_metres(depth: np.ndarray, encoding: str, depth_scale: float = 0.001,
                    max_m: float = 0.0) -> np.ndarray:
    """A ROS depth image -> float32 metres with 0 meaning invalid.

    `16UC1` is millimetres by convention (RealSense publishes it) and already
    uses 0 for "no return", which is the pipeline's spelling too. `32FC1` is
    metres with NaN/inf for no-return, so those become 0 here -- the costmap
    and the ellipsoid layer both test `depth > 0`, and a NaN would sail past
    that into a silently poisoned point cloud.
    """
    d = np.asarray(depth)
    enc = str(encoding).lower()
    if enc in ("16uc1", "mono16"):
        out = d.astype(np.float32) * float(depth_scale)
    elif enc in ("32fc1", ""):
        out = d.astype(np.float32)
    else:
        raise ValueError(f"unsupported depth encoding {encoding!r}")
    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32)
    out[out < 0.0] = 0.0
    if max_m > 0.0:
        out[out > float(max_m)] = 0.0
    return out


def _rot90_once(rgb: np.ndarray, depth: np.ndarray, intr: CameraIntrinsics,
                T_wc: np.ndarray):
    """One 90-degree counter-clockwise turn of the image, carried through the
    intrinsics and the pose.

    `np.rot90` sends pixel (u, v) to (v, W-1-u). Matching that in the pinhole
    model means the new camera axes are X' = Y, Y' = -X, Z' = Z, so fx and fy
    swap, the principal point moves to (cy, W-1-cx), and the pose picks up the
    inverse of that rotation on its camera side. The unit test fixes all of it
    by projecting a point both ways rather than by trusting this comment.
    """
    rgb_r = np.ascontiguousarray(np.rot90(rgb, k=1))
    depth_r = np.ascontiguousarray(np.rot90(depth, k=1))
    intr_r = CameraIntrinsics(
        fx=intr.fy, fy=intr.fx,
        cx=intr.cy, cy=float(intr.width - 1) - intr.cx,
        width=intr.height, height=intr.width,
    )
    R = np.eye(4)
    R[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return rgb_r, depth_r, intr_r, np.asarray(T_wc, dtype=float) @ R


def rotate_frame(rgb: np.ndarray, depth: np.ndarray, intr: CameraIntrinsics,
                 T_wc: np.ndarray, rotate_deg: float = 0.0):
    """Undo a physically rotated camera, in image space.

    The Stretch's D435i is mounted portrait on the head, so the published image
    is the room on its side. Every downstream consumer assumes an upright
    image: the detector was trained on upright photographs, `stair_sense` reads
    the bottom of the frame as the ground, and the costmap's obstacle band is
    a height band. Rotating here -- rather than anywhere later -- keeps that
    assumption true with one knob, `ros2.rotate_deg`.

    Positive is counter-clockwise. Which sign this camera needs is a
    verify-on-robot item; `scripts/check_ros2_pipeline.py` writes the rotated
    RGB out so it can be settled by looking at it.
    """
    k = int(round(float(rotate_deg) / 90.0)) % 4
    for _ in range(k):
        rgb, depth, intr, T_wc = _rot90_once(rgb, depth, intr, T_wc)
    return rgb, depth, intr, T_wc


def looks_like_rotation(T: np.ndarray, tol: float = 1e-6) -> bool:
    """Is the upper-left 3x3 a proper rotation? Used by the checks, because a
    reflection here is the failure that still renders a convincing map."""
    R = np.asarray(T, dtype=float)[:3, :3]
    return bool(
        np.allclose(R @ R.T, np.eye(3), atol=tol)
        and abs(float(np.linalg.det(R)) - 1.0) < 1e-6
    )
