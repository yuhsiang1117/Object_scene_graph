from __future__ import annotations

import numpy as np
import pytest

from osg.core.types import CameraIntrinsics, FrameData


@pytest.fixture
def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0, width=640, height=480)


def make_camera(position, look_at, up=(0.0, -1.0, 0.0)) -> np.ndarray:
    """T_wc for an OpenCV camera at `position` looking at `look_at`.
    OpenCV: z forward, y down, x right; default up = world -y (habitat y-up)."""
    position = np.asarray(position, dtype=float)
    z = np.asarray(look_at, dtype=float) - position
    z = z / np.linalg.norm(z)
    up = np.asarray(up, dtype=float)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross(np.array([0.0, 0.0, 1.0]), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, position
    return T


def draw_ellipse_mask(h, w, mu, semi_axes, angle=0.0) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w]
    ca, sa = np.cos(angle), np.sin(angle)
    dx, dy = xs - mu[0], ys - mu[1]
    u = ca * dx + sa * dy
    v = -sa * dx + ca * dy
    return (u / semi_axes[0]) ** 2 + (v / semi_axes[1]) ** 2 <= 1.0


def make_frame(intrinsics, T_wc, depth_value=3.0, frame_id=0) -> FrameData:
    h, w = intrinsics.height, intrinsics.width
    return FrameData(
        frame_id=frame_id,
        rgb=np.zeros((h, w, 3), dtype=np.uint8),
        depth=np.full((h, w), depth_value, dtype=np.float32),
        T_wc=T_wc,
        intrinsics=intrinsics,
    )
