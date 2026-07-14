from __future__ import annotations

import numpy as np
import pytest

from osg.core.geometry import ellipse_from_mask
from osg.core.types import Detection
from osg.objects.ellipsoid import Ellipsoid

from .conftest import draw_ellipse_mask, make_camera, make_frame


def test_projection_center(intrinsics):
    """An ellipsoid straight ahead projects to the principal point."""
    T_wc = np.eye(4)  # camera at origin looking +z (world == camera)
    e = Ellipsoid(center=np.array([0.0, 0.0, 4.0]), axes=np.array([0.5, 0.3, 0.4]), R=np.eye(3))
    ell = e.project(intrinsics.K(), np.linalg.inv(T_wc))
    assert ell is not None
    assert np.allclose(ell.mu, [intrinsics.cx, intrinsics.cy], atol=1.0)
    # Semi-axes in pixels ~ axis * f / z
    semi, _ = ell.axes_angle()
    assert semi[0] == pytest.approx(0.5 * intrinsics.fx / 4.0, rel=0.05)


def test_projection_behind_camera(intrinsics):
    e = Ellipsoid(center=np.array([0.0, 0.0, -2.0]), axes=np.array([0.3, 0.3, 0.3]), R=np.eye(3))
    assert e.project(intrinsics.K(), np.eye(4)) is None


def test_projection_off_center(intrinsics):
    e = Ellipsoid(center=np.array([1.0, 0.5, 5.0]), axes=np.array([0.3, 0.3, 0.3]), R=np.eye(3))
    ell = e.project(intrinsics.K(), np.eye(4))
    assert ell is not None
    expected_u = intrinsics.cx + 1.0 / 5.0 * intrinsics.fx
    expected_v = intrinsics.cy + 0.5 / 5.0 * intrinsics.fy
    assert np.allclose(ell.mu, [expected_u, expected_v], atol=2.0)


def test_init_from_detection_roundtrip(intrinsics):
    """Draw the projection of a known object, init from it, and check the
    recovered center."""
    z = 3.0
    center_px = (400.0, 200.0)
    semi_px = (60.0, 40.0)
    mask = draw_ellipse_mask(intrinsics.height, intrinsics.width, center_px, semi_px)
    frame = make_frame(intrinsics, np.eye(4), depth_value=z)
    det = Detection(label="chair", score=0.9, bbox_xyxy=np.array([340, 160, 460, 240]), mask=mask)

    e = Ellipsoid.init_from_detection(det, frame)
    assert e is not None
    expected = np.array(
        [(center_px[0] - intrinsics.cx) / intrinsics.fx * z,
         (center_px[1] - intrinsics.cy) / intrinsics.fy * z,
         z]
    )
    assert np.linalg.norm(e.center - expected) < 0.05
    assert e.axes[0] == pytest.approx(semi_px[0] * z / intrinsics.fx, rel=0.15)


def test_ellipse_from_mask_recovers_shape():
    mask = draw_ellipse_mask(480, 640, (320, 240), (80, 30))
    ell = ellipse_from_mask(mask)
    assert ell is not None
    assert np.allclose(ell.mu, [320, 240], atol=1.0)
    semi, _ = ell.axes_angle()
    assert semi[0] == pytest.approx(80, rel=0.05)
    assert semi[1] == pytest.approx(30, rel=0.1)


def test_projection_from_side_view(intrinsics):
    """Camera off to the side still sees the object at the right pixel."""
    obj = np.array([2.0, 0.0, 2.0])
    T_wc = make_camera(position=[0.0, 0.0, 0.0], look_at=obj)
    e = Ellipsoid(center=obj, axes=np.array([0.3, 0.3, 0.3]), R=np.eye(3))
    T_cw = np.linalg.inv(T_wc)
    ell = e.project(intrinsics.K(), T_cw)
    assert ell is not None
    # Object is straight ahead of this camera -> principal point
    assert np.allclose(ell.mu, [intrinsics.cx, intrinsics.cy], atol=1.5)
