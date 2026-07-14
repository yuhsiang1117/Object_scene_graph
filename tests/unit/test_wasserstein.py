from __future__ import annotations

import numpy as np

from osg.core.geometry import sqrtm_2x2_spd
from osg.objects.association import Observation
from osg.objects.ellipsoid import Ellipsoid
from osg.objects.optimization import WassersteinRefiner, residuals, _params_from
from osg.objects.association import ObjectTrack

from .conftest import make_camera


def test_sqrtm_2x2():
    M = np.array([[4.0, 0.0], [0.0, 9.0]])
    S = sqrtm_2x2_spd(M)
    assert np.allclose(S @ S, M)
    M2 = np.array([[5.0, 2.0], [2.0, 3.0]])
    S2 = sqrtm_2x2_spd(M2)
    assert np.allclose(S2 @ S2, M2, atol=1e-8)


def _observations_of(e: Ellipsoid, K: np.ndarray, cameras) -> list:
    obs = []
    for i, T_wc in enumerate(cameras):
        T_cw = np.linalg.inv(T_wc)
        ell = e.project(K, T_cw)
        assert ell is not None
        obs.append(Observation(frame_id=i, mu=ell.mu, cov=ell.cov, K=K, T_cw=T_cw,
                               mean_depth=e.mean_depth_at(T_cw)))
    return obs


def test_zero_residual_for_ground_truth(intrinsics):
    gt = Ellipsoid(center=np.array([0.5, 0.2, 3.0]), axes=np.array([0.4, 0.3, 0.35]), R=np.eye(3))
    cameras = [
        make_camera([0, 0, 0], gt.center),
        make_camera([1.5, 0, 0.5], gt.center),
        make_camera([-1.0, 0.3, 0.5], gt.center),
    ]
    obs = _observations_of(gt, intrinsics.K(), cameras)
    r = residuals(_params_from(gt), obs)
    assert np.abs(r).max() < 1e-4


def test_refine_recovers_center(intrinsics):
    """Perturbed init + 5 synthetic views -> center error < 5 cm."""
    gt = Ellipsoid(center=np.array([0.5, 0.2, 3.0]), axes=np.array([0.4, 0.3, 0.35]), R=np.eye(3))
    cameras = [
        make_camera([0, 0, 0], gt.center),
        make_camera([1.5, 0, 0.5], gt.center),
        make_camera([-1.2, 0.2, 0.3], gt.center),
        make_camera([0.8, -0.4, 0.0], gt.center),
        make_camera([-0.5, 0.5, 1.0], gt.center),
    ]
    obs = _observations_of(gt, intrinsics.K(), cameras)

    init = Ellipsoid(
        center=gt.center + np.array([0.25, -0.15, 0.3]),
        axes=gt.axes * 1.5,
        R=np.eye(3),
    )
    track = ObjectTrack(id=0, label="chair", ellipsoid=init, observations=obs)
    refined = WassersteinRefiner(max_nfev=100).refine(track)
    assert refined is not None
    assert np.linalg.norm(refined.center - gt.center) < 0.05
