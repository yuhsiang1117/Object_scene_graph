"""Multi-view ellipsoid refinement.

Minimizes a 2nd-order-Wasserstein-style discrepancy between the projected
and observed 2D ellipses over all recorded keyframes (VOOM objective).
Implemented with scipy.optimize.least_squares over 9 parameters per object
(center 3, log semi-axes 3, rotation vector 3) — the problem is tiny, so a
general graph optimizer (g2o) is unnecessary.

Residual per observation (5-dim): [dmu_x, dmu_y, vech(sqrtm(S_proj) - sqrtm(S_obs))]
— the Bures metric surrogate for W2 between zero-mean Gaussians.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.optimize import least_squares

from ..core.geometry import matrix_to_rotvec, rotvec_to_matrix, sqrtm_2x2_spd
from .association import Observation, ObjectTrack
from .ellipsoid import MAX_AXIS_M, MIN_AXIS_M, Ellipsoid

# Pixel-scale weighting so center and shape residuals are comparable.
_SHAPE_WEIGHT = 1.0
_CENTER_WEIGHT = 1.0


def _params_from(e: Ellipsoid) -> np.ndarray:
    return np.concatenate([e.center, np.log(np.clip(e.axes, MIN_AXIS_M, MAX_AXIS_M)), matrix_to_rotvec(e.R)])


def _ellipsoid_from(params: np.ndarray) -> Ellipsoid:
    return Ellipsoid(
        center=params[:3].copy(),
        axes=np.clip(np.exp(params[3:6]), MIN_AXIS_M, MAX_AXIS_M),
        R=rotvec_to_matrix(params[6:9]),
    )


def residuals(params: np.ndarray, observations: List[Observation]) -> np.ndarray:
    e = _ellipsoid_from(params)
    res = []
    for obs in observations:
        proj = e.project(obs.K, obs.T_cw)
        if proj is None:
            res.extend([1e3] * 5)  # heavily penalize configurations that leave the view
            continue
        dmu = (proj.mu - obs.mu) * _CENTER_WEIGHT
        ds = (sqrtm_2x2_spd(proj.cov) - sqrtm_2x2_spd(obs.cov)) * _SHAPE_WEIGHT
        res.extend([dmu[0], dmu[1], ds[0, 0], ds[1, 1], ds[0, 1]])
    return np.asarray(res)


class WassersteinRefiner:
    def __init__(self, max_nfev: int = 50) -> None:
        self.max_nfev = max_nfev

    def refine(self, track: ObjectTrack) -> Optional[Ellipsoid]:
        obs = track.observations
        if len(obs) < 3:
            return None
        x0 = _params_from(track.ellipsoid)
        r0 = residuals(x0, obs)
        try:
            sol = least_squares(
                residuals, x0, args=(obs,), method="trf", max_nfev=self.max_nfev, x_scale="jac"
            )
        except Exception:
            return None
        # Reject steps that blow up the residual (degenerate geometry).
        if not np.isfinite(sol.x).all() or sol.cost > 0.5 * float(r0 @ r0) + 1e3:
            return None
        refined = _ellipsoid_from(sol.x)
        if not np.isfinite(refined.center).all():
            return None
        return refined
