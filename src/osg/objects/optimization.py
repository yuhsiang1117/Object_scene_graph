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

from typing import List, NamedTuple, Optional

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


_EYE2 = np.eye(2)


class ViewConstants(NamedTuple):
    """Per view, what no parameter changes, stacked over the views."""
    P: np.ndarray         # (n, 3, 4) the camera matrix K [R|t]
    R: np.ndarray         # (n, 3, 3) world -> camera rotation
    t: np.ndarray         # (n, 3)    and translation
    mu: np.ndarray        # (n, 2)    the observed ellipse's centre
    sqrt_cov: np.ndarray  # (n, 2, 2) and its covariance's square root


def view_constants(observations: List[Observation]) -> ViewConstants:
    """The least-squares solver evaluates the residual ~175 times per refine
    (a 9-parameter numeric Jacobian per step), and recomputing these each time
    was a fifth of the object layer's cost."""
    return ViewConstants(
        P=np.stack([obs.K @ obs.T_cw[:3, :] for obs in observations]),
        R=np.stack([obs.T_cw[:3, :3] for obs in observations]),
        t=np.stack([obs.T_cw[:3, 3] for obs in observations]),
        mu=np.stack([obs.mu for obs in observations]),
        sqrt_cov=np.stack([sqrtm_2x2_spd(obs.cov) for obs in observations]),
    )


def residuals(
    params: np.ndarray, observations: List[Observation],
    consts: Optional[ViewConstants] = None,
) -> np.ndarray:
    """Every view at once, bit-identical to projecting one view at a time
    (`Ellipsoid.project`, then `sqrtm_2x2_spd` on the projection): a stacked
    matmul makes the same BLAS call per view, every other operation is the
    same elementwise one in the same order, and `eigvalsh` is asked about
    exactly the views `project` would ask about, so even its failures are the
    same (tests/unit/test_hot_path_equivalence.py). Half the time of the loop."""
    if not observations:
        return np.asarray([])
    e = _ellipsoid_from(params)
    c = view_constants(observations) if consts is None else consts
    Q = e.Q_dual()
    n = len(c.P)
    with np.errstate(all="ignore"):  # rows that fail below are discarded, as project discards them
        centre_c = c.R @ e.center + c.t
        C = c.P @ Q @ c.P.transpose(0, 2, 1)
        c22 = C[:, 2, 2]
        ok = ~(centre_c[:, 2] < 0.1) & ~(np.abs(c22) < 1e-9)  # cheirality; degenerate conic
        C = C / (-c22)[:, None, None]  # bottom-right -> -1
        mu = -C[:, :2, 2]
        cov = C[:, :2, :2] + mu[:, :, None] * mu[:, None, :]
        cov = 0.5 * (cov + cov.transpose(0, 2, 1))
    seen = np.flatnonzero(ok)
    if len(seen):
        vals = np.linalg.eigvalsh(cov[seen])
        ok[seen[(vals[:, 0] <= 1e-6) | (vals[:, 1] > 1e8)]] = False
    # Heavily penalize configurations that leave the view.
    res = np.full((n, 5), 1e3)
    k = np.flatnonzero(ok)
    if len(k):
        M = cov[k]
        with np.errstate(all="ignore"):
            # sqrtm_2x2_spd, a view per row.
            s = np.sqrt(np.maximum(M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0], 0.0))
            denom = np.sqrt(np.maximum(M[:, 0, 0] + M[:, 1, 1] + 2.0 * s, 1e-12))
            sqrt_proj = (M + s[:, None, None] * _EYE2) / denom[:, None, None]
        dmu = (mu[k] - c.mu[k]) * _CENTER_WEIGHT
        ds = (sqrt_proj - c.sqrt_cov[k]) * _SHAPE_WEIGHT
        res[k, 0], res[k, 1] = dmu[:, 0], dmu[:, 1]
        res[k, 2], res[k, 3], res[k, 4] = ds[:, 0, 0], ds[:, 1, 1], ds[:, 0, 1]
    return res.reshape(-1)


class WassersteinRefiner:
    def __init__(self, max_nfev: int = 50, max_center_move_m: float = 0.5) -> None:
        self.max_nfev = max_nfev
        # VOOM/OA-SLAM reject: the reprojection objective constrains depth only
        # through parallax, so with the narrow view arc typical of ObjectNav a
        # refine step can slide the centre metres along the ray while keeping 2D
        # error low. Measured (analyze_refine_accuracy) that this degrades the
        # 3D centre more often than it helps (+1..+5 m tails). Reject any refine
        # that moves the centre further than this from its pre-refine value.
        self.max_center_move_m = max_center_move_m

    def refine(self, track: ObjectTrack, max_obs: int = 10) -> Optional[Ellipsoid]:
        obs = track.observations[-max_obs:]  # bound the residual count
        if len(obs) < 3:
            return None
        x0 = _params_from(track.ellipsoid)
        consts = view_constants(obs)
        r0 = residuals(x0, obs, consts)
        try:
            sol = least_squares(
                residuals, x0, args=(obs, consts), method="trf", max_nfev=self.max_nfev,
                x_scale="jac",
            )
        except Exception:
            return None
        # Reject steps that blow up the residual (degenerate geometry).
        if not np.isfinite(sol.x).all() or sol.cost > 0.5 * float(r0 @ r0) + 1e3:
            return None
        refined = _ellipsoid_from(sol.x)
        if not np.isfinite(refined.center).all():
            return None
        # Reject centre jumps the parallax can't justify (see __init__).
        if self.max_center_move_m > 0.0:
            if float(np.linalg.norm(refined.center - track.ellipsoid.center)) > self.max_center_move_m:
                return None
        return refined
