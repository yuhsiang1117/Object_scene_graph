"""Ellipsoid object representation (dual quadric), adapted from VOOM.

An ellipsoid with center t, rotation R and semi-axes (a, b, c) has the dual
quadric Q* = Z diag(a^2, b^2, c^2, -1) Z^T with Z = [[R, t], [0, 1]].
Projected through P = K [R_cw | t_cw] it yields a dual conic
C* = P Q* P^T which, normalized so its bottom-right entry is -1, reads
[[Sigma - mu mu^T, -mu], [-mu^T, -1]].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..core.geometry import Ellipse2D, backproject_pixel, ellipse_from_mask
from ..core.types import Detection, FrameData

MIN_AXIS_M = 0.01
MAX_AXIS_M = 3.0


@dataclass
class Ellipsoid:
    center: np.ndarray  # (3,)
    axes: np.ndarray  # (3,) semi-axes, meters
    R: np.ndarray  # (3, 3)

    def Q_dual(self) -> np.ndarray:
        Z = np.eye(4)
        Z[:3, :3] = self.R
        Z[:3, 3] = self.center
        D = np.diag([self.axes[0] ** 2, self.axes[1] ** 2, self.axes[2] ** 2, -1.0])
        return Z @ D @ Z.T

    def project(self, K: np.ndarray, T_cw: np.ndarray) -> Optional[Ellipse2D]:
        """Project to an image ellipse; None if behind camera or degenerate."""
        center_c = T_cw[:3, :3] @ self.center + T_cw[:3, 3]
        if center_c[2] < 0.1:  # cheirality
            return None
        P = K @ T_cw[:3, :]
        C = P @ self.Q_dual() @ P.T
        if abs(C[2, 2]) < 1e-9:
            return None
        C = C / (-C[2, 2])  # bottom-right -> -1
        mu = -C[:2, 2]
        cov = C[:2, :2] + np.outer(mu, mu)
        cov = 0.5 * (cov + cov.T)
        vals = np.linalg.eigvalsh(cov)
        if vals[0] <= 1e-6 or vals[1] > 1e8:
            return None
        return Ellipse2D(mu=mu, cov=cov)

    @staticmethod
    def init_from_detection(
        det: Detection, frame: FrameData, n_depth_samples: int = 50, rng: Optional[np.random.Generator] = None
    ) -> Optional["Ellipsoid"]:
        """Paper recipe: 2D ellipse from mask moments + averaged sampled depth,
        back-projected to 3D; semi-axes scaled by z/f; R aligned to camera."""
        ellipse = ellipse_from_mask(det.mask)
        if ellipse is None:
            return None
        ys, xs = np.nonzero(det.mask)
        d = frame.depth[ys, xs]
        valid = d > 1e-3
        if valid.sum() < 8:
            return None
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(np.nonzero(valid)[0], size=min(n_depth_samples, int(valid.sum())), replace=False)
        z = float(np.mean(d[idx]))
        center = backproject_pixel(ellipse.mu[0], ellipse.mu[1], z, frame.intrinsics, frame.T_wc)

        semi_px, _ = ellipse.axes_angle()
        a = float(np.clip(semi_px[0] * z / frame.intrinsics.fx, MIN_AXIS_M, MAX_AXIS_M))
        b = float(np.clip(semi_px[1] * z / frame.intrinsics.fy, MIN_AXIS_M, MAX_AXIS_M))
        c = float(np.clip(0.5 * (a + b), MIN_AXIS_M, MAX_AXIS_M))
        # Axes initially aligned with the camera frame (x, y image axes, z depth)
        return Ellipsoid(center=center, axes=np.array([a, b, c]), R=frame.T_wc[:3, :3].copy())

    def mean_depth_at(self, T_cw: np.ndarray) -> float:
        return float((T_cw[:3, :3] @ self.center + T_cw[:3, 3])[2])

    # --------------------------------------------------------------- extents
    # R is initialised to the CAMERA rotation and then refined, so it is never
    # world-axis-aligned: reading axes[HEIGHT_AXIS] to answer "how tall is this"
    # is wrong by however much the object is tilted in the map frame. The
    # shape matrix below is the only correct route, and three consumers share
    # it -- container surface heights, the C1 depth band, the C3 affordance
    # test (docs/DYNAMIC_SCENES.md).

    def shape_matrix(self) -> np.ndarray:
        """Q = R diag(a^2, b^2, c^2) R^T -- the ellipsoid's 3x3 shape matrix.

        The surface is {x : (x - t)^T Q^-1 (x - t) = 1}, so Q plays the role of
        a covariance and every extent question below is a statement about it.
        """
        return self.R @ np.diag(self.axes ** 2) @ self.R.T

    def world_extent(self, axis_unit: np.ndarray) -> float:
        """Half-extent along a world direction d: sqrt(d^T Q d).

        The ellipsoid's support function. For an axis-aligned ellipsoid and
        d = e_y this is just the y semi-axis; for a tilted one it is the real
        silhouette half-height, which is what a support-surface test needs.
        """
        d = np.asarray(axis_unit, dtype=float)
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            return 0.0
        d = d / n
        return float(np.sqrt(max(d @ self.shape_matrix() @ d, 0.0)))

    def ground_footprint(self, plane: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """(center_xy, cov_xy) of the ellipsoid's shadow on the ground plane.

        The orthogonal projection of an ellipsoid onto a coordinate plane is the
        ellipse whose shape matrix is the corresponding 2x2 block of Q -- the
        same marginalisation rule as for a Gaussian covariance. A point p is
        inside iff (p - mu)^T cov^-1 (p - mu) <= 1, and the area is
        pi * sqrt(det(cov)).
        """
        idx = np.array(plane, dtype=int)
        Q = self.shape_matrix()
        return self.center[idx].copy(), Q[np.ix_(idx, idx)].copy()
