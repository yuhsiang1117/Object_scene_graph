"""Ellipsoid object representation (dual quadric), adapted from VOOM.

An ellipsoid with center t, rotation R and semi-axes (a, b, c) has the dual
quadric Q* = Z diag(a^2, b^2, c^2, -1) Z^T with Z = [[R, t], [0, 1]].
Projected through P = K [R_cw | t_cw] it yields a dual conic
C* = P Q* P^T which, normalized so its bottom-right entry is -1, reads
[[Sigma - mu mu^T, -mu], [-mu^T, -1]].
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..core.geometry import Ellipse2D, backproject_pixel, det_ellipse, mask_pixels
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

    def project(
        self, K: np.ndarray, T_cw: np.ndarray,
        Q: Optional[np.ndarray] = None, P: Optional[np.ndarray] = None,
    ) -> Optional[Ellipse2D]:
        """Project to an image ellipse; None if behind camera or degenerate.

        `Q` (this ellipsoid's `Q_dual()`) and `P` (`K @ T_cw[:3, :]`) may be
        passed precomputed by a caller that projects one ellipsoid through many
        cameras, or one camera many times -- the refiner does both, ~175
        residual evaluations of up to ten views each. They are the same
        expressions, so the result is bit-identical."""
        center_c = T_cw[:3, :3] @ self.center + T_cw[:3, 3]
        if center_c[2] < 0.1:  # cheirality
            return None
        if P is None:
            P = K @ T_cw[:3, :]
        if Q is None:
            Q = self.Q_dual()
        C = P @ Q @ P.T
        if abs(C[2, 2]) < 1e-9:
            return None
        C = C / (-C[2, 2])  # bottom-right -> -1
        mu = -C[:2, 2]
        cov = C[:2, :2] + mu[:, None] * mu[None, :]  # np.outer, without its call
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
        ellipse = det_ellipse(det)
        if ellipse is None:
            return None
        ys, xs = mask_pixels(det)
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

    @staticmethod
    def init_from_points(pts, frame, ecfg, yaw: Optional[float] = None) -> Optional["Ellipsoid"]:
        """One view's cleaned depth points (objects/point_buffer.clean_points)
        fitted in a gravity-aligned frame: up is world +y, forward is the
        camera's view direction flattened onto the floor.

        Lateral and vertical extents are what the surface spans (percentiles,
        rescaled as for a uniform spread). The axis along the view cannot be
        measured from one side, so its half-length follows
        `ecfg.depth_axis`, and the centre sits that far BEHIND the nearest
        visible surface -- the visible surface is the front of the object,
        not its middle, which is where the moments initialisation put it.
        """
        if pts is None or len(pts.world) < 8:
            return None
        forward = frame.T_wc[:3, 2]
        if yaw is not None:
            forward = snap_yaw(forward, yaw)
        R = gravity_frame(forward, frame.T_wc[:3, 0])
        return _fit_in_frame(pts.world, R, ecfg, one_sided=True)

    @staticmethod
    def fit_points(world: np.ndarray, view_dirs: np.ndarray, ecfg,
                   yaw: Optional[float] = None) -> Optional["Ellipsoid"]:
        """Many views' buffered points: yaw from the horizontal principal axis,
        extents per axis. When every view saw the object from about one side
        (horizontal spread under 60 deg) the far side is completed exactly as
        `init_from_points` completes it, along the mean view direction."""
        world = np.asarray(world, float)
        if len(world) < 8:
            return None
        horiz = world[:, [0, 2]] - world[:, [0, 2]].mean(axis=0)
        cov = horiz.T @ horiz / max(len(horiz) - 1, 1)
        vals, vecs = np.linalg.eigh(cov)
        major = vecs[:, int(np.argmax(vals))]
        dirs = np.asarray(view_dirs, float).reshape(-1, 2)
        one_sided = True
        mean_dir = np.array([major[0], 0.0, major[1]])
        if len(dirs):
            m = dirs.mean(axis=0)
            spread = float(np.max(np.arccos(np.clip(
                dirs @ (m / max(np.linalg.norm(m), 1e-9)), -1.0, 1.0))))
            one_sided = 2.0 * spread < np.radians(60.0) and np.linalg.norm(m) > 1e-6
            if np.linalg.norm(m) > 1e-6:
                mean_dir = np.array([m[0], 0.0, m[1]])
        if yaw is not None:
            # Manhattan: the object's axes are the room's, not its points'
            # principal axis or the line of sight (S78).
            mean_dir = snap_yaw(mean_dir, yaw)
            major = snap_yaw(np.array([major[0], 0.0, major[1]]), yaw)[[0, 2]]
        if one_sided:
            R = gravity_frame(mean_dir, None)
        else:
            a = np.array([major[0], 0.0, major[1]])
            up = np.array([0.0, 1.0, 0.0])
            R = np.stack([a, up, np.cross(a, up)], axis=1)
        return _fit_in_frame(world, R, ecfg, one_sided=one_sided)

    def mean_depth_at(self, T_cw: np.ndarray) -> float:
        return float((T_cw[:3, :3] @ self.center + T_cw[:3, 3])[2])

    # --------------------------------------------------------------- extents
    # R is initialised to the CAMERA rotation and then refined, so it is never
    # world-axis-aligned: reading axes[HEIGHT_AXIS] to answer "how tall is this"
    # is wrong by however much the object is tilted in the map frame. The
    # shape matrix below is the only correct route, and three consumers share
    # it -- container surface heights, the C1 depth band, the C3 affordance
    # test (docs/history/DYNAMIC_SCENES.md).

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


def snap_yaw(direction: np.ndarray, yaw: float) -> np.ndarray:
    """The horizontal unit vector at `yaw + k * 90 deg` (radians, measured as
    atan2(z, x)) closest to `direction`'s horizontal part."""
    d = np.asarray(direction, float)
    a = math.atan2(float(d[2]), float(d[0])) if np.hypot(d[0], d[2]) > 1e-9 else float(yaw)
    k = round((a - float(yaw)) / (math.pi / 2))
    s = float(yaw) + k * math.pi / 2
    return np.array([math.cos(s), 0.0, math.sin(s)])


def gravity_frame(forward: np.ndarray, right_hint: Optional[np.ndarray]) -> np.ndarray:
    """Columns (lateral, up, forward): up is world +y, forward is `forward`
    with its vertical part removed. Looking straight down, `right_hint`
    (the camera's x axis) fixes the yaw instead."""
    up = np.array([0.0, 1.0, 0.0])
    f = np.asarray(forward, float).copy()
    f[1] = 0.0
    if np.linalg.norm(f) < 1e-6 and right_hint is not None:
        r = np.asarray(right_hint, float).copy()
        r[1] = 0.0
        f = np.cross(r, up)
    if np.linalg.norm(f) < 1e-6:
        f = np.array([0.0, 0.0, 1.0])
    f /= np.linalg.norm(f)
    lat = np.cross(up, f)
    return np.stack([lat, up, f], axis=1)


def _fit_in_frame(world: np.ndarray, R: np.ndarray, ecfg, one_sided: bool) -> Optional[Ellipsoid]:
    mean = world.mean(axis=0)
    q = (world - mean) @ R
    p = float(ecfg.extent_pct)
    lo = np.percentile(q, p, axis=0)
    hi = np.percentile(q, 100.0 - p, axis=0)
    scale = max(1.0 - 2.0 * p / 100.0, 1e-3)
    pad = float(getattr(ecfg, "extent_pad_m", 0.0))
    half = (hi - lo) / 2.0 / scale + pad
    centre = (hi + lo) / 2.0
    min_ax = float(ecfg.min_axis_m)
    if one_sided:
        vis = max(float(half[2]), min_ax)
        rule = str(ecfg.depth_axis)
        if rule == "lateral":
            depth_half = max(float(half[0]), vis)
        elif rule == "mean_ab":
            depth_half = max(0.5 * float(half[0] + half[1]), vis)
        elif rule == "thickness":
            depth_half = vis
        else:
            raise ValueError(f"scene_graph.ellipsoid.depth_axis: unknown {rule!r}")
        near = float(lo[2]) - pad
        centre[2] = near + depth_half
        half[2] = depth_half
    half = np.clip(half, max(MIN_AXIS_M, min_ax), MAX_AXIS_M)
    if not np.all(np.isfinite(half)) or not np.all(np.isfinite(centre)):
        return None
    return Ellipsoid(center=mean + R @ centre, axes=half, R=R.copy())
