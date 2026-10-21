"""Geometry helpers: back-projection, mask-moment ellipses, rotations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def backproject(
    depth: np.ndarray,
    intrinsics,
    T_wc: np.ndarray,
    mask: Optional[np.ndarray] = None,
    stride: int = 1,
    max_depth: float = 10.0,
) -> np.ndarray:
    """Back-project depth pixels to world points (N, 3). OpenCV convention."""
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[::stride, ::stride]
    valid = (d > 1e-3) & (d < max_depth)
    if mask is not None:
        valid &= mask[::stride, ::stride]
    u, v, d = us[valid], vs[valid], d[valid]
    x = (u - intrinsics.cx) / intrinsics.fx * d
    y = (v - intrinsics.cy) / intrinsics.fy * d
    pts_c = np.stack([x, y, d], axis=1)
    return pts_c @ T_wc[:3, :3].T + T_wc[:3, 3]


def backproject_horizontal(
    depth: np.ndarray,
    intrinsics,
    T_wc: np.ndarray,
    stride: int = 1,
    max_depth: float = 10.0,
    min_up: float = 0.7,
    up_axis: int = 1,
) -> np.ndarray:
    """World points (N, 3) of the depth pixels that lie on a roughly HORIZONTAL
    surface: the surface normal, from central differences of the back-projected
    pixel grid, within `arccos(min_up)` of the world's up axis -- 45 degrees by
    default, because a scanned HM3D flight can be a 30-37 degree ramp.

    A floor, a tread or a table top passes; a wall, a riser face or a railing
    does not. The normal is taken twice, from the forward and from the backward
    neighbours, and both must point up: a pixel at a corner -- a tread meeting
    the next riser, a floor meeting a wall -- has one horizontal side and one
    vertical, and a central difference across it can look level. Border pixels
    and pixels with an invalid neighbour are dropped.
    """
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[::stride, ::stride].astype(np.float64)
    valid = (d > 1e-3) & (d < max_depth)
    if d.shape[0] < 3 or d.shape[1] < 3:
        return np.zeros((0, 3))
    x = (us - intrinsics.cx) / intrinsics.fx * d
    y = (vs - intrinsics.cy) / intrinsics.fy * d
    pts = np.stack([x, y, d], axis=-1) @ T_wc[:3, :3].T + T_wc[:3, 3]
    centre = pts[1:-1, 1:-1]
    ok = valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1]
    for across, down in ((pts[1:-1, 2:] - centre, pts[2:, 1:-1] - centre),
                         (centre - pts[1:-1, :-2], centre - pts[:-2, 1:-1])):
        normal = np.cross(across, down)
        length = np.linalg.norm(normal, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            up = np.abs(normal[..., up_axis]) / length
        ok &= (length > 0) & (up >= float(min_up))
    return centre[ok]


def backproject_pixel(u: float, v: float, d: float, intrinsics, T_wc: np.ndarray) -> np.ndarray:
    p = np.array(
        [(u - intrinsics.cx) / intrinsics.fx * d, (v - intrinsics.cy) / intrinsics.fy * d, d]
    )
    return T_wc[:3, :3] @ p + T_wc[:3, 3]


@dataclass
class Ellipse2D:
    """Image ellipse: points x with (x - mu)^T Sigma^{-1} (x - mu) = 1."""

    mu: np.ndarray  # (2,)
    cov: np.ndarray  # (2, 2) SPD

    @property
    def area(self) -> float:
        return float(np.pi * np.sqrt(max(np.linalg.det(self.cov), 0.0)))

    def bbox(self) -> np.ndarray:
        """Axis-aligned bbox [x1, y1, x2, y2] enclosing the ellipse."""
        half = np.sqrt(np.maximum(np.diag(self.cov), 0.0))
        return np.array(
            [self.mu[0] - half[0], self.mu[1] - half[1], self.mu[0] + half[0], self.mu[1] + half[1]]
        )

    def axes_angle(self) -> Tuple[np.ndarray, float]:
        vals, vecs = np.linalg.eigh(self.cov)
        semi = np.sqrt(np.maximum(vals, 0.0))
        angle = float(np.arctan2(vecs[1, -1], vecs[0, -1]))
        return semi[::-1], angle  # major first


def ellipse_from_mask(mask: np.ndarray) -> Optional[Ellipse2D]:
    """Fit an Ellipse2D to a boolean mask via image moments.

    For a uniformly filled ellipse the second central moment equals Sigma / 4,
    so we scale by 4 to recover the boundary ellipse.
    """
    ys, xs = np.nonzero(mask)
    if xs.size < 8:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    mu = pts.mean(axis=0)
    cov = 4.0 * np.cov(pts.T)
    cov = np.atleast_2d(cov)
    # Guard against degenerate (thin) masks
    cov += np.eye(2) * 1.0
    return Ellipse2D(mu=mu, cov=cov)


def rotvec_to_matrix(rv: np.ndarray) -> np.ndarray:
    """Rodrigues formula."""
    theta = np.linalg.norm(rv)
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos)
    if theta < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-12:  # theta ~ pi
        # Fallback: eigenvector of R for eigenvalue 1
        vals, vecs = np.linalg.eig(R)
        axis = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
        return axis / np.linalg.norm(axis) * theta
    return axis / n * theta


def quat_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Unit quaternion (w, x, y, z) to rotation matrix."""
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def bresenham(r0: int, c0: int, r1: int, c1: int):
    """Canonical integer Bresenham: yields (r, c) from (r0, c0) to (r1, c1)
    inclusive. Guaranteed to terminate at the endpoint."""
    dr = abs(r1 - r0)
    dc = -abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr + dc
    while True:
        yield r0, c0
        if r0 == r1 and c0 == c1:
            return
        e2 = 2 * err
        if e2 >= dc:
            err += dc
            r0 += sr
        if e2 <= dr:
            err += dr
            c0 += sc


def sqrtm_2x2_spd(M: np.ndarray) -> np.ndarray:
    """Closed-form principal square root of a 2x2 SPD matrix."""
    det = max(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0], 0.0)
    s = np.sqrt(det)
    tr = M[0, 0] + M[1, 1]
    denom = np.sqrt(max(tr + 2.0 * s, 1e-12))
    return (M + s * np.eye(2)) / denom
