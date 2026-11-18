"""Depth points of one detection, cleaned; and a per-track voxel buffer of them.

The moments initialisation (`Ellipsoid.init_from_detection`) reads the mask's
depth as one number -- the mean of 50 random pixels -- and a mask's edge is
exactly where segmentation bleeds onto whatever is behind the object, so that
mean is pulled back by the background. `clean_points` keeps the pixels that
are the object's own surface:

  * erosion by `erode_px`, which removes the bleed band itself;
  * flying pixels: a pixel whose depth jumps by more than `flying_rel` of
    itself to a 4-neighbour straddles an occlusion boundary;
  * a robust depth band, `mad_k` scaled MADs around the median, which drops
    what erosion left of the background.

`VoxelBuffer` accumulates those points across views, one centroid per voxel,
so memory is bounded by the object's surface area, not by how long the agent
looked at it. `view_ok` is the "specific viewing range" gate: only an
untruncated view, in range, with enough valid depth, may add to it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..core.geometry import mask_pixels


@dataclass
class CleanPoints:
    world: np.ndarray        # (N, 3)
    cam: np.ndarray          # (N, 3) camera frame (OpenCV)
    valid_frac: float        # kept / mask pixels sampled
    median_depth: float


def truncated(det, width: int, height: int, border_px: int) -> bool:
    x1, y1, x2, y2 = (float(v) for v in det.bbox_xyxy)
    return (x1 <= border_px or y1 <= border_px
            or x2 >= width - border_px or y2 >= height - border_px)


def _eroded_pixels(det, erode_px: int):
    ys, xs = mask_pixels(det)
    if erode_px <= 0 or len(ys) == 0:
        return ys, xs
    import cv2

    m = np.asarray(det.mask, dtype=np.uint8)
    er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=int(erode_px))
    eys, exs = np.nonzero(er)
    # A mask thinner than the erosion would vanish; keep it rather than lose it.
    return (eys, exs) if len(eys) >= 8 else (ys, xs)


def clean_points(det, frame, ecfg) -> Optional[CleanPoints]:
    ys, xs = _eroded_pixels(det, int(ecfg.erode_px))
    n_all = len(ys)
    if n_all == 0:
        return None
    step = max(1, int(np.ceil(n_all / float(max(int(ecfg.max_pixels), 1)))))
    ys, xs = ys[::step], xs[::step]
    depth = frame.depth
    d = depth[ys, xs].astype(np.float64)
    keep = d > 1e-3
    rel = float(ecfg.flying_rel)
    if rel > 0.0:
        h, w = depth.shape
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny = np.clip(ys + dy, 0, h - 1)
            nx = np.clip(xs + dx, 0, w - 1)
            nd = depth[ny, nx].astype(np.float64)
            keep &= ~((nd > 1e-3) & (np.abs(nd - d) > rel * np.maximum(d, 1e-3)))
    if keep.sum() < 8:
        return None
    med = float(np.median(d[keep]))
    k = float(ecfg.mad_k)
    if k > 0.0:
        # An object has depth of its own -- a table's top recedes from its
        # front edge -- so the band is never narrower than the object's own
        # size at that range, read off the box. What it drops is what lies
        # well beyond the object: the wall behind it, seen through the bleed.
        mad = float(np.median(np.abs(d[keep] - med))) * 1.4826
        x1, y1, x2, y2 = (float(v) for v in det.bbox_xyxy)
        intr = frame.intrinsics
        size_m = max((x2 - x1) / intr.fx, (y2 - y1) / intr.fy) * med
        band = max(k * mad, 0.75 * size_m, 0.05)
        keep &= np.abs(d - med) <= band
    if keep.sum() < 8:
        return None
    u, v, z = xs[keep].astype(np.float64), ys[keep].astype(np.float64), d[keep]
    intr = frame.intrinsics
    cam = np.stack([(u - intr.cx) / intr.fx * z, (v - intr.cy) / intr.fy * z, z], axis=1)
    world = cam @ frame.T_wc[:3, :3].T + frame.T_wc[:3, 3]
    return CleanPoints(world=world, cam=cam, valid_frac=float(keep.sum()) / float(len(d)),
                       median_depth=float(np.median(z)))


def largest_cluster(pts: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """The largest DBSCAN cluster of `pts`; all of them when none forms (too
    few points for a core). ConceptGraphs' denoise step."""
    from sklearn.cluster import DBSCAN

    pts = np.asarray(pts, float)
    if len(pts) < max(int(min_pts), 1):
        return pts
    lab = DBSCAN(eps=float(eps), min_samples=int(min_pts)).fit_predict(pts)
    good = lab[lab >= 0]
    if not len(good):
        return pts
    return pts[lab == np.bincount(good).argmax()]


def anchored_cluster(pts: np.ndarray, eps: float, min_pts: int,
                     anchor: np.ndarray) -> Optional[np.ndarray]:
    """The DBSCAN cluster of `pts` holding the clustered point nearest to
    `anchor`, or None when no cluster forms. `fuse_report`: the anchor is the
    track's own ellipsoid, so a buffer that also took a far detection of
    another object (2D association matches in the image, not in depth) still
    reports the object association placed, not whichever is bigger."""
    from sklearn.cluster import DBSCAN

    pts = np.asarray(pts, float)
    if len(pts) < max(int(min_pts), 1):
        return None
    lab = DBSCAN(eps=float(eps), min_samples=int(min_pts)).fit_predict(pts)
    good = np.nonzero(lab >= 0)[0]
    if not len(good):
        return None
    near = good[int(np.argmin(np.linalg.norm(pts[good] - np.asarray(anchor, float), axis=1)))]
    return pts[lab == lab[near]]


def voxel_centroids(pts: np.ndarray, voxel_m: float) -> np.ndarray:
    if not len(pts):
        return pts
    keys = np.floor(pts / float(voxel_m)).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    sums = np.zeros((int(inv.max()) + 1, 3))
    np.add.at(sums, inv, pts)
    return sums / np.bincount(inv)[:, None]


def fused_points(det, frame, ecfg) -> Optional[np.ndarray]:
    """One view's contribution to a fused buffer (`scene_graph.ellipsoid.fuse`):
    every mask pixel with a real depth (after `fuse_erode_px` of erosion; no
    depth band), the raw points ConceptGraphs fuses -- at most `fuse_max_pixels` of them
    (evenly strided), voxelised, then their largest DBSCAN cluster, which is
    what drops the background a mask bleeds onto."""
    ys, xs = _eroded_pixels(det, int(ecfg.fuse_erode_px))
    if not len(ys):
        return None
    z = frame.depth[ys, xs].astype(np.float64)
    lo, hi = (float(v) for v in ecfg.fuse_depth_range_m)
    ok = (z > lo + 1e-2) & (z < hi - 1e-2)
    ys, xs, z = ys[ok], xs[ok], z[ok]
    cap = int(ecfg.fuse_max_pixels)
    if len(z) > cap:
        sel = np.linspace(0, len(z) - 1, cap).astype(int)
        ys, xs, z = ys[sel], xs[sel], z[sel]
    if len(z) < 8:
        return None
    intr = frame.intrinsics
    cam = np.stack([(xs - intr.cx) / intr.fx * z, (ys - intr.cy) / intr.fy * z, z], axis=1)
    world = cam @ frame.T_wc[:3, :3].T + frame.T_wc[:3, 3]
    pts = largest_cluster(voxel_centroids(world, float(ecfg.fuse_voxel_m)),
                          float(ecfg.fuse_eps_m), int(ecfg.fuse_min_pts))
    return pts if len(pts) else None


def view_ok(det, frame, pts: Optional[CleanPoints], ecfg) -> bool:
    if pts is None:
        return False
    h, w = frame.depth.shape
    if not bool(getattr(ecfg, "buffer_truncated", False)) and truncated(det, w, h, int(ecfg.border_px)):
        return False
    lo, hi = (float(v) for v in ecfg.view_range_m)
    return lo <= pts.median_depth <= hi and pts.valid_frac >= float(ecfg.min_valid_frac)


class VoxelBuffer:
    """Voxel hash of world points: per voxel, the running sum and count."""

    def __init__(self, voxel_m: float = 0.03, cap: int = 6000) -> None:
        self.voxel_m = float(voxel_m)
        self.cap = int(cap)
        self._cells: Dict[Tuple[int, int, int], list] = {}
        self.n_views = 0

    def __len__(self) -> int:
        return len(self._cells)

    def _keys(self, pts: np.ndarray) -> np.ndarray:
        return np.floor(np.asarray(pts, float) / self.voxel_m).astype(np.int64)

    def add(self, pts: np.ndarray) -> None:
        if pts is None or not len(pts):
            return
        self.n_views += 1
        keys = self._keys(pts)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        sums = np.zeros((len(uniq), 3))
        np.add.at(sums, inv, pts)
        counts = np.bincount(inv, minlength=len(uniq))
        for key, s, c in zip(map(tuple, uniq), sums, counts):
            cell = self._cells.get(key)
            if cell is not None:
                cell[0] += s
                cell[1] += int(c)
            elif len(self._cells) < self.cap:
                self._cells[key] = [s.copy(), int(c)]

    def points(self) -> np.ndarray:
        if not self._cells:
            return np.zeros((0, 3))
        return np.stack([s / c for s, c in self._cells.values()])

    def weights(self) -> np.ndarray:
        return np.array([c for _, c in self._cells.values()], dtype=float)

    def contains(self, pts: np.ndarray, dilate: int = 1) -> np.ndarray:
        """Per point: is it in (or within `dilate` voxels of) an occupied voxel?"""
        if not self._cells or not len(pts):
            return np.zeros(len(pts), dtype=bool)
        keys = self._keys(pts)
        occ = self._cells
        out = np.zeros(len(pts), dtype=bool)
        r = range(-dilate, dilate + 1)
        offsets = [(a, b, c) for a in r for b in r for c in r]
        for i, k in enumerate(map(tuple, keys)):
            for o in offsets:
                if (k[0] + o[0], k[1] + o[1], k[2] + o[2]) in occ:
                    out[i] = True
                    break
        return out

    def merge(self, other: "VoxelBuffer") -> None:
        """Fold `other`'s cells into this grid. A key is an index at its own
        buffer's voxel size: one from a buffer at another size (a snapshot's
        compressed summary beside a fresh 3 cm buffer) names a different place
        here, so such cells are re-keyed by their centroid first."""
        items = other._cells.items()
        if other.voxel_m != self.voxel_m:
            rekeyed: Dict[Tuple[int, int, int], list] = {}
            for s, c in other._cells.values():
                cell = rekeyed.setdefault(tuple(self._keys(s / c)), [np.zeros(3), 0])
                cell[0] += s
                cell[1] += int(c)
            items = rekeyed.items()
        for key, (s, c) in items:
            cell = self._cells.get(key)
            if cell is not None:
                cell[0] += s
                cell[1] += c
            elif len(self._cells) < self.cap:
                self._cells[key] = [s.copy(), int(c)]
        self.n_views += other.n_views

    def compress(self, max_voxels: int) -> None:
        """Re-voxelise at doubling sizes until at most `max_voxels` remain.
        The voxel size grows; the summary keeps the mass-weighted centroids."""
        while len(self._cells) > max_voxels:
            pts, w = self.points(), self.weights()
            self.voxel_m *= 2.0
            keys = self._keys(pts)
            cells: Dict[Tuple[int, int, int], list] = {}
            for key, p, c in zip(map(tuple, keys), pts, w):
                cell = cells.setdefault(key, [np.zeros(3), 0])
                cell[0] += p * c
                cell[1] += int(c)
            self._cells = cells

    # ------------------------------------------------------------ persistence

    def to_record(self) -> dict:
        return {"voxel_m": self.voxel_m, "n_views": self.n_views,
                "points": [[float(v) for v in p] for p in self.points()],
                "counts": [int(c) for c in self.weights()]}

    @classmethod
    def from_record(cls, rec: dict, cap: int = 6000) -> "VoxelBuffer":
        buf = cls(voxel_m=float(rec.get("voxel_m", 0.03)), cap=cap)
        pts = np.asarray(rec.get("points", []), float).reshape(-1, 3)
        counts = rec.get("counts", [1] * len(pts))
        for key, p, c in zip(map(tuple, buf._keys(pts)), pts, counts):
            cell = buf._cells.setdefault(key, [np.zeros(3), 0])
            cell[0] += p * c
            cell[1] += int(c)
        buf.n_views = int(rec.get("n_views", 0))
        return buf


class ManhattanFrame:
    """The room's dominant horizontal direction, modulo 90 deg, from the
    walls in the depth images: every keyframe's vertical surface normals vote
    into a 1-degree histogram of their yaw modulo 90; `yaw` is its circular
    mode. Indoors, furniture stands square to the walls, so a fit takes its
    axes from here instead of from the line of sight (one-sided) or the
    points' principal axis (multi-sided) -- refitting the dev buffers in the
    world's axes lifted IoU 0.170 -> 0.267 (S78), and the world axes of an
    HM3D house are its walls'; this recovers them without assuming so."""

    def __init__(self, stride: int = 16, max_ny: float = 0.15) -> None:
        self.stride = int(stride)
        self.max_ny = float(max_ny)
        self.hist = np.zeros(90)

    def add(self, frame) -> None:
        d = np.asarray(frame.depth, dtype=np.float64)[:: self.stride, :: self.stride]
        h, w = d.shape
        intr = frame.intrinsics
        v, u = np.mgrid[0:h, 0:w]
        u = u * self.stride
        v = v * self.stride
        p = np.stack([(u - intr.cx) / intr.fx * d, (v - intr.cy) / intr.fy * d, d], axis=-1)
        dx = p[:-1, 1:] - p[:-1, :-1]
        dy = p[1:, :-1] - p[:-1, :-1]
        dc = d[:-1, :-1]
        ok = ((dc > 1e-3) & (d[:-1, 1:] > 1e-3) & (d[1:, :-1] > 1e-3)
              & (np.abs(d[:-1, 1:] - dc) < 0.05 * dc) & (np.abs(d[1:, :-1] - dc) < 0.05 * dc))
        n = np.cross(dx[ok], dy[ok])
        norm = np.linalg.norm(n, axis=1)
        keep = norm > 1e-9
        n = (n[keep] / norm[keep, None]) @ np.asarray(frame.T_wc[:3, :3]).T
        n = n[np.abs(n[:, 1]) < self.max_ny]
        if not len(n):
            return
        ang = np.degrees(np.arctan2(n[:, 2], n[:, 0])) % 90.0
        self.hist += np.bincount(np.floor(ang).astype(int) % 90, minlength=90)[:90]

    @property
    def yaw(self) -> Optional[float]:
        if self.hist.sum() < 50:
            return None
        k = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        sm = np.zeros(90)
        for i, wgt in zip(range(-2, 3), k):
            sm += wgt * np.roll(self.hist, i)
        return float(np.radians(int(np.argmax(sm)) + 0.5))
