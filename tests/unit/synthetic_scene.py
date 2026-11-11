"""A ray-cast scene of axis-aligned boxes on a floor, for object-layer tests.

World is habitat's: y up, floor at y=0. Cameras are OpenCV T_wc (conftest's
make_camera). `render` returns metric z-depth and an instance image (-1 =
floor or wall), and `detections` turns the instance image into Detection
objects with optional mask bleed -- the dilation that drags background depth
into a mask, which is one of the failure modes the ellipsoid work is about.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from osg.core.types import CameraIntrinsics, Detection, FrameData


@dataclass
class Box:
    label: str
    center: np.ndarray   # (3,) world
    half: np.ndarray     # (3,) half sizes, world axes

    @property
    def lo(self):
        return np.asarray(self.center, float) - np.asarray(self.half, float)

    @property
    def hi(self):
        return np.asarray(self.center, float) + np.asarray(self.half, float)


def render(boxes: Sequence[Box], intr: CameraIntrinsics, T_wc: np.ndarray,
           room_half: float = 6.0):
    h, w = intr.height, intr.width
    vs, us = np.mgrid[0:h, 0:w].astype(float)
    d_cam = np.stack([(us + 0.5 - intr.cx) / intr.fx, (vs + 0.5 - intr.cy) / intr.fy,
                      np.ones_like(us)], axis=-1)            # z = 1, so t along it is z-depth
    R, o = T_wc[:3, :3], T_wc[:3, 3]
    d = d_cam @ R.T
    best = np.full((h, w), np.inf)
    inst = np.full((h, w), -1, dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        # floor y=0 and four walls / ceiling of a room box: background
        for axis, val in ((1, 0.0), (1, 3.0), (0, -room_half), (0, room_half),
                          (2, -room_half), (2, room_half)):
            t = (val - o[axis]) / d[..., axis]
            ok = (t > 1e-6) & (t < best)
            best = np.where(ok, t, best)
        for i, b in enumerate(boxes):
            t1 = (b.lo - o) / d
            t2 = (b.hi - o) / d
            tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
            hit = (tmax >= np.maximum(tmin, 1e-6)) & (tmin < best)
            best = np.where(hit, tmin, best)
            inst = np.where(hit, i, inst)
    depth = np.where(np.isfinite(best), best, 0.0).astype(np.float32)
    return depth, inst


def frame_at(boxes, intr, T_wc, frame_id=0):
    depth, inst = render(boxes, intr, T_wc)
    frame = FrameData(frame_id=frame_id, rgb=np.zeros((intr.height, intr.width, 3), np.uint8),
                      depth=depth, T_wc=T_wc, intrinsics=intr)
    return frame, inst


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(px):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def detections(boxes, inst, frame: FrameData, bleed_px: int = 0, score: float = 0.9,
               min_px: int = 30) -> List[Detection]:
    dets = []
    for i, b in enumerate(boxes):
        mask = inst == i
        if mask.sum() < min_px:
            continue
        if bleed_px:
            mask = _dilate(mask, bleed_px)
        ys, xs = np.nonzero(mask)
        bbox = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
        det = Detection(label=b.label, score=score, bbox_xyxy=bbox, mask=mask)
        det.crop_from(frame.rgb)
        det.gt_instance = i
        dets.append(det)
    return dets
