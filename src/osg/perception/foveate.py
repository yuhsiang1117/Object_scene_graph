"""A second look at the furniture, because a 40 px can is not a small sofa.

The detector runs once per keyframe over the whole 1280 px frame, and that is
the right budget for furniture. It is the wrong budget for the objects this
benchmark actually asks for. Measured on 00848's tin can, over the 21 keyframes
where the ground-truth instrument says the object was in view and unoccluded:

    condition                                     named at the 0.20 gate
    ------------------------------------------------------------------
    the run's own vocabulary and imgsz                      0 / 21
    the same, minus sofa/pillow/cushion/bed/...             0 / 21
    ten alternative names, one swapped in at a time      <= 1 / 21
    a 320 px window around it, upscaled to 1280             5 / 21

So it is not the name and it is not class competition -- removing every
furniture class that was outscoring the can changes nothing, and neither does
calling it a soup can, a food can or a red cylinder. It is scale. The can
subtends about 40 px at 2 m and the only condition that recovers it is the one
that gives it more of them.

The window matters as much as the magnification: a 192 px window scores 0/21,
worse than doing nothing, and the 320 px window works at 2.6-3.4 m and fails at
1.9-2.4 m. Both failures are the same one -- an object that fills its crop has
lost the context the detector needs to call it an object. So the region to crop
is not a box around the target (which the agent cannot know) but the SURFACE it
would be resting on, which the map already holds as a container track, and whose
projection shrinks as the agent backs away exactly as the required window does.

One extra detector call per keyframe per region, capped. Off by default.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import Detection

Region = Tuple[int, int, int, int]


def container_regions(
    object_layer,
    frame,
    categories: Sequence[str],
    max_range_m: float,
    min_px: float,
    max_regions: int,
) -> List[Region]:
    """Image boxes of the container surfaces in view, nearest first.

    Nearest first because the budget is small and the near surface is the one
    the agent is standing at; ranking by projected area would prefer whichever
    sofa happens to fill the frame from across the room.
    """
    T_cw = np.linalg.inv(frame.T_wc)
    K = frame.intrinsics.K()
    h, w = frame.rgb.shape[:2]
    cam = frame.camera_position
    wanted = {str(c).lower() for c in categories}
    scored: List[Tuple[float, Region]] = []
    for track in object_layer.tracks():
        if str(track.label).lower() not in wanted:
            continue
        centre = object_layer.center_of(track)
        rng = float(np.linalg.norm(np.asarray(centre) - np.asarray(cam)))
        if rng > max_range_m:
            continue
        ellipse = track.ellipsoid.project(K, T_cw)
        if ellipse is None:
            continue
        x1, y1, x2, y2 = ellipse.bbox()
        box = (max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)))
        if (box[2] - box[0]) * (box[3] - box[1]) < min_px:
            continue
        scored.append((rng, box))
    scored.sort(key=lambda item: item[0])
    return [box for _, box in scored[:max_regions]]


def _square(box: Region, w: int, h: int, pad: float) -> Region:
    """A padded square window, clipped to the image.

    Square because the detector's letterbox pads a lopsided crop with grey bars
    and spends the magnification on them; padded because the 192 px probe says a
    crop that ends at the object's own edge is worse than no crop at all.
    """
    x1, y1, x2, y2 = box
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    half = 0.5 * max(x2 - x1, y2 - y1) * (1.0 + pad)
    half = max(half, 32.0)
    return (max(0, int(cx - half)), max(0, int(cy - half)),
            min(w, int(cx + half)), min(h, int(cy + half)))


def foveated_detect(
    detector,
    rgb: np.ndarray,
    regions: Sequence[Region],
    pad: float = 0.15,
    min_score: float = 0.0,
) -> List[Detection]:
    """Detections found in upscaled crops, in FULL-FRAME coordinates.

    The detector's own `imgsz` does the upscaling: handing it a 400 px crop is
    exactly handing the object 3x the pixels it had in the whole frame.
    """
    h, w = rgb.shape[:2]
    out: List[Detection] = []
    for box in regions:
        x1, y1, x2, y2 = _square(box, w, h, pad)
        if x2 - x1 < 32 or y2 - y1 < 32:
            continue
        crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])
        for det in detector.detect(crop) or []:
            if float(det.score) < min_score:
                continue
            remapped = _to_full_frame(det, x1, y1, crop.shape[:2], (h, w))
            if remapped is not None:
                out.append(remapped)
    return out


def _to_full_frame(det: Detection, x0: int, y0: int, crop_hw, full_hw) -> Optional[Detection]:
    """Move one crop-space detection back into the frame it came from.

    The detector returns masks and boxes in the coordinates of the image it was
    handed -- the crop -- so the move is a translation. The resize guards the
    case where a backend hands back a mask at its own working resolution; a mask
    pasted at the wrong size would silently mislabel the pixels underneath it.
    """
    import cv2

    ch, cw = crop_hw
    fh, fw = full_hw
    if det.mask is None:
        return None
    mask = det.mask
    if mask.shape != (ch, cw):
        sy, sx = ch / float(mask.shape[0]), cw / float(mask.shape[1])
        mask = cv2.resize(mask.astype(np.uint8), (cw, ch),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        sy = sx = 1.0
    full_mask = np.zeros((fh, fw), dtype=bool)
    full_mask[y0:y0 + ch, x0:x0 + cw] = mask
    bx1, by1, bx2, by2 = [float(v) for v in det.bbox_xyxy]
    bbox = np.array([x0 + bx1 * sx, y0 + by1 * sy,
                     x0 + bx2 * sx, y0 + by2 * sy], dtype=float)
    return Detection(label=det.label, score=float(det.score),
                     bbox_xyxy=bbox, mask=full_mask)


def merge(base: List[Detection], extra: List[Detection], iou_thresh: float = 0.5
          ) -> Tuple[List[Detection], int]:
    """`extra` detections the full-frame pass did not already have.

    A foveated detection that duplicates one the base pass made is not evidence
    of anything; the whole claim is about what the second look ADDS, so the
    counter has to be the additions.
    """
    added: List[Detection] = []
    for det in extra:
        if any(_same(det, b) >= iou_thresh for b in base):
            continue
        added.append(det)
    return base + added, len(added)


def _same(a: Detection, b: Detection) -> float:
    if str(a.label).lower() != str(b.label).lower():
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a.bbox_xyxy]
    bx1, by1, bx2, by2 = [float(v) for v in b.bbox_xyxy]
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return 0.0 if union <= 0 else inter / union
