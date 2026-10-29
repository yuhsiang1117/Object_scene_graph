"""Find the object when the detector will not name it.

21 of the 50 remaining failures are perception: the target in view, often close
and centred and fully visible, and never named. Measured on the dumped
keyframes where the object's pixel is known
(`scripts/legacy/probe_perception_gap.py`), the detector is not mislabelling the
object so much as never proposing it. A proposal tight enough to BE the object
rather than the furniture under it comes from YOLOE on 0% of scissors frames,
20% of cracker box, 25% of soup can, 48% of banana -- and from FastSAM-s on
72/92/100/87%. What YOLOE calls the boxes that DO cover the object says the
rest: scissors `mirror` x26, soup can `lamp` x32, banana `chair` x87. It
proposes the furniture.

That is DualMap's structural advantage measured on our own frames: its detector
emits a tight crop and every stage downstream inherits it.

So: segment class-agnostically, rank the regions by appearance against the
query text, and hand the winner back as an ordinary Detection under the target
label. Everything downstream -- admission, the presence filter, the identity
channel, the candidate gate, the VLM, the attempt protocol -- then judges it
exactly as it judges the detector's own output. Nothing here can stop an
approach or score a trial by itself.

The measured limits, which are the reason this is threshold-gated and capped:

* Appearance can only pick the region out for compact objects. Ranking the true
  region against every other region in the frame (median 59 of them), rank-1 is
  91% for cracker box, 60% soup can, 39% banana -- and 3% for scissors, whose
  region scores 0.149 against distractors. Thin metal at two metres carries too
  little signal for the encoder, and no proposer fixes that.
* A threshold exists at 0.24: 40 correct admissions against 7 wrong, 85%
  precision, 24% recall. Below it precision collapses, above it recall does.
* Crop treatment matters as much as the encoder. Padding the box by a quarter
  of its longer side and asking for "a pair of scissors" rather than "a photo of
  a scissors" moves overall rank-1 from 52% to 70%. ViT-B/32 is worse than
  MobileCLIP-S2 at every setting tried.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..core.types import Detection

# The phrasing a caption corpus would use. CLIP is prompt-sensitive and the
# benchmark's label strings are not how anyone writes a caption; measured, this
# is worth more than the choice of backbone.
NATURAL = {
    "scissors": "a pair of scissors",
    "tin can": "a tin can of soup",
    "soup can": "a tin can of soup",
    "cracker box": "a cardboard box of crackers",
    "banana": "a ripe banana",
    "mug": "a coffee mug",
    "bowl": "a bowl",
    "red plate": "a red plate",
    "plate": "a red plate",
    "blue plastic pitcher": "a blue plastic pitcher",
    "pitcher": "a blue plastic pitcher",
    "bleach bottle": "a bottle of bleach cleanser",
}


def phrase(target: str) -> str:
    t = str(target).strip().lower()
    return NATURAL.get(t, f"a photo of a {t}")


class RegionProposer:
    """Class-agnostic regions, ranked by appearance against the query."""

    def __init__(self, model, encoder, cfg) -> None:
        self.model = model
        self.encoder = encoder
        self.cfg = cfg
        self.text: Optional[np.ndarray] = None
        self.target: Optional[str] = None
        # The winning cosine on the last frame the stage actually scored, so a
        # probe can sweep tau over the production path instead of re-deriving
        # the scoring. -1.0 when the stage declined before ranking.
        self.last_score: float = -1.0
        self.counters = {
            "region_frames": 0,        # keyframes the stage ran on
            "region_proposals": 0,     # regions scored
            "region_admitted": 0,      # regions handed back as detections
            "region_below_tau": 0,     # ran, best region did not clear the bar
        }

    def set_target(self, target: str) -> None:
        """Encode the query once per episode."""
        self.target = target
        self.text = self.encoder.text_feature(phrase(target))
        for k in self.counters:
            self.counters[k] = 0

    def _regions(self, rgb: np.ndarray):
        r = self.model.predict(
            rgb, imgsz=int(self.cfg.imgsz), conf=float(self.cfg.conf), iou=float(self.cfg.iou),
            retina_masks=True, verbose=False, device=str(self.cfg.device),
        )[0]
        if r.boxes is None or r.masks is None or len(r.boxes) == 0:
            return [], None
        boxes = r.boxes.xyxy.detach().cpu().numpy()
        # Masks stay on the device: only the winning region's is ever used,
        # so only that one is copied to the host (`propose`), not every
        # full-resolution mask of every keyframe.
        return list(boxes), r.masks.data

    def propose(self, rgb: np.ndarray) -> Optional[Detection]:
        """The best region for the query on this frame, or None.

        None whenever the stage declines: no target, nothing segmented, every
        region outside the size band, or the best one short of the threshold.
        """
        self.last_score = -1.0
        if self.text is None or self.target is None:
            return None
        boxes, masks = self._regions(rgb)
        if not boxes:
            return None
        H, W = rgb.shape[:2]
        frame_px = float(H * W)
        keep = []
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = [float(v) for v in box]
            a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            # Too small to carry signal, or big enough to be the furniture.
            if a < float(self.cfg.min_area_px) or a > float(self.cfg.max_area_frac) * frame_px:
                continue
            keep.append((box, index, a))
        if not keep:
            return None
        pad = float(self.cfg.pad_frac)
        crops = []
        for box, _, _ in keep:
            x1, y1, x2, y2 = [float(v) for v in box]
            p = pad * max(x2 - x1, y2 - y1)
            cx1, cy1 = max(0, int(x1 - p)), max(0, int(y1 - p))
            cx2, cy2 = min(W, int(np.ceil(x2 + p))), min(H, int(np.ceil(y2 + p)))
            if cx2 - cx1 < 2 or cy2 - cy1 < 2:
                cx1, cy1, cx2, cy2 = int(x1), int(y1), min(W, int(x2) + 2), min(H, int(y2) + 2)
            crops.append(rgb[cy1:cy2, cx1:cx2])
        crops = crops[: int(self.cfg.max_regions)]
        keep = keep[: int(self.cfg.max_regions)]
        feats = self.encoder.encode_images(crops)
        sims = feats @ self.text
        self.counters["region_frames"] += 1
        self.counters["region_proposals"] += len(sims)
        best = int(np.argmax(sims))
        score = float(sims[best])
        self.last_score = score
        if score < float(self.cfg.tau):
            self.counters["region_below_tau"] += 1
            return None
        box, index, _ = keep[best]
        mask = masks[index].detach().cpu().numpy().astype(bool)
        self.counters["region_admitted"] += 1
        # The region's own feature rides along so the track can keep a running
        # mean over views without a second encode. What decides a commit is the
        # cosine of THAT mean against the query -- per-frame argmax was measured
        # not to separate (docs/history/REGION_FALSE_ADMISSION.md), and a mean over
        # several views of one 3D object is DualMap's actual matching unit.
        det = Detection(
            label=str(self.target),
            # The detector's own confidence scale, not the cosine: downstream
            # gates (`min_det_score`, the evidence sum, the VLM's bbox gate) are
            # calibrated against detector scores, and feeding a 0.24 cosine into
            # them would read as an almost-certainly-false detection. The cosine
            # is recorded on the counter instead.
            score=float(self.cfg.admit_score),
            bbox_xyxy=np.asarray(box, dtype=float),
            mask=np.asarray(mask, dtype=bool),
            clip_ft=np.asarray(feats[best], dtype=np.float32),
            source="proposal",
        )
        return det


def build_region_proposer(cfg):
    """The proposal stage, or None when it is off (nothing is loaded)."""
    rp = getattr(cfg, "region_proposal", None)
    if rp is None or not bool(rp.enabled):
        return None
    from ultralytics import FastSAM

    from .feature_encoder import FeatureEncoder

    model = FastSAM(str(rp.weights))
    encoder = FeatureEncoder(str(rp.backbone), str(rp.checkpoint), str(rp.device), half=True)
    return RegionProposer(model, encoder, rp)
