"""Open-vocabulary detection + instance segmentation.

Improvement A over the paper: YOLOE (ultralytics) replaces YOLO-World and
provides masks directly, removing the SAM stage entirely.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..core.labels import normalize_label
from ..core.types import Detection


class Detector(ABC):
    @abstractmethod
    def set_vocabulary(self, classes: List[str]) -> None: ...

    @abstractmethod
    def detect(self, rgb: np.ndarray) -> List[Detection]: ...


class YoloeDetector(Detector):
    def __init__(
        self,
        weights: str = "data/weights/yoloe-11s-seg.pt",
        vocabulary: List[str] | None = None,
        conf: float = 0.3,
        imgsz: int = 512,
        half: bool = True,
        device: str = "cuda",
        class_conf: Optional[Dict[str, float]] = None,
    ) -> None:
        from ultralytics import YOLOE  # deferred: heavy import

        self._weights_dir = Path(weights).expanduser().resolve().parent
        self.model = YOLOE(weights)
        # Checkpoints ship fp16 weights; get_text_pe feeds the fp32 mobileclip
        # features through the checkpoint's text head -> dtype mismatch unless
        # the model is fp32. Inference still runs fp16 via predict(half=True).
        self.model.model.float()
        self.conf = conf
        # Per-class admission thresholds, overriding `conf` for the labels named.
        #
        # One global threshold prices every class the same, and they are not the
        # same. Measured at imgsz 1280 over 900 random navigable poses in three
        # scenes, moving the gate 0.30 -> 0.20 and counting detections above the
        # 1200 px node gate -- extra false positives against the recall gained at
        # the objects' own authored viewpoints:
        #
        #   tomato soup can    +0 FP   +0.09 recall     cracker box  +18 FP  +0.06
        #   banana             +0 FP   +0.08            bleach bottle +9 FP  +0.09
        #   plate              +2 FP   +0.06            bowl          +0 FP  +0.00
        #   blue plastic pitcher +3 FP +0.13
        #
        # Globally that is +8 true positives for +32 false ones, which is a bad
        # trade. Per class it is a good one for exactly the labels that are weak
        # to begin with, and no trade at all for the two promiscuous ones.
        self.class_conf = {
            self._normalize(k): float(v) for k, v in (class_conf or {}).items()
        }
        self.imgsz = imgsz
        self.half = half
        self.device = device
        self._classes: List[str] = []
        if vocabulary:
            self.set_vocabulary(vocabulary)

    @staticmethod
    def _normalize(label: str) -> str:
        return normalize_label(label)

    def set_vocabulary(self, classes: List[str]) -> None:
        """Normalized + sorted so the per-episode call (target already in the
        default list) compares equal and is a no-op. A genuine vocabulary
        change must undo predict()'s in-place fp16 cast before re-encoding
        (the text head is fp32-only) and rebuild the cached predictor."""
        classes = sorted({self._normalize(c) for c in classes})
        if classes == self._classes:
            return
        import torch

        self.model.model.float().to(self.device)
        # Ultralytics resolves mobileclip_blt.ts relative to the process CWD,
        # not relative to the YOLOE checkpoint. Keep all downloaded model
        # assets in the mounted weights volume and restore the caller's CWD.
        previous_cwd = Path.cwd()
        try:
            os.chdir(self._weights_dir)
            text_pe = self.model.get_text_pe(classes)
        finally:
            os.chdir(previous_cwd)
        self.model.set_classes(classes, text_pe)
        self.model.predictor = None  # AutoBackend cached the fp16 view
        if hasattr(self.model.model, "clip_model"):
            del self.model.model.clip_model  # free the 572MB encoder
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        self._classes = classes

    def _floor_conf(self) -> float:
        """Inference has to run at the LOWEST threshold anyone asks for, because
        a detection ultralytics never returns cannot be admitted afterwards."""
        if not self.class_conf:
            return self.conf
        return min(self.conf, min(self.class_conf.values()))

    def _admits(self, label: str, score: float) -> bool:
        return score >= self.class_conf.get(self._normalize(label), self.conf)

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            rgb[..., ::-1],  # ultralytics expects BGR ndarray
            conf=self._floor_conf(),
            imgsz=self.imgsz,
            half=self.half,
            device=self.device,
            verbose=False,
            project="/tmp/yolo_runs",  # keep ultralytics' save_dir out of the repo
        )
        out: List[Detection] = []
        r = results[0]
        if r.masks is None or r.boxes is None:
            return out
        h, w = rgb.shape[:2]
        masks = r.masks.data.cpu().numpy()  # (N, mh, mw)
        for i in range(len(r.boxes)):
            label = r.names[int(r.boxes.cls[i])]
            score = float(r.boxes.conf[i])
            # Before the mask resize, which is the expensive part of this loop.
            if not self._admits(label, score):
                continue
            mask = masks[i]
            if mask.shape != (h, w):
                import cv2

                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            det = Detection(
                label=label,
                score=score,
                bbox_xyxy=r.boxes.xyxy[i].cpu().numpy(),
                mask=mask.astype(bool),
            )
            det.crop_from(rgb)
            out.append(det)
        return out


# HM3D ObjectNav's six goal categories are all COCO classes, under COCO's
# names. This is the whole reason a closed-set detector is usable here.
COCO_TO_HM3D = {
    "chair": "chair",
    "bed": "bed",
    "toilet": "toilet",
    "tv": "tv_monitor",
    "couch": "sofa",
    "potted plant": "plant",
}


class YoloDetector(Detector):
    """Closed-set COCO YOLO, the stand-in for ASCENT's D-FINE.

    WHY THIS EXISTS. Native ASCENT admits a target detection at
    `coco_threshold` 0.8 from a COCO-trained detector; this repo was ingesting
    anything YOLOE emitted above 0.3 from an open-vocabulary, text-prompted
    head. Porting the 0.8 alone (S56, `outputs/s57_thresh08`) removed 80% of the
    commits that end more than 3 m from a goal viewpoint and converted every one
    of them into a timeout rather than a success: 53% SR, unchanged. The score
    that ASCENT thresholds and the score YOLOE emits are not the same quantity,
    and only 52% of episodes ever contain a YOLOE target detection at 0.8 at all.

    WHAT IT GIVES UP. The open vocabulary. Under this preset that costs nothing
    that decides an episode: stairs come from RedNet (`_stair_det_mask` -- YOLOE
    `stairs` recall 10% against RedNet's 54%, every YOLOE firing already inside
    RedNet's), and the scene graph is read-only with respect to navigation.
    `set_vocabulary` therefore only narrows which COCO classes are returned.
    """

    def __init__(
        self,
        weights: str = "data/weights/yolo11x-seg.pt",
        vocabulary: List[str] | None = None,
        conf: float = 0.3,
        imgsz: int = 640,
        half: bool = True,
        device: str = "cuda",
        class_conf: Optional[Dict[str, float]] = None,
    ) -> None:
        from ultralytics import YOLO  # deferred: heavy import

        self.model = YOLO(weights)
        self.conf = conf
        self.class_conf = {
            self._normalize(k): float(v) for k, v in (class_conf or {}).items()
        }
        self.imgsz = int(imgsz)
        self.half = bool(half)
        self.device = device
        self._wanted: Optional[set] = None

    @staticmethod
    def _normalize(label: str) -> str:
        return str(label).strip().lower().replace("_", " ")

    def set_vocabulary(self, classes: List[str]) -> None:
        """Narrow the COCO output; a closed set cannot be widened."""
        wanted = {self._normalize(c) for c in classes}
        # Accept either naming, so callers may ask for `tv_monitor` or `tv`.
        wanted |= {self._normalize(k) for k, v in COCO_TO_HM3D.items()
                   if self._normalize(v) in wanted}
        keep = {c for c in COCO_TO_HM3D if c in wanted}
        self._wanted = keep or None

    def _floor_conf(self) -> float:
        if not self.class_conf:
            return self.conf
        return min(self.conf, min(self.class_conf.values()))

    def _admits(self, label: str, score: float) -> bool:
        return score >= self.class_conf.get(self._normalize(label), self.conf)

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            rgb[..., ::-1],
            conf=self._floor_conf(),
            imgsz=self.imgsz,
            half=self.half,
            device=self.device,
            verbose=False,
            project="/tmp/yolo_runs",
        )
        out: List[Detection] = []
        r = results[0]
        if r.masks is None or r.boxes is None:
            return out
        h, w = rgb.shape[:2]
        masks = r.masks.data.cpu().numpy()
        for i in range(len(r.boxes)):
            coco = self._normalize(r.names[int(r.boxes.cls[i])])
            if coco not in COCO_TO_HM3D:
                continue
            if self._wanted is not None and coco not in self._wanted:
                continue
            label = COCO_TO_HM3D[coco]
            score = float(r.boxes.conf[i])
            if not self._admits(label, score):
                continue
            mask = masks[i]
            if mask.shape != (h, w):
                import cv2

                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            det = Detection(
                label=label,
                score=score,
                bbox_xyxy=r.boxes.xyxy[i].cpu().numpy(),
                mask=mask.astype(bool),
            )
            det.crop_from(rgb)
            out.append(det)
        return out


class DFineDetector(Detector):
    """ASCENT's own detector, over HTTP: D-FINE, with MobileSAM masks.

    This is the third of the three substitutions the ascentnav preset lists --
    "YOLOE for D-FINE + GroundingDINO + MobileSAM" -- and the only one that was
    substituted for a model that is actually here. `model_api/dfine_out.py` and
    `model_api/sam_out.py` ship in `relative_work/ascent`; they run in the
    `ascent` env behind Flask (`scripts/serve_perception.sh`), which is ASCENT's
    own process-per-model architecture rather than a workaround.

    WHY THE MASK MATTERS AS MUCH AS THE BOX. The object cloud is built by
    projecting the mask through the depth image, so the mask decides WHERE the
    agent thinks the object is. ASCENT feeds MobileSAM the detector's box and
    projects the segment; a filled rectangle projects the wall behind the object
    as if it were the object. `use_sam: false` falls back to the rectangle, for
    an A/B of exactly that.

    The GroundingDINO half of ASCENT's stack is deliberately not ported: on
    HM3D its target names are all plain COCO classes, so the non-COCO branch in
    `map_controller.py:715-719` never fires and the reference run never called
    it. Stairs come from RedNet under `stair_up_mode: rednet`, so a COCO-only
    vocabulary costs nothing a navigation decision reads.
    """

    def __init__(
        self,
        url: str = "http://localhost:13186/dfine",
        sam_url: str = "http://localhost:13183/mobile_sam",
        conf: float = 0.8,
        timeout_s: float = 15.0,
        use_sam: bool = True,
        vocabulary: List[str] | None = None,
        class_conf: Optional[Dict[str, float]] = None,
        strict: bool = False,
        **_ignored: object,
    ) -> None:
        self.url = url
        self.sam_url = sam_url
        self.conf = float(conf)
        self.timeout_s = float(timeout_s)
        self.use_sam = bool(use_sam)
        # Fail loud: see `osg/perception/ascent_models.py`.
        self.strict = bool(strict)
        self.class_conf = {self._normalize(k): float(v)
                           for k, v in (class_conf or {}).items()}
        self._wanted: Optional[set] = None
        self.n_calls = 0
        self.n_errors = 0
        self.n_sam_errors = 0
        self._warned = False

    @staticmethod
    def _normalize(label: str) -> str:
        return str(label).strip().lower().replace("_", " ")

    def set_vocabulary(self, classes: List[str]) -> None:
        wanted = {self._normalize(c) for c in classes}
        wanted |= {self._normalize(k) for k, v in COCO_TO_HM3D.items()
                   if self._normalize(v) in wanted}
        keep = {c for c in COCO_TO_HM3D if c in wanted}
        self._wanted = keep or None

    def _post(self, url: str, payload: dict) -> Optional[dict]:
        import json as _json
        import urllib.request

        body = _json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return _json.loads(r.read())
        except Exception as exc:  # noqa: BLE001
            if self.strict:
                from .ascent_models import PerceptionUnavailable

                raise PerceptionUnavailable(f"{url}: {exc}") from exc
            if not self._warned:
                self._warned = True
                print(f"[dfine] unreachable at {url} ({exc}); "
                      "start it with scripts/serve_perception.sh")
            return None

    @staticmethod
    def _encode(rgb: np.ndarray) -> Optional[str]:
        import base64

        import cv2

        # No channel swap: ASCENT hands the RGB array straight to cv2.imencode
        # and the server decodes it back, so the round trip preserves what was
        # passed. Swapping to BGR here would feed the detector swapped images
        # and it would still return plausible-looking boxes.
        ok, buf = cv2.imencode(".jpg", rgb)
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None

    def _mask_for(self, rgb: np.ndarray, box_px, img_b64: str) -> np.ndarray:
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = [int(round(float(v))) for v in box_px]
        x0, x1 = max(0, min(x0, w - 1)), max(0, min(x1, w))
        y0, y1 = max(0, min(y0, h - 1)), max(0, min(y1, h))
        if self.use_sam:
            resp = self._post(self.sam_url,
                              {"image": img_b64, "bbox": [x0, y0, x1, y1]})
            if resp is not None and "cropped_mask" in resp:
                import base64

                raw = np.frombuffer(base64.b64decode(resp["cropped_mask"]),
                                    dtype=np.uint8)
                if raw.size == h * w:
                    return raw.reshape(h, w).astype(bool)
            self.n_sam_errors += 1
            if self.strict:
                from .ascent_models import PerceptionUnavailable

                raise PerceptionUnavailable(
                    f"{self.sam_url}: no usable mask for box {[x0, y0, x1, y1]}")
        mask = np.zeros((h, w), bool)
        mask[y0:y1, x0:x1] = True
        return mask

    def _admits(self, label: str, score: float) -> bool:
        return score >= self.class_conf.get(self._normalize(label), self.conf)

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        img = self._encode(rgb)
        if img is None:
            self.n_errors += 1
            return []
        resp = self._post(self.url, {"image": img})
        if resp is None or "boxes" not in resp:
            self.n_errors += 1
            return []
        self.n_calls += 1
        h, w = rgb.shape[:2]
        out: List[Detection] = []
        for box, logit, phrase in zip(resp["boxes"], resp["logits"],
                                      resp.get("phrases", [])):
            coco = self._normalize(phrase)
            if coco not in COCO_TO_HM3D:
                continue
            if self._wanted is not None and coco not in self._wanted:
                continue
            label = COCO_TO_HM3D[coco]
            score = float(logit)
            if not self._admits(label, score):
                continue
            # `to_json` sends NORMALISED xyxy; ASCENT denormalises the same way
            # (`map_controller.py:757`).
            box_px = np.asarray(box, float) * np.array([w, h, w, h], float)
            det = Detection(label=label, score=score, bbox_xyxy=box_px,
                            mask=self._mask_for(rgb, box_px, img))
            det.crop_from(rgb)
            out.append(det)
        return out


class StubDetector(Detector):
    """Returns queued detections (tests / pipeline development without GPU)."""

    def __init__(self) -> None:
        self._queue: List[List[Detection]] = []
        self.vocabulary: List[str] = []

    def push(self, dets: List[Detection]) -> None:
        self._queue.append(dets)

    def set_vocabulary(self, classes: List[str]) -> None:
        self.vocabulary = classes

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        return self._queue.pop(0) if self._queue else []
