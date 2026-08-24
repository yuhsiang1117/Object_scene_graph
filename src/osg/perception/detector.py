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
