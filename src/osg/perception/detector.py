"""Open-vocabulary detection + instance segmentation.

Improvement A over the paper: YOLOE (ultralytics) replaces YOLO-World and
provides masks directly, removing the SAM stage entirely.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

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
    ) -> None:
        from ultralytics import YOLOE  # deferred: heavy import

        self.model = YOLOE(weights)
        # Checkpoints ship fp16 weights; get_text_pe feeds the fp32 mobileclip
        # features through the checkpoint's text head -> dtype mismatch unless
        # the model is fp32. Inference still runs fp16 via predict(half=True).
        self.model.model.float()
        self.conf = conf
        self.imgsz = imgsz
        self.half = half
        self.device = device
        self._classes: List[str] = []
        if vocabulary:
            self.set_vocabulary(vocabulary)

    @staticmethod
    def _normalize(label: str) -> str:
        return label.lower().replace("_", " ").strip()

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
        self.model.set_classes(classes, self.model.get_text_pe(classes))
        self.model.predictor = None  # AutoBackend cached the fp16 view
        if hasattr(self.model.model, "clip_model"):
            del self.model.model.clip_model  # free the 572MB encoder
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        self._classes = classes

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            rgb[..., ::-1],  # ultralytics expects BGR ndarray
            conf=self.conf,
            imgsz=self.imgsz,
            half=self.half,
            device=self.device,
            verbose=False,
        )
        out: List[Detection] = []
        r = results[0]
        if r.masks is None or r.boxes is None:
            return out
        h, w = rgb.shape[:2]
        masks = r.masks.data.cpu().numpy()  # (N, mh, mw)
        for i in range(len(r.boxes)):
            mask = masks[i]
            if mask.shape != (h, w):
                import cv2

                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            det = Detection(
                label=r.names[int(r.boxes.cls[i])],
                score=float(r.boxes.conf[i]),
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
