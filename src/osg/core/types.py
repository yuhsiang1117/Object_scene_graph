"""Core data types. FrameData is the single currency between the simulator
(or any future data source: ScanNet, real robot) and the pipeline.

Camera convention: OpenCV pinhole (z forward, x right, y down). The habitat
wrapper converts from OpenGL before constructing FrameData. World frame is
whatever the source uses (habitat: y-up); mapping modules pick their plane
axes explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @classmethod
    def from_hfov(cls, hfov_deg: float, width: int, height: int) -> "CameraIntrinsics":
        fx = width / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0, width=width, height=height)


@dataclass
class FrameData:
    frame_id: int
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, meters, 0 = invalid
    T_wc: np.ndarray  # (4, 4) camera-to-world, OpenCV convention
    intrinsics: CameraIntrinsics
    timestamp: float = 0.0

    @property
    def T_cw(self) -> np.ndarray:
        R = self.T_wc[:3, :3]
        t = self.T_wc[:3, 3]
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ t
        return T

    @property
    def camera_position(self) -> np.ndarray:
        return self.T_wc[:3, 3].copy()


@dataclass
class Detection:
    label: str
    score: float
    bbox_xyxy: np.ndarray  # (4,) float
    mask: np.ndarray  # (H, W) bool
    crop: Optional[np.ndarray] = None  # bbox-cropped rgb for VLM verification

    def crop_from(self, rgb: np.ndarray, pad: int = 8) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox_xyxy.astype(int)
        h, w = rgb.shape[:2]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        self.crop = rgb[y1:y2, x1:x2].copy()
        return self.crop
