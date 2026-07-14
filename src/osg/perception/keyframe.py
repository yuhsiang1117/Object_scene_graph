"""Keyframe selection and storage. Detection and scene-graph updates run on
keyframes only; keyframe JPEGs feed the VLM scorer (improvement B) and the
target verifier (improvement C).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..core.geometry import matrix_to_rotvec
from ..core.types import FrameData


@dataclass
class KeyframeRef:
    frame_id: int
    path: Optional[str]  # jpg on disk (None = in-memory only)
    position: np.ndarray  # (3,) camera position
    view_dir: np.ndarray  # (3,) camera forward (world)


class KeyframeSelector:
    def __init__(self, trans_thresh_m: float = 0.25, rot_thresh_deg: float = 30.0) -> None:
        self.trans_thresh = trans_thresh_m
        self.rot_thresh = np.radians(rot_thresh_deg)
        self._last_T: Optional[np.ndarray] = None

    def is_keyframe(self, T_wc: np.ndarray) -> bool:
        if self._last_T is None:
            self._last_T = T_wc.copy()
            return True
        dt = np.linalg.norm(T_wc[:3, 3] - self._last_T[:3, 3])
        dR = self._last_T[:3, :3].T @ T_wc[:3, :3]
        dr = np.linalg.norm(matrix_to_rotvec(dR))
        if dt > self.trans_thresh - 1e-6 or dr > self.rot_thresh - 1e-6:
            self._last_T = T_wc.copy()
            return True
        return False

    def reset(self) -> None:
        self._last_T = None


class KeyframeStore:
    def __init__(self, save_dir: Optional[str] = None, downscale: int = 2, max_in_memory: int = 200) -> None:
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.downscale = downscale
        self.max_in_memory = max_in_memory
        self._refs: List[KeyframeRef] = []
        self._images: Dict[int, np.ndarray] = {}

    def add(self, frame: FrameData) -> KeyframeRef:
        img = frame.rgb[:: self.downscale, :: self.downscale]
        path = None
        if self.save_dir is not None:
            import imageio.v2 as imageio

            path = str(self.save_dir / f"kf_{frame.frame_id:05d}.jpg")
            imageio.imwrite(path, img, quality=85)
        else:
            self._images[frame.frame_id] = img.copy()
            if len(self._images) > self.max_in_memory:
                self._images.pop(next(iter(self._images)))
        # OpenCV camera forward is +z
        ref = KeyframeRef(
            frame_id=frame.frame_id,
            path=path,
            position=frame.camera_position,
            view_dir=frame.T_wc[:3, 2].copy(),
        )
        self._refs.append(ref)
        return ref

    def refs(self) -> List[KeyframeRef]:
        return list(self._refs)

    def load_image(self, ref: KeyframeRef) -> Optional[np.ndarray]:
        if ref.path is not None:
            import imageio.v2 as imageio

            return np.asarray(imageio.imread(ref.path))
        return self._images.get(ref.frame_id)

    def nearest_facing(self, target_xy: np.ndarray, k: int = 2, plane=(0, 2)) -> List[KeyframeRef]:
        """Keyframes closest to target (in the ground plane) that face it."""
        scored = []
        for ref in self._refs:
            pos = ref.position[list(plane)]
            to_target = target_xy - pos
            dist = np.linalg.norm(to_target)
            if dist < 1e-3:
                continue
            facing = float(np.dot(to_target / dist, ref.view_dir[list(plane)]))
            if facing < 0.3:  # must roughly look toward the target
                continue
            scored.append((dist, ref))
        scored.sort(key=lambda s: s[0])
        return [r for _, r in scored[:k]]

    def reset(self) -> None:
        self._refs.clear()
        self._images.clear()
