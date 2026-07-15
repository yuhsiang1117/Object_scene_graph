"""Improvement C, part 2: VLM verification of candidate targets. A rejected
candidate is blacklisted in the object layer and exploration resumes —
false-positive detections are a dominant failure mode of open-vocabulary
ObjectNav pipelines.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..llm import prompts
from ..llm.client import ChatClient
from ..objects.association import ObjectTrack


def _upscale_small(img: np.ndarray, min_side: float = 320.0) -> np.ndarray:
    """Small crops sit below the VLM's reliable resolution: a 156px chair
    crop was rejected while its 3x upscale was accepted (prompt-lab)."""
    import cv2

    h, w = img.shape[:2]
    scale = min_side / max(h, w)
    if scale <= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


class TargetVerifier:
    def __init__(
        self,
        vlm: ChatClient,
        accept_confidence: float = 0.5,
        debug_dir: Optional[str] = None,
    ) -> None:
        self.vlm = vlm
        self.accept_confidence = accept_confidence
        self.debug_dir = debug_dir
        self.n_calls = 0
        self.n_rejections = 0

    def _dump(self, images, target: str, resp) -> None:
        if self.debug_dir is None:
            return
        from pathlib import Path

        import imageio.v2 as imageio

        d = Path(self.debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            imageio.imwrite(d / f"verify{self.n_calls:03d}_{target}_{i}.jpg", img)
        (d / f"verify{self.n_calls:03d}_{target}.json").write_text(str(resp))

    def verify(
        self,
        track: ObjectTrack,
        target: str,
        live_view: Optional[np.ndarray] = None,
    ) -> bool:
        # Verify the detector's own evidence: the best crop. The live view
        # from the approach viewpoint often misses the object (a chair tucked
        # under a desk) and biases small VLMs into rejecting real targets;
        # with candidate quality gates the crop is meaningful evidence.
        images = []
        if track.best_crop is not None:
            images.append(_upscale_small(track.best_crop))
        elif live_view is not None:
            images.append(live_view)
        if not images:
            return False  # nothing to verify against — do not stop blindly

        target_text = target.replace("_", " ")
        # Borderline crops flip between runs (sampling variance at the
        # model's decision boundary). Quality gates already filter garbage
        # candidates, and a false rejection blacklists the true target and
        # usually ends the episode — so ask twice and accept if either
        # attempt accepts.
        accepted = False
        for attempt in range(2):
            try:
                resp = self.vlm.chat(
                    prompts.VERIFY_SYSTEM,
                    prompts.VERIFY_USER.format(target=target_text),
                    images=images,
                    temperature=0.3 if attempt else 0.0,
                )
            except Exception:
                # VLM unavailable: fail open (paper behavior = no verification)
                self.n_calls += 1
                return True
            self.n_calls += 1
            self._dump(images, target, resp)
            is_target = bool(resp.get("is_target", False))
            try:
                conf = float(resp.get("confidence", 0.6))
            except (TypeError, ValueError):
                conf = 0.6  # describe-then-decide sometimes omits confidence
            if is_target and conf >= self.accept_confidence:
                accepted = True
                break
        if not accepted:
            self.n_rejections += 1
        return accepted
