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
        # The live view (agent at the approach viewpoint, facing the object)
        # is the primary evidence; the stored crop is supporting context.
        images = []
        if live_view is not None:
            images.append(live_view)
        if track.best_crop is not None:
            images.append(track.best_crop)
        if not images:
            return False  # nothing to verify against — do not stop blindly

        target_text = target.replace("_", " ")
        try:
            resp = self.vlm.chat(
                prompts.VERIFY_SYSTEM,
                prompts.VERIFY_USER.format(target=target_text),
                images=images,
            )
        except Exception:
            # VLM unavailable: fail open (paper behavior = no verification)
            return True
        finally:
            self.n_calls += 1

        self._dump(images, target, resp)
        is_target = bool(resp.get("is_target", False))
        conf = float(resp.get("confidence", 0.0))
        accepted = is_target and conf >= self.accept_confidence
        if not accepted:
            self.n_rejections += 1
        return accepted
