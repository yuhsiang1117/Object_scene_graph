"""Image-text similarity: how much a view looks like the thing being searched for.

One scalar per frame, painted into a top-down map (mapping/value_map.py) so
frontier selection can prefer directions that look like the target's habitat --
the mechanism VLFM and ASCENT use, and the main thing this pipeline's purely
geometric exploration lacks.

The backend is behind an interface because the two candidates differ in what
they cost, not in what they provide: CLIP is already installed and runs in
milliseconds, BLIP-2 ITM is what ASCENT uses and would need transformers plus a
~1.2 GB checkpoint. Absolute scores are not comparable between them, but nothing
downstream depends on the scale -- the value map's confidence weighting is
independent of the value, and frontier selection only ranks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class ImageTextScorer(ABC):
    @abstractmethod
    def score(self, rgb: np.ndarray, texts: List[str]) -> np.ndarray:
        """Similarity of one image to each text, as (len(texts),) float32."""

    def reset(self) -> None:
        """Called at episode start (the prompt changes with the target)."""


class ConstantScorer(ImageTextScorer):
    """Returns 1.0 for everything. For tests, and for isolating the value map's
    geometry from the model's judgement in an ablation."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value
        self.n_calls = 0

    def score(self, rgb: np.ndarray, texts: List[str]) -> np.ndarray:
        self.n_calls += 1
        return np.full(len(texts), self.value, dtype=np.float32)


class ClipScorer(ImageTextScorer):
    """OpenAI CLIP via the ultralytics fork already installed for YOLOE.

    Text embeddings are cached: the prompt changes once per episode, so the
    per-frame cost is a single 224x224 image forward (~2 ms on a 3090).
    Preprocessing is done with cv2 rather than CLIP's PIL transform to avoid
    the conversion round-trip on every frame.
    """

    # CLIP's normalisation constants.
    _MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    _STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        half: bool = True,
        download_root: str = "data/clip",
    ) -> None:
        import clip
        import torch

        self.torch = torch
        self.clip = clip
        self.device = device
        # The nav container has no outbound network, so the checkpoint must be
        # staged from the host (scripts/download_weights.py --clip) into a
        # bind-mounted path -- NOT data/weights, which is a named volume and is
        # therefore invisible to the host.
        self.model, _ = clip.load(model_name, device=device, download_root=download_root)
        self.model.eval()
        self.half = half and str(device).startswith("cuda")
        self._text_cache: dict = {}
        self.n_calls = 0

    def reset(self) -> None:
        self._text_cache.clear()

    def _text_features(self, texts: List[str]):
        key = tuple(texts)
        feats = self._text_cache.get(key)
        if feats is None:
            with self.torch.no_grad():
                tok = self.clip.tokenize(list(texts)).to(self.device)
                feats = self.model.encode_text(tok).float()
                feats = feats / feats.norm(dim=-1, keepdim=True)
            self._text_cache[key] = feats
        return feats

    def _preprocess(self, rgb: np.ndarray):
        import cv2

        img = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        t = self.torch.from_numpy(img).permute(2, 0, 1)[None].to(self.device)
        return t

    def score(self, rgb: np.ndarray, texts: List[str]) -> np.ndarray:
        self.n_calls += 1
        with self.torch.no_grad():
            img = self.model.encode_image(self._preprocess(rgb)).float()
            img = img / img.norm(dim=-1, keepdim=True)
            sims = (img @ self._text_features(texts).T).squeeze(0)
        return sims.detach().cpu().numpy().astype(np.float32)


def build_image_text_scorer(cfg) -> Optional[ImageTextScorer]:
    """None when the value map is off, so nothing is loaded."""
    if not getattr(cfg.exploration, "value_map", False):
        return None
    model = getattr(cfg.exploration, "value_model", "clip")
    if model == "constant":
        return ConstantScorer()
    if model == "clip":
        return ClipScorer(
            model_name=getattr(cfg.exploration, "value_clip_name", "ViT-B/32"),
            device=cfg.detector.device,
            download_root=getattr(cfg.exploration, "value_clip_root", "data/clip"),
        )
    raise ValueError(f"unknown value_model: {model}")
