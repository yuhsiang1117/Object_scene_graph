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


class Blip2ItmScorer(ImageTextScorer):
    """ASCENT's own value model, over HTTP.

    The value map is what ranks frontiers, and ASCENT scores it with BLIP-2
    image-text matching (`map_controller.py:110`, `BLIP2ITMClient`). OSG has
    been using CLIP's whole-image cosine, which is a different quantity: CLIP is
    trained for retrieval over a shared embedding, BLIP-2 ITM is trained to
    answer "does this text match this image", which is the question the value
    map asks.

    BLIP-2 will not co-exist with this environment -- lavis needs numpy 1.x
    builds and transformers pins that habitat's env does not have -- so it runs
    where it does work, in the `ascent` conda env, behind the Flask server
    ASCENT itself ships (`model_api/blip2itm_out.py`). That is not a workaround
    bolted on: process-per-model over HTTP is ASCENT's own architecture, and
    this speaks its wire format.

    Falls back to a constant on any failure. A value map stuck at one value is
    a value map that ranks nothing, which is visible in `value_errors` rather
    than silently reshaping exploration.
    """

    def __init__(self, url: str = "http://localhost:13182/blip2itm",
                 timeout_s: float = 10.0, strict: bool = False) -> None:
        self.url = url
        self.timeout_s = float(timeout_s)
        # Fail loud. A cosine of 0 from a dead server means ASCENT's commit
        # gate never latches, which means the agent never STOPs.
        self.strict = bool(strict)
        self.n_calls = 0
        self.n_errors = 0
        self._warned = False

    def reset(self) -> None:
        self.n_calls = 0
        self.n_errors = 0

    def score(self, rgb: np.ndarray, texts: List[str]) -> np.ndarray:
        import base64
        import json as _json
        import urllib.request

        import cv2

        out = np.zeros(len(texts), dtype=np.float32)
        # No channel swap. ASCENT's own client hands the RGB array straight to
        # `cv2.imencode` (`server_wrapper_out.py:59`) and the server decodes it
        # and calls `Image.fromarray`, so the round trip preserves whatever was
        # passed. Converting to BGR here would feed BLIP-2 channel-swapped
        # images -- and it would still return plausible-looking scores.
        ok, buf = cv2.imencode(".jpg", rgb)
        if not ok:
            self.n_errors += 1
            return out
        payload = base64.b64encode(buf.tobytes()).decode("ascii")
        for i, text in enumerate(texts):
            body = _json.dumps({"image": payload, "txt": text,
                                "method": "cosine"}).encode()
            req = urllib.request.Request(
                self.url, data=body, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    out[i] = float(_json.loads(r.read())["response"])
                self.n_calls += 1
            except Exception as exc:  # noqa: BLE001 -- any failure means no signal
                self.n_errors += 1
                if self.strict:
                    from .ascent_models import PerceptionUnavailable

                    raise PerceptionUnavailable(f"{self.url}: {exc}") from exc
                if not self._warned:
                    self._warned = True
                    print(f"[blip2itm] unreachable at {self.url} ({exc}); "
                          "value map will read 0 until it answers")
        return out


def build_scorer_by_name(model: str, cfg) -> Optional[ImageTextScorer]:
    """One scorer by name, independent of whether the value map wants one.

    The value map and the COMMIT GATE ask the same model two different
    questions, and ASCENT answers both with the same BLIP-2 call
    (`map_controller.py:540-562` computes the value-map cosine and keeps
    `cosines[0][0]` as `_blip_cosine`, which `:770-776` thresholds at 0.15 to
    latch `_double_check_goal`). OSG splits them so the gate can run on BLIP-2
    while the value map stays on whatever measured best.
    """
    if model in ("none", "", None):
        return None
    if model == "constant":
        return ConstantScorer()
    if model == "clip":
        return ClipScorer(
            model_name=getattr(cfg.exploration, "value_clip_name", "ViT-B/32"),
            device=cfg.detector.device,
            download_root=getattr(cfg.exploration, "value_clip_root", "data/clip"),
        )
    if model == "blip2itm":
        return Blip2ItmScorer(
            url=str(getattr(cfg.exploration, "value_blip2_url",
                            "http://localhost:13182/blip2itm")),
            timeout_s=float(getattr(cfg.exploration, "value_blip2_timeout_s", 10.0)),
            strict=bool(getattr(cfg.exploration, "value_strict", False)),
        )
    raise ValueError(f"unknown image-text model: {model}")


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
    if model == "blip2itm":
        return Blip2ItmScorer(
            url=str(getattr(cfg.exploration, "value_blip2_url",
                            "http://localhost:13182/blip2itm")),
            timeout_s=float(getattr(cfg.exploration, "value_blip2_timeout_s", 10.0)),
            strict=bool(getattr(cfg.exploration, "value_strict", False)),
        )
    raise ValueError(f"unknown value_model: {model}")
