"""Image and text features for matching an object by what it LOOKS like.

The pipeline names objects: a detection becomes a track under the label the
detector chose, and every later question -- is this a candidate, is this the
surface to search, is this the thing in front of me -- is asked by comparing
that label string to the target's. DualMap never asks the label. It stores a
CLIP image feature per mapped object and ranks by cosine against the query's
TEXT feature, which is why an object its detector called `speaker` at 0.40 is
still the top candidate for a red cup, and why a scissors its detector never
named is still findable.

Measured here, that gap is one step wide: our prior maps already store a crop
for every track (343/343 in 00848), and the mapping pass DID see the missing
objects -- the scissors sits 0.09-0.10 m from a track labelled `bleach bottle`
in all three scenes. The crop is there; the feature is not.

Two backbones, because the comparison is the point:

  `vitb32`         OpenAI CLIP ViT-B/32, the checkpoint already staged for the
                   value map (data/clip/ViT-B-32.pt).
  `mobileclip_s2`  DualMap's own (open_clip `MobileCLIP-S2` / `datacompdr`),
                   reparameterised for inference. Its preprocessing is NOT
                   CLIP's -- 256 px and no mean/std normalisation -- which is
                   why this class always uses the model's own transform rather
                   than the hand-rolled cv2 path in `image_text.ClipScorer`.

Deliberately separate from `ClipScorer`. That class is the value map's
per-frame scalar scorer, it loads through the ultralytics `clip` fork (no
MobileCLIP), and `NavAgent` turns the value map on for any non-None
`image_text` -- so sharing one object between the two would silently change
frontier selection. The probe and the runtime share THIS class instead, so an
offline threshold means the same thing online.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BACKBONES = ("vitb32", "mobileclip_s2")

# open_clip's model name and pretrained tag per backbone. MobileCLIP-S2 resolves
# through the HuggingFace cache (HF_HUB_OFFLINE=1 works once it is populated);
# ViT-B-32 is loaded from the staged checkpoint file, because the nav container
# has no outbound network.
_OPEN_CLIP = {
    "vitb32": ("ViT-B-32", None),
    "mobileclip_s2": ("MobileCLIP-S2", "datacompdr"),
}


_PREP_POOL = None


def _prep_pool():
    """One small thread pool for crop preprocessing, shared by every encoder."""
    global _PREP_POOL
    if _PREP_POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _PREP_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="clip_prep")
    return _PREP_POOL


def split_preprocess(preprocess):
    """`(pil_steps, table)` when `preprocess` is PIL steps then ToTensor then
    Normalize -- both shipped backbones' transforms are -- else None.

    ToTensor and Normalize map each uint8 value of each channel to one float,
    so `table` (3, 256) holds exactly what they make of every value, computed
    by those transforms themselves. A crop then needs only its PIL steps on the
    host; the uint8 batch goes to the device and is looked up there. The
    obvious device version, `x.float().div(255)`, is NOT the same tensor:
    CUDA divides by a scalar as a multiply by its reciprocal, 1 ulp off on
    some values (measured 6e-8).
    """
    from PIL import Image

    steps = list(getattr(preprocess, "transforms", ()))
    names = [type(t).__name__ for t in steps]
    if len(steps) < 2 or not names[-2].endswith("ToTensor") or names[-1] != "Normalize":
        return None
    ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)   # 1 x 256 RGB
    out = Image.fromarray(ramp, mode="RGB")
    for t in steps[-2:]:
        out = t(out)
    table = out[:, 0, :].clone()
    if tuple(table.shape) != (3, 256):
        return None
    return steps[:-2], table


def prepare_batch(crops, preprocess, split, device, pool=None):
    """The (N, 3, H, W) model input on `device` for RGB uint8 crops: what
    `torch.stack([preprocess(crop) ...]).to(device)` gives, bit for bit, with
    the per-pixel float work moved to the device when `split` allows it."""
    import torch
    from PIL import Image

    def full(c):
        return preprocess(Image.fromarray(np.asarray(c, dtype=np.uint8)).convert("RGB"))

    def host(c):
        im = Image.fromarray(np.asarray(c, dtype=np.uint8)).convert("RGB")
        for t in split[0]:
            im = t(im)
        return np.asarray(im) if im.mode == "RGB" else None

    run = (lambda f: list(pool.map(f, crops))) if pool is not None and len(crops) > 1 \
        else (lambda f: [f(c) for c in crops])
    if split is None:
        return torch.stack(run(full)).to(device)
    arrays = run(host)
    if any(a is None or a.shape != arrays[0].shape for a in arrays):
        return torch.stack(run(full)).to(device)
    x = torch.from_numpy(np.stack(arrays)).to(device).long()       # N, H, W, 3
    table = split[1].to(device)
    return torch.stack([table[c][x[..., c]] for c in range(3)], dim=1)


def _reparameterize(model):
    """Apple's `reparameterize_model` when `mobileclip` is installed, else timm's.

    Apple's walks every module and calls `.reparameterize()`; timm's does the
    same and also folds `fuse()` / `switch_to_deploy()` modules. Both rewrite a
    trained multi-branch block as the single conv it is equivalent to at
    inference, so features agree to float precision -- and `mobileclip` is not
    on PyPI, so a container rebuilt without GitHub access lost the whole
    region-proposal stage over this one function (pyproject.toml)."""
    try:
        from mobileclip.modules.common.mobileone import reparameterize_model
    except ImportError:
        from timm.utils.model import reparameterize_model
    return reparameterize_model(model)


class FeatureEncoder:
    """Unit-norm 512-d features for crops and for text, one backbone at a time."""

    def __init__(
        self,
        backbone: str = "vitb32",
        checkpoint: str = "data/clip/ViT-B-32.pt",
        device: str = "cuda",
        half: bool = True,
    ) -> None:
        import open_clip
        import torch

        backbone = str(backbone).lower()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown feature backbone: {backbone!r} (want one of {BACKBONES})")
        self.torch = torch
        self.backbone = backbone
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.half = bool(half) and self.device.type == "cuda"

        name, pretrained = _OPEN_CLIP[backbone]
        if backbone == "vitb32":
            path = Path(checkpoint).expanduser()
            if not path.is_file():
                raise FileNotFoundError(
                    f"CLIP checkpoint not found: {path} "
                    "(stage it with scripts/download_weights.py --clip)"
                )
            pretrained = str(path)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained
        )
        if backbone == "mobileclip_s2":
            # Fuses the reparameterisable branches; DualMap does the same
            # (utils/object_detector.py:161-164) and inference differs without it.
            self.model = _reparameterize(self.model)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(name)
        self._split = split_preprocess(self.preprocess)
        # Both shipped backbones are 512-d; read it off the model where the
        # attribute exists rather than hard-coding, and never fail over it.
        self.dim = 512
        proj = getattr(self.model, "text_projection", None)
        shape = getattr(proj, "shape", None)
        if shape is not None and len(shape):
            self.dim = int(shape[-1])
        self._text_cache: Dict[Tuple[str, ...], np.ndarray] = {}
        self.n_images = 0
        self.n_text = 0

    # ------------------------------------------------------------------ images

    def encode_images(self, crops: Sequence[np.ndarray], batch_size: int = 128) -> np.ndarray:
        """(N, D) unit features for RGB uint8 crops. Empty input -> (0, D)."""
        usable = [c for c in crops if c is not None and getattr(c, "size", 0) > 0]
        if not usable:
            return np.empty((0, self.dim), dtype=np.float32)
        out: List[np.ndarray] = []
        for start in range(0, len(usable), batch_size):
            chunk = usable[start:start + batch_size]
            # Each crop's resize is independent and PIL releases the GIL for
            # it, so the region stage's crops (up to 64 a keyframe) resize on 4
            # threads; ToTensor + Normalize run on the device as a table lookup.
            # Measured with MobileCLIP-S2 on 77 keyframes (23 crops each): 18.1
            # -> 9.6 ms, and the input and the features bit-identical to the
            # serial loop's on every one.
            tensor = prepare_batch(chunk, self.preprocess, self._split, self.device, _prep_pool())
            with self.torch.inference_mode():
                if self.half:
                    with self.torch.autocast("cuda", dtype=self.torch.float16):
                        feats = self.model.encode_image(tensor)
                else:
                    feats = self.model.encode_image(tensor)
                feats = self.torch.nn.functional.normalize(feats.float(), dim=-1)
            out.append(feats.cpu().numpy().astype(np.float32))
        self.n_images += len(usable)
        return np.concatenate(out, axis=0)

    def encode_image(self, crop: np.ndarray) -> Optional[np.ndarray]:
        feats = self.encode_images([crop])
        return None if len(feats) == 0 else feats[0]

    # -------------------------------------------------------------------- text

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """(M, D) unit features, cached per prompt tuple (it changes once per
        episode, so the encoder must not run once per keyframe)."""
        key = tuple(str(t) for t in texts)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached
        with self.torch.inference_mode():
            tokens = self.tokenizer(list(key)).to(self.device)
            feats = self.model.encode_text(tokens)
            feats = self.torch.nn.functional.normalize(feats.float(), dim=-1)
        arr = feats.cpu().numpy().astype(np.float32)
        self._text_cache[key] = arr
        self.n_text += len(key)
        return arr

    def text_feature(self, text: str) -> np.ndarray:
        return self.encode_text([text])[0]

    def reset(self) -> None:
        """Episode boundary. The text cache is keyed on the prompt, so it stays
        valid across episodes and is deliberately NOT cleared here."""


def build_feature_encoder(cfg) -> Optional[FeatureEncoder]:
    """None when feature memory is off, so nothing is loaded."""
    fm = getattr(cfg, "feature_memory", None)
    if fm is None or not bool(getattr(fm, "enabled", False)):
        return None
    return FeatureEncoder(
        backbone=str(fm.backbone),
        checkpoint=str(fm.checkpoint),
        device=str(fm.device) if str(fm.device) != "auto" else cfg.detector.device,
        half=bool(fm.half),
    )
