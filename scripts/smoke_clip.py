"""Smoke test for the semantic value map's image-text backend.

Checks the two things that silently produce a useless value map rather than an
error: that the checkpoint loads from the staged path with no network, and that
image and text embeddings actually discriminate. A wrong preprocessing
normalisation, or a text-only checkpoint, still returns finite similarities --
they are just uninformative, and the value map degrades to noise that looks
plausible on a heatmap.

The nav container has no route to the CLIP CDN, so the checkpoint is staged
host-side into the bind-mounted data/clip (see scripts/download_weights.py
--clip). Run inside the container:

    python scripts/smoke_clip.py
    python scripts/smoke_clip.py --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from osg.perception.image_text import ClipScorer


def solid(channel: int) -> np.ndarray:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :, channel] = 220
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--root", default="data/clip")
    args = ap.parse_args()

    scorer = ClipScorer(device=args.device, half=False, download_root=args.root)
    print(f"loaded {args.root} on {args.device} (no network needed)")

    # Colour is the crudest possible semantic probe, which is the point: if this
    # fails, nothing subtler will work either.
    prompts = ["a red image", "a blue image"]
    ok = True
    for name, channel in (("red", 0), ("blue", 2)):
        s = scorer.score(solid(channel), prompts)
        pick = prompts[int(np.argmax(s))]
        ok &= pick == f"a {name} image"
        print(f"  {name:5s} -> {pick:14s} {np.round(s, 4)}")

    # A degenerate encoder returns near-identical scores for every prompt; the
    # spread matters as much as the argmax.
    spread = float(abs(scorer.score(solid(0), prompts)[0] - scorer.score(solid(0), prompts)[1]))
    print(f"  score spread on one image: {spread:.4f}")

    t0 = time.perf_counter()
    n = 10
    for _ in range(n):
        scorer.score(solid(0), ["Seems like there is a bed ahead."])
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"  {ms:.0f} ms/frame, text-cache entries={len(scorer._text_cache)}")

    if not ok or spread < 1e-3:
        sys.exit("FAIL: embeddings do not discriminate -- check preprocessing")
    print("PASS")


if __name__ == "__main__":
    main()
