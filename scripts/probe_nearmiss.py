#!/usr/bin/env python3
"""Why the detector missed the target on frames where it was plainly in view.

`eval.gt_dump_dir` leaves a directory of raw keyframes with the object's
projected pixel in each filename. Those frames are the whole experiment: every
"would it have been named if..." question is a re-run over them, which costs a
minute instead of the half hour a 500-step episode costs, and asks the question
against the FRAME THAT ACTUALLY FAILED rather than a synthetic best viewpoint.

Four conditions, chosen because they are the four different bugs:

  base        the run's own vocabulary and imgsz -- reproduces the miss
  nocompete   the same, minus the furniture classes whose masks swallow the
              object. An open-vocabulary head runs class-competitive NMS, so a
              10 cm can lying on a sofa cushion is competing with "sofa",
              "pillow" and "cushion" for its own pixels
  labels      one candidate name at a time, SWAPPED IN for the target's (adding
              them measures the competition instead of the name)
  zoom        a window around the projected pixel, upscaled to the full imgsz.
              This separates "the model cannot recognise this asset" from "the
              model cannot recognise 40 px", and only the second has a cheap fix

    python scripts/probe_nearmiss.py +experiment=ycb_dynamic_best \
        +probe.dump=outputs/nearmiss/base +probe.target='tin can' \
        +probe.out=outputs/nearmiss/probe
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig

from osg.core.config import register_configs
from osg.core.labels import normalize_label

register_configs()

# The classes that own the pixels a small object sits on. Not a guess: these are
# what `index.tsv` records under top_labels on the frames that missed.
COMPETITORS = ["sofa", "pillow", "cushion", "bed", "armchair", "chair",
               "clothes", "towel", "rug", "counter", "table", "desk", "shelf"]

CANDIDATES: Dict[str, List[str]] = {
    "tin can": ["tin can", "soup can", "can", "canned food", "red and white can",
                "tomato soup can", "food can", "soda can", "red can", "cylinder"],
    "red plate": ["red plate", "plate", "dish", "saucer", "round plate", "bowl",
                  "dinner plate", "red dish", "red disc"],
}

_UV = re.compile(r"_u(\d+)_v(\d+)\.png$")


def _frames(dump: Path):
    """(path, u, v) for every raw keyframe the dump wrote, deepest first."""
    out = []
    for raw in sorted(dump.glob("*/raw/*.png")):
        m = _UV.search(raw.name)
        if m:
            out.append((raw, int(m.group(1)), int(m.group(2))))
    return out


def _covers(det, u: float, v: float, pad: float = 8.0) -> bool:
    """Does this detection's box sit over the object's projected pixel?"""
    x1, y1, x2, y2 = [float(c) for c in det.bbox_xyxy]
    return x1 - pad <= u <= x2 + pad and y1 - pad <= v <= y2 + pad


def _hit(dets, want: str, u: float, v: float, pad: float = 8.0) -> float:
    """Best score for `want` on a box covering the object's projected pixel."""
    best = 0.0
    for det in dets:
        if normalize_label(det.label) == normalize_label(want) and _covers(det, u, v, pad):
            best = max(best, float(det.score))
    return best


def _zoom(rgb: np.ndarray, u: int, v: int, half: int, imgsz: int):
    """A square window around (u, v), upscaled. Returns (crop, u', v', scale)."""
    h, w = rgb.shape[:2]
    x0, y0 = max(0, u - half), max(0, v - half)
    x1, y1 = min(w, u + half), min(h, v + half)
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return None, 0, 0, 1.0
    scale = imgsz / max(crop.shape[0], crop.shape[1])
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return big, (u - x0) * scale, (v - y0) * scale, scale


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from osg.pipeline.components import build_detector

    probe = cfg.get("probe", {})
    dump = Path(str(probe.get("dump", "outputs/nearmiss/base")))
    target = str(probe.get("target", "tin can"))
    out_dir = Path(str(probe.get("out", "outputs/nearmiss/probe")))
    out_dir.mkdir(parents=True, exist_ok=True)
    halves = [int(v) for v in probe.get("zoom_half", [160, 96])]

    base_vocab = [str(v) for v in cfg.detector.vocabulary]
    detector = build_detector(cfg)
    imgsz = int(cfg.detector.imgsz)

    # Which episodes' frames to probe. The dump is keyed by episode id, and an
    # episode's target is not written into the path, so the caller names the
    # substring -- `probe.match=__17__` for the tin can's episodes. Without it
    # every frame in the dump is probed for this target, which measures the
    # detector on episodes whose object is something else entirely.
    match = str(probe.get("match", "") or "")
    frames = [f for f in _frames(dump) if match in str(f[0])]
    print(f"# {len(frames)} raw keyframes under {dump}"
          + (f" matching {match!r}" if match else " (ALL episodes)"))

    # Condition OUTER, frames inner. `set_vocabulary` re-runs the text encoder
    # over the whole class list, and swapping it per frame made the probe spend
    # all its time encoding the same thirteen vocabularies 139 times each --
    # slower than the episodes it exists to avoid re-running.
    rows = [{"frame": p.parent.parent.name + "/" + p.name, "u": u, "v": v}
            for p, u, v in frames]
    images = [cv2.imread(str(p))[..., ::-1] for p, _, _ in frames]
    detector.imgsz = imgsz

    def sweep(vocab, key: str, name: str, transform=None) -> None:
        detector.set_vocabulary(vocab)
        for rec, rgb, (_, u, v) in zip(rows, images, frames):
            img, uu, vv = (rgb, float(u), float(v)) if transform is None \
                else transform(rgb, u, v)
            rec[key] = 0.0 if img is None else _hit(detector.detect(img), name, uu, vv)
        print(f"#   {key} done")

    sweep(base_vocab, "base", target)
    # What the detector DID say where the object is: the near-miss labels are
    # the whole diagnosis when the target's own label scores zero.
    detector.set_vocabulary(base_vocab)
    for rec, rgb, (_, u, v) in zip(rows, images, frames):
        near = sorted((d for d in detector.detect(rgb) if _covers(d, u, v)),
                      key=lambda d: -float(d.score))[:3]
        rec["base_top"] = ",".join(f"{d.label}:{d.score:.2f}" for d in near) or "-"

    sweep([c for c in base_vocab
           if normalize_label(c) not in {normalize_label(x) for x in COMPETITORS}],
          "nocompete", target)

    for name in CANDIDATES.get(target, [target]):
        sweep([name if normalize_label(c) == normalize_label(target) else c
               for c in base_vocab], f"label:{name}", name)
    for rec in rows:
        best = max(CANDIDATES.get(target, [target]),
                   key=lambda n: rec.get(f"label:{n}", 0.0))
        rec["best_label"], rec["best_label_score"] = best, rec.get(f"label:{best}", 0.0)

    for half in halves:
        def _t(rgb, u, v, half=half):
            big, uu, vv, _ = _zoom(rgb, int(u), int(v), half, imgsz)
            return big, uu, vv
        sweep(base_vocab, f"zoom{half}", target, transform=_t)

    for rec in rows:
        print(json.dumps(rec))

    (out_dir / f"{target.replace(' ', '_')}.json").write_text(json.dumps(rows, indent=1))

    # --------------------------------------------------------------- summary
    def rate(key: str, thresh: float) -> str:
        vals = [r.get(key, 0.0) for r in rows]
        n = sum(1 for x in vals if x >= thresh)
        return f"{n}/{len(vals)} ({n/max(1,len(vals)):.2f})  mean {np.mean(vals):.3f}"

    gate = float(cfg.detector.class_conf.get(target, cfg.detector.conf)) \
        if cfg.detector.get("class_conf") else float(cfg.detector.conf)
    print(f"\n=== {target}  (admission gate {gate:.2f}, imgsz {imgsz}) ===")
    for key in ["base", "nocompete"] + [f"zoom{h}" for h in halves]:
        print(f"  {key:<14} named at gate: {rate(key, gate)}")
    print("  --- one name at a time, swapped in ---")
    for name in CANDIDATES.get(target, [target]):
        print(f"  {name:<24} {rate('label:' + name, gate)}")
    counts = defaultdict(int)
    for r in rows:
        counts[r["best_label"]] += 1
    print("  best name per frame:", dict(counts))


if __name__ == "__main__":
    main()
