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


def _hit(dets, want: str, u: float, v: float, pad: float = 8.0) -> float:
    """Best score for `want` on a box covering the object's projected pixel."""
    best = 0.0
    for det in dets:
        if normalize_label(det.label) != normalize_label(want):
            continue
        x1, y1, x2, y2 = [float(c) for c in det.bbox_xyxy]
        if x1 - pad <= u <= x2 + pad and y1 - pad <= v <= y2 + pad:
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

    frames = [f for f in _frames(dump) if target.replace(" ", "") in
              str(f[0]).replace(" ", "").lower() or True]
    print(f"# {len(frames)} raw keyframes under {dump}")

    rows = []
    for path, u, v in frames:
        rgb = cv2.imread(str(path))[..., ::-1]
        rec = {"frame": path.parent.parent.name + "/" + path.name, "u": u, "v": v}

        detector.set_vocabulary(base_vocab)
        detector.imgsz = imgsz
        dets = detector.detect(rgb)
        rec["base"] = _hit(dets, target, u, v)
        rec["base_top"] = ",".join(
            f"{d.label}:{d.score:.2f}" for d in sorted(
                (d for d in dets if _hit([d], d.label, u, v) > 0),
                key=lambda d: -d.score)[:3]) or "-"

        detector.set_vocabulary([c for c in base_vocab
                                 if normalize_label(c) not in
                                 {normalize_label(x) for x in COMPETITORS}])
        rec["nocompete"] = _hit(detector.detect(rgb), target, u, v)

        best_label, best_score = target, 0.0
        for name in CANDIDATES.get(target, [target]):
            swapped = [name if normalize_label(c) == normalize_label(target) else c
                       for c in base_vocab]
            detector.set_vocabulary(swapped)
            s = _hit(detector.detect(rgb), name, u, v)
            rec[f"label:{name}"] = s
            if s > best_score:
                best_label, best_score = name, s
        rec["best_label"], rec["best_label_score"] = best_label, best_score

        detector.set_vocabulary(base_vocab)
        for half in halves:
            big, uu, vv, scale = _zoom(rgb, u, v, half, imgsz)
            rec[f"zoom{half}"] = (0.0 if big is None
                                  else _hit(detector.detect(big), target, uu, vv))
        rows.append(rec)
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
