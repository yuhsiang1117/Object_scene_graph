#!/usr/bin/env python3
"""Our detector against DualMap's, on the frames our agent actually failed on.

DualMap names objects ours never does -- it scores `scissors` 9/18 on the
released benchmark where we score 1/18 -- and three explanations were live:
the name we hand the detector, the classes competing for the object's pixels,
and the model itself. `scripts/legacy/probe_nearmiss.py` has now closed the first two:
swapping in nine alternative names for the soup can moved recall by three
frames, and removing all thirteen competing classes moved it by zero, on four
different targets. That leaves the model.

This runs DualMap's own detector -- `yolov8l-world.pt` against its 305-class
list, both taken from the vendored checkout the benchmark ran from -- over the
same dumped keyframes, and asks the same question ours was asked: did anything
labelled with the target cover the object's projected pixel, and at what score.

It is not a like-for-like architecture comparison and is not meant to be. It is
the question that decides where the perception work goes: if DualMap's detector
also fails these frames, our ceiling is the benchmark's and the remaining work
is elsewhere; if it names them, the detector is the work.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

VENDOR = Path("outputs/dualmap_comparison/vendor/DualMap")
WEIGHTS = VENDOR / "model/yolov8l-world.pt"
CLASSES = VENDOR / "config/class_list/hm3d300_classes_ycb.txt"

# Our detector label -> the name DualMap's class list uses for the same asset.
# Both sides get the string their own system was built with; that is the point.
OURS_TO_THEIRS = {
    "tin can": "soup can",
    "red plate": "plate",
    "blue plastic pitcher": "pitcher",
    "cracker box": "cracker box",
    "bowl": "bowl",
    "banana": "banana",
    "mug": "mug",
    "scissors": "scissors",
}
# The dump directory names carry the benchmark's query, not our label.
MATCH_TO_OURS = {
    "soup_can": "tin can", "plate": "red plate", "pitcher": "blue plastic pitcher",
    "cracker_box": "cracker box", "bowl": "bowl", "banana": "banana",
    "mug": "mug", "scissors": "scissors",
}
_UV = re.compile(r"_u(\d+)_v(\d+)\.png$")


def frames(dump: Path, match: str):
    out = []
    for raw in sorted(dump.glob("*/raw/*.png")):
        if match not in str(raw):
            continue
        found = _UV.search(raw.name)
        if found:
            out.append((raw, int(found.group(1)), int(found.group(2))))
    return out


def covers(box, u: float, v: float, pad: float = 8.0) -> bool:
    x1, y1, x2, y2 = [float(c) for c in box]
    return x1 - pad <= u <= x2 + pad and y1 - pad <= v <= y2 + pad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=Path("outputs/osg_dualmap_tightring/gt_dump"))
    parser.add_argument("--out", type=Path, default=Path("outputs/nearmiss_tightring/BAKEOFF.md"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--gate", type=float, default=0.20)
    args = parser.parse_args()

    if not WEIGHTS.is_file():
        raise SystemExit(f"DualMap weights not found at {WEIGHTS}")
    names = [n.strip() for n in CLASSES.read_text(encoding="utf-8").splitlines() if n.strip()]

    from ultralytics import YOLOWorld

    model = YOLOWorld(str(WEIGHTS))
    model.set_classes(names)

    rows: List[Dict] = []
    for match, ours in sorted(MATCH_TO_OURS.items()):
        theirs = OURS_TO_THEIRS[ours]
        if theirs not in names:
            print(f"# {theirs!r} absent from DualMap's class list -- skipping")
            continue
        found = frames(args.dump, match)
        if not found:
            continue
        print(f"# {ours} ({theirs}): {len(found)} frames")
        hits = 0
        scores = []
        for path, u, v in found:
            image = cv2.imread(str(path))
            if image is None:
                continue
            result = model.predict(image[..., ::-1], imgsz=args.imgsz, verbose=False)[0]
            best = 0.0
            for box, cls, conf in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.cls.tolist(),
                result.boxes.conf.tolist(),
            ):
                if names[int(cls)] == theirs and covers(box, u, v):
                    best = max(best, float(conf))
            scores.append(best)
            hits += int(best >= args.gate)
        rows.append({
            "target": ours, "their_name": theirs, "frames": len(scores),
            "hits": hits, "recall": hits / max(len(scores), 1),
            "mean_score": float(np.mean(scores)) if scores else 0.0,
        })

    lines = [
        "# Our detector vs DualMap's, on the frames ours failed",
        "",
        f"`{WEIGHTS.name}` against its own {len(names)}-class list, over the raw "
        f"keyframes in `{args.dump}` -- the frames on which our agent had the object "
        f"genuinely in view. Gate {args.gate:.2f}, imgsz {args.imgsz}. "
        "Each system is given the name its own pipeline uses.",
        "",
        "| Target | Their class | Frames | DualMap recall | mean score |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target']} | `{row['their_name']}` | {row['frames']} | "
            f"{row['hits']}/{row['frames']} = {row['recall']:.2f} | {row['mean_score']:.3f} |"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
