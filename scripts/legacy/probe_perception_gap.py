#!/usr/bin/env python3
"""Why does the detector emit nothing on the objects we fail?

21 of the 50 remaining failures on `_sensor_v2` are perception: the target was
in view, often close and centred and fully visible, and was never named. The
episode instrument records a zero box and a zero score for those trials, but it
only counts detections whose label ALREADY matches the target
(`eval/instruments.py:154`), so a zero there means "no correctly-labelled box",
not "no box". Those are different defects with different fixes:

  a box is proposed, wrongly labelled  -> rank regions by appearance (DualMap's
                                          structure; the matcher is the fix)
  no box is proposed at all            -> the region proposer is the fix, and
                                          no amount of matching helps

This measures which, on the dumped keyframes where the object's pixel is known,
for three proposers:

  yoloe    the live detector at the live settings, every detection, any label
  fastsam  class-agnostic segmentation (DualMap's FastSAM-s)
  sam      MobileSAM, the one the merge brought in

and reports, per query:

  cover       any proposal contains the object's pixel
  tight       a proposal contains it AND is small enough to BE the object
              rather than the furniture it rests on (area <= --tight-frac of
              the frame, and <= --tight-ratio of the largest covering box)
  label       what the covering detections were called

Usage:
  python scripts/legacy/probe_perception_gap.py --queries scissors soup_can cracker_box banana
"""
from __future__ import annotations

import argparse, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

UV_RE = re.compile(r"_u(?P<u>\d+)_v(?P<v>\d+)\.png$")


def frames(dump: Path, queries: List[str], limit_per_query: int) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = defaultdict(list)
    for d in sorted(dump.iterdir()):
        if not d.is_dir():
            continue
        query = d.name.rsplit("__", 1)[-1]
        if query not in queries:
            continue
        if len(out[query]) >= limit_per_query:
            continue
        ranges: Dict[int, float] = {}
        index = d / "index.tsv"
        if index.exists():
            for line in index.read_text(encoding="utf-8").splitlines()[1:]:
                p = line.split("\t")
                if len(p) > 2:
                    try:
                        ranges[int(p[0])] = float(p[2])
                    except ValueError:
                        pass
        for png in sorted((d / "raw").glob("*.png")):
            m = UV_RE.search(png.name)
            if not m:
                continue
            kf = int(png.name[2:6]) if png.name[:2] == "kf" else -1
            out[query].append({
                "path": png, "u": int(m.group("u")), "v": int(m.group("v")),
                "range_m": ranges.get(kf), "trial": d.name,
            })
            if len(out[query]) >= limit_per_query:
                break
    return out


def covers(box, u, v, pad=8.0) -> bool:
    return box[0] - pad <= u <= box[2] + pad and box[1] - pad <= v <= box[3] + pad


def area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def yoloe_boxes(det_model, image: np.ndarray):
    dets = det_model.detect(image)
    return [([float(c) for c in d.bbox_xyxy], str(d.label), float(d.score)) for d in dets]


def sam_boxes(model, image: np.ndarray, imgsz: int, conf: float, device: str):
    r = model.predict(image, imgsz=imgsz, conf=conf, iou=0.7, retina_masks=True,
                      verbose=False, device=device)[0]
    if r.boxes is None:
        return []
    b = r.boxes.xyxy.detach().cpu().numpy()
    return [([float(x) for x in box], "<region>", 1.0) for box in b]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=Path("outputs/osg_dualmap_tightring/gt_dump"))
    ap.add_argument("--queries", nargs="+",
                    default=["scissors", "soup_can", "cracker_box", "banana", "mug", "pitcher"])
    ap.add_argument("--proposers", nargs="+", default=["yoloe", "fastsam", "sam"])
    ap.add_argument("--preset", default="dualmap_protocol_osg_sensor_v2")
    ap.add_argument("--limit", type=int, default=120, help="frames per query")
    ap.add_argument("--tight-frac", type=float, default=0.02,
                    help="a proposal this fraction of the frame or smaller could BE the object")
    ap.add_argument("--tight-ratio", type=float, default=0.25,
                    help="...and this fraction of the largest covering proposal")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("outputs/PERCEPTION_GAP.md"))
    args = ap.parse_args()

    work = frames(args.dump, args.queries, args.limit)
    print({k: len(v) for k, v in work.items()})

    models: Dict[str, Any] = {}
    if "yoloe" in args.proposers:
        from hydra import compose, initialize_config_dir
        from osg.core.config import register_configs
        from osg.pipeline.components import build_detector
        from osg.perception.vocabulary import target_vocabulary
        register_configs()
        with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
            cfg = compose(config_name="config", overrides=[f"+experiment={args.preset}"])
        models["yoloe"] = ("yoloe", build_detector(cfg), cfg)
    if "fastsam" in args.proposers:
        from ultralytics import FastSAM
        p = Path("outputs/dualmap_comparison/vendor/DualMap/model/FastSAM-s.pt")
        if p.exists():
            models["fastsam"] = ("sam", FastSAM(str(p)), None)
    if "sam" in args.proposers:
        from ultralytics import SAM
        p = Path("data/weights/mobile_sam.pt")
        if p.exists():
            models["sam"] = ("sam", SAM(str(p)), None)

    rows: List[dict] = []
    for query, items in work.items():
        agent_label = query.replace("_", " ")
        for name, (kind, model, cfg) in models.items():
            if kind == "yoloe":
                from osg.perception.vocabulary import target_vocabulary
                model.set_vocabulary(target_vocabulary(agent_label, cfg.detector.vocabulary))
            for it in items:
                img = Image.open(it["path"]).convert("RGB")
                arr = np.asarray(img)
                W, H = img.size
                if kind == "yoloe":
                    boxes = yoloe_boxes(model, arr)
                else:
                    boxes = sam_boxes(model, arr, 1024, 0.25, args.device)
                cov = [b for b in boxes if covers(b[0], it["u"], it["v"])]
                tight = [b for b in cov
                         if area(b[0]) <= args.tight_frac * W * H
                         and (not cov or area(b[0]) <= args.tight_ratio * max(area(c[0]) for c in cov))]
                rows.append({
                    "query": query, "proposer": name, "range_m": it["range_m"],
                    "n_boxes": len(boxes), "cover": len(cov) > 0, "tight": len(tight) > 0,
                    "cover_labels": [b[1] for b in cov],
                    "tight_labels": [b[1] for b in tight],
                    "smallest_cover_px": min((area(b[0]) for b in cov), default=None),
                })

    L: List[str] = []
    def say(s: str = "") -> None:
        print(s); L.append(s)
    say("# Does anything propose a region on the objects we fail?")
    say()
    say(f"{len(rows)} (frame, proposer) measurements. `tight` means a proposal covers the")
    say(f"object's pixel AND is at most {args.tight_frac:.0%} of the frame and {args.tight_ratio:.0%}")
    say("of the largest covering proposal -- i.e. it could BE the object rather than the")
    say("furniture it rests on.")
    say()
    say("| query | proposer | frames | any box covers it | a TIGHT box covers it |")
    say("|---|---|---:|---:|---:|")
    for query in args.queries:
        for name in models:
            s = [r for r in rows if r["query"] == query and r["proposer"] == name]
            if not s:
                continue
            c = sum(r["cover"] for r in s)
            t = sum(r["tight"] for r in s)
            say(f"| {query} | {name} | {len(s)} | {c} ({100*c/len(s):.0f}%) | {t} ({100*t/len(s):.0f}%) |")
    say()
    say("## What the covering detections were called (yoloe)")
    say()
    for query in args.queries:
        s = [r for r in rows if r["query"] == query and r["proposer"] == "yoloe"]
        lab = Counter(l for r in s for l in r["cover_labels"])
        if lab:
            say(f"- **{query}**: " + ", ".join(f"`{k}` x{v}" for k, v in lab.most_common(8)))
    say()
    say("## By range (tight-proposal rate)")
    say()
    say("| proposer | <1 m | 1-2 m | 2-3 m | >=3 m |")
    say("|---|---:|---:|---:|---:|")
    bands = [(0, 1), (1, 2), (2, 3), (3, 99)]
    for name in models:
        cells = []
        for lo, hi in bands:
            s = [r for r in rows if r["proposer"] == name and r["range_m"] is not None
                 and lo <= r["range_m"] < hi]
            cells.append(f"{sum(r['tight'] for r in s)}/{len(s)}" if s else "-")
        say(f"| {name} | " + " | ".join(cells) + " |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(rows, open(str(args.out).replace(".md", ".json"), "w"), indent=1, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
