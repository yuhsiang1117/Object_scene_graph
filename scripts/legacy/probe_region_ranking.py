#!/usr/bin/env python3
"""Given a class-agnostic region on the object, can appearance find it?

`probe_perception_gap.py` established the first half: FastSAM puts a tight
region on the objects we fail on 72-100% of frames, where YOLOE puts one on
0-48% and otherwise proposes the furniture (`mirror` for scissors, `lamp` for a
soup can, `chair` for a banana). That is DualMap's structural advantage.

This measures the second half, which decides whether a live proposal stage is
worth building: among ALL of a frame's regions, where does the one on the
object rank by cosine against the query text? A proposer that finds the region
is useless if the matcher cannot pick it out of eighty others.

Reports rank-1 / rank-5 rates by range band, and the margin between the true
region and the best distractor, per backbone.
"""
from __future__ import annotations

import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_perception_gap import frames, covers, area  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=Path("outputs/osg_dualmap_tightring/gt_dump"))
    ap.add_argument("--queries", nargs="+",
                    default=["scissors", "soup_can", "cracker_box", "banana"])
    ap.add_argument("--backbone", default="mobileclip_s2", choices=["mobileclip_s2", "vitb32"])
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--tight-frac", type=float, default=0.02)
    ap.add_argument("--tight-ratio", type=float, default=0.25)
    ap.add_argument("--max-regions", type=int, default=64)
    ap.add_argument("--prompt", default="a photo of a {q}")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("outputs/REGION_RANKING.md"))
    args = ap.parse_args()

    from ultralytics import FastSAM
    from osg.perception.feature_encoder import FeatureEncoder

    sam = FastSAM("outputs/dualmap_comparison/vendor/DualMap/model/FastSAM-s.pt")
    enc = FeatureEncoder(args.backbone, "", args.device, half=True)

    work = frames(args.dump, args.queries, args.limit)
    rows: List[dict] = []
    for query, items in work.items():
        text = enc.text_feature(args.prompt.format(q=query.replace("_", " ")))
        for it in items:
            img = Image.open(it["path"]).convert("RGB")
            W, H = img.size
            r = sam.predict(np.asarray(img), imgsz=1024, conf=0.25, iou=0.7,
                            retina_masks=True, verbose=False, device=args.device)[0]
            if r.boxes is None or len(r.boxes) == 0:
                rows.append({"query": query, "range_m": it["range_m"], "found": False})
                continue
            boxes = [[float(x) for x in b] for b in r.boxes.xyxy.detach().cpu().numpy()]
            boxes = boxes[: args.max_regions]
            cov = [b for b in boxes if covers(b, it["u"], it["v"])]
            if not cov:
                rows.append({"query": query, "range_m": it["range_m"], "found": False})
                continue
            big = max(area(b) for b in cov)
            tight = [b for b in cov
                     if area(b) <= args.tight_frac * W * H and area(b) <= args.tight_ratio * big]
            if not tight:
                rows.append({"query": query, "range_m": it["range_m"], "found": False})
                continue
            crops = [img.crop((int(b[0]), int(b[1]), int(b[2]), int(b[3]))) for b in boxes]
            crops = [c for c in crops if c.size[0] >= 2 and c.size[1] >= 2]
            if not crops:
                rows.append({"query": query, "range_m": it["range_m"], "found": False})
                continue
            feats = enc.encode_images([np.asarray(c) for c in crops])
            sims = feats @ text
            truth = {id(b) for b in tight}
            idx = [i for i, b in enumerate(boxes) if id(b) in truth and i < len(sims)]
            if not idx:
                rows.append({"query": query, "range_m": it["range_m"], "found": False})
                continue
            best_true = max(float(sims[i]) for i in idx)
            others = [float(sims[i]) for i in range(len(sims)) if i not in idx]
            rank = 1 + sum(1 for s in others if s > best_true)
            rows.append({"query": query, "range_m": it["range_m"], "found": True,
                         "n_regions": len(sims), "rank": rank,
                         "margin": best_true - (max(others) if others else 0.0),
                         "sim_true": best_true})

    L: List[str] = []
    def say(s: str = "") -> None:
        print(s); L.append(s)
    say(f"# Ranking the object's region among all of a frame's regions ({args.backbone})")
    say()
    say(f"{len(rows)} frames. A frame counts only when FastSAM actually produced a tight")
    say("region on the object; `rank 1` means appearance picked it over every other region")
    say("in the frame.")
    say()
    say("| query | frames | tight region found | rank 1 | rank <=5 | median regions |")
    say("|---|---:|---:|---:|---:|---:|")
    for query in args.queries:
        s = [r for r in rows if r["query"] == query]
        f = [r for r in s if r["found"]]
        if not s:
            continue
        r1 = sum(1 for r in f if r["rank"] == 1)
        r5 = sum(1 for r in f if r["rank"] <= 5)
        med = int(np.median([r["n_regions"] for r in f])) if f else 0
        say(f"| {query} | {len(s)} | {len(f)} | {r1} ({100*r1/max(len(f),1):.0f}%) | "
            f"{r5} ({100*r5/max(len(f),1):.0f}%) | {med} |")
    f = [r for r in rows if r["found"]]
    if f:
        r1 = sum(1 for r in f if r["rank"] == 1)
        r5 = sum(1 for r in f if r["rank"] <= 5)
        say(f"| **all** | {len(rows)} | {len(f)} | {r1} ({100*r1/len(f):.0f}%) | "
            f"{r5} ({100*r5/len(f):.0f}%) | {int(np.median([r['n_regions'] for r in f]))} |")
    say()
    say("## By range")
    say()
    say("| band | frames with a tight region | rank 1 | rank <=5 |")
    say("|---|---:|---:|---:|")
    for lo, hi, lbl in [(0, 1, "<1 m"), (1, 2, "1-2 m"), (2, 3, "2-3 m"), (3, 99, ">=3 m")]:
        s = [r for r in rows if r.get("range_m") is not None and lo <= r["range_m"] < hi]
        ff = [r for r in s if r["found"]]
        if not s:
            continue
        r1 = sum(1 for r in ff if r["rank"] == 1)
        r5 = sum(1 for r in ff if r["rank"] <= 5)
        say(f"| {lbl} | {len(ff)}/{len(s)} | {r1} | {r5} |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(rows, open(str(args.out).replace(".md", ".json"), "w"), indent=1, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
