#!/usr/bin/env python3
"""Score ConceptGraphs maps of a recorded pass 1 against the same instance ground truth, and with
the same metrics, as the object layer's replays (scripts/analyze_ellipsoids.py).

Inputs:
  * the labelled recording (`--rec`, label_object_frames.py's gt.json and _instances/);
  * export_recording.py's scenes (`--cg`), each with ConceptGraphs' map under pcd_saves/ and the
    semantic image of every exported frame;
  * any number of object-layer replays (`--ours NAME=dir`).

A ConceptGraphs object is the union of the detections it merged (its `image_idx` / `mask`
lists). Each detection is labelled exactly as the recording's own detections are
(gt_semantic.label_masks: the majority instance under the mask, None below purity 0.5), on the
semantic image of its frame, sampled at the mask's resolution. The object becomes a "track" whose
extent is its point cloud's axis-aligned box, and analyze_ellipsoids.score_map scores it. Every
map is scored on ONE instance set: the instances the recording's detector labelled at least
twice (analyze_ellipsoids.evaluable_instances), which does not depend on either method.

    python scripts/baselines/conceptgraphs/score_recording.py --rec outputs/objrec_v2 \\
        --cg outputs/baselines/cg_rec --cg-map 'pcd_saves/full_pcd_none_*_post.pkl.gz' \\
        --ours I6r1=outputs/objreplay/I6r1_all --ours I6r1_s5=outputs/objreplay/I6r1_s5
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from collections import ChainMap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # scripts/
from analyze_ellipsoids import evaluable_instances, last_episode, load_gt, score_map  # noqa: E402

from osg.eval.gt_semantic import label_masks  # noqa: E402


def _sample(sem: np.ndarray, shape) -> np.ndarray:
    """Nearest-neighbour `sem` at `shape` (pixel centres)."""
    h, w = shape
    H, W = sem.shape
    rows = np.minimum(((np.arange(h) + 0.5) * H / h).astype(int), H - 1)
    cols = np.minimum(((np.arange(w) + 0.5) * W / w).astype(int), W - 1)
    return sem[rows][:, cols]


def cg_blob(scene_dir: Path, map_file: Path, labels: dict) -> dict:
    """ConceptGraphs' objects as a replay blob; puts each detection's label in `labels`."""
    with gzip.open(map_file, "rb") as fh:
        res = pickle.load(fh)
    sems = {}
    tracks, refs = [], {}
    for oid, obj in enumerate(res["objects"]):
        pts = np.asarray(obj["pcd_np"], float)
        if len(pts) == 0:
            continue
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        r = []
        for j, (img, mask) in enumerate(zip(obj["image_idx"], obj["mask"])):
            if img not in sems:
                sems[img] = np.load(scene_dir / "semantic" / f"sem{img:06d}.npz")["sem"].astype(np.int64)
            mask = np.asarray(mask, bool)
            (iid, _p), = label_masks(_sample(sems[img], mask.shape), [mask])
            key = ("cg", oid, j)
            labels[key] = iid
            r.append(list(key))
        tracks.append({"id": oid, "center": (0.5 * (lo + hi)).tolist(), "axes": (0.5 * (hi - lo)).tolist(),
                       "R": np.eye(3).reshape(-1).tolist(), "observations": [{}] * len(r),
                       "floor_key": 0, "linked_ids": []})
        refs[str(oid)] = r
    return {"tracks": tracks, "det_refs": refs}


def wilcoxon(a, b) -> str:
    from scipy.stats import wilcoxon as wx

    d = np.asarray(b, float) - np.asarray(a, float)
    if not np.any(d):
        return "p=1"
    return f"p={wx(a, b).pvalue:.2g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--cg", required=True, help="export_recording.py's output root")
    ap.add_argument("--cg-map", action="append", default=[],
                    help="NAME=glob under <cg>/<scene>/ (first match), e.g. CG='pcd_saves/*_post.pkl.gz'")
    ap.add_argument("--ours", action="append", default=[], help="NAME=replay dir")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--min-dets", type=int, default=2)
    ap.add_argument("--max-half-m", type=float, default=2.5)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    labels, insts, ep_meta = load_gt(Path(args.rec))
    cg_root = Path(args.cg)
    scenes = args.scenes or sorted(d.name for d in cg_root.iterdir() if (d / "frames.json").exists())
    arms = [("ours",) + tuple(v.split("=", 1)) for v in args.ours] + \
           [("cg",) + tuple(v.split("=", 1)) for v in args.cg_map]
    out = {name: {} for _, name, _ in arms}
    for s in scenes:
        gt = evaluable_instances(labels, insts, ep_meta, s, args.min_dets, args.max_half_m)
        for kind, name, where in arms:
            if kind == "ours":
                f = last_episode(Path(where), s)
            else:
                hits = sorted((cg_root / s).glob(where))
                if not hits:
                    print(f"  {name} {s}: no map matching {where}")
                    continue
                f = cg_root / s / f"_scored_{name}.json"
                own = {}
                f.write_text(json.dumps(cg_blob(cg_root / s, hits[0], own)))
                out[name][s] = score_map(f, ChainMap(own, labels), gt)
                continue
            out[name][s] = score_map(f, labels, gt)

    def rows(name, pred=lambda r: True):
        return {(s, i): r for s, res in out[name].items() for i, r in res["per_inst"].items() if pred(r)}

    def line(name, rr, scenes_):
        hit = [r for r in rr.values() if not r["missing"]]
        pur = np.mean([out[name][s]["purity"] for s in scenes_]) if scenes_ else float("nan")
        return (f"{name:10s} inst {len(rr):5d}  mapped {100 * len(hit) / max(len(rr), 1):5.1f}%  "
                f"extra/inst {np.mean([r['extra'] for r in rr.values()]):6.3f}  "
                f"dup% {100 * np.mean([r['extra'] > 0 for r in rr.values()]):5.1f}  "
                f"ctr_med {np.median([r['ctr_err'] for r in hit]):.3f}  "
                f"iou_med {np.median([r['iou'] for r in hit]):.3f}  "
                f"iou>=.25 {100 * np.mean([r['iou'] >= 0.25 for r in hit]):5.1f}%  "
                f"iou>=.5 {100 * np.mean([r['iou'] >= 0.5 for r in hit]):5.1f}%  "
                f"purity {pur:.3f}  objects {sum(out[name][s]['n_tracks'] for s in scenes_)}")

    names = [n for n in out if out[n]]
    common = sorted(set.intersection(*(set(out[n]) for n in names))) if names else []
    print(f"scenes scored by every arm: {common}")
    for label, pred in (("all", lambda r: True), ("hm3d", lambda r: r["source"] == "hm3d"),
                        ("ycb", lambda r: r["source"] == "ycb")):
        print(f"\n[{label}]")
        for n in names:
            rr = {k: r for k, r in rows(n, pred).items() if k[0] in common}
            print("  " + line(n, rr, common))
    print("\n[paired, instances mapped by both; first arm is the base]")
    base = names[0]
    for n in names[1:]:
        a, b = rows(base), rows(n)
        keys = [k for k in a if k[0] in common and k in b]
        ex = [a[k]["extra"] for k in keys], [b[k]["extra"] for k in keys]
        print(f"  {n} vs {base}  extra (all {len(keys)}): {np.mean(ex[0]):.3f} vs {np.mean(ex[1]):.3f} "
              f"{wilcoxon(*ex)}")
        both = [k for k in keys if not a[k]["missing"] and not b[k]["missing"]]
        for m in ("ctr_err", "iou"):
            x, y = [a[k][m] for k in both], [b[k][m] for k in both]
            print(f"    {m:7s} n={len(both)}  median {np.median(x):.3f} vs {np.median(y):.3f}  "
                  f"median diff {np.median(np.subtract(y, x)):+.3f}  {wilcoxon(x, y)}")
    print("\n[by scene: mapped% / extra / ctr_med / iou_med / purity]")
    for s in common:
        cells = []
        for n in names:
            rr = out[n][s]["per_inst"].values()
            hit = [r for r in rr if not r["missing"]]
            cells.append(f"{n} {100 * len(hit) / max(len(rr), 1):.0f}/{np.mean([r['extra'] for r in rr]):.2f}/"
                         f"{np.median([r['ctr_err'] for r in hit]):.3f}/{np.median([r['iou'] for r in hit]):.3f}/"
                         f"{out[n][s]['purity']:.3f}")
        print(f"  {s[:5]}  " + "  ".join(cells))
    if args.json:
        Path(args.json).write_text(json.dumps({n: {s: {"purity": v["purity"], "per_inst": {
            str(i): r for i, r in v["per_inst"].items()}} for s, v in out[n].items()} for n in names}))


if __name__ == "__main__":
    main()
