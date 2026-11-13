#!/usr/bin/env python3
"""How good are the object layer's ellipsoids? Scored against instance ground truth.

Inputs are a labelled recording (scripts/label_object_frames.py wrote gt.json
and _instances/) and one or more replays of it (scripts/replay_object_layer.py).
Every replay is scored on the SAME set of ground-truth instances -- the ones
the detector labelled cleanly at least `--min-dets` times in the recording,
which does not depend on the variant -- so variants are paired instance by
instance.

    python scripts/analyze_ellipsoids.py --rec outputs/objrec_v2 \\
        --variant V0=outputs/objreplay/V0 --variant V1=outputs/objreplay/V1 \\
        --scenes 00800-TEEsavR23oF 00821-eF36g7L6Z9M 00862-LT9Jq6dN3Ea

Per instance, on the map at the end of each scene's last episode:

  extra     tracks whose majority instance is this one, minus one (0 if none):
            the duplication the work is about
  missing   no track at all
  ctr_err   ||centre - AABB centre|| of the instance's main track (most
            observations), with linking's component mean (`center_of`)
  iou       3D IoU of that track's world-axis extent box (sqrt diag of the
            shape matrix) against the AABB
  log_ext   per world axis, log(estimated half extent / true half extent)

A track's box is its `extent` when the replay wrote one
(scene_graph.ellipsoid.fuse_report: the fused box the map reports, while
association keeps the ellipsoid), else the ellipsoid's.

and per track, `purity`: the share of its labelled observations that are of
its majority instance -- the guard against merging two objects into one.

The first variant is the base; each other is compared to it with a paired
Wilcoxon signed-rank test and a bootstrap CI of the median paired difference.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

STOP_CATEGORIES = {"wall", "floor", "ceiling", "unknown", "", "misc", "void", "objects"}
BUCKETS = ((1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 10 ** 9, "6+"))


def bucket(n: int) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= n <= hi:
            return name
    return "0"


def shape_half_extents(rec) -> np.ndarray:
    R = np.asarray(rec["R"], float).reshape(3, 3)
    axes = np.asarray(rec["axes"], float)
    Q = R @ np.diag(axes ** 2) @ R.T
    return np.sqrt(np.maximum(np.diag(Q), 0.0))


def reported(rec):
    """(centre, half extents) of the box the map reports for this track."""
    e = rec.get("extent")
    if e is not None:
        return np.asarray(e["center"], float), np.asarray(e["axes"], float)
    return np.asarray(rec["center"], float), shape_half_extents(rec)


def iou3d(c1, h1, c2, h2) -> float:
    lo = np.maximum(c1 - h1, c2 - h2)
    hi = np.minimum(c1 + h1, c2 + h2)
    inter = float(np.prod(np.clip(hi - lo, 0.0, None)))
    union = float(np.prod(2 * h1) + np.prod(2 * h2) - inter)
    return inter / union if union > 0 else 0.0


def load_gt(rec_root: Path):
    labels = {}   # (episode, kf, det) -> instance id or None
    insts = {}    # (scene, layout) -> {id: instance}
    ep_meta = {}
    for d in sorted(rec_root.iterdir()):
        if not (d / "gt.json").exists():
            continue
        meta = json.loads((d / "prior.json").read_text()).get("meta", {})
        ep_meta[d.name] = meta
        gt = json.loads((d / "gt.json").read_text())
        for k, dets in enumerate(gt["kf"]):
            for i, (iid, _p) in enumerate(dets):
                labels[(d.name, k, i)] = iid
    for f in (rec_root / "_instances").glob("*.json"):
        scene, layout = f.stem.split("__", 1)
        insts[(scene, layout)] = {int(k): v for k, v in json.loads(f.read_text()).items()}
    return labels, insts, ep_meta


def evaluable_instances(labels, insts, ep_meta, scene, min_dets, max_half_m):
    """Instances labelled at least `min_dets` times in this scene's recording,
    excluding structure (walls, floors) and anything too big to be an object."""
    counts = Counter(iid for (ep, _k, _i), iid in labels.items()
                     if iid is not None and ep_meta[ep].get("scene") == scene)
    layouts = {ep_meta[ep].get("layout_id") for ep in ep_meta if ep_meta[ep].get("scene") == scene}
    table = {}
    for lay in layouts:
        table.update(insts.get((scene, lay), {}))
    out = {}
    for iid, n in counts.items():
        g = table.get(iid)
        if g is None or n < min_dets:
            continue
        if str(g["category"]).lower() in STOP_CATEGORIES or max(g["half"]) > max_half_m:
            continue
        out[iid] = g
    return out


def score_map(replay_file: Path, labels, gt_insts) -> Dict:
    blob = json.loads(replay_file.read_text())
    tracks = {int(t["id"]): t for t in blob["tracks"]
              if not t.get("blacklisted") and not t.get("proposal_only")}
    refs = blob.get("det_refs", {})
    owner = {}       # track id -> (majority instance, purity, n labelled)
    for tid, t in tracks.items():
        ids = [labels.get(tuple(r)) for r in refs.get(str(tid), []) if r is not None]
        ids = [i for i in ids if i is not None]
        if not ids:
            owner[tid] = (None, 0.0, 0)
            continue
        iid, n = Counter(ids).most_common(1)[0]
        owner[tid] = (iid, n / len(ids), len(ids))

    def center_of(t):
        cs = [reported(t)[0]]
        for lid in t.get("linked_ids", []):
            o = tracks.get(int(lid))
            if o is not None and o["floor_key"] == t["floor_key"]:
                cs.append(reported(o)[0])
        return np.mean(cs, axis=0)

    by_inst = defaultdict(list)
    for tid, (iid, _p, _n) in owner.items():
        if iid is not None:
            by_inst[iid].append(tid)
    per_inst = {}
    for iid, g in gt_insts.items():
        tids = by_inst.get(iid, [])
        row = {"n_tracks": len(tids), "extra": max(len(tids) - 1, 0), "missing": not tids,
               "category": g["category"], "source": g["source"]}
        if tids:
            main = max(tids, key=lambda i: (len(tracks[i]["observations"]), -i))
            t = tracks[main]
            c_gt, h_gt = np.asarray(g["center"], float), np.asarray(g["half"], float)
            c, h = reported(t)
            row.update(
                ctr_err=float(np.linalg.norm(center_of(t) - c_gt)),
                iou=iou3d(c, h, c_gt, h_gt),
                log_ext=[float(v) for v in np.log(np.maximum(h, 1e-3) / np.maximum(h_gt, 1e-3))],
                n_obs=len(t["observations"]),
            )
        per_inst[iid] = row
    labelled = [(p, n) for (iid, p, n) in owner.values() if iid is not None]
    purity = (sum(p * n for p, n in labelled) / max(sum(n for _, n in labelled), 1))
    kf = blob.get("kf_seconds", [])
    return {
        "per_inst": per_inst,
        "purity": float(purity),
        "n_tracks": len(tracks),
        "n_unlabelled_tracks": sum(1 for iid, _p, _n in owner.values() if iid is None),
        "kf_ms_mean": 1000.0 * float(np.mean(kf)) if kf else float("nan"),
        "funnel": blob.get("funnel", {}),
    }


def last_episode(replay_dir: Path, scene: str) -> Path:
    files = []
    for f in replay_dir.glob("*.json"):
        meta = json.loads(f.read_text()).get("meta", {})
        if meta.get("scene") == scene:
            files.append((float(meta.get("wall_start", 0.0)), f))
    if not files:
        raise SystemExit(f"{replay_dir}: no episode of {scene}")
    return max(files)[1]


def paired(a: List[float], b: List[float], rng) -> str:
    from scipy.stats import wilcoxon

    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    if len(d) == 0:
        return "n=0"
    if np.allclose(d, 0):
        return f"n={len(d)} identical"
    p = float(wilcoxon(a, b, zero_method="zsplit").pvalue)
    boots = [np.median(rng.choice(d, len(d))) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (f"n={len(d)} median diff {np.median(d):+.3f} [{lo:+.3f},{hi:+.3f}] "
            f"mean {np.mean(d):+.3f} p={p:.3g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--variant", action="append", required=True, help="NAME=replay_dir; first is the base")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--min-dets", type=int, default=2)
    ap.add_argument("--max-half-m", type=float, default=2.5)
    ap.add_argument("--json", default="", help="write the per-instance table here")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    labels, insts, ep_meta = load_gt(Path(args.rec))
    scenes = args.scenes or sorted({m.get("scene") for m in ep_meta.values()})
    variants = [v.split("=", 1) for v in args.variant]

    results = {name: {} for name, _ in variants}
    gt_by_scene = {s: evaluable_instances(labels, insts, ep_meta, s, args.min_dets, args.max_half_m)
                   for s in scenes}
    for name, d in variants:
        for s in scenes:
            results[name][s] = score_map(last_episode(Path(d), s), labels, gt_by_scene[s])

    def rows(name, pred=lambda r: True):
        return [(s, iid, r) for s in scenes for iid, r in results[name][s]["per_inst"].items() if pred(r)]

    print(f"evaluable instances: " + ", ".join(f"{s[:5]} {len(gt_by_scene[s])}" for s in scenes))
    hdr = f"{'variant':10s} {'tracks':>6s} {'extra/inst':>10s} {'dup%':>6s} {'miss%':>6s} " \
          f"{'ctr_med':>7s} {'iou_med':>7s} {'|logext|':>8s} {'purity':>6s} {'kf_ms':>6s}"
    print(hdr)
    for name, _ in variants:
        rr = [r for _, _, r in rows(name)]
        found = [r for r in rr if not r["missing"]]
        print(f"{name:10s} {sum(results[name][s]['n_tracks'] for s in scenes):6d} "
              f"{np.mean([r['extra'] for r in rr]):10.3f} "
              f"{100 * np.mean([r['extra'] > 0 for r in rr]):6.1f} "
              f"{100 * np.mean([r['missing'] for r in rr]):6.1f} "
              f"{np.median([r['ctr_err'] for r in found]):7.3f} "
              f"{np.median([r['iou'] for r in found]):7.3f} "
              f"{np.median([np.mean(np.abs(r['log_ext'])) for r in found]):8.3f} "
              f"{np.mean([results[name][s]['purity'] for s in scenes]):6.3f} "
              f"{np.mean([results[name][s]['kf_ms_mean'] for s in scenes]):6.1f}")

    print("\nby n_obs of the base's main track (median ctr_err / iou):")
    base = variants[0][0]
    keyed = {name: {(s, iid): r for s, iid, r in rows(name)} for name, _ in variants}
    for _lo, _hi, b in BUCKETS:
        keys = [k for k, r in keyed[base].items() if not r["missing"] and bucket(r["n_obs"]) == b]
        line = f"  n_obs {b:>3s} ({len(keys):4d}): "
        for name, _ in variants:
            rs = [keyed[name][k] for k in keys if not keyed[name][k]["missing"]]
            if rs:
                line += f"{name} {np.median([r['ctr_err'] for r in rs]):.3f}/{np.median([r['iou'] for r in rs]):.3f}  "
        print(line)

    for src in ("ycb", "hm3d"):
        keys = [k for k, r in keyed[base].items() if r["source"] == src]
        line = f"  {src:4s} ({len(keys):4d}): "
        for name, _ in variants:
            rs = [keyed[name][k] for k in keys if not keyed[name][k]["missing"]]
            if rs:
                line += (f"{name} ctr {np.median([r['ctr_err'] for r in rs]):.3f} "
                         f"extra {np.mean([keyed[name][k]['extra'] for k in keys]):.2f}  ")
        print(line)

    if len(variants) > 1:
        print(f"\npaired against {base} (per instance; negative is better for extra/ctr_err):")
        for name, _ in variants[1:]:
            keys = sorted(keyed[base])
            print(f"  {name}:")
            print(f"    extra    {paired([keyed[base][k]['extra'] for k in keys], [keyed[name][k]['extra'] for k in keys], rng)}")
            both = [k for k in keys if not keyed[base][k]['missing'] and not keyed[name][k]['missing']]
            for m in ("ctr_err", "iou"):
                print(f"    {m:8s} {paired([keyed[base][k][m] for k in both], [keyed[name][k][m] for k in both], rng)}")
            small = [k for k in both if keyed[base][k]["n_obs"] <= 2]
            for m in ("ctr_err", "iou"):
                print(f"    {m:8s} (base n_obs<=2) "
                      f"{paired([keyed[base][k][m] for k in small], [keyed[name][k][m] for k in small], rng)}")
            ycb = [k for k in both if keyed[base][k]["source"] == "ycb"]
            print(f"    ycb ctr  {paired([keyed[base][k]['ctr_err'] for k in ycb], [keyed[name][k]['ctr_err'] for k in ycb], rng)}")
            miss = sum(keyed[name][k]["missing"] for k in keys) - sum(keyed[base][k]["missing"] for k in keys)
            print(f"    missing  {miss:+d} instances")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {name: {s: {str(i): r for i, r in results[name][s]["per_inst"].items()} for s in scenes}
             for name, _ in variants}))


if __name__ == "__main__":
    main()
