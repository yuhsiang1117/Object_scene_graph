#!/usr/bin/env python
"""Does the commit gate's score know whether the target is actually there?

    python scripts/audit_gate_signal.py outputs/probe_itm

Reads a behaviour log and scores every step against the dataset's own goal
geometry, which the agent never sees: a step is POSITIVE when a true instance
of the goal category is within `--range` metres and inside the horizontal FOV.
Then reports, for each per-step signal in the trace, the area under the ROC --
the probability that a random positive step scores above a random negative one.

This exists because S26 asserted CLIP's whole-image cosine carried commit signal
here, measured it at AUC 0.479, and the conclusion ("no signal") was then
generalised to every image-text model. BLIP-2 ITM is a different model trained
for a different question, and that generalisation is exactly the kind of thing
that should be measured rather than inherited.

0.5 is a coin flip. Anything under about 0.6 cannot carry a gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_behaviour import load_goal_geometry  # noqa: E402


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U / |pos||neg|, ties counted as half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    both = np.concatenate([pos, neg])
    order = both.argsort(kind="mergesort")
    ranks = np.empty(len(both), float)
    ranks[order] = np.arange(1, len(both) + 1)
    # average ranks over ties, so a signal that is constant scores exactly 0.5
    _, inv, counts = np.unique(both, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--episodes-root",
                    default="data/datasets/objectnav/hm3d/v1/val/content")
    ap.add_argument("--range", type=float, default=5.0)
    ap.add_argument("--hfov", type=float, default=79.0)
    ap.add_argument("--fields", default="itm,det",
                    help="per-step trace keys to score")
    args = ap.parse_args()

    geo = load_goal_geometry(args.episodes_root)
    half = np.radians(args.hfov) / 2
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    pos: dict = {f: [] for f in fields}
    neg: dict = {f: [] for f in fields}
    n_steps = 0

    for line in (Path(args.run) / "episodes.jsonl").open():
        r = json.loads(line)
        g = geo.get(f"{r['scene']}:{r['episode_id']}")
        if not g or g[1] is None:
            continue
        objs = g[1]
        for s in r.get("step_trace") or []:
            yaw = s.get("yaw")
            if yaw is None:
                continue
            v = objs - np.asarray(s["xy"], float)
            d = np.linalg.norm(v, axis=1)
            ang = np.abs(np.arctan2(v[:, 1], v[:, 0]) - yaw)
            ang = np.minimum(ang, 2 * np.pi - ang)
            visible = bool(np.any((d < args.range) & (ang < half)))
            n_steps += 1
            for f in fields:
                (pos if visible else neg)[f].append(float(s.get(f) or 0.0))

    print(f"{args.run}: {n_steps} steps, "
          f"{len(pos[fields[0]])} with the target in frame within {args.range:.0f} m")
    print(f"\n  {'signal':8s} {'AUC':>7s} {'mean+':>9s} {'mean-':>9s} "
          f"{'p50+':>8s} {'p50-':>8s}")
    for f in fields:
        p, n = np.array(pos[f]), np.array(neg[f])
        print(f"  {f:8s} {auc(p, n):7.3f} {p.mean():9.4f} {n.mean():9.4f} "
              f"{np.median(p):8.4f} {np.median(n):8.4f}")

    # What a threshold would actually do, for the signal named first.
    f = fields[0]
    p, n = np.array(pos[f]), np.array(neg[f])
    print(f"\n  operating points for `{f}`")
    print(f"  {'thr':>6s} {'recall':>8s} {'fpr':>8s}")
    for thr in np.quantile(np.concatenate([p, n]), [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]):
        print(f"  {thr:6.3f} {float((p >= thr).mean()):8.3f} {float((n >= thr).mean()):8.3f}")


if __name__ == "__main__":
    main()
