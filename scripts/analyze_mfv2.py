#!/usr/bin/env python3
"""Benchmark v2 campaign tables from `episodes.jsonl` alone: every arm per seed, each
ablation paired against the arm it removes one mechanism from, and where the wall time goes.

A campaign root holds `<arm>_s<seed>/**/episodes.jsonl` -- `scripts/run_mfv2.sh`'s layout,
kept as it is by the release bundle (`scripts/release/package.py`). An episode is keyed by
(seed, episode_id): the ids carry the scene and repeat across seeds. Success is on the
object's storey, re-scored from the episode's own stop poses (`analyze_multifloor.scored`).

  arms       every arm x seed: successes, first-attempt successes, failure buckets, commits
  ablations  each ablation against its base arm on the episodes both ran: successes,
             discordant pairs (ablation better / worse), exact two-sided sign test, buckets.
             Runs are deterministic at a fixed seed, so the episodes pair one to one.
  wall       wall time per arm, and against a reference on the episodes whose step count
             is identical in both (the same trajectory, so the difference is compute)

Each failed episode lands in exactly one bucket:
  A_down / A_up  started on another storey and never reached the object's
  B              started on the object's storey, left it, and failed
  C1 .. C4       on the object's storey: never in view / in view, never detected /
                 detected, never committed / committed to the true object, still failed
TRUE / STALE / FP count the goal commits of the C episodes (`analyze_multifloor.commits`):
at the object, at its prior pose, anywhere else.

    python scripts/analyze_mfv2.py arms --root outputs/mfv2 --csv arms_by_seed.csv
    python scripts/analyze_mfv2.py ablations --root <bundle>/benchmark_v2/runs
    python scripts/analyze_mfv2.py ablations --root outputs/mfv2 --set mouth_walk
    python scripts/analyze_mfv2.py wall --root outputs/mfv2 \\
        --ref osg_landing_vet_live osg_landing_vet_live_follows
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_multifloor import FLOOR_DY_M, commits, load, scored, target_floor_y  # noqa: E402

SEEDS = (42, 43, 44)
BUCKETS = ("A_down", "A_up", "B", "C1", "C2", "C3", "C4")
KINDS = ("TRUE", "STALE", "FP")

# What each ablation removes, in the order of the paper's ablation table. The final arm is
# `mfv2_osg_landing_vet_live`; every `mfv2_osg_vet_live_<x>` preset changes one thing on it.
FINAL = "osg_landing_vet_live"
SETS: Dict[str, Tuple[str, List[Tuple[str, str]]]] = {
    "final": (FINAL, [
        ("storey belief", "osg_vet_live_no_belief"),
        ("stair route", "osg_vet_live_no_route"),
        ("presence belief", "osg_vet_live_no_presence"),
        ("search posterior", "osg_vet_live_no_search"),
        ("remembered stair mouth", "osg_vet_live_no_arrival"),
        ("STOP gate and vetting", "osg_vet_live_no_gate"),
        ("affinity table -> all LLM", "osg_vet_live_affinity_llm"),
        ("affinity table -> flat", "osg_vet_live_affinity_flat"),
        ("floor LLM", "osg_vet_live_no_floor_llm"),
        ("absence VLM", "osg_vet_live_no_vlm"),
        ("stale-twin retirement *", "osg_vet_live_no_twins"),
        ("ceiling: navmesh climbs", "osg_vet_live_oracle"),
        ("ceiling: GT vetting, FP only", "osg_landing_vet_oracle_fp"),
        ("ceiling: GT vetting, full", "osg_landing_vet_oracle_full"),
        ("ceiling: GT refutes, every storey", "osg_landing_vet_oracle_sweep"),
    ]),
    # The ablations of the previous final arm (`mfv2_osg_mouth_walk`).
    "mouth_walk": ("osg_mouth_walk", [
        (x, f"osg_mouth_walk_{x}") for x in
        ("no_belief", "no_route", "no_search", "no_arrival", "no_gate", "affinity_llm",
         "no_presence", "no_vlm", "affinity_flat", "vet", "oracle")
    ]),
}
SUBSET_NOTE = ("* run only on the episodes of the final arm where the mechanism fired "
               "(`agent_stats.stale_twins_retired`), so the pair covers those alone.")

Episodes = Dict[Tuple[int, str], Dict[str, Any]]


def run_dirs(root: Path) -> "OrderedDict[str, Dict[int, Path]]":
    """arm -> {seed: dir} for every `<arm>_s<seed>` below root (names starting `_` skipped)."""
    out: "OrderedDict[str, Dict[int, Path]]" = OrderedDict()
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        m = re.fullmatch(r"(.+)_s(\d+)", d.name)
        if m:
            out.setdefault(m.group(1), {})[int(m.group(2))] = d
    return out


def episodes(root: Path, arm: str, seeds=SEEDS) -> Episodes:
    out: Episodes = {}
    for seed in seeds:
        d = root / f"{arm}_s{seed}"
        if d.is_dir():
            for e in load(d):
                out[(seed, e["episode_id"])] = e  # a re-run episode: the last record wins
    return out


def bucket(e: Dict[str, Any]) -> Optional[str]:
    """The failure bucket of an episode; None when it succeeded."""
    if scored(e)["success"]:
        return None
    fy, sy = target_floor_y(e.get("authored_layout") or {}), e.get("start_y")
    if fy is None or sy is None:
        return "unknown"
    start_on = abs(float(sy) - float(fy)) <= FLOOR_DY_M
    yr = e.get("traj_y_range")
    if not start_on and not e.get("goal_floor_reached"):
        return "A_up" if fy > float(sy) else "A_down"
    if start_on and yr is not None and float(yr) > FLOOR_DY_M:
        return "B"
    kinds = [k for k, _ in commits(e)]
    if (e.get("gt_in_view_frames") or 0) == 0:
        return "C1"
    if (e.get("gt_kf_detected") or 0) == 0:
        return "C2"
    return "C4" if "TRUE" in kinds else "C3"


def tally(eps: List[Dict[str, Any]]) -> Dict[str, Any]:
    b, c = Counter(), Counter()
    ok = first = 0
    for e in eps:
        s = scored(e)
        ok += s["success"]
        first += s["success_first"]
        k = bucket(e)
        if k:
            b[k] += 1
            if k.startswith("C"):
                c.update(kind for kind, _ in commits(e))
    return {"n": len(eps), "ok": ok, "first": first, "buckets": b, "commits": c}


def sign_p(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def cmd_arms(args) -> int:
    rows = []
    for arm, seeds in run_dirs(args.root).items():
        if args.arms and arm not in args.arms:
            continue
        pooled: List[Dict[str, Any]] = []
        for seed in sorted(seeds):
            eps = list(episodes(args.root, arm, [seed]).values())
            if not eps:
                continue
            pooled += eps
            rows.append((arm, str(seed), tally(eps), len(list(seeds[seed].glob("*/summary.json")))))
        if len(seeds) > 1 and pooled:
            rows.append((arm, "+".join(str(s) for s in sorted(seeds)), tally(pooled), None))
    head = ["arm", "seed", "episodes", "successes", "sr", "first_attempt", *BUCKETS,
            *(f"commits_{k}" for k in KINDS), "scene_summaries"]
    print(f"{'arm':38s} {'seed':9s} {'n':>4s} {'ok':>4s} {'SR':>5s} {'1st':>4s} "
          + " ".join(f"{k:>6s}" for k in BUCKETS + KINDS))
    table = []
    for arm, seed, t, scenes in rows:
        sr = 100.0 * t["ok"] / t["n"]
        print(f"{arm:38s} {seed:9s} {t['n']:4d} {t['ok']:4d} {sr:5.1f} {t['first']:4d} "
              + " ".join(f"{t['buckets'][k]:6d}" for k in BUCKETS)
              + " " + " ".join(f"{t['commits'][k]:6d}" for k in KINDS))
        table.append([arm, seed, t["n"], t["ok"], round(sr / 100, 4), t["first"],
                      *(t["buckets"][k] for k in BUCKETS), *(t["commits"][k] for k in KINDS),
                      "" if scenes is None else scenes])
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(table)
        print(f"\nwrote {args.csv}")
    return 0


def cmd_ablations(args) -> int:
    base_arm, arms = SETS[args.set]
    base_arm = args.base or base_arm
    base = episodes(args.root, base_arm)
    if not base:
        print(f"no episodes of {base_arm} under {args.root}", file=sys.stderr)
        return 2
    bt = tally(list(base.values()))
    print(f"base {base_arm}: {bt['ok']}/{bt['n']} on the object's storey, seeds {SEEDS}\n")
    print(f"{'removed / replaced':36s} {'arm':32s} {'n':>4s} {'abl':>4s} {'base':>4s} {'diff':>5s} "
          f"{'better':>6s} {'worse':>6s} {'p':>6s}  per seed (abl/base)")
    detail = [("base", base_arm, bt)]
    for label, arm in arms:
        abl = episodes(args.root, arm)
        common = sorted(set(abl) & set(base))
        if not common:
            print(f"{label:36s} {arm:32s}    - (no run under {args.root})")
            continue
        a_ok = {k: scored(abl[k])["success"] for k in common}
        b_ok = {k: scored(base[k])["success"] for k in common}
        better = sum(a_ok[k] and not b_ok[k] for k in common)
        worse = sum(b_ok[k] and not a_ok[k] for k in common)
        per = []
        for seed in SEEDS:
            ks = [k for k in common if k[0] == seed]
            if ks:
                per.append(f"s{seed} {sum(a_ok[k] for k in ks)}/{sum(b_ok[k] for k in ks)}")
        sa, sb = sum(a_ok.values()), sum(b_ok.values())
        print(f"{label:36s} {arm:32s} {len(common):4d} {sa:4d} {sb:4d} {sa - sb:+5d} {better:6d} "
              f"{worse:6d} {sign_p(better, worse):6.3f}  {', '.join(per)}")
        detail.append((label, arm, tally([abl[k] for k in common])))
    print(f"\n{SUBSET_NOTE}" if args.set == "final" else "")
    print(f"\n{'arm':32s} {'n':>4s} {'ok':>4s} " + " ".join(f"{k:>6s}" for k in BUCKETS + KINDS))
    for label, arm, t in detail:
        print(f"{arm:32s} {t['n']:4d} {t['ok']:4d} " + " ".join(f"{t['buckets'][k]:6d}" for k in BUCKETS)
              + " " + " ".join(f"{t['commits'][k]:6d}" for k in KINDS))
    print("\nbuckets and commits on the paired episodes; TRUE/STALE/FP are the goal commits of the C episodes")
    return 0


def cmd_wall(args) -> int:
    ref_arm, others = args.ref[0], args.ref[1:]
    ref = episodes(args.root, ref_arm)
    for arm in [ref_arm, *others]:
        eps = episodes(args.root, arm)
        if not eps:
            print(f"{arm:38s} -")
            continue
        wall = sum(float(e.get("wall_time_s") or 0) for e in eps.values())
        steps = sum(int(e.get("steps") or 0) for e in eps.values())
        print(f"{arm:38s} {len(eps):4d} eps  {wall / 3600:6.1f} h  {wall / len(eps) / 60:5.1f} min/ep  "
              f"{wall / max(steps, 1):5.2f} s/step  {steps / len(eps):5.0f} steps/ep")
        if arm == ref_arm:
            continue
        same = [k for k in eps if k in ref and int(eps[k].get("steps") or 0) == int(ref[k].get("steps") or 0)]
        if not same:
            continue
        a = sum(float(ref[k].get("wall_time_s") or 0) for k in same)
        b = sum(float(eps[k].get("wall_time_s") or 0) for k in same)
        s = sum(int(ref[k].get("steps") or 0) for k in same)
        print(f"   paired with {ref_arm} on {len(same)} identical-step episodes: "
              f"{a / len(same) / 60:5.1f} -> {b / len(same) / 60:5.1f} min/ep ({100 * (b / a - 1):+.0f}%), "
              f"{a / s:4.2f} -> {b / s:4.2f} s/step")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("arms", help="every arm x seed")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--arms", nargs="*", help="only these arms (default: every run below root)")
    p.add_argument("--csv", type=Path)
    p.set_defaults(fn=cmd_arms)
    p = sub.add_parser("ablations", help="each ablation against its base arm")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--set", choices=sorted(SETS), default="final")
    p.add_argument("--base", help="pair against this arm instead of the set's")
    p.set_defaults(fn=cmd_ablations)
    p = sub.add_parser("wall", help="wall time, paired with a reference arm")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--ref", nargs="+", required=True, help="reference arm, then the arms to compare")
    p.set_defaults(fn=cmd_wall)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
