#!/usr/bin/env python3
"""The hard subset: the trials an experiment can actually move, and nothing else.

A full arm on the released benchmark is 107 dynamic trials and about five hours
of process time, and 68% of that is the failures that run to the 500-step
budget. Those failures are also the only trials a search-order or approach
change is aimed at; the successes and the trials no mechanism touches are
paid for every run and never change the answer.

This writes a trial list picked by rule from a reference run, so an arm can be
run on the subset and still be paired, trial by trial, against any full run:

  hard_cross     cross-anchor trials where the target was never named in the
                 reference run -- the search never brought it into a usable
                 view (the population of every search-order arm)
  cross_named    cross-anchor trials named and still failed
  inanchor_leave in-anchor trials whose first commit was already within 1 m
                 of the object and which still failed (the arrive-and-leave
                 population of the look before absence)
  inanchor_other the remaining in-anchor failures

`--rules` picks which of those go in. The output records the rule, the
reference run and each trial's reference outcome, so a later reader knows what
the subset was selected on. Selection on a reference run's outcome is a
selection on noise as well as on difficulty -- a trial that failed once by
chance is in, one that scored once by chance is out -- so subset results are
for iterating on a mechanism, and the full 107 remain the number reported.

    python scripts/make_hard_subset.py --reference outputs/osg_closelook/inanchor
    TRIAL_SET=data/splits/dualmap_hard.json ARMS=flat_anchor_v2_island_close scripts/run_close_look_ab.sh
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

CONDITIONS = ("in_anchor", "cross_anchor")
RULES = ("hard_cross", "cross_named", "inanchor_leave", "inanchor_other")


def horizontal(a, b) -> float:
    ax, az = (a[0], a[2]) if len(a) == 3 else (a[0], a[1])
    bx, bz = (b[0], b[2]) if len(b) == 3 else (b[0], b[1])
    return float(np.hypot(ax - bx, az - bz))


def load(root: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for path in sorted(root.glob("**/episodes.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            block = (r.get("authored_layout") or {}).get("dualmap") or {}
            if block.get("condition") in CONDITIONS:
                out[block["trial_id"]] = r
    return out


def classify(r: Dict[str, Any]) -> str:
    block = r["authored_layout"]["dualmap"]
    if block["success"]:
        return "success"
    named = int(r.get("gt_kf_detected") or 0) > 0
    if block["condition"] == "cross_anchor":
        return "cross_named" if named else "hard_cross"
    log = r.get("goal_commit_log") or []
    if log and horizontal(log[0]["center"], r["authored_layout"]["target_position"]) <= 1.0:
        return "inanchor_leave"
    return "inanchor_other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/osg_closelook/inanchor"))
    parser.add_argument("--rules", nargs="+", default=["hard_cross", "cross_named", "inanchor_leave"],
                        choices=RULES)
    parser.add_argument("--drop-queries", nargs="*", default=["scissors", "mug"],
                        help="queries the detector cannot name in situ (recall 0.02 / 0.00); "
                             "no search or approach change can convert them, so they are "
                             "paid for and never move. Pass an empty list to keep them.")
    parser.add_argument("--out", type=Path, default=Path("data/splits/dualmap_hard.json"))
    args = parser.parse_args()

    recs = load(args.reference)
    if not recs:
        raise SystemExit(f"no dynamic trials under {args.reference}")
    chosen = []
    by_rule = collections.Counter()
    hours = 0.0
    for trial_id in sorted(recs):
        r = recs[trial_id]
        rule = classify(r)
        if rule not in args.rules:
            continue
        if r["authored_layout"]["dualmap"]["query"] in set(args.drop_queries or []):
            by_rule["dropped_query"] += 1
            continue
        by_rule[rule] += 1
        hours += float(r.get("wall_time_s") or 0.0) / 3600.0
        chosen.append({
            "trial_id": trial_id,
            "scene": r["scene"],
            "condition": r["authored_layout"]["dualmap"]["condition"],
            "query": r["authored_layout"]["dualmap"]["query"],
            "rule": rule,
            "reference_success": int(r["authored_layout"]["dualmap"]["success"]),
            "reference_steps": int(r.get("steps") or 0),
        })
    per_scene = collections.Counter(c["scene"] for c in chosen)
    payload = {
        "reference_run": str(args.reference),
        "rules": list(args.rules),
        "dropped_queries": list(args.drop_queries or []),
        "selected_on": "the reference run's outcome; a selection on noise as well as difficulty",
        "counts": {"total": len(chosen), "by_rule": dict(by_rule), "by_scene": dict(per_scene)},
        "reference_wall_hours": round(hours, 2),
        "trials": chosen,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(chosen)} of {len(recs)} trials, "
          f"{dict(by_rule)}, per scene {dict(per_scene)}, "
          f"reference wall {hours:.2f} h of "
          f"{sum(float(r.get('wall_time_s') or 0) for r in recs.values()) / 3600:.2f} h")


if __name__ == "__main__":
    main()
