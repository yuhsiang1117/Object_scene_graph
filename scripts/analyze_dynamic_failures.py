#!/usr/bin/env python3
"""Where our dynamic episodes fail, one bucket per trial.

The comparison says we lose; the gap analysis says roughly where. This says which
single stage each of the 107 dynamic trials died at, so a fix can be aimed.

The stages are ordered and a trial lands in the first one it fails, because the
later evidence is meaningless once an earlier stage has failed -- an episode that
never put the object in frame has nothing to say about the approach controller.

  1 never in view          the search never pointed the camera at it
  2 seen, never named      detector recall
  3 named, never admitted  the admission gate
  4 admitted, no commit    the candidate gate
  5 committed elsewhere    we aimed at something that is not the object
  6 committed, no arrival  we aimed right and did not get there

Stages 1-4 read the ground-truth instrument (`gt_*`, observed beside the run and
never given to the agent). Stage 5 is the mirror of the flaw we charge DualMap
with: arriving at a confidently-named wrong place. `goal_commit_log` shows how
many such places one episode visits.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osg.eval import dualmap_release as release

STAGES = [
    ("success", "succeeded"),
    ("never_in_view", "never in view"),
    ("seen_not_named", "seen, never named"),
    ("named_not_admitted", "named, never admitted"),
    ("admitted_not_committed", "admitted, never committed"),
    ("committed_elsewhere", "committed elsewhere"),
    ("committed_no_arrival", "committed, never arrived"),
]


def load(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("**/episodes.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            block = (record.get("authored_layout") or {}).get("dualmap") or {}
            if block.get("condition") in ("in_anchor", "cross_anchor"):
                rows.append(record)
    return rows


def classify(record: Dict[str, Any], targets) -> str:
    block = record["authored_layout"]["dualmap"]
    if block["success"]:
        return "success"
    if not (record.get("gt_kf_in_view") or 0):
        return "never_in_view"
    if not (record.get("gt_kf_detected") or 0):
        return "seen_not_named"
    if not (record.get("gt_kf_admitted") or 0):
        return "named_not_admitted"
    if record.get("target_obj_xy") is None:
        return "admitted_not_committed"
    if not release.position_correct(record["target_obj_xy"], targets):
        return "committed_elsewhere"
    return "committed_no_arrival"


def commit_errors(record: Dict[str, Any], targets) -> List[float]:
    """How far each goal this episode committed to was from the real object."""
    out = []
    for entry in record.get("goal_commit_log") or []:
        centre = entry.get("center")
        if centre and len(centre) >= 3:
            out.append(release.horizontal_distance_xz((centre[0], centre[2]), targets))
    return out


def table(title: str, header: List[str], rows: List[List[str]]) -> List[str]:
    out = ["", f"## {title}", "", "| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osg", type=Path, default=Path("outputs/osg_dualmap_protocol"))
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/osg_dualmap_protocol/DYNAMIC_FAILURES.md")
    )
    args = parser.parse_args()

    protocol = {t["trial_id"]: t for t in release.protocol()}
    records = load(args.osg)
    rows = []
    for record in records:
        block = record["authored_layout"]["dualmap"]
        trial = protocol[block["trial_id"]]
        targets = release.trial_targets(trial)
        errors = commit_errors(record, targets)
        rows.append(
            {
                "trial_id": block["trial_id"],
                "condition": block["condition"],
                "query": block["query"],
                "scene": record["scene"],
                "stage": classify(record, targets),
                "steps": int(record.get("steps") or 0),
                "at_budget": int(record.get("steps") or 0) >= 500,
                "travelled_m": float(block.get("travelled_m") or 0.0),
                "stop_reason": record.get("approach_stop_reason"),
                "commits": len(errors),
                "commits_wrong": sum(1 for e in errors if e > release.SUCCESS_DISTANCE_M),
                "commit_errors_m": errors,
                "final_distance_m": float(block["final_distance_horizontal_m"]),
                "gt_kf_in_view": int(record.get("gt_kf_in_view") or 0),
                "gt_kf_detected": int(record.get("gt_kf_detected") or 0),
                "gt_best_det_score": float(record.get("gt_best_det_score") or 0.0),
            }
        )

    conditions = ("in_anchor", "cross_anchor")
    out = [
        "# Where our dynamic episodes fail",
        "",
        f"{len(rows)} dynamic trials from {args.osg}. Each trial is assigned to the first "
        "stage it failed; the buckets are exhaustive and mutually exclusive.",
    ]

    body = []
    for key, label in STAGES:
        cells = [label]
        for condition in conditions:
            subset = [r for r in rows if r["condition"] == condition]
            n = sum(1 for r in subset if r["stage"] == key)
            cells.append(f"{n} ({100 * n / len(subset):.0f}%)")
        cells.append(str(sum(1 for r in rows if r["stage"] == key)))
        body.append(cells)
    out += table("The funnel", ["Stage", "in-anchor (54)", "cross-anchor (53)", "All"], body)

    # Per query, because the detector floor is not uniform across the eight objects.
    body = []
    for query in sorted({r["query"] for r in rows}):
        subset = [r for r in rows if r["query"] == query]
        ok = sum(1 for r in subset if r["stage"] == "success")
        counts = collections.Counter(r["stage"] for r in subset)
        worst = max(
            (k for k, _ in STAGES if k != "success"),
            key=lambda k: counts.get(k, 0),
        )
        body.append([
            query, str(len(subset)), f"{ok}/{len(subset)}",
            f"{dict(STAGES)[worst]} ({counts.get(worst, 0)})",
            f"{np.median([r['gt_kf_detected'] / max(r['gt_kf_in_view'], 1) for r in subset]):.2f}",
        ])
    out += table(
        "By query",
        ["Query", "Trials", "Success", "Dominant failure", "Median in-situ recall"],
        body,
    )

    # Stage 5 is the one we charge DualMap with, so it is measured on ourselves too.
    out += [
        "",
        "## Chasing the wrong object",
        "",
        "`goal_commit_log` records every goal the candidate gate committed to. A commit "
        "more than 1 m from the real object is an episode spending its budget walking to "
        "something that is not the target -- the same failure we charge DualMap's anchor "
        "successes with, except ours does not get scored for it.",
        "",
        "| Condition | Trials | Episodes with >=1 commit | Median commits | Commits that were wrong | Episodes whose every commit was wrong |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        subset = [r for r in rows if r["condition"] == condition]
        committed = [r for r in subset if r["commits"]]
        total_commits = sum(r["commits"] for r in subset)
        wrong = sum(r["commits_wrong"] for r in subset)
        all_wrong = sum(1 for r in committed if r["commits_wrong"] == r["commits"])
        out.append(
            f"| {condition} | {len(subset)} | {len(committed)} | "
            f"{np.median([r['commits'] for r in committed]) if committed else 0:.0f} | "
            f"{wrong}/{total_commits} = {100 * wrong / max(total_commits, 1):.0f}% | "
            f"{all_wrong}/{len(committed)} |"
        )

    # Budget, which is where the difference in travelled distance comes from.
    out += [
        "",
        "## Budget",
        "",
        "| Condition | At the 500-step budget | Median steps | Median travelled |",
        "|---|---:|---:|---:|",
    ]
    for condition in conditions:
        subset = [r for r in rows if r["condition"] == condition]
        out.append(
            f"| {condition} | {sum(1 for r in subset if r['at_budget'])}/{len(subset)} | "
            f"{np.median([r['steps'] for r in subset]):.0f} | "
            f"{np.median([r['travelled_m'] for r in subset]):.1f} m |"
        )

    # Why the terminal approach ended, for the episodes that got that far.
    body = []
    reasons = collections.Counter(
        (r["condition"], r["stop_reason"] or "none")
        for r in rows if r["stage"] in ("committed_elsewhere", "committed_no_arrival")
    )
    for (condition, reason), n in sorted(reasons.items()):
        body.append([condition, reason, str(n)])
    out += table("How the terminal approach ended, for committed failures",
                 ["Condition", "Stop reason", "Trials"], body)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
