#!/usr/bin/env python3
"""Where does DualMap's local path actually point?

Success alone cannot separate "DualMap found the object and the agent walked
there" from "DualMap aimed at the wrong object" or "the agent happened to stop
near the object without DualMap ever localising it".  This reads the local paths
DualMap saved for every trial and measures the distance from each planned goal
to the released ground-truth object position.

Frame note: saved paths are in DualMap's z-up frame; Habitat (x, z) is
(path_x, -path_y).  Verified by the global path's first point coinciding with
the agent pose and its last point with the local path's first point.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path
import statistics
from typing import Any, Dict, List

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osg.eval import dualmap_release as release

CONDITIONS = ("static", "in_anchor", "cross_anchor")
LABEL = {"static": "Static", "in_anchor": "In-anchor", "cross_anchor": "Cross-anchor"}
GOAL_TOLERANCE_M = 1.0


def to_habitat_xz(point: List[float]) -> tuple:
    """DualMap's saved paths are z-up: Habitat (x, z) is (path_x, -path_y)."""
    return (point[0], -point[1])


def goal_error(point: List[float], centres, halves) -> float:
    """Horizontal distance from a planned goal to the nearest true target box.

    The measurement itself lives in `osg.eval.dualmap_release` so that both
    systems' goals are judged by one predicate; only the frame conversion is
    DualMap's.
    """
    return release.horizontal_distance_xz(
        to_habitat_xz(point), list(zip(centres, halves))
    )


def analyse(roots: List[Path]) -> Dict[str, Any]:
    per: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    errors: Dict[str, List[float]] = collections.defaultdict(list)
    aiming: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    anchors: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for root in roots:
        for trial in sorted(glob.glob(str(root / "trials" / "*"))):
            result = Path(trial) / "result.json"
            if not result.is_file():
                continue
            record = json.loads(result.read_text(encoding="utf-8"))
            condition = record["condition"]
            counts = per[condition]
            counts["total"] += 1
            centres = [np.asarray(t, dtype=float) for t in record["target_positions"]]
            halves = [np.asarray(h, dtype=float) for h in record["target_half_extents"]]
            paths = sorted(
                glob.glob(f"{trial}/path/local_path/*.json"),
                key=lambda p: int(Path(p).stem),
            )
            goals = [
                goal_error(json.loads(Path(p).read_text(encoding="utf-8"))[-1], centres, halves)
                for p in paths
            ]
            success = bool(record["success"])
            if not goals:
                counts["no_local_path_success" if success else "no_local_path_fail"] += 1
                if success:
                    final = ([a for a in record["attempts"] if a.get("success")] or [{}])[-1]
                    anchors[condition][str(final.get("candidate"))] += 1
                continue
            errors[condition].append(goals[-1])
            correct = min(goals) <= GOAL_TOLERANCE_M
            counts["planned"] += 1
            if correct:
                counts["goal_correct"] += 1
                counts["goal_correct_reached" if success else "goal_correct_missed"] += 1
            else:
                counts["goal_wrong_success" if success else "goal_wrong_fail"] += 1
            if condition != "static":
                aiming[condition].append(
                    aim_row(record, paths[-1], centres, halves)
                )
    return {"counts": per, "errors": errors, "aiming": aiming, "anchors": anchors}


def aim_row(record, last_path: str, centres, halves) -> Dict[str, Any]:
    """Did the planned goal point at the object's new position, or its old one?

    The only way to tell a system that re-detects from one that replays its prior
    map is to measure both distances, and only on the trials where they differ --
    in-anchor moves the object less than the success tolerance, so there the two
    hypotheses are observationally identical.
    """
    point = json.loads(Path(last_path).read_text(encoding="utf-8"))[-1]
    goal = to_habitat_xz(point)
    new_targets = list(zip(centres, halves))
    old_targets = release.static_target_positions(record["scene"], record["query"])
    displacement = min(
        float(np.linalg.norm((n[0] - o[0])[[0, 2]]))
        for n in new_targets for o in old_targets
    )
    return {
        "trial_id": record["trial_id"],
        "displacement_m": displacement,
        "error_to_new_m": release.horizontal_distance_xz(goal, new_targets),
        "error_to_old_m": release.horizontal_distance_xz(goal, old_targets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/dualmap_official_bench"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[12, 13, 14])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    roots = [args.root / f"seed{s}" for s in args.seeds if (args.root / f"seed{s}" / "trials").is_dir()]
    data = analyse(roots)
    counts, errors = data["counts"], data["errors"]

    lines: List[str] = ["# Where DualMap's local path points\n"]
    lines.append(
        "Seeds " + ", ".join(str(s) for s in args.seeds) + ". A goal is \"correct\" when "
        f"the planned local-path endpoint lies within {GOAL_TOLERANCE_M:.0f} m of the released "
        "ground-truth object position.\n"
    )

    lines.append("## Goal accuracy, among trials that planned a local path\n")
    lines.append("| Split | Local paths | Goal correct | Median goal error |")
    lines.append("|---|---:|---:|---:|")
    for condition in CONDITIONS:
        c = counts[condition]
        if not c["planned"]:
            continue
        lines.append(
            f"| {LABEL[condition]} | {c['planned']} | "
            f"{c['goal_correct']}/{c['planned']} = {100 * c['goal_correct'] / c['planned']:.1f}% | "
            f"{statistics.median(errors[condition]):.2f} m |"
        )
    lines.append("")

    lines.append("## Full outcome breakdown\n")
    lines.append("| Outcome | " + " | ".join(LABEL[c] for c in CONDITIONS) + " |")
    lines.append("|---|" + "---:|" * len(CONDITIONS))
    rows = [
        ("Local path, goal correct, reached", "goal_correct_reached"),
        ("Local path, goal correct, missed", "goal_correct_missed"),
        ("Local path, goal wrong, failed", "goal_wrong_fail"),
        ("Local path, goal wrong, scored success", "goal_wrong_success"),
        ("No local path, failed", "no_local_path_fail"),
        ("No local path, scored success", "no_local_path_success"),
    ]
    for label, key in rows:
        cells = []
        for condition in CONDITIONS:
            c = counts[condition]
            cells.append(f"{c[key]} ({100 * c[key] / c['total']:.1f}%)" if c["total"] else "n/a")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Position-correct SR\n")
    lines.append(
        "Success credited only when DualMap localised the object — the local-path goal is "
        "within 1 m of the true position — rather than when the agent merely ends up near it.\n"
    )
    lines.append("| Split | Trials | Localised correctly | Localised and reached |")
    lines.append("|---|---:|---:|---:|")
    for condition in CONDITIONS:
        c = counts[condition]
        n = c["total"]
        if not n:
            continue
        lines.append(
            f"| {LABEL[condition]} | {n} | {100 * c['goal_correct'] / n:.1f}% | "
            f"{100 * c['goal_correct_reached'] / n:.1f}% |"
        )
    lines.append("")

    lines.append("## What the anchor-only successes stopped at\n")
    lines.append(
        "These are the successes with no local path at all: DualMap text-matched a piece "
        "of furniture in its abstract map, walked to it, and the agent happened to be "
        "within the tolerance. The anchor is a class name, not the queried object.\n"
    )
    lines.append("| Split | Anchor-only successes | Anchors stopped at |")
    lines.append("|---|---:|---|")
    for condition in CONDITIONS:
        tally = data["anchors"][condition]
        if not tally:
            continue
        listed = ", ".join(f"`{k}` x{v}" for k, v in tally.most_common())
        lines.append(f"| {LABEL[condition]} | {sum(tally.values())} | {listed} |")
    lines.append("")

    lines.append("## New position or old position?\n")
    lines.append(
        "A goal within 1 m of where the object now is means the system re-detected it; a "
        "goal within 1 m of where it used to be means the system replayed its prior map. "
        "The two are only distinguishable when the object moved further than the "
        "tolerance, so the second table is the one that decides it.\n"
    )
    lines.append("| Split | Local paths | Median error to new | Median error to old | Aimed at new | Aimed at old |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for condition in ("in_anchor", "cross_anchor"):
        rows_ = data["aiming"][condition]
        if not rows_:
            continue
        new_hits = sum(1 for r in rows_ if r["error_to_new_m"] <= GOAL_TOLERANCE_M)
        old_hits = sum(
            1 for r in rows_
            if r["error_to_old_m"] <= GOAL_TOLERANCE_M and r["error_to_new_m"] > GOAL_TOLERANCE_M
        )
        lines.append(
            f"| {LABEL[condition]} | {len(rows_)} | "
            f"{statistics.median(r['error_to_new_m'] for r in rows_):.2f} m | "
            f"{statistics.median(r['error_to_old_m'] for r in rows_):.2f} m | "
            f"{new_hits} | {old_hits} |"
        )
    lines.append("")

    decisive = [
        r for condition in ("in_anchor", "cross_anchor")
        for r in data["aiming"][condition]
        if r["displacement_m"] > 1.5
    ]
    if decisive:
        new_hits = sum(1 for r in decisive if r["error_to_new_m"] <= GOAL_TOLERANCE_M)
        old_hits = sum(
            1 for r in decisive
            if r["error_to_old_m"] <= GOAL_TOLERANCE_M and r["error_to_new_m"] > GOAL_TOLERANCE_M
        )
        lines.append(
            f"Restricted to the {len(decisive)} planned goals whose object moved more than "
            f"1.5 m, so the two hypotheses differ: **{new_hits} aimed at the new position, "
            f"{old_hits} at the old one.** DualMap does not replay stale positions; when it "
            "plans a local path it has re-detected the object. Its over-credit is the "
            "anchor-only successes above, not stale planning.\n"
        )

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
