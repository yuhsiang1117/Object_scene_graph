#!/usr/bin/env python3
"""Did the tighter, extent-aware viewpoint ring convert the near misses?

The expected effect is about nine trials out of 107, which is inside the noise
floor two arms of this size can resolve (docs/history/AB_RESULTS.md). An SR delta alone
therefore cannot answer the question, and two of five arms in an earlier campaign
returned confident nulls while testing nothing. So this leads with the MECHANISM
counter -- the distance from the placed goal to the object, which the change acts
on directly -- and reports SR second, on the paired trials only.

Read it in this order:

  1. Did the knob move? `goal_to_obj_m` must fall from 0.80. If it did not, the
     ring never engaged and nothing below means anything.
  2. Did it move the right trials? The near-miss band -- committed to the right
     object, stopped between 1.0 and 1.6 m -- is the population the change targets.
  3. Did it cost anything? A ring that is too tight lands in occupied cells,
     `approach_viewpoint` returns None, and the run falls back to a goal that
     historically scored 0.100 against 0.516. `approach_viewpoint_none` is that
     alarm.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osg.eval import dualmap_release as release


def load(root: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.glob("**/episodes.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            block = (record.get("authored_layout") or {}).get("dualmap") or {}
            if not block.get("trial_id"):
                continue
            diag = record.get("approach_diag") or {}
            stats = record.get("agent_stats") or {}
            out[block["trial_id"]] = {
                "condition": block["condition"],
                "query": block["query"],
                "success": int(block["success"]),
                "final_m": float(block["final_distance_horizontal_m"]),
                "goal_to_obj_m": diag.get("goal_to_obj_m"),
                "shortfall_m": diag.get("min_dist_to_goal_m"),
                "committed": record.get("target_obj_xy") is not None,
                "viewpoint": int(stats.get("approach_viewpoint", 0) or 0),
                "viewpoint_none": int(stats.get("approach_viewpoint_none", 0) or 0),
                "steps": int(record.get("steps") or 0),
            }
    return out


def med(values: List[float]) -> str:
    clean = [float(v) for v in values if v is not None]
    return f"{np.median(clean):.2f}" if clean else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("outputs/osg_dualmap_protocol"))
    parser.add_argument("--arm", type=Path, default=Path("outputs/osg_dualmap_tightring"))
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/osg_dualmap_tightring/TIGHTRING_AB.md")
    )
    args = parser.parse_args()

    base, arm = load(args.baseline), load(args.arm)
    paired = sorted(set(base) & set(arm))
    lines = [
        "# Tighter, extent-aware viewpoint rings: A/B",
        "",
        f"Baseline `{args.baseline}` ({len(base)} trials), arm `{args.arm}` "
        f"({len(arm)} trials), paired on trial id: **{len(paired)}**.",
        "",
        "## 1. Did the knob move?",
        "",
        "`goal_to_obj_m` is the distance from the goal the approach actually placed "
        "to the committed object. The change acts on exactly this number; if it has "
        "not fallen from 0.80, nothing below is evidence of anything.",
        "",
        "| Condition | Approaches (base / arm) | Median goal_to_obj (base) | (arm) |",
        "|---|---:|---:|---:|",
    ]
    for condition in ("in_anchor", "cross_anchor", "static"):
        b = [base[t] for t in paired if base[t]["condition"] == condition]
        a = [arm[t] for t in paired if arm[t]["condition"] == condition]
        bg = [r["goal_to_obj_m"] for r in b if r["goal_to_obj_m"] is not None]
        ag = [r["goal_to_obj_m"] for r in a if r["goal_to_obj_m"] is not None]
        if not b:
            continue
        lines.append(
            f"| {condition} | {len(bg)} / {len(ag)} | {med(bg)} m | **{med(ag)} m** |"
        )

    lines += [
        "",
        "## 2. Did it convert the population it targets?",
        "",
        "The near-miss band is the trials that committed to the right object and "
        "stopped between 1.0 and 1.6 m -- close enough that the ring change can "
        "reach them, and failing only because of it.",
        "",
        "| Condition | Trials | SR base | SR arm | Delta | Near-miss band (base -> arm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in ("static", "in_anchor", "cross_anchor"):
        keep = [t for t in paired if base[t]["condition"] == condition]
        if not keep:
            continue
        sb = sum(base[t]["success"] for t in keep)
        sa = sum(arm[t]["success"] for t in keep)
        nb = sum(1 for t in keep if not base[t]["success"] and 1.0 <= base[t]["final_m"] <= 1.6)
        na = sum(1 for t in keep if not arm[t]["success"] and 1.0 <= arm[t]["final_m"] <= 1.6)
        lines.append(
            f"| {condition} | {len(keep)} | {sb}/{len(keep)} = {100 * sb / len(keep):.1f}% | "
            f"{sa}/{len(keep)} = {100 * sa / len(keep):.1f}% | "
            f"{100 * (sa - sb) / len(keep):+.1f} pts | {nb} -> {na} |"
        )

    flips = collections.Counter()
    for t in paired:
        flips[(base[t]["success"], arm[t]["success"])] += 1
    lines += [
        "",
        f"Per-trial: **{flips[(0, 1)]} gained**, **{flips[(1, 0)]} lost**, "
        f"{flips[(1, 1)]} both, {flips[(0, 0)]} neither. A change that only converts "
        "near misses should show gains with very few losses; losses mean the tighter "
        "ring cost a viewpoint somewhere.",
        "",
        "## 3. What did it cost?",
        "",
        "| Quantity | Baseline | Arm |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("Approaches that found no viewpoint (`approach_viewpoint_none`)", "viewpoint_none"),
        ("Median final distance, all trials", "final_m"),
        ("Median steps", "steps"),
    ):
        b = [base[t][key] for t in paired]
        a = [arm[t][key] for t in paired]
        if key == "viewpoint_none":
            lines.append(f"| {label} | {sum(b)} | {sum(a)} |")
        else:
            lines.append(f"| {label} | {med(b)} | {med(a)} |")

    missing = sorted(set(base) - set(arm))
    if missing:
        lines += ["", f"**{len(missing)} baseline trials are missing from the arm** -- "
                      "the comparison above is on the paired subset only, so re-read it "
                      "once the run completes.",
                  "", *[f"- `{m}`" for m in missing[:20]]]
        if len(missing) > 20:
            lines.append(f"- ... and {len(missing) - 20} more")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
