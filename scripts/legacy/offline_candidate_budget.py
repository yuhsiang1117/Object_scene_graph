#!/usr/bin/env python3
"""Replay distinct failed-candidate budgets over frozen episode telemetry.

The observable replay uses failed attempts, which the multi-attempt evaluator
feeds back to the agent.  The separate oracle audit labels goal commits by their
distance to the authored target and is diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from osg.eval import dualmap_release as release


def load_records(root: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(root.glob("*/episodes.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            block = (record.get("authored_layout") or {}).get("dualmap") or {}
            if block.get("condition") in {"in_anchor", "cross_anchor"}:
                records.append(record)
    return records


def assign_spatial_ids(commits: Sequence[Dict[str, Any]], radius_m: float) -> List[int]:
    representatives: List[np.ndarray] = []
    ids: List[int] = []
    for commit in commits:
        center = np.asarray(commit.get("center") or [np.nan, np.nan, np.nan], dtype=float)
        point = center[[0, 2]]
        match = None
        for index, representative in enumerate(representatives):
            if float(np.linalg.norm(point - representative)) <= radius_m:
                match = index
                break
        if match is None:
            match = len(representatives)
            representatives.append(point)
        ids.append(match)
    return ids


def targets_for(record: Dict[str, Any], protocol: Dict[str, Dict[str, Any]]):
    trial_id = record["authored_layout"]["dualmap"]["trial_id"]
    return release.trial_targets(protocol[trial_id])


def active_commit_index(commits: Sequence[Dict[str, Any]], step: int) -> Optional[int]:
    candidates = [i for i, commit in enumerate(commits) if int(commit.get("step") or 0) <= step]
    return candidates[-1] if candidates else None


def nth_distinct(events: Sequence[Tuple[int, int]], budget: int) -> Optional[int]:
    seen = set()
    for step, candidate_id in sorted(events):
        seen.add(candidate_id)
        if len(seen) >= budget:
            return int(step)
    return None


def median_or_none(values: Sequence[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def summarize_budget(rows: Sequence[Dict[str, Any]], mode: str, budget: int) -> Dict[str, Any]:
    triggered = [r for r in rows if r[mode][str(budget)]["trigger_step"] is not None]
    successes = [r for r in rows if r["success"]]
    success_triggered = [r for r in successes if r[mode][str(budget)]["trigger_step"] is not None]
    at_budget_failures = [r for r in rows if r["at_budget"] and not r["success"]]
    at_budget_triggered = [
        r for r in at_budget_failures if r[mode][str(budget)]["trigger_step"] is not None
    ]
    return {
        "budget": budget,
        "episodes_triggered": len(triggered),
        "episode_rate": len(triggered) / len(rows) if rows else 0.0,
        "successful_episodes_triggered": len(success_triggered),
        "successful_episode_count": len(successes),
        "budget_exhausted_failures_triggered": len(at_budget_triggered),
        "budget_exhausted_failure_count": len(at_budget_failures),
        "median_trigger_step": median_or_none([
            r[mode][str(budget)]["trigger_step"] for r in triggered
        ]),
        "median_remaining_steps": median_or_none([
            r[mode][str(budget)]["remaining_steps"] for r in triggered
        ]),
        "commits_after_trigger": int(sum(
            r[mode][str(budget)]["commits_after_trigger"] for r in triggered
        )),
        "wrong_commits_after_trigger": int(sum(
            r[mode][str(budget)]["wrong_commits_after_trigger"] for r in triggered
        )),
    }


def analyze_record(
    record: Dict[str, Any],
    protocol: Dict[str, Dict[str, Any]],
    budgets: Sequence[int],
    cluster_radius_m: float,
) -> Dict[str, Any]:
    block = record["authored_layout"]["dualmap"]
    commits = sorted(record.get("goal_commit_log") or [], key=lambda x: int(x.get("step") or 0))
    spatial_ids = assign_spatial_ids(commits, cluster_radius_m)
    targets = targets_for(record, protocol)
    errors = []
    for commit in commits:
        center = commit.get("center")
        error = (
            release.horizontal_distance_xz((center[0], center[2]), targets)
            if center and len(center) >= 3 else float("inf")
        )
        errors.append(float(error))

    # A failed benchmark attempt is observable in this runner: rearm_after_failed_attempt
    # applies negative presence and identity evidence to the active candidate.
    observable_events: List[Tuple[int, int]] = []
    linked_attempts = []
    for attempt in record.get("attempt_log") or []:
        if bool(attempt.get("success")):
            continue
        step = int(attempt.get("step") or 0)
        index = active_commit_index(commits, step)
        if index is None:
            continue
        observable_events.append((step, spatial_ids[index]))
        linked_attempts.append({
            "step": step,
            "track_id": commits[index].get("track_id"),
            "spatial_id": spatial_ids[index],
            "commit_error_m_oracle": errors[index],
        })

    oracle_events = [
        (int(commit.get("step") or 0), spatial_ids[index])
        for index, commit in enumerate(commits)
        if errors[index] > release.SUCCESS_DISTANCE_M
    ]

    result = {
        "trial_id": block["trial_id"],
        "condition": block["condition"],
        "query": block["query"],
        "success": bool(block.get("success")),
        "steps": int(record.get("steps") or 0),
        "at_budget": int(record.get("steps") or 0) >= 500,
        "commit_count": len(commits),
        "distinct_commit_locations": len(set(spatial_ids)),
        "commit_errors_m_oracle": errors,
        "linked_failed_attempts": linked_attempts,
        "excluded_candidate_rejections": list(record.get("candidate_reject_log") or []),
        "observable": {},
        "oracle": {},
    }
    for mode, events in (("observable", observable_events), ("oracle", oracle_events)):
        for budget in budgets:
            trigger = nth_distinct(events, budget)
            after = [i for i, commit in enumerate(commits) if trigger is not None and int(commit.get("step") or 0) > trigger]
            result[mode][str(budget)] = {
                "trigger_step": trigger,
                "remaining_steps": max(0, result["steps"] - trigger) if trigger is not None else None,
                "commits_after_trigger": len(after),
                "wrong_commits_after_trigger": sum(
                    errors[i] > release.SUCCESS_DISTANCE_M for i in after
                ),
            }
    return result


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.1f}%)" if d else "0/0"


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.0f}"


def table(lines: List[str], title: str, data: Sequence[Dict[str, Any]]) -> None:
    lines += [
        "",
        f"## {title}",
        "",
        "| Distinct budget | Episodes triggered | Budget-exhausted failures covered | Successful episodes triggered | Median trigger step | Median steps left | Later commits (wrong) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data:
        lines.append(
            f"| {row['budget']} | {pct(row['episodes_triggered'], row['episode_count'])} | "
            f"{pct(row['budget_exhausted_failures_triggered'], row['budget_exhausted_failure_count'])} | "
            f"{pct(row['successful_episodes_triggered'], row['successful_episode_count'])} | "
            f"{fmt(row['median_trigger_step'])} | {fmt(row['median_remaining_steps'])} | "
            f"{row['commits_after_trigger']} ({row['wrong_commits_after_trigger']}) |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/osg_dualmap_tightring"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--cluster-radius-m", type=float, default=0.75)
    args = parser.parse_args()
    out = args.out or args.run / "offline_improvements" / "CANDIDATE_BUDGET.md"

    records = load_records(args.run)
    if not records:
        raise SystemExit(f"no dynamic records under {args.run}")
    protocol = {trial["trial_id"]: trial for trial in release.protocol()}
    rows = [analyze_record(r, protocol, args.budgets, args.cluster_radius_m) for r in records]

    summaries: Dict[str, Any] = {}
    for mode in ("observable", "oracle"):
        summaries[mode] = {}
        for condition in ("all", "in_anchor", "cross_anchor"):
            subset = rows if condition == "all" else [r for r in rows if r["condition"] == condition]
            values = []
            for budget in args.budgets:
                value = summarize_budget(subset, mode, budget)
                value["episode_count"] = len(subset)
                values.append(value)
            summaries[mode][condition] = values

    reasons = Counter(
        event.get("reason", "unknown")
        for row in rows for event in row["excluded_candidate_rejections"]
    )
    result = {
        "input_run": str(args.run),
        "parameters": {"budgets": args.budgets, "cluster_radius_m": args.cluster_radius_m},
        "telemetry": {
            "episodes": len(rows),
            "failed_attempt_events": sum(len(r["linked_failed_attempts"]) for r in rows),
            "episodes_with_failed_attempt": sum(bool(r["linked_failed_attempts"]) for r in rows),
            "candidate_rejection_reasons_excluded": dict(reasons),
            "exclusion_reason": "unreachable is navigation uncertainty, not conclusive identity failure",
        },
        "summaries": summaries,
        "episodes": rows,
    }

    lines = [
        "# Offline distinct-candidate budget replay",
        "",
        f"Frozen input: `{args.run}` ({len(rows)} dynamic trials). Candidate locations "
        f"are deduplicated within {args.cluster_radius_m:.2f} m.",
        "",
        "The deployable replay uses failed attempts linked to the most recent goal commit. "
        "The runner feeds these failures back through `rearm_after_failed_attempt`, so they "
        "are observable. All recorded `candidate_reject_log` events were `unreachable` "
        f"({reasons.get('unreachable', 0)} events) and are intentionally excluded: planner "
        "uncertainty is not identity evidence.",
        "",
        "The oracle audit labels commit locations with authored destination geometry. It "
        "measures avoidable-looking tail pressure but cannot be used as a runtime policy or "
        "converted into hypothetical successes.",
    ]
    table(lines, "Observable replay — all trials", summaries["observable"]["all"])
    table(lines, "Observable replay — in-anchor", summaries["observable"]["in_anchor"])
    table(lines, "Observable replay — cross-anchor", summaries["observable"]["cross_anchor"])
    table(lines, "Oracle waste audit — all trials", summaries["oracle"]["all"])
    lines += [
        "",
        "## Decision rule",
        "",
        "A budget is not selected from the oracle table. The observable table is used to "
        "choose candidates that cover budget-exhausted failures while rarely firing during "
        "successful episodes and while leaving useful search time. If the current logs do "
        "not expose enough conclusive failures, add telemetry before changing policy.",
        "",
        f"Machine-readable details: [{out.with_suffix('.json').name}]({out.with_suffix('.json').name}).",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
