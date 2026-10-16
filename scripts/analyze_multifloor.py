#!/usr/bin/env python3
"""A multi-floor relocation run, scored on the object's storey and split the way
the floor decision has to be judged.

Four questions, each one a table:

  1. **What does the run score when a stop must be on the object's storey?**
     The released rule is horizontal-only. Episodes recorded before
     `ycb.object_success_max_floor_dy_m` existed are re-scored here from their
     own stop poses; the episodes the horizontal rule accepted from another
     storey are listed. SR carries a Wilson 95% interval, because at n = 25 one
     episode is four points.

  2. **Where is it won and lost?** By relocation class, read off the authored
     geometry rather than the layout's folder name (the multi-floor
     `in_anchor_01` layouts move several objects to another anchor and storey):
     `static`, `same_floor_near` (moved under the success radius),
     `same_floor`, `upward`, `downward`; and by whether the episode STARTS on
     the object's storey.

  3. **Did the agent change storey when it should, and only then?** Reached the
     object's storey, left the start storey when the object was on it
     (an unnecessary switch), first switch step, switch attempts.

  4. **What did it commit to?** Every goal commit, classified against the
     object's true pose: TRUE (within the success radius, same storey), STALE
     (at the object's prior pose -- the correct opening move for a moved
     object), FP (anything else), and by source (prior-map track, live
     detection, region proposal).

Reads `episodes.jsonl` only; no simulator.

    python scripts/analyze_multifloor.py outputs/RERUN/mf5
    python scripts/analyze_multifloor.py OSG=outputs/RERUN/mf5 ASCENT=outputs/ascent_crossanchor
    python scripts/analyze_multifloor.py outputs/RERUN/mf5 --flips --json out.json
    python scripts/analyze_multifloor.py OSG=outputs/mfv2/osg_s42 OSG=outputs/mfv2/osg_s43  # pooled
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SUCCESS_M = 1.0          # the released rule's horizontal radius
FLOOR_DY_M = 1.0         # ycb.object_success_max_floor_dy_m
STOREY_M = 0.5           # ycb.relocation_floor_tolerance_m: same storey or not
MAX_ATTEMPTS = 3
CLASSES = ("static", "same_floor_near", "same_floor", "upward", "downward", "unknown")


def load(root: Path) -> List[Dict[str, Any]]:
    paths = [root] if root.is_file() else sorted(root.glob("**/episodes.jsonl"))
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _hyp(a: Iterable[float], b: Iterable[float]) -> float:
    a, b = list(a), list(b)
    return math.hypot(a[0] - b[0], a[2] - b[2])


def target_floor_y(layout: Dict[str, Any]) -> Optional[float]:
    value = layout.get("destination_floor_y")
    if value is None:
        vps = layout.get("viewpoints") or []
        if vps and vps[0].get("position") is not None:
            value = vps[0]["position"][1]
    return None if value is None else float(value)


def relocation_class(layout: Dict[str, Any]) -> str:
    if str(layout.get("layout_type", "")) == "static" or layout.get("prior_position") is None:
        return "static"
    prior_y, dest_y = layout.get("prior_floor_y"), target_floor_y(layout)
    if prior_y is None or dest_y is None:
        return "unknown"
    dy = float(dest_y) - float(prior_y)
    if abs(dy) > STOREY_M:
        return "upward" if dy > 0 else "downward"
    moved = _hyp(layout["prior_position"], layout["target_position"])
    return "same_floor_near" if moved <= SUCCESS_M else "same_floor"


def attempts_of(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    block = (record.get("authored_layout") or {}).get("object_distance") or {}
    return list(block.get("attempts") or [])


def scored(record: Dict[str, Any]) -> Dict[str, Any]:
    """Success under both rules, from the episode's own stop poses."""
    layout = record.get("authored_layout") or {}
    floor_y = target_floor_y(layout)
    horizontal = on_storey = first = False
    for i, attempt in enumerate(attempts_of(record)[:MAX_ATTEMPTS]):
        within = float(attempt["distance_horizontal_m"]) <= SUCCESS_M
        horizontal |= within
        if within and floor_y is not None:
            hit = abs(float(attempt["position"][1]) - floor_y) <= FLOOR_DY_M
        else:
            hit = within            # no floor to check against: horizontal rule
        on_storey |= hit
        first |= hit and i == 0
    if not attempts_of(record):     # a record without the rule's block
        horizontal = on_storey = bool(record.get("success"))
        first = on_storey and int(record.get("attempts_used") or 1) == 1
    # First-attempt success is what an arm allowed ONE attempt scores: the
    # trajectory up to the first STOP does not depend on how many follow, so it
    # compares a single-attempt run with a three-attempt one without a rerun.
    return {"success": int(on_storey), "success_horizontal": int(horizontal),
            "success_first": int(first)}


def commits(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(kind, source) per goal commit: TRUE / STALE / FP, prior / live / proposal."""
    layout = record.get("authored_layout") or {}
    target = layout.get("target_position")
    prior = layout.get("prior_position")
    floor_y = target_floor_y(layout)
    out = []
    for entry in record.get("goal_commit_log") or []:
        centre = entry.get("center")
        if not centre or target is None:
            continue
        same_storey = floor_y is None or abs(float(centre[1]) - float(target[1])) <= 1.5
        if _hyp(centre, target) <= SUCCESS_M and same_storey:
            kind = "TRUE"
        elif prior is not None and _hyp(centre, prior) <= SUCCESS_M \
                and abs(float(centre[1]) - float(prior[1])) <= 1.5:
            kind = "STALE"
        else:
            kind = "FP"
        if entry.get("prior") and not entry.get("seen_live"):
            source = "prior"
        elif entry.get("proposal_only"):
            source = "proposal"
        else:
            source = "live"
        out.append((kind, source))
    return out


def episode_row(record: Dict[str, Any]) -> Dict[str, Any]:
    layout = record.get("authored_layout") or {}
    floor_y = target_floor_y(layout)
    start_y = record.get("start_y")
    stats = record.get("agent_stats") or {}
    portal_log = record.get("portal_log") or []
    y_range = record.get("traj_y_range")
    row = {
        "episode_id": record.get("episode_id"),
        "scene": str(record.get("scene", ""))[:5],
        "target": record.get("target"),
        "class": relocation_class(layout),
        "start_on_target_storey": (
            None if start_y is None or floor_y is None
            else abs(float(start_y) - floor_y) <= FLOOR_DY_M),
        "reached_target_storey": record.get("goal_floor_reached"),
        "left_start_storey": None if y_range is None else float(y_range) > FLOOR_DY_M,
        "switch_attempts": int(stats.get("floor_switch_attempts", 0) or 0),
        "first_switch_step": (int(portal_log[0][0]) if portal_log else None),
        "steps": int(record.get("steps") or 0),
        "spl_recorded": float(record.get("spl") or 0.0),
        "commits": commits(record),
        # The climb, for section 3b.
        "need": (None if start_y is None or floor_y is None
                 else 0 if abs(float(start_y) - floor_y) <= FLOOR_DY_M
                 else (1 if floor_y > float(start_y) else -1)),
        "flights_up": int(stats.get("flight_switch_up", 0) or 0),
        "flights_down": int(stats.get("flight_switch_down", 0) or 0),
        "climbs": int(stats.get("climb_start", 0) or 0),
        "climbs_ok": int(stats.get("climb_ok", 0) or 0),
        "climb_ends": {k[len("climb_end_"):]: int(v) for k, v in stats.items()
                       if k.startswith("climb_end_")},
        "climb_steps": climb_steps(record),
        "y_range": None if y_range is None else float(y_range),
    }
    row.update(scored(record))
    return row


def climb_steps(record: Dict[str, Any]) -> int:
    """Steps spent in the CLIMB state, from `state_log` ([step, state] at each change)."""
    log = record.get("state_log") or []
    end = int(record.get("steps") or 0)
    total = 0
    for i, (step, state) in enumerate(log):
        if state == "climb":
            nxt = int(log[i + 1][0]) if i + 1 < len(log) else end
            total += max(0, nxt - int(step))
    return total


def pct(k: int, n: int) -> str:
    if n == 0:
        return "-"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100.0 * k / n:.1f}% [{100 * lo:.0f}, {100 * hi:.0f}]"


def table(title: str, header: List[str], rows: List[List[str]]) -> None:
    print(f"\n## {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")


def report(runs: "OrderedDict[str, List[Dict[str, Any]]]", flips: bool) -> Dict[str, Any]:
    names = list(runs)
    table(
        "1. Success on the object's storey (horizontal-only rule beside it)",
        ["run", "n", "SR, same storey [Wilson 95%]", "SR, first attempt",
         "SR, horizontal only", "rejected"],
        [[name, str(len(rows)),
          pct(sum(r["success"] for r in rows), len(rows)),
          pct(sum(r["success_first"] for r in rows), len(rows)),
          pct(sum(r["success_horizontal"] for r in rows), len(rows)),
          str(sum(r["success_horizontal"] - r["success"] for r in rows))]
         for name, rows in runs.items()],
    )
    if flips:
        for name, rows in runs.items():
            for r in rows:
                if r["success_horizontal"] and not r["success"]:
                    print(f"  {name}: {r['scene']} {r['target']} -- scored from another storey")

    by_class = []
    for cls in CLASSES:
        line, any_n = [cls], False
        for name in names:
            rows = [r for r in runs[name] if r["class"] == cls]
            any_n |= bool(rows)
            line.append(pct(sum(r["success"] for r in rows), len(rows)))
        if any_n:
            by_class.append(line)
    table("2a. By relocation class (from the authored geometry)", ["class"] + names, by_class)

    by_start = []
    for flag, label in ((True, "starts on the object's storey"),
                        (False, "starts on another storey")):
        line = [label]
        for name in names:
            rows = [r for r in runs[name] if r["start_on_target_storey"] is flag]
            line.append(pct(sum(r["success"] for r in rows), len(rows)))
        by_start.append(line)
    table("2b. By start storey", ["start"] + names, by_start)

    floor_rows = []
    for name, rows in runs.items():
        away = [r for r in rows if r["start_on_target_storey"] is False]
        home = [r for r in rows if r["start_on_target_storey"] is True]
        reached = [r for r in away if r["reached_target_storey"]]
        firsts = sorted(r["first_switch_step"] for r in rows if r["first_switch_step"] is not None)
        floor_rows.append([
            name,
            pct(len(reached), len(away)),
            pct(sum(r["success"] for r in reached), len(reached)),
            pct(sum(bool(r["left_start_storey"]) for r in home), len(home)),
            (f"{sum(r['switch_attempts'] for r in rows) / max(len(rows), 1):.2f}"),
            (str(firsts[len(firsts) // 2]) if firsts else "-"),
        ])
    table(
        "3. Storey decisions",
        ["run", "reached object's storey (started elsewhere)", "SR once reached",
         "left the start storey when the object was on it", "switch attempts / ep",
         "median first switch step"],
        floor_rows,
    )

    climb_rows = []
    for name, rows in runs.items():
        moving = [r for r in rows if r["need"] in (1, -1)]
        pursued = [r for r in moving if r["flights_up"] + r["flights_down"] > 0]
        wrong = [r for r in pursued
                 if (r["flights_up"] if r["need"] > 0 else r["flights_down"]) == 0]
        ends = Counter()
        for r in rows:
            ends.update(r["climb_ends"])
        n_climbs = sum(r["climbs"] for r in rows)
        climb_rows.append([
            name,
            str(n_climbs),
            pct(sum(r["climbs_ok"] for r in rows), n_climbs),
            ", ".join(f"{k} {v}" for k, v in ends.most_common(4)) or "-",
            f"{sum(r['climb_steps'] for r in rows) / max(len(rows), 1):.0f}",
            pct(len(wrong), len(pursued)),
            pct(sum((r["y_range"] or 0.0) >= 1.5 for r in moving), len(moving)),
        ])
    table(
        "3b. Stair climbs",
        ["run", "climbs", "ended on a storey", "how climbs ended", "steps climbing / ep",
         "only wrong-way flights (of those pursuing one)", "rose 1.5 m+ (had to change storey)"],
        climb_rows,
    )

    commit_rows = []
    for name, rows in runs.items():
        c = Counter(kind for r in rows for kind, _ in r["commits"])
        s = Counter(f"{kind}/{src}" for r in rows for kind, src in r["commits"])
        total = sum(c.values())
        commit_rows.append([
            name, str(total), str(c["TRUE"]), str(c["STALE"]), str(c["FP"]),
            (f"{100.0 * c['FP'] / total:.0f}%" if total else "-"),
            ", ".join(f"{k} {v}" for k, v in sorted(s.items())),
        ])
    table("4. Goal commits", ["run", "commits", "TRUE", "STALE", "FP", "FP share", "by source"],
          commit_rows)
    return {name: rows for name, rows in runs.items()}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run dir, or NAME=run_dir")
    parser.add_argument("--flips", action="store_true",
                        help="list episodes the horizontal-only rule scored from another storey")
    parser.add_argument("--json", type=Path, help="write the per-episode rows here")
    args = parser.parse_args(argv)
    runs: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for spec in args.runs:
        name, _, path = spec.partition("=") if "=" in spec else (spec, "", spec)
        records = load(Path(path))
        if not records:
            print(f"no episodes under {path}", file=sys.stderr)
            return 2
        # A repeated name pools its directories: OSG=s42 OSG=s43 OSG=s44.
        runs.setdefault(name, []).extend(episode_row(r) for r in records)
    out = report(runs, args.flips)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
