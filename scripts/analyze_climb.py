"""Why do climbs fail to change floor?

The stair A/B established that attempt *count* is not the bottleneck: floor
switches held at 0.40 per episode whether the agent made 3.5 climb attempts or
0.4 of them. So the question is what happens to the attempts that are made.

This decomposes them from what run_eval already records:

  agent_stats.climb_attempt / climb_ok / climb_fail
  state_log                  every CLIMB entry and what state followed it
  floor_switches, y_range_m  whether the agent actually ended up elsewhere

Three outcomes are distinguished, because they call for different fixes:

  INSTANT      entered and left CLIMB inside one step -- the state never even
               appears in state_log. Navigation had nowhere to go: the goal did
               not snap to the navmesh at all. A detection/goal-placement
               problem, not a traversal one.
  SHORT        a handful of steps, no height gained. The agent moved but the
               route was not a staircase, or it could not get onto it.
  CLIMBED      real height gained. If the floor still did not switch, the
               problem is in FloorStack's commit, not in navigation.

Usage: python scripts/analyze_climb.py [outputs/<dir>/ ...]
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys
from collections import Counter

# A climb that gains less than this never left the floor in any real sense.
CLIMBED_M = 0.5


def latest_runs():
    runs = sorted(
        (p for p in glob.glob("outputs/*/") if os.path.exists(p + "episodes.jsonl")),
        key=os.path.getmtime,
    )
    return runs[-1:]


def climb_spans(row):
    """(entry_step, duration) for every CLIMB visible in state_log.

    state_log records a state only when it differs from the previous step, so a
    climb that starts and ends within one step leaves no trace -- which is
    exactly what makes the count discrepancy below meaningful.
    """
    sl = row.get("state_log") or []
    out = []
    for i, (step, state) in enumerate(sl):
        if state != "climb":
            continue
        nxt = sl[i + 1][0] if i + 1 < len(sl) else row.get("steps", step)
        out.append((step, nxt - step))
    return out


def report(run):
    rows = [json.loads(line) for line in open(os.path.join(run, "episodes.jsonl"))]
    n = len(rows)
    att = sum(r["agent_stats"].get("climb_attempt", 0) for r in rows)
    ok = sum(r["agent_stats"].get("climb_ok", 0) for r in rows)
    fail = sum(r["agent_stats"].get("climb_fail", 0) for r in rows)
    visible = [s for r in rows for s in climb_spans(r)]
    instant = att - len(visible)

    print(f"\n=== {run}  ({n} episodes) ===")
    if att == 0:
        print("  no climb attempts (multi_floor off, or stair_prior 0)")
        return

    print(f"  attempts {att}   ok {ok} ({ok / att:.1%})   fail {fail}")
    print(f"  outcome split:")
    print(f"    INSTANT (never lasted a step)  {instant:4d}  {instant / att:6.1%}")
    if visible:
        durs = [d for _, d in visible]
        print(f"    lasted >=1 step                {len(visible):4d}  "
              f"{len(visible) / att:6.1%}   median {st.median(durs):.0f} steps")

    # Did the agent actually go anywhere? y_range is the honest check: an
    # episode can log climb_ok and still be on the floor it started on.
    climbed = [r for r in rows if r.get("y_range_m", 0.0) > CLIMBED_M]
    switched = [r for r in rows if r.get("floor_switches", 0) > 0]
    succ_switched = [r for r in switched if r["success"]]
    print(f"  episodes: attempted a climb {sum(1 for r in rows if r['agent_stats'].get('climb_attempt')):3d}"
          f"   gained >{CLIMBED_M} m {len(climbed):3d}"
          f"   switched floor {len(switched):3d}"
          f"   switched AND succeeded {len(succ_switched):3d}")

    # The gap that matters: attempts that moved the agent vertically but never
    # committed a floor change would point at FloorStack, not navigation.
    stuck = [r for r in climbed if r.get("floor_switches", 0) == 0]
    if stuck:
        print(f"  gained height but never committed a floor: {len(stuck)}"
              f"  (median gain {st.median([r['y_range_m'] for r in stuck]):.2f} m)")
        print("    -> look at FloorStack band/commit_steps, not at navigation")

    by_target = Counter(r["target"] for r in rows if r.get("floor_switches", 0) > 0)
    if by_target:
        print(f"  floor changes by target: {dict(by_target)}")


def main() -> None:
    runs = sys.argv[1:] or latest_runs()
    if not runs:
        sys.exit("no runs found under outputs/")
    for run in runs:
        report(run if run.endswith("/") else run + "/")


if __name__ == "__main__":
    main()
