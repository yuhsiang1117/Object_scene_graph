"""Decompose ObjectNav failures into exploration vs approach stages.

Reads a run's episodes.jsonl and classifies every episode by the furthest
phase of the NavAgent state machine it reached, cross-tabbed with success and
final distance-to-goal. This isolates where the SR is lost:

  * exploration failure  -> episode never produced a target candidate
    (never entered GOTO_VERIFY_VIEW / VERIFYING / APPROACH): the agent either
    never navigated to the object's region or the detector never mapped it as
    a candidate track. Timeouts here are wandering/coverage failures.

  * approach/localization failure -> episode DID reach APPROACH (target found,
    verified, walked toward) but STOP landed outside the success viewpoint set
    (distance_to_goal > success_distance). The last few metres failed.

Usage: python scripts/analyze_stages.py [outputs/<dir>/]
"""
import glob
import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict

def success_dist(run_dir):
    """The run's own success_distance, read from summary.json.

    Must not be hardcoded: the protocol moved from 0.18 (old ROS-matched) to
    0.10 (habitat/ASCENT default) in S0, so a fixed constant silently
    mis-classifies the "<=succ" column for one era of runs or the other.
    Falls back to 0.18 for pre-S0 runs, whose summary.json predates the field.
    """
    try:
        with open(os.path.join(run_dir, "summary.json")) as f:
            return float(json.load(f)["config"]["success_distance"])
    except (OSError, KeyError, ValueError, TypeError):
        print("[warn] no success_distance in summary.json; assuming 0.18 "
              "(pre-S0 run)")
        return 0.18


def furthest_phase(states):
    """Highest-ranked state visited in the episode's state_log."""
    rank = {
        "init": 0, "explore": 1, "goto_frontier": 2,
        "goto_verify_view": 3, "verifying": 4, "approach": 5, "done": 6,
    }
    best = "init"
    for _, s in states:
        if rank.get(s, -1) > rank.get(best, -1):
            best = s
    return best


# "done" is only reachable through _do_approach, and a same-step
# VERIFYING->APPROACH->DONE transition logs only "done" (state_log records a
# state solely when it differs from the previous step) -- so both mark that the
# episode reached the approach stage.
_APPROACH_STATES = {"approach", "done"}
_CANDIDATE_STATES = {"goto_verify_view", "verifying", "approach", "done"}


def phase_bucket(row):
    """Coarse bucket for exploration-vs-approach accounting."""
    states = {s for _, s in row.get("state_log", [])}
    if row.get("success"):
        return "SUCCESS"
    reached_candidate = bool(states & _CANDIDATE_STATES)
    reached_approach = bool(states & _APPROACH_STATES)
    if not reached_candidate:
        return "FAIL_explore"        # never even found/mapped a candidate
    if not reached_approach:
        return "FAIL_verify_stall"   # found candidate, stalled before approach
    return "FAIL_approach"           # reached approach, stopped in wrong place


def main():
    if len(sys.argv) > 1:
        d = sys.argv[1]
    else:
        d = sorted(
            [p for p in glob.glob("outputs/*/") if os.path.exists(p + "episodes.jsonl")],
            key=os.path.getmtime,
        )[-1]
    if not d.endswith("/"):
        d += "/"
    rows = [json.loads(l) for l in open(d + "episodes.jsonl")]
    n = len(rows)
    succ_dist = success_dist(d)
    print(f"=== stage decomposition: {d}  ({n} episodes, "
          f"success_distance={succ_dist}) ===\n")

    # 1. coarse exploration-vs-approach accounting
    buckets = Counter(phase_bucket(r) for r in rows)
    order = ["SUCCESS", "FAIL_explore", "FAIL_verify_stall", "FAIL_approach"]
    print("failure attribution")
    print(f"  {'bucket':20s} {'n':>4s}  {'%':>6s}")
    for b in order:
        c = buckets.get(b, 0)
        print(f"  {b:20s} {c:>4d}  {c/n:>5.1%}")
    print()

    # 2. did the agent even get near the goal in each bucket? (dtg stats)
    print("distance-to-goal (m) by bucket")
    print(f"  {'bucket':20s} {'n':>4s} {'mean':>6s} {'med':>6s} {'min':>6s} {'<=succ':>7s} {'<1m':>5s}")
    byb = defaultdict(list)
    for r in rows:
        byb[phase_bucket(r)].append(r.get("distance_to_goal"))
    for b in order:
        v = [x for x in byb.get(b, []) if x is not None]
        if not v:
            continue
        near = sum(1 for x in v if x <= succ_dist)
        near1 = sum(1 for x in v if x <= 1.0)
        print(f"  {b:20s} {len(v):>4d} {st.mean(v):>6.2f} {st.median(v):>6.2f} "
              f"{min(v):>6.2f} {near:>7d} {near1:>5d}")
    print()

    # 3. of the episodes that REACHED approach, how did the approach end?
    reached_ap = [r for r in rows if ({s for _, s in r.get("state_log", [])} & _APPROACH_STATES)]
    print(f"episodes reaching APPROACH: {len(reached_ap)}  "
          f"(success {sum(1 for r in reached_ap if r.get('success'))})")
    print("  approach_stop_reason x success:")
    sr = defaultdict(lambda: [0, 0])
    for r in reached_ap:
        reason = r.get("approach_stop_reason")
        sr[reason][0] += 1
        sr[reason][1] += int(bool(r.get("success")))
    for reason, (tot, s) in sorted(sr.items(), key=lambda kv: -kv[1][0]):
        v = [r.get("distance_to_goal") for r in reached_ap
             if r.get("approach_stop_reason") == reason and r.get("distance_to_goal") is not None]
        med = st.median(v) if v else float("nan")
        print(f"    {str(reason):14s} {s}/{tot:<3d} success   median_dtg={med:.2f}m")
    print()

    # 4. conditional SR: given the agent reached approach, how often does it
    #    convert? this is the "approach quality" number.
    if reached_ap:
        conv = sum(1 for r in reached_ap if r.get("success")) / len(reached_ap)
        print(f"P(success | reached APPROACH) = {conv:.1%}   "
              f"(approach-stage conversion)")
    cand = [r for r in rows if {s for _, s in r.get("state_log", [])} & _CANDIDATE_STATES]
    if cand:
        pc = len(cand) / n
        print(f"P(reached candidate)          = {pc:.1%}   "
              f"(exploration+detection reach)")
    print()

    # 5. how bad is the approach miss when it reaches approach but fails?
    ap_fail = [r for r in reached_ap if not r.get("success")]
    v = [r.get("distance_to_goal") for r in ap_fail if r.get("distance_to_goal") is not None]
    if v:
        near_miss = sum(1 for x in v if x <= 0.5)
        print(f"approach failures: {len(v)}  "
              f"near-miss (<=0.5m): {near_miss}  "
              f"(these are pure last-metre/viewpoint losses)")
        far = sum(1 for x in v if x > 2.0)
        print(f"                   far miss (>2m): {far}  "
              f"(approached wrong object / wrong region)")


if __name__ == "__main__":
    main()
