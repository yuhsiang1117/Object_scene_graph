"""Approach-navigation failure diagnostics.

Requires a run produced with approach_diag logging (runner logs agent.approach_diag:
goal_xy, obj_xy, goal_to_obj_m, goal_cell, min_dist_to_goal_m, plan_fail,
path_consumed_cause). Focuses on the NAVIGATION-failure bucket identified by
scripts/analyze_trackloc.py -- episodes that correctly mapped the target
(track center within 1.5m of a real goal object) but stopped short -- and asks
WHY the terminal approach failed to reach a viewpoint.

Key questions answered:
  * For path_consumed stops: planner_no_path (couldn't route to the goal cell)
    vs controller_arrived (false arrival at a stub path)?
  * Was the approach goal cell reachable-looking (free) yet unreached
    (min_dist_to_goal large)? -> planner reachability failure.
  * Is there a GT success viewpoint near the approach goal? -> confirms the
    target/goal was valid and the loss is purely navigation.

Usage: python scripts/analyze_approach.py [outputs/<dir>/] \
           [--episodes data/datasets/objectnav/hm3d/v1/val]
"""
import argparse
import glob
import gzip
import json
import os
import statistics as st
from collections import Counter

import numpy as np

XZ = [0, 2]
NEAR_M = 1.5


def load_gt(episodes_dir):
    gobj, gvp = {}, {}
    for f in glob.glob(os.path.join(episodes_dir, "content", "*.json.gz")):
        dd = json.load(gzip.open(f))
        for k, ol in dd.get("goals_by_category", {}).items():
            gobj[k] = np.array([np.asarray(g["position"], float)[XZ] for g in ol])
            vps = [np.asarray(vp["agent_state"]["position"], float)[XZ]
                   for g in ol for vp in (g.get("view_points") or [])]
            gvp[k] = np.array(vps) if vps else np.zeros((0, 2))
    return gobj, gvp


def mind(pt, pts):
    if pt is None or pts is None or len(pts) == 0:
        return None
    return float(np.min(np.linalg.norm(pts - np.asarray(pt, float), axis=1)))


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--episodes", default="data/datasets/objectnav/hm3d/v1/val")
    args = ap.parse_args()
    d = args.run_dir or sorted(
        [p for p in glob.glob("outputs/*/") if os.path.exists(p + "episodes.jsonl")],
        key=os.path.getmtime)[-1]
    if not d.endswith("/"):
        d += "/"
    rows = [json.loads(l) for l in open(d + "episodes.jsonl")]
    if not any(r.get("approach_diag") for r in rows):
        raise SystemExit("run has no approach_diag logging; rerun with the updated agent")
    gobj, gvp = load_gt(args.episodes)
    print(f"=== approach-navigation diagnostics: {d} ({len(rows)} eps) ===\n")

    # navigation-failure bucket: committed (target_obj_xy set), failed, track
    # well-localized (err_loc <= 1.5m)
    nav = []
    for r in rows:
        if not r.get("target_obj_xy") or r.get("success"):
            continue
        el = mind(r["target_obj_xy"], gobj.get(f"{r['scene']}_{r['target']}"))
        if el is not None and el <= NEAR_M:
            nav.append(r)
    print(f"NAVIGATION-failure episodes (well-localized target, stopped short): {len(nav)}\n")

    print("stop-reason distribution:", dict(Counter(r.get("approach_stop_reason") for r in nav)))
    print()

    # path_consumed breakdown
    pc = [r for r in nav if r.get("approach_stop_reason") == "path_consumed"]
    causes = Counter((r.get("approach_diag") or {}).get("path_consumed_cause") for r in pc)
    print(f"path_consumed stops: {len(pc)}")
    for cause, c in causes.most_common():
        print(f"    {str(cause):18s} {c}")
    print()

    # approach geometry for the nav-failure bucket
    print("approach geometry (nav-failure bucket):")
    diags = [r.get("approach_diag") or {} for r in nav]
    print(f"  goal_to_obj_m (goal cell vs mapped object): med={med([x.get('goal_to_obj_m') for x in diags]):.2f}m")
    print(f"  min_dist_to_goal_m (closest agent got to its goal): "
          f"med={med([x.get('min_dist_to_goal_m') for x in diags]):.2f}m")
    gc = Counter(x.get("goal_cell") for x in diags)
    print(f"  goal_cell status: {dict(gc)}")
    print()

    # did the agent ever get near its goal? (min_dist small => reached goal but
    # goal was wrong; min_dist large => never reached goal => planner failure)
    reached = [r for r in nav if (r.get("approach_diag") or {}).get("min_dist_to_goal_m") is not None
               and (r["approach_diag"]["min_dist_to_goal_m"]) <= 0.5]
    stalled = [r for r in nav if (r.get("approach_diag") or {}).get("min_dist_to_goal_m") is not None
               and (r["approach_diag"]["min_dist_to_goal_m"]) > 1.0]
    n = len(nav)
    print(f"reached approach goal (min_dist<=0.5m): {len(reached):3d}  {len(reached)/n:5.1%}"
          f"  -> goal reached but not a success viewpoint (goal-SELECTION)")
    print(f"stalled far from goal (min_dist>1.0m) : {len(stalled):3d}  {len(stalled)/n:5.1%}"
          f"  -> never reached a free, object-adjacent goal (planner REACHABILITY)")
    print()

    # validate the goal was worth reaching: GT viewpoint near the approach goal
    dvp = [mind((r.get("approach_diag") or {}).get("goal_xy"),
                gvp.get(f"{r['scene']}_{r['target']}")) for r in nav]
    print(f"d(approach goal -> nearest GT success viewpoint): med={med(dvp):.2f}m "
          f"(<=0.5m: {sum(1 for x in dvp if x is not None and x <= 0.5)}/{n})")
    print("  -> confirms the approach goal sits on/next to a real success pose;")
    print("     the loss is the planner/controller failing to reach it.")


if __name__ == "__main__":
    main()
