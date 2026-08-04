"""GT-localization-error analysis for ObjectNav failures.

For every episode in a run, compare where the agent STOPPED (final_xy) against
the episode's ground-truth goal geometry (all annotated goal objects of the
target category and their success view_points, from the HM3D dataset's
goals_by_category). This separates the two failure mechanisms the stage
decomposition (scripts/analyze_stages.py) could not tell apart:

  * approach-geometry miss  -> agent stopped NEAR a real goal object
    (d_obj small) but outside its success viewpoint set (dtg > success_dist):
    the object was found and reached, the final stop pose/orientation missed
    the viewpoint region. Fix = approach termination geometry.

  * perception / localization miss -> agent stopped FAR from every annotated
    goal object of the category (d_obj large): it committed to a location with
    no real target -- a detector false positive or a mislocalized scene-graph
    track pointing at empty space. Fix = detection precision / 3D localization.

In HM3D ObjectNav essentially every annotated instance of the target category
is a goal, so "far from all goal objects" means the agent stood where there is
no real target of that category.

Coordinates: habitat world, y up; ground plane = (x, z) = position[[0, 2]],
matching final_xy = camera_position[[0, 2]].

Usage: python scripts/analyze_localization.py [outputs/<dir>/] \
           [--episodes data/datasets/objectnav/hm3d/v1/val]
"""
import argparse
import glob
import gzip
import json
import os
import statistics as st
from collections import defaultdict

import numpy as np

SUCCESS_DIST = 0.18
NEAR_OBJ_M = 1.5   # stop within this of a goal object center -> "reached object"
FAR_OBJ_M = 3.0    # stop beyond this of every goal object -> "wrong location"
XZ = [0, 2]


def load_goals(episodes_dir):
    """{f'{scene}_{category}': {'objs': [xz...], 'vps': [xz...]}} from all
    content shards of a split."""
    goals = {}
    files = glob.glob(os.path.join(episodes_dir, "content", "*.json.gz"))
    for f in files:
        d = json.load(gzip.open(f))
        for key, objlist in d.get("goals_by_category", {}).items():
            objs, vps = [], []
            for g in objlist:
                objs.append(np.asarray(g["position"], float)[XZ])
                for vp in g.get("view_points", []) or []:
                    vps.append(np.asarray(vp["agent_state"]["position"], float)[XZ])
            goals[key] = {
                "objs": np.array(objs) if objs else np.zeros((0, 2)),
                "vps": np.array(vps) if vps else np.zeros((0, 2)),
            }
    return goals


def min_dist(pt, pts):
    if pts.shape[0] == 0:
        return None
    return float(np.min(np.linalg.norm(pts - pt, axis=1)))


def reached_approach(row):
    states = {s for _, s in row.get("state_log", [])}
    return bool(states & {"approach", "done"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--episodes", default="data/datasets/objectnav/hm3d/v1/val")
    args = ap.parse_args()

    d = args.run_dir
    if d is None:
        d = sorted([p for p in glob.glob("outputs/*/") if os.path.exists(p + "episodes.jsonl")],
                   key=os.path.getmtime)[-1]
    if not d.endswith("/"):
        d += "/"
    rows = [json.loads(l) for l in open(d + "episodes.jsonl")]
    goals = load_goals(args.episodes)
    print(f"=== GT-localization analysis: {d}  ({len(rows)} episodes) ===")
    print(f"    goal catalog: {len(goals)} scene_category keys from {args.episodes}\n")

    # annotate every episode with d_obj (nearest goal object) and d_vp
    miss = 0
    for r in rows:
        key = f"{r['scene']}_{r['target']}"
        g = goals.get(key)
        if g is None or r.get("final_xy") is None:
            r["_d_obj"] = r["_d_vp"] = None
            miss += 1
            continue
        pt = np.asarray(r["final_xy"], float)
        r["_d_obj"] = min_dist(pt, g["objs"])
        r["_d_vp"] = min_dist(pt, g["vps"])
    if miss:
        print(f"  (warning: {miss} episodes had no goal-catalog match / no final_xy)\n")

    ap_rows = [r for r in rows if reached_approach(r)]
    ap_fail = [r for r in ap_rows if not r.get("success") and r.get("_d_obj") is not None]
    succ = [r for r in ap_rows if r.get("success") and r.get("_d_obj") is not None]

    def stats(name, grp):
        do = [r["_d_obj"] for r in grp]
        dv = [r["_d_vp"] for r in grp]
        if not do:
            return
        print(f"  {name:22s} n={len(grp):3d}  "
              f"d_obj med={st.median(do):5.2f}m  d_vp med={st.median(dv):5.2f}m")

    print("stop-pose distance to nearest GT goal object / viewpoint (Euclidean, x-z):")
    stats("SUCCESS", succ)
    stats("approach FAIL", ap_fail)
    print()

    # the money split: of the approach failures, where did the agent stop?
    reached_obj = [r for r in ap_fail if r["_d_obj"] <= NEAR_OBJ_M]
    wrong_loc = [r for r in ap_fail if r["_d_obj"] > FAR_OBJ_M]
    mid = [r for r in ap_fail if NEAR_OBJ_M < r["_d_obj"] <= FAR_OBJ_M]
    n = len(ap_fail)
    print(f"approach failures decomposed by distance to nearest real goal object (n={n}):")
    print(f"  reached goal object (<= {NEAR_OBJ_M}m): {len(reached_obj):3d}  {len(reached_obj)/n:5.1%}"
          f"   -> APPROACH-GEOMETRY miss (found+reached object, missed viewpoint)")
    print(f"  intermediate ({NEAR_OBJ_M}-{FAR_OBJ_M}m)     : {len(mid):3d}  {len(mid)/n:5.1%}")
    print(f"  wrong location (> {FAR_OBJ_M}m)   : {len(wrong_loc):3d}  {len(wrong_loc)/n:5.1%}"
          f"   -> PERCEPTION/LOCALIZATION miss (no real target where it stopped)")
    print()

    # for the 'reached object' group, how far outside the viewpoint set? this
    # bounds how much a better terminal-geometry rule could recover.
    if reached_obj:
        dv = sorted(r["_d_vp"] for r in reached_obj)
        recoverable = sum(1 for r in reached_obj if r["_d_vp"] <= 0.5)
        print(f"of the {len(reached_obj)} that reached a goal object but failed:")
        print(f"  d_vp (to nearest success viewpoint): "
              f"med={st.median(dv):.2f}m  min={dv[0]:.2f}m  "
              f"<=0.5m: {recoverable}")
        print(f"  -> ~{recoverable} are within 0.5m of a real viewpoint: a tighter/"
              f"viewpoint-aware terminal stop could plausibly convert these.")
    print()

    # whole-run headline
    total = len(rows)
    succ_all = sum(1 for r in rows if r.get("success"))
    print(f"headline: SR {succ_all}/{total} = {succ_all/total:.1%};  "
          f"approach reached {len(ap_rows)}, failed {len(ap_fail)+ (len(ap_rows)-len(ap_fail)-len(succ))}")
    print(f"  attribution of approach failures: "
          f"{len(reached_obj)/n:.0%} geometry / {len(wrong_loc)/n:.0%} perception-localization"
          f" / {len(mid)/n:.0%} mid")


if __name__ == "__main__":
    main()
