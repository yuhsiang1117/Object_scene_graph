"""Definitive localization-vs-navigation split for ObjectNav approach failures.

Requires a run produced with track-center logging (runner._target_track_fields:
target_obj_xy, cand_best_cam_xy, cand_best_score). For every episode that
committed to a target (reached APPROACH), computes against GT goal geometry:

  err_loc = d(target_obj_xy, nearest GT goal object)
      scene-graph LOCALIZATION error: how far the mapped 3D track center is
      from any real object of the target category.

  err_nav = d(final_xy, target_obj_xy)
      residual NAVIGATION error: how far the agent stopped from its own
      committed target center.

  err_cam = d(cand_best_cam_xy, nearest GT goal object)
      how far the pose that produced the track's best detection was from a real
      goal object -- used to split a large err_loc into:
        * localization DRIFT  (err_cam small, err_loc large): the detection
          view was near a real object, but the 3D center estimate drifted away
          (monocular parallax fit) -> fix the ellipsoid localization.
        * false-positive DETECTION (err_cam also large): the detector fired on
          a category-adjacent / non-goal object far from any real target
          -> detection precision problem.

Usage: python scripts/analyze_trackloc.py [outputs/<dir>/] \
           [--episodes data/datasets/objectnav/hm3d/v1/val]
"""
import argparse
import glob
import gzip
import json
import os
import statistics as st

import numpy as np

XZ = [0, 2]
NEAR_M = 1.5
FAR_M = 3.0


def load_goals(episodes_dir):
    goals = {}
    for f in glob.glob(os.path.join(episodes_dir, "content", "*.json.gz")):
        d = json.load(gzip.open(f))
        for key, objlist in d.get("goals_by_category", {}).items():
            objs = [np.asarray(g["position"], float)[XZ] for g in objlist]
            goals[key] = np.array(objs) if objs else np.zeros((0, 2))
    return goals


def mind(pt, pts):
    if pt is None or pts is None or pts.shape[0] == 0:
        return None
    return float(np.min(np.linalg.norm(pts - np.asarray(pt, float), axis=1)))


def reached_approach(row):
    return bool({s for _, s in row.get("state_log", [])} & {"approach", "done"})


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
    if not any("target_obj_xy" in r for r in rows):
        raise SystemExit("run has no track-center logging; rerun with the updated runner")
    goals = load_goals(args.episodes)
    print(f"=== track-center localization split: {d} ({len(rows)} eps) ===\n")

    committed = []
    for r in rows:
        if r.get("target_obj_xy") is None:
            continue
        gpts = goals.get(f"{r['scene']}_{r['target']}")
        r["_err_loc"] = mind(r["target_obj_xy"], gpts)
        r["_err_cam"] = mind(r.get("cand_best_cam_xy"), gpts)
        r["_err_nav"] = (
            float(np.linalg.norm(np.asarray(r["final_xy"], float)
                                 - np.asarray(r["target_obj_xy"], float)))
            if r.get("final_xy") is not None else None
        )
        committed.append(r)

    succ = [r for r in committed if r.get("success")]
    fail = [r for r in committed if not r.get("success")]
    print(f"committed to a target (reached APPROACH): {len(committed)}  "
          f"(success {len(succ)}, fail {len(fail)})\n")

    print("median errors (m):            err_loc  err_nav  err_cam")
    for name, grp in [("SUCCESS", succ), ("FAIL", fail)]:
        print(f"  {name:20s} {med([r['_err_loc'] for r in grp]):7.2f} "
              f"{med([r['_err_nav'] for r in grp]):8.2f} {med([r['_err_cam'] for r in grp]):8.2f}")
    print()

    # PRIMARY SPLIT: is the mapped track center near a real goal object?
    loc_ok = [r for r in fail if r["_err_loc"] is not None and r["_err_loc"] <= NEAR_M]
    loc_bad = [r for r in fail if r["_err_loc"] is not None and r["_err_loc"] > FAR_M]
    loc_mid = [r for r in fail if r["_err_loc"] is not None and NEAR_M < r["_err_loc"] <= FAR_M]
    n = len(fail)
    print(f"approach FAILURES split by scene-graph localization error (n={n}):")
    print(f"  track WELL-localized (err_loc<= {NEAR_M}m): {len(loc_ok):3d}  {len(loc_ok)/n:5.1%}"
          f"  -> NAVIGATION failure (correct target, agent didn't reach its viewpoint)")
    print(f"  intermediate ({NEAR_M}-{FAR_M}m)            : {len(loc_mid):3d}  {len(loc_mid)/n:5.1%}")
    print(f"  track MIS-localized (err_loc> {FAR_M}m)  : {len(loc_bad):3d}  {len(loc_bad)/n:5.1%}"
          f"  -> track center far from any real target")
    print()

    # for well-localized: confirm nav is the problem (agent stopped far from its
    # own correctly-placed track)
    if loc_ok:
        print(f"of the {len(loc_ok)} well-localized failures: "
              f"median err_nav={med([r['_err_nav'] for r in loc_ok]):.2f}m "
              f"(agent stopped this far from its own correctly-mapped target)")
        print()

    # SECONDARY SPLIT: for mis-localized tracks, drift vs false-positive
    if loc_bad:
        drift = [r for r in loc_bad if r["_err_cam"] is not None and r["_err_cam"] <= NEAR_M]
        fp = [r for r in loc_bad if r["_err_cam"] is not None and r["_err_cam"] > FAR_M]
        camnone = [r for r in loc_bad if r["_err_cam"] is None]
        m = len(loc_bad)
        print(f"mis-localized tracks split by detection-view error err_cam (n={m}):")
        print(f"  localization DRIFT (err_cam<= {NEAR_M}m): {len(drift):3d}  {len(drift)/m:5.1%}"
              f"  -> real object seen, 3D center estimate drifted (ellipsoid fit)")
        print(f"  false-positive DET (err_cam> {FAR_M}m) : {len(fp):3d}  {len(fp)/m:5.1%}"
              f"  -> detector fired far from any real target")
        mid_cam = m - len(drift) - len(fp) - len(camnone)
        print(f"  intermediate/unknown                : {mid_cam + len(camnone):3d}")
        if fp:
            print(f"    (false-positive tracks: median best_score="
                  f"{med([r.get('cand_best_score') for r in fp]):.2f}, "
                  f"n_obs={med([r.get('cand_n_obs') for r in fp]):.0f})")
    print()

    # headline attribution over ALL approach failures
    print("=== attribution of approach failures ===")
    print(f"  navigation (well-localized target, missed):   {len(loc_ok)/n:5.1%}")
    print(f"  intermediate:                                 {len(loc_mid)/n:5.1%}")
    if loc_bad:
        drift_frac = len(drift) / n
        fp_frac = len(fp) / n
        other_ml = (len(loc_bad) - len(drift) - len(fp)) / n
        print(f"  localization drift (bad 3D center):           {drift_frac:5.1%}")
        print(f"  false-positive detection:                     {fp_frac:5.1%}")
        print(f"  mis-localized (other/unknown view):           {other_ml:5.1%}")


if __name__ == "__main__":
    main()
