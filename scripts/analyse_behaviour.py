#!/usr/bin/env python
"""Turn behaviour logs into the numbers that explain a run.

    python scripts/analyse_behaviour.py outputs/RUN [outputs/OTHER]

Reads `episodes.jsonl` (needs `eval.debug_frames=true`, which is what carries
the per-step log) and overlays the dataset's own goal geometry, which the agent
never sees. Answers, per run and side by side:

  reach       did the agent ever get near a goal, and did it get there before
              it committed somewhere else?
  sight       when a goal was in frame, did the detector fire?  (the denominator
              is the honest part: a view-point is a place to STAND, so it is
              reported against the OBJECT position as well)
  framing     of the steps spent near the object, how often was it even in view
  motion      commanded forwards that produced no displacement
  taxonomy    why each failure failed, in one bucket each

Every one of these was reconstructed by hand for S47-S50 and cost a bespoke
re-run each time.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

HFOV_DEG = 79.0


def load_goal_geometry(episodes_root: str) -> dict:
    """{uid: (view_points_xy, object_xy)} in the agent's (x, -z) plane."""
    out = {}
    for f in sorted(glob.glob(f"{episodes_root}/*.json.gz")):
        d = json.load(gzip.open(f))
        gbc = d.get("goals_by_category", {})
        for i, e in enumerate(d["episodes"]):        # habitat: episode_id = str(index)
            key = f"{e['scene_id'].split('/')[-1]}_{e['object_category']}"
            goals = gbc.get(key)
            if not goals:
                continue
            vps = [vp["agent_state"]["position"] for g in goals for vp in g.get("view_points", [])]
            objs = [g["position"] for g in goals if "position" in g]
            flip = lambda a: np.stack([np.asarray(a)[:, 0], -np.asarray(a)[:, 2]], 1)
            out[f"{e['scene_id'].split('/')[-1]}:{i}"] = (
                flip(vps) if vps else None, flip(objs) if objs else None)
    return out


def load_goal_geometry_3d(episodes_root: str) -> dict:
    """{uid: object positions as (x, -z, y)} -- the plane the agents log in,
    plus habitat's height, for gating instances to a storey."""
    out = {}
    for f in sorted(glob.glob(f"{episodes_root}/*.json.gz")):
        d = json.load(gzip.open(f))
        gbc = d.get("goals_by_category", {})
        for i, e in enumerate(d["episodes"]):
            key = f"{e['scene_id'].split('/')[-1]}_{e['object_category']}"
            objs = [g["position"] for g in gbc.get(key, []) if "position" in g]
            if not objs:
                continue
            a = np.asarray(objs, float)
            out[f"{e['scene_id'].split('/')[-1]}:{i}"] = np.stack([a[:, 0], -a[:, 2], a[:, 1]], 1)
    return out


def uid(row: dict) -> str:
    return row.get("uid") or f"{row.get('scene','?')}:{row['episode_id']}"


def is_cross(row: dict) -> bool:
    if row.get("floor_class") is not None:
        return row["floor_class"] != "same_floor"
    return (row.get("goal_floor_gap_m") or 0) > 0.5


def in_frame(xy, yaw, targets, max_d, half_deg):
    """Steps where any target is within range and inside the horizontal FOV."""
    rel = targets[None, :, :] - xy[:, None, :]
    dist = np.linalg.norm(rel, axis=2)
    ang = (np.arctan2(rel[..., 1], rel[..., 0]) - yaw[:, None] + np.pi) % (2 * np.pi) - np.pi
    return ((dist < max_d) & (np.abs(ang) < np.radians(half_deg))).any(axis=1), dist.min(axis=1)


def analyse(run: Path, geom: dict) -> dict:
    rows = [json.loads(l) for l in open(run / "episodes.jsonl")]
    n = len(rows)
    res = {"run": run.name, "n": n,
           "SR": 100 * sum(r["success"] for r in rows) / n,
           "SPL": float(np.mean([r["spl"] for r in rows])),
           "steps": float(np.mean([r["steps"] for r in rows]))}
    cf = [r for r in rows if is_cross(r)]
    sf = [r for r in rows if not is_cross(r)]
    res["SR_same"] = 100 * sum(r["success"] for r in sf) / max(len(sf), 1)
    res["SR_cross"] = 100 * sum(r["success"] for r in cf) / max(len(cf), 1)

    tax, blocked, fwd = Counter(), 0, 0
    iv_vp = fire_vp = iv_obj = fire_obj = near_obj = 0
    passed_then_committed = closest = 0
    dists = []
    for r in rows:
        trace = r.get("step_trace") or []
        for t in trace:
            if t.get("act") == "move_forward":
                fwd += 1
                blocked += int(t.get("blocked", 0))
        vps, objs = geom.get(uid(r), (None, None))
        if trace and vps is not None:
            xy = np.array([t["xy"] for t in trace], float)
            yaw = np.array([t["yaw"] for t in trace], float)
            det = np.array([bool(t.get("ndet", 0)) for t in trace])
            m, dmin = in_frame(xy, yaw, vps, 3.0, 20.0)
            iv_vp += int(m.sum()); fire_vp += int((m & det).sum())
            dists.append(float(dmin.min()))
            if objs is not None:
                mo, dobj = in_frame(xy, yaw, objs, 3.0, HFOV_DEG / 2)
                iv_obj += int(mo.sum()); fire_obj += int((mo & det).sum())
                near_obj += int((dobj < 3.0).sum())
            if not r["success"]:
                app = [i for i, t in enumerate(trace) if t.get("state") in ("approach", "navigate")]
                if app and int(np.argmin(dmin)) < app[-1] and dmin.min() <= 3.0:
                    passed_then_committed += 1
                if dmin.min() <= 3.0:
                    closest += 1
        if r["success"]:
            continue
        if r["steps"] >= 500:
            tax["timeout"] += 1
        elif r.get("approach_stop_reason") and r["distance_to_goal"] > 3.0:
            tax["stopped_on_wrong_object"] += 1
        elif r.get("approach_stop_reason"):
            tax["stopped_near_miss"] += 1
        else:
            tax["other"] += 1
    res.update({
        "blocked_forwards": blocked, "forwards": fwd,
        "blocked_frac": round(blocked / max(fwd, 1), 3),
        "recall_viewpoint_in_frame": 100 * fire_vp / max(iv_vp, 1), "n_vp_in_frame": iv_vp,
        "recall_object_in_frame": 100 * fire_obj / max(iv_obj, 1), "n_obj_in_frame": iv_obj,
        "framing_rate": 100 * iv_obj / max(near_obj, 1), "n_near_obj": near_obj,
        "failures_that_reached_a_goal": closest,
        "passed_goal_then_committed": passed_then_committed,
        "median_closest_approach_m": float(np.median(dists)) if dists else None,
        "taxonomy": dict(tax),
    })
    return res


def show(a: dict, b: dict | None) -> None:
    keys = [("SR", "%"), ("SPL", ""), ("SR_same", "%"), ("SR_cross", "%"), ("steps", ""),
            ("blocked_forwards", ""), ("blocked_frac", ""),
            ("recall_viewpoint_in_frame", "%"), ("recall_object_in_frame", "%"),
            ("framing_rate", "%"), ("median_closest_approach_m", " m"),
            ("failures_that_reached_a_goal", ""), ("passed_goal_then_committed", "")]
    w = max(len(k) for k, _ in keys) + 2
    head = f"{'metric':{w}s} {a['run']:>22s}" + (f" {b['run']:>22s}" if b else "")
    print(head); print("-" * len(head))
    for k, unit in keys:
        va = a.get(k)
        line = f"{k:{w}s} {('' if va is None else f'{va:.2f}{unit}'):>22s}"
        if b:
            vb = b.get(k)
            line += f" {('' if vb is None else f'{vb:.2f}{unit}'):>22s}"
        print(line)
    print("\nfailure taxonomy")
    for run in (a, b) if b else (a,):
        print(f"  {run['run']:22s} {run['taxonomy']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--episodes-root",
                    default="data/datasets/objectnav/hm3d/v1/val/content")
    args = ap.parse_args()
    geom = load_goal_geometry(args.episodes_root)
    out = [analyse(r, geom) for r in args.runs]
    show(out[0], out[1] if len(out) > 1 else None)
    for o in out:
        (o_path := args.runs[out.index(o)] / "behaviour_summary.json").write_text(
            json.dumps(o, indent=2))
        print(f"\nwrote {o_path}")
