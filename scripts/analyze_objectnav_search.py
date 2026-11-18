#!/usr/bin/env python
"""Where an ObjectNav run loses its episodes, and whether a second run moved them.

    python scripts/analyze_objectnav_search.py outputs/RUN
    python scripts/analyze_objectnav_search.py outputs/BASE outputs/NEW     # paired
    python scripts/analyze_objectnav_search.py outputs/RUN --episodes       # one line each

Reads `episodes.jsonl` (the run needs `eval.behaviour_log=true`: every number
here comes from the per-step trace) and the dataset's goal geometry, which the
agent never sees. Built for the search additions on top of ASCENT
(`src/navigation/search/`, docs/history/AB_RESULTS.md S83+), whose claims are
about WHERE an episode fails, so each episode gets exactly one class:

  success
  xfloor: no stairs       cross-floor, never on the goal's storey, and no stair
                          toward it was ever on the map
  xfloor: no climb        ... a stair toward it was mapped, no climb was started
  xfloor: climb failed    ... climbs were started, none ended on the goal's storey
  saw -> wrong STOP       a true instance was in view at some step; the agent
                          STOPped more than NEAR_MISS_M from every viewpoint
  saw -> near miss        ... STOPped within NEAR_MISS_M of one, and still failed
  saw -> timeout          ... and ran out of steps
  never saw -> STOP / timeout

"In view" is the S71 rule (a true instance within 3 m and +-40 deg of the
heading) with one refinement the S71 traces could not make: the trace carries
the camera height, so an instance counts only while the agent stands on its
storey. The S71 rule itself is printed beside it so the two can be compared.

With two runs the episodes are paired by (scene, episode id): the outcome
table with an exact McNemar test, the class each flipped episode moved between,
and every trace metric as a paired difference with a sign test. At n = 100 SR
moves +-3 points on noise, so read the trace metrics, not the SR line.

    python scripts/analyze_objectnav_search.py outputs/RUN --storey

replays the storey belief (`src/navigation/search/storey.py`) over the run's
own trace (which needs `agent.search.trace=true`): at every step the agent was
choosing a frontier, would the belief have asked for another storey, and was
that toward the goal's? The run did not follow the request, so only the FIRST
one in an episode means anything -- it is the offline gate a switch rule has
to pass before it is run. Its two measured inputs are printed with it: the
area one live frontier turned out to promise (`a_f`), and the steps a storey
change needs to be worth starting (a climb that changed storey, plus getting
from a first sighting to STOP).

    python scripts/analyze_objectnav_search.py outputs/RUN --diagnose

answers, from the same trace, the questions that decide whether each addition
is worth building: did RedNet alone see the staircases the fused mask never
mapped; how the climbs that did not end on the goal's storey ended; in the
episodes that had the target in view and failed, whether the 0.8 detector
fired on it, whether a sub-threshold box landed on it, whether the agent
committed to it and whether it then gave it up; and in the ones that never had
it in view, whether the agent was stuck or went the other way.

    python scripts/analyze_objectnav_search.py outputs/RUN --candidates v0=outputs/RUN/replay_v0

asks whether the OSG object layer held the target where ASCENT's 0.8 detector
did not (the inspection addition, M3). The directory is an object-layer replay
of the run's recording made with `+replay.target_log=true`
(scripts/replay_object_layer.py): every track of the target's label, per
keyframe. A track within AT_INSTANCE_M of a true instance is TRUE. Printed: how
well each of the layer's scores ranks true above false (AUC), and for a sweep
of thresholds on the best of them, how many false places an episode would be
sent to inspect against how many failed episodes held a true track the
baseline never committed to. Repeat the flag to compare ellipsoid variants.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_behaviour import in_frame, load_goal_geometry_3d  # noqa: E402
from navigation.search.storey import (FloorSearchBelief, TraceStoreys, floor_table,  # noqa: E402
                                      unexplored_beyond)

CAMERA_HEIGHT_M = 0.88      # the trace logs the camera's height; the base is this far below
SAME_FLOOR_M = 0.5          # eval/floors.py: |base - viewpoint height| on one storey
SETTLE_STEPS = 5            # consecutive steps at a height before it counts as "on" that storey
SAW_RANGE_M = 3.0           # the S71 trace metric
SAW_HALF_DEG = 40.0
OBJ_BELOW_M, OBJ_ABOVE_M = 0.5, 2.5   # an instance's centre, relative to the base, on this storey
NEAR_MISS_M = 1.0           # a failed STOP this close (geodesic, to a viewpoint) had the right object
TIMEOUT_STEP = 499
INIT_TURNS = 13             # `AscentNavAgent._initialize`: every storey opens with 13 turns
CHANGED_STOREY_M = 1.5      # a climb that moved the base this far changed storey (the S71 rule's gap)
AT_INSTANCE_M = 1.5         # a point this close to a true instance is "on" it (analyze_trackloc.py)
MIN_STAIR_PIXELS = 20       # osg/perception/stair_seg.py: RedNet's stair class counts above this
STUCK_M = 0.05              # a forward step that moved less than this did not move

CLASSES = (
    "success",
    "xfloor: no stairs", "xfloor: no climb", "xfloor: climb failed",
    "saw -> wrong STOP", "saw -> near miss", "saw -> timeout",
    "never saw -> STOP", "never saw -> timeout",
)


# ------------------------------------------------------------------ loading

def load_rows(run: Path) -> list:
    path = run / "episodes.jsonl"
    if not path.exists():
        raise SystemExit(f"no episodes.jsonl in {run}")
    return [json.loads(line) for line in path.open() if line.strip()]


def key(row: dict) -> tuple:
    return (str(row.get("scene")), str(row.get("episode_id")))


def load_viewpoint_heights(episodes_root: str) -> dict:
    """{uid: sorted heights of every goal viewpoint}. A viewpoint is a place to
    stand, so its height IS a storey's floor height; an object's is not."""
    out = {}
    for f in sorted(glob.glob(f"{episodes_root}/*.json.gz")):
        d = json.load(gzip.open(f))
        gbc = d.get("goals_by_category", {})
        for i, e in enumerate(d["episodes"]):
            scene = e["scene_id"].split("/")[-1]
            ys = [vp["agent_state"]["position"][1]
                  for g in gbc.get(f"{scene}_{e['object_category']}", [])
                  for vp in g.get("view_points", [])]
            if ys:
                out[f"{scene}:{i}"] = np.unique(np.round(np.asarray(ys, float), 2))
    return out


# ------------------------------------------------------- one episode's facts

def _runs(mask: np.ndarray) -> list:
    """[(start, end_exclusive)] of every run of True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(d[0::2].tolist(), d[1::2].tolist()))


def _settled(mask: np.ndarray, n: int = SETTLE_STEPS) -> int:
    """First step of the first run of True at least `n` long; -1 if none."""
    for a, b in _runs(mask):
        if b - a >= n:
            return a
    return -1


def visible(xy, yaw, base_y, objs) -> np.ndarray:
    """(T, M): instance m in view at step t -- in range, inside the cone, and on
    the storey the agent is standing on."""
    rel = objs[None, :, :2] - xy[:, None, :]
    dist = np.linalg.norm(rel, axis=2)
    ang = (np.arctan2(rel[..., 1], rel[..., 0]) - yaw[:, None] + np.pi) % (2 * np.pi) - np.pi
    dy = objs[None, :, 2] - base_y[:, None]
    return ((dist < SAW_RANGE_M) & (np.abs(ang) < np.radians(SAW_HALF_DEG))
            & (dy > -OBJ_BELOW_M) & (dy < OBJ_ABOVE_M))


def believed_switches(trace: list, base_y: np.ndarray) -> list:
    """Every time the agent's own storey changed (`floor_uid`, so the run needs
    `agent.search.trace`), and what its height did: the median base height it
    stood at on the storey it left against the one it entered, climb steps
    left out. ASCENT calls a climb done when the agent walks off the stair map
    having been at its centroid (`stairs.py:process_climb_state`), which does
    not say it went anywhere."""
    if not trace or "floor_uid" not in trace[0]:
        return []
    uid = np.array([int(t["floor_uid"]) for t in trace])
    level = np.array([not int(t.get("climb") or 0) and t.get("state") != "climb" for t in trace])
    cuts = [0] + (np.flatnonzero(uid[1:] != uid[:-1]) + 1).tolist() + [len(trace)]
    height = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        ys = base_y[a:b][level[a:b]]
        height.append(float(np.median(ys if len(ys) else base_y[a:b])))
    return [{"step": int(cuts[k + 1]), "dy": round(height[k + 1] - height[k], 2),
             "real": bool(abs(height[k + 1] - height[k]) > SAME_FLOOR_M)}
            for k in range(len(height) - 1)]


def climb_attempts(trace: list, base_y: np.ndarray, vp_ys: np.ndarray) -> list:
    """Every run of climb-mode steps: its start, length, direction (the trace's
    1 = up, 2 = down) and where it left the agent."""
    flag = np.array([int(t.get("climb") or 0) for t in trace])
    on = (flag != 0) | np.array([t.get("state") == "climb" for t in trace])
    out = []
    for a, b in _runs(on):
        end_y = float(base_y[min(b, len(base_y) - 1)])
        out.append({
            "start": int(a), "steps": int(b - a),
            "dir": "up" if flag[a:b].max() == 1 else "down",
            "dy": round(end_y - float(base_y[a]), 2),
            "to_goal_storey": bool(np.abs(vp_ys - end_y).min() <= SAME_FLOOR_M),
        })
    return out


def features(row: dict, objs, vp_ys) -> dict | None:
    """Everything the tables below are built from; None when the row has no trace
    or the dataset has no geometry for it."""
    trace = row.get("step_trace") or []
    if not trace or objs is None or vp_ys is None or not len(objs):
        return None
    xy = np.array([t["xy"] for t in trace], float)
    yaw = np.array([float(t["yaw"]) for t in trace])
    base_y = np.array([float(t["h"]) for t in trace]) - CAMERA_HEIGHT_M
    start_y = float(row.get("start_y", base_y[0]))
    success = bool(row["success"])
    stopped = int(row["steps"]) < TIMEOUT_STEP
    cross = row.get("floor_class") not in (None, "same_floor")

    on_goal_storey = np.abs(base_y[:, None] - vp_ys[None, :]).min(axis=1) <= SAME_FLOOR_M
    reach_step = _settled(on_goal_storey)
    goal_dy = float(vp_ys[np.abs(vp_ys - start_y).argmin()] - start_y)
    on_start = np.abs(base_y - start_y) <= SAME_FLOOR_M
    px = np.array([int(t.get("up_px" if goal_dy > 0 else "dn_px") or 0) for t in trace])
    stair_toward_goal = bool((px[on_start] > 0).any())
    climbs = climb_attempts(trace, base_y, vp_ys)
    climb_steps = int(sum(c["steps"] for c in climbs))

    vis = visible(xy, yaw, base_y, objs)
    saw_steps = np.flatnonzero(vis.any(axis=1))
    saw = bool(len(saw_steps))
    # The S71 rule, for comparison with docs/history/AB_RESULTS.md S71: storey
    # by the START height on same-floor episodes, no height gate on cross-floor.
    o71 = objs if cross else objs[np.abs(objs[:, 2] - start_y) < 1.5]
    saw_s71 = bool(len(o71)) and bool(in_frame(xy, yaw, o71[:, :2], SAW_RANGE_M, SAW_HALF_DEG)[0].any())

    try_nav = np.array([bool(t.get("try_nav")) for t in trace])
    d_obj = np.linalg.norm(objs[None, :, :2] - xy[:, None, :], axis=2)
    dy = objs[None, :, 2] - base_y[:, None]
    d_obj = np.where((dy > -OBJ_BELOW_M) & (dy < OBJ_ABOVE_M), d_obj, np.inf)

    if success:
        cls = "success"
    elif cross and reach_step < 0:
        cls = ("xfloor: climb failed" if climbs
               else "xfloor: no climb" if stair_toward_goal else "xfloor: no stairs")
    elif not stopped:
        cls = f"{'saw' if saw else 'never saw'} -> timeout"
    elif not saw:
        cls = "never saw -> STOP"
    else:
        cls = ("saw -> near miss" if float(row["distance_to_goal"]) <= NEAR_MISS_M
               else "saw -> wrong STOP")
    # Where the STOP came from. Only a commit is a claim about the target: the
    # agent also STOPs when a storey runs out (`explored_out`), and it passes on
    # a PointNav STOP raised while walking to a staircase (`agent.py:
    # _navigate_stair_if_unexplored_floor`, the reference's `:847`).
    last = trace[-1]
    stop_from = (None if not stopped
                 else "commit" if last.get("try_nav")
                 else "explored_out" if row.get("approach_stop_reason") == "explored_out"
                 else str(last.get("state")))
    return {
        "key": key(row), "target": row.get("target"), "class": cls, "stop_from": stop_from,
        "success": success, "stopped": stopped, "cross": cross,
        "steps": len(trace), "spl": float(row.get("spl") or 0.0),
        "dtg": float(row["distance_to_goal"]),
        "reached_goal_storey": reach_step >= 0, "reach_step": reach_step,
        "stair_toward_goal": stair_toward_goal,
        "climbs": climbs, "climb_steps": climb_steps,
        "switches": believed_switches(trace, base_y),
        "saw": saw, "saw_s71": saw_s71,
        "first_saw": int(saw_steps[0]) if saw else -1, "saw_steps": int(len(saw_steps)),
        "committed": bool(try_nav.any()),
        "first_commit": int(np.flatnonzero(try_nav)[0]) if try_nav.any() else -1,
        "min_obj_m": float(d_obj.min()) if np.isfinite(d_obj).any() else None,
        "giveups": len(row.get("giveup_log") or []),
        "path_m": float((row.get("behaviour") or {}).get("path_len_m") or 0.0),
    }


def analyse(run: Path, geo3: dict, vps: dict) -> list:
    out, skipped = [], 0
    for r in load_rows(run):
        uid = f"{r.get('scene')}:{r.get('episode_id')}"
        f = features(r, geo3.get(uid), vps.get(uid))
        if f is None:
            skipped += 1
        else:
            out.append(f)
    if skipped:
        print(f"  ({run.name}: {skipped} episodes skipped -- no step_trace or no goal geometry; "
              f"was the run made with eval.behaviour_log=true?)")
    if not out:
        raise SystemExit(f"{run}: nothing to analyse")
    return out


# ------------------------------------------------------------------- metrics

def metrics(fs: list) -> dict:
    """Run-level numbers. Counts are counts of episodes unless the name says %."""
    n = len(fs)
    same = [f for f in fs if not f["cross"]]
    cross = [f for f in fs if f["cross"]]
    saw = [f for f in fs if f["saw"]]
    com = [f for f in fs if f["committed"]]
    steps = sum(f["steps"] for f in fs)
    attempts = [c for f in fs for c in f["climbs"]]
    first_saw = [f["first_saw"] for f in saw]
    return {
        "episodes": n,
        "success": sum(f["success"] for f in fs),
        "SPL": round(float(np.mean([f["spl"] for f in fs])), 3),
        "same-floor success": sum(f["success"] for f in same),
        "same-floor episodes": len(same),
        "cross-floor success": sum(f["success"] for f in cross),
        "cross-floor episodes": len(cross),
        "cross-floor: reached goal storey": sum(f["reached_goal_storey"] for f in cross),
        "cross-floor: success | reached": sum(f["success"] for f in cross if f["reached_goal_storey"]),
        "SAW episodes": len(saw),
        "SAW episodes (S71 rule)": sum(f["saw_s71"] for f in fs),
        "conversion given SAW": round(sum(f["success"] for f in saw) / max(len(saw), 1), 3),
        "median first-SAW step": float(np.median(first_saw)) if first_saw else None,
        "committed episodes": len(com),
        "P(success | committed)": round(sum(f["success"] for f in com) / max(len(com), 1), 3),
        "give-ups": sum(f["giveups"] for f in fs),
        "climb attempts": len(attempts),
        "climb attempts ending on goal storey": sum(c["to_goal_storey"] for c in attempts),
        "storey changes the agent believed": sum(len(f["switches"]) for f in fs),
        "... that moved it off its storey": sum(sw["real"] for f in fs for sw in f["switches"]),
        "episodes with a storey change in belief only": sum(
            any(not sw["real"] for sw in f["switches"]) for f in fs),
        "climb-mode steps %": round(100 * sum(f["climb_steps"] for f in fs) / max(steps, 1), 1),
        "climb-mode steps, same-floor %": round(
            100 * sum(f["climb_steps"] for f in same) / max(sum(f["steps"] for f in same), 1), 1),
        "mean steps": round(steps / n, 1),
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar (a sign test on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)


# ------------------------------------------------------------------ printing

def show_run(name: str, fs: list) -> None:
    m = metrics(fs)
    print(f"\n== {name}: {m['success']}/{m['episodes']} "
          f"({100 * m['success'] / m['episodes']:.1f}%), SPL {m['SPL']:.3f}")
    tab = Counter(f["class"] for f in fs)
    print("\n  class                      n   same  cross")
    for c in CLASSES:
        if tab[c]:
            s = sum(f["class"] == c and not f["cross"] for f in fs)
            print(f"  {c:24s} {tab[c]:3d}   {s:4d}  {tab[c] - s:5d}")
    origin = Counter((f["stop_from"], f["success"]) for f in fs if f["stopped"])
    print("\n  STOP raised from           n   succeeded")
    for o in sorted({k for k, _ in origin}, key=str):
        print(f"  {str(o):24s} {origin[(o, True)] + origin[(o, False)]:3d}   {origin[(o, True)]:9d}")
    print()
    for k, v in m.items():
        if k not in ("episodes", "success", "SPL"):
            print(f"  {k:40s} {'-' if v is None else v}")


def show_episodes(fs: list) -> None:
    print("\n  scene        ep  target      class                 steps  dtg   saw@  commit@  stop from    climbs")
    for f in sorted(fs, key=lambda f: f["key"]):
        climbs = " ".join(f"{c['dir'][0]}{c['start']}+{c['steps']}{'*' if c['to_goal_storey'] else ''}"
                          for c in f["climbs"])
        print(f"  {f['key'][0][:11]:11s} {f['key'][1]:>3s}  {str(f['target'])[:10]:10s}  "
              f"{f['class']:20s}  {f['steps']:4d} {f['dtg']:5.1f}  {f['first_saw']:4d}  "
              f"{f['first_commit']:6d}  {str(f['stop_from'] or '-'):12s} {climbs}")


def show_paired(na: str, a: list, nb: str, b: list) -> dict:
    by_a = {f["key"]: f for f in a}
    by_b = {f["key"]: f for f in b}
    shared = sorted(set(by_a) & set(by_b))
    if not shared:
        raise SystemExit("no episodes in common -- are these the same split?")
    pa = [by_a[k] for k in shared]
    pb = [by_b[k] for k in shared]
    both = sum(x["success"] and y["success"] for x, y in zip(pa, pb))
    only_a = [k for k, x, y in zip(shared, pa, pb) if x["success"] and not y["success"]]
    only_b = [k for k, x, y in zip(shared, pa, pb) if y["success"] and not x["success"]]
    p = mcnemar_exact(len(only_a), len(only_b))
    print(f"\n== paired: {na} vs {nb}, {len(shared)} episodes")
    print(f"  both {both} | {na} only {len(only_a)} | {nb} only {len(only_b)} | "
          f"neither {len(shared) - both - len(only_a) - len(only_b)}    McNemar exact p = {p:.3f}")

    ma, mb = metrics(pa), metrics(pb)
    print(f"\n  {'metric':40s} {na[:14]:>14s} {nb[:14]:>14s} {'delta':>8s}")
    for k in ma:
        va, vb = ma[k], mb[k]
        d = "" if va is None or vb is None else f"{vb - va:+.3g}"
        print(f"  {k:40s} {str(va):>14s} {str(vb):>14s} {d:>8s}")

    # Per-episode differences with a sign test: which way did each episode move?
    print(f"\n  {'per-episode':40s} {'better':>7s} {'worse':>7s} {'sign p':>8s}")
    for label, f, lower_is_better in (
        ("reached goal storey (cross-floor)", lambda x: float(x["reached_goal_storey"]) if x["cross"] else None, False),
        ("saw the target", lambda x: float(x["saw"]), False),
        ("first-SAW step (both saw)", lambda x: float(x["first_saw"]) if x["saw"] else None, True),
        ("climb-mode steps (same-floor)", lambda x: None if x["cross"] else float(x["climb_steps"]), True),
        ("steps (both succeeded)", lambda x: float(x["steps"]) if x["success"] else None, True),
    ):
        better = worse = 0
        for x, y in zip(pa, pb):
            u, v = f(x), f(y)
            if u is None or v is None or u == v:
                continue
            if (v < u) == lower_is_better:
                better += 1
            else:
                worse += 1
        print(f"  {label:40s} {better:7d} {worse:7d} {mcnemar_exact(better, worse):8.3f}")

    moves = Counter((x["class"], y["class"]) for x, y in zip(pa, pb) if x["class"] != y["class"])
    if moves:
        print("\n  class changes (episodes)")
        for (ca, cb), n in moves.most_common():
            print(f"    {n:3d}  {ca:22s} -> {cb}")
    for title, ks in ((f"{nb} wins", only_b), (f"{na} wins", only_a)):
        if ks:
            print(f"\n  {title}:")
            for k in ks:
                print(f"    {k[0][:11]:11s} ep{k[1]:>3s} {str(by_a[k]['target'])[:10]:10s} "
                      f"{by_a[k]['class']:22s} -> {by_b[k]['class']}")
    return {"n_paired": len(shared), "both": both, "only_a": [list(k) for k in only_a],
            "only_b": [list(k) for k in only_b], "mcnemar_p": p, "a": ma, "b": mb}


# ------------------------------------------------- the storey belief, replayed

def choosing_frontier(t: dict) -> bool:
    """`_explore` got as far as `get_best_frontier` on this step: exploring,
    past the opening turns, not on a staircase, a live frontier to pick."""
    return (t.get("state") == "explore" and int(t.get("nfront_live") or 0) > 0
            and int(t.get("floor_n") or 0) >= INIT_TURNS and not int(t.get("climb") or 0))


def frontier_area(rows: list) -> dict:
    """`a_f`: what one live frontier promised, fitted to the storeys the run
    finished. On a storey that ended exhausted with explored area A_end the
    true searched fraction at a step is A / A_end; `a_f` is the value whose
    A / (A + a_f F) is closest to it (mean absolute error over the storey's
    frontier-choosing steps, every storey weighing the same). A storey the
    episode left unfinished only bounds it from below, so those are used when
    fewer than 10 were finished, and the output says which it was."""
    per = {"exhausted": [], "unfinished": []}
    for r in rows:
        by = {}
        for t in r.get("step_trace") or []:
            if "floor_uid" in t:
                by.setdefault(t["floor_uid"], []).append(t)
        for ts in by.values():
            a_end = max(float(t.get("explored_m2") or 0.0) for t in ts)
            pts = np.array([(float(t["explored_m2"]), int(t["nfront_live"]))
                            for t in ts if choosing_frontier(t)], float)
            if len(pts) and a_end > 0:
                done = any(int(t.get("floor_done") or 0) for t in ts)
                per["exhausted" if done else "unfinished"].append((pts, a_end))
    use = per["exhausted"] if len(per["exhausted"]) >= 10 else per["exhausted"] + per["unfinished"]
    out = {"a_f": None, "from": "exhausted storeys" if len(per["exhausted"]) >= 10 else "all storeys",
           "n_exhausted": len(per["exhausted"]), "n_unfinished": len(per["unfinished"])}
    if not use:
        return out

    def err(a_f: float) -> float:
        return float(np.mean([np.abs(p[:, 0] / (p[:, 0] + a_f * p[:, 1]) - p[:, 0] / a_end).mean()
                              for p, a_end in use]))

    grid = np.arange(0.5, 80.01, 0.5)
    errs = [err(a) for a in grid]
    best = int(np.argmin(errs))
    # The comparison a frontier count has to beat: every storey is the median
    # storey, and the searched fraction is A over that.
    typical = float(np.median([a_end for _, a_end in use]))
    flat = float(np.mean([np.abs(np.minimum(p[:, 0] / typical, 1.0) - p[:, 0] / a_end).mean()
                          for p, a_end in use]))
    out.update({
        "a_f": float(grid[best]), "coverage_mae": round(errs[best], 3),
        "coverage_mae_fixed_area": round(flat, 3),
        "median_storey_m2": round(float(np.median([a_end for _, a_end in use])), 1),
        "median_ratio": round(float(np.median(
            [np.median((a_end - p[:, 0]) / p[:, 1]) for p, a_end in use])), 1),
    })
    return out


def steps_needed(fs: list) -> dict:
    """What has to be left of the episode for leaving a storey to pay: a climb
    that changed storey, and the steps from a first sighting to STOP."""
    climbs = [c["steps"] for f in fs for c in f["climbs"] if abs(c["dy"]) >= CHANGED_STOREY_M]
    convert = [f["steps"] - f["first_saw"] for f in fs if f["success"] and f["first_saw"] >= 0]
    t_climb = float(np.median(climbs)) if climbs else None
    t_conv = float(np.median(convert)) if convert else None
    return {"climb": t_climb, "n_climbs": len(climbs), "convert": t_conv, "n_convert": len(convert),
            "min_left": int(round((t_climb or 0.0) + (t_conv or 0.0)))}


def replay_storey(row: dict, vp_ys: np.ndarray, prior: dict, par: dict) -> dict | None:
    """The first step the belief would have asked to leave, and every later one
    counted. None when the trace carries no storey fields."""
    trace = row.get("step_trace") or []
    if not trace or "floor_uid" not in trace[0]:
        return None
    belief = FloorSearchBelief(switch_ratio=par["ratio"], detect_prob=par["d"],
                               prior_weight=par["w"], frontier_area_m2=par["a_f"])
    floors = TraceStoreys()
    first, asks = None, 0
    for i, t in enumerate(trace):
        here, storeys = floors.update(t)
        if not choosing_frontier(t):
            continue
        enough = (TIMEOUT_STEP + 1 - i) >= par["min_left"]
        req = belief.decide(
            i, here, storeys, floor_table(prior, row.get("target"), len(storeys)),
            can_up=enough and bool(t.get("has_up")) and unexplored_beyond(storeys, here, "up"),
            can_down=enough and bool(t.get("has_dn")) and unexplored_beyond(storeys, here, "down"),
            here_id=int(t["floor_uid"]))
        if req is None:
            continue
        asks += 1
        if first is None:
            base = float(t["h"]) - CAMERA_HEIGHT_M
            gap = float(vp_ys[np.abs(vp_ys - base).argmin()] - base)
            post = belief.log[-1]
            first = {"step": i, "dir": req,
                     "toward_goal": abs(gap) > SAME_FLOOR_M and (gap > 0) == (req == "up"),
                     "on_goal_storey": abs(gap) <= SAME_FLOOR_M,
                     "P": post["P"], "cover": post["cover"], "here": here}
    return {"key": key(row), "first": first, "asks": asks, "n_storeys": len(floors.storeys)}


def show_storey(run: Path, fs: list, vps: dict, args) -> dict:
    rows = load_rows(run)
    area, need = frontier_area(rows), steps_needed(fs)
    par = {"ratio": args.storey_ratio, "d": args.storey_detect, "w": args.storey_prior_weight,
           "a_f": args.storey_af if args.storey_af > 0 else area["a_f"],
           "min_left": args.storey_min_left if args.storey_min_left >= 0 else need["min_left"]}
    print(f"\n== storey belief replayed on {run.name}")
    print(f"  a_f: {area['a_f']} m2 per live frontier, fitted on {area['from']} "
          f"(finished {area['n_exhausted']}, unfinished {area['n_unfinished']}; coverage error "
          f"{area.get('coverage_mae')} against {area.get('coverage_mae_fixed_area')} for a fixed "
          f"storey area, median storey {area.get('median_storey_m2')} m2, "
          f"median (A_end - A) / F {area.get('median_ratio')})")
    print(f"  steps needed: climb that changed storey {need['climb']} (n {need['n_climbs']}) "
          f"+ first sighting to STOP {need['convert']} (n {need['n_convert']}) = {need['min_left']}")
    if par["a_f"] is None:
        print("  no frontier-choosing step carries the storey fields: was the run made with "
              "agent.search.trace=true?")
        return {}
    print(f"  rule: ask when P[other] >= {par['ratio']} x P[here]; d {par['d']}, prior weight "
          f"{par['w']}, a_f {par['a_f']}, >= {par['min_left']} steps left")
    try:
        prior = json.loads(Path(args.floor_prior).read_text())
    except OSError:
        prior = {}
        print(f"  ({args.floor_prior} missing: uniform prior everywhere)")
    by_key = {f["key"]: f for f in fs}
    res = []
    for r in rows:
        f = by_key.get(key(r))
        rep = replay_storey(r, vps.get(f"{r.get('scene')}:{r.get('episode_id')}"), prior, par) if f else None
        if rep is not None:
            res.append((f, rep))
    print("\n  class                      n   asked  toward  away   first ask (median step)")
    for c in CLASSES:
        sel = [(f, rep) for f, rep in res if f["class"] == c]
        if not sel:
            continue
        asked = [rep["first"] for _, rep in sel if rep["first"]]
        toward = sum(a["toward_goal"] for a in asked)
        med = f"{np.median([a['step'] for a in asked]):.0f}" if asked else "-"
        print(f"  {c:24s} {len(sel):3d}   {len(asked):5d}  {toward:6d}  {len(asked) - toward:4d}   {med}")

    def before_sighting(f, rep) -> bool:
        a = rep["first"]
        ref = f["first_saw"] if f["first_saw"] >= 0 else f["first_commit"]
        return a is not None and (ref < 0 or a["step"] < ref)

    noclimb = [(f, rep) for f, rep in res if f["class"] == "xfloor: no climb"]
    same_ok = [(f, rep) for f, rep in res if f["success"] and not f["cross"]]
    g1 = sum(bool(rep["first"] and rep["first"]["toward_goal"]) for _, rep in noclimb)
    g2 = sum(before_sighting(f, rep) for f, rep in same_ok)
    cross = [(f, rep) for f, rep in res if f["cross"] and not f["reached_goal_storey"]]
    g3 = sum(bool(rep["first"] and rep["first"]["toward_goal"]) for _, rep in cross)
    print(f"\n  stairs mapped, never climbed: asked toward the goal's storey   {g1} of {len(noclimb)}")
    print(f"  cross-floor, never reached the goal's storey: asked toward it  {g3} of {len(cross)}")
    print(f"  same-floor successes asked to leave before the first sighting  {g2} of {len(same_ok)}")
    # The stricter reading. The rule only speaks on a frontier-choosing step,
    # so ANY request in an episode the baseline won on its own storey is a
    # departure the baseline did not make -- "in view" is geometry, and the
    # baseline had often not detected what it was looking at yet.
    g4 = sum(bool(rep["first"] and rep["first"]["on_goal_storey"]) for _, rep in same_ok)
    print(f"  same-floor successes asked to leave the goal's storey at all   {g4} of {len(same_ok)}")
    if args.episodes:
        print("\n  scene        ep  target      class                 ask@  dir   toward  saw@  P / cover")
        for f, rep in sorted(res, key=lambda x: x[0]["key"]):
            a = rep["first"]
            if a is None and not f["cross"]:
                continue
            tail = (f"{a['step']:4d}  {a['dir']:4s}  {'yes' if a['toward_goal'] else 'no ':3s}   "
                    f"{f['first_saw']:4d}  {a['P']} / {a['cover']}") if a else "   -"
            print(f"  {f['key'][0][:11]:11s} {f['key'][1]:>3s}  {str(f['target'])[:10]:10s}  "
                  f"{f['class']:20s}  {tail}")
    return {"params": par, "frontier_area": area, "steps_needed": need,
            "no_climb_asked_toward": [g1, len(noclimb)], "cross_unreached_asked_toward": [g3, len(cross)],
            "same_floor_success_asked_before_sighting": [g2, len(same_ok)],
            "same_floor_success_asked_at_all": [g4, len(same_ok)],
            "episodes": [{"key": rep["key"], "class": f["class"], "first": rep["first"],
                          "asks": rep["asks"], "n_storeys": rep["n_storeys"]} for f, rep in res]}


# ------------------------------------------- what each addition would act on

def episodic_to_world(trace: list):
    """The rigid map from the trace's episodic plane (`xy_ep`, `sel`) to its
    world plane (`xy`), fitted to the trajectory; None when it cannot be."""
    pairs = [(t["xy_ep"], t["xy"]) for t in trace if t.get("xy_ep") is not None]
    if len(pairs) < 3:
        return None
    ep, w = (np.asarray(v, float) for v in zip(*pairs))
    a, b = ep - ep.mean(0), w - w.mean(0)
    if np.linalg.norm(a) < 0.5:               # the agent never left its start
        return None
    u, _, vt = np.linalg.svd(a.T @ b)
    rot = (u @ vt).T
    if np.linalg.det(rot) < 0:
        return None
    return lambda q: (np.asarray(q, float) - ep.mean(0)) @ rot.T + w.mean(0)


def facts(row: dict, objs: np.ndarray, vp_ys: np.ndarray) -> dict:
    """One episode's answers to the questions in the module docstring."""
    trace = row["step_trace"]
    xy = np.array([t["xy"] for t in trace], float)
    yaw = np.array([float(t["yaw"]) for t in trace])
    base_y = np.array([float(t["h"]) for t in trace]) - CAMERA_HEIGHT_M
    start_y = float(row.get("start_y", base_y[0]))
    on_start = np.abs(base_y - start_y) <= SAME_FLOOR_M
    step_of = {int(t["n"]): i for i, t in enumerate(trace)}

    def at_instance(point, i) -> bool:
        dy = objs[:, 2] - base_y[i]
        near = np.linalg.norm(objs[:, :2] - np.asarray(point, float)[:2], axis=1) <= AT_INSTANCE_M
        return bool((near & (dy > -OBJ_BELOW_M) & (dy < OBJ_ABOVE_M)).any())

    seg = np.array([int(t.get("seg_px") or 0) > MIN_STAIR_PIXELS for t in trace])
    det = np.array([int(t.get("stair_det_px") or 0) > 0 for t in trace])
    in_view = visible(xy, yaw, base_y, objs).any(axis=1)
    fired = np.array([float(t.get("det") or 0.0) for t in trace])

    weak_true, weak_best, weak_else = 0, 0.0, 0
    for i, t in enumerate(trace):
        for w in t.get("weak") or []:
            if len(w) < 3:
                continue
            if at_instance(w[1:3], i):
                weak_true += 1
                weak_best = max(weak_best, float(w[0]))
            else:
                weak_else += 1

    goal = [(i, t["goal_xy"]) for i, t in enumerate(trace) if t.get("try_nav") and t.get("goal_xy")]
    commit_true = sum(at_instance(g, i) for i, g in goal)
    # A give-up is logged at the step it happens; what was given up is the
    # goal the agent was walking to on the step before.
    gave_true = gave_else = 0
    for step, _why, _xy in row.get("giveup_log") or []:
        i = step_of.get(int(step))
        before = [g for j, g in goal if i is not None and j < i]
        if not before:
            continue
        if at_instance(before[-1], i):
            gave_true += 1
        else:
            gave_else += 1

    # Did the frontier the agent chose lie nearer the target than the agent
    # stood? The trace holds the chosen frontier only, not the ones passed over.
    to_world = episodic_to_world(trace)
    toward = chose = 0
    if to_world is not None:
        for i, t in enumerate(trace):
            if not choosing_frontier(t) or t.get("sel") is None:
                continue
            dy = objs[:, 2] - base_y[i]
            here = objs[(dy > -OBJ_BELOW_M) & (dy < OBJ_ABOVE_M), :2]
            if not len(here):
                continue
            chose += 1
            sel = to_world(t["sel"])
            toward += bool(np.linalg.norm(here - sel, axis=1).min()
                           < np.linalg.norm(here - xy[i], axis=1).min())
    fwd = np.array([t.get("act") == "move_forward" for t in trace])
    moved = np.array([float(t.get("moved") or 0.0) for t in trace])
    return {
        "seg_steps": int((seg & on_start).sum()), "det_steps": int((det & on_start).sum()),
        "fused_steps": int((seg & det & on_start).sum()),
        "in_view_steps": int(in_view.sum()),
        "fired_in_view": int((in_view & (fired > 0)).sum()),
        "best_fired_in_view": round(float(fired[in_view].max()), 2) if in_view.any() else 0.0,
        "weak_true": weak_true, "weak_best": round(weak_best, 2), "weak_else": weak_else,
        "commit_steps": len(goal), "commit_true": int(commit_true),
        "gave_true": gave_true, "gave_else": gave_else,
        "frontier_choices": chose, "toward": toward,
        "stuck": round(float((fwd & (moved < STUCK_M)).sum()) / max(int(fwd.sum()), 1), 2),
        "explored_m2": round(max(float(t.get("explored_m2") or 0.0) for t in trace), 1),
        "storey_done": bool(any(int(t.get("floor_done") or 0) for t in trace)),
    }


def show_diagnose(run: Path, fs: list, geo3: dict, vps: dict) -> dict:
    rows = {key(r): r for r in load_rows(run)}
    ds = []
    for f in fs:
        r = rows[f["key"]]
        uid = f"{r.get('scene')}:{r.get('episode_id')}"
        ds.append((f, facts(r, geo3[uid], vps[uid])))
    name = lambda f: f"{f['key'][0][:11]:11s} {f['key'][1]:>3s}  {str(f['target'])[:10]:10s}"  # noqa: E731
    out = {}
    print(f"\n== what each addition would act on, {run.name}")

    # -- M2a: the staircases that never reached the map
    sel = [(f, d) for f, d in ds if f["class"].startswith("xfloor")]
    print(f"\n  cross-floor, never on the goal's storey ({len(sel)}): frames on the START storey with")
    print("  stairs in RedNet's mask / in the detector's mask / in both (only both reach the map)")
    print("  scene        ep  target      class                 rednet  detector  both")
    for f, d in sel:
        print(f"  {name(f)}  {f['class']:20s}  {d['seg_steps']:6d}  {d['det_steps']:8d}  {d['fused_steps']:4d}")
    nost = [d for f, d in sel if f["class"] == "xfloor: no stairs"]
    out["no_stairs"] = {"n": len(nost), "rednet_saw": sum(d["seg_steps"] > 0 for d in nost),
                        "detector_saw": sum(d["det_steps"] > 0 for d in nost)}
    print(f"  no stairs mapped: {out['no_stairs']['n']} episodes; RedNet had stairs in "
          f"{out['no_stairs']['rednet_saw']}, the detector in {out['no_stairs']['detector_saw']}")

    # -- M2b: how the climbs ended
    att = [(f, c) for f, _ in ds for c in f["climbs"]]
    kinds = Counter("to the goal's storey" if c["to_goal_storey"] and abs(c["dy"]) >= CHANGED_STOREY_M
                    else "to another storey" if abs(c["dy"]) >= CHANGED_STOREY_M
                    else "no storey change" for _, c in att)
    print(f"\n  climb attempts ({len(att)}): outcome, n, median steps")
    for k in ("to the goal's storey", "to another storey", "no storey change"):
        steps = [c["steps"] for _, c in att
                 if k == ("to the goal's storey" if c["to_goal_storey"] and abs(c["dy"]) >= CHANGED_STOREY_M
                          else "to another storey" if abs(c["dy"]) >= CHANGED_STOREY_M
                          else "no storey change")]
        if steps:
            print(f"    {k:22s} {kinds[k]:3d}  {np.median(steps):.0f}")
    rep = []
    for f, _ in ds:
        failed = [c for c in f["climbs"] if abs(c["dy"]) < CHANGED_STOREY_M]
        if len(failed) >= 2:
            rep.append((f, failed))
    print(f"  episodes with 2+ climbs that changed nothing: {len(rep)} "
          f"(cross-floor {sum(f['cross'] for f, _ in rep)}, succeeded {sum(f['success'] for f, _ in rep)})")
    for f, failed in rep:
        after = sum(c["steps"] for c in failed[2:])
        print(f"    {name(f)}  {f['class']:20s}  {len(failed)} attempts, "
              f"{sum(c['steps'] for c in failed)} steps, {after} of them after the second")
    ph = [(f, [sw for sw in f["switches"] if not sw["real"]]) for f, _ in ds]
    ph = [(f, sw) for f, sw in ph if sw]
    n_sw = sum(len(f["switches"]) for f, _ in ds)
    print(f"  the agent changed storey in its own books {n_sw} times; {sum(len(sw) for _, sw in ph)} of "
          f"them left it within {SAME_FLOOR_M} m of the height it started at, in {len(ph)} episodes "
          f"(cross-floor {sum(f['cross'] for f, _ in ph)}, succeeded {sum(f['success'] for f, _ in ph)})")
    for f, sw in ph:
        print(f"    {name(f)}  {f['class']:20s}  " + " ".join(f"@{x['step']} {x['dy']:+.2f} m" for x in sw))
    out["phantom_switches"] = {"switches": n_sw, "phantom": sum(len(sw) for _, sw in ph),
                               "episodes": len(ph), "cross": sum(f["cross"] for f, _ in ph),
                               "succeeded": sum(f["success"] for f, _ in ph)}
    out["climbs"] = {"outcomes": dict(kinds), "repeat_episodes": len(rep),
                     "steps_after_second": int(sum(sum(c["steps"] for c in fl[2:]) for _, fl in rep))}

    # -- M3 / M5: the target was in view and the episode failed
    sel = [(f, d) for f, d in ds if f["class"].startswith("saw")]
    print(f"\n  had the target in view and failed ({len(sel)})")
    print("  scene        ep  target      class                 view  0.8@view  weak@true  "
          "commit true/all  gave up true/else")
    for f, d in sel:
        print(f"  {name(f)}  {f['class']:20s}  {d['in_view_steps']:4d}  "
              f"{d['fired_in_view']:3d} {d['best_fired_in_view']:4.2f}  "
              f"{d['weak_true']:3d} {d['weak_best']:4.2f}   "
              f"{d['commit_true']:5d}/{d['commit_steps']:<5d}    {d['gave_true']:3d}/{d['gave_else']}")
    for c in ("saw -> timeout", "saw -> wrong STOP", "saw -> near miss"):
        g = [d for f, d in sel if f["class"] == c]
        if not g:
            continue
        out[c] = {
            "n": len(g),
            "detector_never_fired_in_view": sum(d["fired_in_view"] == 0 for d in g),
            "weak_box_on_target_only": sum(d["fired_in_view"] == 0 and d["weak_true"] > 0 for d in g),
            "committed_to_target": sum(d["commit_true"] > 0 for d in g),
            "gave_up_target": sum(d["gave_true"] > 0 for d in g),
        }
        o = out[c]
        print(f"  {c}: {o['n']}; the 0.8 detector never fired while in view {o['detector_never_fired_in_view']}"
              f" (a weak box on the target in {o['weak_box_on_target_only']} of those); committed to the "
              f"target {o['committed_to_target']}; gave the target up {o['gave_up_target']}")
    gave = [(f, d) for f, d in ds if not f["success"] and d["gave_true"] > 0]
    out["gave_up_target"] = {"failures": len(gave), "timeouts": sum(not f["stopped"] for f, _ in gave)}
    print(f"  failures that gave up a commit on a true instance: {len(gave)} "
          f"(timeouts {out['gave_up_target']['timeouts']})")
    succ = [d for f, d in ds if f["success"]]
    print(f"  for scale, successes: gave up a true instance in {sum(d['gave_true'] > 0 for d in succ)} "
          f"of {len(succ)}; weak boxes off any instance, per episode, median "
          f"{np.median([d['weak_else'] for _, d in ds]):.0f}")

    # -- M4: the target was never in view
    sel = [(f, d) for f, d in ds if f["class"].startswith("never saw")]
    print(f"\n  never had the target in view, on its storey ({len(sel)})")
    print("  scene        ep  target      class                 steps  path m  stuck  explored  "
          "done  nearest m  chose toward  weak@true")
    for f, d in sel:
        near = "-" if f["min_obj_m"] is None else f"{f['min_obj_m']:.1f}"
        print(f"  {name(f)}  {f['class']:20s}  {f['steps']:5d}  {f['path_m']:6.1f}  {d['stuck']:5.2f}  "
              f"{d['explored_m2']:8.1f}  {'yes' if d['storey_done'] else 'no ':4s}  {near:>9s}  "
              f"{d['toward']:5d}/{d['frontier_choices']:<5d}  {d['weak_true']:5d}")
    out["never_saw"] = {
        "n": len(sel), "stopped": sum(f["stopped"] for f, _ in sel),
        "stuck_over_30pct": sum(d["stuck"] > 0.3 for _, d in sel),
        "storey_done": sum(d["storey_done"] for _, d in sel),
    }
    out["episodes"] = [{"key": f["key"], "class": f["class"], **d} for f, d in ds]
    return out


# ---------------------------------------------- the object layer's candidates

CAND_SCORES = ("evidence", "best_score", "n_obs", "p")
CAND_SWEEP = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
CAND_SITE_M = 1.0           # false tracks this close together are one place to inspect
CAND_MIN_LEFT = 50          # steps an inspection needs left to matter (the M1 gate's number)


def load_candidates(replay: Path, k: tuple, objs: np.ndarray) -> list:
    """The target's tracks over one episode: [{"id", "true", "xy", "first",
    "when": [(step, evidence)], and the maximum of each score}]."""
    path = replay / f"{k[0].split('.')[0]}_ep{k[1]}.targets.jsonl"
    if not path.exists():
        return []
    tracks = {}
    for line in path.open():
        if not line.strip():
            continue
        kf = json.loads(line)
        for t in kf["tracks"]:
            d = tracks.setdefault(t["id"], {"id": t["id"], "first": int(kf["frame_id"]), "when": [],
                                            **{sc: 0.0 for sc in CAND_SCORES}})
            for sc in CAND_SCORES:
                d[sc] = max(d[sc], float(t[sc]))
            d["when"].append((int(kf["frame_id"]), float(t["evidence"])))
            d["centre"] = t["center"]
    out = []
    for d in tracks.values():
        c = np.asarray(d.pop("centre"), float)            # habitat (x, y, z)
        d["xy"] = np.array([c[0], -c[2]])
        near = np.linalg.norm(objs[:, :2] - d["xy"], axis=1) <= AT_INSTANCE_M
        d["true"] = bool((near & (np.abs(objs[:, 2] - c[1]) < CHANGED_STOREY_M)).any())
        out.append(d)
    return out


def reached(track: dict, tau: float) -> int:
    """The first step the track's evidence was at least `tau`; -1 if never."""
    return next((step for step, ev in track["when"] if ev >= tau), -1)


def auc(pos: list, neg: list) -> float | None:
    if not pos or not neg:
        return None
    a, b = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))


def sites(tracks: list) -> int:
    """Distinct places among these tracks (greedy, CAND_SITE_M apart)."""
    kept = []
    for t in tracks:
        if all(np.linalg.norm(t["xy"] - q) > CAND_SITE_M for q in kept):
            kept.append(t["xy"])
    return len(kept)


def show_candidates(run: Path, fs: list, geo3: dict, vps: dict, name: str, replay: Path) -> dict:
    rows = {key(r): r for r in load_rows(run)}
    eps = []
    for f in fs:
        uid = f"{f['key'][0]}:{f['key'][1]}"
        tracks = load_candidates(replay, f["key"], geo3[uid])
        eps.append({"f": f, "tracks": tracks,
                    "commit_true": facts(rows[f["key"]], geo3[uid], vps[uid])["commit_true"]})
    every = [t for e in eps for t in e["tracks"]]
    true = [t for t in every if t["true"]]
    false = [t for t in every if not t["true"]]
    print(f"\n=== {run.name}: the object layer's target tracks, replay '{name}' ({replay}) ===")
    print(f"  {len(every)} tracks in {sum(bool(e['tracks']) for e in eps)} of {len(eps)} episodes: "
          f"{len(true)} within {AT_INSTANCE_M} m of a true instance, {len(false)} not")
    out = {"tracks": len(every), "true": len(true), "auc": {}, "sweep": []}
    print("  score         AUC    median true   median false")
    for sc in CAND_SCORES:
        a = auc([t[sc] for t in true], [t[sc] for t in false])
        out["auc"][sc] = a
        if a is not None:
            print(f"  {sc:11s} {a:6.3f} {np.median([t[sc] for t in true]):12.2f} "
                  f"{np.median([t[sc] for t in false]):13.2f}")

    # A failed episode the layer could have helped: it held a true track, and
    # the baseline never walked to that instance.
    open_ = [e for e in eps if not e["f"]["success"] and not e["commit_true"]
             and any(t["true"] for t in e["tracks"])]
    print(f"\n  failed episodes holding a true track the baseline never committed to: {len(open_)}")
    for e in open_:
        f = e["f"]
        best = max((t for t in e["tracks"] if t["true"]), key=lambda t: t["evidence"])
        print(f"    {f['key'][0].split('.')[0]}:{f['key'][1]:3s} {f['class']:22s} first held at step "
              f"{best['first']:3d} of {f['steps']}, evidence {best['evidence']:.2f}, "
              f"best score {best['best_score']:.2f}, {int(best['n_obs'])} observations")

    print(f"\n  threshold on evidence: false places to inspect per episode, against the failed "
          f"episodes above whose true track reached it with >= {CAND_MIN_LEFT} steps left")
    print("     tau   false places/ep   in successes, before the first commit   failed episodes reached")
    for tau in CAND_SWEEP:
        n_false, n_before, hit = 0, 0, []
        for e in eps:
            f = e["f"]
            fl = [t for t in e["tracks"] if not t["true"] and reached(t, tau) >= 0]
            n_false += sites(fl)
            if f["success"]:
                first = f["first_commit"] if f["first_commit"] >= 0 else f["steps"]
                n_before += sites([t for t in fl if reached(t, tau) < first])
            elif e in open_:
                steps = [reached(t, tau) for t in e["tracks"] if t["true"] and reached(t, tau) >= 0]
                if steps and f["steps"] - min(steps) >= CAND_MIN_LEFT:
                    hit.append(f"{f['key'][0][:4]}:{f['key'][1]}")
        n_succ = max(sum(e["f"]["success"] for e in eps), 1)
        print(f"    {tau:5.2f} {n_false / len(eps):12.2f} {n_before / n_succ:24.2f}"
              f"{'':17s}{len(hit)}  {' '.join(hit)}")
        out["sweep"].append({"tau": tau, "false_per_episode": n_false / len(eps),
                             "false_before_commit_per_success": n_before / n_succ, "reached": hit})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="one run, or BASE NEW for a paired comparison")
    ap.add_argument("--episodes-root", default="data/datasets/objectnav/hm3d/v1/val/content")
    ap.add_argument("--episodes", action="store_true", help="also print one line per episode")
    ap.add_argument("--json", default="", help="also write the numbers here")
    ap.add_argument("--storey", action="store_true",
                    help="replay the storey belief over the (first) run's trace")
    ap.add_argument("--diagnose", action="store_true",
                    help="what the (first) run's failures were missing, per addition")
    ap.add_argument("--candidates", action="append", default=[], metavar="NAME=DIR",
                    help="an object-layer replay of the (first) run made with +replay.target_log=true")
    ap.add_argument("--storey-ratio", type=float, default=1.8, help="exploration.storey_switch_ratio")
    ap.add_argument("--storey-detect", type=float, default=0.85, help="exploration.storey_detect_prob")
    ap.add_argument("--storey-prior-weight", type=float, default=0.8,
                    help="exploration.storey_anchor_weight")
    ap.add_argument("--storey-af", type=float, default=0.0, help="m2 per live frontier; 0 = measure")
    ap.add_argument("--storey-min-left", type=int, default=-1, help="steps left to ask; -1 = measure")
    ap.add_argument("--floor-prior", default="data/priors/hm3d_floor_prior.json")
    args = ap.parse_args()
    if len(args.runs) > 2:
        raise SystemExit("one run, or two to pair")

    geo3 = load_goal_geometry_3d(args.episodes_root)
    vps = load_viewpoint_heights(args.episodes_root)
    if not geo3:
        raise SystemExit(f"no goal geometry under {args.episodes_root} "
                         f"(python scripts/download_data.py --episodes-only --episodes-version v1)")
    out = {}
    per_run = []
    for run in args.runs:
        fs = analyse(run, geo3, vps)
        per_run.append(fs)
        show_run(run.name, fs)
        if args.episodes:
            show_episodes(fs)
        out[run.name] = {"metrics": metrics(fs), "classes": dict(Counter(f["class"] for f in fs)),
                         "episodes": fs}
    if args.storey:
        out["storey"] = show_storey(args.runs[0], per_run[0], vps, args)
    if args.diagnose:
        out["diagnose"] = show_diagnose(args.runs[0], per_run[0], geo3, vps)
    for spec in args.candidates:
        name, _, path = spec.partition("=")
        out.setdefault("candidates", {})[name] = show_candidates(
            args.runs[0], per_run[0], geo3, vps, name, Path(path))
    if len(per_run) == 2:
        out["paired"] = show_paired(args.runs[0].name, per_run[0], args.runs[1].name, per_run[1])
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
