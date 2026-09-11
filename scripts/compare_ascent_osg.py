#!/usr/bin/env python
"""Put a native-ASCENT run and an OSG run side by side, episode for episode.

    python scripts/compare_ascent_osg.py \
        relative_work/ascent/debug/behaviour_100 outputs/s68_behaviour100

The two runs are the SAME 100 episodes (`configs/eval/scenes20_ep0to4.yaml` and
`experiments/eval_ascent_hm3d_100.yaml` select the same set), so every number
here is paired: the interesting cell is not "ASCENT scores X" but "which
episodes does ASCENT win that OSG loses, and what does its trace do there".

`floor_class` and the goal geometry live only on the OSG record; they are the
dataset's, not the agent's, so they are joined onto the ASCENT rows rather than
recomputed.

Category names differ between the two stacks (COCO's `couch`/`tv`/`potted
plant` against HM3D's `sofa`/`tv_monitor`/`plant`); `CANON` folds them.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CANON = {
    "couch": "sofa", "tv": "tv_monitor", "potted plant": "plant",
    "toilet": "toilet", "bed": "bed", "chair": "chair",
}


def canon(name: str | None) -> str:
    n = (name or "").lower().replace("_", " ").strip()
    return CANON.get(n, n.replace(" ", "_"))


def load(path: Path) -> list:
    return [json.loads(line) for line in path.open() if line.strip()]


def key(row: dict) -> tuple:
    return (str(row.get("scene")), str(row.get("episode_id")))


def rate(rows: list, field: str = "success") -> float:
    vals = [float(r.get(field) or 0.0) for r in rows]
    return 100 * float(np.mean(vals)) if vals else 0.0


def in_frame(xy, yaw, targets, max_d, half_deg):
    """Steps where any target is within range and inside the horizontal FOV."""
    rel = targets[None, :, :] - xy[:, None, :]
    dist = np.linalg.norm(rel, axis=2)
    ang = (np.arctan2(rel[..., 1], rel[..., 0]) - yaw[:, None] + np.pi) % (2 * np.pi) - np.pi
    return ((dist < max_d) & (np.abs(ang) < np.radians(half_deg))).any(axis=1), dist.min(axis=1)


def _asc_step_blocks(path: Path) -> list:
    """steps.jsonl is one flat stream; `n` resets at every episode boundary."""
    out, prev = [[]], 1e9
    for line in path.open():
        s = json.loads(line)
        if s["n"] <= prev and out[-1]:
            out.append([])
        prev = s["n"]
        out[-1].append(s)
    return out


def block(title: str, asc: list, osg: list) -> None:
    print(f"\n{title}")
    print(f"  {'':22s} {'ASCENT':>10s} {'OSG':>10s} {'delta':>8s}")
    for label, field, scale in (
        ("SR %", "success", 100.0),
        ("SPL", "spl", 1.0),
        ("distance to goal m", "distance_to_goal", 1.0),
        ("steps", "steps", 1.0),
    ):
        a = np.mean([float(r.get(field) or 0.0) for r in asc]) * scale if asc else 0.0
        o = np.mean([float(r.get(field) or 0.0) for r in osg]) * scale if osg else 0.0
        print(f"  {label:22s} {a:10.2f} {o:10.2f} {a - o:+8.2f}")
    print(f"  {'n':22s} {len(asc):10d} {len(osg):10d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ascent_dir", help="dir holding the ASCENT episodes.jsonl/steps.jsonl")
    ap.add_argument("osg_run", help="an OSG outputs/<run> directory")
    ap.add_argument("--json", default="", help="also write the comparison here")
    ap.add_argument("--episodes-root",
                    default="data/datasets/objectnav/hm3d/v1/val/content",
                    help="episode jsons, for the goal geometry the agents never see")
    args = ap.parse_args()

    asc_rows = load(Path(args.ascent_dir) / "episodes.jsonl")
    osg_rows = load(Path(args.osg_run) / "episodes.jsonl")
    osg_by = {key(r): r for r in osg_rows}
    asc_by = {key(r): r for r in asc_rows}
    shared = sorted(set(asc_by) & set(osg_by))
    print(f"ASCENT {len(asc_rows)} episodes | OSG {len(osg_rows)} | paired {len(shared)}")
    if not shared:
        raise SystemExit("no episodes in common -- are these the same split?")

    asc = [asc_by[k] for k in shared]
    osg = [osg_by[k] for k in shared]
    for r, o in zip(asc, osg):
        r["_floor_class"] = o.get("floor_class")

    block("overall (paired)", asc, osg)
    for fc in ("same_floor", "cross_floor"):
        a = [r for r in asc if r["_floor_class"] == fc]
        o = [osg_by[key(r)] for r in a]
        block(f"{fc} (n={len(a)})", a, o)

    # ---------------------------------------------------------- per category
    print("\nper category (SR %, ASCENT vs OSG)")
    cats = defaultdict(lambda: [[], []])
    for r, o in zip(asc, osg):
        c = canon(r.get("target")) or canon(o.get("target"))
        cats[c][0].append(r)
        cats[c][1].append(o)
    print(f"  {'category':14s} {'n':>4s} {'ASCENT':>8s} {'OSG':>8s} {'delta':>8s}")
    for c in sorted(cats):
        a, o = cats[c]
        print(f"  {c:14s} {len(a):4d} {rate(a):8.1f} {rate(o):8.1f} {rate(a) - rate(o):+8.1f}")

    # -------------------------------------------------------------- the pairs
    won = [k for k in shared if asc_by[k]["success"] and not osg_by[k]["success"]]
    lost = [k for k in shared if not asc_by[k]["success"] and osg_by[k]["success"]]
    both = [k for k in shared if asc_by[k]["success"] and osg_by[k]["success"]]
    neither = [k for k in shared if not asc_by[k]["success"] and not osg_by[k]["success"]]
    print(f"\npaired outcome: both {len(both)} | ASCENT only {len(won)} | "
          f"OSG only {len(lost)} | neither {len(neither)}")

    def why_osg_failed(k) -> str:
        o = osg_by[k]
        if o["steps"] >= 500:
            return "timeout"
        if o.get("approach_stop_reason") and o["distance_to_goal"] > 3.0:
            return "stopped_on_wrong_object"
        if o.get("approach_stop_reason"):
            return "stopped_near_miss"
        return "other"

    print("\n  where ASCENT wins, what was OSG doing:")
    for cause, n in Counter(why_osg_failed(k) for k in won).most_common():
        print(f"    {cause:26s} {n}")
    print("\n  episodes ASCENT wins:")
    for k in won:
        a, o = asc_by[k], osg_by[k]
        print(f"    {k[0][:12]:12s} ep{k[1]:>3s} {canon(a.get('target')):10s} "
              f"{a['_floor_class'] or '-':11s} asc_steps={a.get('steps') or 0:3d} "
              f"osg_dtg={o['distance_to_goal']:5.1f} osg={why_osg_failed(k)}")

    # ------------------------------------------------ what the traces differ on
    print("\nbehaviour (paired means)")
    def asc_mode_frac(r, *modes) -> float:
        m = r.get("modes") or {}
        tot = sum(m.values()) or 1
        return 100 * sum(m.get(x, 0) for x in modes) / tot

    rows = [
        ("steps in a stair mode %", [asc_mode_frac(r, "climb_stair", "get_close_to_stair",
                                                   "look_for_downstair", "climb_stair_initialize")
                                     for r in asc],
         [100 * np.mean([("climb" in str(t.get("state", "")) or "stair" in str(t.get("state", "")))
                         for t in (o.get("step_trace") or [{}])]) for o in osg]),
        ("steps navigating to a target %", [asc_mode_frac(r, "navigate") for r in asc], None),
        ("LLM calls / episode", [r.get("llm_calls", 0) for r in asc],
         [o.get("llm_calls", 0) or (o.get("agent_stats") or {}).get("llm_calls", 0) for o in osg]),
        ("floors visited", [r.get("n_floors_seen", 1) for r in asc],
         [o.get("n_floors_seen", 1) for o in osg]),
        ("path length m", [r.get("path_len_m", 0) for r in asc],
         [(o.get("behaviour") or {}).get("path_len_m", 0) for o in osg]),
    ]
    for label, a, o in rows:
        am = float(np.mean(a)) if a else 0.0
        om = f"{float(np.mean(o)):10.2f}" if o else f"{'-':>10s}"
        print(f"  {label:32s} {am:10.2f} {om}")

    print("\nASCENT failure causes (its own logger):")
    for cause, n in Counter(r.get("failure_cause") for r in asc).most_common():
        print(f"    {str(cause):26s} {n}")

    # ------------------------------------------- reach / convert, against truth
    #
    # The only frame-exact comparison available: ASCENT logs GPS-relative poses
    # and OSG logs world poses, but the two runs share every episode, so the
    # OSG record's first step supplies the start pose that maps one into the
    # other. Validated by construction -- ASCENT's successes land a median
    # 0.04 m from a goal viewpoint once transformed.
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from analyse_behaviour import load_goal_geometry, load_goal_geometry_3d
        geo = load_goal_geometry(args.episodes_root)
    except Exception as exc:  # noqa: BLE001
        print(f"\n(no goal geometry: {exc})")
        geo = {}
    if geo:
        # steps.jsonl is in RUN order and `asc` is in sorted-key order, so the
        # two must be joined by episode, not zipped. Zipping them silently
        # scored each ASCENT trajectory against a different episode's goals.
        blocks_by_key = dict(zip(
            (key(r) for r in asc_rows),
            _asc_step_blocks(Path(args.ascent_dir) / "steps.jsonl"),
        ))
        blocks = [blocks_by_key.get(key(r), []) for r in asc]
        print("\nreach and convert, scored against the dataset's own object positions")
        print(f"  {'':22s} {'ASCENT':>10s} {'OSG':>10s}")
        rows = []
        for r, o, st in zip(asc, osg, blocks):
            g = geo.get(f"{o['scene']}:{o['episode_id']}")
            tr = o.get("step_trace") or []
            if not g or g[1] is None or not tr or not st:
                continue
            t = np.asarray(tr[0]["xy"], float)
            th = float(tr[0]["yaw"])
            c, s_ = np.cos(th), np.sin(th)
            rot = np.array([[c, -s_], [s_, c]])
            aw = np.array([rot @ np.asarray(x["xy"], float) + t for x in st])
            ow = np.array([x["xy"] for x in tr], float)
            rows.append((
                float(np.linalg.norm(aw[:, None, :] - g[1][None], axis=2).min()),
                float(np.linalg.norm(ow[:, None, :] - g[1][None], axis=2).min()),
                float(r["success"]), float(o["success"]),
                len(st) < 499, o["steps"] < 499,
            ))
        n = len(rows)
        ar = [x for x in rows if x[0] < 2.0]
        orr = [x for x in rows if x[1] < 2.0]
        print(f"  {'episodes scored':22s} {n:10d} {n:10d}")
        print(f"  {'stopped at all':22s} {sum(x[4] for x in rows):10d} {sum(x[5] for x in rows):10d}")
        print(f"  {'got within 2 m':22s} {len(ar):10d} {len(orr):10d}")
        print(f"  {'converted those':22s} "
              f"{100 * sum(x[2] for x in ar) / max(len(ar), 1):9.0f}% "
              f"{100 * sum(x[3] for x in orr) / max(len(orr), 1):9.0f}%")

    # ------------------------------------------ the S71 trace metrics (H1/H2)
    #
    # "SAW": some step had a true instance of the target within 3 m and ±40°
    # of the heading. Both traces are scored with the same rule; neither
    # carries the agent's height, so on same-floor episodes instances on other
    # storeys (|y - start_y| > 1.5 m) are dropped, and on cross-floor episodes
    # no height gate is applied. An episode that ended before step 499 was
    # STOPped by the agent; 499+ steps is a timeout (the harness forces the
    # last STOP). "committed" means `try_nav` was ever set.
    if geo:
        geo3 = load_goal_geometry_3d(args.episodes_root)
        ASC_CLIMB = {"climb_stair", "get_close_to_stair", "look_for_downstair", "climb_stair_initialize"}

        def osg_is_climb(t) -> bool:
            st = str(t.get("state", ""))
            return "climb" in st or "stair" in st

        def classify(saw: bool, success: bool, stopped: bool) -> str:
            if success:
                return "success"
            return f"{'saw' if saw else 'never_saw'} -> {'STOP' if stopped else 'timeout'}"

        tab = {"ASCENT": Counter(), "OSG": Counter()}
        agg = {"ASCENT": defaultdict(float), "OSG": defaultdict(float)}
        early_stops = {"ASCENT": [], "OSG": []}
        for r, o, st in zip(asc, osg, blocks):
            g3 = geo3.get(f"{o['scene']}:{o['episode_id']}")
            tr = o.get("step_trace") or []
            if g3 is None or not tr or not st:
                continue
            objs = g3
            same = (o.get("floor_class") == "same_floor")
            if same:
                start_y = float(o.get("start_y", tr[0].get("h", 0.88) - 0.88))
                objs = objs[np.abs(objs[:, 2] - start_y) < 1.5]
            if len(objs) == 0:
                continue
            t = np.asarray(tr[0]["xy"], float)
            th = float(tr[0]["yaw"])
            c, s_ = np.cos(th), np.sin(th)
            rot = np.array([[c, -s_], [s_, c]])
            for name, xy, yaw, n_steps, climb, try_nav, succ in (
                ("ASCENT",
                 np.array([rot @ np.asarray(x["xy"], float) + t for x in st]),
                 np.array([float(x["yaw"]) + th for x in st]),
                 len(st),
                 [x.get("mode") in ASC_CLIMB for x in st],
                 any(bool(x.get("try_nav")) for x in st),
                 bool(r["success"])),
                ("OSG",
                 np.array([x["xy"] for x in tr], float),
                 np.array([float(x["yaw"]) for x in tr]),
                 int(o["steps"]),
                 [osg_is_climb(x) for x in tr],
                 any(bool(x.get("try_nav")) for x in tr),
                 bool(o["success"])),
            ):
                seen, _ = in_frame(xy, yaw, objs[:, :2], 3.0, 40.0)
                saw = bool(seen.any())
                stopped = n_steps < 499
                tab[name][classify(saw, succ, stopped)] += 1
                A = agg[name]
                A["episodes"] += 1
                A["saw"] += saw
                A["steps"] += len(climb)
                A["climb_steps"] += sum(climb)
                if same:
                    A["same_steps"] += len(climb)
                    A["same_climb_steps"] += sum(climb)
                A["committed"] += try_nav
                A["committed_success"] += (try_nav and succ)
                if stopped:
                    early_stops[name].append(n_steps - 1)

        print("\ntrace metrics (paired, scored against the dataset's object positions)")
        print(f"  {'':40s} {'ASCENT':>10s} {'OSG':>10s}")
        for label in ("success", "saw -> STOP", "saw -> timeout", "never_saw -> STOP", "never_saw -> timeout"):
            print(f"  {label:40s} {tab['ASCENT'][label]:10d} {tab['OSG'][label]:10d}")
        def _row(label, f):
            print(f"  {label:40s} {f(agg['ASCENT']):>10s} {f(agg['OSG']):>10s}")
        _row("episodes scored", lambda A: f"{int(A['episodes'])}")
        _row("SAW episodes (target in frame, 3 m/40deg)", lambda A: f"{int(A['saw'])}")
        for name in ("ASCENT", "OSG"):
            agg[name]["saw_success"] = tab[name]["success"]          # every success saw the target
        _row("conversion given SAW", lambda A: f"{A['saw_success'] / max(A['saw'], 1):.3f}")
        _row("climb-mode steps, all episodes %", lambda A: f"{100 * A['climb_steps'] / max(A['steps'], 1):.1f}")
        _row("climb-mode steps, same-floor episodes %", lambda A: f"{100 * A['same_climb_steps'] / max(A['same_steps'], 1):.1f}")
        print(f"  {'earliest STOP step':40s} {min(early_stops['ASCENT'], default=-1):10d} {min(early_stops['OSG'], default=-1):10d}")
        print(f"  {'STOPs before step 33':40s} {sum(x < 33 for x in early_stops['ASCENT']):10d} {sum(x < 33 for x in early_stops['OSG']):10d}")
        _row("committed episodes", lambda A: f"{int(A['committed'])}")
        _row("P(success | committed)", lambda A: f"{A['committed_success'] / max(A['committed'], 1):.3f}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "n_paired": len(shared),
            "ascent": {"sr": rate(asc), "spl": float(np.mean([r["spl"] for r in asc]))},
            "osg": {"sr": rate(osg), "spl": float(np.mean([r["spl"] for r in osg]))},
            "ascent_only": [list(k) for k in won],
            "osg_only": [list(k) for k in lost],
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
