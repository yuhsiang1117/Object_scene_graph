#!/usr/bin/env python3
"""Two runs, side by side, down the funnel that decides an episode.

`analyze_campaign.py` discovers runs by the `+run_tag` recorded in Hydra's
parallel tree, which ties it to `outputs/`. This one takes explicit directories,
so an A/B run anywhere can be compared without moving files around, and it
reports the instrumented counters the campaign analyzer predates.

    python scripts/compare_runs.py CONTROL=path/to/a TREATMENT=path/to/b
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def load(path: Path):
    f = path / "episodes.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def load_paired(path: Path):
    rows = load(path)
    paired = {
        row.get("uid") or f"{row.get('scene', '?')}:{row['episode_id']}": row
        for row in rows
    }
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return paired, summary.get("config", {}), summary.get("algorithm", {})


def _is_cross_floor(a: dict, b: dict, threshold: float) -> bool:
    for row in (a, b):
        if row.get("floor_class") in ("same_floor", "cross_floor"):
            return row["floor_class"] == "cross_floor"
        if row.get("goal_floor_gap_m") is not None:
            return float(row["goal_floor_gap_m"]) > threshold
    return max(float(a.get("traj_y_range", 0.0) or 0.0),
               float(b.get("traj_y_range", 0.0) or 0.0)) > threshold


def paired_report(argv) -> None:
    parser = argparse.ArgumentParser(description="paired source/treatment comparison")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("treatment", type=Path)
    parser.add_argument("--split", choices=("all", "multi", "single"), default="all")
    parser.add_argument("--floor-span", type=float, default=1.0)
    parser.add_argument("--flips", action="store_true")
    args = parser.parse_args(argv)
    a, a_cfg, a_alg = load_paired(args.baseline)
    b, b_cfg, b_alg = load_paired(args.treatment)
    shared = sorted(set(a) & set(b))
    if args.split != "all":
        want_cross = args.split == "multi"
        shared = [uid for uid in shared
                  if _is_cross_floor(a[uid], b[uid], args.floor_span) == want_cross]
    if not shared:
        raise SystemExit("no paired episodes match this split")

    protocol_keys = ("success_distance", "dataset_version", "split")
    mismatch = {key: (a_cfg.get(key), b_cfg.get(key)) for key in protocol_keys
                if key in a_cfg and key in b_cfg and a_cfg[key] != b_cfg[key]}
    if mismatch:
        print(f"ERROR: protocol mismatch: {mismatch}")
    alg_diff = {key: (a_alg.get(key), b_alg.get(key))
                for key in sorted(set(a_alg) | set(b_alg))
                if a_alg.get(key) != b_alg.get(key)}
    print(f"algorithm differences: {alg_diff or 'none'}")

    gained = [uid for uid in shared if a[uid]["success"] < 0.5 <= b[uid]["success"]]
    lost = [uid for uid in shared if b[uid]["success"] < 0.5 <= a[uid]["success"]]
    sr_a = statistics.mean(a[uid]["success"] for uid in shared)
    sr_b = statistics.mean(b[uid]["success"] for uid in shared)
    print(f"paired={len(shared)} baseline_SR={sr_a:.3f} treatment_SR={sr_b:.3f}")
    print(f"gained={len(gained)} lost={len(lost)} net={len(gained)-len(lost):+d}")
    if args.flips:
        for label, ids in (("gained", gained), ("lost", lost)):
            for uid in ids:
                print(f"{label:7s} {uid} target={a[uid].get('target', '?')}")

    mechanisms = (
        "floor_switches", "floor_switch_attempts", "directed_floor_switch_attempts",
        "climb_attempt", "goal_floor_reached", "cross_floor_search_requests",
        "search_surface", "glance_updates", "steps", "spl", "distance_to_goal",
    )
    for key in mechanisms:
        av = [a[uid].get("agent_stats", {}).get(key, a[uid].get(key)) for uid in shared]
        bv = [b[uid].get("agent_stats", {}).get(key, b[uid].get(key)) for uid in shared]
        pairs = [(x, y) for x, y in zip(av, bv) if x is not None and y is not None]
        if pairs:
            before = statistics.mean(float(x) for x, _ in pairs)
            after = statistics.mean(float(y) for _, y in pairs)
            print(f"{key:34s} {before:9.3f} -> {after:9.3f} ({after-before:+.3f})")


def d2(a, b):
    return math.dist((a[0], a[2]), (b[0], b[2]))


def stat(eps, key, default=0):
    return sum(e["agent_stats"].get(key, default) for e in eps)


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else float("nan")


def funnel(eps) -> dict:
    n = len(eps)
    out = {"episodes": n}
    if not n:
        return out
    out["SR"] = statistics.mean([e["success"] for e in eps])
    out["SPL"] = statistics.mean([e["spl"] for e in eps])
    for kind in ("in_anchor", "cross_anchor"):
        sub = [e for e in eps if e["authored_layout"].get("layout_type") == kind]
        out[f"SR {kind}"] = statistics.mean([e["success"] for e in sub]) if sub else float("nan")
    out["timed out"] = sum(1 for e in eps if e["steps"] >= 500)
    out["median steps"] = med([e["steps"] for e in eps])

    # --- perception -> map -> commit -> stop
    localized = 0
    named_no_track = 0
    for e in eps:
        tp = e["authored_layout"].get("target_position")
        if not tp:
            continue
        near = any(d2(t["center"], tp) <= 0.5 for t in (e.get("target_tracks") or []))
        localized += int(near)
        if not near and e.get("gt_kf_detected", 0) > 0 and not e["success"]:
            named_no_track += 1
    out["mapped within 0.5 m"] = localized
    out["named it, no track (fail)"] = named_no_track
    out["in-situ recall"] = (
        sum(e.get("gt_kf_detected", 0) for e in eps)
        / max(1, sum(e.get("gt_kf_in_view", 0) for e in eps))
    )
    if any("gt_kf_admitted" in e for e in eps):
        out["ADMITTED recall"] = (
            sum(e.get("gt_kf_admitted", 0) for e in eps)
            / max(1, sum(e.get("gt_kf_in_view", 0) for e in eps))
        )
        out["named but not admitted"] = (
            sum(e.get("gt_kf_detected", 0) for e in eps)
            - sum(e.get("gt_kf_admitted", 0) for e in eps)
        )

    # --- the counters
    out["unreachable_skip"] = stat(eps, "unreachable_skip")
    out["  ...episodes hit"] = sum(1 for e in eps if e["agent_stats"].get("unreachable_skip"))
    out["absence_abandon"] = stat(eps, "absence_abandon")
    out["surface inspections"] = stat(eps, "search_surface")
    out["frontier goals"] = stat(eps, "select_ok")
    if any("glance_updates" in e["agent_stats"] for e in eps):
        out["glance updates"] = stat(eps, "glance_updates")
        out["surfaces touched"] = med([e["agent_stats"].get("surfaces_touched") for e in eps])
        out["surfaces retired (<0.1)"] = med([e["agent_stats"].get("surfaces_retired") for e in eps])
        out["min surface factor"] = med([e["agent_stats"].get("surface_factor_min") for e in eps])
    if any("det_admitted" in e["agent_stats"] for e in eps):
        out["det admitted"] = stat(eps, "det_admitted")
        out["  obs_rejected"] = stat(eps, "obs_rejected")
        out["  ellipsoid_rejected"] = stat(eps, "ellipsoid_rejected")
        out["  tracks_created"] = stat(eps, "tracks_created")
    return out


def main(argv):
    # Teammate mode: two positional run directories with paired flip analysis.
    # Dynamic-scene mode below remains compatible with NAME=directory columns.
    if argv and all("=" not in arg for arg in argv if not arg.startswith("--")):
        paired_report(argv)
        return
    named = []
    for arg in argv:
        label, _, path = arg.partition("=")
        named.append((label, Path(path)))
    cols = [(label, funnel(load(path))) for label, path in named]
    keys = []
    for _, f in cols:
        for k in f:
            if k not in keys:
                keys.append(k)
    w = max(len(k) for k in keys) + 2
    head = "".join(f"{label:>16}" for label, _ in cols)
    print(f"{'':{w}}{head}")
    print("-" * (w + 16 * len(cols)))
    for k in keys:
        row = f"{k:{w}}"
        for _, f in cols:
            v = f.get(k)
            if v is None:
                row += f"{'-':>16}"
            elif isinstance(v, float):
                row += f"{v:>16.3f}"
            else:
                row += f"{v:>16}"
        print(row)


if __name__ == "__main__":
    main(sys.argv[1:])
