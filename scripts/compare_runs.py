#!/usr/bin/env python3
"""Two runs, side by side, down the funnel that decides an episode.

`analyze_campaign.py` discovers runs by the `+run_tag` recorded in Hydra's
parallel tree, which ties it to `outputs/`. This one takes explicit directories,
so an A/B run anywhere can be compared without moving files around, and it
reports the instrumented counters the campaign analyzer predates.

    python scripts/compare_runs.py CONTROL=path/to/a TREATMENT=path/to/b
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path


def load(path: Path):
    f = path / "episodes.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


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
