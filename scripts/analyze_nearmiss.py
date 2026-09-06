#!/usr/bin/env python3
"""The near-miss band: episodes that got close and did not score.

Success on this benchmark is P(the agent ends within `success_distance` of an
authored view point), and those sit on rings around the object's TRUE centre.
So an approach is only as good as the centre it aimed at, and this reports that
number directly: how far the committed estimate was from truth, how far the same
track's estimate ended up, and whether the gap between them was ever spent.

    python scripts/analyze_nearmiss.py outputs/<control> [outputs/<treatment>]

With two runs it pairs on episode_id and leads with the mechanism counter --
`approach_retargeted` -- because a difference without it is not this arm's.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def load(path: str) -> list:
    return [json.loads(l) for l in (Path(path) / "episodes.jsonl").read_text()
            .splitlines() if l.strip()]


def _xz(p):
    return (float(p[0]), float(p[2]))


def _d(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def errors(e: dict) -> dict:
    """Where the agent thought the object was, at the two moments that matter."""
    truth = _xz(e["authored_layout"]["target_position"])
    gc = e.get("goal_commit_log") or []
    tt = e.get("target_tracks") or []
    ad = e.get("approach_diag") or {}
    out = {
        "commit_err": _d(_xz(gc[-1]["center"]), truth) if gc else None,
        "final_err": _d(_xz(tt[0]["center"]), truth) if tt else None,
        "aimed_err": _d(tuple(ad["obj_xy"]), truth) if ad.get("obj_xy") else None,
        "retargets": len(e.get("approach_retarget_log") or []),
        "gt_min": e.get("gt_min_range_any_m"),
        "ok": bool(e.get("success")),
    }
    # The refinement the approach could have spent and did not -- but only the
    # SAME track's. `target_tracks` is every track carrying the target's label,
    # so tt[0] is routinely a different object from the one the agent committed
    # to, and differencing the two measures nothing. Where the committed track
    # survives to the end this is the arm's whole thesis in one number; where it
    # does not, the episode was lost to picking the wrong track and belongs to a
    # different fix.
    if gc:
        same = next((t for t in tt if t.get("track_id") == gc[-1].get("track_id")),
                    None)
        if same is not None:
            out["same_track_err"] = _d(_xz(same["center"]), truth)
            out["unspent"] = out["commit_err"] - out["same_track_err"]
    return out


def report(eps: list, title: str) -> None:
    print(f"\n===== {title}  n={len(eps)} =====")
    ok = sum(1 for e in eps if e.get("success"))
    print(f"  SR {ok}/{len(eps)} = {ok/max(1,len(eps)):.3f}")
    rt = sum(len(e.get("approach_retarget_log") or []) for e in eps)
    nrt = sum(1 for e in eps if e.get("approach_retarget_log"))
    print(f"  approach_retargeted: {rt} over {nrt} episodes"
          + ("   <-- the knob never fired" if rt == 0 else ""))
    print(f"\n  {'episode':<26}{'ok':>3}{'gt_min':>8}{'commit':>8}{'same_tr':>9}"
          f"{'unspent':>9}{'retgt':>6}")
    print("  (same_tr = the COMMITTED track's error at episode end; blank means"
          "\n   that track was gone, so the loss was the choice, not the aim)")
    for e in sorted(eps, key=lambda x: x.get("gt_min_range_any_m") or 99):
        r = errors(e)
        f = lambda v: f"{v:8.3f}" if isinstance(v, float) else "       -"
        lay = e["authored_layout"]["layout_id"]
        print(f"  {lay + ' ' + e['target']:<26}{int(r['ok']):>3}"
              f"{f(r['gt_min'])}{f(r['commit_err'])}{f(r.get('same_track_err'))}"
              f"{f(r.get('unspent'))}{r['retargets']:>6}")


def main() -> None:
    runs = sys.argv[1:]
    if not runs:
        print(__doc__)
        return
    control = load(runs[0])
    report(control, f"control  {runs[0]}")
    if len(runs) < 2:
        band = [e for e in control
                if not e.get("success")
                and 1.25 < (e.get("gt_min_range_any_m") or 99) <= 2.0]
        print(f"\n  near-miss band (failed, 1.25-2.0 m): {len(band)}/{len(control)}")
        return

    treat = load(runs[1])
    report(treat, f"treatment  {runs[1]}")
    ci = {e["episode_id"]: e for e in control}
    print("\n===== paired =====")
    flips = []
    for e in treat:
        c = ci.get(e["episode_id"])
        if c is None:
            continue
        if bool(c["success"]) != bool(e["success"]):
            flips.append((e["episode_id"], bool(c["success"]), bool(e["success"])))
    for eid, a, b in flips:
        print(f"  {'WON ' if b else 'LOST'}  {eid}   {a} -> {b}")
    if not flips:
        print("  no episode changed outcome")
    moved = [e for e in treat if e.get("approach_retarget_log")]
    print(f"\n  episodes where the goal moved: {len(moved)}")
    for e in moved:
        c = ci.get(e["episode_id"])
        rc, rt = errors(c) if c else {}, errors(e)
        print(f"    {e['authored_layout']['layout_id']:<16} {e['target']:<12}"
              f" aimed {rc.get('aimed_err')} -> {rt.get('aimed_err')}"
              f"   ok {int(bool(c and c['success']))} -> {int(bool(e['success']))}"
              f"   moves {e['approach_retarget_log']}")


if __name__ == "__main__":
    main()
