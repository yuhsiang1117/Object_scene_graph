#!/usr/bin/env python3
"""Swap a target's STATIC and CROSS_ANCHOR poses, reversing the storey it must cross.

The cross-anchor arm measured 0/11 on ascents and 5/15 on descents, and the
agent always starts on the storey the PRIOR puts the object on
(`ycb.start_on_prior_floor: True`, composed into the arm), so the direction an
episode demands is exactly `sign(cross_anchor_floor - static_floor)`.  Nothing
in the config can invert that: `relocation_directions` only DROPS episodes, and
the start is pinned to the static floor.  Reversing an episode therefore means
re-authoring `static`, and the cheapest correct way to do that is to exchange
the two poses the scene already has -- each was generated and validated by the
same rules, so the exchange cannot invent an unreachable or floating pose.

    python scripts/flip_cross_anchor.py --scene 00821-eF36g7L6Z9M --auto-upward
    python scripts/flip_cross_anchor.py --all --auto-upward --dry-run

What the exchange does NOT carry, and must be repaired afterwards:

* **Separation between layouts.** Each pose was checked for clearance against
  the other objects of ITS OWN layout.  A partial swap mixes the two sets, so
  two objects can end up inside each other's keep-out or on one anchor.  This
  script reports both collisions; `verify_dualmap_layouts.py` is still the
  authority and must be run after.
* **`in_anchor`.** Its rule is that it must not change the storey relative to
  static (`validate_dualmap_authoring.py:176`), and a flip moves static to the
  other storey.  Every flipped target is listed as needing an in-anchor repair.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/habitat-data-collector/outputs/dualmap_multifloor")
TOL_M = 0.5          # ycb.relocation_floor_tolerance_m -- what OSG calls a storey change
SEPARATION_M = 0.35  # the generator's target-to-target keep-out


def direction(static_y: float, other_y: float) -> str:
    d = other_y - static_y
    return "same" if abs(d) <= TOL_M else ("up" if d > 0 else "down")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def scene_paths(root: Path, scene: str, index: int):
    d = root / scene
    return (d / "static_scene_config.json",
            d / "dynamic_scene_config" / "cross_anchor" / f"layout_{index:02d}.json",
            d / "dynamic_scene_config" / "in_anchor" / f"layout_{index:02d}.json")


def by_id(data: dict) -> dict:
    return {int(o["semantic_id"]): o for o in data["objects"]}


def flip_scene(root: Path, scene: str, index: int, wanted, *, dry_run: bool) -> dict:
    sp, cp, ip = scene_paths(root, scene, index)
    static, cross = load(sp), load(cp)
    s_by, c_by = by_id(static), by_id(cross)

    targets = []
    for sid in sorted(set(s_by) & set(c_by)):
        d = direction(s_by[sid]["anchor"]["floor_height"],
                      c_by[sid]["anchor"]["floor_height"])
        if (wanted == "upward" and d == "up") or (wanted == "all" and d != "same") \
                or (isinstance(wanted, (list, tuple, set)) and sid in wanted):
            targets.append(sid)

    report = {"scene": scene, "flipped": [], "collisions": [],
              "double_anchored": [], "in_anchor_repair": []}
    if not targets:
        return report

    for sid in targets:
        s, c = s_by[sid], c_by[sid]
        before = direction(s["anchor"]["floor_height"], c["anchor"]["floor_height"])
        for key in ("translation", "rotation", "object_id", "anchor"):
            s[key], c[key] = c.get(key), s.get(key)
        after = direction(s["anchor"]["floor_height"], c["anchor"]["floor_height"])
        report["flipped"].append({
            "semantic_id": sid,
            "handle": cross["id_handle_mapping"].get(str(sid)),
            "before": before, "after": after,
            "static_floor": s["anchor"]["floor_height"],
            "cross_floor": c["anchor"]["floor_height"],
        })
        report["in_anchor_repair"].append(sid)

    # Post-swap consistency, checked on BOTH layouts because the swap mixes two
    # independently validated sets.
    for name, data in (("static", static), ("cross_anchor", cross)):
        objs = data["objects"]
        seen = {}
        for o in objs:
            a = o["anchor"]["object_id"]
            seen.setdefault(a, []).append(int(o["semantic_id"]))
        for anchor, sids in seen.items():
            if len(sids) > 1:
                report["double_anchored"].append(
                    {"layout": name, "anchor": anchor, "semantic_ids": sorted(sids)})
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                a, b = objs[i]["translation"], objs[j]["translation"]
                gap = math.dist(a, b)
                if gap < SEPARATION_M:
                    report["collisions"].append({
                        "layout": name, "gap_m": round(gap, 3),
                        "semantic_ids": [int(objs[i]["semantic_id"]),
                                         int(objs[j]["semantic_id"])]})

    # `cross_floor_semantic_ids` is validated against the anchors, so it has to
    # be recomputed from what the anchors now say (validate_dualmap_authoring:184).
    s_floor = {sid: o["anchor"].get("floor_index") for sid, o in by_id(static).items()}
    cross["authoring"]["cross_floor_semantic_ids"] = sorted(
        int(o["semantic_id"]) for o in cross["objects"]
        if o["anchor"].get("floor_index") != s_floor.get(int(o["semantic_id"]))
    )
    report["cross_floor_semantic_ids"] = cross["authoring"]["cross_floor_semantic_ids"]

    if not dry_run:
        sp.write_text(json.dumps(static, indent=2))
        cp.write_text(json.dumps(cross, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--auto-upward", action="store_true",
                    help="flip every target whose cross_anchor pose is ABOVE static")
    ap.add_argument("--semantic-id", type=int, action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    scenes = args.scene or (sorted(p.name for p in args.root.iterdir() if p.is_dir())
                            if args.all else [])
    if not scenes:
        print("nothing to do: pass --scene or --all", file=sys.stderr)
        return 2
    wanted = set(args.semantic_id) if args.semantic_id else (
        "upward" if args.auto_upward else "all")

    reports, bad = [], False
    for scene in scenes:
        r = flip_scene(args.root, scene, args.index, wanted, dry_run=args.dry_run)
        reports.append(r)
        print(f"=== {scene}: {len(r['flipped'])} flipped"
              f"{' (dry run)' if args.dry_run else ''}")
        for f in r["flipped"]:
            print(f"    {f['semantic_id']} {str(f['handle'])[:26]:<26} "
                  f"{f['before']:>4} -> {f['after']:<4}  "
                  f"static {f['static_floor']:+.2f}  cross {f['cross_floor']:+.2f}")
        if r["flipped"]:
            print(f"    cross_floor_semantic_ids -> {r['cross_floor_semantic_ids']}")
        for c in r["collisions"]:
            bad = True
            print(f"    COLLISION {c['layout']}: {c['semantic_ids']} {c['gap_m']} m "
                  f"< {SEPARATION_M}")
        for d in r["double_anchored"]:
            bad = True
            print(f"    TWO ON ONE ANCHOR {d['layout']}: {d['anchor']} {d['semantic_ids']}")
        if r["in_anchor_repair"]:
            print(f"    in_anchor must follow static to the other storey for "
                  f"{r['in_anchor_repair']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, indent=2))
        print(f"wrote {args.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
