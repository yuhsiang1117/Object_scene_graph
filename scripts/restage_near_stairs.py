#!/usr/bin/env python3
"""Re-place authored targets beside the stairs, through the collector's generator.

Two levers set how hard a cross-floor episode is, and only one of them is the
climb.  The other is how much floor the agent has to sweep at each end: a
target at the far side of an unexplored storey costs that storey's exploration
before the approach starts.  Measured on the 26-episode arm, every success sat
1.4-9.2 m from a stair mouth and the worst failures sat 9-15 m out.

So this puts BOTH ends of a cross-floor episode by the stairs:

* the CROSS_ANCHOR pose, because that is the goal; and
* the STATIC pose, because the agent starts on the static storey and walks to
  the prior position first -- finishing that check next to the stair mouth is
  most of the saving.

`replace_placement.py` steers only through `--exclude-anchor`, so "near the
stairs, on this storey" is spelled as "exclude every anchor that is not", from
the ranking `stair_anchors.py` produces.  Everything the generator enforces
still applies to the new pose, which is the point of going through it rather
than writing coordinates: rests on its anchor, upright, clear of the other
objects, visible from a navigable viewpoint on the connected navmesh.

The radius escalates per target (--radii) because anchor supply is thin and
each anchor takes one object: 00878's ground storey offers 7 anchors for 6
targets, so a fixed radius would fail rather than place slightly further out.

    python scripts/restage_near_stairs.py --stairs outputs/audit/stair_anchors.json
    python scripts/restage_near_stairs.py --scene 00821-eF36g7L6Z9M --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COLLECTOR = Path("/habitat-data-collector")
DEFAULT_ROOT = COLLECTOR / "outputs" / "dualmap_multifloor"
TOL_M = 0.5
STOREY_M = 2.0   # an anchor belongs to the storey whose mouth it matches within this


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def scene_files(root: Path, scene: str, index: int):
    d = root / scene
    return (d / "static_scene_config.json",
            d / "dynamic_scene_config" / "cross_anchor" / f"layout_{index:02d}.json",
            d / "dynamic_scene_config" / "in_anchor" / f"layout_{index:02d}.json")


def allowed_anchors(anchors, storey_y, radius_m):
    """Anchors on `storey_y` within `radius_m` of a stair mouth."""
    return [a for a in anchors
            if a.get("mouth_height") is not None
            and abs(a["mouth_height"] - storey_y) <= TOL_M
            and a["stair_m"] is not None and a["stair_m"] <= radius_m]


def run_replace(scene, layout, index, handle, exclude, *, root, seed, dry_run):
    cmd = ["xvfb-run", "-a", "python", "scripts/replace_placement.py",
           "--root", str(root.resolve()), "--scene", scene, "--layout", layout,
           "--target", handle, "--seed", str(seed)]
    if layout != "static":
        cmd += ["--index", str(index)]
    if exclude:
        cmd += ["--exclude-anchor", *exclude]
    if dry_run:
        cmd += ["--dry-run"]
    p = subprocess.run(cmd, cwd=COLLECTOR, capture_output=True, text=True)
    noise = ("Gym has been", "Please upgrade", "See the migration")
    def lines(stream):
        return [l for l in (stream or "").splitlines()
                if l.strip() and not l.startswith(noise)]
    # On failure the reason is on STDERR ("No other affordable anchor is free"),
    # while stdout still carries the header line -- so preferring stdout hides
    # exactly the message that says why the placement could not be made.
    err, out = lines(p.stderr), lines(p.stdout)
    if p.returncode != 0:
        return p.returncode, (err[-1] if err else (out[-1] if out else "?"))
    return 0, (out[-1] if out else "")


def restage_scene(scene, stairs, args) -> list:
    root = args.root
    sp, cp, ip = scene_files(root, scene, args.index)
    anchors = stairs["scenes"].get(scene, {}).get("anchors") or []
    if not anchors:
        return [{"scene": scene, "note": "no anchor ranking; skipped"}]
    all_ids = [a["object_id"] for a in anchors]
    out = []

    def place(layout, handle, storey_y, sid, why):
        for radius in args.radii:
            keep = {a["object_id"] for a in allowed_anchors(anchors, storey_y, radius)}
            if not keep:
                continue
            exclude = [i for i in all_ids if i not in keep]
            code, msg = run_replace(scene, layout, args.index, handle, exclude,
                                    root=root, seed=args.seed, dry_run=args.dry_run)
            if code == 0:
                out.append({"scene": scene, "sid": sid, "layout": layout, "why": why,
                            "radius_m": radius, "ok": True, "msg": msg,
                            "n_allowed": len(keep)})
                return True
        out.append({"scene": scene, "sid": sid, "layout": layout, "why": why,
                    "ok": False, "msg": msg if 'msg' in dir() else "no anchor in any radius"})
        return False

    static, cross = load(sp), load(cp)
    s_by = {int(o["semantic_id"]): o for o in static["objects"]}
    c_by = {int(o["semantic_id"]): o for o in cross["objects"]}
    handles = {int(k): v for k, v in cross["id_handle_mapping"].items()}

    if args.ends == "in_anchor":
        # in_anchor's rule is that it must NOT change the storey relative to
        # static (validate_dualmap_authoring.py:176), and in this dataset it
        # always sits on static's own anchor.  Moving static therefore breaks
        # it, so force it back by excluding every anchor except static's.
        inn = load(ip)
        i_by = {int(o["semantic_id"]): o for o in inn["objects"]}
        for sid in sorted(set(s_by) & set(i_by)):
            want = s_by[sid]["anchor"]["object_id"]
            if i_by[sid]["anchor"]["object_id"] == want:
                continue
            exclude = [i for i in all_ids if i != want]
            code, msg = run_replace(scene, "in_anchor", args.index, handles.get(sid),
                                    exclude, root=root, seed=args.seed,
                                    dry_run=args.dry_run)
            out.append({"scene": scene, "sid": sid, "layout": "in_anchor",
                        "why": f"follow static onto {want}", "ok": code == 0,
                        "msg": msg, "radius_m": "-"})
            inn = load(ip)
            i_by = {int(o["semantic_id"]): o for o in inn["objects"]}
        return out

    for sid in sorted(set(s_by) & set(c_by)):
        sy = s_by[sid]["anchor"]["floor_height"]
        cy = c_by[sid]["anchor"]["floor_height"]
        if abs(cy - sy) <= TOL_M:
            continue                       # same-floor target: not this script's business
        handle = handles.get(sid)
        if args.ends in ("both", "static"):
            place("static", handle, sy, sid, "prior end, by the stairs")
            static = load(sp); s_by = {int(o["semantic_id"]): o for o in static["objects"]}
            sy = s_by[sid]["anchor"]["floor_height"]
        if args.ends in ("both", "cross"):
            place("cross_anchor", handle, cy, sid, "goal end, by the stairs")
            cross = load(cp); c_by = {int(o["semantic_id"]): o for o in cross["objects"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--stairs", type=Path, default=Path("outputs/audit/stair_anchors.json"))
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--radii", type=float, nargs="*", default=[4.0, 6.0, 8.0, 12.0])
    ap.add_argument("--ends", choices=("both", "static", "cross", "in_anchor"),
                    default="both")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    stairs = load(args.stairs)
    scenes = args.scene or sorted(stairs["scenes"])
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda s: restage_scene(s, stairs, args), scenes))

    flat, bad = [], 0
    for rows in results:
        for r in rows:
            flat.append(r)
            if r.get("note"):
                print(f"  {r['scene']}: {r['note']}")
                continue
            mark = "ok " if r["ok"] else "FAIL"
            if not r["ok"]:
                bad += 1
            print(f"  {mark} {r['scene'][:5]} {r['sid']} {r['layout']:<12} "
                  f"r={r.get('radius_m', '-')!s:<5} {r['msg'][:96]}")
    print(f"\n{len(flat) - bad}/{len(flat)} placements succeeded")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(flat, indent=2))
        print(f"wrote {args.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
