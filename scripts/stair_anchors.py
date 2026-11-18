#!/usr/bin/env python3
"""Rank every usable anchor in a scene by how far it is from the stairs.

The cross-anchor arm measured 0/11 on ascents and 5/15 on descents, so the
lever the benchmark has for difficulty is *where on the storey* the target
sits: a target at the far end of an unexplored floor costs the whole floor's
exploration before the approach even starts, while one beside the stair mouth
costs the climb and little else.

`replace_placement.py` steers only through `--exclude-anchor`, so "put it near
the stairs" has to be spelled as "exclude everything that is not". This script
prints that list.

Stairs are found from the navmesh rather than from geometry: `navmesh_storeys`
buckets an area-uniform sample of navigable points onto the storey peaks and
reports everything further than `band_m` from every peak as `off_band` -- and
on a scene whose stairs are navigable, those off-band points ARE the stairs
(1.2% of 00821's navigable area, in one connected run between its two floors).
A flight's MOUTH on a storey is then the point of that storey nearest to the
flight, which is the landing the agent actually steps onto.

    python scripts/stair_anchors.py --scene 00821-eF36g7L6Z9M
    python scripts/stair_anchors.py --all --near-m 4.0 --out outputs/audit/stairs.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/habitat-data-collector")
sys.path.insert(0, "/habitat-data-collector/scripts")

from osg.core.paths import hm3d_scene_root  # noqa: E402
from osg.eval.floors import (  # noqa: E402
    HEIGHT_AXIS,
    load_pathfinder,
    navmesh_floor_heights,
    navmesh_path_for,
    navmesh_storeys,
)

PLANE = [0, 2]


def stair_points(pathfinder, *, n_points: int, seed: int, band_m: float) -> tuple:
    """(storeys, stair_xyz) -- the storeys and the points that belong to none."""
    peaks = navmesh_floor_heights(pathfinder)
    info = navmesh_storeys(pathfinder, n_points=n_points, seed=seed, band_m=band_m)
    pts = info["points"]
    if not len(pts):
        return info["storeys"], np.zeros((0, 3))
    gaps = np.abs(pts[:, HEIGHT_AXIS][:, None] - np.asarray(peaks)[None, :])
    off = gaps.min(axis=1) > band_m
    return info["storeys"], pts[off]


def cluster_xy(points: np.ndarray, link_m: float) -> list:
    """Single-link clustering in the ground plane.

    A staircase is one connected run of navigable polygons, so single link is
    the right shape: it follows the run regardless of how it bends, where a
    centroid method would split a switchback into two.
    """
    if not len(points):
        return []
    xy = points[:, PLANE]
    unassigned = set(range(len(xy)))
    out = []
    while unassigned:
        seed = unassigned.pop()
        group, frontier = [seed], [seed]
        while frontier:
            i = frontier.pop()
            near = [j for j in unassigned
                    if np.linalg.norm(xy[j] - xy[i]) <= link_m]
            for j in near:
                unassigned.discard(j)
                group.append(j)
                frontier.append(j)
        out.append(points[group])
    out.sort(key=len, reverse=True)
    return out


def flight_mouths(storeys: list, flights: list, *, touch_m: float) -> list:
    """Per flight, the nearest point of each storey it actually touches."""
    out = []
    for flight in flights:
        entry = {"n_points": int(len(flight)),
                 "y_range": [float(flight[:, HEIGHT_AXIS].min()),
                             float(flight[:, HEIGHT_AXIS].max())],
                 "mouths": {}}
        for storey in storeys:
            pts = storey["points"]
            if not len(pts):
                continue
            d = np.linalg.norm(
                pts[:, None, PLANE] - flight[None, :, PLANE], axis=2
            ).min(axis=1)
            if d.min() > touch_m:
                continue
            entry["mouths"][storey["index"]] = {
                "xy": [float(v) for v in pts[int(np.argmin(d)), PLANE]],
                "height": float(storey["height"]),
                "gap_m": float(d.min()),
            }
        if entry["mouths"]:
            out.append(entry)
    return out


def anchor_candidates(scene: str, root: Path, split: str, config: Path) -> list:
    """Every anchor the generator would consider, with its centre."""
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    from auto_dualmap_authoring import (  # noqa: E402
        collect_anchor_candidates, detect_floors, load_config, open_simulator,
        scene_glb,
    )
    from habitat_data_collector.auto_authoring import (  # noqa: E402
        restrict_to_floors, restrict_to_island,
    )

    cfg = load_config(config, scene_glb(split, scene), root)
    cfg.scene_name = scene
    sim = open_simulator(cfg)
    try:
        levels = detect_floors(sim)
        cands = restrict_to_floors(
            restrict_to_island(collect_anchor_candidates(sim, levels=levels)),
            max_floors=2,
        )
        return [{"object_id": c.object_id, "category": c.category, "kind": c.kind,
                 "room": c.room, "center": list(c.center), "size": list(c.size),
                 "floor_index": int(c.floor_index),
                 "floor_height": float(c.floor_height)}
                for c in cands]
    finally:
        sim.close()


def nearest_mouth_m(center, mouths_by_height, tol_m=1.2):
    """Distance from an anchor to the nearest stair mouth on ITS OWN storey.

    Matched by height, not by the collector's floor_index: 00878's detector
    merges its basement and ground storey into one index, so an index match
    would compare an anchor against a mouth three metres below it.
    """
    xy = np.asarray([center[0], center[2]], dtype=float)
    best = None
    for height, mouth_xy in mouths_by_height:
        if abs(float(center[1]) - height) > 2.0:
            continue
        d = float(np.linalg.norm(xy - np.asarray(mouth_xy)))
        if best is None or d < best[0]:
            best = (d, height)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--root", type=Path,
                    default=Path("/habitat-data-collector/outputs/dualmap_multifloor"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--config", type=Path,
                    default=Path("/habitat-data-collector/config/habitat_data_collector.yaml"))
    ap.add_argument("--n-points", type=int, default=40000,
                    help="navmesh samples; stairs are ~1%% of them, so this "
                         "buys the flight its resolution")
    ap.add_argument("--band-m", type=float, default=0.75)
    ap.add_argument("--link-m", type=float, default=1.0,
                    help="single-link radius joining stair samples into a flight")
    ap.add_argument("--touch-m", type=float, default=1.5,
                    help="how close a storey must come to a flight to have a mouth on it")
    ap.add_argument("--near-m", type=float, default=4.0,
                    help="an anchor this close to a mouth counts as 'by the stairs'")
    ap.add_argument("--no-anchors", action="store_true",
                    help="stairs only; skips opening the simulator")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    scenes = args.scene
    if args.all or not scenes:
        scenes = sorted(p.name for p in args.root.iterdir() if p.is_dir())

    report = {"gates": {"near_m": args.near_m, "band_m": args.band_m,
                        "link_m": args.link_m, "touch_m": args.touch_m,
                        "n_points": args.n_points}, "scenes": {}}
    for scene in scenes:
        navmesh = navmesh_path_for(scene, hm3d_scene_root())
        if navmesh is None:
            print(f"{scene}: no navmesh", file=sys.stderr)
            continue
        pf = load_pathfinder(navmesh)
        storeys, stairs = stair_points(pf, n_points=args.n_points, seed=0,
                                       band_m=args.band_m)
        flights = cluster_xy(stairs, args.link_m)
        mouths = flight_mouths(storeys, flights, touch_m=args.touch_m)
        print(f"=== {scene}   storeys="
              f"{[round(s['height'], 2) for s in storeys]}  "
              f"stair_points={len(stairs)}  flights={len(mouths)}")
        flat = []
        for i, f in enumerate(mouths):
            spans = sorted(f["mouths"])
            print(f"  flight {i}: {f['n_points']:>4} pts  "
                  f"y {f['y_range'][0]:+.2f}..{f['y_range'][1]:+.2f}  "
                  f"connects storeys {spans}")
            for idx, m in sorted(f["mouths"].items()):
                print(f"      mouth on storey {idx} (y={m['height']:+.2f}): "
                      f"xy=({m['xy'][0]:+.2f}, {m['xy'][1]:+.2f})  gap={m['gap_m']:.2f} m")
                flat.append((m["height"], m["xy"]))
        entry = {"storeys": [{"index": s["index"], "height": s["height"],
                              "area_m2": s["area_m2"]} for s in storeys],
                 "flights": mouths}
        if not args.no_anchors and flat:
            cands = anchor_candidates(scene, args.root, args.split, args.config)
            ranked = []
            for c in cands:
                best = nearest_mouth_m(c["center"], flat)
                c["stair_m"] = None if best is None else round(best[0], 2)
                c["mouth_height"] = None if best is None else round(best[1], 2)
                ranked.append(c)
            ranked.sort(key=lambda c: (c["stair_m"] is None, c["stair_m"] or 0))
            entry["anchors"] = ranked
            near = [c for c in ranked if c["stair_m"] is not None
                    and c["stair_m"] <= args.near_m]
            print(f"  {len(near)}/{len(ranked)} anchors within {args.near_m} m "
                  f"of a stair mouth:")
            for c in near:
                print(f"      {c['stair_m']:>5.2f} m  {c['object_id']:<26} "
                      f"y={c['center'][1]:+.2f} {c['room']}")
        report["scenes"][scene] = entry

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
