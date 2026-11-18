#!/usr/bin/env python3
"""Is the prior map good enough to run pass 2 on? Answered in numbers, per storey.

The cross-anchor experiment kept failing for a reason upstream of the
navigation: the map. On 00800 the kept obstacle snapshot covered 51% of the
lower storey and five of twelve authored target positions fell 2-5 m outside
it; on 00808 no mapping episode visited both storeys, so the scene graph
recorded its "upper" storey at 1.03 m when the real ones are 0.06 and 3.0.
Both were found by hand, after the GPU had been spent. This is the instrument
that finds them first.

Per scene, per storey, against the scene's own navmesh:

  free / occupied / unknown   what the pasted map calls each navigable point
  storeys                     navmesh truth vs what the union actually covers
  stairs                      cells and the span of the pasted ramp
  graph                       scene-graph layer, its tracks, its YCB targets
  targets                     every authored position, in BOTH layouts: which
                              storey, which cell, how far to mapped free space

FREE IS COVERAGE; OCCUPIED IS SOMETHING ELSE. They are different questions and
they get different gates. A navmesh-navigable point the map calls FREE is floor
the planner can use. A navmesh-navigable point it calls OCCUPIED is the planner
being *blocked* from floor that exists -- which on a correct map is furniture:
ASCENT stamps a 0.61-0.88 m band, and a table top over navigable floor lands in
it. Measured on the correct 00800 union, lower storey 0.000 and upper 0.040,
steady from the second snapshot on while free climbed 0.742 -> 0.876.

So occupied alone does NOT detect a transposed paste, and reading the log's
"0.04-0.05 transposed" as a threshold would have been wrong. What separates the
two is FREE: the transposed map read free 0.04 where the correct one reads 0.80.
Both are reported; both are gated; neither is collapsed into the other. And
"mapped" is never defined as `grid != UNKNOWN`, which would merge them into one
number a transposed paste can score well on.

Everything geometric goes through `osg.eval.prior_map.paste_scene`, which is
the paste a run performs, with the costmaps passed in. A second implementation
is exactly how the transposition bug happened, and it must not be reintroduced
for a diagnostic.

No GPU, no Simulator, no perception servers -- a bare `PathFinder` and the
stored snapshots, about two seconds a scene.

    python scripts/audit_map_coverage.py \\
        --maps outputs/maps_p1500_ascent --graphs outputs/maps_p1500_osg \\
        --layout-root /habitat-data-collector/outputs/dualmap_multifloor \\
        --layout-root outputs/collector_layouts \\
        --out outputs/audit/map_coverage.json --label 1500-union

Exit 0 pass, 1 a gate tripped, 2 NOTHING WAS CHECKED (a missing snapshot or an
unimportable habitat_sim is not a pass).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osg.core.config.ycb import YCB_TARGET_LABELS  # noqa: E402
from osg.core.paths import hm3d_scenes_dir  # noqa: E402
from osg.eval.floors import (  # noqa: E402
    load_pathfinder,
    navmesh_path_for,
    navmesh_storeys,
)
from osg.eval.prior_map import (  # noqa: E402
    _map_path,
    _scene_map_paths,
    paste_scene,
    storeys_from_snapshots,
)
from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN  # noqa: E402

EXIT_OK, EXIT_GATE, EXIT_NOTHING = 0, 1, 2


# --------------------------------------------------------------- authored data

def read_layouts(layout_roots: Sequence[Path], scene: str,
                 layout_types: Sequence[str],
                 layout_indices: Sequence[int]) -> List[dict]:
    """Every authored layout for a scene, as raw JSON.

    Raw, not `AuthoredLayout`: the validated dataclass drops `anchor.floor_index`,
    `anchor.floor_height` and `anchor.top_y`, and this needs all three to
    cross-check its own storey assignment against the authoring's.
    """
    out: List[dict] = []
    for root in layout_roots:
        base = Path(root) / scene
        if not base.is_dir():
            continue
        wanted = []
        if "static" in layout_types:
            wanted.append(("static", base / "static_scene_config.json"))
        for kind in ("in_anchor", "cross_anchor"):
            if kind not in layout_types:
                continue
            for index in layout_indices:
                wanted.append((f"{kind}_{index:02d}",
                               base / "dynamic_scene_config" / kind
                               / f"layout_{index:02d}.json"))
        for layout_id, path in wanted:
            if not path.exists():
                continue
            try:
                blob = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"  ! unreadable layout {path}: {exc}")
                continue
            blob["_layout_id"] = layout_id
            blob["_path"] = str(path)
            out.append(blob)
        if out:
            break  # the first root that holds the scene owns it
    return out


def storey_of(base_y: float, heights: Sequence[float], eps_m: float = 0.15) -> int:
    """The storey an object at this base height is standing ON.

    The HIGHEST storey at or below it, not the NEAREST. A target sits 0.4-1.2 m
    up on furniture, so nearest-height puts a bottle on a 1.2 m counter on the
    storey above as soon as two storeys are less than 2.4 m apart -- and 00878's
    are 2.85. `eps_m` forgives a placement a few centimetres below its own
    floor height.
    """
    below = [i for i, h in enumerate(heights) if base_y >= h - eps_m]
    if below:
        return max(below, key=lambda i: heights[i])
    return int(np.argmin([abs(base_y - h) for h in heights])) if heights else 0


# ------------------------------------------------------------------- measuring

def coverage(costmap, points_xz: np.ndarray) -> dict:
    """How the pasted map classifies each navigable point of one storey."""
    n = int(len(points_xz))
    if n == 0:
        return {"n": 0, "free": 0.0, "occupied": 0.0, "unknown": 0.0, "oob": 0.0}
    rc = np.floor((points_xz - costmap.origin) / costmap.resolution).astype(int)
    h, w = costmap.grid.shape
    inside = ((rc[:, 0] >= 0) & (rc[:, 1] >= 0) & (rc[:, 0] < h) & (rc[:, 1] < w))
    values = np.full(n, -999, dtype=np.int64)
    values[inside] = costmap.grid[rc[inside, 0], rc[inside, 1]]
    return {
        "n": n,
        "free": float(np.mean(values == FREE)),
        "occupied": float(np.mean(values == OCCUPIED)),
        "unknown": float(np.mean(values == UNKNOWN)),
        "oob": float(np.mean(~inside)),
    }


def free_distance_m(costmap) -> np.ndarray:
    """Metres from every cell to the nearest FREE cell.

    A full distance transform, NOT `costmap.nearest_free_xy`: that searches a
    1.5 m box and returns the query point unchanged when it finds nothing, so a
    target 5 m outside the map would be reported at 0.00 m -- the exact reading
    this script exists to make impossible.
    """
    from scipy import ndimage

    free = costmap.grid == FREE
    if not free.any():
        return np.full(costmap.grid.shape, np.inf, dtype=np.float64)
    return ndimage.distance_transform_edt(~free) * costmap.resolution


def stair_report(costmap, storey_gap_m: Optional[float]) -> dict:
    """The staircase as PASTED: how many cells, and how much of the gap the
    height ramp spans.

    A flight cut short is not a cosmetic defect: `find_flights` keeps cells
    within a band of the floor, and when the ramp stopped at 1.6 m of a 3.0 m
    gap the carrot had no waypoint past that point, the climb fell back to
    ASCENT's depth ray, and the agent drove into the banister for 120 steps.
    """
    mask = getattr(costmap, "stair_mask", None)
    cells = int(np.count_nonzero(mask)) if mask is not None else 0
    span = None
    if cells and getattr(costmap, "height", None) is not None:
        hs = costmap.height[mask]
        hs = hs[np.isfinite(hs)]
        if hs.size:
            span = float(hs.max() - hs.min())
    return {
        "cells": cells,
        "ramp_span_m": span,
        "storey_gap_m": storey_gap_m,
        "ramp_span_frac": (None if span is None or not storey_gap_m
                           else float(span / storey_gap_m)),
    }


def graph_report(graphs_dir: Optional[Path], scene: str, heights: Sequence[float],
                 match_m: float) -> dict:
    """The scene-graph prior: which storeys it has, and what is on each.

    Tracks are bucketed by their own `center[1]`, NOT by `track.floor_key`.
    Measured on 00808: the prior's single YCB target track is filed under a
    phantom 1.03 m storey while its centre sits at y=0.56. Reporting the
    disagreement is the point -- it is the same class of defect as the storey
    the estimator invented there.
    """
    if graphs_dir is None:
        return {"present": False}
    path = _map_path(str(graphs_dir), scene)
    if not path.exists():
        return {"present": False, "path": str(path)}
    blob = json.loads(path.read_text(encoding="utf-8"))
    floors = blob.get("floors") or []
    tracks = blob.get("tracks") or []
    labels = set(YCB_TARGET_LABELS.values())

    per_storey: List[dict] = []
    for index, height in enumerate(heights):
        layer = None
        for floor in floors:
            fy = floor.get("height_y")
            if fy is None:
                continue
            if abs(float(fy) - height) <= match_m:
                if layer is None or abs(float(fy) - height) < abs(
                        float(layer.get("height_y", 1e9)) - height):
                    layer = floor
        mine, ycb = 0, Counter()
        for track in tracks:
            centre = track.get("center") or track.get("centre")
            if not centre or len(centre) < 2:
                continue
            if storey_of(float(centre[1]), list(heights)) == index:
                mine += 1
                if str(track.get("label", "")) in labels:
                    ycb[str(track["label"])] += 1
        per_storey.append({
            "floor_key": (None if layer is None else layer.get("key")),
            "height_y": (None if layer is None else layer.get("height_y")),
            "tracks_by_centre_y": mine,
            "ycb_tracks": dict(ycb),
        })

    # A track is MIS-FILED only when some other storey is a better fit for its
    # own centre, not merely when it sits high above the one it is filed under:
    # a ceiling lamp is 2.5 m above its floor and perfectly well filed. The
    # first version of this counted height alone and reported 211 of 402 tracks
    # on 00800 as disagreeing, which is an artefact of tall furniture. Measured
    # properly it is the 00808 defect this is looking for -- a track filed under
    # a phantom storey the estimator invented.
    mismatched = 0
    key_to_height = {f.get("key"): f.get("height_y") for f in floors}
    known = [(k, float(h)) for k, h in key_to_height.items() if h is not None]
    for track in tracks:
        centre = track.get("center") or track.get("centre")
        fy = key_to_height.get(track.get("floor_key"))
        if not centre or len(centre) < 2 or fy is None or len(known) < 2:
            continue
        base = float(centre[1])
        mine = abs(base - float(fy))
        best = min(abs(base - h) for _, h in known)
        # 0.5 m of slack so a track midway between storeys is not counted twice
        if mine > best + 0.5:
            mismatched += 1
    return {
        "present": True, "path": str(path),
        "graph_floors": [f.get("height_y") for f in floors],
        "n_tracks": len(tracks),
        "storeys": per_storey,
        "tracks_whose_floor_key_disagrees_with_geometry": mismatched,
    }


# -------------------------------------------------------------------- one scene

def audit_scene(scene: str, args, layout_roots: Sequence[Path]) -> dict:
    row: dict = {"scene": scene, "failures": [], "notes": []}

    navmesh = navmesh_path_for(scene, args.scenes_dir)
    if navmesh is None:
        row["verdict"] = "not-checked"
        row["failures"].append(f"no navmesh under {args.scenes_dir}")
        return row
    pathfinder = load_pathfinder(navmesh)
    if pathfinder is None:
        row["verdict"] = "not-checked"
        row["failures"].append("habitat_sim unavailable or navmesh would not load")
        return row

    nav = navmesh_storeys(pathfinder, n_points=args.nav_points, seed=args.seed,
                          band_m=args.storey_band_m)
    nav_heights = [s["height"] for s in nav["storeys"]]
    row["navmesh"] = {
        "path": str(navmesh),
        "navigable_area_m2": nav["navigable_area_m2"],
        "n_storeys": len(nav_heights),
        "off_band_frac": nav["off_band_frac"],
        "storeys": [{k: v for k, v in s.items() if k != "points"}
                    for s in nav["storeys"]],
    }
    # The start island: coverage on an island the agent cannot walk to is not a
    # map defect. 00848 has 3 islands and 00880 has 4, so 100% is unreachable
    # there by construction.
    sample = nav["points"]
    islands = Counter(int(pathfinder.get_island(pathfinder.snap_point(p)))
                      for p in sample[:2000])
    main_island = islands.most_common(1)[0][0] if islands else 0
    row["navmesh"]["islands"] = dict(islands)
    row["navmesh"]["main_island"] = main_island

    paths = _scene_map_paths(str(args.maps), scene)
    if not paths:
        row["verdict"] = "not-checked"
        row["failures"].append(f"no obstacle snapshot for {scene} under {args.maps}")
        return row

    graph = graph_report(args.graphs, scene, nav_heights, args.storey_match_m)
    row["graph"] = graph

    # Which storey heights the paste is told. `pass2` replicates what
    # `load_obstacle_map` + `_seed_storeys` will do on the night: the scene
    # graph's own storeys, plus any snapshot cluster they lack.
    union_heights = storeys_from_snapshots(paths)
    if args.storeys == "navmesh":
        heights, source = list(nav_heights), "navmesh"
    elif args.storeys == "union":
        heights, source = list(union_heights), "union"
    else:
        heights = [float(h) for h in (graph.get("graph_floors") or [])
                   if h is not None]
        seeded = []
        for h in union_heights:
            if all(abs(h - k) > args.storey_match_m for k in heights):
                heights.append(h)
                seeded.append(round(h, 3))
        heights = sorted(heights)
        row["storeys_seeded"] = seeded
        source = "pass2"
    if not heights:
        heights, source = list(nav_heights), source + "->navmesh(empty)"
    heights = sorted(float(h) for h in heights)
    row["storey_source"] = source
    row["union_heights"] = [round(h, 3) for h in union_heights]
    row["paste_heights"] = [round(h, 3) for h in heights]

    pasted = paste_scene(str(args.maps), scene, heights, union=True,
                         size_m=args.size_m)
    row["snapshots"] = [
        {k: v for k, v in s.items() if k != "floors"} | {
            "mapped_floors": sum(1 for f in s["floors"] if f["explored_cells"] > 0),
            "floor_y": [f["floor_y"] for f in s["floors"] if f["explored_cells"] > 0],
            "traj_on_map": [f.get("traj_on_map") for f in s["floors"]
                            if f["explored_cells"] > 0],
        }
        for s in pasted["snapshots"]
    ]
    row["union"] = {
        "n_snapshots": len(pasted["paths"]),
        "used": len(pasted["used"]),
        "skipped": pasted["skipped"],
        "cells_written": pasted["cells_written"],
        "matched_by": sorted({p["matched_by"] for p in pasted["pasted"]}),
    }
    if args.expect_layout:
        wrong = [s["file"] for s in pasted["snapshots"]
                 if s["layout_id"] and s["layout_id"] != args.expect_layout]
        if wrong:
            row["failures"].append(
                f"{len(wrong)} snapshot(s) not built on the "
                f"'{args.expect_layout}' layout: {wrong[:3]}")

    # ---- per storey
    nav_to_paste: Dict[int, Optional[int]] = {}
    for i, nh in enumerate(nav_heights):
        gaps = [abs(nh - h) for h in heights]
        j = int(np.argmin(gaps)) if gaps else -1
        nav_to_paste[i] = j if gaps and gaps[j] <= args.storey_match_m else None

    storeys: List[dict] = []
    for i, nav_storey in enumerate(nav["storeys"]):
        j = nav_to_paste[i]
        pts = nav_storey["points"]
        on_main = np.array([
            int(pathfinder.get_island(pathfinder.snap_point(p))) == main_island
            for p in pts
        ]) if len(pts) else np.zeros(0, dtype=bool)
        mine = pts[on_main][:, [0, 2]] if len(pts) else np.zeros((0, 2))
        entry = {
            "index": i,
            "height": nav_storey["height"],
            "area_m2": nav_storey["area_m2"],
            "n_points": nav_storey["n_points"],
            "off_island_frac": float(1.0 - on_main.mean()) if len(pts) else 0.0,
            "paste_storey": j,
        }
        # A storey the agent cannot WALK to is out of scope whatever the map
        # says: no mapping episode could have reached it and no episode is
        # scored on it. Measured on 00808's basement -- 44.9 m2 of navigable
        # area, 100% of it on an island disconnected from the start island.
        # Deriving this beats hand-listing the storey, which is what
        # --exempt-storey was doing and why that flag missed this check.
        entry["unreachable"] = bool(entry["off_island_frac"] >= args.off_island_out_of_scope)
        if entry["unreachable"]:
            row["notes"].append(
                f"storey {i} (y={nav_storey['height']:.2f}) is "
                f"{entry['off_island_frac']:.0%} off the start island -- "
                "out of scope, not gated")
        if j is None:
            entry["navigable"] = coverage(None, np.zeros((0, 2)))
            entry["stairs"] = {"cells": 0, "ramp_span_m": None,
                               "storey_gap_m": None, "ramp_span_frac": None}
            if not entry["unreachable"] and i not in args.exempt_storeys.get(scene, ()):
                row["failures"].append(
                    f"storey {i} (y={nav_storey['height']:.2f}) has no pasted storey "
                    f"within {args.storey_match_m} m")
            storeys.append(entry)
            continue
        costmap = pasted["costmaps"][j]
        entry["navigable"] = coverage(costmap, mine)
        gap = None
        if len(heights) > 1:
            gap = (heights[j + 1] - heights[j]) if j + 1 < len(heights) else (
                heights[j] - heights[j - 1])
        entry["stairs"] = stair_report(costmap, gap)
        entry["graph"] = (graph.get("storeys") or [{}] * len(nav_heights))[i] \
            if graph.get("present") else None
        storeys.append(entry)
    row["storeys"] = storeys

    # ---- authored targets
    layouts = read_layouts(layout_roots, scene, args.layout_types,
                           args.layout_indices)
    dists = {j: None for j in range(len(heights))}
    targets: List[dict] = []
    for layout in layouts:
        idm = layout.get("id_handle_mapping", {})
        for obj in layout.get("objects", []):
            handle = idm.get(str(obj.get("semantic_id")), "?")
            pos = np.asarray(obj["translation"], dtype=float)
            anchor = obj.get("anchor") or {}
            snapped = pathfinder.snap_point(pos.astype(np.float32))
            base = float(snapped[1]) if np.all(np.isfinite(snapped)) else float(pos[1])
            nav_i = storey_of(base, nav_heights)
            j = nav_to_paste.get(nav_i)
            item = {
                "layout_id": layout["_layout_id"],
                "handle": handle,
                "label": YCB_TARGET_LABELS.get(handle, handle),
                "position": [float(v) for v in pos],
                "base_y": base,
                "storey": nav_i,
                "storey_source": "height_rule",
                "authored_floor_index": anchor.get("floor_index"),
                "authored_floor_height": anchor.get("floor_height"),
                "anchor": anchor.get("object_id"),
                "island": int(pathfinder.get_island(snapped))
                if np.all(np.isfinite(snapped)) else None,
            }
            if j is None:
                item.update(cell="no-storey", dist_to_free_m=None, gate_ok=False)
            else:
                costmap = pasted["costmaps"][j]
                costmap.ensure_contains(pos[[0, 2]], margin_m=3.0)
                if dists.get(j) is None or dists[j].shape != costmap.grid.shape:
                    dists[j] = free_distance_m(costmap)
                rc = costmap.world_to_grid(pos[[0, 2]])
                if costmap.in_bounds(rc):
                    value = int(costmap.grid[rc[0], rc[1]])
                    cell = {FREE: "free", OCCUPIED: "occupied",
                            UNKNOWN: "unknown"}.get(value, str(value))
                    d = float(dists[j][rc[0], rc[1]])
                else:
                    cell, d = "oob", float("inf")
                item.update(cell=cell, dist_to_free_m=d,
                            gate_ok=bool(d <= args.max_target_free_m))
            item["island_ok"] = (item["island"] == main_island)
            targets.append(item)
    graph_tracks = graph_tracks_of(args.graphs, scene)
    for item in targets:
        if item["layout_id"] != "static" or graph_tracks is None:
            continue
        item["prior_track_m"] = nearest_same_label_m(item, graph_tracks)
        item["in_prior"] = bool(
            item["prior_track_m"] is not None
            and item["prior_track_m"] <= args.max_target_track_m)
    row["targets"] = targets
    return row


def graph_tracks_of(graphs_dir: Optional[Path], scene: str) -> Optional[List[dict]]:
    if graphs_dir is None:
        return None
    path = _map_path(str(graphs_dir), scene)
    if not path.exists():
        return None
    return list(json.loads(path.read_text(encoding="utf-8")).get("tracks") or [])


def nearest_same_label_m(item: dict, tracks: Sequence[dict]) -> Optional[float]:
    """Horizontal distance from an authored static pose to the nearest track of
    the same label on the same storey, or None if the graph names none.

    This is the question the multi-floor prior once failed silently: the
    occupancy covered every target and the scene graph held 0 of the 25 search
    targets at their static poses (outputs/maps_p1500_osg), so pass 2 had no
    stale belief to revise and the benchmark never exercised presence
    revision. 1.5 m vertical is "same storey" for an object on furniture."""
    pos = item["position"]
    best = None
    for track in tracks:
        if str(track.get("label", "")) != item["label"]:
            continue
        centre = track.get("center") or track.get("centre")
        if not centre or len(centre) < 3 or abs(float(centre[1]) - pos[1]) > 1.5:
            continue
        d = float(np.hypot(float(centre[0]) - pos[0], float(centre[2]) - pos[2]))
        best = d if best is None else min(best, d)
    return best


def gate(row: dict, args) -> List[str]:
    """Which thresholds this scene trips. Appended to whatever `audit_scene`
    already found structurally wrong."""
    failures = list(row.get("failures", []))
    multi = len(row.get("navmesh", {}).get("storeys", [])) > 1
    for storey in row.get("storeys", []):
        i, nav = storey["index"], storey["navigable"]
        if storey.get("paste_storey") is None:
            continue
        if i in args.exempt_storeys.get(row["scene"], ()) or storey.get("unreachable"):
            continue
        if nav["free"] < args.min_free_frac:
            failures.append(f"storey {i}: free {nav['free']:.2f} < "
                            f"{args.min_free_frac:.2f}")
        # `occupied` is REPORTED, never gated. See --max-occupied-frac: the
        # transposed map it was meant to catch read FREE 0.04 / OCC 0.04, so an
        # occupied threshold loose enough to pass a furnished storey (0.137
        # measured on a correct 00800) is far too loose to fire on 0.04 -- it
        # could never have caught the one case it existed for, while the free
        # gate catches it easily. The frame test that works is
        # scripts/check_obstacle_map_reuse.py, on the poses the agent walked.
        if multi and storey["stairs"]["cells"] < args.min_stair_cells:
            failures.append(f"storey {i}: stair cells {storey['stairs']['cells']} "
                            f"< {args.min_stair_cells}")
        frac = storey["stairs"].get("ramp_span_frac")
        if multi and frac is not None and frac < args.min_ramp_span_frac:
            failures.append(f"storey {i}: stair ramp spans {frac:.2f} of the "
                            f"storey gap < {args.min_ramp_span_frac:.2f}")
        # A storey needs SOMEWHERE for `apply_map` to put its tracks -- but the
        # graph as stored is not the whole answer, because pass 2 runs with
        # `ycb.seed_storeys_from_obstacle_map` and `_seed_storeys` ADDS a layer
        # for any obstacle-map cluster the graph lacks. So the gate is: no
        # graph floor AND no cluster to seed one from. Measured on 00808, whose
        # graph floors are 1.03 and -0.43 -- neither a real storey -- while the
        # union clusters at 0.09 and 3.04 seed both correctly.
        graph = storey.get("graph")
        if args.require_graph_floor and row.get("graph", {}).get("present"):
            has_layer = graph is not None and graph.get("floor_key") is not None
            seeded = any(abs(float(h) - storey["height"]) <= args.storey_match_m
                         for h in (row.get("union_heights") or []))
            if not has_layer and not seeded:
                failures.append(f"storey {i}: no scene-graph floor within "
                                f"{args.storey_match_m} m, and no obstacle-map "
                                "cluster to seed one from")
    if args.max_target_track_m > 0:
        missing = [t for t in row.get("targets", [])
                   if t.get("in_prior") is False and t["storey"] not in
                   args.exempt_storeys.get(row["scene"], ())]
        if missing:
            failures.append(
                f"{len(missing)} static target(s) with no same-label scene-graph "
                f"track within {args.max_target_track_m} m: "
                f"{sorted(t['handle'] for t in missing)}")
    out = [t for t in row.get("targets", [])
           if not t.get("gate_ok") and t["storey"] not in
           args.exempt_storeys.get(row["scene"], ())]
    if out:
        worst = max(out, key=lambda t: (t.get("dist_to_free_m") or 0.0))
        failures.append(
            f"{len(out)} authored target(s) beyond {args.max_target_free_m} m of "
            f"mapped free space (worst: {worst['layout_id']} {worst['handle']} "
            f"{worst.get('dist_to_free_m')})")
    return failures


# -------------------------------------------------------------------- printing

def print_scene(row: dict, args) -> None:
    scene = row["scene"]
    nav = row.get("navmesh")
    if nav is None:
        print(f"\n== {scene}   NOT CHECKED: {'; '.join(row['failures'])}")
        return
    print(f"\n== {scene}   navmesh {nav['n_storeys']} storeys, "
          f"{nav['navigable_area_m2']:.1f} m2 navigable, "
          f"{nav['off_band_frac']:.1%} of the sample on stairs, "
          f"{len(nav['islands'])} island(s)")
    print(f"   storeys: navmesh {[round(s['height'], 2) for s in nav['storeys']]}"
          f" | union {row.get('union_heights')} | pasted {row.get('paste_heights')}"
          f" ({row.get('storey_source')})")
    graph = row.get("graph", {})
    if graph.get("present"):
        print(f"   graph: {graph['n_tracks']} tracks, floors "
              f"{[None if h is None else round(float(h), 2) for h in graph['graph_floors']]}"
              f", {graph['tracks_whose_floor_key_disagrees_with_geometry']} track(s) "
              "filed under a storey further from their own centre than another")
    union = row.get("union", {})
    print(f"   snapshots: {union.get('n_snapshots')} file(s), "
          f"{union.get('used')} pasted, {len(union.get('skipped') or [])} skipped, "
          f"matched_by={union.get('matched_by')}, "
          f"{union.get('cells_written')} cells")
    for skip in union.get("skipped") or []:
        print(f"      SKIPPED {skip['path']}: {skip['why']} "
              f"({skip['mapped_floors']} mapped floors)")

    print(f"   {'st':>2} {'height':>7} {'area':>7} {'free':>6} {'occ':>6} "
          f"{'unk':>6} {'offisl':>7} {'stair':>6} {'ramp/gap':>10} "
          f"{'graph':>6} {'ycb':>4}")
    for storey in row.get("storeys", []):
        nv, stair = storey["navigable"], storey["stairs"]
        g = storey.get("graph") or {}
        frac = stair.get("ramp_span_frac")
        ramp = "-" if frac is None else (
            f"{stair['ramp_span_m']:.2f}/{stair['storey_gap_m']:.2f}")
        print(f"   {storey['index']:>2} {storey['height']:>7.2f} "
              f"{storey['area_m2']:>7.1f} {nv['free']:>6.3f} {nv['occupied']:>6.3f} "
              f"{nv['unknown']:>6.3f} {storey['off_island_frac']:>7.3f} "
              f"{stair['cells']:>6} {ramp:>10} "
              f"{str(g.get('tracks_by_centre_y', '-')):>6} "
              f"{len(g.get('ycb_tracks') or {}):>4}")

    targets = row.get("targets", [])
    if targets:
        bad = [t for t in targets if not t.get("gate_ok")]
        print(f"   targets: {len(targets)} position(s) over "
              f"{len(set(t['layout_id'] for t in targets))} layout(s); "
              f"{len(bad)} beyond {args.max_target_free_m} m of mapped free space")
        for t in sorted(bad, key=lambda t: -(t.get("dist_to_free_m") or 0))[:8]:
            d = t.get("dist_to_free_m")
            print(f"      {t['layout_id']:<16} {t['handle']:<22} storey {t['storey']} "
                  f"cell={t['cell']:<9} d_free="
                  f"{'inf' if d is None or not np.isfinite(d) else f'{d:.2f}'}")
        checked = [t for t in targets if "in_prior" in t]
        if checked:
            have = [t for t in checked if t["in_prior"]]
            print(f"   prior graph: {len(have)}/{len(checked)} static target(s) have a "
                  f"same-label track within {args.max_target_track_m} m")
            for t in checked:
                if not t["in_prior"]:
                    d = t.get("prior_track_m")
                    print(f"      missing {t['handle']:<22} storey {t['storey']} nearest "
                          f"{'none' if d is None else f'{d:.2f} m'}")
        off = [t for t in targets if not t.get("island_ok")]
        if off:
            print(f"      {len(off)} target(s) NOT on the main island: "
                  f"{sorted({t['handle'] for t in off})}")
    for note in row.get("notes", []):
        print(f"   note  {note}")
    for failure in row.get("failures", []):
        print(f"   FAIL  {failure}")


def print_header(args) -> None:
    print(f"audit_map_coverage  label={args.label or '-'}")
    print(f"  maps   {args.maps}   storeys={args.storeys}  "
          f"expect-layout={args.expect_layout or '-'}")
    print(f"  graphs {args.graphs or '-'}")
    print(f"  gates  free>={args.min_free_frac:.2f} (chosen)  "
          f"occupied REPORTED not gated  "
          f"target_free<={args.max_target_free_m:.2f} m (measured bound)  "
          f"stairs>={args.min_stair_cells} cells  "
          f"ramp>={args.min_ramp_span_frac:.2f} of gap  "
          f"graph-floor={'required' if args.require_graph_floor else 'off'}")
    print(f"  sample {args.nav_points} navigable points/scene, seed {args.seed}, "
          f"storey band {args.storey_band_m} m, matched within "
          f"{args.storey_match_m} m")
    print("  free IS coverage. occupied counts navigable floor under waist-height "
          "furniture\n  and is NOT by itself a frame check -- a transposed paste "
          "shows as free COLLAPSING.\n  traj_on_map is recorded and never gated.")
    if args.exempt_storeys:
        for scene, idx in args.exempt_storeys.items():
            print(f"  exempt {scene} storey(s) {sorted(idx)} -- declared, not silent")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", required=True,
                    help="the ycb.obstacle_map_out directory to audit")
    ap.add_argument("--graphs", default="",
                    help="the ycb.map_out directory; without it the graph "
                         "columns read '-' and their gate is skipped")
    ap.add_argument("--layout-root", action="append", default=[],
                    help="repeatable; one per dataset family")
    ap.add_argument("--scenes", default="",
                    help="comma-separated; default every scene with a snapshot")
    ap.add_argument("--scenes-dir", default=str(hm3d_scenes_dir()))
    ap.add_argument("--layout-types", default="static,cross_anchor")
    ap.add_argument("--layout-indices", default="1,2,3")
    ap.add_argument("--storeys", choices=("pass2", "union", "navmesh"),
                    default="pass2",
                    help="which storey heights the paste is told. 'pass2' "
                         "replicates load_obstacle_map + _seed_storeys, which "
                         "is what the run will actually do")
    ap.add_argument("--size-m", type=float, default=60.0)
    ap.add_argument("--nav-points", type=int, default=12000,
                    help="area-uniform navmesh sample; 12000 is about +/-0.5%% "
                         "at 1 sigma on a 0.8 fraction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--storey-band-m", type=float, default=0.75,
                    help="further than this from every storey peak is a stair "
                         "tread, counted in no storey's denominator")
    ap.add_argument("--storey-match-m", type=float, default=1.0,
                    help="MEASURED-ADJACENT: the same tolerance "
                         "ycb.storey_seed_tol_m and _match_floors use")
    ap.add_argument("--expect-layout", default="static",
                    help="fail if a snapshot was built on another layout; "
                         "pass 1 maps the static world. '' disables")
    ap.add_argument("--min-free-frac", type=float, default=0.60,
                    help="CHOSEN, not measured. For reference the 00800 union "
                         "measured 0.81/0.86 and the single best-of 0.51/0.80, "
                         "so 0.60 fails a map the union did not rescue")
    ap.add_argument("--max-occupied-frac", type=float, default=0.08,
                    help="REPORTED, NOT GATED, and kept only to print the "
                         "number. Twice mis-calibrated before that was "
                         "understood: 0.02 read off the log's 'transposed' "
                         "figure failed a correct map at 0.040, and 0.08 then "
                         "failed a correct 1500-step map at 0.137. The reason "
                         "no threshold works is that the transposed map read "
                         "FREE 0.04 / OCC 0.04 -- its occupied fraction was "
                         "LOWER than a healthy furnished storey's, so no cutoff "
                         "separates them. What collapses under a transposition "
                         "is FREE (0.80 -> 0.04), which --min-free-frac catches. "
                         "Occupied counts navmesh-navigable floor under "
                         "waist-height furniture and rises with observation: on "
                         "00800's lower storey 0.000 at 500 steps and 0.137 at "
                         "1500, while KNOWN coverage rose 0.802 -> 0.887")
    ap.add_argument("--max-target-free-m", type=float, default=1.5,
                    help="MEASURED as a bound: the 00800 union put all 12 "
                         "authored positions within 1.2 m; before it, one was "
                         "5.1 m outside")
    ap.add_argument("--max-target-track-m", type=float, default=1.0,
                    help="every STATIC target needs a same-label scene-graph track "
                         "this close (horizontal, same storey), or pass 2 has no "
                         "stale belief to revise. The success radius. <= 0 disables; "
                         "skipped without --graphs")
    ap.add_argument("--min-stair-cells", type=int, default=50,
                    help="CHOSEN, order-of-magnitude: 00800 holds ~933")
    ap.add_argument("--min-ramp-span-frac", type=float, default=0.60,
                    help="CHOSEN, from the measured failure: a ramp cut at 0.53 "
                         "of the gap left the climber no waypoint and it drove "
                         "into the banister")
    ap.add_argument("--require-graph-floor", action="store_true", default=True)
    ap.add_argument("--no-require-graph-floor", dest="require_graph_floor",
                    action="store_false")
    ap.add_argument("--off-island-out-of-scope", type=float, default=0.9,
                    help="a storey with at least this fraction of its navigable "
                         "area off the START island is out of scope and not "
                         "gated -- no mapping episode can reach it and no "
                         "episode is scored on it (00808's basement measured "
                         "1.00). Derived, so it does not need declaring")
    ap.add_argument("--exempt-storey", action="append", default=[],
                    metavar="SCENE:INDEX",
                    help="declare a storey out of scope, e.g. an empty basement "
                         "no authored target uses. Printed in the header")
    ap.add_argument("--no-gate", action="store_true",
                    help="report only; always exit 0")
    ap.add_argument("--out", default="", help="write the JSON report here")
    ap.add_argument("--label", default="", help="names this run inside the JSON")
    args = ap.parse_args()

    args.maps = Path(args.maps)
    args.graphs = Path(args.graphs) if args.graphs else None
    args.scenes_dir = Path(args.scenes_dir)
    args.layout_types = [s for s in args.layout_types.split(",") if s]
    args.layout_indices = [int(s) for s in args.layout_indices.split(",") if s]
    exempt: Dict[str, set] = {}
    for spec in args.exempt_storey:
        scene, _, index = spec.partition(":")
        exempt.setdefault(scene, set()).add(int(index))
    args.exempt_storeys = exempt

    layout_roots = [Path(r) for r in args.layout_root]
    if args.scenes:
        scenes = [s for s in args.scenes.split(",") if s]
    else:
        seen = set()
        for path in sorted(args.maps.glob("*.json")):
            seen.add(path.stem.split("__", 1)[0])
        scenes = sorted(seen)
    if not scenes:
        print(f"NOTHING WAS CHECKED: no snapshots under {args.maps}")
        return EXIT_NOTHING

    print_header(args)
    rows = []
    for scene in scenes:
        row = audit_scene(scene, args, layout_roots)
        if row.get("verdict") != "not-checked":
            row["failures"] = gate(row, args)
            row["verdict"] = "fail" if row["failures"] else "pass"
        print_scene(row, args)
        rows.append(row)

    print("\n" + "=" * 78)
    print(f"{'scene':<22}{'nav':>4}{'paste':>6}{'snaps':>6}{'skip':>5}"
          f"{'worst_free':>11}{'worst_occ':>10}{'targets_out':>12}  verdict")
    for row in rows:
        nav = row.get("navmesh") or {}
        storeys = [s for s in row.get("storeys", [])
                   if s.get("paste_storey") is not None]
        free = min((s["navigable"]["free"] for s in storeys), default=float("nan"))
        occ = max((s["navigable"]["occupied"] for s in storeys), default=float("nan"))
        targets = row.get("targets", [])
        out = sum(1 for t in targets if not t.get("gate_ok"))
        union = row.get("union", {})
        print(f"{row['scene']:<22}{nav.get('n_storeys', 0):>4}"
              f"{len(row.get('paste_heights') or []):>6}"
              f"{union.get('n_snapshots', 0):>6}{len(union.get('skipped') or []):>5}"
              f"{free:>11.3f}{occ:>10.3f}{f'{out}/{len(targets)}':>12}  "
              f"{row.get('verdict', '?').upper()}")

    failed = [r for r in rows if r.get("verdict") == "fail"]
    nothing = [r for r in rows if r.get("verdict") == "not-checked"]
    print(f"\nRESULT  {len(rows)} scene(s): "
          f"{sum(1 for r in rows if r.get('verdict') == 'pass')} pass, "
          f"{len(failed)} FAIL, {len(nothing)} not checked")
    for row in failed + nothing:
        for failure in row["failures"][:4]:
            print(f"  {row['scene']}  {failure}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "tool": "audit_map_coverage", "version": 1, "label": args.label,
            "maps": str(args.maps), "graphs": str(args.graphs or ""),
            "layout_roots": [str(r) for r in layout_roots],
            "storey_source": args.storeys,
            "gates": {
                "min_free_frac": args.min_free_frac,
                "max_occupied_frac": args.max_occupied_frac,
                "max_target_free_m": args.max_target_free_m,
                "min_stair_cells": args.min_stair_cells,
                "min_ramp_span_frac": args.min_ramp_span_frac,
                "require_graph_floor": args.require_graph_floor,
                "storey_match_m": args.storey_match_m,
                "expect_layout": args.expect_layout,
                "exempt_storeys": {k: sorted(v) for k, v in exempt.items()},
            },
            "sampling": {"n_points": args.nav_points, "seed": args.seed,
                         "storey_band_m": args.storey_band_m},
            "scenes": rows,
        }
        out_path.write_text(json.dumps(report, indent=2, default=str) + "\n",
                            encoding="utf-8")
        print(f"\nwrote {out_path}")

    if args.no_gate:
        return EXIT_OK
    if nothing:
        return EXIT_NOTHING
    return EXIT_GATE if failed else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
