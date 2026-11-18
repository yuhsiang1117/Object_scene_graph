#!/usr/bin/env python3
"""After a YCB object moves, which prior actually goes stale?

Re-running pass 1 costs ~14 minutes an episode. Before paying that for a layout
edit, this answers the question from geometry: can the move change ASCENT's
obstacle map at all, and does the scene graph hold a belief about the object
that is now wrong?

Three facts do most of the work, and the probe MEASURES each rather than
assuming it:

  1. Pass 1 maps the STATIC layout only (`mf5_ascentnav_map` sets
     `layout_types: [static]`) and both artifacts key on SCENE, not layout
     (`prior_map._map_path`). So a cross_anchor edit cannot invalidate either
     one by construction -- and "rebuild the obstacle map" can only ever mean
     re-running pass 1 on the STATIC layout. Mapping the moved world would make
     pass 2 an incomplete-map benchmark instead of a dynamic-scene one.

  2. ASCENT stamps obstacles only between `agent.ascent_min_obstacle_h` and
     `ascent_max_obstacle_h` (0.61-0.88 m) and the band is RELATIVE to the
     storey the agent stands on, not absolute: `geometry.tf_camera_to_episodic`
     puts the camera at a constant `z = agent.camera_height`, so the episodic z
     that `filter_points_by_height` thresholds is height above the agent's feet.
     The test is therefore `y_obj - storey_floor_y` in [0.61, 0.88], per storey.
     OSG's own band is 0.15-1.50 m, which is why a tabletop YCB object is
     visible to OSG's map and mostly invisible to ASCENT's.

  3. The episode manifest self-invalidates: `manifest_cache_key` hashes
     `layout_sha256` (`osg/sim/ycb_env.py`). Reported as a fact, never as a
     decision.

WHAT IT CANNOT ANSWER, stated because the verdict is easy to over-read: it says
whether the map COULD differ, not whether the difference would change a plan.
The `infl` column is the cheap proxy -- a cell the planner already refuses to
route through cannot be made worse -- and `ring` is a proxy for "the support
furniture already accounts for this cell", not a proof. The only proof is
re-running pass 1, which is the expense being avoided.

    python scripts/probe_remap_need.py --scene 00878-XB4GS9ShBRE \\
        --maps outputs/maps_p1500_ascent --graphs outputs/maps_p1500_osg
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osg.core.config.ycb import YCB_TARGET_LABELS  # noqa: E402
from osg.core.paths import collector_data_root, hm3d_scenes_dir  # noqa: E402
from osg.eval.floors import (  # noqa: E402
    load_pathfinder, navmesh_path_for, navmesh_storeys,
)
from osg.eval.prior_map import _map_path, paste_scene, storeys_from_snapshots  # noqa: E402
from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN  # noqa: E402

MF5 = ("00800-TEEsavR23oF", "00808-y9hTuugGdiq", "00821-eF36g7L6Z9M",
       "00873-bxsVRursffK", "00878-XB4GS9ShBRE")


# ------------------------------------------------------------ object geometry

def glb_bounds(path: Path):
    """Local-frame AABB from a .glb's glTF POSITION accessors.

    Read straight out of the JSON chunk rather than loaded through habitat:
    every accessor for a POSITION attribute carries `min`/`max`, so the AABB is
    exact and needs no Simulator, no GL and no mesh library (there is no
    trimesh in this environment).
    """
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        raise ValueError(f"not a binary glTF: {path}")
    length, = struct.unpack_from("<I", raw, 8)
    offset, meta = 12, None
    while offset < min(length, len(raw)):
        chunk_len, chunk_type = struct.unpack_from("<II", raw, offset)
        body = raw[offset + 8: offset + 8 + chunk_len]
        if chunk_type == 0x4E4F534A:  # 'JSON'
            meta = json.loads(body.decode("utf-8"))
            break
        offset += 8 + chunk_len + ((4 - chunk_len % 4) % 4)
    if meta is None:
        raise ValueError(f"no JSON chunk in {path}")
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    accessors = meta.get("accessors", [])
    for mesh in meta.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            index = primitive.get("attributes", {}).get("POSITION")
            if index is None:
                continue
            accessor = accessors[index]
            if "min" in accessor and "max" in accessor:
                lo = np.minimum(lo, np.asarray(accessor["min"], dtype=float))
                hi = np.maximum(hi, np.asarray(accessor["max"], dtype=float))
    if not np.all(np.isfinite(lo)):
        raise ValueError(f"no POSITION bounds in {path}")
    return lo, hi


def semantic_frame_R(config: dict) -> np.ndarray:
    """Rotation from the asset's own (up, front) to habitat's y-up / -z-front.

    Rows are [right, up, -front] with right = front x up. Every YCB config in
    this dataset declares up=(0,0,1), front=(0,1,0), so the asset's z is
    habitat's y.

    Load-bearing, and checked against published YCB dimensions rather than
    asserted. Standing height, rotated vs raw vs published (m):

        003_cracker_box      0.213  0.164  0.213
        005_tomato_soup_can  0.102  0.068  0.101
        019_pitcher_base     0.242  0.145  0.235
        024_bowl             0.055  0.161  0.055
        002_master_chef_can  0.140  0.102  0.139
        006_mustard_bottle   0.191  0.067  0.195
        011_banana           0.037  0.178  0.036

    Rotated is within 1-7 mm everywhere; raw is wrong by 2-5x and, for the bowl
    and the banana, wrong in the direction that would put a flat object across
    ASCENT's whole 0.27 m band.
    """
    up = np.asarray(config.get("up", [0.0, 1.0, 0.0]), dtype=float)
    front = np.asarray(config.get("front", [0.0, 0.0, -1.0]), dtype=float)
    up = up / (np.linalg.norm(up) or 1.0)
    front = front / (np.linalg.norm(front) or 1.0)
    right = np.cross(front, up)
    right = right / (np.linalg.norm(right) or 1.0)
    return np.vstack([right, up, -front])


def quat_matrix(xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in xyzw]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def world_aabb(translation, rotation_xyzw, handle: str, objects_dir: Path):
    config_path = objects_dir / f"{handle}.object_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    asset = (config_path.parent / config["render_asset"]).resolve()
    lo, hi = glb_bounds(asset)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    corners = corners @ semantic_frame_R(config).T
    corners = corners @ quat_matrix(rotation_xyzw).T
    corners = corners + np.asarray(translation, dtype=float)
    return corners.min(axis=0), corners.max(axis=0)


def object_band(obj: dict, handle: str, objects_dir: Path, pad_m: float):
    """`(y_lo, y_hi, source)` -- the height interval the object occupies.

    When the layout records `anchor.top_y` the BASE is snapped to it and only
    the HEIGHT comes from the mesh: the authoring placed the object's base on
    that surface, so its number is exact by construction, while the mesh AABB
    measured +0.00 to +0.14 m above it across the five multi-floor scenes.
    Released-format layouts carry no `top_y` and fall back to the raw AABB,
    which the `source` column names so a marginal verdict is visibly marginal.
    """
    lo, hi = world_aabb(obj["translation"], obj["rotation"], handle, objects_dir)
    top_y = (obj.get("anchor") or {}).get("top_y")
    if top_y is not None:
        height = float(hi[1] - lo[1])
        base = float(top_y)
        return base - pad_m, base + height + pad_m, "top_y"
    return float(lo[1]) - pad_m, float(hi[1]) + pad_m, "aabb"


def band_overlap_m(y_lo, y_hi, floor_y, lo_m, hi_m) -> float:
    """Length of the object's height interval inside ASCENT's stamping band."""
    band_lo, band_hi = floor_y + lo_m, floor_y + hi_m
    return max(0.0, min(y_hi, band_hi) - max(y_lo, band_lo))


# ---------------------------------------------------------------- map queries

def ring_occupied_frac(costmap, xz, r_in_m: float, r_out_m: float) -> float:
    """Occupied fraction of an annulus around a point.

    A YCB object is under 0.2 m across -- four cells at 0.05 m -- while the
    furniture holding it is metres wide. So a predominantly OCCUPIED ring says
    the SUPPORT is what fills these cells, which the centre cell alone cannot
    tell you: in the static map the object's own stamp is already there.

    The inner radius therefore has to clear the object itself. At 0.10 m it did
    not: an object standing entirely alone on free floor read 22% occupied,
    because its own two-cell stamp touches the ring. See `--ring-in-m`.
    """
    rc = costmap.world_to_grid(np.asarray(xz, dtype=float))
    # round, not truncate: 0.15 / 0.05 is 2.9999... in binary floating point,
    # so int() gave a 2-cell inner radius for a 0.15 m request and the ring
    # swallowed the object it was meant to exclude.
    r_in = int(round(r_in_m / costmap.resolution))
    r_out = int(round(r_out_m / costmap.resolution))
    h, w = costmap.grid.shape
    r0, r1 = max(0, rc[0] - r_out), min(h, rc[0] + r_out + 1)
    c0, c1 = max(0, rc[1] - r_out), min(w, rc[1] + r_out + 1)
    if r0 >= r1 or c0 >= c1:
        return 0.0
    rows, cols = np.ogrid[r0:r1, c0:c1]
    dist = np.hypot(rows - rc[0], cols - rc[1])
    ring = (dist >= r_in) & (dist <= r_out)
    if not ring.any():
        return 0.0
    patch = costmap.grid[r0:r1, c0:c1][ring]
    known = patch != UNKNOWN
    if not known.any():
        return 0.0
    return float(np.mean(patch[known] == OCCUPIED))


def nearest_track(tracks, label: str, xz, y, heights, match_m: float):
    """The prior's nearest track with this label, on this object's storey.

    Bucketed by the track's OWN `center[1]`, never by `track.floor_key`:
    measured on 00808, the prior's single YCB target track is filed under a
    phantom 1.03 m storey while its centre sits at 0.56.
    """
    best, best_d = None, float("inf")
    for track in tracks:
        if str(track.get("label", "")) != label:
            continue
        centre = track.get("center") or track.get("centre")
        if not centre or len(centre) < 3:
            continue
        d = float(np.hypot(centre[0] - xz[0], centre[2] - xz[1]))
        if d < best_d:
            best, best_d = track, d
    if best is None:
        return None, None
    return (best, best_d) if best_d <= match_m else (best, best_d)


# --------------------------------------------------------------------- verdict

def probe_scene(scene: str, args) -> dict:
    row: dict = {"scene": scene, "objects": [], "notes": []}
    root = Path(args.layout_root)
    static_path = root / scene / "static_scene_config.json"
    moved_path = (root / scene / "dynamic_scene_config" / args.layout_type
                  / f"layout_{args.layout_index:02d}.json")
    if not static_path.exists() or not moved_path.exists():
        row["error"] = f"missing layout ({static_path} / {moved_path})"
        return row
    static = json.loads(static_path.read_text(encoding="utf-8"))
    moved = json.loads(moved_path.read_text(encoding="utf-8"))
    row["layout_id"] = f"{args.layout_type}_{args.layout_index:02d}"

    pathfinder = load_pathfinder(navmesh_path_for(scene, args.scenes_dir))
    if pathfinder is None:
        row["error"] = "no navmesh / habitat_sim unavailable"
        return row
    nav = navmesh_storeys(pathfinder, n_points=args.nav_points, seed=args.seed)
    nav_heights = [s["height"] for s in nav["storeys"]]

    heights = storeys_from_snapshots(
        [p for p in Path(args.maps).glob(f"{scene}*.json")]) if args.maps else []
    if not heights:
        heights = list(nav_heights)
    heights = sorted(float(h) for h in heights)
    row["storeys"] = [round(h, 3) for h in heights]

    pasted = paste_scene(str(args.maps), scene, heights, union=True) if args.maps \
        else {"costmaps": []}
    inflated = []
    for costmap in pasted.get("costmaps", []):
        try:
            inflated.append(costmap.inflated(args.agent_radius_m))
        except Exception:
            inflated.append(None)

    tracks = []
    if args.graphs:
        graph_path = _map_path(str(args.graphs), scene)
        if graph_path.exists():
            tracks = json.loads(graph_path.read_text(encoding="utf-8")).get(
                "tracks") or []
    row["graph_tracks"] = len(tracks)

    objects_dir = Path(args.objects_dir)
    idm = static.get("id_handle_mapping", {})
    by_id = {int(o["semantic_id"]): o for o in moved.get("objects", [])}

    def look(obj, label_handle):
        pos = np.asarray(obj["translation"], dtype=float)
        y_lo, y_hi, source = object_band(obj, label_handle, objects_dir,
                                         args.aabb_pad_m)
        base = (obj.get("anchor") or {}).get("top_y")
        base = float(base) if base is not None else float(y_lo)
        # the storey the object STANDS on: highest floor at or below its base
        below = [i for i, h in enumerate(heights) if base >= h - 0.15]
        st = max(below, key=lambda i: heights[i]) if below else 0
        overlap = band_overlap_m(y_lo, y_hi, heights[st],
                                 args.ascent_band[0], args.ascent_band[1])
        info = {"xz": [float(pos[0]), float(pos[2])], "storey": st,
                "y_lo": y_lo, "y_hi": y_hi, "band_source": source,
                "band_overlap_m": float(overlap),
                "anchor": (obj.get("anchor") or {}).get("object_id")}
        if st < len(pasted.get("costmaps", [])):
            costmap = pasted["costmaps"][st]
            rc = costmap.world_to_grid(pos[[0, 2]])
            if costmap.in_bounds(rc):
                value = int(costmap.grid[rc[0], rc[1]])
                info["cell"] = {FREE: "free", OCCUPIED: "occupied",
                                UNKNOWN: "unknown"}.get(value, str(value))
                mask = inflated[st] if st < len(inflated) else None
                info["inflated"] = (bool(mask[rc[0], rc[1]])
                                    if mask is not None else None)
            else:
                info["cell"], info["inflated"] = "oob", None
            info["ring_occupied_frac"] = ring_occupied_frac(
                costmap, pos[[0, 2]], args.ring_in_m, args.ring_out_m)
        else:
            info.update(cell="no-map", inflated=None, ring_occupied_frac=None)
        return info

    for obj in static.get("objects", []):
        handle = idm.get(str(obj["semantic_id"]), "?")
        dest_obj = by_id.get(int(obj["semantic_id"]))
        if dest_obj is None:
            row["notes"].append(f"{handle} is not in the moved layout")
            continue
        origin = look(obj, handle)
        dest = look(dest_obj, handle)

        label = YCB_TARGET_LABELS.get(handle, handle)
        track, track_d = nearest_track(tracks, label, origin["xz"],
                                       origin["y_lo"], heights,
                                       args.track_match_m)
        # The destination is the case that hurts: the map says traversable
        # where an obstacle now stands. The origin is the mirror: the map says
        # blocked where nothing stands any more, and the ring says whether the
        # support furniture already accounts for those cells.
        dest_differs = bool(
            dest["band_overlap_m"] > 0 and dest.get("cell") == "free"
            and not dest.get("inflated"))
        origin_differs = bool(
            origin["band_overlap_m"] > 0 and origin.get("cell") == "occupied"
            and (origin.get("ring_occupied_frac") or 0.0) < args.ring_thresh)
        row["objects"].append({
            "handle": handle, "label": label,
            "origin": origin, "destination": dest,
            "map_differs": dest_differs or origin_differs,
            "why": ("destination on FREE inside the band" if dest_differs else
                    "origin on OCCUPIED the furniture does not explain"
                    if origin_differs else "band never enters ASCENT's stamp"),
            "track_dist_m": track_d,
            "graph": ("stale" if track is not None and track_d is not None
                      and track_d <= args.track_match_m else
                      "mislocated" if track is not None else "absent"),
        })

    differing = [o for o in row["objects"] if o["map_differs"]]
    tracked = [o for o in row["objects"] if o["graph"] == "stale"]
    n = max(1, len(row["objects"]))
    row["verdict"] = {
        "obstacle_map": "rebuild" if differing else "reuse",
        "scene_graph": ("rebuild" if len(tracked) / n < args.min_tracked_frac
                        else "reuse"),
        "n_differing": len(differing),
        "n_tracked": len(tracked),
        "n_objects": len(row["objects"]),
    }
    return row


def print_scene(row: dict, args) -> None:
    if row.get("error"):
        print("\n== %s   SKIPPED: %s" % (row["scene"], row["error"]))
        return
    print("\n== %s  %s   storeys %s   graph tracks %d"
          % (row["scene"], row["layout_id"], row["storeys"],
             row.get("graph_tracks", 0)))
    print("  %-22s%22s%22s%7s%7s%10s%10s%6s%5s%8s  verdict"
          % ("object", "from", "to", "ov_f", "ov_t", "cell_f", "cell_t",
             "ring", "infl", "track"))
    for o in row["objects"]:
        f, t = o["origin"], o["destination"]
        ring = f.get("ring_occupied_frac")
        td = o.get("track_dist_m")
        where_f = "s%d %.2f-%.2f (%s)" % (f["storey"], f["y_lo"], f["y_hi"],
                                          f["band_source"])
        where_t = "s%d %.2f-%.2f (%s)" % (t["storey"], t["y_lo"], t["y_hi"],
                                          t["band_source"])
        infl = "-" if f.get("inflated") is None else ("y" if f["inflated"] else "n")
        print("  %-22s%22s%22s%7.3f%7.3f%10s%10s%6s%5s%8s  map %-7s graph %s"
              % (o["handle"], where_f, where_t,
                 f["band_overlap_m"], t["band_overlap_m"],
                 str(f.get("cell")), str(t.get("cell")),
                 "-" if ring is None else "%.2f" % ring,
                 infl,
                 "-" if td is None else "%.1fm" % td,
                 "DIFFERS" if o["map_differs"] else "SAME",
                 o["graph"].upper()))
    v = row["verdict"]
    print("  obstacle map : %s -- %d of %d object(s) could change it"
          % (v["obstacle_map"].upper(), v["n_differing"], v["n_objects"]))
    print("  scene graph  : %s -- %d of %d target(s) carry a prior track "
          "(threshold %.2f)"
          % (v["scene_graph"].upper(), v["n_tracked"], v["n_objects"],
             args.min_tracked_frac))
    for note in row.get("notes", []):
        print("  note  %s" % note)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", action="append", default=[],
                    help="repeatable; default the five multi-floor scenes")
    ap.add_argument("--layout-root",
                    default="/habitat-data-collector/outputs/dualmap_multifloor")
    ap.add_argument("--layout-type", default="cross_anchor",
                    choices=("in_anchor", "cross_anchor"))
    ap.add_argument("--layout-index", type=int, default=1)
    ap.add_argument("--maps", default="outputs/maps_p1500_ascent")
    ap.add_argument("--graphs", default="outputs/maps_p1500_osg")
    ap.add_argument("--scenes-dir", default=str(hm3d_scenes_dir()))
    ap.add_argument("--objects-dir",
                    default=str(collector_data_root() / "objects/ycb/configs"))
    ap.add_argument("--ascent-band", default="0.61,0.88",
                    help="agent.ascent_min/max_obstacle_h, ABOVE THE STOREY")
    ap.add_argument("--aabb-pad-m", type=float, default=0.05)
    ap.add_argument("--agent-radius-m", type=float, default=0.18)
    ap.add_argument("--ring-in-m", type=float, default=0.15,
                    help="MEASURED: the widest YCB target here is 0.164 m "
                         "across, so its own stamp reaches 0.08 m and at 0.10 m "
                         "the ring was reading the OBJECT as its own support "
                         "(22%% occupied for an object standing alone). 0.15 m "
                         "is three cells at the 0.05 m resolution and clears it")
    ap.add_argument("--ring-out-m", type=float, default=0.35)
    ap.add_argument("--ring-thresh", type=float, default=0.5,
                    help="above this, the support furniture accounts for the cell")
    ap.add_argument("--track-match-m", type=float, default=1.5)
    ap.add_argument("--min-tracked-frac", type=float, default=0.5)
    ap.add_argument("--nav-points", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    args.scenes_dir = Path(args.scenes_dir)
    args.ascent_band = [float(v) for v in args.ascent_band.split(",")]
    scenes = args.scene or list(MF5)

    print("probe_remap_need")
    print(f"  ASCENT stamps {args.ascent_band[0]}-{args.ascent_band[1]} m ABOVE "
          "THE STOREY THE AGENT STANDS ON (episodic z), not above sea level.")
    print(f"  OSG maps 0.15-1.50 m, which is why a tabletop object is in its band "
          "and mostly out of ASCENT's.")
    print(f"  AABB pad {args.aabb_pad_m} m; ring {args.ring_in_m}-{args.ring_out_m} m, "
          f"occupied>{args.ring_thresh} means the support explains the cell.")
    print("  Pass 1 maps the STATIC layout and its artifacts key on SCENE, so a "
          "cross_anchor edit\n  invalidates neither by construction, and "
          "'rebuild' can only mean re-running pass 1 on static.")
    print("  The episode manifest self-invalidates: manifest_cache_key hashes "
          "layout_sha256.")

    rows = [probe_scene(scene, args) for scene in scenes]
    for row in rows:
        print_scene(row, args)
    print("\n  Limits: this says whether the map COULD differ, from geometry. It "
          "cannot say whether\n  the difference would change a plan -- 'infl' is "
          "the proxy for that, and 'ring' is a proxy\n  for 'the furniture "
          "already explains this cell', not a proof.")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"tool": "probe_remap_need", "version": 1,
             "bands": {"ascent": args.ascent_band, "osg": [0.15, 1.5]},
             "scenes": rows}, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
