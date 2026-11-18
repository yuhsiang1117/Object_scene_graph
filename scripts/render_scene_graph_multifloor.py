#!/usr/bin/env python3
"""Cross-floor scene graph: the hierarchy on the left, the building on the right.

The right panel is an orthographic *side* section of the HM3D mesh, taken with
the camera on the building's mid-plane so the near half is clipped away and
both storeys are visible in cut-away, staircase included.  The left panel hangs
the map's own data structure off it -- building -> floor -> room -> container
-> object -- with each floor node drawn at the height of the storey it stands
for, so the two halves read against each other.

Only schema-2 maps carry per-floor grids; single-floor maps will render one
floor band.

    python scripts/render_scene_graph_multifloor.py --scene 00808-y9hTuugGdiq
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

WORKSPACE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from osg.core.paths import hm3d_scene_root
from osg.graph.map_store import load_map, _track_from_record
from osg.graph.scene_graph import SceneGraph
from osg.mapping.costmap import Costmap2D, PLANE
from osg.objects.object_layer import ObjectLayer

C_BUILDING = "#3b3f8f"
C_FLOOR = "#0f7f8f"
C_ROOM = "#1f8a2f"
C_CONTAINER = "#b8471f"
C_OBJECT = "#d81f26"
C_EDGE = "#8a8a8a"
FLOOR_TINT = ["#e8f2f4", "#f6efe4", "#eef0f8"]


# --------------------------------------------------------------------- graph

def load_multifloor(scene: str, maps_root: pathlib.Path):
    path = maps_root / scene / f"{scene}.json"
    if not path.exists():
        path = maps_root / f"{scene}.json"
    if not path.exists():
        raise SystemExit(f"no saved map for {scene} under {maps_root}")
    blob = load_map(path)
    grids = blob["_grids"]

    layer = ObjectLayer()
    for rec in blob["tracks"]:
        track = _track_from_record(rec)
        layer._tracks[track.id] = track
    layer._next_id = int(blob["next_track_id"])

    floors = blob.get("floors")
    if not floors:  # schema 1: one unnamed storey
        floors = [{"key": 0, "height_y": 0.0, "prefix": "", "resolution": blob["resolution"]}]

    # `containers.DEFAULT_TOP_H_M` is an ABSOLUTE world-height band (0.2-1.4 m)
    # and `top_height` returns an absolute y, so on a storey at y = 2.86 every
    # desk top lands at ~3.6 m and nothing qualifies: measured on this map, all
    # 132 support surfaces came out on floor 0 and none on floor 1.  That is a
    # pipeline bug, not a drawing one.  Until it is fixed upstream the figure
    # rebuilds each storey with the band offset to its own floor, which is what
    # the rule was always meant to express.
    graph = SceneGraph()
    costmaps = {}
    from osg.graph import containers as containers_mod
    lo_h, hi_h = containers_mod.DEFAULT_TOP_H_M
    for spec in floors:
        key = int(spec["key"])
        prefix = spec.get("prefix", "")
        height = float(spec["height_y"])
        grid = grids[f"{prefix}grid"]
        res = float(spec.get("resolution", blob["resolution"]))
        costmap = Costmap2D(resolution=res, size_m=grid.shape[0] * res)
        costmap.grid = grid
        costmap.origin = np.asarray(grids[f"{prefix}origin"], dtype=float)
        costmaps[key] = costmap

        per_floor = SceneGraph(container_top_h_m=(lo_h + height, hi_h + height))
        per_floor.rebuild_floor(grids[f"{prefix}room_labels"], costmap, layer,
                                floor_key=key, floor_height=height)
        graph.rooms.update(per_floor.rooms)
        graph.containers.update(per_floor.containers)
        graph.objects.extend(per_floor.objects)
        graph.floors.update(per_floor.floors)
    return blob, grids, costmaps, layer, graph, floors


# ---------------------------------------------------------------- side view

def render_side(scene: str, res: int, pad_m: float, frac: float,
                cache: pathlib.Path):
    """Orthographic section through the building, cropped to its own ink.

    Returns (rgb, (x0, x1, y0, y1)) with the world extents of the crop: world
    x along the image columns, world height along the rows (screen up is +y).
    """
    if cache.exists():
        blob = np.load(cache)
        return blob["img"], tuple(float(v) for v in blob["extent"])

    import habitat_sim

    root = pathlib.Path(hm3d_scene_root())
    stem = scene.split("-", 1)[1]
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(root / f"val/{scene}/{stem}.basis.glb")
    cfg.scene_dataset_config_file = str(
        root / "hm3d_annotated_basis.scene_dataset_config.json")
    cfg.enable_physics = False

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    sensor.resolution = [res, res]
    sensor.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    sensor.orientation = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # along -Z
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]

    sensor.ortho_scale = 0.05
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
    lo, hi = sim.pathfinder.get_bounds()
    sim.close()

    # The ORTHOGRAPHic sensor covers 1/ortho_scale metres across a square frame.
    span = max(float(hi[0] - lo[0]), float(hi[1] - lo[1])) + 2 * pad_m
    sensor.ortho_scale = 1.0 / span
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
    cx = float((lo[0] + hi[0]) / 2)
    cy = float((lo[1] + hi[1]) / 2)
    cz = float(lo[2] + (hi[2] - lo[2]) * frac)
    state = habitat_sim.AgentState()
    state.position = np.array([cx, cy, cz], dtype=np.float32)
    state.rotation = np.quaternion(1, 0, 0, 0)
    sim.get_agent(0).set_state(state)
    img = np.asarray(sim.get_sensor_observations()["rgb"][..., :3], dtype=np.uint8)
    sim.close()

    ink = np.nonzero(img.sum(axis=2) > 16)
    if ink[0].size == 0:
        raise SystemExit("section render is empty; try a different --section-frac")
    r0, r1 = int(ink[0].min()), int(ink[0].max())
    c0, c1 = int(ink[1].min()), int(ink[1].max())
    crop = img[r0:r1 + 1, c0:c1 + 1]

    mpp = span / res
    x0 = cx + (c0 - res / 2) * mpp
    x1 = cx + (c1 + 1 - res / 2) * mpp
    # Screen up is +y, so the top row is the HIGHEST world y.
    y1 = cy - (r0 - res / 2) * mpp
    y0 = cy - (r1 + 1 - res / 2) * mpp
    extent = (x0, x1, y0, y1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, img=crop, extent=np.array(extent))
    return crop, extent


# --------------------------------------------------------------------- tree

def build_tree(graph, layer, floors, *, max_rooms, max_containers, max_objects):
    """Trim the hierarchy to what a page can hold, largest-first at each level."""
    tree = []
    for spec in sorted(floors, key=lambda s: float(s["height_y"])):
        key = int(spec["key"])
        rooms = [r for r in graph.rooms.values() if r.floor_id == key]
        rooms.sort(key=lambda r: r.n_cells, reverse=True)
        room_entries = []
        for room in rooms[:max_rooms]:
            conts = [c for c in graph.containers.values() if c.room_id == room.id]
            conts.sort(key=lambda c: c.area_m2, reverse=True)
            cont_entries = []
            for cont in conts[:max_containers]:
                objs = [o for o in graph.objects if o.track_id in set(cont.object_ids)]
                objs.sort(key=lambda o: float(
                    getattr(layer._tracks.get(o.track_id), "evidence", 0.0)), reverse=True)
                cont_entries.append((cont, objs[:max_objects]))
            room_entries.append((room, cont_entries))
        tree.append((key, float(spec["height_y"]), room_entries))
    return tree


def leaf_count(tree):
    n = 0
    for _, _, rooms in tree:
        for _, conts in rooms:
            for _, objs in conts:
                n += max(1, len(objs))
    return max(n, 1)


def draw_tree(ax, tree, floor_y_of):
    """Columns building | floor | room | container | object, rows by leaf order."""
    X = {"building": 0.04, "floor": 0.26, "room": 0.48, "container": 0.70, "object": 0.92}

    # Lay leaves out inside each floor's own vertical band so the floor node can
    # sit at the storey's real height without dragging its subtree off-panel.
    bands = {}
    n_floors = len(tree)
    for i, (key, _, _) in enumerate(tree):
        lo = 0.06 + i * (0.88 / n_floors)
        bands[key] = (lo, lo + 0.88 / n_floors - 0.06)

    floor_pts = {}
    for key, height, rooms in tree:
        lo, hi = bands[key]
        leaves = sum(max(1, sum(max(1, len(objs)) for _, objs in conts))
                     for _, conts in rooms) or 1
        step = (hi - lo) / leaves
        cursor = lo + step / 2
        room_pts = []
        for room, conts in rooms:
            cont_pts = []
            for cont, objs in conts:
                obj_pts = []
                for obj in objs:
                    obj_pts.append(cursor)
                    cursor += step
                if not objs:
                    obj_pts.append(cursor)
                    cursor += step
                cy = float(np.mean(obj_pts))
                cont_pts.append((cont, cy, list(zip(objs, obj_pts))))
            if cont_pts:
                ry = float(np.mean([c[1] for c in cont_pts]))
            else:
                ry = cursor
                cursor += step
            room_pts.append((room, ry, cont_pts))
        fy = float(np.mean([r[1] for r in room_pts])) if room_pts else (lo + hi) / 2
        floor_pts[key] = (fy, height, room_pts)

        ax.add_patch(FancyBboxPatch(
            (0.015, lo - 0.028), 0.97, (hi - lo) + 0.056,
            boxstyle="round,pad=0.004", linewidth=0,
            facecolor=FLOOR_TINT[key % len(FLOOR_TINT)], zorder=0))

    by = float(np.mean([v[0] for v in floor_pts.values()]))
    ax.plot([X["building"]], [by], marker="s", ms=17, color=C_BUILDING, zorder=5)
    ax.annotate("building", (X["building"], by), textcoords="offset points",
                xytext=(0, 15), ha="center", fontsize=8.5, fontweight="bold")

    for key, (fy, height, room_pts) in floor_pts.items():
        ax.plot([X["building"], X["floor"]], [by, fy], color=C_EDGE, lw=1.1, zorder=1)
        ax.plot([X["floor"]], [fy], marker="s", ms=14, color=C_FLOOR, zorder=5)
        ax.annotate(f"floor {key}\ny = {height:.2f} m", (X["floor"], fy),
                    textcoords="offset points", xytext=(0, 16), ha="center",
                    fontsize=8, fontweight="bold")
        floor_y_of[key] = fy
        for room, ry, cont_pts in room_pts:
            ax.plot([X["floor"], X["room"]], [fy, ry], color=C_EDGE, lw=0.9, zorder=1)
            ax.plot([X["room"]], [ry], marker="s", ms=10, color=C_ROOM, zorder=5)
            ax.annotate(room.label or f"room {room.id % 1_000_000}",
                        (X["room"], ry), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=6.5)
            for cont, cy, objs in cont_pts:
                ax.plot([X["room"], X["container"]], [ry, cy], color=C_EDGE,
                        lw=0.8, zorder=1)
                ax.plot([X["container"]], [cy], marker="o", ms=7,
                        color=C_CONTAINER, zorder=5)
                ax.annotate(cont.label, (X["container"], cy),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=6.0, color="#7d2f11")
                for obj, oy in objs:
                    ax.plot([X["container"], X["object"]], [cy, oy],
                            color=C_EDGE, lw=0.6, alpha=0.8, zorder=1)
                    ax.plot([X["object"]], [oy], marker="o", ms=4.5,
                            color=C_OBJECT, zorder=5)
                    ax.annotate(obj.label, (X["object"], oy),
                                textcoords="offset points", xytext=(7, -2),
                                ha="left", va="center", fontsize=5.8)

    for x, name in ((X["building"], "Building"), (X["floor"], "Floor"),
                    (X["room"], "Room"), (X["container"], "Container"),
                    (X["object"], "Object")):
        ax.annotate(name, (x, 1.0), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    family="serif")
    ax.set_xlim(-0.02, 1.10)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="00808-y9hTuugGdiq")
    ap.add_argument("--maps-root", default=str(WORKSPACE / "outputs/maps_15"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--side-res", type=int, default=1600)
    ap.add_argument("--section-frac", type=float, default=0.5,
                    help="where along z to cut; 0 is the near wall, 1 the far one")
    ap.add_argument("--pad-m", type=float, default=1.0)
    ap.add_argument("--max-rooms", type=int, default=3)
    ap.add_argument("--max-containers", type=int, default=3)
    ap.add_argument("--max-objects", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=350)
    args = ap.parse_args()

    blob, grids, costmaps, layer, graph, floors = load_multifloor(
        args.scene, pathlib.Path(args.maps_root))
    cache = WORKSPACE / (f"outputs/figures/side_{args.scene}_{args.side_res}"
                         f"_{args.section_frac:.2f}.npz")
    side, extent = render_side(args.scene, args.side_res, args.pad_m,
                               args.section_frac, cache)

    fig = plt.figure(figsize=(17.0, 8.2))
    ax_t = fig.add_axes([0.005, 0.02, 0.53, 0.90])
    ax_s = fig.add_axes([0.565, 0.02, 0.43, 0.90])

    floor_y_of = {}
    tree = build_tree(graph, layer, floors,
                      max_rooms=args.max_rooms,
                      max_containers=args.max_containers,
                      max_objects=args.max_objects)
    draw_tree(ax_t, tree, floor_y_of)

    rgba = np.dstack([side, np.where(side.sum(axis=2) > 16, 255, 0)
                      .astype(np.uint8)])
    ax_s.imshow(rgba, extent=[extent[0], extent[1], extent[2], extent[3]],
                origin="upper", interpolation="bilinear")
    ax_s.set_xlim(extent[0], extent[1])
    ax_s.set_ylim(extent[2], extent[3])
    ax_s.set_aspect("equal")
    ax_s.set_axis_off()
    ax_s.annotate("HM3D section (near half clipped)", (0.5, 1.0),
                  xycoords="axes fraction", ha="center", va="bottom",
                  fontsize=11, fontweight="bold", family="serif")

    # Tie each floor node to the storey it stands for.
    for key, fy in floor_y_of.items():
        height = next(float(s["height_y"]) for s in floors if int(s["key"]) == key)
        ax_s.axhline(height, color=C_FLOOR, lw=1.1, ls="--", alpha=0.85)
        ax_s.annotate(f"floor {key}", (extent[0], height), xytext=(4, 4),
                      textcoords="offset points", fontsize=8, color=C_FLOOR,
                      fontweight="bold")
        con = matplotlib.patches.ConnectionPatch(
            xyA=(1.045, fy), coordsA=ax_t.transData,
            xyB=(extent[0], height), coordsB=ax_s.transData,
            color=C_FLOOR, lw=1.0, ls=":", alpha=0.9)
        fig.add_artist(con)

    n_obj = len(graph.objects)
    n_cont = len(graph.containers)
    fig.text(0.005, 0.965,
             f"{args.scene}   {len(floors)} floors · {len(graph.rooms)} rooms · "
             f"{n_cont} support surfaces · {n_obj} objects"
             f"   (tree shows the largest {args.max_rooms}/{args.max_containers}/"
             f"{args.max_objects} at each level)",
             fontsize=10.5, va="bottom")

    out = pathlib.Path(args.out) if args.out else (
        WORKSPACE / f"outputs/figures/scene_graph_multifloor_{args.scene}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.png')}")
    for key, _, rooms in tree:
        print(f"  floor {key}: {len(rooms)} rooms shown, "
              f"{sum(len(c) for _, c in rooms)} containers shown")


if __name__ == "__main__":
    main()
