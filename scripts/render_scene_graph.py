#!/usr/bin/env python3
"""Render a saved prior map as the scene graph the agent actually built.

Reads `outputs/maps_v5/<scene>/<scene>.json` (+ its `.npz` grids), rebuilds the
floor -> room -> container -> object hierarchy with the same code the agent
runs (`osg.graph.scene_graph.SceneGraph.rebuild`), and draws it over the
occupancy grid.  Nothing here re-derives geometry: the ellipsoids, room
segmentation and container qualification are the map's own.

    python scripts/render_scene_graph.py --scene 00848-ziup5kvtCCR
    python scripts/render_scene_graph.py --scene 00848-ziup5kvtCCR \
        --episode outputs/osg_dualmap_protocol/00848-ziup5kvtCCR/episodes.jsonl \
        --episode-id 00848-ziup5kvtCCR__cross_anchor__0128-1__bowl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse

from osg.graph.map_store import load_map, _track_from_record
from osg.graph.scene_graph import SceneGraph
from osg.graph import containers as containers_mod
from osg.mapping.costmap import Costmap2D, PLANE, UNKNOWN, FREE, OCCUPIED
from osg.objects.object_layer import ObjectLayer

WORKSPACE = pathlib.Path(__file__).resolve().parents[1]

# Muted fills: the rooms are context, the graph is the subject.
ROOM_COLORS = [
    "#cfe3f2", "#f6dfd0", "#d8ecd6", "#eddaea", "#f4eccb",
    "#d6e4e8", "#f0d8d8", "#dfe0f0", "#e6efd4", "#f3e2d2",
]
CONTAINER_EDGE = "#b8471f"
CONTAINER_FILL = "#f7e2d8"
OBJECT_ON = "#1f5c8b"
OBJECT_FREE = "#9aa4ad"
SEARCH = "#1b5e20"


def load_graph(scene: str, maps_root: pathlib.Path):
    path = maps_root / scene / f"{scene}.json"
    if not path.exists():
        raise SystemExit(f"no saved map at {path}")
    blob = load_map(path)
    grids = blob["_grids"]
    res = float(blob["resolution"])
    grid = grids["grid"]
    costmap = Costmap2D(resolution=res, size_m=grid.shape[0] * res)
    costmap.grid = grid
    costmap.origin = np.asarray(grids["origin"], dtype=float)

    layer = ObjectLayer()
    for rec in blob["tracks"]:
        track = _track_from_record(rec)
        layer._tracks[track.id] = track
    layer._next_id = int(blob["next_track_id"])

    graph = SceneGraph()
    graph.rebuild(grids["room_labels"], costmap, layer)
    return blob, grids, costmap, layer, graph


def known_extent(grid: np.ndarray, costmap: Costmap2D, pad_m: float = 1.0):
    """World-frame bbox of everything the agent actually saw."""
    rows, cols = np.nonzero(grid != UNKNOWN)
    if rows.size == 0:
        raise SystemExit("map has no observed cells")
    lo = costmap.grid_to_world(np.array([rows.min(), cols.min()], dtype=float))
    hi = costmap.grid_to_world(np.array([rows.max(), cols.max()], dtype=float))
    return (min(lo[0], hi[0]) - pad_m, max(lo[0], hi[0]) + pad_m,
            min(lo[1], hi[1]) - pad_m, max(lo[1], hi[1]) + pad_m)


def cov_ellipse(ax, center_xy, cov, *, n_std=2.0, **kw):
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=float))
    vals = np.clip(vals, 1e-6, None)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2.0 * n_std * np.sqrt(vals)
    e = Ellipse(xy=center_xy, width=w, height=h, angle=angle, **kw)
    ax.add_patch(e)
    return e


def draw_base(ax, grids, costmap, extent):
    """Occupancy under soft per-room fills."""
    grid = grids["grid"]
    rooms = grids["room_labels"]
    h, w = grid.shape
    x0, y0 = costmap.origin
    img_extent = [x0, x0 + w * costmap.resolution, y0, y0 + h * costmap.resolution]

    # Room fills first, so walls draw on top of them.
    room_ids = [int(r) for r in np.unique(rooms) if r != 0]
    canvas = np.zeros((*rooms.shape, 4), dtype=float)
    observed = grid != UNKNOWN
    for i, rid in enumerate(room_ids):
        rgba = matplotlib.colors.to_rgba(ROOM_COLORS[i % len(ROOM_COLORS)])
        canvas[(rooms == rid) & observed] = rgba
    # Observed-but-unroomed free space stays a neutral off-white.
    canvas[(rooms == 0) & (grid == FREE)] = matplotlib.colors.to_rgba("#f2f2f0")
    ax.imshow(np.transpose(canvas, (1, 0, 2)), origin="lower", extent=img_extent,
              interpolation="nearest", zorder=0)

    walls = np.ma.masked_where(grid != OCCUPIED, grid)
    ax.imshow(walls.T, origin="lower", extent=img_extent, interpolation="nearest",
              cmap=ListedColormap(["#3a3f44"]), zorder=1)
    return room_ids


def draw_graph(ax, graph, layer, *, min_area=0.0, max_labels=14):
    # container id -> centre, for the support edges
    centres = {}
    for cid, node in graph.containers.items():
        if node.area_m2 < min_area:
            continue
        xy = np.asarray(node.center, dtype=float)[list(PLANE)]
        centres[cid] = xy
        for tid in node.track_ids:
            track = layer._tracks.get(tid)
            if track is None:
                continue
            c_xy, cov = containers_mod.footprint(layer.center_of(track), track.ellipsoid)
            cov_ellipse(ax, c_xy, cov, n_std=1.0, facecolor=CONTAINER_FILL,
                        edgecolor=CONTAINER_EDGE, lw=1.0, alpha=0.85, zorder=4)


    for node in sorted(graph.containers.values(),
                       key=lambda n: n.area_m2, reverse=True)[:max_labels]:
        xy = np.asarray(node.center, dtype=float)[list(PLANE)]
        ax.annotate(node.label, xy, fontsize=6.5, color="#7d2f11",
                    ha="center", va="center", zorder=7,
                    path_effects=[matplotlib.patheffects.withStroke(
                        linewidth=2.2, foreground="white")])

    on_container = {}
    for cid, node in graph.containers.items():
        for oid in node.object_ids:
            on_container[oid] = cid

    for obj in graph.objects:
        xy = np.asarray(obj.center, dtype=float)[list(PLANE)]
        cid = on_container.get(obj.track_id)
        if cid is not None and cid in centres:
            ax.plot([xy[0], centres[cid][0]], [xy[1], centres[cid][1]],
                    color=OBJECT_ON, lw=0.45, alpha=0.45, zorder=3)
        ax.plot(*xy, marker="o", ms=2.4 if cid is not None else 1.8,
                color=OBJECT_ON if cid is not None else OBJECT_FREE,
                mec="none", zorder=5)
    return on_container


def draw_thumbnails(fig, ax, graph, layer, on_container, *, n=6, only=None):
    """Callouts of the agent's own best crop for a few objects."""
    cands = []
    for obj in graph.objects:
        track = layer._tracks.get(obj.track_id)
        if track is None or track.best_crop is None:
            continue
        if only and obj.label not in only:
            continue
        # Prefer objects that sit on a surface: those are the graph's point.
        cands.append((float(getattr(track, "best_score", 0.0)),
                      obj.track_id in on_container, obj, track))
    cands.sort(key=lambda t: (t[1], t[0]), reverse=True)

    seen, picked = set(), []
    for _, _, obj, track in cands:
        if obj.label in seen:
            continue
        seen.add(obj.label)
        picked.append((obj, track))
        if len(picked) >= n:
            break

    for i, (obj, track) in enumerate(picked):
        xy = np.asarray(obj.center, dtype=float)[list(PLANE)]
        col = i % 2
        row = i // 2
        box = fig.add_axes([0.745 + col * 0.126, 0.735 - row * 0.235, 0.112, 0.185])
        box.imshow(track.best_crop, aspect="auto")
        box.set_xticks([]); box.set_yticks([])
        for s in box.spines.values():
            s.set_edgecolor(OBJECT_ON); s.set_linewidth(1.2)
        box.set_title(f"{obj.label}  ({track.best_score:.2f})", fontsize=6.5, pad=2)
        ax.plot(*xy, marker="o", ms=6, mfc="none", mec=OBJECT_ON, mew=1.4, zorder=8)
    return picked


def draw_search(ax, graph, episodes_path, episode_id):
    """Overlay one episode's use of the graph.

    These runs do not log per-step poses, so there is no trajectory to draw.
    What they do log is better for a teaser: `search_log_events` is the ordered
    list of support surfaces the agent chose to inspect, with the prior and
    utility it scored them by, and `goal_commit_log` is the moment it committed
    to a track.  That is the graph being used, not just built.
    """
    ep = None
    with open(episodes_path) as fh:
        for line in fh:
            cand = json.loads(line)
            if episode_id in (None, "", cand["episode_id"]):
                ep = cand
                break
    if ep is None:
        raise SystemExit(f"episode {episode_id!r} not in {episodes_path}")

    # One marker per surface, in visit order; repeats keep their first number.
    order, seen = [], set()
    for ev in ep.get("search_log_events") or []:
        cid = ev.get("container_id")
        if cid is None or cid in seen or cid not in graph.containers:
            continue
        seen.add(cid)
        order.append((cid, ev))

    pts = []
    for cid, _ in order:
        node = graph.containers[cid]
        pts.append(np.asarray(node.center, dtype=float)[list(PLANE)])
    if len(pts) > 1:
        arr = np.asarray(pts)
        ax.plot(arr[:, 0], arr[:, 1], color=SEARCH, lw=1.6, alpha=0.9,
                zorder=8, solid_capstyle="round")
    for i, ((cid, ev), xy) in enumerate(zip(order, pts), start=1):
        ax.plot(*xy, marker="o", ms=11, mfc="white", mec=SEARCH, mew=1.8, zorder=9)
        ax.annotate(str(i), xy, fontsize=6.5, color=SEARCH, ha="center",
                    va="center", zorder=10, fontweight="bold")

    for ev in ep.get("goal_commit_log") or []:
        c = np.asarray(ev["center"], dtype=float)[list(PLANE)]
        ax.plot(*c, marker="*", ms=20, mfc="#f2b705", mec="#7a5c00", mew=1.2,
                zorder=11)
        ax.annotate(f"commit: {ev['label']}  p={ev['p']:.2f}", c,
                    textcoords="offset points", xytext=(10, 8), fontsize=7.5,
                    color="#5c4600", zorder=11,
                    path_effects=[matplotlib.patheffects.withStroke(
                        linewidth=2.4, foreground="white")])
    return ep, order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--maps-root", default=str(WORKSPACE / "outputs/maps_v5"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--thumbs", type=int, default=6)
    ap.add_argument("--thumb-labels", default="", help="comma-separated labels to prefer")
    ap.add_argument("--min-container-area", type=float, default=0.0)
    ap.add_argument("--max-labels", type=int, default=14)
    ap.add_argument("--episode", default=None)
    ap.add_argument("--episode-id", default=None)
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    blob, grids, costmap, layer, graph = load_graph(
        args.scene, pathlib.Path(args.maps_root))

    fig = plt.figure(figsize=(13.0, 8.0))
    ax = fig.add_axes([0.015, 0.02, 0.71, 0.92])
    extent = known_extent(grids["grid"], costmap)
    room_ids = draw_base(ax, grids, costmap, extent)
    on_container = draw_graph(ax, graph, layer,
                              min_area=args.min_container_area,
                              max_labels=args.max_labels)

    ep, order = None, []
    if args.episode:
        ep, order = draw_search(ax, graph, args.episode, args.episode_id)

    only = [s.strip() for s in args.thumb_labels.split(",") if s.strip()]
    if args.thumbs:
        draw_thumbnails(fig, ax, graph, layer, on_container,
                        n=args.thumbs, only=only or None)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    n_obj_on = len(on_container)
    title = (f"{args.scene}   "
             f"{len(room_ids)} rooms · {len(graph.containers)} support surfaces · "
             f"{len(graph.objects)} objects ({n_obj_on} on a surface)")
    if ep is not None:
        title += (f"\n{ep['episode_id']}   target={ep['target']}   "
                  f"surfaces inspected={len(order)}   success={ep['success']}")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=CONTAINER_FILL, edgecolor=CONTAINER_EDGE,
              label='support surface (container node)'),
        Line2D([], [], marker='o', ls='none', ms=4, color=OBJECT_ON,
               label='object on a surface'),
        Line2D([], [], marker='o', ls='none', ms=3, color=OBJECT_FREE,
               label='object, no surface'),
        Patch(facecolor=ROOM_COLORS[0], edgecolor='none', label='room'),
    ], loc='lower left', fontsize=7.5, frameon=True, framealpha=0.9,
       borderpad=0.6)
    ax.set_title(title, fontsize=10, pad=8)

    out = pathlib.Path(args.out) if args.out else (
        WORKSPACE / f"outputs/figures/scene_graph_{args.scene}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.png')}")
    print(f"rooms={len(room_ids)} containers={len(graph.containers)} "
          f"objects={len(graph.objects)} on_surface={n_obj_on}")
    print("top container labels:",
          Counter(c.label for c in graph.containers.values()).most_common(8))


if __name__ == "__main__":
    import matplotlib.patheffects  # noqa: F401  (used in draw_graph)
    main()
