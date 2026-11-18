"""The stored ASCENT map, drawn with the episode's target marked on it.

A cross-anchor failure is geometric -- which storey the target is on, which
end of the staircase the agent drove to, whether the prior map covers the
place the object moved to -- and a table of coordinates does not settle a
geometric question. Three things came out of the first one of these that the
numbers had not: the stair paste and the flight ends were correct, the lower
storey's northern room was never mapped at all, and the two episodes were
failing for different reasons (docs/history/CROSS_ANCHOR_OBSTACLE_MAP.md).

Written once per scene at the end of a run that read `ycb.obstacle_map_in`.
Best effort throughout: a drawing must never be able to fail a run that has
already produced its episodes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..mapping.costmap import FREE, OCCUPIED, UNKNOWN

FREE_RGB = (1.0, 1.0, 1.0)
OCC_RGB = (0.15, 0.15, 0.18)
UNKNOWN_RGB = (0.72, 0.72, 0.72)
STAIR_RGB = (0.95, 0.55, 0.10)


def _render(costmap) -> np.ndarray:
    """Grid -> RGB. White free, black obstacle, grey unmapped, orange stairs."""
    grid = costmap.grid
    img = np.full(grid.shape + (3,), UNKNOWN_RGB[0], dtype=float)
    img[grid == FREE] = FREE_RGB
    img[grid == OCCUPIED] = OCC_RGB
    mask = getattr(costmap, "stair_mask", None)
    if mask is not None:
        img[mask] = STAIR_RGB
    return img


def _extent(costmap):
    rows, cols = costmap.grid.shape
    x0, z0 = float(costmap.origin[0]), float(costmap.origin[1])
    return x0, x0 + rows * costmap.resolution, z0, z0 + cols * costmap.resolution


def _storey_heights(episodes: Sequence[dict]) -> List[float]:
    """Bottom-up storey heights, taken from the run's own `floor_log`."""
    seen = {round(float(row[2]), 3)
            for e in episodes for row in (e.get("floor_log") or [])}
    return sorted(seen)


def _stair_endpoints(maps_dir, snapshots) -> dict:
    """`{storey slot: [(label, (x, z)), ...]}` over EVERY pasted snapshot.

    Under `ycb.obstacle_map_union` a scene has one snapshot per mapping
    episode, each with its own anchor and its own recorded staircase, so the
    endpoints have to be collected per file rather than read off one.
    """
    from navigation.mapping.map_store import (_anchor_from, _px_to_world,
                                              load_obstacle_maps)

    maps_dir = Path(maps_dir)
    out: dict = {}
    for snapshot in snapshots:
        path = Path(snapshot["path"])
        blob = load_obstacle_maps(str(path))
        anchor = _anchor_from(blob)
        npz_path = path.with_suffix(".npz")
        if not npz_path.exists():
            continue
        npz = np.load(npz_path)
        by_index = {int(f["index"]): f for f in blob.get("floors", [])}
        for pair in snapshot.get("pasted", []):
            record = by_index.get(int(pair["ascent_floor"]))
            if record is None:
                continue
            size = int(record["size"])
            epo = np.asarray(record["episode_pixel_origin"], dtype=float)
            ppm = float(record["pixels_per_meter"])
            for name, colour in (("_up_stair_start", "#1b9e77"),
                                 ("_up_stair_end", "#d95f02"),
                                 ("_down_stair_start", "#d95f02"),
                                 ("_down_stair_end", "#1b9e77")):
                key = record["prefix"] + name
                if key not in getattr(npz, "files", []):
                    continue
                px = np.asarray(npz[key], dtype=float).ravel()
                if px.size < 2:
                    continue
                # `robot_px`: geometric (row, col), the frame `_px_to_world`
                # takes (see `_to_geometric_layout` for why the masks are the
                # odd ones).
                world = _px_to_world(px[0], px[1], size, epo, ppm, anchor)
                label = f"ASCENT {name.strip('_').replace('_', ' ')}"
                out.setdefault(int(pair["osg_floor"]), []).append(
                    (label, colour, world))
    return out


def plot_obstacle_map(maps_dir, scene: str, episodes: Sequence[dict], out_path,
                      heights: Optional[Sequence[float]] = None,
                      union: bool = True) -> Optional[str]:
    """Draw every storey, with each episode's markers on the storey it is on.

    One panel per STOREY, not per stored floor: under the union a storey is
    built from several snapshots' floors, and a panel per floor would draw the
    same storey several times over. Returns the path written, or None if there
    was nothing to draw.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .prior_map import paste_scene

    maps_dir = Path(maps_dir)
    storeys = list(heights) if heights else _storey_heights(episodes)
    if not storeys:
        storeys = [0.0]

    pasted = paste_scene(str(maps_dir), scene, storeys, union=union)
    if not pasted["costmaps"] or not pasted["cells_written"]:
        return None
    # Attribute each snapshot's placements back to it, for the stair endpoints.
    by_file: dict = {}
    for pair in pasted["pasted"]:
        by_file.setdefault(pair["from"], []).append(pair)
    for snapshot in pasted["snapshots"]:
        snapshot["pasted"] = by_file.get(snapshot["file"], [])
    endpoints = _stair_endpoints(maps_dir, pasted["snapshots"])
    drawn = sorted({int(p["osg_floor"]) for p in pasted["pasted"]})

    fig, axes = plt.subplots(1, len(drawn), figsize=(9 * len(drawn), 9))
    axes = np.atleast_1d(axes)
    for ax, index in zip(axes, drawn):
        lo = storeys[index]
        costmap = pasted["costmaps"][index]
        cells = sum(int(p["cells"]) for p in pasted["pasted"]
                    if int(p["osg_floor"]) == index)
        ax.imshow(np.transpose(_render(costmap), (1, 0, 2)), origin="lower",
                  extent=_extent(costmap), interpolation="nearest")

        for label, colour, world in endpoints.get(index, []):
            ax.plot(*world, marker="P", ms=15, mfc=colour, mec="k", mew=1.4,
                    ls="none", label=label)

        for episode in episodes:
            # A panel is now a STOREY, which is the numbering `goal_floor` and
            # `start_floor` already use, so no reconciliation is needed -- but
            # only storeys this run actually pasted onto get a marker, because
            # drawing every episode on every panel makes a target one storey
            # away look reachable, which is the whole point of the figure.
            pasted_storeys = {int(r["osg_floor"]) for r in
                              (episode.get("prior_obstacle_map") or {}).get(
                                  "matched_floors", [])}
            goal = int(episode.get("goal_floor", -1))
            here = int(episode.get("start_floor", -1))
            goal_panel = goal if goal in pasted_storeys else None
            here_panel = here if here in pasted_storeys else None
            # GROUND TRUTH is `authored_layout.target_position`, a 3D habitat
            # point. `target_obj_xy` is NOT the target: it is the agent's last
            # committed candidate (`record.target_track_fields`), and the
            # first version of this figure drew that as the target -- which put
            # a red star on a false positive 8.9 m from the real object and
            # led to a wrong reading about coverage. Both are drawn now, with
            # different marks, so the two can never be confused again.
            truth = (episode.get("authored_layout") or {}).get("target_position")
            if truth and len(truth) >= 3 and goal_panel == index:
                ax.plot(truth[0], truth[2], marker="*", ms=30, mfc="#e41a1c",
                        mec="k", mew=1.6, ls="none",
                        label=f"TRUE TARGET {episode['target']}")
                ax.annotate(f"   {episode['target']} (truth)", (truth[0], truth[2]),
                            color="#e41a1c", fontsize=13, weight="bold")
            belief = episode.get("target_obj_xy")
            if belief and here_panel == index:
                ax.plot(belief[0], belief[1], marker="v", ms=15, mfc="#ff7f00",
                        mec="k", mew=1.2, ls="none",
                        label=f"agent's last commit ({episode['target']})")
            final = episode.get("final_xy")
            if final and here_panel == index:
                ax.plot(final[0], final[1], marker="X", ms=16, mfc="#377eb8",
                        mec="k", mew=1.4, ls="none",
                        label=f"agent ended ({episode['target']})")
            if here_panel == index:
                for step, foot, kind, _n in (episode.get("portal_log") or []):
                    ax.plot(foot[0], foot[1], marker="o", ms=15, mfc="none",
                            mec="#984ea3", mew=3.2, ls="none",
                            label=f"flight {kind} @{step}")

        ax.set_title(f"storey {index}  (y={lo:.2f})   {cells} cells pasted from "
                     f"{len(pasted['used'])} snapshot(s)", fontsize=13)
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world z (m)")
        ax.grid(alpha=0.25, ls=":")
        handles, labels = ax.get_legend_handles_labels()
        seen, keep = set(), []
        for handle, label in zip(handles, labels):
            if label not in seen:
                seen.add(label)
                keep.append((handle, label))
        if keep:
            ax.legend(*zip(*keep), loc="upper left", fontsize=9, framealpha=0.92)
        known = np.argwhere(costmap.grid != UNKNOWN)
        if known.size:
            xs = costmap.origin[0] + known[:, 0] * costmap.resolution
            zs = costmap.origin[1] + known[:, 1] * costmap.resolution
            ax.set_xlim(xs.min() - 2, xs.max() + 2)
            ax.set_ylim(zs.min() - 2, zs.max() + 2)

    fig.suptitle(f"{scene}   orange = stairs carried from ASCENT   "
                 "white = free   black = obstacle   grey = never mapped",
                 fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=115, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def save_obstacle_map_pngs(cfg, out_dir) -> List[str]:
    """One figure per scene, from the run's own `episodes.jsonl`.

    Silent no-op when the run did not read an obstacle map -- there is nothing
    to draw -- and best effort otherwise: the episodes are already on disk and
    a drawing must not be able to take them away.
    """
    maps_dir = str(getattr(cfg.ycb, "obstacle_map_in", "") or "")
    if not maps_dir or not bool(getattr(cfg.eval, "obstacle_map_png", True)):
        return []
    out_dir = Path(out_dir)
    episodes_file = out_dir / "episodes.jsonl"
    if not episodes_file.exists():
        return []
    try:
        episodes = [json.loads(line) for line in
                    episodes_file.read_text().splitlines() if line.strip()]
    except Exception:
        return []
    by_scene: dict = {}
    for episode in episodes:
        by_scene.setdefault(str(episode.get("scene", "")), []).append(episode)
    written: List[str] = []
    for scene, eps in by_scene.items():
        if not scene:
            continue
        try:
            path = plot_obstacle_map(
                maps_dir, scene, eps, out_dir / "viz" / f"obstacle_map_{scene}.png")
        except Exception as exc:  # never fail a finished run over a picture
            print(f"[obstacle-map png] {scene}: {type(exc).__name__}: {exc}")
            continue
        if path:
            written.append(path)
    return written
