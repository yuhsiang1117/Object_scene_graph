"""Does a stored ASCENT obstacle map mean the same thing to the OSG pipeline?

The unit tests prove the transform inverts the obstacle map's own projection on
synthetic data. This is the end-to-end check on a REAL pass, and it uses the
one piece of ground truth a run always has: **the agent walked there**.

Every pose in the behaviour trace is a place the agent physically stood, so
after the snapshot is resampled into an OSG costmap those poses must land on
cells the map calls FREE. If the frame conversion were wrong -- a missed axis
swap, an unrotated anchor -- the poses would fall on UNKNOWN cells outside the
mapped region, and that is exactly what this counts.

The trace records `xy` in ASCENT's world convention, which is habitat's
`(x, -z)` (`geometry.robot_xy_heading`); OSG's `PLANE` is `(x, z)`. Converting
here rather than assuming they agree is the point of the exercise.

NEEDS `eval.behaviour_log=true` on the run being checked: the episode record
keeps only the final pose, and one pose cannot distinguish a correct map from
a map rotated about it.

    python scripts/run_eval.py +experiment=mf5_ascentnav_map \\
        eval.behaviour_log=true ycb.obstacle_map_out=outputs/maps_mf5_ascent \\
        output_dir=outputs/mf5_pass1
    python scripts/check_obstacle_map_reuse.py \\
        --maps outputs/maps_mf5_ascent --run outputs/mf5_pass1 [--png out.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from navigation.mapping.map_store import (
    apply_to_costmap,
    floor_summaries,
    load_obstacle_maps,
)
from osg.eval.prior_map import _scene_map_paths
from osg.eval.record import safe_tag
from osg.mapping.costmap import Costmap2D

# How far outside the mapped region a pose may fall and still be the depth
# camera's coverage fringe rather than a broken transform. Measured on 00800:
# every off-map pose was within 0.27 m, median 0.10 m; a copy-with-offset puts
# them 10-22 m out.
NEAR_M = 0.5


def _episode_id(ep) -> str:
    """The tag `save_obstacle_maps` stored, rebuilt from the jsonl record.

    `eval.record.episode_tag` wants a habitat episode object; the record is a
    dict, so the same string is assembled from its fields rather than by
    calling that helper on the wrong type -- which silently returned something
    that never matched, and sent every check down the fallback path.
    """
    from osg.eval.record import safe_tag

    authored = ep.get("authored_layout") or {}
    scene = safe_tag(authored.get("scene") or ep.get("scene") or "scene")
    layout = safe_tag(authored.get("layout_id") or "layout")
    return f"{scene}_{layout}_ep{safe_tag(ep.get('episode_id', ''))}"


def _poses(episodes) -> np.ndarray:
    """Every traced pose, in OSG world PLANE coordinates.

    `step_trace` rows carry `xy` in ASCENT's world frame -- habitat's (x, -z),
    per `navigation/geometry.robot_xy_heading` -- so the sign of the second
    component is flipped back here. Getting this wrong is the failure the whole
    script exists to catch, so it is done in one place and named.
    """
    out = []
    for ep in episodes:
        for row in (ep.get("step_trace") or []):
            xy = row.get("xy")
            if xy and len(xy) >= 2:
                out.append([float(xy[0]), -float(xy[1])])
    return np.array(out, dtype=float)


def _near_fraction(costmap, poses):
    """(distances, fraction within NEAR_M, poses outside the grid).

    DISTANCE, not a hit rate. A few percent of poses always fall just outside
    the mapped region -- ASCENT marks `explored_area` from what the depth
    camera saw, and with min_depth 0.5 m the cell under the agent is unmarked
    until it is viewed from somewhere else. So "was it on a mapped cell"
    quietly fails on healthy maps. What separates a coverage fringe from a
    broken frame is HOW FAR outside: the fringe is centimetres, an unrotated
    anchor is 10-22 m (measured).
    """
    from scipy import ndimage

    dist_cells = ndimage.distance_transform_edt(costmap.grid == -1)
    distances, outside = [], 0
    for xy in poses:
        rc = costmap.world_to_grid(xy)
        if not costmap.in_bounds(rc):
            outside += 1
            continue
        distances.append(float(dist_cells[rc[0], rc[1]]) * costmap.resolution)
    d = np.array(distances) if distances else np.zeros(0)
    return d, (float((d <= NEAR_M).mean()) if len(d) else 0.0), outside


def _episodes(run_dir: Path):
    for path in sorted(run_dir.glob("**/episodes.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True, help="ycb.obstacle_map_out directory")
    ap.add_argument("--run", required=True, help="the run that wrote them")
    ap.add_argument("--png", default="", help="also draw the map + trajectory here")
    args = ap.parse_args()

    maps_dir, run_dir = Path(args.maps), Path(args.run)
    episodes = list(_episodes(run_dir))
    if not episodes:
        raise SystemExit(f"no episodes.jsonl under {run_dir}")

    by_scene: dict = {}
    for ep in episodes:
        scene = str((ep.get("authored_layout") or {}).get("scene") or ep.get("scene") or "")
        by_scene.setdefault(scene, []).append(ep)

    worst = 1.0
    checked_nothing = False
    any_outside = False
    for scene, eps in sorted(by_scene.items()):
        # Every snapshot for the scene: the best-of file, and the per-episode
        # ones `ycb.obstacle_map_union` writes. Reading only `<scene>.json`
        # made this print "NO SNAPSHOT" for a whole union directory, because
        # union mode never writes that file (osg/eval/prior_map.py).
        paths = _scene_map_paths(str(maps_dir), scene)
        if not paths:
            print(f"{scene}: NO SNAPSHOT under {maps_dir}")
            continue
        print(f"\n{scene}  ({len(paths)} snapshot(s))")
        floors = []
        for path in paths:
            blob = load_obstacle_maps(path)
            summaries = floor_summaries(blob)
            floors.extend(summaries)
            print(f"  {Path(path).name}: {len(summaries)} stored floors")
            for f in summaries:
                print(f"    floor {f['index']}: explored {f['explored_cells']:>8} cells, "
                      f"obstacle {f['obstacle_cells']:>7}, steps {f['steps']:>4}, "
                      f"floor_y={f['floor_y']}, "
                      f"up={f['has_up_stair']} down={f['has_down_stair']}")

        # Merge every storey of every snapshot into ONE costmap: the trajectory
        # crosses floors and this check is about the FRAME, not about which
        # storey a pose was on. Heights are all zero for the same reason --
        # the stair ramp is irrelevant here and a single plane keeps every
        # snapshot's floors landing on the one grid.
        mapped = [f for f in floors if f["explored_cells"] > 0]
        if not mapped:
            print("  NO MAPPED FLOOR in any snapshot")
            checked_nothing = True
            continue
        costmap = Costmap2D(resolution=float(mapped[0]["resolution"]), size_m=20.0)
        for path in paths:
            blob = load_obstacle_maps(path)
            for f in floor_summaries(blob):
                if f["explored_cells"] > 0:
                    apply_to_costmap(blob, f["index"], costmap, overwrite=False)

        # ONE episode's map, checked against THAT episode's poses -- unless the
        # snapshots ARE per episode, in which case every pose in the run was
        # walked onto some snapshot in this union and all of them count. With a
        # single best-of file a different episode starts elsewhere and
        # legitimately walks where that map was never built, so scoring it
        # against all of them would measure the protocol, not the frame.
        ids = {str(load_obstacle_maps(p).get("episode_id") or "") for p in paths}
        if len(paths) > 1:
            print(f"  union of {len(paths)} episodes; checking every pose in the run")
        else:
            want = next(iter(ids))
            mine = [e for e in eps if _episode_id(e) == want] if want else []
            if mine:
                print(f"  snapshot written by episode: {want}")
            else:
                if want:
                    print(f"  snapshot names episode {want}, not in this run; "
                          "falling back to the best-matching episode")
                else:
                    print("  snapshot predates episode provenance; "
                          "using the best-matching episode")
                mine = eps
            eps = mine

        if len(eps) > 1:
            best, best_near = None, -1.0
            for candidate in eps:
                p_c = _poses([candidate])
                if not len(p_c):
                    continue
                near_c = _near_fraction(costmap, p_c)[1]
                if near_c > best_near:
                    best, best_near = candidate, near_c
            if best is not None:
                eps = [best]
                print(f"  best-matching episode: {_episode_id(best)}")

        poses = _poses(eps)
        if not len(poses):
            print("  NO POSES: rerun with eval.behaviour_log=true -- without a "
                  "trace there is nothing to check the frame against")
            checked_nothing = True
            continue
        d, near, outside = _near_fraction(costmap, poses)
        on_map = float((d == 0).mean()) if len(d) else 0.0
        worst = min(worst, near)
        print(f"  trajectory poses: {len(poses)}")
        print(f"    exactly on a mapped cell : {on_map:6.1%}")
        print(f"    within {NEAR_M} m of the map  : {near:6.1%}   <- the frame check")
        if len(d[d > 0]):
            off = d[d > 0]
            print(f"    off-map poses: {len(off)}, median {np.median(off):.2f} m, "
                  f"max {off.max():.2f} m")
        print(f"    outside the grid entirely: {outside}")
        any_outside = any_outside or bool(outside)
        if outside:
            print("    ** poses fell OUTSIDE the grid entirely -- the only "
                  "symptom here that a wrong frame produces and a sparse map "
                  "does not **")

        if args.png:
            _draw(costmap, poses, Path(args.png), scene)

    if checked_nothing:
        print("\nNOTHING WAS CHECKED -- see above. Not a pass.")
        raise SystemExit(2)
    print(f"\nworst 'within {NEAR_M} m' fraction across scenes: {worst:.1%}")
    print(
        "\nREAD THIS AS COVERAGE, NOT CORRECTNESS. A low fraction does not mean\n"
        "a wrong frame: ASCENT erases an agent-radius band around every obstacle\n"
        "(obstacle_map.py:374) and never reveals a stairwell interior, so a valid\n"
        "map leaves much of the path over UNKNOWN -- measured on 00800, one\n"
        "episode each: bowl 100%, banana 42%, pitcher 11% on the strict test,\n"
        "and all three are correctly framed. The frame itself is pinned by\n"
        "tests/unit/test_navigation_map_store.py against the obstacle map's own\n"
        "projection at four start headings; the --png is how a human confirms it."
    )
    raise SystemExit(2 if any_outside else 0)


def _draw(costmap, poses, png: Path, scene: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    known = np.argwhere(costmap.grid != -1)
    if not len(known):
        return
    r0, c0 = known.min(axis=0)
    r1, c1 = known.max(axis=0) + 1
    view = costmap.grid[r0:r1, c0:c1]
    extent = [
        costmap.origin[1] + c0 * costmap.resolution,
        costmap.origin[1] + c1 * costmap.resolution,
        costmap.origin[0] + r1 * costmap.resolution,
        costmap.origin[0] + r0 * costmap.resolution,
    ]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(np.where(view == -1, 0.5, np.where(view == 100, 0.0, 1.0)),
              cmap="gray", vmin=0, vmax=1, extent=extent)
    ax.plot(poses[:, 1], poses[:, 0], "-", color="#d1495b", lw=1.2,
            label="agent trajectory")
    ax.plot(poses[0, 1], poses[0, 0], "o", color="#2e86ab", ms=7, label="start")
    ax.set_title(f"{scene}: ASCENT obstacle map, resampled into an OSG costmap")
    ax.set_xlabel("world z (m)")
    ax.set_ylabel("world x (m)")
    ax.legend(loc="upper right")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {png}")


if __name__ == "__main__":
    main()
