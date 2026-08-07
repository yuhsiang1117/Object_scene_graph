"""Classify HM3D ObjectNav scenes and episodes by floor structure.

Reads the episode dataset directly (no simulator, no scene meshes needed) and
answers two different questions:

  * per SCENE  -- how many distinct floors do the goal view points occupy?
    A scene is `single` when every goal view point in it lies within
    `--scene-spread` of one height. This is what produced the ten-scene list in
    `configs/eval/hm3d_val_single_floor.yaml`.

  * per EPISODE -- is the goal reachable without changing level? An episode is
    `cross_floor` when no goal view point sits within SAME_FLOOR_M of the
    agent's start height, i.e. the agent MUST take the stairs to succeed.
    This is the split reported as `per_floor_class` in a run's summary.json.

The two are independent: a multi-floor scene still yields mostly `same_floor`
episodes, because the agent usually starts on the goal's floor.

Usage:
    python scripts/scene_floors.py                        # v2 val
    python scripts/scene_floors.py --version v1 --split val
    python scripts/scene_floors.py --out data/floor_classes.json
"""
import argparse
import glob
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from osg.eval.floors import (  # noqa: E402
    HEIGHT_AXIS,
    SAME_FLOOR_M,
    classify_episode,
    navmesh_floor_heights,
)


def navmesh_floors(scene, scenes_dir):
    """Scene floor heights from the navmesh, or None when unavailable.

    Loads the .navmesh directly into a bare PathFinder -- no Simulator, no GPU,
    no scene mesh parsing (~1 s/scene). This is the stronger scene-level
    signal: it sees floors that hold no goal objects. Optional because it needs
    habitat_sim and the (license-gated) scene download.
    """
    try:
        import habitat_sim
    except ImportError:
        return None
    paths = glob.glob(os.path.join(scenes_dir, "*", f"*-{scene}", f"{scene}.basis.navmesh"))
    if not paths:
        return None
    pf = habitat_sim.nav.PathFinder()
    pf.load_nav_mesh(paths[0])
    if not pf.is_loaded:
        return None
    return navmesh_floor_heights(pf)


def cluster_heights(heights, gap):
    """Sorted heights -> list of cluster means, splitting wherever consecutive
    values differ by more than `gap`. Enough to count floors: HM3D storeys are
    >2 m apart while view points on one storey scatter by centimetres."""
    if not heights:
        return []
    hs = sorted(heights)
    clusters = [[hs[0]]]
    for h in hs[1:]:
        if h - clusters[-1][-1] > gap:
            clusters.append([h])
        else:
            clusters[-1].append(h)
    return [sum(c) / len(c) for c in clusters]


def goal_view_heights_json(goal):
    """Heights of a raw (JSON) goal's view points, or its centre as fallback."""
    vps = goal.get("view_points") or []
    out = [float(vp["agent_state"]["position"][HEIGHT_AXIS]) for vp in vps]
    if not out and goal.get("position") is not None:
        out = [float(goal["position"][HEIGHT_AXIS])]
    return out


def load_scene(path):
    """One content/<scene>.json.gz -> (scene_id, episodes, goals_by_category)."""
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return os.path.basename(path).replace(".json.gz", ""), data.get(
        "episodes", []
    ), data.get("goals_by_category", {})


def analyze_scene(path, scene_spread, scenes_dir=None):
    scene, episodes, goals_by_cat = load_scene(path)

    all_heights = []
    for goals in goals_by_cat.values():
        for goal in goals:
            all_heights.extend(goal_view_heights_json(goal))
    floors = cluster_heights(all_heights, gap=scene_spread)

    ep_rows = []
    for ep in episodes:
        start_y = float(ep["start_position"][HEIGHT_AXIS])
        # habitat builds this key as basename(scene_id) + "_" + category
        key = os.path.basename(ep["scene_id"]) + "_" + ep["object_category"]
        goal_ys = []
        for goal in goals_by_cat.get(key, []):
            goal_ys.extend(goal_view_heights_json(goal))
        ep_rows.append({
            "episode_id": str(ep["episode_id"]),
            "scene": scene,
            "target": ep["object_category"],
            "start_y": round(start_y, 3),
            "floor_class": classify_episode(start_y, goal_ys),
        })

    nav_floors = navmesh_floors(scene, scenes_dir) if scenes_dir else None

    return {
        "scene": scene,
        "n_floors": len(floors),
        "floor_heights": [round(h, 3) for h in floors],
        "goal_y_span": round(max(all_heights) - min(all_heights), 3) if all_heights else 0.0,
        "scene_class": "single" if len(floors) <= 1 else "multi",
        # Independent navmesh signal: sees floors with no goal objects on them,
        # which the goal-view-point clustering cannot. None when habitat_sim or
        # the scene meshes are unavailable.
        "navmesh_n_floors": len(nav_floors) if nav_floors is not None else None,
        "navmesh_heights": nav_floors,
        "n_episodes": len(ep_rows),
        "episodes": ep_rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v2", help="episode dataset version (v1 / v2)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--root", default="data/datasets/objectnav/hm3d",
                    help="episode dataset root")
    ap.add_argument("--scene-spread", type=float, default=SAME_FLOOR_M,
                    help="max height gap within one floor (m)")
    ap.add_argument("--scenes-dir", default=None,
                    help="scene_datasets/hm3d_v0.2 root; enables the independent "
                         "navmesh floor count (needs habitat_sim + scene meshes)")
    ap.add_argument("--out", default=None,
                    help="write the full per-episode classification to this JSON")
    args = ap.parse_args()

    pattern = os.path.join(args.root, args.version, args.split, "content", "*.json.gz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"no episode files at {pattern}")

    scenes = [analyze_scene(p, args.scene_spread, args.scenes_dir) for p in paths]
    scenes.sort(key=lambda s: (s["n_floors"], s["scene"]))

    print(f"{'scene':<16} {'floors':>6} {'nav':>4} {'y-span':>7} {'eps':>5} {'cross':>6}  heights")
    ep_classes = Counter()
    single_scenes = []
    disagree = []
    for s in scenes:
        cross = sum(1 for e in s["episodes"] if e["floor_class"] == "cross_floor")
        ep_classes.update(e["floor_class"] for e in s["episodes"])
        if s["scene_class"] == "single":
            single_scenes.append(s["scene"])
        nav = s["navmesh_n_floors"]
        if nav is not None and nav != s["n_floors"]:
            disagree.append((s["scene"], s["n_floors"], nav))
        heights = ", ".join(f"{h:.2f}" for h in s["floor_heights"][:6])
        print(f"{s['scene']:<16} {s['n_floors']:>6} {('-' if nav is None else nav):>4} "
              f"{s['goal_y_span']:>7.2f} {s['n_episodes']:>5} {cross:>6}  {heights}")

    n_scenes = len(scenes)
    n_single = len(single_scenes)
    n_eps = sum(s["n_episodes"] for s in scenes)
    print(f"\nscenes: {n_single}/{n_scenes} single-floor, {n_scenes - n_single} multi-floor")
    print(f"episodes: {n_eps} total, " + ", ".join(
        f"{k} {v} ({100.0 * v / n_eps:.1f}%)" for k, v in sorted(ep_classes.items())))
    if disagree:
        print("\ngoal-view-point vs navmesh floor count disagree "
              "(navmesh also sees floors with no goal objects):")
        for scene, n_goal, n_nav in disagree:
            print(f"  {scene:<16} goals {n_goal}  navmesh {n_nav}")
    print("\nsingle-floor scenes (for configs/eval/*.yaml content_scenes):")
    for s in single_scenes:
        print(f"  - {s}")

    if args.out:
        by_scene = {s["scene"]: {k: v for k, v in s.items() if k != "episodes"} for s in scenes}
        by_episode = {}
        for s in scenes:
            for e in s["episodes"]:
                by_episode.setdefault(s["scene"], {})[e["episode_id"]] = e["floor_class"]
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "version": args.version, "split": args.split,
                "same_floor_m": args.scene_spread,
                "scenes": by_scene, "episode_floor_class": by_episode,
            }, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
