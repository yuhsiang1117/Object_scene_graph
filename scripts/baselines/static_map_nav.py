#!/usr/bin/env python3
"""Navigate DualMap's released benchmark with a static map built by another method.

A static-map baseline (Khronos, Clio, HOV-SG, ConceptGraphs) maps the release's static
sequence offline and answers each query with a ranking of its objects: `rankings.json`, the
format `scripts/baselines/khronos/query_map.py` writes. This runner turns that ranking into
navigation, with everything else held at DualMap's released protocol:

  * the trials, targets, start poses and success rule: `osg.eval.dualmap_release` and the
    collector seed, through `run_dualmap_released_native` -- the same simulator and the same
    random start as the DualMap rows of the table;
  * the planner: DualMap's own `GlobalMapManager.calculate_global_path` on its released global
    map and walls, unchanged, with the method's candidate in place of DualMap's best anchor.
    It plans to the Voronoi node nearest the candidate;
  * the last leg: a straight segment from that node to the free cell DualMap's
    `snap_to_free_space_directional` picks beside the candidate. It stands in for DualMap's
    local planner, which runs on DualMap's online local map; a static map has none;
  * the follower: the released collector's ContinuousPathFollower (`ReleasedFollower`).

An attempt ends when the follower reaches the end of its path. It succeeds within 1 m
(horizontal, to the box) of any instance of the query. There are up to three attempts, each on
the best-ranked candidate more than 1 m (horizontal) from every candidate already tried: several
of a map's objects are often fragments of one thing, and going back to the same place is not a
second attempt. A candidate the planner finds no path to uses up its attempt.

The map is never updated. In the dynamic splits the agent goes where the static map says the
object was; that is the comparison the table makes, and it is labelled so.

    python scripts/baselines/static_map_nav.py \\
        --rankings 'outputs/baselines/khronos/map/{scene}/rankings.json' \\
        --output outputs/baselines/khronos/nav/seed12 --seed 12 [--scene 00829] [--condition static]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_dualmap_released_native as native  # noqa: E402

from osg.eval.dualmap_release import (  # noqa: E402
    MAX_ATTEMPTS,
    SUCCESS_DISTANCE_M,
    as_targets,
    distance_to_target,
    protocol,
    raw_layout,
    shortest_success_path,
    static_target_positions,
    target_positions,
)

# Simulated seconds an attempt may take before it is cut off. The follower integrates its
# velocity without collisions, so it always reaches the end of its path; this only bounds a
# path that is absurdly long.
MAX_ATTEMPT_SECONDS = 600.0


class Candidate:
    """What `GlobalMapManager.get_goal_position` reads from the object it navigates to."""

    def __init__(self, uid: str, center: Sequence[float], dimensions: Sequence[float]) -> None:
        import open3d as o3d

        c = np.asarray(center, dtype=float)
        h = np.maximum(np.asarray(dimensions, dtype=float), 1e-3) / 2.0
        self.uid = uid
        self.bbox_2d = o3d.geometry.AxisAlignedBoundingBox(c - h, c + h)


def system_pose(service) -> np.ndarray:
    """The camera pose in DualMap's world frame, as `data_input` builds it for DualMap."""
    from scipy.spatial.transform import Rotation
    from habitat_data_collector.utils.coordinate_transform import CoordinateTransform

    sensor = service.get_agent_state().sensor_states["color_sensor"]
    vector = CoordinateTransform.get_agent_pose_in_system(sensor)
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = Rotation.from_quat(vector[3:]).as_matrix()
    pose[:3, 3] = vector[:3]
    return pose


def horizontal(a: Sequence[float], b: Sequence[float]) -> float:
    """Horizontal distance in DualMap's world frame (z up)."""
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def choose(ranking: Sequence[Sequence[Any]], objects: Dict[str, Dict[str, Any]],
           tried: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for uid, score in ranking:
        obj = objects[uid]
        if all(horizontal(obj["center"], t["center"]) > SUCCESS_DISTANCE_M for t in tried):
            return {**obj, "score": float(score)}
    return None


def plan(cfg, pose: np.ndarray, candidate: Dict[str, Any]):
    """DualMap's global path to the candidate, then the leg to the free cell beside it."""
    from utils.global_map_manager import GlobalMapManager
    from utils.types import GoalMode
    from utils.visualizer import ReRunVisualizer

    # A singleton every map manager reaches for without a config; Dualmap.__init__ creates it
    # first, with one, and so does this.
    ReRunVisualizer(cfg).set_use_rerun(cfg.use_rerun)
    manager = GlobalMapManager(cfg)
    manager.load_map()
    manager.load_wall()
    target = Candidate(candidate["uid"], candidate["center"], candidate["dimensions"])
    manager.find_best_candidate_with_inquiry = lambda: (target, candidate["score"])
    path = manager.calculate_global_path(pose, goal_mode=GoalMode.INQUIRY)
    if not path:
        return None, {"global_path": False}
    graph = manager.nav_graph
    end = graph.calculate_pos_2d(path[-1])
    goal = graph.calculate_pos_2d(candidate["center"])
    snapped = graph.snap_to_free_space_directional(goal, end, graph.free_space)
    path = [tuple(float(v) for v in p) for p in path]
    leg = 0.0
    if snapped is not None:
        last = tuple(float(v) for v in graph.calculate_pos_3d(int(snapped[0]), int(snapped[1])))
        leg = horizontal(path[-1], last)
        if leg > 1e-6:
            path.append(last)
    return path, {"global_path": True, "last_leg_m": leg, "path_points": len(path),
                  "path_end": list(path[-1])}


def run_trial(trial: Dict[str, Any], rankings: Dict[str, Any], out: Path) -> Dict[str, Any]:
    layout_path = Path(trial["layout_path"])
    if trial["condition"] == "static":
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        targets = static_target_positions(trial["scene"], trial["query"])
    else:
        layout = raw_layout(trial["scene"], trial["condition"], trial["layout"])
        targets = as_targets(target_positions(layout, trial["query"]))
    _, service, objects, start = native.make_collector(trial["scene"], layout_path)
    cfg = native.setup_dualmap(trial["scene"], out, trial["query"])
    sys.path.insert(0, str(native.DUALMAP_ROOT))
    mapped = {o["uid"]: o for o in rankings["objects"]}
    ranking = rankings["rankings"][trial["query"]]
    shortest = shortest_success_path(service.sim.pathfinder, start, targets)
    follower = native.ReleasedFollower(service)
    attempts: List[Dict[str, Any]] = []
    travelled = 0.0
    success = False
    terminal = "attempts_exhausted"
    try:
        while len(attempts) < MAX_ATTEMPTS:
            candidate = choose(ranking, mapped, attempts)
            if candidate is None:
                terminal = "no_candidate"
                break
            try:
                path, info = plan(cfg, system_pose(service), candidate)
            except ImportError:
                raise  # a broken environment is not a failed plan
            except Exception as exc:  # DualMap's planner raising is a failed plan, not a crash
                path, info = None, {"global_path": False, "error": f"{type(exc).__name__}: {exc}"}
            attempt: Dict[str, Any] = {
                "attempt": len(attempts) + 1,
                "uid": candidate["uid"],
                "center": candidate["center"],
                "score": candidate["score"],
                "rank": next(i for i, (uid, _) in enumerate(ranking) if uid == candidate["uid"]),
                **info,
            }
            ticks = 0
            if path is not None:
                follower.path_key = None
                follower.update_path(path)
                done = follower.follower is None
                while not done and ticks < MAX_ATTEMPT_SECONDS * native.SIM_HZ:
                    delta, done = follower.advance(native.OBSERVATION_TICKS)
                    travelled += delta
                    ticks += native.OBSERVATION_TICKS
            state = service.get_agent_state()
            dist_h, dist_3d = distance_to_target(state.position, targets)
            attempt.update({
                "simulated_s": ticks / native.SIM_HZ,
                "position": np.asarray(state.position, dtype=float).tolist(),
                "distance_horizontal_m": dist_h,
                "distance_3d_m": dist_3d,
                "success": dist_h <= SUCCESS_DISTANCE_M,
                "travelled_m": travelled,
            })
            attempts.append(attempt)
            if attempt["success"]:
                success, terminal = True, "success"
                break
        state = service.get_agent_state()
        dist_h, dist_3d = distance_to_target(state.position, targets)
        spl = shortest / max(shortest, travelled, 1e-12) if success and math.isfinite(shortest) else 0.0
        result = {
            **trial,
            "method": rankings["method"],
            "map": rankings["map"],
            "success": int(success),
            "spl": float(spl),
            "shortest_success_path_m": None if not math.isfinite(shortest) else shortest,
            "travelled_m": travelled,
            "final_distance_horizontal_m": dist_h,
            "final_distance_3d_m": dist_3d,
            "attempts": attempts,
            "attempt_count": len(attempts),
            "terminal": terminal,
            "start_position": start.tolist(),
            "target_positions": [centre.tolist() for centre, _ in targets],
            "target_half_extents": [half.tolist() for _, half in targets],
            "success_definition": "horizontal distance to any queried instance <= 1.0 m",
            "map_updates": "none: the method's static map, built from the release's static sequence",
            "navigator": "DualMap calculate_global_path + snapped last leg; released collector follower",
            "raw_layout_object_count": len(layout["objects"]),
            "runtime_object_count": len(objects),
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        service.sim.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rankings", required=True,
                        help="path to a scene's rankings.json, with {scene} in place of the scene")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=native.COLLECTOR_SEED,
                        help="collector seed; controls the agent start pose (default: %(default)s)")
    parser.add_argument("--scene", help="full scene name or numeric prefix")
    parser.add_argument("--condition", choices=native.CONDITIONS)
    parser.add_argument("--trial", help="run one exact protocol trial ID")
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()

    native.COLLECTOR_SEED = args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.WARNING)
    trials = protocol()
    selected = native.select_trials(trials, args)
    if not selected:
        raise SystemExit("no protocol trials matched the selection")
    native.write_manifest(args.output, trials)
    rankings_by_scene: Dict[str, Dict[str, Any]] = {}
    for scene in sorted({t["scene"] for t in selected}):
        path = Path(args.rankings.format(scene=scene)).resolve()
        rankings_by_scene[scene] = json.loads(path.read_text(encoding="utf-8"))
    # DualMap's modules resolve their config and model paths from its checkout.
    os.chdir(native.DUALMAP_ROOT)
    for index, trial in enumerate(selected, start=1):
        out = args.output / "trials" / trial["trial_id"]
        if (out / "result.json").is_file() and not args.overwrite:
            print(f"SKIP {index}/{len(selected)} {trial['trial_id']}", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"RUN {index}/{len(selected)} {trial['trial_id']}", flush=True)
        try:
            result = run_trial(trial, rankings_by_scene[trial["scene"]], out)
            print(f"DONE {trial['trial_id']} success={result['success']} spl={result['spl']:.3f} "
                  f"attempts={result['attempt_count']} final={result['final_distance_horizontal_m']:.2f}m",
                  flush=True)
        except Exception as exc:
            failure = {**trial, "error": type(exc).__name__, "message": str(exc)}
            (out / "error.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
            logging.exception("trial failed: %s", trial["trial_id"])
        summary = native.aggregate(args.output, trials)
        print(f"PROGRESS {summary['completed']}/{summary['expected']} SR={summary['sr']} SPL={summary['spl']}",
              flush=True)


if __name__ == "__main__":
    main()
