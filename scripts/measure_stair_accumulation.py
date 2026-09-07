"""Can accumulation turn a 19% per-pose stair signal into a detected staircase?

S14a measured YOLOE's per-POSE up-stair recall at 19% (11l @ 640; 0% on 11s).
S15 then found that the bigger detector produced no extra climb attempts at all
-- 5 to 4 over 21 cross-floor episodes. So the per-pose number does not survive
`StairDetector`, which requires `min_hits` frames on the same cells and a
component of at least `min_cells`.

This measures the thing that actually gates behaviour: **per-STAIRCASE** recall
after accumulation. For each cross-floor episode it walks the geodesic path,
folds every frame into a StairDetector exactly as NavAgent does, and then
extracts components under a sweep of (min_hits, min_cells) -- so the cost of
relaxing each threshold is visible against the false components it admits.

Ground truth is the navmesh, as in measure_stair_recall.py: on a cross-floor
episode the path from start to goal must cross a staircase, so the waypoints
where height climbs steeply are where one is. A component counts as a HIT if its
centroid lands within --hit-m of such a waypoint, and as a FALSE component
otherwise.

Run inside the nav container:

    python scripts/measure_stair_accumulation.py --episodes 12 -- eval=dev50_mf detector=yoloe
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from osg.core.config import register_configs

register_configs()

MIN_HITS = [1, 2, 3]
MIN_CELLS = [5, 10, 25]


class _Layer:
    """The handful of FloorLayer fields StairDetector touches."""

    def __init__(self, costmap, floor_y: float) -> None:
        self.costmap = costmap
        self.floor_y = floor_y
        self.up_stair_hits = None
        self.down_stair_hits = None
        self.disabled_stair = None


def _rising_waypoints(points: np.ndarray, rise_m: float = 0.3) -> np.ndarray:
    """Path waypoints whose next step climbs -- i.e. where the stairs are."""
    out = []
    for a, b in zip(points[:-1], points[1:]):
        if abs(b[1] - a[1]) >= rise_m:
            out.append(a)
    return np.array(out) if out else np.zeros((0, 3))


def run(cfg: DictConfig, n_episodes: int, hit_m: float, stride: int) -> None:
    import habitat_sim

    from osg.core.geometry import quat_to_matrix
    from osg.core.types import FrameData
    from osg.eval.runner import _episode_uid, _goal_floor_gap_m, build_detector
    from osg.mapping.costmap import PLANE, Costmap2D
    from osg.mapping.stairs import StairDetector
    from osg.sim.habitat_env import HabitatObjectNavEnv, _GL_TO_CV

    env = HabitatObjectNavEnv(cfg)
    # The episode_ids filter lives in runner.run_eval, which this script does
    # not use -- without applying it here the "cross-floor" split is ignored and
    # the sample is ordinary val episodes, whose geodesic paths are flat.
    if cfg.eval.episode_ids:
        wanted = {str(e) for e in cfg.eval.episode_ids}
        kept = [e for e in env.env.episodes if _episode_uid(e) in wanted]
        if kept:
            env.env.episodes = kept
    # Keep only episodes that genuinely need a floor change, whatever split was
    # given: a flat path has no staircase to find.
    cross = [e for e in env.env.episodes if (_goal_floor_gap_m(e) or 0.0) > 1.0]
    if cross:
        env.env.episodes = cross
    print(f"[info] {len(env.env.episodes)} cross-floor episodes available")
    sim = env.env.sim
    detector = build_detector(cfg)
    detector.set_vocabulary(["stairs"] + list(cfg.detector.vocabulary))
    intr = env.intrinsics

    # (min_hits, min_cells) -> [episodes with a hit, total false components]
    hits = defaultdict(int)
    false_c = defaultdict(int)
    used = 0
    skipped = defaultdict(int)

    for ep_i in range(n_episodes):
        env.env.reset()
        ep = env.env.current_episode
        start = np.array(ep.start_position, dtype=float)
        goals = [np.array(v.agent_state.position, dtype=float)
                 for g in ep.goals for v in getattr(g, "view_points", [])]
        if not goals:
            skipped["no view points"] += 1
            continue
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = min(goals, key=lambda g: abs(g[1] - start[1]))
        if not sim.pathfinder.find_path(path):
            skipped["no navmesh path"] += 1
            continue
        pts = np.array(path.points, dtype=float)
        truth = _rising_waypoints(pts)
        if truth.shape[0] == 0:
            skipped["path never rises"] += 1
            gaps = np.abs(np.diff(pts[:, 1])) if len(pts) > 1 else np.array([0.0])
            skipped[f"  (largest step {gaps.max():.2f} m over {len(pts)} waypoints)"] += 1
            continue
        used += 1

        cm = Costmap2D(resolution=cfg.mapping.resolution_m)
        layer = _Layer(cm, float(start[1]))
        det = StairDetector(resolution_m=cfg.mapping.resolution_m,
                            max_range_m=cfg.mapping.max_range_m, min_hits=1)

        # Walk the path, folding every frame in exactly as NavAgent does.
        for i in range(0, len(pts) - 1, stride):
            pos, nxt = pts[i], pts[i + 1]
            d = nxt - pos
            yaw = float(np.arctan2(-d[0], -d[2]))
            rot = np.quaternion(np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0)
            obs = sim.get_observations_at(pos.tolist(), rot, keep_agent_at_new_pose=False)
            if obs is None:
                continue
            R = quat_to_matrix(rot.w, rot.x, rot.y, rot.z)
            T_wc = np.eye(4)
            T_wc[:3, :3] = R @ _GL_TO_CV
            T_wc[:3, 3] = pos + np.array([0.0, cfg.agent.camera_height, 0.0])
            depth = obs["depth"]
            depth = depth[..., 0] if depth.ndim == 3 else depth
            frame = FrameData(frame_id=i, rgb=np.ascontiguousarray(obs["rgb"][..., :3]),
                              depth=depth.astype(np.float32), T_wc=T_wc, intrinsics=intr)
            cm.update(frame, floor_y=float(start[1]))
            det.accumulate(frame, layer, detector.detect(frame.rgb))

        # One accumulation, every threshold: the grids are already built, so the
        # sweep costs nothing beyond re-thresholding them.
        truth_xy = truth[:, list(PLANE)]
        for mh in MIN_HITS:
            for mc in MIN_CELLS:
                det.min_hits, det.min_cells = mh, mc
                comps = det.extract(layer)
                found = False
                for c in comps:
                    if np.linalg.norm(truth_xy - c.centroid_xy, axis=1).min() <= hit_m:
                        found = True
                    else:
                        false_c[(mh, mc)] += 1
                hits[(mh, mc)] += int(found)

    env.close()

    if skipped:
        print(f"\nskipped: {dict(skipped)}")
    print(f"\n=== per-STAIRCASE recall after accumulation ({used} cross-floor episodes) ===")
    print("what fraction of episodes end up with a stair component near the real "
          "staircase,\nand how many components land nowhere near one.\n")
    print(f"{'min_hits':>8s} {'min_cells':>10s} {'episodes with a hit':>21s} "
          f"{'false components':>18s}")
    for mh in MIN_HITS:
        for mc in MIN_CELLS:
            h, f = hits[(mh, mc)], false_c[(mh, mc)]
            mark = "   <- current default" if (mh, mc) == (3, 25) else ""
            print(f"{mh:>8d} {mc:>10d} {h:>14d}/{used:<6d} {f:>18d}{mark}")
    print("\nThe system-level number to beat: 4-5 of 21 cross-floor episodes"
          "\ncurrently attempt a climb at all (docs/AB_RESULTS S15).")


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--hit-m", type=float, default=2.0,
                    help="a component counts as found within this of a rising waypoint")
    ap.add_argument("--stride", type=int, default=1, help="path waypoints per frame")
    ap.add_argument("-h", "--help", action="store_true")
    args, hydra_argv = ap.parse_known_args()
    if args.help:
        ap.print_help()
        print(__doc__)
        return
    sys.argv = [sys.argv[0], *hydra_argv]

    @hydra.main(config_path="../configs", config_name="config", version_base="1.3")
    def _run(cfg: DictConfig) -> None:
        run(cfg, args.episodes, args.hit_m, args.stride)

    _run()


if __name__ == "__main__":
    main()
