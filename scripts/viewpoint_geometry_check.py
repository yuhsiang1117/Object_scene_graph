"""Compare our agent's final stop position against HM3D's actual view_points
geometry for a set of episodes, to determine whether near-miss failures are
a real geometric gap (position estimate / viewpoint selection error) or an
artifact of habitat's geodesic distance_to_goal metric diverging from
straight-line distance (e.g. routed around a wall).

Usage: python scripts/viewpoint_geometry_check.py <episodes.jsonl>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from osg.core.config import register_configs

register_configs()
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
    cfg = compose(config_name="config", overrides=["eval=hm3d_val_mini"])

from osg.sim.habitat_env import HabitatObjectNavEnv  # noqa: E402

env = HabitatObjectNavEnv(cfg)

# scene_basename -> episode_id -> habitat episode object
lookup: dict = {}
for ep in env.env.episodes:
    scene = str(ep.scene_id).split("/")[-1]
    lookup.setdefault(scene, {})[str(ep.episode_id)] = ep

rows = [json.loads(line) for line in open(sys.argv[1])]

print(f"{'ep':>3} {'scene':22s} {'target':11s} {'succ':4s} {'reported_dtg':12s} "
      f"{'nearest_vp_2d':13s} {'vp_world_y':13s} note")
for r in rows:
    scene = r["scene"]
    ep = lookup.get(scene, {}).get(r["episode_id"])
    if ep is None:
        print(f"{r['episode_id']:>3} {scene:22s} {r['target']:11s}  -- episode not found in current split --")
        continue
    final_xy = np.array(r.get("final_xy", [None, None]))
    if final_xy[0] is None:
        continue

    # Track the viewpoint that minimizes 2D (ground-plane) distance, since
    # that's what our system actually optimizes for (it has no notion of
    # height); report that same viewpoint's height gap alongside it.
    best_2d, best_2d_dy = float("inf"), None
    n_vp = 0
    for g in ep.goals:
        for vp in g.view_points:
            n_vp += 1
            p = np.array(vp.agent_state.position)  # (x, y, z)
            d2 = float(np.linalg.norm(p[[0, 2]] - final_xy))
            if d2 < best_2d:
                best_2d = d2
                best_2d_dy = float(p[1])

    start_y = float(ep.start_position[1])
    same_floor = "same-floor" if best_2d_dy is not None and abs(best_2d_dy - start_y) < 1.0 else "DIFF-FLOOR"
    note = f"n_vp={n_vp} start_y={start_y:.2f} {same_floor}"
    print(f"{r['episode_id']:>3} {scene:22s} {r['target']:11s} {r['success']:4.0f} "
          f"{r['distance_to_goal']:12.3f} {best_2d:13.3f} {best_2d_dy:13.3f} {note}")

env.close()
