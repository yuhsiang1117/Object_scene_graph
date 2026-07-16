"""Measure the real "how close can the agent actually stand" distance for a
handful of episodes, to test the P1f-followup hypothesis that costmap
inflation (mapping.inflate_margin_m + agent.agent_radius) is what caps how
close APPROACH's final position gets to the object -- not tolerance slack.

For each episode this re-runs the full pipeline (same components as
run_eval.run_eval) and, right after the episode ends, measures on the
agent's own live costmap:
  - nearest_occ_from_final: distance from final_xy to the nearest OCCUPIED
    (raw sensed obstacle surface) cell.
  - nearest_occ_from_goal: same, but from the APPROACH goal
    (_nearest_free_xy(obj_xy) -- what the agent was actually walking toward).
  - inflate_radius_m: agent_radius + inflate_margin_m, the soft-cost band
    width, for reference.
It also looks up HM3D's real view_points for the same episode (as
viewpoint_geometry_check.py does) to relate the standoff distance to the
actual success-radius gap.

Usage: python scripts/standoff_check.py <episode_id>:<target> [<episode_id>:<target> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from scipy import ndimage

from osg.core.config import register_configs

register_configs()
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
    cfg = compose(config_name="config", overrides=["eval=hm3d_val_mini"])

from osg.agent.nav_agent import NavAgent  # noqa: E402
from osg.eval.runner import _unload_ollama_models, build_detector, build_scorer, build_verifier  # noqa: E402
from osg.mapping.costmap import OCCUPIED, PLANE  # noqa: E402
from osg.sim.habitat_env import HabitatObjectNavEnv  # noqa: E402

wanted = set()
for tok in sys.argv[1:]:
    eid, target = tok.split(":")
    wanted.add((eid, target))

_unload_ollama_models(cfg)  # free VRAM for YOLOE's fp32 load, same as run_eval.py

env = HabitatObjectNavEnv(cfg)
lookup: dict = {}
for ep in env.env.episodes:
    scene = str(ep.scene_id).split("/")[-1]
    lookup.setdefault(scene, {})[str(ep.episode_id)] = ep

detector = build_detector(cfg)
scorer = build_scorer(cfg)
verifier = build_verifier(cfg)

inflate_radius_m = cfg.agent.agent_radius + cfg.mapping.inflate_margin_m
print(f"inflate_radius_m (agent_radius + inflate_margin_m) = {inflate_radius_m:.3f} m")
print(f"costmap resolution = {cfg.mapping.resolution_m} m")
print()
header = (f"{'ep':>3} {'target':11s} {'succ':4s} {'dtg':>6s} {'occ<-final':>10s} "
          f"{'occ<-goal':>9s} {'final<-goal':>11s} {'vp_2d':>7s} note")
print(header)

n_total = len(env.env.episodes)
for ep_i in range(n_total):
    frame = env.reset()
    episode = env.current_episode
    eid = str(episode.episode_id)
    target = env.target_category()
    if (eid, target) not in wanted:
        continue

    profiler_agent = NavAgent(cfg, detector, scorer, verifier, target, profiler=None)
    trajectory = [frame.camera_position[list(PLANE)]]
    while not env.episode_over:
        action = profiler_agent.act(frame)
        frame = env.step(action)
        trajectory.append(frame.camera_position[list(PLANE)])

    m = env.metrics()
    final_xy = np.asarray(trajectory[-1])
    cm = profiler_agent.costmap
    occ = cm.grid == OCCUPIED
    if occ.any():
        dist_to_occ = ndimage.distance_transform_edt(~occ) * cm.resolution
        final_rc = cm.world_to_grid(final_xy)
        occ_from_final = float(dist_to_occ[final_rc[0], final_rc[1]])
    else:
        occ_from_final = float("nan")

    goal_xy = getattr(profiler_agent, "_goal_xy", None)
    if goal_xy is not None and occ.any():
        goal_rc = cm.world_to_grid(goal_xy)
        occ_from_goal = float(dist_to_occ[goal_rc[0], goal_rc[1]])
        final_from_goal = float(np.linalg.norm(final_xy - goal_xy))
    else:
        occ_from_goal = float("nan")
        final_from_goal = float("nan")

    scene = str(episode.scene_id).split("/")[-1]
    hep = lookup.get(scene, {}).get(eid)
    best_2d = float("nan")
    if hep is not None:
        for g in hep.goals:
            for vp in g.view_points:
                p = np.array(vp.agent_state.position)
                d2 = float(np.linalg.norm(p[[0, 2]] - final_xy))
                if d2 < best_2d or np.isnan(best_2d):
                    best_2d = d2

    print(f"{eid:>3} {target:11s} {m.get('success', 0.0):4.0f} "
          f"{m.get('distance_to_goal', -1.0):6.3f} {occ_from_final:10.3f} "
          f"{occ_from_goal:9.3f} {final_from_goal:11.3f} {best_2d:7.3f} "
          f"stop={profiler_agent.approach_stop_reason}")

env.close()
