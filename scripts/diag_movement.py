"""P0 diagnosis: why doesn't the agent move? Pure navigation stack —
stub detector, nearest scorer, no LLM. Logs per-step state/action/position,
plan stats, and dumps costmap + inflated planning view."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from osg.agent.nav_agent import NavAgent
from osg.core.config import register_configs
from osg.exploration.async_scorer import AsyncScorer
from osg.exploration.scorer import NearestScorer
from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN, PLANE
from osg.perception.detector import StubDetector
from osg.sim.habitat_env import HabitatObjectNavEnv

register_configs()
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
    cfg = compose(config_name="config", overrides=[
        "eval=hm3d_val_mini", "agent.max_steps=200", "verification=off",
    ])

env = HabitatObjectNavEnv(cfg)
frame = env.reset()
agent = NavAgent(cfg, StubDetector(), AsyncScorer(NearestScorer()), None, env.target_category())

positions = [frame.camera_position[list(PLANE)].copy()]
actions = Counter()
state_at = []
n = 0
while not env.episode_over and n < 200:
    action = agent.act(frame)
    actions[action] += 1
    state_at.append(agent.state.value)
    frame = env.step(action)
    positions.append(frame.camera_position[list(PLANE)].copy())
    n += 1
    if n % 25 == 0:
        pos = positions[-1]
        print(f"step {n:3d} state={agent.state.value:14s} pos=({pos[0]:6.2f},{pos[1]:6.2f}) "
              f"stats={agent.stats} blocked={sum(1 for _, u in agent._blocked_frontier_pts if u > agent.step_count)}")

positions = np.stack(positions)
disp = np.linalg.norm(positions - positions[0], axis=1).max()
path_len = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
print("\n=== RESULT ===")
print("actions:", dict(actions))
print("states:", dict(Counter(state_at)))
print("stats:", agent.stats)
print(f"max displacement from start: {disp:.2f} m, path length: {path_len:.2f} m")
print("state transitions:", agent.state_log[:25])

g = agent.costmap.grid
print(f"costmap: free={int((g==FREE).sum())} occ={int((g==OCCUPIED).sum())} unknown={int((g==UNKNOWN).sum())}")

# --- Plan probe: WHY does planning to frontiers fail from here? ---
from osg.exploration.selector import frontier_goal_xy

np.savez("outputs/diag_costmap.npz", grid=g, origin=agent.costmap.origin,
         resolution=agent.costmap.resolution, agent_xy=positions[-1])
frontiers = agent.frontier_extractor.extract(agent.costmap)
print(f"=== PLAN PROBE ({len(frontiers)} frontiers) ===")
for f in sorted(frontiers, key=lambda f: -f.size)[:6]:
    goal = frontier_goal_xy(f, agent.costmap)
    res = agent.planner.plan(agent.costmap, positions[-1], goal)
    why = "OK cost=%.2f" % res.cost if res.success else agent.planner.last_failure
    grc = agent.costmap.world_to_grid(goal)
    print(f"frontier id={f.id:4d} size={f.size:4d} goal=({goal[0]:6.2f},{goal[1]:6.2f}) "
          f"goal_cell={g[grc[0], grc[1]]:3d} -> {why}")

infl = agent.costmap.inflated(agent.planner.inflate_radius_m)
plannable = (g == FREE) & ~infl
print(f"inflate_radius={agent.planner.inflate_radius_m}m -> plannable free cells: {int(plannable.sum())} / {int((g==FREE).sum())}")

# Dump visual: raw grid vs plannable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
img = np.full((*g.shape, 3), 0.5); img[g == FREE] = (1, 1, 1); img[g == OCCUPIED] = (0, 0, 0)
axes[0].imshow(img); axes[0].set_title("costmap (white=free)")
img2 = img.copy(); img2[infl & (g == FREE)] = (1, 0.6, 0.6)
rc = agent.costmap.world_to_grid(positions[-1]); img2[max(0,rc[0]-2):rc[0]+3, max(0,rc[1]-2):rc[1]+3] = (0, 0, 1)
axes[1].imshow(img2); axes[1].set_title("pink=free-but-inflated, blue=agent")
fig.savefig("outputs/diag_movement.png", dpi=110, bbox_inches="tight")
print("viz -> outputs/diag_movement.png")
env.close()
