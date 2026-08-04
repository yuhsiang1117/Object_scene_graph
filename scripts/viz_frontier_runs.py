"""Run N episodes and, per episode, render a top-down map showing the walked
trajectory and every frontier the agent *selected* (numbered in selection
order, connected in sequence) over the costmap. Also emits a contact sheet of
all episodes and a summary line per episode.

Usage (local, no API key):
  python scripts/viz_frontier_runs.py --num 30 --max-steps 200
Outputs: outputs/frontier_viz/ep<i>_<id>_<target>.png (+ _contact.png, index.txt)
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from osg.core.config import register_configs

register_configs()
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
    cfg = compose(config_name="config", overrides=["eval=hm3d_val_single_floor", "llm=ollama"])

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from osg.eval.runner import build_detector, build_scorer  # noqa: E402
from osg.agent.nav_agent import NavAgent  # noqa: E402
from osg.mapping.costmap import FREE, OCCUPIED, PLANE  # noqa: E402
from osg.sim.habitat_env import HabitatObjectNavEnv  # noqa: E402


def _costmap_img(costmap):
    grid = costmap.grid
    img = np.full((*grid.shape, 3), 0.55)
    img[grid == FREE] = (0.95, 0.95, 0.95)
    img[grid == OCCUPIED] = (0.15, 0.15, 0.2)
    extent = [costmap.origin[0], costmap.origin[0] + grid.shape[0] * costmap.resolution,
              costmap.origin[1], costmap.origin[1] + grid.shape[1] * costmap.resolution]
    return np.transpose(img, (1, 0, 2)), extent


def _draw(ax, rec):
    img, extent = rec["img"], rec["extent"]
    ax.imshow(img, origin="lower", extent=extent)
    traj = np.asarray(rec["traj"])
    if len(traj):
        ax.plot(traj[:, 0], traj[:, 1], color="orange", lw=1.6, alpha=0.9, zorder=2)
        ax.plot(traj[0, 0], traj[0, 1], "o", color="tab:blue", ms=7, zorder=5)   # start
        ax.plot(traj[-1, 0], traj[-1, 1], "X", color="black", ms=8, zorder=5)    # end
    sel = rec["sel"]
    if sel:
        fx = np.asarray([s["frontier_xy"] for s in sel])
        # sequence line between consecutive selected frontiers
        ax.plot(fx[:, 0], fx[:, 1], "--", color="magenta", lw=0.8, alpha=0.5, zorder=3)
        colors = plt.cm.viridis(np.linspace(0, 1, len(fx)))
        ax.scatter(fx[:, 0], fx[:, 1], c=colors, s=90, edgecolors="k",
                   linewidths=0.6, zorder=4)
        for i, (x, y) in enumerate(fx):
            ax.annotate(str(i + 1), (x, y), ha="center", va="center",
                        fontsize=6, color="white", zorder=6)
    for xy in rec["targets"]:
        ax.plot(xy[0], xy[1], "*", color="red", ms=13, zorder=5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()

    detector = build_detector(cfg)
    scorer = build_scorer(cfg)
    env = HabitatObjectNavEnv(cfg)
    out = Path("outputs/frontier_viz")
    out.mkdir(parents=True, exist_ok=True)

    records, index = [], []
    for ep_i in range(min(args.num, len(env.env.episodes))):
        frame = env.reset()
        ep = env.current_episode
        target = env.target_category()
        scorer.reset()
        agent = NavAgent(cfg, detector, scorer, None, target, profiler=None)

        sel = []
        orig = agent._select_new_frontier
        def wrapped(fr, _o=orig, _a=agent, _s=sel):
            before = _a._current_frontier; nb = _a.stats.get("select_ok", 0)
            _o(fr)
            f = _a._current_frontier
            if f is not None and f is not before and _a.stats.get("select_ok", 0) > nb:
                _s.append({"step": _a.step_count,
                           "frontier_xy": [round(float(x), 3) for x in f.centroid_xy]})
        agent._select_new_frontier = wrapped

        traj = [frame.camera_position[list(PLANE)].copy()]
        step = 0
        while not env.episode_over and step < args.max_steps:
            frame = env.step(agent.act(frame)); step += 1
            traj.append(frame.camera_position[list(PLANE)].copy())
        m = env.metrics()
        succ = float(m.get("success", 0.0))

        # target-category object footprints found in the scene graph
        targets = []
        for t in agent.object_layer.tracks():
            if t.label.lower().replace(" ", "_") == target.lower().replace(" ", "_"):
                c = agent.object_layer.center_of(t)[list(PLANE)]
                targets.append([float(c[0]), float(c[1])])
        img, extent = _costmap_img(agent.costmap)
        rec = dict(idx=ep_i, ep=str(ep.episode_id), target=target, succ=succ, steps=step,
                   n_sel=len(sel), img=img, extent=extent, traj=traj, sel=sel, targets=targets)
        records.append(rec)

        fig, ax = plt.subplots(figsize=(7, 7))
        _draw(ax, rec)
        tag = "OK" if succ >= 1 else "fail"
        ax.set_title(f"ep{ep.episode_id} {target} [{tag}] steps={step} "
                     f"selections={len(sel)}", fontsize=11)
        fig.savefig(out / f"ep{ep_i:02d}_{ep.episode_id}_{target}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        index.append(f"ep{ep_i:02d} id={ep.episode_id} {target:<10} {tag:<4} "
                     f"steps={step:>3} selections={len(sel):>2} target_objs={len(targets)}")
        print(index[-1])

    env.close(); scorer.shutdown()

    # contact sheet
    n = len(records); cols = 6; rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    for k, ax in enumerate(np.asarray(axes).ravel()):
        if k < n:
            _draw(ax, records[k])
            r = records[k]
            ax.set_title(f"ep{r['ep']} {r['target']} {'OK' if r['succ']>=1 else 'x'} "
                         f"s={r['n_sel']}", fontsize=8)
        else:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "_contact.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    (out / "index.txt").write_text("\n".join(index) + "\n")
    sr = sum(1 for r in records if r["succ"] >= 1)
    print(f"\nwrote {n} episode figs + _contact.png to {out}  SR={sr}/{n}")


if __name__ == "__main__":
    main()
