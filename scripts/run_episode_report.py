"""Run one or more episodes end-to-end and emit a COMPLETE per-episode report:
  outputs/report/ep<id>_<target>/
    keyframes/kf_<NNNN>_step<SSS>.jpg   YOLOE segmentation overlay per keyframe
    trajectory.png                      top-down: costmap + path + selected frontiers
    summary.txt                         per-keyframe detection index + result line

One forward pass per episode produces both the perception (keyframe seg) and the
navigation (trajectory/frontier) views, using the current agent stack.

Usage (local, no API key):
  python scripts/run_episode_report.py --episodes 3:sofa,6:bed,17:toilet
  python scripts/run_episode_report.py --num 3            # first 3 in iteration order
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
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

PALETTE = [(80, 180, 255), (80, 255, 140), (255, 170, 80), (200, 120, 255),
           (80, 220, 255), (255, 120, 160), (150, 255, 80), (255, 210, 90)]
TARGET_BGR = (60, 60, 255)


def _norm(s: str) -> str:
    return s.lower().replace("_", " ").strip()


def seg_overlay(rgb, dets, target):
    img = rgb[..., ::-1].copy()
    tgt = _norm(target)
    layer = img.copy()
    for i, d in enumerate(dets):
        col = TARGET_BGR if _norm(d.label) == tgt else PALETTE[i % len(PALETTE)]
        layer[d.mask.astype(bool)] = col
    img = cv2.addWeighted(layer, 0.45, img, 0.55, 0)
    for i, d in enumerate(dets):
        is_t = _norm(d.label) == tgt
        col = TARGET_BGR if is_t else PALETTE[i % len(PALETTE)]
        x1, y1, x2, y2 = d.bbox_xyxy.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if is_t else 1)
        t = f"{d.label} {d.score:.2f}"
        cv2.putText(img, t, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return img


def save_trajectory(png, costmap, traj, sel, targets, title):
    grid = costmap.grid
    im = np.full((*grid.shape, 3), 0.55)
    im[grid == FREE] = (0.95, 0.95, 0.95); im[grid == OCCUPIED] = (0.15, 0.15, 0.2)
    extent = [costmap.origin[0], costmap.origin[0] + grid.shape[0] * costmap.resolution,
              costmap.origin[1], costmap.origin[1] + grid.shape[1] * costmap.resolution]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.transpose(im, (1, 0, 2)), origin="lower", extent=extent)
    traj = np.asarray(traj)
    if len(traj):
        ax.plot(traj[:, 0], traj[:, 1], color="orange", lw=1.6, zorder=2)
        ax.plot(traj[0, 0], traj[0, 1], "o", color="tab:blue", ms=7, zorder=5)
        ax.plot(traj[-1, 0], traj[-1, 1], "X", color="black", ms=8, zorder=5)
    if sel:
        fx = np.asarray([s["frontier_xy"] for s in sel])
        ax.plot(fx[:, 0], fx[:, 1], "--", color="magenta", lw=0.8, alpha=0.5, zorder=3)
        cols = plt.cm.viridis(np.linspace(0, 1, len(fx)))
        ax.scatter(fx[:, 0], fx[:, 1], c=cols, s=90, edgecolors="k", linewidths=0.6, zorder=4)
        for i, (x, y) in enumerate(fx):
            ax.annotate(str(i + 1), (x, y), ha="center", va="center", fontsize=6, color="white", zorder=6)
    for xy in targets:
        ax.plot(xy[0], xy[1], "*", color="red", ms=13, zorder=5)
    ax.set_title(title, fontsize=11); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(png, dpi=120, bbox_inches="tight"); plt.close(fig)


def run_one(env, agent_cfg, detector, scorer, ep, frame, target, max_steps, kf_stride=1):
    out = Path(f"outputs/report/ep{ep.episode_id}_{target}")
    (out / "keyframes").mkdir(parents=True, exist_ok=True)
    agent = NavAgent(agent_cfg, detector, scorer, None, target, profiler=None)
    kf_index = []
    sel = []
    orig = agent._select_new_frontier
    def wrapped(fr, _o=orig, _a=agent, _s=sel):
        before = _a._current_frontier; nb = _a.stats.get("select_ok", 0)
        _o(fr)
        f = _a._current_frontier
        if f is not None and f is not before and _a.stats.get("select_ok", 0) > nb:
            _s.append({"frontier_xy": [round(float(x), 3) for x in f.centroid_xy]})
    agent._select_new_frontier = wrapped

    def on_kf(fr, dets):
        if agent._kf_count % kf_stride == 0:  # thin keyframes for big sweeps
            img = seg_overlay(fr.rgb, dets, target)
            cv2.imwrite(str(out / "keyframes" / f"kf_{agent._kf_count:04d}_step{agent.step_count:03d}.jpg"),
                        img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        n_t = sum(1 for d in dets if _norm(d.label) == _norm(target))
        kf_index.append(f"kf {agent._kf_count:>3} step {agent.step_count:>3} dets {len(dets):>2} "
                        f"target_seen {n_t} [{','.join(sorted({d.label for d in dets}))}]")
    agent.on_keyframe_detections = on_kf

    traj = [frame.camera_position[list(PLANE)].copy()]
    step = 0
    while not env.episode_over and step < max_steps:
        frame = env.step(agent.act(frame)); step += 1
        traj.append(frame.camera_position[list(PLANE)].copy())
    m = env.metrics(); succ = float(m.get("success", 0.0))
    targets = []
    for t in agent.object_layer.tracks():
        if _norm(t.label) == _norm(target):
            c = agent.object_layer.center_of(t)[list(PLANE)]
            targets.append([float(c[0]), float(c[1])])
    tag = "OK" if succ >= 1 else "fail"
    save_trajectory(out / "trajectory.png", agent.costmap, traj, sel, targets,
                    f"ep{ep.episode_id} {target} [{tag}] steps={step} selections={len(sel)}")
    seen = sum(1 for l in kf_index if "target_seen 0" not in l)
    (out / "summary.txt").write_text(
        f"episode {ep.episode_id} target={target} result={tag} steps={step} "
        f"selections={len(sel)} keyframes={len(kf_index)} target_seen_in={seen} "
        f"target_objs_mapped={len(targets)}\n\n" + "\n".join(kf_index) + "\n")
    print(f"ep{ep.episode_id} {target}: {tag} steps={step} kf={len(kf_index)} "
          f"target_seen_in={seen} selections={len(sel)} -> {out}")
    return {"episode_id": str(ep.episode_id), "target": target, "success": succ,
            "spl": float(m.get("spl", 0.0)), "steps": step, "selections": len(sel),
            "keyframes": len(kf_index), "target_seen_in": seen,
            "target_objs_mapped": len(targets)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="", help="comma list id:target, e.g. 3:sofa,6:bed")
    ap.add_argument("--num", type=int, default=3, help="if --episodes empty, first N (-1 = all)")
    ap.add_argument("--max-steps", type=int, default=500)  # benchmark default budget
    ap.add_argument("--kf-stride", type=int, default=1, help="save every Nth keyframe (thin big sweeps)")
    args = ap.parse_args()

    detector = build_detector(cfg); scorer = build_scorer(cfg); env = HabitatObjectNavEnv(cfg)
    wanted = None
    if args.episodes:
        wanted = set(tuple(x.split(":")) for x in args.episodes.split(","))

    done = 0; results = []
    n_total = len(env.env.episodes)
    for _ in range(n_total):
        frame = env.reset(); ep = env.current_episode; target = env.target_category()
        if wanted is not None:
            if (str(ep.episode_id), target) not in wanted:
                continue
        scorer.reset()
        results.append(run_one(env, cfg, detector, scorer, ep, frame, target,
                               args.max_steps, args.kf_stride))
        done += 1
        if wanted is None and args.num >= 0 and done >= args.num:
            break
        if wanted is not None and done >= len(wanted):
            break
    env.close(); scorer.shutdown()

    import json
    sr = sum(1 for r in results if r["success"] >= 1)
    spl = sum(r["spl"] for r in results) / max(1, len(results))
    agg = {"episodes": done, "success_rate": sr / max(1, done), "spl": spl,
           "max_steps": args.max_steps, "per_episode": results}
    Path("outputs/report").mkdir(parents=True, exist_ok=True)
    Path("outputs/report/summary.json").write_text(json.dumps(agg, indent=2))
    print(f"\n{done} episodes -> outputs/report/  SR={sr}/{done}={sr/max(1,done):.3f} "
          f"SPL={spl:.3f}  (keyframes+trajectory.png per episode)")


if __name__ == "__main__":
    main()
