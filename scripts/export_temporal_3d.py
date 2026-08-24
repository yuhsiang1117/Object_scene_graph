"""Export a time-indexed sequence of the scene graph state built during one
episode -- costmap (free/occupied), room segmentation, object ellipsoids and
the agent trajectory -- so a 3D viewer can scrub/play the map growing over
time (see scripts/temporal_3d_viewer.html.template).

Snapshots are taken every `--every-kf` keyframes; each snapshot downsamples the
costmap to `--tile-m` ground tiles (free tiles carry their room id) and records
every object track's current ellipsoid, plus the agent pose. The full
trajectory is stored once with a per-point step index so the viewer can draw it
up to the current frame's step.

Usage (local LLM, no API key needed):
  python scripts/export_temporal_3d.py --episode-id 3 --target sofa --max-steps 160
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from osg.core.config import register_configs

register_configs()
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
    cfg = compose(config_name="config", overrides=[
        "eval=hm3d_val_single_floor", "llm=ollama",  # local, self-contained
    ])

from osg.pipeline.components import build_detector, build_scorer  # noqa: E402
from osg.agent.nav_agent import NavAgent  # noqa: E402
from osg.mapping.costmap import FREE, OCCUPIED, PLANE  # noqa: E402
from osg.sim.habitat_env import HabitatObjectNavEnv  # noqa: E402


def _snapshot(agent, tile_px):
    """Downsample the current costmap to tiles and record ellipsoids + pose."""
    cm = agent.costmap
    grid = cm.grid
    h, w = grid.shape
    room = agent._room_labels if agent._room_labels is not None and agent._room_labels.shape == grid.shape else None
    free, occ = [], []
    for r in range(0, h - tile_px, tile_px):
        for c in range(0, w - tile_px, tile_px):
            blk = grid[r:r + tile_px, c:c + tile_px]
            xy = cm.grid_to_world(np.array([r + tile_px / 2.0, c + tile_px / 2.0]))
            pt = [round(float(xy[0]), 2), round(float(xy[1]), 2)]
            if (blk == OCCUPIED).any():
                occ += pt
            elif (blk == FREE).any():
                rid = 0
                if room is not None:
                    rblk = room[r:r + tile_px, c:c + tile_px]
                    rv = rblk[rblk > 0]
                    if rv.size:
                        rid = int(np.bincount(rv).argmax())
                free += pt + [rid]
    ellipsoids = []
    for t in agent.object_layer.tracks():
        if t.n_obs < 2:  # skip single-view noise so the timeline stays readable
            continue
        e = t.ellipsoid
        ellipsoids.append({
            "id": t.id, "label": t.label,
            "center": [round(float(x), 3) for x in e.center],
            "axes": [round(float(x), 3) for x in e.axes],
            "R": [[round(float(x), 4) for x in row] for row in e.R],
            "is_target": t.label.lower().replace(" ", "_") == agent.target.lower().replace(" ", "_"),
        })
    return {"free": free, "occ": occ, "ellipsoids": ellipsoids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-id", default="3")
    ap.add_argument("--target", default="sofa")
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--every-kf", type=int, default=2, help="snapshot every N keyframes")
    ap.add_argument("--tile-m", type=float, default=0.15, help="ground-tile downsample size")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    detector = build_detector(cfg)
    scorer = build_scorer(cfg)
    env = HabitatObjectNavEnv(cfg)

    picked = None
    for _ in range(len(env.env.episodes)):
        frame = env.reset()
        ep = env.current_episode
        if str(ep.episode_id) == args.episode_id and env.target_category() == args.target:
            picked = (ep, frame)
            break
    if picked is None:
        raise SystemExit(f"episode_id={args.episode_id} target={args.target} not found")
    ep, frame = picked
    target = env.target_category()

    agent = NavAgent(cfg, detector, scorer, None, target, profiler=None)
    tile_px = max(1, int(args.tile_m / cfg.mapping.resolution_m))

    frames = []
    trajectory = [[round(float(x), 3) for x in frame.camera_position[list(PLANE)]] + [0]]
    step, last_kf_snapped = 0, -999
    while not env.episode_over and step < args.max_steps:
        frame = env.step(agent.act(frame))
        step += 1
        trajectory.append([round(float(x), 3) for x in frame.camera_position[list(PLANE)]] + [step])
        # snapshot on keyframe boundaries
        if agent._kf_count - last_kf_snapped >= args.every_kf:
            last_kf_snapped = agent._kf_count
            snap = _snapshot(agent, tile_px)
            snap["step"] = step
            snap["n_kf"] = agent._kf_count
            snap["agent_xy"] = [round(float(x), 3) for x in frame.camera_position[list(PLANE)]]
            frames.append(snap)
    m = env.metrics()
    # final snapshot
    snap = _snapshot(agent, tile_px)
    snap["step"] = step
    snap["n_kf"] = agent._kf_count
    snap["agent_xy"] = [round(float(x), 3) for x in frame.camera_position[list(PLANE)]]
    frames.append(snap)
    env.close()
    scorer.shutdown()

    floor_y = float(agent._floor_y) if agent._floor_y is not None else 0.0
    data = {
        "episode_id": str(ep.episode_id), "target": target,
        "success": float(m.get("success", 0.0)), "spl": float(m.get("spl", 0.0)),
        "steps": step, "resolution": cfg.mapping.resolution_m, "tile_m": args.tile_m,
        "plane_axes": list(PLANE), "floor_y": floor_y,
        "trajectory": trajectory, "frames": frames,
    }
    out = Path(args.out) if args.out else Path(f"outputs/temporal_3d_ep{ep.episode_id}_{target}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(data, separators=(",", ":"))
    out.write_text(data_json)

    # Fill the viewer template: a standalone HTML you can open in a browser, and
    # an artifact-ready fragment (the slice between the ARTIFACT_CONTENT markers,
    # no <html>/<head>/<body> wrapper).
    tpl = Path("scripts/temporal_3d_viewer.html.template")
    if tpl.exists():
        t = tpl.read_text()
        token = "/*__DATA__*/ null"  # placeholder + JS fallback, replaced wholesale
        standalone = t.replace(token, data_json)
        out.with_suffix(".html").write_text(standalone)
        a, b = t.find("<!--ARTIFACT_CONTENT_START-->"), t.find("<!--ARTIFACT_CONTENT_END-->")
        if a >= 0 and b > a:
            frag = t[a:b].replace(token, data_json)
            out.with_name(out.stem + "_artifact.html").write_text(frag)

    print(f"wrote {out}: {len(frames)} time-frames, {len(trajectory)} traj pts, "
          f"success={data['success']:.0f} steps={step}")


if __name__ == "__main__":
    main()
