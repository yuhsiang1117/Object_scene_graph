"""Dump every keyframe's full detection set (every object YOLOE reports,
across the whole vocabulary -- not just the ObjectNav target) as annotated
images, one per episode subfolder. This is exactly what feeds
object_layer.update() when the scene graph is built (see NavAgent.
on_keyframe_detections, a debug hook fired right after the keyframe
detector call in _on_keyframe()) -- not a resampled/independent detector
call like the other diagnostic scripts in this directory.

Usage:
  python scripts/detection_dump.py --num-episodes 3 --max-steps 300
  python scripts/detection_dump.py --episode-ids 10,12 --max-steps 500
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from hydra import compose, initialize_config_dir

from osg.core.config import register_configs

register_configs()
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
    cfg = compose(config_name="config", overrides=["eval=hm3d_val_mini"])

from osg.agent.nav_agent import NavAgent  # noqa: E402
from osg.eval.runner import _unload_ollama_models, build_detector, build_scorer, build_verifier  # noqa: E402
from osg.sim.habitat_env import HabitatObjectNavEnv  # noqa: E402

_PALETTE = [
    (60, 60, 220), (60, 200, 60), (220, 140, 40), (200, 60, 200),
    (40, 200, 220), (200, 200, 40), (120, 120, 220), (60, 220, 140),
]


def _color_for(label: str) -> tuple:
    return _PALETTE[hash(label) % len(_PALETTE)]


def _save_annotated(path: Path, rgb: np.ndarray, dets) -> None:
    img = np.ascontiguousarray(rgb[..., ::-1])  # rgb -> bgr
    for d in dets:
        color = _color_for(d.label)
        x1, y1, x2, y2 = d.bbox_xyxy.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{d.label} {d.score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1, th + 4)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(img, text, (x1 + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-episodes", type=int, default=3)
    ap.add_argument("--episode-ids", default=None, help="comma-separated episode_id filter (any target)")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    wanted_ids = set(args.episode_ids.split(",")) if args.episode_ids else None
    out_root = Path(args.out_dir) if args.out_dir else Path(f"outputs/detection_dump_{time.strftime('%Y%m%d_%H%M%S')}")
    out_root.mkdir(parents=True, exist_ok=True)

    _unload_ollama_models(cfg)
    env = HabitatObjectNavEnv(cfg)
    detector = build_detector(cfg)
    scorer = build_scorer(cfg)
    verifier = build_verifier(cfg)

    n_total = len(env.env.episodes)
    n_run = 0
    for ep_i in range(n_total):
        if wanted_ids is None and n_run >= args.num_episodes:
            break
        frame = env.reset()
        episode = env.current_episode
        if wanted_ids is not None and str(episode.episode_id) not in wanted_ids:
            continue
        n_run += 1
        target = env.target_category()
        agent = NavAgent(cfg, detector, scorer, verifier, target, profiler=None)

        ep_dir = out_root / f"ep{episode.episode_id}_{target}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        kf_counter = [0]

        def _on_kf(fr, dets, ep_dir=ep_dir, kf_counter=kf_counter):
            kf_counter[0] += 1
            fname = ep_dir / f"kf{kf_counter[0]:04d}_step{fr.frame_id}_n{len(dets)}.png"
            _save_annotated(fname, fr.rgb, dets)

        agent.on_keyframe_detections = _on_kf

        step = 0
        while not env.episode_over and step < args.max_steps:
            action = agent.act(frame)
            frame = env.step(action)
            step += 1

        print(f"[{n_run}] ep={episode.episode_id} target={target} steps={step} "
              f"keyframes={kf_counter[0]} -> {ep_dir}")

    env.close()
    scorer.shutdown()
    print(f"\nAll detection images written under: {out_root}")


if __name__ == "__main__":
    main()
