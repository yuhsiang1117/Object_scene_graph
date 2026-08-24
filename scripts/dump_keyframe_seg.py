"""Run one episode and save every keyframe's YOLOE segmentation as an image
overlay, for eyeballing what perception actually sees (the remaining bottleneck
once navigation freezes are fixed).

Each saved frame shows the RGB with translucent per-instance masks, bounding
boxes and label(score); detections of the episode's target category are drawn
in a bright highlight colour so it's obvious whether/when the target is seen.

Usage (local, no API key):
  python scripts/dump_keyframe_seg.py --episode-id 20 --target chair --max-steps 200
Outputs: outputs/seg/ep<ID>_<target>/kf_<NNNN>_step<SSS>.jpg  (+ index.txt)
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

from osg.pipeline.components import build_detector, build_scorer  # noqa: E402
from osg.agent.nav_agent import NavAgent  # noqa: E402
from osg.sim.habitat_env import HabitatObjectNavEnv  # noqa: E402

# distinct BGR colours for non-target instances (cycled)
PALETTE = [(80, 180, 255), (80, 255, 140), (255, 170, 80), (200, 120, 255),
           (80, 220, 255), (255, 120, 160), (150, 255, 80), (255, 210, 90)]
TARGET_BGR = (60, 60, 255)  # red-ish highlight for the target category


def _norm(s: str) -> str:
    return s.lower().replace("_", " ").strip()


def overlay(rgb: np.ndarray, dets, target: str) -> np.ndarray:
    img = rgb[..., ::-1].copy()  # RGB -> BGR for cv2
    tgt = _norm(target)
    layer = img.copy()
    for i, d in enumerate(dets):
        is_t = _norm(d.label) == tgt
        col = TARGET_BGR if is_t else PALETTE[i % len(PALETTE)]
        m = d.mask.astype(bool)
        layer[m] = col
    img = cv2.addWeighted(layer, 0.45, img, 0.55, 0)
    for i, d in enumerate(dets):
        is_t = _norm(d.label) == tgt
        col = TARGET_BGR if is_t else PALETTE[i % len(PALETTE)]
        x1, y1, x2, y2 = d.bbox_xyxy.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if is_t else 1)
        txt = f"{d.label} {d.score:.2f}"
        cv2.putText(img, txt, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, col, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-id", default="20")
    ap.add_argument("--target", default="chair")
    ap.add_argument("--max-steps", type=int, default=200)
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
            picked = (ep, frame); break
    if picked is None:
        raise SystemExit(f"episode_id={args.episode_id} target={args.target} not found")
    ep, frame = picked
    target = env.target_category()

    out = Path(args.out) if args.out else Path(f"outputs/seg/ep{ep.episode_id}_{target}")
    out.mkdir(parents=True, exist_ok=True)
    index = []

    agent = NavAgent(cfg, detector, scorer, None, target, profiler=None)

    def on_kf(fr, dets):
        kf = agent._kf_count
        img = overlay(fr.rgb, dets, target)
        n_t = sum(1 for d in dets if _norm(d.label) == _norm(target))
        fn = out / f"kf_{kf:04d}_step{agent.step_count:03d}.jpg"
        cv2.imwrite(str(fn), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        labels = ",".join(sorted({d.label for d in dets}))
        index.append(f"kf {kf:>3} step {agent.step_count:>3}  dets {len(dets):>2}  "
                     f"target_seen {n_t}  [{labels}]")

    agent.on_keyframe_detections = on_kf

    step = 0
    while not env.episode_over and step < args.max_steps:
        frame = env.step(agent.act(frame)); step += 1
    m = env.metrics()
    env.close(); scorer.shutdown()

    (out / "index.txt").write_text("\n".join(index) + "\n")
    n_seen = sum(1 for line in index if "target_seen 0" not in line)
    print(f"ep{ep.episode_id} {target}: success={m.get('success',0):.0f} steps={step} "
          f"keyframes={len(index)} target_seen_in={n_seen} -> {out}")


if __name__ == "__main__":
    main()
