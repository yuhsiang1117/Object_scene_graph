"""Record a scripted trajectory in one HM3D scene to .npz FrameData bundles
for offline pipeline development (M1/M2) — iterate on perception/mapping
without paying simulator startup per run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from osg.core.config import register_configs

register_configs()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from osg.sim.habitat_env import HabitatObjectNavEnv

    out = Path("data/recordings")
    out.mkdir(parents=True, exist_ok=True)

    env = HabitatObjectNavEnv(cfg)
    frame = env.reset()
    rng = np.random.default_rng(cfg.seed)

    frames = []
    # Scripted sweep: full spin, then random-ish forward exploration
    actions = ["turn_left"] * 12
    for _ in range(120):
        actions.append(str(rng.choice(["move_forward"] * 3 + ["turn_left", "turn_right"])))

    for action in actions:
        frames.append(
            dict(
                frame_id=frame.frame_id,
                rgb=frame.rgb,
                depth=frame.depth,
                T_wc=frame.T_wc,
                fx=frame.intrinsics.fx, fy=frame.intrinsics.fy,
                cx=frame.intrinsics.cx, cy=frame.intrinsics.cy,
            )
        )
        if env.episode_over:
            break
        frame = env.step(action)

    path = out / "recording.npz"
    np.savez_compressed(
        path,
        n=len(frames),
        **{f"f{i}_{k}": v for i, fr in enumerate(frames) for k, v in fr.items()},
    )
    print(f"saved {len(frames)} frames -> {path}")
    env.close()


if __name__ == "__main__":
    main()
