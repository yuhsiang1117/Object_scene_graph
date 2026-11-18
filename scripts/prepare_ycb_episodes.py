#!/usr/bin/env python3
"""Discover authored YCB layouts and precompute deterministic episode manifests."""
from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from osg.core.config import register_configs

register_configs()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if str(cfg.eval.mode) != "ycb_authored":
        raise ValueError("select a YCB experiment, for example +experiment=ycb_authored_nav")
    from osg.sim.ycb_env import prepare_ycb_benchmark

    prepared = prepare_ycb_benchmark(cfg)
    print(
        json.dumps(
            {
                "layouts": [
                    f"{layout.scene_name}/{layout.layout_id}"
                    for layout in prepared.discovery.layouts
                ],
                "skipped": list(prepared.discovery.skipped),
                "manifests": [str(path) for path in prepared.cache_files],
                "episodes": sum(len(value["episodes"]) for value in prepared.manifests),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
