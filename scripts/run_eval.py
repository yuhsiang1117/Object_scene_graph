"""Hydra entrypoint for HM3D ObjectNav evaluation.

Examples:
    python scripts/run_eval.py eval=hm3d_val_mini
    python scripts/run_eval.py eval=hm3d_val eval.num_episodes=200 detector=yoloe
    python scripts/run_eval.py --multirun +ablation=full,no_verify,paper_baseline,no_llm
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from osg.core.config import register_configs

register_configs()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from osg.eval.runner import run_eval

    run_eval(cfg)


if __name__ == "__main__":
    main()
