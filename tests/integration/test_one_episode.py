"""Integration: one short episode through the full agent stack with a stub
detector (isolates the pipeline from model weights). Requires habitat-sim,
HM3D minival scenes and the ObjectNav v2 episodes -> marked `sim`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.sim

habitat = pytest.importorskip("habitat")


def _missing() -> str:
    """Which half is absent -- the scenes or the episodes. They arrive
    separately and it is never obvious which one a bare skip meant."""
    from osg.core.paths import hm3d_scene_root

    if not hm3d_scene_root().is_dir():
        return f"HM3D scenes not mounted at {hm3d_scene_root()}"
    episodes = Path("data/datasets/objectnav/hm3d")
    if not (episodes.is_dir() and any(episodes.rglob("*.json.gz"))):
        return f"ObjectNav episodes not downloaded under {episodes} (scripts/download_data.py)"
    return ""


@pytest.mark.timeout(600)
def test_one_episode_runs():
    reason = _missing()
    if reason:
        pytest.skip(reason)

    from hydra import compose, initialize_config_dir

    from osg.agent.nav_agent import NavAgent
    from osg.core.config import register_configs
    from osg.exploration.async_scorer import AsyncScorer
    from osg.exploration.scorer import FrontierScorer

    class _StubScorer(FrontierScorer):
        def score(self, frontiers, sg, target, keyframes=None):
            return {f.id: 1.0 for f in frontiers}
    from osg.perception.detector import StubDetector
    from osg.sim.habitat_env import HabitatObjectNavEnv

    register_configs()
    cfg_dir = str(Path("configs").resolve())
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["eval=hm3d_val_mini", "agent.max_steps=100", "verification=off"],
        )

    env = HabitatObjectNavEnv(cfg)
    frame = env.reset()
    agent = NavAgent(
        cfg,
        detector=StubDetector(),
        scorer=AsyncScorer(_StubScorer()),
        verifier=None,
        target_category=env.target_category(),
    )
    steps = 0
    while not env.episode_over and steps < 100:
        frame = env.step(agent.act(frame))
        steps += 1

    assert steps > 5
    assert agent.costmap.coverage_cells() > 100  # map grew
    m = env.metrics()
    assert "success" in m and "spl" in m
    env.close()
