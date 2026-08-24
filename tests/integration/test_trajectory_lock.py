"""Behaviour locks: a real episode must take the identical actions.

Almost every constant in this pipeline was chosen by an experiment, and almost
every module is about to move to a different file. Unit tests prove the pieces
still work; they cannot prove the assembled agent still does the SAME thing --
and "the same thing" is the only claim a refactor of calibrated code is allowed
to make.

So: run a real authored YCB episode and hash the action sequence. Both fixtures
are deterministic, verified by running each twice:

  exploration   StubDetector, 120 steps, ~5 s, no GPU. Covers the costmap,
                keyframing, frontier extraction and selection, the blacklist and
                give-up nets, and navmesh following.
  dynamic       real YOLOE + presence filter + search posterior, 300 steps,
                ~55 s, needs the GPU. Covers everything above plus the belief
                updates, surface inspection, candidate admission, and the
                terminal approach -- the run reaches DONE at step 236 having
                inspected three surfaces and made 1501 belief updates.

Regenerate the goldens ONLY from behaviour you have decided is correct:

    python tests/integration/test_trajectory_lock.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.sim

pytest.importorskip("habitat")

DATA_ROOT = Path("/datasets/habitat-data-collector/data")
LAYOUT_ROOT = Path("/datasets/habitat-data-collector/outputs/dualmap_authoring")

# One scene, one target, one authored start -- the manifest is cached under
# outputs/ycb_manifests/, so the episode is byte-identical run to run.
SCENE = "00829-QaLdnwvtxbs"
TARGET = "bowl"

EXPLORATION_STEPS = 120
DYNAMIC_STEPS = 300

EXPLORATION_ACTIONS_SHA256 = "c14d33461af8a7e4c3e7002b51a3522ef08e14e8370cac56a81a74ae3d1ae31d"
DYNAMIC_ACTIONS_SHA256 = "b58c82b7f9dd4ae9abebd42a2988e660e9a7d49eacf50f8e26b37e0688c8075c"
# The mechanisms, not just the motion: a refactor that moved a belief update
# would change these long before it changed a foot placement.
DYNAMIC_FINGERPRINT: dict = {"state_log": [[13, "goto_frontier"], [212, "approach"], [236, "done"]], "presence_expected": 1260, "presence_positive": 1024, "presence_negative": 477, "presence_disbelieved": 12, "search_surface": 3, "frontier_reached": 6, "select_ok": 6, "select_none": 0, "plan_ok": 6, "plan_fail": 0, "n_search_events": 5, "n_presence_events": 17}


def _mounted() -> bool:
    return DATA_ROOT.is_dir() and LAYOUT_ROOT.is_dir()


def _config(extra=()):
    from hydra import compose, initialize_config_dir

    from osg.core.config import register_configs

    register_configs()
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[
                "+experiment=ycb_authored_nav",
                # No VLM anywhere: the README records that runs are not
                # reproducible while the verifier is on, and a lock that is not
                # reproducible locks nothing.
                "verification=off",
                f"ycb.scenes=[{SCENE}]",
                f"ycb.targets=[{TARGET}]",
                "eval.save_viz=false",
                "eval.debug_frames=false",
                *extra,
            ],
        )


def run_episode(max_steps: int, real_detector: bool, extra=()):
    """(actions, agent) for one episode, driven exactly as the runner drives it."""
    from osg.agent.nav_agent import NavAgent
    from osg.exploration.async_scorer import AsyncScorer
    from osg.exploration.scorer import NullScorer
    from osg.perception.detector import StubDetector
    from osg.sim.ycb_env import YCBAuthoredNavEnv

    cfg = _config(extra)
    env = YCBAuthoredNavEnv(cfg)
    try:
        if real_detector:
            from osg.pipeline.components import build_detector

            detector = build_detector(cfg)
        else:
            detector = StubDetector()
        frame = env.reset()
        agent = NavAgent(
            cfg, detector, AsyncScorer(NullScorer()), None, env.target_category(),
            nav_fn=env.action_to_goal if cfg.agent.use_habitat_navmesh else None,
            reachable_fn=env.is_reachable if cfg.agent.use_habitat_navmesh else None,
        )
        actions = []
        while not env.episode_over and len(actions) < max_steps:
            action = agent.act(frame)
            actions.append(action)
            frame = env.step(action)
        return actions, agent
    finally:
        env.close()


def action_hash(actions) -> str:
    return hashlib.sha256(",".join(actions).encode("utf-8")).hexdigest()


DYNAMIC_OVERRIDES = (
    "scene_graph.presence.enabled=true",
    "exploration.search_posterior=true",
)


def fingerprint(agent) -> dict:
    """The belief and search counters, which drift before the trajectory does."""
    stats = agent.stats
    return {
        "state_log": [list(entry) for entry in agent.state_log],
        "presence_expected": stats.get("presence_expected"),
        "presence_positive": stats.get("presence_positive"),
        "presence_negative": stats.get("presence_negative"),
        "presence_disbelieved": stats.get("presence_disbelieved"),
        "search_surface": stats.get("search_surface"),
        "frontier_reached": stats.get("frontier_reached"),
        "select_ok": stats.get("select_ok"),
        "select_none": stats.get("select_none"),
        "plan_ok": stats.get("plan_ok"),
        "plan_fail": stats.get("plan_fail"),
        "n_search_events": len(agent.search_log_events),
        "n_presence_events": len(agent.presence_events),
    }


@pytest.mark.timeout(300)
def test_exploration_trajectory_is_unchanged():
    if not _mounted():
        pytest.skip("collector data is not mounted")
    actions, _ = run_episode(EXPLORATION_STEPS, real_detector=False)
    assert len(actions) == EXPLORATION_STEPS
    assert action_hash(actions) == EXPLORATION_ACTIONS_SHA256, (
        "exploration behaviour changed: " + "".join(a[0] for a in actions)
    )


@pytest.mark.gpu
@pytest.mark.timeout(900)
def test_dynamic_pipeline_trajectory_is_unchanged():
    if not _mounted():
        pytest.skip("collector data is not mounted")
    actions, agent = run_episode(
        DYNAMIC_STEPS, real_detector=True, extra=DYNAMIC_OVERRIDES
    )
    # Report the mechanism first: "presence_negative 477 -> 471" says which
    # module moved, where a bare hash mismatch says only that something did.
    assert fingerprint(agent) == DYNAMIC_FINGERPRINT
    assert action_hash(actions) == DYNAMIC_ACTIONS_SHA256


if __name__ == "__main__":
    import json
    import re

    explore_actions, _ = run_episode(EXPLORATION_STEPS, real_detector=False)
    dyn_actions, dyn_agent = run_episode(
        DYNAMIC_STEPS, real_detector=True, extra=DYNAMIC_OVERRIDES
    )
    path = Path(__file__)
    text = path.read_text(encoding="utf-8")
    # Rewrite by NAME, never by the value's own pattern: a regex containing the
    # literal it is replacing rewrites this block as well as the constant.
    for name, value in (
        ("EXPLORATION_ACTIONS_SHA256", json.dumps(action_hash(explore_actions))),
        ("DYNAMIC_ACTIONS_SHA256", json.dumps(action_hash(dyn_actions))),
        ("DYNAMIC_FINGERPRINT: dict", json.dumps(fingerprint(dyn_agent))),
    ):
        text, n = re.subn(
            r"^" + re.escape(name) + r" = .*$", name + " = " + value,
            text, count=1, flags=re.M,
        )
        assert n == 1, f"could not find {name} to rewrite"
    path.write_text(text, encoding="utf-8")
    print("goldens written to", path)
