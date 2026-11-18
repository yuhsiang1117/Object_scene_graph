"""Behaviour locks: a real episode must take the identical actions.

Almost every constant in this pipeline was chosen by an experiment, and almost
every module is about to move to a different file. Unit tests prove the pieces
still work; they cannot prove the assembled agent still does the SAME thing --
and "the same thing" is the only claim a refactor of calibrated code is allowed
to make.

So: run a real authored YCB episode and hash the action sequence. Two fixtures
per scene, both verified deterministic by running each twice:

  exploration   StubDetector, 120 steps, ~5 s, no GPU. Covers the costmap,
                keyframing, frontier extraction and selection, the blacklist and
                give-up nets, and navmesh following.
  dynamic       real YOLOE + presence filter + search posterior, 300 steps,
                ~55 s, needs the GPU. Covers everything above plus the belief
                updates, surface inspection, candidate admission, and the
                terminal approach.

A lock is only worth what its fixture scene is worth, and the authored corpus is
regenerated from time to time: five of the fifteen scenes these locks were first
written against are no longer authored, 00829 among them. So each case names its
own scene and SKIPS when that scene is absent. A lock that FAILS for want of data
teaches people to ignore it, which costs more than the lock was worth.

  LEGACY   00829-QaLdnwvtxbs. The tape these locks were recorded from. Its
           authored layout is not on the current collector mount, so it skips
           here; the hashes stay as the historical record and become live again
           if the scene is ever re-authored.
  CURRENT  00848-ziup5kvtCCR. Recorded 2026-09-17 on the present corpus, and the
           case a refactor is actually measured against. It exercises more of
           the machinery than the legacy one did -- 1797 presence expectations
           against 1260, 869 negative updates against 477, eight surface
           inspections against three.

Regenerate the goldens ONLY from behaviour you have decided is correct:

    python tests/integration/test_trajectory_lock.py
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest

pytestmark = pytest.mark.sim

pytest.importorskip("habitat")

from osg.core.paths import collector_data_root, ycb_authoring_root

DATA_ROOT = collector_data_root()
LAYOUT_ROOT = ycb_authoring_root()

EXPLORATION_STEPS = 120
DYNAMIC_STEPS = 300


@dataclass(frozen=True)
class LockCase:
    """One scene's recorded tape. The manifest is cached under
    outputs/ycb_manifests/, so the episode is byte-identical run to run."""

    name: str
    scene: str
    target: str
    exploration_sha256: str
    dynamic_sha256: str
    # The mechanisms, not just the motion: a refactor that moved a belief update
    # would change these long before it changed a foot placement.
    dynamic_fingerprint: dict = field(default_factory=dict)


LEGACY = LockCase(
    name="legacy",
    scene="00829-QaLdnwvtxbs",
    target="bowl",
    exploration_sha256="c14d33461af8a7e4c3e7002b51a3522ef08e14e8370cac56a81a74ae3d1ae31d",
    dynamic_sha256="b58c82b7f9dd4ae9abebd42a2988e660e9a7d49eacf50f8e26b37e0688c8075c",
    dynamic_fingerprint={"state_log": [[13, "goto_frontier"], [212, "approach"], [236, "done"]], "presence_expected": 1260, "presence_positive": 1024, "presence_negative": 477, "presence_disbelieved": 12, "search_surface": 3, "frontier_reached": 6, "select_ok": 6, "select_none": 0, "plan_ok": 6, "plan_fail": 0, "n_search_events": 5, "n_presence_events": 17},
)

CURRENT = LockCase(
    name="current",
    scene="00848-ziup5kvtCCR",
    target="bowl",
    exploration_sha256="97a03b0e29378a46eb55e4cf5557c44302c59071df438145f5b46794388df10a",
    dynamic_sha256="21e7db0dc9916f51f795399613f438ceef950909c010f9bca10037e514f86f41",
    dynamic_fingerprint={"state_log": [[13, "explore"], [23, "goto_frontier"], [25, "explore"], [28, "goto_frontier"], [271, "explore"], [278, "goto_frontier"], [288, "explore"], [291, "goto_frontier"]], "presence_expected": 1797, "presence_positive": 1230, "presence_negative": 869, "presence_disbelieved": 23, "search_surface": 8, "frontier_reached": 8, "select_ok": 9, "select_none": 0, "plan_ok": 9, "plan_fail": 0, "n_search_events": 16, "n_presence_events": 36},
)

CASES = (LEGACY, CURRENT)
CASE_IDS = tuple(case.name for case in CASES)


def _mounted() -> bool:
    return DATA_ROOT.is_dir() and LAYOUT_ROOT.is_dir()


def _authored(scene: str) -> bool:
    """The corpus is re-authored from time to time; an absent scene is missing
    data, not a regression."""
    return (LAYOUT_ROOT / scene / "static_scene_config.json").is_file()


def _require(case: LockCase) -> None:
    if not _mounted():
        pytest.skip("collector data is not mounted")
    if not _authored(case.scene):
        pytest.skip(f"{case.scene} is not in the authored corpus under {LAYOUT_ROOT}")


def _config(scene: str, target: str, extra=()):
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
                f"ycb.scenes=[{scene}]",
                f"ycb.targets=[{target}]",
                "eval.save_viz=false",
                "eval.debug_frames=false",
                *extra,
            ],
        )


def run_episode(case: LockCase, max_steps: int, real_detector: bool, extra=()):
    """(actions, agent) for one episode, driven exactly as the runner drives it."""
    from osg.agent.nav_agent import NavAgent
    from osg.exploration.async_scorer import AsyncScorer
    from osg.exploration.scorer import NullScorer
    from osg.perception.detector import StubDetector
    from osg.sim.ycb_env import YCBAuthoredNavEnv

    cfg = _config(case.scene, case.target, extra)
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


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
@pytest.mark.timeout(300)
def test_exploration_trajectory_is_unchanged(case: LockCase):
    _require(case)
    actions, _ = run_episode(case, EXPLORATION_STEPS, real_detector=False)
    assert len(actions) == EXPLORATION_STEPS
    assert action_hash(actions) == case.exploration_sha256, (
        f"{case.scene}: exploration behaviour changed: "
        + "".join(a[0] for a in actions)
    )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
@pytest.mark.gpu
@pytest.mark.timeout(900)
def test_dynamic_pipeline_trajectory_is_unchanged(case: LockCase):
    _require(case)
    actions, agent = run_episode(
        case, DYNAMIC_STEPS, real_detector=True, extra=DYNAMIC_OVERRIDES
    )
    # Report the mechanism first: "presence_negative 869 -> 863" says which
    # module moved, where a bare hash mismatch says only that something did.
    assert fingerprint(agent) == case.dynamic_fingerprint
    assert action_hash(actions) == case.dynamic_sha256


if __name__ == "__main__":
    import json
    import re

    path = Path(__file__)
    text = path.read_text(encoding="utf-8")
    for case in CASES:
        if not (_mounted() and _authored(case.scene)):
            print(f"skipping {case.name} ({case.scene}): not in the authored corpus")
            continue
        explore_actions, _ = run_episode(case, EXPLORATION_STEPS, real_detector=False)
        dyn_actions, dyn_agent = run_episode(
            case, DYNAMIC_STEPS, real_detector=True, extra=DYNAMIC_OVERRIDES
        )
        # Rewrite by FIELD NAME within this case's block, never by the value's
        # own pattern: a regex containing the literal it is replacing rewrites
        # this block as well as the constant.
        block = re.search(
            r"^" + case.name.upper() + r" = LockCase\(\n(?:.*\n)*?\)\n", text, flags=re.M
        )
        assert block is not None, f"could not find the {case.name.upper()} block"
        updated = block.group(0)
        for field_name, value in (
            ("exploration_sha256", json.dumps(action_hash(explore_actions))),
            ("dynamic_sha256", json.dumps(action_hash(dyn_actions))),
            # repr(), not json.dumps(): a Python literal, and any bool in the
            # fingerprint would come back as `false`.
            ("dynamic_fingerprint", repr(fingerprint(dyn_agent))),
        ):
            updated, n = re.subn(
                r"^(\s*)" + field_name + r"=.*$",
                lambda m: f"{m.group(1)}{field_name}={value},",
                updated, count=1, flags=re.M,
            )
            assert n == 1, f"could not find {case.name}.{field_name} to rewrite"
        text = text.replace(block.group(0), updated)
    path.write_text(text, encoding="utf-8")
    print("goldens written to", path)
