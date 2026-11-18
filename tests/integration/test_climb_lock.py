"""Behaviour lock for the climb, which no other test reaches.

`test_trajectory_lock.py` hashes two real episodes, and neither of them climbs:
both run `+experiment=ycb_authored_nav`, which composes `legacy_defaults.yaml`
with `multi_floor: false` and `floor.enabled=False`, so `agent/floor_policy.py`,
`NavAgent._do_climb` and the whole carrot cascade -- roughly 700 lines -- execute
zero times under test. Everything about the multi-floor arm was unguarded.

It needs guarding more than the rest, not less. `configs/experiment/
mf5_osg_on_ascent_map.yaml` records two runs that differ by ONE alignment turn
and then diverge from arriving at step 161 to step 513. On a staircase, a change
too small to reason about is not too small to matter.

So: drive the real `_start_climb` + `_do_climb` from a fixed pose at the foot of
a ground-truth flight on 00800-TEEsavR23oF, and pin the action sequence and the
climb counters. Three height variants, because they fail differently and the
difference is the diagnosis:

    depth   the flight has no cells, so `_carrot_action` falls through to
            `_update_carrot` -- the depth-ray carrot transcribed from ASCENT,
            which every other variant shadows.
    exact   the flight fed the navmesh route's true tread heights.
    ramp    the same cells with the linear ramp `_ramp_stair_heights` writes
            when it pastes an ASCENT map. If `exact` climbs and `ramp` does not,
            the heights are the problem rather than the loop.

The `pasted` variant is deliberately absent: its input lives under
`outputs/maps_mf5_ascent`, which is gitignored, so it cannot be a lock.

This imports the fixture and the runner from `scripts/probe_climb_osg.py` rather
than copying them. A copy would not drift with the production code it is
supposed to be measuring; the cost is that editing the probe's setup fails this
test, which is the correct alarm with a slightly confusing name.

Regenerate ONLY from behaviour you have decided is correct:

    python tests/integration/test_climb_lock.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.sim, pytest.mark.gpu]

pytest.importorskip("habitat")

from osg.core.paths import ycb_multi_floor_root

REPO = Path(__file__).resolve().parents[2]
SCENE = "00800-TEEsavR23oF"
EXPERIMENT = "mf5_osg_on_ascent_map"
MAX_STEPS = 300
VARIANTS = ("depth", "exact", "ramp")

# What is pinned: the motion, and the counters that move before the motion does.
PINNED = ("direction", "steps", "dy_max", "y_end", "fell_back_m", "end",
          "carrot_flight", "carrot_top", "forced_forward", "blocked_turn",
          "actions", "actions_sha256")

GOLDEN: dict = {'depth': {'direction': 1, 'steps': 51, 'dy_max': 0.148, 'y_end': 0.163, 'fell_back_m': 0.148, 'end': {'ok': False, 'why': 'stalled'}, 'carrot_flight': 0, 'carrot_top': 0, 'forced_forward': 0, 'blocked_turn': 0, 'actions': {'move_forward': 24, 'turn_left': 10, 'turn_right': 17}, 'actions_sha256': 'b62b69df872703857de8bb2f7662c5b49daba9c63ef72a281c10b3c7b1ad613e'}, 'exact': {'direction': 1, 'steps': 275, 'dy_max': 2.802, 'y_end': 2.965, 'fell_back_m': 0.0, 'end': {'ok': True, 'why': 'storey_of_height'}, 'carrot_flight': 4, 'carrot_top': 0, 'forced_forward': 0, 'blocked_turn': 0, 'actions': {'move_forward': 163, 'turn_left': 54, 'turn_right': 58}, 'actions_sha256': '8720f62f68ba447390042d2febae2e60f556dd5ee7e9930354ada404dcb5ccf5'}, 'ramp': {'direction': 1, 'steps': 300, 'dy_max': 1.25, 'y_end': 0.163, 'fell_back_m': 1.25, 'end': {}, 'carrot_flight': 3, 'carrot_top': 0, 'forced_forward': 0, 'blocked_turn': 0, 'actions': {'move_forward': 169, 'turn_left': 70, 'turn_right': 61}, 'actions_sha256': '9976aea53e2ff082660dcec69e9da2a482c057cf91ed4b350e78097fc55fb374'}}


def _probe():
    spec = importlib.util.spec_from_file_location(
        "probe_climb_osg", REPO / "scripts" / "probe_climb_osg.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config():
    from hydra import compose, initialize_config_dir

    from osg.core.config import register_configs

    register_configs()
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        return compose(config_name="config", overrides=[
            f"+experiment={EXPERIMENT}",
            f"ycb.scenes=[{SCENE}]",
            "eval.save_viz=false",
            "eval.debug_frames=false",
        ])


def run_variants() -> dict:
    """One environment, three climbs. Building the env is the expensive part."""
    probe = _probe()
    cfg = _config()
    fx = probe.build_climb_fixture(cfg, scene=SCENE)
    try:
        out = {}
        for variant in VARIANTS:
            result = probe._run_variant(
                fx.env, cfg, fx.pointnav, fx.foot, fx.top, fx.path3d,
                variant, MAX_STEPS, fx.place)
            out[variant] = {k: result[k] for k in PINNED}
        return out
    finally:
        fx.env.close()


def _available() -> bool:
    return (ycb_multi_floor_root() / SCENE).is_dir()


@pytest.mark.timeout(2400)
def test_climb_behaviour_is_unchanged():
    if not _available():
        pytest.skip(f"{SCENE} is not under {ycb_multi_floor_root()}")
    assert GOLDEN, "no golden recorded; run `python tests/integration/test_climb_lock.py`"
    got = run_variants()
    # Per variant, so a failure names the one that moved rather than the set.
    for variant in VARIANTS:
        assert got[variant] == GOLDEN[variant], f"climb behaviour changed for {variant!r}"


if __name__ == "__main__":
    import re

    recorded = run_variants()
    path = Path(__file__)
    text = path.read_text(encoding="utf-8")
    # repr(), not json.dumps(): this is rewriting a Python literal, and the
    # `end` dicts carry booleans, which JSON spells `false`.
    text, n = re.subn(r"^GOLDEN: dict = .*$",
                      lambda _m: "GOLDEN: dict = " + repr(recorded),
                      text, count=1, flags=re.M)
    assert n == 1, "could not find GOLDEN to rewrite"
    path.write_text(text, encoding="utf-8")
    print("golden written to", path)
