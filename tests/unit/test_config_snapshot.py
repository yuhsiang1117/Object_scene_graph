"""Every calibrated constant, pinned.

Almost every default in `OSGConfig` was chosen by an experiment and carries the
measurement in a comment beside it: `search_proximity_len_m` is 1.0 because a
114-relocation offline ranking said so, `min_presence` is 0.45 because that is
where one detector-strength absence reading lands and one VLM reading does not,
`stair_min_rise_m` is 1.0 because every false positive on the single-floor gate
rose 0.33-0.75 m.

A refactor that moves those files must not move those numbers. This flattens the
whole config to `{dotted.key: value}` and compares it against a committed
snapshot, so a constant can only change when someone changes the snapshot too --
which makes it show up in the diff as what it is, rather than as a line lost in a
file move.

To change a default on purpose:  python tests/unit/test_config_snapshot.py
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from osg.core.config import OSGConfig

SNAPSHOT = Path(__file__).parent / "golden" / "config_snapshot.json"


def flatten(obj, prefix: str = "") -> dict:
    """`{dotted.key: value}` over a nested dataclass tree.

    Tuples become lists so the JSON round-trip is exact -- `container_top_h_m`
    is a tuple in the dataclass and a list once it has been through a file.
    """
    out = {}
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        key = f"{prefix}{field.name}"
        if dataclasses.is_dataclass(value):
            out.update(flatten(value, key + "."))
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def test_no_calibrated_default_has_moved():
    current = flatten(OSGConfig())
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    added = sorted(set(current) - set(expected))
    removed = sorted(set(expected) - set(current))
    changed = {
        k: (expected[k], current[k])
        for k in sorted(set(current) & set(expected))
        if current[k] != expected[k]
    }
    assert not changed, f"defaults changed (was, is): {changed}"
    assert not removed, f"config fields removed: {removed}"
    assert not added, f"config fields added without updating the snapshot: {added}"


def test_every_field_is_reachable_by_its_dotted_name():
    """The snapshot is only a lock if its keys are the names Hydra overrides
    use -- `exploration.search_posterior=true` on the command line has to reach
    the same field this file pins."""
    cfg = OSGConfig()
    for dotted in flatten(cfg):
        node = cfg
        for part in dotted.split("."):
            node = getattr(node, part)


if __name__ == "__main__":  # regenerate deliberately, never automatically
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(
        json.dumps(flatten(OSGConfig()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {SNAPSHOT}")


# ------------------------------------------------- the best-known config (M)
#
# The five flags that separate condition M from K each default to OFF, so the
# ladder stays reproducible from its own overrides and the winning combination
# is asserted in exactly one place. That makes the preset load-bearing: if it
# drifts, the best result on the benchmark stops being reproducible and nothing
# else would notice.

def test_the_best_known_configuration_still_composes():
    from hydra import compose, initialize_config_dir

    from osg.core.config import register_configs

    register_configs()
    root = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(root), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["+experiment=ycb_dynamic_best"])

    # condition N: the six flags, and the K campaign line they sit on
    assert cfg.agent.reachable_via_viewpoint is True
    assert cfg.verification.unreachable_is_absorbing is False
    assert cfg.exploration.search_glance_floor == 0.2
    assert cfg.scene_graph.target_bypasses_gates is True
    assert cfg.verification.target_bypasses_bbox_gate is True
    assert cfg.verification.rank_candidates_by_presence is True

    assert cfg.eval.attempts == 3
    assert cfg.detector.imgsz == 1280
    assert cfg.eval.rgb_width == 1280
    assert cfg.scene_graph.presence.enabled is True
    assert cfg.exploration.search_posterior is True
    assert cfg.verification.absence_only is True
    assert cfg.scene_graph.min_det_bbox_px == 1200
    assert cfg.verification.min_bbox_px == 800

    # the prior map is deliberately not baked in: which snapshot the agent
    # navigates from IS the experiment.
    assert not cfg.ycb.map_in


def test_the_cross_anchor_preset_is_one_flag_from_the_best_one():
    """The two presets are the trade, and it must stay legible as one flag:
    +6 cross_anchor episodes for -5 in_anchor. If they ever drift apart on
    anything else, the comparison stops meaning what the writeup says."""
    from hydra import compose, initialize_config_dir

    from osg.core.config import register_configs

    register_configs()
    root = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(root), version_base="1.3"):
        best = compose(config_name="config", overrides=["+experiment=ycb_dynamic_best"])
        cross = compose(config_name="config", overrides=["+experiment=ycb_dynamic_cross"])

    assert best.exploration.search_drop_proximity_after_absence is False
    assert cross.exploration.search_drop_proximity_after_absence is True

    from omegaconf import OmegaConf

    def flat(cfg, prefix=""):
        out = {}
        for k, v in OmegaConf.to_container(cfg, resolve=False).items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(flat(OmegaConf.create(v), key + "."))
            else:
                out[key] = v
        return out

    a, b = flat(best), flat(cross)
    differ = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert differ == {"exploration.search_drop_proximity_after_absence"}, (
        f"the presets differ on more than the trade: {sorted(differ)}"
    )


def test_the_saturation_preset_is_one_flag_from_the_best_one():
    """Condition V is N plus room saturation and nothing else. The whole claim
    is that ONE change moved the container-dense scenes, so the presets drifting
    apart on anything else would destroy the comparison."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from osg.core.config import register_configs

    register_configs()
    root = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(root), version_base="1.3"):
        best = compose(config_name="config", overrides=["+experiment=ycb_dynamic_best"])
        sat = compose(config_name="config", overrides=["+experiment=ycb_dynamic_sat"])

    assert best.exploration.search_room_saturation == 0.0
    assert sat.exploration.search_room_saturation > 0.0
    # The floor DOES move, and deliberately. Cancelling the bonus (floor 1.0,
    # condition V3) cost nothing and fixed nothing: a room at bonus 1.0 still
    # produced surface utilities of 0.098 against the best frontier's 0.023, so
    # exploration never got a turn. Only a floor below 1.0 makes an exhausted
    # room lose to a frontier. The allowance is what keeps that off the 91% of
    # successes that never reach it.
    assert best.exploration.search_room_saturation_floor == 1.0
    assert sat.exploration.search_room_saturation_floor == 1.0
    assert sat.exploration.search_room_saturation_free > 0

    def flat(cfg, prefix=""):
        out = {}
        for k, v in OmegaConf.to_container(cfg, resolve=False).items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(flat(OmegaConf.create(v), key + "."))
            else:
                out[key] = v
        return out

    a, b = flat(best), flat(sat)
    differ = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert differ == {
        "exploration.search_room_saturation",
        "exploration.search_room_saturation_free",
    }, f"V is meant to be room saturation and nothing else, but differs on: {sorted(differ)}"


def test_the_grounded_preset_is_one_flag_from_the_saturation_one():
    """W = V3 + grounded affinity, nothing else. The claim under test is that
    grounding alone moves 00848, so any other drift would confound it."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from osg.core.config import register_configs

    register_configs()
    root = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(root), version_base="1.3"):
        sat = compose(config_name="config", overrides=["+experiment=ycb_dynamic_sat"])
        gnd = compose(config_name="config", overrides=["+experiment=ycb_dynamic_grounded"])

    assert sat.exploration.affinity_grounded is False
    assert gnd.exploration.affinity_grounded is True

    def flat(cfg, prefix=""):
        out = {}
        for k, v in OmegaConf.to_container(cfg, resolve=False).items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(flat(OmegaConf.create(v), key + "."))
            else:
                out[key] = v
        return out

    a, b = flat(sat), flat(gnd)
    differ = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert differ == {"exploration.affinity_grounded"}, (
        f"W is meant to be grounding and nothing else, but differs on: {sorted(differ)}"
    )


def test_the_five_flags_still_default_off():
    """If one of these ever ships ON, the A-M ladder stops being reproducible
    from the overrides recorded against it."""
    cfg = OSGConfig()
    assert cfg.agent.reachable_via_viewpoint is False
    assert cfg.verification.unreachable_is_absorbing is True
    assert cfg.exploration.search_glance_floor == 0.0
    assert cfg.scene_graph.target_bypasses_gates is False
    assert cfg.verification.target_bypasses_bbox_gate is False
    assert cfg.verification.rank_candidates_by_presence is False
