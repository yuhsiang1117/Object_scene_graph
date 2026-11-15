"""The `objsearch_*` presets lay over the default arm and change only what they name."""
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from osg.core.config import register_configs

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _compose(*overrides):
    register_configs()
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        return OmegaConf.to_container(compose(config_name="config", overrides=list(overrides)),
                                      resolve=False)


def test_ellipsoid_a_is_the_adopted_block_and_nothing_else():
    base = _compose()
    a = _compose("+experiment=objsearch_ellipsoid_a")
    adopted = _compose("+experiment=mfv2_osg_vet_live_ellipsoid_v3")
    assert a["scene_graph"]["ellipsoid"] == adopted["scene_graph"]["ellipsoid"]
    assert a["scene_graph"]["ellipsoid"] != base["scene_graph"]["ellipsoid"]
    a["scene_graph"]["ellipsoid"] = base["scene_graph"]["ellipsoid"]
    a.pop("experiment", None)
    base.pop("experiment", None)
    assert a == base


def test_storey_turns_on_m1_and_m2b_and_nothing_else():
    base = _compose()
    s = _compose("+experiment=objsearch_storey")
    on = {"storey_belief": True, "verify_climb": True, "climb_fail_cap": 2, "stair_stop_guard": True}
    changed = {k: v for k, v in s["agent"]["search"].items() if base["agent"]["search"][k] != v}
    assert changed == on
    assert not base["agent"]["search"]["storey_belief"] and base["agent"]["search"]["climb_fail_cap"] == 0
    s["agent"]["search"] = base["agent"]["search"]
    s.pop("experiment", None)
    base.pop("experiment", None)
    assert s == base


def test_relay_is_storey_plus_the_one_switch():
    s = _compose("+experiment=objsearch_storey")
    r = _compose("+experiment=objsearch_relay")
    assert not s["agent"]["search"]["climb_relay"] and r["agent"]["search"]["climb_relay"]
    r["agent"]["search"]["climb_relay"] = False
    s.pop("experiment", None)
    r.pop("experiment", None)
    assert r == s
