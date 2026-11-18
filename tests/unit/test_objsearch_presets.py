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
    base = _compose("+experiment=ascentnav")              # the S71 arm, every addition off
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


def test_the_default_is_the_s71_arm_with_m1_m2b_the_heading_memory_and_no_llm():
    """S89: the default is `ascentnav` plus the `objsearch_storey` switches,
    `climb_heading_memory` (`agent: s87`) and no LLM ranker, RAM++ or
    Places365 (`exploration: s89`, `scene_graph: default`);
    `objsearch_no_llm` pins it."""
    default = _compose()
    no_llm = _compose("+experiment=objsearch_no_llm")
    s71 = _compose("+experiment=ascentnav")
    for c in (default, no_llm, s71):
        c.pop("experiment", None)
    assert default == no_llm
    on = {k: v for k, v in default["agent"]["search"].items() if s71["agent"]["search"][k] != v}
    assert on == {"storey_belief": True, "verify_climb": True, "climb_fail_cap": 2,
                  "stair_stop_guard": True, "climb_heading_memory": True}
    assert (default["exploration"]["ranker"], default["exploration"]["ram_tags"],
            default["scene_graph"]["room_classifier"]) == ("none", False, "none")
    s71["agent"]["search"] = default["agent"]["search"]
    s71["exploration"]["ranker"], s71["exploration"]["ram_tags"] = "none", False
    s71["scene_graph"]["room_classifier"] = "none"
    assert s71 == default


def test_climb_still_composes_to_the_s87_default():
    """`objsearch_climb` pins `agent: s87` with the LLM ranker, as S87 ran."""
    climb = _compose("+experiment=objsearch_climb")
    s87 = _compose("agent=s87", "exploration=s71", "scene_graph=place365")
    for c in (climb, s87):
        c.pop("experiment", None)
    assert climb == s87


def test_the_robot_presets_keep_the_room_layer_and_the_ranker():
    """S89 changed the ascentnav default; OSG's nav_agent on the Stretch reads
    Places365 rooms and the ranker, and keeps them."""
    for name in ("stretch3_map", "stretch3_search"):
        c = _compose(f"+experiment={name}")
        assert c["agent"]["policy"] == "nav_agent"
        assert c["scene_graph"]["room_classifier"] == "place365"
        assert c["exploration"]["ranker"] == "ascent" and c["exploration"]["ram_tags"]


def test_storey_still_composes_to_the_s86_default():
    """`objsearch_storey` pins `agent: s86`, so S85/S86 re-run as measured."""
    storey = _compose("+experiment=objsearch_storey")
    s86 = _compose("agent=s86", "exploration=s71", "scene_graph=place365")
    for c in (storey, s86):
        c.pop("experiment", None)
    assert storey == s86


def test_climb_is_storey_plus_the_heading_memory():
    """S87: `objsearch_climb` is the S86 arm plus `climb_heading_memory` only."""
    s = _compose("+experiment=objsearch_storey")
    c = _compose("+experiment=objsearch_climb")
    assert not s["agent"]["search"]["climb_heading_memory"] and c["agent"]["search"]["climb_heading_memory"]
    assert c["agent"]["search"]["climb_mount_m"] == 0.0 and not c["agent"]["search"]["climb_carrot_free"]
    c["agent"]["search"]["climb_heading_memory"] = False
    s.pop("experiment", None)
    c.pop("experiment", None)
    assert c == s


def test_no_llm_is_s87_without_the_ranker_and_its_prompt_models():
    """S89: `objsearch_no_llm` is `objsearch_climb` with the LLM ranker,
    RAM++ and Places365 off, and nothing else."""
    d = _compose("+experiment=objsearch_climb")
    n = _compose("+experiment=objsearch_no_llm")
    assert (d["exploration"]["ranker"], d["exploration"]["ram_tags"],
            d["scene_graph"]["room_classifier"]) == ("ascent", True, "place365")
    assert (n["exploration"]["ranker"], n["exploration"]["ram_tags"],
            n["scene_graph"]["room_classifier"]) == ("none", False, "none")
    n["exploration"]["ranker"], n["exploration"]["ram_tags"] = "ascent", True
    n["scene_graph"]["room_classifier"] = "place365"
    for c in (d, n):
        c.pop("experiment", None)
    assert n == d
