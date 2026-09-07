"""Episode identity across runs.

habitat's ObjectNav loader assigns episode_id per CONTENT FILE
(object_nav_dataset.py: `episode.episode_id = str(i)` inside the per-scene
loop), so EVERY scene contains an episode "0". Anything that subsets episodes
(the dev50 / dev50_mf A/B splits) or joins two runs together
(scripts/compare_runs.py) must key on the scene as well, or a 50-episode split
silently selects one episode per scene from all 20 scenes and the paired
comparison lines up unrelated episodes.
"""
import types

from osg.eval.runner import _episode_uid, _goal_floor_gap_m


def _ep(scene: str, eid: str):
    return types.SimpleNamespace(scene_id=scene, episode_id=eid)


def test_uid_is_scene_qualified():
    ep = _ep("data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb", "12")
    assert _episode_uid(ep) == "TEEsavR23oF.basis.glb:12"


def test_same_episode_id_in_different_scenes_is_distinct():
    a = _ep("data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb", "0")
    b = _ep("data/scene_datasets/hm3d/val/00801-HaxA7YrQdEC/HaxA7YrQdEC.basis.glb", "0")
    assert _episode_uid(a) != _episode_uid(b)


def _goal(view_point_ys):
    return types.SimpleNamespace(
        view_points=[
            types.SimpleNamespace(agent_state=types.SimpleNamespace(position=[0.0, y, 0.0]))
            for y in view_point_ys
        ]
    )


def _episode(start_y, goal_view_point_ys):
    return types.SimpleNamespace(
        start_position=[0.0, start_y, 0.0], goals=[_goal(goal_view_point_ys)]
    )


def test_gap_is_zero_when_goal_is_on_the_starting_floor():
    assert _goal_floor_gap_m(_episode(0.1, [0.1, 0.15])) < 0.1


def test_gap_measures_a_storey():
    assert _goal_floor_gap_m(_episode(0.0, [2.8])) == 2.8


def test_gap_takes_the_nearest_view_point():
    """A goal category with instances on several floors is reachable via the
    closest one, so the episode is only cross-floor if EVERY instance is."""
    assert _goal_floor_gap_m(_episode(0.0, [2.8, 0.05])) == 0.05


def test_gap_is_absolute_so_downstairs_counts():
    assert _goal_floor_gap_m(_episode(2.8, [0.0])) == 2.8


def test_gap_is_none_without_view_points():
    assert _goal_floor_gap_m(types.SimpleNamespace(start_position=[0, 0, 0], goals=[])) is None


def test_gap_matches_make_dev_split():
    """The split selects episodes with this measure and compare_runs filters
    A/B results with it; the two implementations must agree."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "make_dev_split.py"
    spec = importlib.util.spec_from_file_location("_mds_gap", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ep = _episode(0.0, [2.8, 5.6])
    assert mod.goal_floor_gap(ep) == _goal_floor_gap_m(ep)


def test_uid_matches_make_dev_split():
    """make_dev_split.py writes the ids that run_eval filters on, so the two
    implementations must agree character for character."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "make_dev_split.py"
    spec = importlib.util.spec_from_file_location("_mds", path)
    mod = importlib.util.module_from_spec(spec)
    # The module registers hydra configs at import; that is side-effect free.
    spec.loader.exec_module(mod)

    ep = _ep("data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb", "7")
    assert mod.episode_uid(ep) == _episode_uid(ep)


def test_algorithm_fingerprint_covers_every_flag_an_ab_can_switch():
    """The fingerprint exists so a paired A/B can prove exactly one variable
    moved. A flag added to the config but not to the block is worse than no
    fingerprint at all: compare_runs then reports "algorithm config: IDENTICAL"
    and tells the reader that any difference is noise -- which is what happened
    to the first floor_llm A/B, where floor_llm itself was the missing key.
    """
    import dataclasses
    import inspect

    from osg.core import config
    from osg.eval import runner

    src = inspect.getsource(runner.run_eval)
    block = src.split('"algorithm": {', 1)[1].split("},", 1)[0]

    # Every dataclass an experiment can override, not a hand-kept list. The
    # hand-kept version missed VerificationConfig entirely, so a
    # verification.min_score A/B also reported "IDENTICAL" -- the same failure
    # this test was added for, one config class over.
    behavioural = set()
    for cls, prefix in (
        (config.ExplorationConfig, ""),
        (config.VerificationConfig, "verify_"),
        # AgentConfig was missing, and it owns use_habitat_navmesh -- the S8
        # variable, which sat in the "config" block that compare_runs does not
        # diff. Same failure as floor_llm and verification.min_score before it:
        # the A/B would have printed "algorithm config: IDENTICAL".
        (config.AgentConfig, ""),
    ):
        behavioural |= {prefix + f.name for f in dataclasses.fields(cls)}
    behavioural |= {
        "detector_weights", "detector_imgsz",
        "multi_floor", "room_classifier", "room_erode_iters",
        "terminal_rule", "terminal_stop_m", "terminal_percentile", "fp_retraction",
    }
    # Tuning constants that no A/B in this repo varies; listing them would be
    # noise in every diff.
    exempt = {
        "frontier_text_scorer", "subgraph_radius_m", "images_per_frontier", "top_n_frontiers",
        "frontier_dedup_m", "min_path_cost_m", "unscored_prior",
        "info_gain_radius_m", "max_frontiers_per_call", "knowledge_radius_m",
        "knowledge_prior_path", "value_clip_name", "value_clip_root",
        "value_stride", "value_radius_m", "value_prompt", "value_max_depth_m",
        "stair_explored_boost", "floor_exp_steps", "stair_min_hits",
        "stair_retire_cells", "stair_probe_every", "score_cache_radius_m",
        "score_cache_ttl", "ssim_thresh",
        # VerificationConfig knobs no A/B in this repo varies.
        "verify_enabled", "verify_vlm_model", "verify_min_evidence",
        "verify_max_verify_calls", "verify_debug_dir", "verify_reverify_every",
        "verify_ring_radii_m", "verify_center_tol_deg", "verify_center_max_turns",
        # AgentConfig: protocol constants (recorded in the "config" block
        # instead) and geometry/tuning no A/B in this repo varies.
        "max_steps", "forward_m", "turn_deg", "success_distance", "initial_scan",
        "camera_height", "agent_radius", "approach_stop_depth_m",
        "approach_stop_bbox_px", "approach_depth_stop", "approach_max_steps",
        "approach_goal_tolerance_m", "approach_arrival_tol_m",
        "approach_standoff_m", "navmesh_goal_radius", "navmesh_approach_steps",
        "stair_reach_m", "stair_overshoot_m", "climb_max_steps",
        "stair_climb_state", "floor_gap_min_m", "stair_exit_m",
        "check_candidates_all_states", "terminal_engage_m",
        "terminal_progress_eps", "terminal_stall_steps",
    }
    missing = sorted(f for f in behavioural - exempt if f'"{f}"' not in block)
    assert not missing, f"not recorded in summary.json['algorithm']: {missing}"


def test_composed_config_matches_the_tuned_defaults():
    """A dataclass default is not what runs.

    Every evaluation loads a config group from configs/, and those files
    override the dataclass. `VerificationConfig.min_score` was raised to 0.70 on
    the strength of a 100-episode measurement, but configs/verification/nim.yaml
    still said 0.35 -- so the change was inert and the next run silently used
    the old gate while the fingerprint dutifully recorded it.

    This asserts the value that actually executes, for the knobs whose numbers
    are quoted in docs/AB_RESULTS.
    """
    import hydra

    from osg.core.config import register_configs

    register_configs()
    with hydra.initialize(config_path="../../configs", version_base="1.3"):
        cfg = hydra.compose(config_name="config")

    assert cfg.verification.min_score == 0.70, (
        "the composed gate disagrees with the measured one; "
        "check configs/verification/*.yaml, not just VerificationConfig"
    )
    assert cfg.scene_graph.room_erode_iters == 6
    assert cfg.agent.terminal_percentile == 5.0


def test_frontier_text_scorer_says_what_it_controls():
    """It gates the legacy LLM text scorer, and nothing else.

    The old name was `scorer` with values `nearest|geometric|none|vlm|random`.
    "nearest" read as "no semantics", but the CLIP value map is consulted
    BEFORE the flat prior it falls back to (selector.py:106-108), so every
    "nearest" arm in this log was still running a semantic frontier signal --
    two of them, counting the ASCENT ranker. "vlm" and "random" had no dispatch
    branch and silently built the text scorer.
    """
    from osg.core.config import ExplorationConfig

    cfg = ExplorationConfig()
    assert not hasattr(cfg, "scorer"), "the misleading name is back"
    assert cfg.frontier_text_scorer == "disabled"
    # The switches it does NOT control, so a future reader does not repeat the
    # mistake of reading one field as "semantics on/off".
    for other in ("value_map", "ranker"):
        assert hasattr(cfg, other), f"{other} is a separate switch and must stay one"


def test_an_unrecognised_frontier_text_scorer_fails_loudly():
    """A stale `=nearest` override must not silently do something."""
    import types

    import pytest

    from osg.eval.runner import build_scorer

    cfg = types.SimpleNamespace(
        exploration=types.SimpleNamespace(frontier_text_scorer="nearest")
    )
    with pytest.raises(ValueError, match="not recognised"):
        build_scorer(cfg)


def test_no_config_value_is_a_yaml_boolean_in_disguise():
    """`frontier_text_scorer: off` silently became the string "False".

    YAML 1.1 parses off/on/yes/no as booleans. The field is typed str, so
    OmegaConf coerced the bool back to "False", which matched neither branch.
    Only the explicit raise caught it -- the previous implementation would have
    treated "False" as "not disabled" and quietly built the LLM scorer for a
    whole run.

    Checked against the dataclass's real field types rather than a hand-kept
    allowlist, so a new bool field does not have to be remembered here.
    """
    import dataclasses
    from pathlib import Path

    import yaml

    from osg.core import config

    declared = {f.name: f.type for f in dataclasses.fields(config.ExplorationConfig)}
    for f in sorted(Path("configs/exploration").glob("*.yaml")):
        loaded = yaml.safe_load(f.read_text()) or {}
        for key, value in loaded.items():
            if not isinstance(value, bool):
                continue
            declared_type = declared.get(key)
            assert declared_type in (bool, "bool"), (
                f"{f.name}: `{key}: {value}` parsed as a YAML bool but the "
                f"dataclass types it as {declared_type!r}; it will reach the "
                f"code as the string '{value}'. Quote it, or use a word YAML "
                f"does not treat as a boolean (off/on/yes/no all do)."
            )
