"""The `episodes.jsonl` key set, pinned.

Nine analysis scripts read this record by name -- `analyze_campaign.py` splits
three funnels out of `gt_kf_*`, `search_log_events` and `target_tracks`;
`analyze_localization.py` needs `target_obj_xy` beside `final_xy`;
`analyze_dynamic_bench.py` needs `authored_layout.relocation`. None of them
would fail loudly on a dropped field: they would report zero and the conclusion
drawn from that zero would be wrong.

So the record's key set is pinned here, taken from a real run
(`outputs/20260823_053711/episodes.jsonl`). The keys are derived the way the
runner actually builds them -- the literal block plus the four `**` helpers,
each really called -- so this survives the record assembly moving to its own
module.
"""
from __future__ import annotations

import ast
import inspect
import types
from pathlib import Path

# From a real 96-episode campaign run. Sorted, so a diff reads alphabetically.
EXPECTED_KEYS = {
    "agent_stats", "approach_bbox_log", "approach_diag", "approach_stop_reason",
    "attempt_log", "attempts_used", "authored_layout", "cand_best_cam_xy",
    "cand_best_score", "cand_n_obs", "control_fps", "detector",
    "distance_to_goal", "episode_id", "final_xy", "final_y", "floor_changes",
    "floor_class", "floor_log", "floor_transitions", "floor_y_drift",
    "frontier_select_log", "giveup_log", "goal_commit_log", "goal_y_span",
    "gt_best_det_score", "gt_best_offaxis", "gt_frames",
    "gt_in_view_close_frames", "gt_in_view_frames", "gt_kf_close_centred",
    "gt_kf_close_centred_detected", "gt_kf_close_peripheral",
    "gt_kf_close_peripheral_detected", "gt_kf_detected", "gt_kf_far_centred",
    "gt_kf_far_centred_detected", "gt_kf_far_peripheral",
    "gt_kf_far_peripheral_detected", "gt_kf_in_view",
    "gt_mean_visible_fraction", "gt_min_range_m", "llm_calls", "llm_errors",
    "llm_last_error", "n_floors_seen", "n_stair_tracks", "portal_log",
    "presence_events", "prior_map", "scene", "search_log_events", "spl",
    "stair_tracks", "start_y", "state_log", "steps", "success", "target",
    "target_obj_xy", "target_tracks", "traj_y_range", "verify_calls",
    "verify_errors", "wall_time_s",
}

# The four blocks spliced in with `**`, and the analyses that would go quiet
# without them.
CONTRIBUTED_BY_HELPERS = {
    "target_obj_xy", "cand_best_cam_xy", "cand_best_score", "cand_n_obs",
    "floor_class", "start_y", "final_y", "traj_y_range", "floor_changes",
    "goal_y_span", "n_stair_tracks", "stair_tracks",
}


def _record_module():
    """Wherever the record is assembled today. `runner.py` now; `record.py`
    after the split -- found by the symbol, not by the filename."""
    from osg.eval import runner

    return runner


def _literal_keys() -> set:
    """String keys of the record's dict literal, read from the source.

    Building the real record needs a live Habitat episode, so the literal half
    is read statically: the record is the one dict literal in the module with
    more than twenty string keys.
    """
    tree = ast.parse(inspect.getsource(_record_module()))
    best: set = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if len(keys) > len(best):
            best = keys
    assert len(best) > 20, "could not find the episode record's dict literal"
    return best


def _helper_keys() -> set:
    """The `**` blocks, actually called. Each takes stub inputs happily: the
    point is the key set, not the values."""
    from osg.eval.floors import episode_floor_fields
    from osg.eval.runner import _GroundTruthVisibility, _stair_track_fields, _target_track_fields
    from osg.objects.object_layer import ObjectLayer

    agent = types.SimpleNamespace(object_layer=ObjectLayer())
    return (
        set(_target_track_fields(agent))
        | set(episode_floor_fields(types.SimpleNamespace(), [0.0]))
        | set(_stair_track_fields(agent))
        | set(_GroundTruthVisibility(None).fields())
    )


def test_the_record_still_has_every_field_the_analyzers_read():
    produced = _literal_keys() | _helper_keys()
    assert not (EXPECTED_KEYS - produced), (
        f"episodes.jsonl fields dropped: {sorted(EXPECTED_KEYS - produced)}"
    )
    assert not (produced - EXPECTED_KEYS), (
        "episodes.jsonl gained fields; add them to EXPECTED_KEYS on purpose: "
        f"{sorted(produced - EXPECTED_KEYS)}"
    )


def test_the_spliced_blocks_are_still_spliced_in():
    """A `**helper()` that stops being splatted drops a dozen fields at once and
    the literal half of the record still looks fine."""
    assert CONTRIBUTED_BY_HELPERS <= _helper_keys()
    assert not (CONTRIBUTED_BY_HELPERS & _literal_keys()), (
        "a helper's field is now also a literal -- one of the two is dead"
    )


def test_every_spliced_value_survives_json_dumps():
    """`json.dumps` is what writes the record, and numpy scalars do not survive
    it. The helpers are where numpy gets in -- ellipsoid centres, camera poses,
    trajectory heights -- so they are the ones worth checking."""
    import json
    import types

    from osg.eval.floors import episode_floor_fields
    from osg.eval.runner import _GroundTruthVisibility, _stair_track_fields, _target_track_fields
    from osg.objects.object_layer import ObjectLayer

    agent = types.SimpleNamespace(object_layer=ObjectLayer())
    for block in (
        _target_track_fields(agent),
        episode_floor_fields(types.SimpleNamespace(), [0.0]),
        _stair_track_fields(agent),
        _GroundTruthVisibility(None).fields(),
    ):
        json.dumps(block)
