"""Floor classification for episodes and trajectories (osg.eval.floors)."""
import types

import pytest

from osg.eval.floors import (
    FLOOR_CHANGE_M,
    SAME_FLOOR_M,
    classify_episode,
    count_floor_changes,
    episode_floor_fields,
    goal_view_heights,
    runtime_floor_fields,
)
from osg.eval.metrics import per_floor_class, per_relocation


# --------------------------------------------------------------- fake episode


class _State:
    def __init__(self, y):
        self.position = [0.0, y, 0.0]


class _ViewPoint:
    def __init__(self, y):
        self.agent_state = _State(y)


class _Goal:
    def __init__(self, ys, center_y=None):
        self.view_points = [_ViewPoint(y) for y in ys]
        self.position = [0.0, center_y if center_y is not None else 0.0, 0.0]


class _Episode:
    def __init__(self, start_y, goals):
        self.start_position = [0.0, start_y, 0.0]
        self.goals = goals


# ------------------------------------------------------------------- classify


def test_same_floor_when_a_goal_shares_the_start_height():
    assert classify_episode(0.0, [0.05, 3.1]) == "same_floor"


def test_cross_floor_when_every_goal_is_a_storey_away():
    assert classify_episode(0.0, [3.1, 3.2]) == "cross_floor"


def test_unknown_without_goal_heights():
    assert classify_episode(0.0, []) == "unknown"


def test_same_floor_boundary_is_inclusive():
    assert classify_episode(0.0, [SAME_FLOOR_M]) == "same_floor"
    assert classify_episode(0.0, [SAME_FLOOR_M + 0.01]) == "cross_floor"


def test_goal_view_heights_falls_back_to_the_goal_centre():
    assert goal_view_heights(_Episode(0.0, [_Goal([], center_y=2.5)])) == [2.5]


# -------------------------------------------------------------- floor changes


def test_flat_trajectory_has_no_floor_change():
    assert count_floor_changes([0.0] * 50) == 0


def test_climbing_and_settling_counts_once():
    ys = [0.0] * 20 + list(_ramp(0.0, 3.0, 12)) + [3.0] * 20
    assert count_floor_changes(ys) == 1


def test_up_and_back_down_counts_zero():
    """Half a staircase and back is not a floor change -- a bare threshold
    crossing would wrongly count two."""
    ys = [0.0] * 10 + list(_ramp(0.0, 2.0, 8)) + list(_ramp(2.0, 0.0, 8)) + [0.0] * 10
    assert count_floor_changes(ys) == 0


def test_two_storeys_count_twice():
    ys = ([0.0] * 15 + list(_ramp(0.0, 3.0, 12)) + [3.0] * 15
          + list(_ramp(3.0, 6.0, 12)) + [6.0] * 15)
    assert count_floor_changes(ys) == 2


def test_change_threshold_is_above_the_same_floor_threshold():
    """Standing mid-staircase must not read as a new floor."""
    assert FLOOR_CHANGE_M > SAME_FLOOR_M
    assert count_floor_changes([0.0] * 10 + [1.0] * 40) == 0


def test_empty_trajectory():
    assert count_floor_changes([]) == 0


def _ramp(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


# ----------------------------------------------------------- episode fields


def test_episode_floor_fields_cross_floor():
    ep = _Episode(0.0, [_Goal([3.2, 3.3])])
    fields = episode_floor_fields(ep, [0.0] * 10 + [3.2] * 20)
    assert fields["floor_class"] == "cross_floor"
    assert fields["start_y"] == 0.0
    assert fields["final_y"] == 3.2
    assert fields["floor_changes"] == 1
    assert fields["traj_y_range"] == pytest.approx(3.2)


def test_episode_floor_fields_without_trajectory():
    ep = _Episode(1.0, [_Goal([1.0])])
    fields = episode_floor_fields(ep, [])
    assert fields["floor_class"] == "same_floor"
    assert fields["final_y"] is None
    assert fields["floor_changes"] == 0


def test_runtime_fields_use_stable_keys_and_report_downward_relocation():
    ep = _Episode(2.8, [_Goal([0.0])])
    stack = types.SimpleNamespace(stair_edges=[object()])
    floors = types.SimpleNamespace(levels={4: 0.0, 9: 2.8}, stack=stack)
    exploration = types.SimpleNamespace(selected_search_floor=4)
    agent = types.SimpleNamespace(
        floors=floors, exploration=exploration,
        stats={"floor_switch_attempts": 1},
    )
    authored = {"relocation": {
        "origin_position": [0.0, 2.8, 0.0],
        "destination_position": [0.0, 0.0, 0.0],
    }}

    fields = runtime_floor_fields(ep, [2.8, 1.4, 0.0], agent, authored)

    assert fields == {
        "start_floor": 9,
        "goal_floor": 4,
        "prior_floor": 9,
        "relocation_direction": "downward",
        "selected_search_floor": 4,
        "floor_switches": 1,
        "climb_attempts": 1,
        "goal_floor_reached": True,
    }


# ------------------------------------------------------------------- metrics


def test_per_floor_class_splits_sr():
    results = [
        {"floor_class": "same_floor", "success": 1.0, "spl": 0.5},
        {"floor_class": "same_floor", "success": 0.0, "spl": 0.0},
        {"floor_class": "cross_floor", "success": 0.0, "spl": 0.0},
    ]
    out = per_floor_class(results)
    assert out["same_floor"]["success_rate"] == 0.5
    assert out["same_floor"]["num_episodes"] == 2
    assert out["cross_floor"]["success_rate"] == 0.0


def test_per_floor_class_defaults_missing_field_to_unknown():
    assert "unknown" in per_floor_class([{"success": 1.0, "spl": 1.0}])


def test_relocation_metrics_include_cross_floor_and_direction_groups():
    results = [
        {"relocation_direction": "same_floor", "success": 1.0, "spl": 0.5},
        {"relocation_direction": "upward", "success": 1.0, "spl": 0.25},
        {"relocation_direction": "downward", "success": 0.0, "spl": 0.0},
    ]
    out = per_relocation(results)
    assert out["same_floor"]["num_episodes"] == 1
    assert out["cross_floor"]["num_episodes"] == 2
    assert out["cross_floor"]["success_rate"] == 0.5
    assert out["upward"]["success_rate"] == 1.0
    assert out["downward"]["success_rate"] == 0.0
