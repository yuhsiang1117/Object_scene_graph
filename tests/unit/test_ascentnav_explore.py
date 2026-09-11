"""The explore path of the ascentnav transcription (S71).

`Ascent_Policy._explore` (`ascent_policy.py:699-762`) and the vendored
`Ascent_LLM_Planner` (`llm_planner.py`). These are the mechanisms the trace
diagnosis found carrying the same-floor budget into the ground: a floor's
last frontier retired by the sticky rule, a per-episode disabled set, no
20-selection disable, and no stairwell reinitialisation.
"""
from __future__ import annotations

import numpy as np

from ascentnav.planner import REPEATED_SELECTION_THRESHOLD, STICKY_FRONTIER_STEP_THRESHOLD

from .test_ascentnav_stairs import DEPTH, _agent


def _frame_for(frontier):
    """A distinct frame per frontier, so the SSIM > 0.75 dedup keeps them all."""
    rng = np.random.RandomState(int(abs(frontier[0]) * 10 + abs(frontier[1])))
    return rng.randint(0, 255, (16, 16, 3)).astype(np.uint8)


class _Value:
    """A value map that ranks frontiers by x, highest first."""

    def sort_waypoints(self, frontiers, _r):
        f = np.asarray(frontiers, dtype=float).reshape(-1, 2)
        order = np.argsort(-f[:, 0])
        return f[order], [float(v) for v in f[order, 0]]


def _explorer(frontiers, **over):
    a = _agent(**over)
    a.obstacle_map.frontiers = np.asarray(frontiers, dtype=float)
    a.obstacle_map._floor_num_steps = 60           # past the reinit window
    a._floors[0]["value"] = _Value()
    return a


def test_a_single_frontier_is_taken_without_the_sticky_rule():
    """`llm_planner.py:82-86` returns before the bookkeeping. The port ran the
    sticky rule here and could retire a floor's ONLY frontier."""
    a = _explorer([[3.0, 0.0]])
    for _ in range(STICKY_FRONTIER_STEP_THRESHOLD + 5):
        a._explore(DEPTH, np.zeros(2), 0.0)
    assert (3.0, 0.0) not in a.obstacle_map._disabled_frontiers
    assert a.planner.frontier_stick_step == 0


def test_the_sticky_rule_retires_a_frontier_the_agent_cannot_close_on():
    """`:256-274`: same frontier, no 0.3 m change, 20 steps -> disabled."""
    a = _explorer([[3.0, 0.0], [1.0, 0.0]])
    for _ in range(STICKY_FRONTIER_STEP_THRESHOLD + 2):
        a._explore(DEPTH, np.zeros(2), 0.0)
    assert (3.0, 0.0) in a.obstacle_map._disabled_frontiers


def test_disabled_frontiers_are_per_floor():
    """They live on the obstacle map (`:704`), not on the agent."""
    a = _explorer([[3.0, 0.0], [1.0, 0.0]])
    a.obstacle_map._disabled_frontiers.add((3.0, 0.0))
    a._floors.append(a._new_floor())
    a._floor_idx = 1
    assert (3.0, 0.0) not in a.obstacle_map._disabled_frontiers


def test_the_first_choice_becomes_a_commitment():
    """`:119-125`: `_force_frontier` is latched at the END of the first call,
    and `_try_force_frontier` (:168-175) returns it while it is still extracted."""
    a = _explorer([[3.0, 0.0], [1.0, 0.0]])
    a._explore(DEPTH, np.zeros(2), 0.0)
    assert np.allclose(a.planner._force_frontier, [3.0, 0.0])
    assert a.obstacle_map._finish_first_explore
    a.obstacle_map.frontiers = np.array([[1.0, 0.0], [3.0, 0.0]])
    a._explore(DEPTH, np.zeros(2), 0.0)
    assert np.allclose(a.cur_frontier, [3.0, 0.0])


def test_a_nearby_frontier_short_circuits_once_the_first_explore_is_done():
    """`:100-113`: within `nearby_distance` (3.0 m), the closest wins and
    `_neighbor_search` is set."""
    a = _explorer([[10.0, 0.0], [2.0, 0.0]])
    a.obstacle_map._finish_first_explore = True
    a._explore(DEPTH, np.zeros(2), 0.0)
    assert np.allclose(a.cur_frontier, [2.0, 0.0])
    assert a.obstacle_map._neighbor_search is True


def test_the_sticky_reset_is_suppressed_under_neighbor_search():
    """`:265`: `and not obstacle_map._neighbor_search`."""
    a = _explorer([[10.0, 0.0], [2.0, 0.0]])
    a.obstacle_map._finish_first_explore = True
    a._explore(DEPTH, np.zeros(2), 0.0)                    # picks (2,0), neighbour mode
    for i in range(STICKY_FRONTIER_STEP_THRESHOLD + 2):
        # 0.5 m back and forth: > 0.3 m of change every step, always within
        # the 3 m neighbourhood so the shortcut keeps picking (2, 0)
        a._explore(DEPTH, np.array([0.5 * (i % 2), 0.0]), 0.0)
    assert (2.0, 0.0) in a.obstacle_map._disabled_frontiers, (
        "in neighbour mode the stick counter is not re-armed by movement")


def test_repeated_non_consecutive_selection_retires_a_frontier():
    """`:281-288`: bouncing between two frontiers is bounded at 20."""
    a = _explorer([[3.0, 0.0], [1.0, 0.0]])
    om = a.obstacle_map
    om._finish_first_explore = True
    a.planner._force_frontier = np.zeros(2)
    for i in range(2 * REPEATED_SELECTION_THRESHOLD + 2):
        # alternate which one ranks first so the selection is non-consecutive
        om.frontiers = np.array([[3.0, 0.0], [1.0, 0.0]] if i % 2 else [[1.0, 0.0], [3.0, 0.0]])
        a._floors[0]["value"] = _Value()
        om._neighbor_search = False
        om._finish_first_explore = False   # skip the nearby shortcut
        a._explore(DEPTH, np.array([20.0, 0.0]), 0.0)
        om._finish_first_explore = False
        a.planner._force_frontier = np.zeros(2)
    assert om._disabled_frontiers, "neither frontier was ever retired"


def test_the_llm_is_asked_only_when_no_shortcut_fires():
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return '{"Index": "2", "Reason": "test"}'

    a = _explorer([[3.0, 0.0], [1.0, 0.0], [-2.0, 0.0]])
    a.planner._llm = llm
    a.obstacle_map._each_step_rgb = {}
    a.obstacle_map.extract_frontiers_with_image = lambda f: (0, _frame_for(f))
    a._explore(DEPTH, np.array([20.0, 0.0]), 0.0)
    assert len(calls) == 1 and a.stats["llm_calls"] == 1
    assert np.allclose(a.cur_frontier, [1.0, 0.0]), "Index 2 is the second-ranked frontier"
    assert a.stats["rank_overrides"] == 1


def test_a_failing_llm_keeps_the_value_ranking():
    a = _explorer([[3.0, 0.0], [1.0, 0.0], [-2.0, 0.0]])
    a.planner._llm = lambda p: (_ for _ in ()).throw(RuntimeError("down"))
    a.obstacle_map.extract_frontiers_with_image = lambda f: (0, np.zeros((16, 16, 3), np.uint8))
    a._explore(DEPTH, np.array([20.0, 0.0]), 0.0)
    assert np.allclose(a.cur_frontier, [3.0, 0.0]) and a.stats["rank_errors"] == 1


def test_a_policy_stop_on_a_frontier_forces_forward():
    """`:759-761`."""
    from .test_ascent_agent import _Driver

    a = _explorer([[3.0, 0.0]], driver=_Driver(action=None))
    assert a._explore(DEPTH, np.zeros(2), 0.0) == "move_forward"
    assert a.stats["explore_forced_forward"] == 1


def test_the_multi_floor_sentinel_routes_to_the_stairs():
    """`:745-752`: GO_UP tries the up-staircase and falls through when there
    is none."""
    from ascentnav.planner import GO_UP

    a = _explorer([[3.0, 0.0], [1.0, 0.0]])
    a.planner.get_best_frontier = lambda *args, **kw: (np.array([3.0, 0.0]), GO_UP)
    assert a._explore(DEPTH, np.zeros(2), 0.0) == "move_forward"   # no stairs: drives at the frontier
    assert not a.stairs.climbing
