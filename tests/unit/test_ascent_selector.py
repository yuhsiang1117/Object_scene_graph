"""ASCENT's frontier selection: value argmax, nearby shortcut, commitment.

The two behaviours with no equivalent in `selector.select_frontier` are that
distance never divides the score, and that a frontier the agent keeps choosing
without reaching gets retired. Both are pinned here, along with the property
that makes the retirement work at all: the bookkeeping is keyed on position, not
on `Frontier.id`, which is reassigned every extraction.
"""
from __future__ import annotations

import numpy as np

from osg.exploration.ascent_selector import (
    FrontierCommitState,
    select_frontier_ascent,
)
from osg.mapping.costmap import FREE, Costmap2D
from osg.mapping.frontier import Frontier


class _StraightLinePlanner:
    """Succeeds for everything, at euclidean cost."""

    def __init__(self, unreachable=()):
        self.unreachable = [np.asarray(u, float) for u in unreachable]
        self.calls = 0

    def plan(self, costmap, start_xy, goal_xy):
        self.calls += 1
        blocked = any(np.linalg.norm(goal_xy - u) < 0.4 for u in self.unreachable)

        class R:
            success = not blocked
            cost = float(np.linalg.norm(np.asarray(goal_xy) - np.asarray(start_xy)))
            path = None
        return R()


def _map() -> Costmap2D:
    cm = Costmap2D(resolution=0.05, size_m=40.0)
    cm.grid[:, :] = FREE
    return cm


def _f(fid: int, x: float, z: float) -> Frontier:
    xy = np.array([x, z], dtype=float)
    return Frontier(id=fid, centroid_xy=xy, cells=np.zeros((0, 2), dtype=int), size=10)


def _select(fronts, values, agent=(0.0, 0.0), state=None, planner=None, **kw):
    state = state or FrontierCommitState()
    planner = planner or _StraightLinePlanner()
    return select_frontier_ascent(
        fronts, values, planner, _map(), np.array(agent, float), state, **kw
    ), state, planner


# ------------------------------------------------------------ value vs distance


def test_value_wins_over_distance():
    """`select_frontier` divides by path cost and would take the near one; this
    must not."""
    near, far = _f(0, 1.0, 0.0), _f(1, 12.0, 0.0)
    best, _, _ = _select([near, far], {0: 0.1, 1: 0.9})
    assert best is far


def test_single_frontier_shortcut_skips_ranking():
    only = _f(0, 9.0, 0.0)
    best, _, planner = _select([only], {})
    assert best is only
    assert planner.calls == 1


# --------------------------------------------------------- the nearby shortcut


def test_nearby_shortcut_beats_a_higher_valued_far_frontier():
    """Within nearby_distance the closest frontier wins outright, whatever the
    value map says -- but only once the first exploration pick has been made."""
    near, far = _f(0, 2.0, 0.0), _f(1, 12.0, 0.0)
    state = FrontierCommitState(finish_first_explore=True)
    best, _, _ = _select([near, far], {0: 0.1, 1: 0.9}, state=state)
    assert best is near


def test_nearby_shortcut_is_inactive_before_the_first_pick():
    near, far = _f(0, 2.0, 0.0), _f(1, 12.0, 0.0)
    best, state, _ = _select([near, far], {0: 0.1, 1: 0.9})
    assert best is far, "value should decide until the shortcut is armed"
    assert state.finish_first_explore, "and the shortcut is armed afterwards"


def test_no_nearby_frontier_commits_to_the_value_pick():
    """What clearing the shortcut flag actually buys.

    In ASCENT the flag goes True -> False -> True within a single call: step 3
    clears it when nothing is nearby (llm_planner.py:106-108), and step 6 sees it
    cleared and sets it back while recording a force frontier (:124-126). So it
    never *ends* a call False; its only observable effect is that the frontier
    just chosen becomes the committed one for the next round.
    """
    far_a, far_b = _f(0, 12.0, 0.0), _f(1, 14.0, 0.0)
    state = FrontierCommitState(finish_first_explore=True)
    best, _, _ = _select([far_a, far_b], {0: 0.9, 1: 0.1}, state=state)
    assert best is far_a
    assert state.force_xy is not None
    assert state.key(state.force_xy) == state.key(far_a.centroid_xy)
    assert state.finish_first_explore is True


# ------------------------------------------------------------------ commitment


def test_repeatedly_choosing_without_closing_retires_the_frontier():
    """The stuck case: the agent keeps picking the same frontier and the
    distance never changes, so it is eventually taken off the table."""
    stuck, other = _f(0, 5.0, 0.0), _f(1, 6.0, 0.0)
    state = FrontierCommitState()
    planner = _StraightLinePlanner()
    for _ in range(STICKY := 25):
        select_frontier_ascent([stuck, other], {0: 0.9, 1: 0.1}, planner, _map(),
                               np.zeros(2), state)
    assert state.is_disabled(stuck.centroid_xy)


def test_progress_keeps_the_frontier_alive():
    """Same frontier every round, but the agent is actually closing on it."""
    target = _f(0, 20.0, 0.0)
    other = _f(1, 21.0, 0.0)
    state = FrontierCommitState()
    planner = _StraightLinePlanner()
    for i in range(25):
        agent = np.array([float(i) * 0.6, 0.0])  # 0.6 m per round > sticky 0.3
        select_frontier_ascent([target, other], {0: 0.9, 1: 0.1}, planner, _map(),
                               agent, state)
    assert not state.is_disabled(target.centroid_xy)


def test_repeated_non_consecutive_selection_retires():
    """Ping-ponging between two frontiers retires them even though neither is
    ever selected twice in a row."""
    a, b = _f(0, 5.0, 0.0), _f(1, -5.0, 0.0)
    state = FrontierCommitState()
    planner = _StraightLinePlanner()
    for i in range(60):
        vals = {0: 0.9, 1: 0.1} if i % 2 == 0 else {0: 0.1, 1: 0.9}
        state.force_xy = None  # otherwise commitment pins one of them
        select_frontier_ascent([a, b], vals, planner, _map(), np.zeros(2), state)
    assert state.is_disabled(a.centroid_xy) or state.is_disabled(b.centroid_xy)


def test_bookkeeping_survives_id_reassignment():
    """The reason state is keyed on position. Frontier ids are handed out fresh
    by every extract() call, so an id-keyed set forgets on the next round."""
    state = FrontierCommitState()
    planner = _StraightLinePlanner()
    for i in range(25):
        # Same place, brand new id every round -- exactly what extract() does.
        f = _f(1000 + i, 5.0, 0.0)
        other = _f(2000 + i, 6.0, 0.0)
        select_frontier_ascent([f, other], {f.id: 0.9, other.id: 0.1}, planner,
                               _map(), np.zeros(2), state)
    assert state.is_disabled(np.array([5.0, 0.0]))


def test_disabled_frontiers_are_excluded_from_ranking():
    top, second = _f(0, 5.0, 0.0), _f(1, 6.0, 0.0)
    state = FrontierCommitState()
    state.disabled.add(state.key(top.centroid_xy))
    best, _, _ = _select([top, second], {0: 0.9, 1: 0.1}, state=state)
    assert best is second


# ----------------------------------------------------- the reachability deviation


def test_unplannable_frontier_falls_through_to_the_next():
    """Marked deviation from ASCENT, which leans on PointNav to make progress
    toward anything. Without this an unreachable frontier is re-selected until
    the repeat counter retires it 20 rounds later."""
    bad, good = _f(0, 5.0, 0.0), _f(1, -5.0, 0.0)
    planner = _StraightLinePlanner(unreachable=[(5.0, 0.0)])
    failed: set = set()
    best, _, _ = _select([bad, good], {0: 0.9, 1: 0.1}, planner=planner,
                         failed_out=failed)
    assert best is good
    assert failed == {0}


def test_all_unplannable_returns_none():
    a, b = _f(0, 5.0, 0.0), _f(1, -5.0, 0.0)
    planner = _StraightLinePlanner(unreachable=[(5.0, 0.0), (-5.0, 0.0)])
    best, _, _ = _select([a, b], {0: 0.9, 1: 0.1}, planner=planner)
    assert best is None


def test_blocked_ids_are_respected():
    a, b = _f(0, 5.0, 0.0), _f(1, -5.0, 0.0)
    best, _, _ = _select([a, b], {0: 0.9, 1: 0.1}, blocked={0})
    assert best is b
