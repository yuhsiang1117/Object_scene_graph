"""`floor.prior_stair_mouth: arrival` -- this storey's stair mouth is where the
prior map ARRIVED on it.

`FloorStack.set_current` runs when the estimator commits the new storey, so both
points of a saved `StairEdge` lie on the storey arrived at. The coordinates
below are outputs/maps_v2_osg's: 00873's two edges, 00800's three storeys with
the 1.74 m landing `drop_prior_landings` takes out.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.agent.floor_policy import FloorPolicy
from osg.mapping.floor_stack import StairEdge

from .test_nav_agent import make_cfg


def _policy(levels: dict, current: int, edges, *, mode="arrival", dropped=None, max_failures=0):
    cfg = make_cfg()
    cfg.floor.enabled = True
    cfg.floor.cross_floor = True
    cfg.floor.stairs = True
    cfg.floor.use_prior_stairs = True
    cfg.floor.prior_stair_mouth = mode
    cfg.floor.prior_stair_max_failures = max_failures
    policy = FloorPolicy(cfg, stats={})
    policy.stack.current_id = current
    policy.stack.stair_edges = list(edges)
    policy.stack._layers = {}
    policy.estimator._levels = dict(levels)
    policy.stack.dropped_levels = dict(dropped or {})
    return policy


def _p(*xy):
    return np.array(xy, dtype=float)


UPPER_00873, LOWER_00873 = 0, 1
EDGES_00873 = [
    # pass 1 went down at step 625 and arrived at (-3.27, -4.96) on the lower storey...
    StairEdge(UPPER_00873, LOWER_00873, entry_xy=_p(-3.27, -4.96), exit_xy=_p(-3.27, -4.96), step=625),
    # ...and up at step 283, arriving 0.3 m from the true lip (-4.4, -4.3).
    StairEdge(LOWER_00873, UPPER_00873, entry_xy=_p(-4.19, -4.10), exit_xy=_p(-4.19, -4.10), step=283),
]
LEVELS_00873 = {UPPER_00873: 0.22, LOWER_00873: -2.98}


def test_the_edge_reading_sends_00873s_descent_to_a_point_on_the_lower_storey():
    """As measured: the most recent edge leaving this storey wins, and its
    `entry_xy` is where it ARRIVED -- over the stairwell, a storey down."""
    policy = _policy(LEVELS_00873, UPPER_00873, EDGES_00873, mode="edge")
    goal, other = policy._remembered_stair(LOWER_00873)
    assert other == LOWER_00873
    assert goal == pytest.approx([-3.27, -4.96])


def test_the_arrival_reading_finds_00873s_lip():
    policy = _policy(LEVELS_00873, UPPER_00873, EDGES_00873)
    goal, other = policy._remembered_stair(LOWER_00873)
    assert other == LOWER_00873
    assert goal == pytest.approx([-4.19, -4.10])


def test_and_from_the_lower_storey_the_foot():
    policy = _policy(LEVELS_00873, LOWER_00873, EDGES_00873)
    goal, other = policy._remembered_stair(UPPER_00873)
    assert other == UPPER_00873
    assert goal == pytest.approx([-3.27, -4.96])


# 00800: storeys 3.16 and 0.64 (the real 0.16), the 1.74 landing dropped.
TOP, LANDING, BOTTOM = 0, 1, 2
EDGES_00800 = [
    StairEdge(TOP, LANDING, entry_xy=_p(-10.52, -4.55), exit_xy=_p(-10.52, -4.55), step=358),
    StairEdge(LANDING, TOP, entry_xy=_p(-8.21, -4.62), exit_xy=_p(-8.21, -4.62), step=1119),
    StairEdge(LANDING, BOTTOM, entry_xy=_p(-11.53, -4.14), exit_xy=_p(-11.53, -4.14), step=457),
    StairEdge(LANDING, BOTTOM, entry_xy=_p(-11.03, -4.16), exit_xy=_p(-11.53, -4.14), step=468),
]


def test_an_arrival_from_a_dropped_landing_counts_by_the_landings_height():
    """00800's flight top is (-7.35, -4.68): pass 1 arrived on the top storey
    from the landing, never from the storey the descent is for."""
    policy = _policy({TOP: 3.16, BOTTOM: 0.64}, TOP, EDGES_00800, dropped={LANDING: 1.74})
    goal, other = policy._remembered_stair(BOTTOM)
    assert other == BOTTOM
    assert goal == pytest.approx([-8.21, -4.62])


def test_the_most_recent_arrival_wins_among_equals():
    """Foot (-11.67, -3.37): both arrivals of step 468 lie beside it."""
    policy = _policy({TOP: 3.16, BOTTOM: 0.64}, BOTTOM, EDGES_00800, dropped={LANDING: 1.74})
    goal, other = policy._remembered_stair(TOP)
    assert other == TOP
    assert goal == pytest.approx([-11.03, -4.16]) or goal == pytest.approx([-11.53, -4.14])


def test_an_arrival_from_the_wrong_side_is_not_a_way_there():
    """00878: pass 1 arrived on the middle storey only from the basement, so it
    knows no mouth there toward the top storey."""
    policy = _policy({0: -2.8, 1: 0.07, 2: 2.8}, 1, [
        StairEdge(0, 1, entry_xy=_p(1.38, 1.23), exit_xy=_p(1.38, 1.23), step=247),
    ])
    assert policy._remembered_stair(2) is None
    goal, other = policy._remembered_stair(0)
    assert other == 0


def test_a_landing_whose_height_is_unknown_is_skipped():
    policy = _policy({TOP: 3.16, BOTTOM: 0.64}, TOP, EDGES_00800)  # no dropped_levels
    assert policy._remembered_stair(BOTTOM) is None


def test_an_undirected_request_names_the_nearest_storey_on_that_side():
    policy = _policy({TOP: 3.16, BOTTOM: 0.64}, TOP, EDGES_00800, dropped={LANDING: 1.74})
    goal, other = policy._remembered_stair(None)
    assert other == BOTTOM
    assert goal == pytest.approx([-8.21, -4.62])


def test_a_mouth_is_given_up_only_after_max_failures_beside_it():
    policy = _policy(LEVELS_00873, UPPER_00873, EDGES_00873, max_failures=2)
    policy._failed_portals = [_p(-4.6, -5.2)]                  # 1.2 m away: one failure
    assert policy._remembered_stair(LOWER_00873) is not None
    policy._failed_portals.append(_p(-4.0, -4.0))
    assert policy._remembered_stair(LOWER_00873) is None
    assert policy.stats["prior_stair_skipped_after_failures"] >= 1


def test_without_max_failures_a_failure_never_hides_it():
    """As measured: the remembered branch had no failure check."""
    policy = _policy(LEVELS_00873, UPPER_00873, EDGES_00873)
    policy._failed_portals = [_p(-4.19, -4.10)] * 5
    assert policy._remembered_stair(LOWER_00873) is not None


def test_an_unknown_mode_fails_fast():
    policy = _policy(LEVELS_00873, UPPER_00873, EDGES_00873, mode="mouth")
    with pytest.raises(ValueError):
        policy._remembered_stair(LOWER_00873)
