"""The staircase the prior map already walked, read back.

`save_map` writes every committed floor transition into the prior map's
`connectivity`, and `apply_map` restores them into `FloorStack.stair_edges`.
Until `floor.use_prior_stairs`, nothing consumed them: an agent that knew
exactly where the stairs were still had to rediscover them by chance.

Measured on outputs/osg_authored_15, that chance is poor. Of 11 cross-floor
episodes, 4 never saw a portal and never attempted a switch, and across the run
405 storey requests produced 12 attempts.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from osg.agent.floor_policy import FloorPolicy
from osg.mapping.floor_stack import StairEdge

from .test_nav_agent import make_cfg

GROUND, UPPER = 0, 1


def _policy(use_prior_stairs: bool, edges):
    cfg = make_cfg()
    cfg.floor.enabled = True
    cfg.floor.cross_floor = True
    cfg.floor.stairs = True
    cfg.floor.use_prior_stairs = use_prior_stairs
    policy = FloorPolicy(cfg, stats={})
    policy.stack.current_id = GROUND
    policy.stack.stair_edges = list(edges)
    policy.estimator._levels = {GROUND: 0.0, UPPER: 2.9}
    return policy


def _remembered(policy, target_floor=None):
    return policy._remembered_stair(target_floor)


def test_the_mouth_on_this_floor_is_the_edge_we_walked_up_from():
    """`entry_xy` is where the agent stood when it committed to the new storey:
    the mouth of the staircase on the floor it LEFT."""
    policy = _policy(True, [StairEdge(GROUND, UPPER, entry_xy=np.array([1.9, 0.5]),
                                      exit_xy=np.array([2.1, 0.9]), step=132)])
    goal, other = _remembered(policy)
    assert other == UPPER
    assert goal == pytest.approx([1.9, 0.5])


def test_an_edge_walked_the_other_way_still_names_a_point_here():
    """Coming DOWN from the upper floor lands at `exit_xy` on this one, which is
    the same staircase seen from the other end."""
    policy = _policy(True, [StairEdge(UPPER, GROUND, entry_xy=np.array([2.1, 0.9]),
                                      exit_xy=np.array([1.9, 0.5]), step=200)])
    goal, other = _remembered(policy)
    assert other == UPPER
    assert goal == pytest.approx([1.9, 0.5])


def test_a_directed_request_ignores_a_staircase_to_the_wrong_storey():
    policy = _policy(True, [StairEdge(GROUND, 2, entry_xy=np.array([5.0, 5.0]), step=10)])
    policy.estimator._levels[2] = 5.8
    assert _remembered(policy, target_floor=UPPER) is None
    assert _remembered(policy, target_floor=2) is not None


def test_the_most_walked_staircase_wins():
    """A staircase pass 1 used repeatedly is the one that works."""
    policy = _policy(True, [
        StairEdge(GROUND, UPPER, entry_xy=np.array([9.0, 9.0]), step=400, n_traversals=1),
        StairEdge(GROUND, UPPER, entry_xy=np.array([1.9, 0.5]), step=132, n_traversals=3),
    ])
    goal, _ = _remembered(policy)
    assert goal == pytest.approx([1.9, 0.5])


def test_edges_that_do_not_touch_this_floor_are_ignored():
    policy = _policy(True, [StairEdge(1, 2, entry_xy=np.array([4.0, 4.0]), step=10)])
    assert _remembered(policy) is None


def test_an_edge_with_no_recorded_point_is_not_a_goal():
    policy = _policy(True, [StairEdge(GROUND, UPPER, entry_xy=None, exit_xy=None)])
    assert _remembered(policy) is None


def test_a_prior_map_with_no_transitions_offers_nothing():
    """Pass 1 never changed floor, so there is no staircase to remember and the
    agent is back to looking for one."""
    assert _remembered(_policy(True, [])) is None


# ------------------------------------ preferring it over a detected portal

def test_a_sightline_portal_loses_to_a_staircase_we_walked():
    """`find_portals` finds a patch of another storey VISIBLE from here. Over a
    balcony rail that is somewhere you can see the next floor and cannot walk up
    from. Measured on 00821's cracker box: 55 such goals, every one a real
    2.69 m gap, the agent arrived at one and rose 0.17 m in 500 steps.

    `connectivity` records where pass 1 actually changed floor, which is a
    staircase by construction."""
    policy = _policy(True, [StairEdge(GROUND, UPPER, entry_xy=np.array([1.9, 0.5]), step=132)])
    policy.cfg.floor.prefer_prior_stairs = True
    goal, other = _remembered(policy)
    assert other == UPPER and goal == pytest.approx([1.9, 0.5])


def test_preferring_is_off_by_default():
    policy = _policy(True, [StairEdge(GROUND, UPPER, entry_xy=np.array([1.9, 0.5]), step=132)])
    assert not bool(getattr(policy.cfg.floor, "prefer_prior_stairs", False))


def test_with_no_remembered_staircase_the_portals_still_stand():
    """Preferring something that does not exist must not veto what does."""
    policy = _policy(True, [])
    policy.cfg.floor.prefer_prior_stairs = True
    assert _remembered(policy) is None


# --------------------------- when the gate itself is what refuses

def test_a_known_staircase_can_satisfy_a_gate_geometry_will_not():
    """`may_switch`'s last clause is "nothing near is left on this floor", which
    a large storey never satisfies. Measured on outputs/crossfloor_ab, 3 of 7
    cross-floor episodes returned here every round and never made one switch
    attempt, rising 0.00 to 0.19 m in 500 steps."""
    from osg.mapping.portals import FloorSwitchPolicy

    policy = FloorSwitchPolicy(max_steps=500, no_switch_before=50,
                               min_interval_steps=50, no_switch_after_frac=0.7)
    # The ordinary gate refuses: a near frontier still exists.
    assert not policy.may_switch(200, best_path_cost=1.0)
    # The narrower one allows it, because the risk that clause prices is gone.
    assert policy.may_switch_to_known_stairs(200, steps_on_floor=200)


def test_the_timing_guards_still_hold():
    """Budget guards are not about evidence, so a known staircase does not lift
    them: not in the last third, not twice in quick succession, and not before
    this floor has been looked at."""
    from osg.mapping.portals import FloorSwitchPolicy

    policy = FloorSwitchPolicy(max_steps=500, no_switch_before=50,
                               min_interval_steps=50, no_switch_after_frac=0.7)
    assert not policy.may_switch_to_known_stairs(400, steps_on_floor=400)  # past 350
    assert not policy.may_switch_to_known_stairs(200, steps_on_floor=10)   # just arrived
    policy.note_switch(190)
    assert not policy.may_switch_to_known_stairs(200, steps_on_floor=200)  # too soon


def test_steps_on_floor_is_what_counts_not_the_episode_step():
    """Arriving on a new storey must not immediately license leaving it."""
    from osg.mapping.portals import FloorSwitchPolicy

    policy = FloorSwitchPolicy(max_steps=500, no_switch_before=50,
                               min_interval_steps=0, no_switch_after_frac=0.7)
    assert not policy.may_switch_to_known_stairs(300, steps_on_floor=5)
    assert policy.may_switch_to_known_stairs(300, steps_on_floor=60)


def test_pursuing_a_storey_known_only_from_the_prior_map_does_not_raise():
    """`known_levels()` holds storeys the online estimator never stood on (the
    stack's layers, seeded from the prior obstacle map). `_remembered_stair`
    may name one; `_pursue_remembered` read its height from the estimator and
    raised KeyError (mfv2 00800 seed 42, 2026-11-06)."""
    policy = _policy(True, [StairEdge(GROUND, 2, entry_xy=np.array([5.0, 5.0]), step=10)])
    policy.stack._layers = {2: SimpleNamespace(floor_y=5.8)}
    assert 2 not in policy.estimator._levels
    remembered = _remembered(policy, target_floor=2)
    assert remembered is not None and remembered[1] == 2
    frame = SimpleNamespace(camera_position=np.array([0.0, 1.2, 0.0]))
    goal = policy._pursue_remembered(frame, 0, remembered, 0.0, None, True, [])
    assert goal is not None and goal.target_y == pytest.approx(5.8)
