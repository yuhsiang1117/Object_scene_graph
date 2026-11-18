"""Three gates added after reading the 00800/00821 cross-anchor episodes.

None of them rescued an episode on its own -- every failure measured there is
the climb gaining 0.00 m, which is a different defect -- so what is pinned here
is that each does what it claims and, in particular, that none of them can make
progress impossible. A verdict that empties the only list an agent has is worse
than the behaviour it replaces.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


# ------------------------------------------- the storey verdict gates flights

class _Flight:
    def __init__(self, kind):
        self.kind = kind


def _policy(levels, current=0, disproved=()):
    from osg.agent.floor_policy import FloorPolicy
    policy = FloorPolicy.__new__(FloorPolicy)
    policy.stats = {}
    policy.disproved_floors = set(disproved)
    policy.estimator = SimpleNamespace(levels=dict(levels))
    policy.stack = SimpleNamespace(
        current_id=current,
        _layers={k: SimpleNamespace(floor_y=v) for k, v in levels.items()},
    )
    return policy


def test_known_levels_unions_the_stack_with_the_estimator():
    """A storey can be KNOWN without having been stood on: `_seed_storeys`
    creates a layer for every cluster the prior obstacle map found, and
    `apply_map` restores the scene graph's. The online estimator only learns a
    level by standing at that height."""
    from osg.agent.floor_policy import FloorPolicy
    policy = _policy({0: 0.13})
    policy.stack._layers[1] = SimpleNamespace(floor_y=-3.47)

    levels = FloorPolicy.known_levels(policy)

    assert levels == {0: 0.13, 1: -3.47}


def test_a_flight_back_to_a_disproved_storey_is_dropped():
    """00800's toy airplane: two attempts failed upstairs, storey 0 was
    disproved, the agent descended and ARRIVED on storey 1 at step 455 -- then
    took a `flight_up` at step 506 and spent its last attempt back on the
    disproved storey. The verdict gated the floor LLM, the candidate channel
    and the posterior, but not the choice of where to walk."""
    from osg.agent.floor_policy import FloorPolicy
    policy = _policy({0: 3.16, 1: 0.16}, current=1, disproved={0})
    flights = [_Flight("up"), _Flight("down")]

    kept = FloorPolicy._drop_disproved(policy, flights, floor_y=0.16)

    assert [f.kind for f in kept] == ["down"]
    assert policy.stats["flights_skipped_disproved"] == 1


def test_it_never_leaves_the_agent_with_no_flight_at_all():
    """A verdict may not make progress impossible -- the same rule
    `_rank_flights` and `ExplorationStrategy` already keep."""
    from osg.agent.floor_policy import FloorPolicy
    policy = _policy({0: 3.16, 1: 0.16}, current=1, disproved={0})
    flights = [_Flight("up")]

    kept = FloorPolicy._drop_disproved(policy, flights, floor_y=0.16)

    assert [f.kind for f in kept] == ["up"]
    assert "flights_skipped_disproved" not in policy.stats


def test_with_no_verdict_nothing_is_dropped():
    from osg.agent.floor_policy import FloorPolicy
    policy = _policy({0: 3.16, 1: 0.16}, current=1)
    flights = [_Flight("up"), _Flight("down")]
    assert len(FloorPolicy._drop_disproved(policy, flights, 0.16)) == 2


def test_the_destination_of_a_flight_is_the_nearest_level_that_way():
    from osg.agent.floor_policy import FloorPolicy
    policy = _policy({0: 3.16, 1: 0.16, 2: -3.0}, current=1)
    assert FloorPolicy._destination_key(policy, "up", 0.16) == 0
    assert FloorPolicy._destination_key(policy, "down", 0.16) == 2
    # nothing below the bottom storey
    policy2 = _policy({0: 3.16, 1: 0.16}, current=1)
    assert FloorPolicy._destination_key(policy2, "down", 0.16) is None


# ----------------------------------------------- the height prior at commit

class _Track:
    def __init__(self, tid, n_obs):
        self.id = tid
        self.n_obs = n_obs


def _channel(limit, min_obs, floor_y, centres):
    from osg.agent.candidate import CandidatePolicy
    channel = CandidatePolicy.__new__(CandidatePolicy)
    channel.nav = SimpleNamespace(
        cfg=SimpleNamespace(verification=SimpleNamespace(
            commit_max_above_storey_m=limit, commit_high_min_obs=min_obs)),
        floors=SimpleNamespace(current_id=0, height_of=lambda _k: floor_y),
        object_layer=SimpleNamespace(center_of=lambda t: np.array(centres[t.id])),
        stats={},
    )
    return channel


def test_a_one_shot_track_metres_above_the_storey_is_held_back():
    """00800's toy airplane spent BOTH early attempts, by step 95, on
    single-observation tracks at y 5.64 and 5.09 above a 3.16 m storey, while
    the true object was at 0.96 on the other storey."""
    from osg.agent.candidate import CandidatePolicy
    high, low = _Track(1, 1), _Track(2, 3)
    channel = _channel(1.0, 2, 3.16, {1: (0, 5.64, 0), 2: (0, 3.9, 0)})

    kept = CandidatePolicy._height_plausible(channel, [high, low])

    assert [t.id for t in kept] == [2]
    assert channel.nav.stats["commit_held_high_track"] == 1


def test_corroboration_lets_a_high_track_compete_again():
    """A real object CAN sit high on a shelf, so this asks for a second look
    rather than striking the track off."""
    from osg.agent.candidate import CandidatePolicy
    seen_twice = _Track(1, 2)
    channel = _channel(1.0, 2, 3.16, {1: (0, 5.64, 0)})
    assert CandidatePolicy._height_plausible(channel, [seen_twice]) == [seen_twice]


def test_when_every_candidate_is_high_the_best_is_still_offered():
    """Refusing to commit at all is how an agent ends an episode having done
    nothing."""
    from osg.agent.candidate import CandidatePolicy
    a, b = _Track(1, 1), _Track(2, 1)
    channel = _channel(1.0, 2, 3.16, {1: (0, 5.64, 0), 2: (0, 5.09, 0)})
    assert CandidatePolicy._height_plausible(channel, [a, b]) == [a, b]


def test_the_gate_is_off_by_default():
    from osg.agent.candidate import CandidatePolicy
    a = _Track(1, 1)
    channel = _channel(0.0, 2, 3.16, {1: (0, 99.0, 0)})
    assert CandidatePolicy._height_plausible(channel, [a]) == [a]


# ------------------------------------------- where the agent joins a flight

def test_the_mouth_of_a_descent_is_its_top_not_its_foot():
    """`Flight.foot_xy` is the LOWEST tread by definition and `top_xy` the
    highest, so the foot is the mouth of an ascent and the top is the mouth of
    a descent. Every place the agent asked "where do I step on" used the foot,
    which on a descent is a point at the bottom of the staircase -- on the
    storey it has not reached yet.
    """
    from osg.mapping.stairs import mouth_xy
    up = SimpleNamespace(kind="up", foot_xy=np.array([1.0, 0.0]),
                         top_xy=np.array([4.0, 0.0]))
    down = SimpleNamespace(kind="down", foot_xy=np.array([4.0, 0.0]),
                           top_xy=np.array([1.0, 0.0]))

    assert np.allclose(mouth_xy(up), [1.0, 0.0])
    assert np.allclose(mouth_xy(down), [1.0, 0.0])


def test_the_mouth_is_what_the_flight_ranking_measures():
    """Ranking by the foot picks, on a descent, the flight whose BOTTOM is
    nearest -- which is not the one whose entrance is nearest."""
    from osg.agent.floor_policy import FloorPolicy
    from osg.mapping.stairs import mouth_xy
    near_mouth = SimpleNamespace(kind="down", foot_xy=np.array([9.0, 0.0]),
                                 top_xy=np.array([1.0, 0.0]), cells_rc=np.zeros((1, 2), int))
    near_foot = SimpleNamespace(kind="down", foot_xy=np.array([0.5, 0.0]),
                                top_xy=np.array([8.0, 0.0]), cells_rc=np.zeros((1, 2), int))
    policy = FloorPolicy.__new__(FloorPolicy)
    # `costmap` is a read-only property; the corroboration lookup is what reads
    # it, so stub that instead and leave both flights uncorroborated.
    policy._on_stair_mask = lambda _f: False
    policy.stats = {}
    agent_xy = np.array([0.0, 0.0])

    chosen = FloorPolicy._best_flight(policy, [near_mouth, near_foot], agent_xy)

    assert chosen is near_mouth, "the flight whose ENTRANCE is nearest"
    assert np.allclose(mouth_xy(chosen), [1.0, 0.0])


def test_with_descent_goal_mouth_a_descent_is_entered_at_its_highest_tread():
    """`find_flights` keeps a descent's highest tread in `foot_xy`; under
    `floor.descent_goal: mouth` that is the entrance the ranking measures."""
    from osg.agent.floor_policy import FloorPolicy
    from osg.mapping.stairs import mouth_xy
    near_top = SimpleNamespace(kind="down", foot_xy=np.array([1.0, 0.0]),
                               top_xy=np.array([9.0, 0.0]), cells_rc=np.zeros((1, 2), int))
    far_top = SimpleNamespace(kind="down", foot_xy=np.array([8.0, 0.0]),
                              top_xy=np.array([0.5, 0.0]), cells_rc=np.zeros((1, 2), int))
    policy = FloorPolicy.__new__(FloorPolicy)
    policy.cfg = SimpleNamespace(floor=SimpleNamespace(descent_goal="mouth"))
    policy._on_stair_mask = lambda _f: False
    policy.stats = {}
    chosen = FloorPolicy._best_flight(policy, [near_top, far_top], np.array([0.0, 0.0]))
    assert chosen is near_top
    assert np.allclose(mouth_xy(chosen, descent="mouth"), [1.0, 0.0])
    # The legacy reading sends the same agent to the far end.
    assert np.allclose(mouth_xy(chosen), [9.0, 0.0])
