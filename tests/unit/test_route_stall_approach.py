"""`agent.climb_route_approach_steps`: the route climb's stall rule allows the
walk to the flight.

A route climb starts up to `climb_route_start_m` out on the flat floor. On 00873
(seed 42, static 50002, mfv2_osg_descent) the walk toward the lip and two
12-turn look-rounds took the climb's first 41 steps with no height change, and
the 40-step stall rule ended it about 1 m from the lip.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from osg.agent.climb import ClimbPolicy


def _climber(approach: int, stall: int = 40):
    c = SimpleNamespace(
        cfg=SimpleNamespace(agent=SimpleNamespace(climb_route_stall_steps=stall,
                                                  climb_route_approach_steps=approach)),
        step_count=0, _climb_max_dy=0.0, _route_best_dy=0.0, _route_progress_step=0)
    return c


def _stalled_at(c, step: int, dy: float = 0.0) -> bool:
    c.step_count = step
    c._climb_max_dy = max(c._climb_max_dy, dy)
    return ClimbPolicy._route_stalled(c, np.zeros(2))


def test_as_measured_41_flat_steps_end_the_climb():
    c = _climber(approach=0)
    assert not _stalled_at(c, 40)
    assert _stalled_at(c, 41)


def test_the_approach_allowance_covers_the_walk_to_the_flight():
    c = _climber(approach=60)
    assert not _stalled_at(c, 41)
    assert not _stalled_at(c, 100)
    assert _stalled_at(c, 101)


def test_once_on_the_treads_the_ordinary_limit_applies():
    c = _climber(approach=60)
    assert not _stalled_at(c, 30, dy=0.3)        # gained: the clock restarts at 30
    assert not _stalled_at(c, 70, dy=0.3)
    assert _stalled_at(c, 71, dy=0.3)
