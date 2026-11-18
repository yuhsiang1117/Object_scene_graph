"""Closing the last metre (agent/approach.py `_close_last_metre`).

With `approach_to_viewpoint` the walk ends on the innermost viewpoint ring
that has a FREE costmap cell; beside a bed the inflated costmap has none
nearer than about a metre, so the agent stops 1.0-1.6 m from a track it
placed within 0.6 m of the object and the 1 m rule scores a miss. With
`approach_close_last_metre_m` set, a consumed path farther than that from the
track asks the navmesh for its nearest navigable point to the track and
walks there before the stop.
"""
from __future__ import annotations

import numpy as np

from osg.agent.nav_agent import State

from .test_nav_agent import _frame, make_agent, make_cfg


class _Nav:
    """A navmesh follower fake: `plan` holds one answer per call (None is
    "arrived"), and every goal it was asked for is recorded."""

    def __init__(self, plan=()):
        self.plan = list(plan)
        self.goals = []

    def __call__(self, goal_xy, floor_y=None):
        self.goals.append(np.asarray(goal_xy, dtype=float).copy())
        return self.plan.pop(0) if self.plan else None


def _arrived_agent(nearest, *, close_m=0.8, plan=(), obj=(3.0, 0.0)):
    cfg = make_cfg(use_habitat_navmesh=True, approach_to_viewpoint=True,
                   approach_close_last_metre_m=close_m)
    cfg.verification.stop_at_stale_anchor_once = False
    nav = _Nav(plan)
    agent = make_agent(cfg, target="chair", nav_fn=nav, reachable_fn=lambda *_: True,
                       nearest_navigable_fn=nearest)
    agent.approach.start(np.asarray(obj, dtype=float), agent_xy=np.zeros(2))
    agent._goto_deadline = 10_000
    return agent, nav


def test_off_by_default():
    assert make_cfg().agent.approach_close_last_metre_m == 0.0


def test_a_consumed_path_a_metre_out_walks_to_the_navmesh_point_nearest_the_track():
    calls = []

    def nearest(xy, floor_y=None):
        calls.append(np.asarray(xy, dtype=float).copy())
        return np.array([2.6, 0.0])  # the floor beside the bed, 0.4 m from the track

    # The viewpoint the rings produced is consumed at (1.8, 0): 1.2 m from the
    # track. The follower answers "arrived" there, then walks the closing goal.
    agent, nav = _arrived_agent(nearest, plan=[None, "move_forward"])
    action = agent.approach.step(_frame([1.8, 0.0]))
    assert action == "move_forward" and agent.state is State.APPROACH
    assert np.allclose(calls[0], [3.0, 0.0])
    assert np.allclose(nav.goals[-1], [2.6, 0.0])
    assert agent.stats["approach_close_started"] == 1
    assert agent.approach.closing and not agent.approach.closed
    # The closing walk is consumed: the stop follows from there.
    action = agent.approach.step(_frame([2.6, 0.0]))
    assert action == "stop" and agent.state is State.DONE
    assert agent.approach.closed and agent.stats["approach_close_arrived"] == 1
    assert abs(agent.approach.diag["close_to_m"] - 0.4) < 1e-6


def test_a_closing_walk_that_ends_farther_out_walks_back_to_the_viewpoint():
    agent, nav = _arrived_agent(lambda xy, floor_y=None: np.array([2.6, 0.0]),
                                plan=[None, "move_forward", None, "move_forward"])
    assert agent.approach.step(_frame([1.8, 0.0])) == "move_forward"  # closing
    # The follower gave up at (1.2, 0): 1.8 m out, farther than the 1.2 m it left.
    assert agent.approach.step(_frame([1.2, 0.0])) == "move_forward"  # walking back
    assert np.allclose(nav.goals[-1], [1.8, 0.0]) and agent.stats["approach_close_worse"] == 1
    assert agent.approach.step(_frame([1.8, 0.0])) == "stop"
    assert agent.approach.closed and agent.stats["approach_close_started"] == 1


def test_no_closing_when_the_navmesh_point_is_no_nearer():
    agent, nav = _arrived_agent(lambda xy, floor_y=None: np.array([1.83, 0.0]))
    action = agent.approach.step(_frame([1.8, 0.0]))
    assert action == "stop"
    assert agent.stats["approach_close_no_gain"] == 1 and "approach_close_started" not in agent.stats


def test_no_closing_when_already_within_the_distance():
    calls = []
    agent, nav = _arrived_agent(lambda xy, floor_y=None: calls.append(1))
    action = agent.approach.step(_frame([2.5, 0.0]))  # 0.5 m out, under 0.8
    assert action == "stop" and not calls


def test_no_closing_when_the_flag_is_off():
    calls = []
    agent, nav = _arrived_agent(lambda xy, floor_y=None: calls.append(1), close_m=0.0)
    action = agent.approach.step(_frame([1.8, 0.0]))
    assert action == "stop" and not calls


def test_closing_happens_once_per_approach():
    agent, nav = _arrived_agent(lambda xy, floor_y=None: np.array([2.6, 0.0]))
    # The fake follower has no plan, so the closing walk is consumed at once.
    action = agent.approach.step(_frame([1.8, 0.0]))
    assert action == "stop" and agent.approach.closed
    assert agent.stats["approach_close_started"] == 1
    assert agent.stats["approach_close_arrived"] == 1


def test_arrival_by_distance_ends_a_walk_the_follower_never_would():
    """`approach_arrival_m`: within that distance of the goal the path is
    consumed even while the follower keeps handing out actions."""
    cfg = make_cfg(use_habitat_navmesh=True, approach_to_viewpoint=True, approach_arrival_m=0.3)
    cfg.verification.stop_at_stale_anchor_once = False
    nav = _Nav(["move_forward"] * 50)  # a follower that never says "arrived"
    agent = make_agent(cfg, target="chair", nav_fn=nav, reachable_fn=lambda *_: True)
    agent.approach.start(np.array([3.0, 0.0]), agent_xy=np.zeros(2))
    agent._goto_deadline = 10_000
    goal = agent._goal_xy
    assert agent.approach.step(_frame(goal + np.array([0.5, 0.0]))) == "move_forward"
    # 0.12 m out: arrived. The viewpoint scan turns run first, then the stop.
    for _ in range(20):
        action = agent.approach.step(_frame(goal + np.array([0.12, 0.0])))
        if action == "stop":
            break
    assert action == "stop" and agent.stats["approach_arrived_by_distance"] >= 1
    assert len(nav.plan) >= 30  # the follower was not consulted for the arrival
