"""ASCENT's control flow (S37), on OSG's perception.

The point of the arm is that the DIFFERENCES are structural, so these tests pin
the structure: no committed states, the object goal re-aimed every step with
hysteresis, and a network STOP never concluding anything.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.agent.ascent_agent import AscentAgent
from osg.agent.nav_agent import State
from osg.planning.pointnav_driver import NavStep

from .test_nav_agent import _frame, make_agent, make_cfg


class _Driver:
    stop_radius = 0.9

    def __init__(self, reason="moving", action="move_forward"):
        self.reason, self.action = reason, action
        self.goals = []

    def observe(self, frame):
        pass

    def reset(self):
        pass

    def step(self, goal_xy, **kw):
        self.goals.append(np.asarray(goal_xy, dtype=float).copy())
        return NavStep(self.action, self.reason)

    def __call__(self, goal_xy, **kw):
        return self.step(goal_xy, **kw).action


def _agent(driver=None, **over):
    from osg.exploration.async_scorer import AsyncScorer
    from osg.perception.detector import StubDetector

    from .test_nav_agent import _StubScorer

    opts = dict(navigation="pointnav", initial_scan=False,
                terminal_rule="nearest_point", approach_abandon_steps=100)
    opts.update(over)
    cfg = make_cfg(**opts)
    a = AscentAgent(cfg, StubDetector(), AsyncScorer(_StubScorer()), None, "chair",
                    pointnav=driver or _Driver())
    a.costmap.grid[:, :] = 0
    a.reset("chair")
    return a


def _give_cloud(agent, xy, dist_axis=0.0):
    from osg.objects.association import ObjectTrack
    from osg.objects.ellipsoid import Ellipsoid

    t = ObjectTrack(id=0, label="chair",
                    ellipsoid=Ellipsoid(center=np.array([xy[0], 0.5, xy[1]]),
                                        axes=np.full(3, 0.1), R=np.eye(3)))
    t.points_w = np.array([[xy[0], 0.5, xy[1]]])
    agent.object_layer._tracks[0] = t
    agent._candidate_id = 0
    return t


# ------------------------------------------------------------ the goal re-aims


def test_goal_tracks_the_cloud_when_it_moves_far_enough():
    a = _agent()
    t = _give_cloud(a, (5.0, 0.0))
    g0 = a._object_goal(np.zeros(2))
    assert g0 == pytest.approx([5.0, 0.0])

    t.points_w = np.array([[3.0, 0.5, 0.0]])  # a 2 m correction
    g1 = a._object_goal(np.zeros(2))
    assert g1 == pytest.approx([3.0, 0.0]), "a large cloud move must re-aim"


def test_small_moves_are_ignored():
    """Without hysteresis the goal jitters on every mask update, and the driver
    resets its recurrent state on any move over 0.1 m."""
    a = _agent()
    t = _give_cloud(a, (5.0, 0.0))
    a._object_goal(np.zeros(2))
    t.points_w = np.array([[5.05, 0.5, 0.0]])  # 5 cm
    assert a._object_goal(np.zeros(2)) == pytest.approx([5.0, 0.0])


def test_medium_moves_are_ignored_only_while_far_away():
    a = _agent()
    t = _give_cloud(a, (5.0, 0.0))
    a._object_goal(np.zeros(2))
    t.points_w = np.array([[5.3, 0.5, 0.0]])  # 0.3 m: under 0.5
    assert a._object_goal(np.zeros(2)) == pytest.approx([5.0, 0.0]), "far: keep"
    # Same 0.3 m move, but now standing close to it: take it.
    assert a._object_goal(np.array([5.0, 0.0])) == pytest.approx([5.3, 0.0])


def test_goal_is_none_without_a_candidate():
    a = _agent()
    assert a._object_goal(np.zeros(2)) is None


# ------------------------------------------------------------ no FSM latching


def test_losing_the_candidate_returns_to_exploring_the_same_step():
    """No committed APPROACH: the dispatch re-decides from the map each step."""
    a = _agent()
    t = _give_cloud(a, (5.0, 0.0))
    a._dispatch(_frame([0.0, 0.0]), 0.0, a.floors.current())
    assert a.state is State.APPROACH
    t.blacklisted = True
    a._dispatch(_frame([0.0, 0.0]), 0.0, a.floors.current())
    assert a.state is State.EXPLORE


def test_the_opening_scan_completes_before_any_target_is_chased():
    """ASCENT orders it this way: the `not done_initializing` branch precedes
    `elif goal is None` (`ascent_policy.py:566-576`), so a target spotted during
    the opening spin waits for the spin to finish. Pinned because the opposite
    is the intuitive guess and would be a silent divergence."""
    a = _agent(initial_scan=True)
    a.reset("chair")
    assert a._ascent_init_left == 12
    _give_cloud(a, (5.0, 0.0))
    for _ in range(12):
        assert a._dispatch(_frame([0.0, 0.0]), 0.0, a.floors.current()) == "turn_left"
    # spin done -> the target it already knows about is chased immediately
    a._dispatch(_frame([0.0, 0.0]), 0.0, a.floors.current())
    assert a.state is State.APPROACH


# --------------------------------------------------- STOP concludes nothing


def test_network_stop_while_exploring_forces_forward():
    from osg.mapping.frontier import Frontier

    d = _Driver(reason="policy_stop", action=None)
    a = _agent(driver=d)
    f = Frontier(id=1, centroid_xy=np.array([6.0, 0.0]),
                 cells=np.zeros((0, 2), dtype=int), size=30)
    f.path_cost = 1.0
    a._select_new_frontier = lambda frame: setattr(a, "_current_frontier", f)
    assert a._ascent_explore(_frame([0.0, 0.0]), np.zeros(2)) == "move_forward"
    assert a.stats["ascent_explore_forced_forward"] == 1


def test_network_stop_while_navigating_forces_forward():
    d = _Driver(reason="policy_stop", action=None)
    a = _agent(driver=d)
    _give_cloud(a, (9.0, 0.0))
    assert a._ascent_navigate(_frame([0.0, 0.0]), np.zeros(2), np.array([9.0, 0.0])) \
        == "move_forward"
    assert a.stats["ascent_navigate_forced_forward"] == 1


def test_navigate_abandons_on_the_step_budget():
    a = _agent()
    _give_cloud(a, (9.0, 0.0))
    a._target_obj_xy = np.array([9.0, 0.0])
    a._navigate_steps = 99
    a._ascent_navigate(_frame([0.0, 0.0]), np.zeros(2), np.array([9.0, 0.0]))
    assert a.stats.get("approach_abandon") == 1
    assert a.state is State.EXPLORE


# ------------------------------------------------------------------ damping


def test_every_frontier_choice_feeds_the_commit_state():
    """The damping S34 left behind: without observe() the sticky counter and
    force_xy never accumulate and re-selection thrashes."""
    from osg.mapping.frontier import Frontier

    a = _agent()
    if a.commit_state is None:
        pytest.skip("commit state only exists on the ascent selector")
    f = Frontier(id=1, centroid_xy=np.array([6.0, 0.0]),
                 cells=np.zeros((0, 2), dtype=int), size=30)
    f.path_cost = 1.0
    a._select_new_frontier = lambda frame: setattr(a, "_current_frontier", f)
    a._ascent_explore(_frame([0.0, 0.0]), np.zeros(2))
    assert a.commit_state.last_xy is not None
