"""The mover that posts a goal instead of steering.

`Nav2Driver` decides almost nothing -- which is the point -- so what is worth
pinning is the handful of places it does decide: when a goal is re-posted, how
the navigator's verdict maps onto the `NavStep` reasons the FSM already knows,
and who owns the base on a given tick.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.core.types import CameraIntrinsics
from osg.planning.nav2_driver import DRIVING, BackendStatus, Nav2Driver

from .conftest import make_frame

_K = CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0, width=640, height=480)


class FakeBackend:
    """A navigator whose verdict the test dictates."""

    def __init__(self, state="active", action=None):
        self.state, self.action = state, action
        self.goals, self.cancels, self.polls = [], 0, 0

    def observe(self, frame):
        self.seen = frame

    def send_goal(self, goal_xy):
        self.goals.append((np.asarray(goal_xy, float).copy(), None))

    def cancel(self):
        self.cancels += 1

    def poll(self):
        self.polls += 1
        return BackendStatus(self.state, action=self.action)


def _driver(backend=None, **kw):
    return Nav2Driver(backend or FakeBackend(), **kw)


def _at(driver, xy):
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = float(xy[0]), 0.88, float(xy[1])
    driver.observe(make_frame(_K, T))
    return driver


# ------------------------------------------------------------------ posting


def test_the_first_step_posts_the_goal():
    b = FakeBackend()
    d = _at(_driver(b), (0.0, 0.0))
    step = d.step(np.array([5.0, 0.0]))
    assert len(b.goals) == 1 and np.allclose(b.goals[0][0], [5.0, 0.0])
    assert step == (DRIVING, "moving")
    assert d.n_goals_sent == 1


def test_a_goal_that_barely_moves_is_not_re_posted():
    """A frontier centroid shifts by a cell as the map fills. Re-posting would
    restart Nav2's global planner every step."""
    b = FakeBackend()
    d = _at(_driver(b, goal_resend_m=0.5), (0.0, 0.0))
    d.step(np.array([5.0, 0.0]))
    d.step(np.array([5.2, 0.0]))
    _at(d, (0.1, 0.0)).step(np.array([5.3, 0.1]))
    assert len(b.goals) == 1


def test_a_goal_that_really_moves_is_re_posted():
    b = FakeBackend()
    d = _at(_driver(b, goal_resend_m=0.5), (0.0, 0.0))
    d.step(np.array([5.0, 0.0]))
    d.step(np.array([-4.0, 2.0]))
    assert len(b.goals) == 2 and np.allclose(b.goals[1][0], [-4.0, 2.0])


# ------------------------------------------------- the navigator's verdicts


def test_arrival_is_tested_before_anything_is_posted():
    """PointNavDriver learned this the hard way: a radius test that runs after
    the creep can never report arriving."""
    b = FakeBackend()
    d = _at(_driver(b, stop_radius=0.9), (0.0, 0.0))
    assert d.step(np.array([0.5, 0.0])) == (None, "arrived")
    assert b.goals == [] and b.polls == 0


def test_an_overridden_radius_wins():
    """`_follow_to` passes `agent.pointnav_arrival_m` for an approach, which is
    much tighter than the frontier radius."""
    d = _at(_driver(stop_radius=0.9), (0.0, 0.0))
    assert d.step(np.array([0.5, 0.0]), stop_radius=0.3) == (DRIVING, "moving")


def test_reaching_the_goal_cancels_the_outstanding_nav_goal():
    """Otherwise the base keeps driving to a goal the FSM has moved on from."""
    b = FakeBackend()
    d = _at(_driver(b, stop_radius=0.9), (0.0, 0.0))
    d.step(np.array([5.0, 0.0]))
    before = b.cancels
    _at(d, (4.8, 0.0)).step(np.array([5.0, 0.0]))
    assert b.cancels == before + 1
    assert d.goal_active is False


def test_nav2_succeeding_is_an_arrival():
    b = FakeBackend(state="succeeded")
    d = _at(_driver(b, stop_radius=0.9), (0.0, 0.0))
    assert d.step(np.array([5.0, 0.0])) == (None, "arrived")
    assert d.goal_active is False


@pytest.mark.parametrize("state", ["aborted", "rejected"])
def test_nav2_giving_up_is_the_blocked_reason_the_fsm_already_knows(state):
    """`policy_stop` is what `agent.pointnav_stop_means_blocked` reads to retire
    a frontier. Reusing it is why the FSM needed no new branch."""
    b = FakeBackend(state=state)
    d = _at(_driver(b), (0.0, 0.0))
    assert d.step(np.array([5.0, 0.0])) == (None, "policy_stop")
    assert d.n_aborts == 1 and d.goal_active is False


def test_the_simulated_backend_action_is_passed_through():
    """In habitat the navigator hands back a real discrete action; on the robot
    there is nothing to execute and the placeholder stands in."""
    d = _at(_driver(FakeBackend(state="active", action="turn_left")), (0.0, 0.0))
    assert d.step(np.array([5.0, 0.0])) == ("turn_left", "moving")


def test_a_goal_cancelled_by_someone_else_is_re_posted_not_reported_failed():
    b = FakeBackend(state="canceled")
    d = _at(_driver(b), (0.0, 0.0))
    assert d.step(np.array([5.0, 0.0])) == (DRIVING, "moving")
    d.step(np.array([5.0, 0.0]))
    assert len(b.goals) == 2, "the pursuit resumes on the next tick"


def test_acting_without_a_pose_is_a_programming_error_not_a_silent_zero():
    with pytest.raises(RuntimeError, match="observe"):
        _driver().step(np.array([1.0, 0.0]))


# -------------------------------------------------------- the base has one owner


def test_consume_tick_reports_whether_the_driver_steered_and_clears():
    b = FakeBackend()
    d = _at(_driver(b), (0.0, 0.0))
    d.step(np.array([5.0, 0.0]))
    assert d.consume_tick() == (True, True)
    assert d.consume_tick() == (False, True), "the flag is per tick"
    _at(d, (0.0, 0.0))  # a new tick, and the FSM does not call step
    assert d.consume_tick() == (False, True)


def test_a_cancelled_goal_is_re_posted_after_the_fsm_took_the_base():
    """The env cancels when a discrete action wins the tick; the pursuit must
    resume rather than be dropped."""
    b = FakeBackend()
    d = _at(_driver(b), (0.0, 0.0))
    d.step(np.array([5.0, 0.0]))
    d.mark_cancelled()
    assert d.goal_active is False
    d.step(np.array([5.0, 0.0]))
    assert len(b.goals) == 2


def test_reset_cancels_whatever_was_outstanding():
    b = FakeBackend()
    d = _at(_driver(b), (0.0, 0.0))
    d.step(np.array([5.0, 0.0]))
    d.reset()
    assert b.cancels >= 1 and d.goal_active is False


def test_the_stop_radius_is_visible_where_the_fsm_reads_it():
    """`nav_agent._frontier_reach_m` and the climb read `stop_radius` off the
    mover; a driver without it silently reports every frontier unreached."""
    assert _driver(stop_radius=1.25).stop_radius == 1.25


def test_the_driver_satisfies_the_pointnav_call_shape():
    """It occupies PointNav's slot in NavAgent, so it has to answer the same
    calls -- including the action-only form and the ignored creep."""
    d = _at(_driver(), (0.0, 0.0))
    assert d(np.array([5.0, 0.0]), creep_below=1.0) == DRIVING
    assert d.step(np.array([5.0, 0.0]), creep_below=1.0).reason == "moving"


# ------------------------------------------------------- the simulated backend


class _FakeEnv:
    """habitat's follower, as `SimNav2Backend` sees it."""

    def __init__(self, action):
        self.action, self.asked = action, []

    def action_to_goal(self, goal_xy, floor_y=None):
        self.asked.append((np.asarray(goal_xy, float).copy(), floor_y))
        return self.action


def _sim(action, agent_xy, goal_xy, arrival_m=0.25):
    from osg.planning.nav2_backends import SimNav2Backend

    env = _FakeEnv(action)
    backend = SimNav2Backend(env, arrival_m=arrival_m)
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = float(agent_xy[0]), 0.88, float(agent_xy[1])
    backend.observe(make_frame(_K, T))
    backend.send_goal(np.asarray(goal_xy, float))
    return backend, env


def test_the_simulated_backend_is_active_while_the_follower_steers():
    backend, env = _sim("move_forward", (0.0, 0.0), (5.0, 0.0))
    status = backend.poll()
    assert status.state == "active" and status.action == "move_forward"
    assert status.distance_remaining == pytest.approx(5.0)
    assert env.asked[0][0].tolist() == [5.0, 0.0]


def test_the_simulated_backend_separates_arriving_from_giving_up():
    """habitat's follower returns None for both, and collapsing them is the
    mistake that made every greedy failure look like an arrival."""
    near, _ = _sim(None, (0.0, 0.0), (0.1, 0.0))
    far, _ = _sim(None, (0.0, 0.0), (5.0, 0.0))
    assert near.poll().state == "succeeded"
    assert far.poll().state == "aborted"


def test_the_simulated_backend_is_idle_until_it_is_given_a_goal():
    from osg.planning.nav2_backends import SimNav2Backend

    assert SimNav2Backend(_FakeEnv("move_forward")).poll().state == "idle"


def test_cancelling_the_simulated_backend_stops_it_asking_the_follower():
    backend, env = _sim("move_forward", (0.0, 0.0), (5.0, 0.0))
    backend.cancel()
    assert backend.poll().state == "idle" and env.asked == []


def test_the_simulated_backend_asks_about_the_agents_own_storey():
    """This mover is same-storey (see Nav2Driver): `None` tells habitat to use
    the agent's own floor, which is what a same-storey goal wants."""
    backend, env = _sim("move_forward", (0.0, 0.0), (5.0, 0.0))
    backend.poll()
    assert env.asked[-1][1] is None


# ------------------------------------------------- a goal that never resolves


def test_a_goal_nav2_never_finishes_is_eventually_treated_as_refused():
    """Nav2's recovery behaviours can spin on an unreachable goal indefinitely
    without ever reporting ABORTED. Waiting forever is a whole run spent on one
    frontier, so `ros2.nav_timeout_s` converts patience into a verdict."""
    b = FakeBackend(state="active")
    d = _at(_driver(b, timeout_steps=3), (0.0, 0.0))
    for _ in range(2):
        assert d.step(np.array([5.0, 0.0])).reason == "moving"
    step = d.step(np.array([5.0, 0.0]))
    assert step == (None, "policy_stop")
    assert d.n_timeouts == 1 and d.goal_active is False
    assert b.cancels >= 1, "the base must be released, not left driving"


def test_the_patience_restarts_when_the_goal_changes():
    b = FakeBackend(state="active")
    d = _at(_driver(b, timeout_steps=3), (0.0, 0.0))
    d.step(np.array([5.0, 0.0]))
    d.step(np.array([5.0, 0.0]))
    d.step(np.array([-5.0, 2.0]))  # a new goal: a fresh allowance
    assert d.step(np.array([-5.0, 2.0])).reason == "moving"
    assert d.n_timeouts == 0


def test_patience_can_be_switched_off():
    b = FakeBackend(state="active")
    d = _at(_driver(b, timeout_steps=0), (0.0, 0.0))
    for _ in range(50):
        assert d.step(np.array([5.0, 0.0])).reason == "moving"
    assert d.n_timeouts == 0


# ------------------------------------------------------- per-episode counters


def test_reset_clears_the_counters_because_the_mover_outlives_the_episode():
    """One mover is built per RUN (pipeline/components.py) and reused for every
    episode, while `pointnav_resets` is written into each episode's record. A
    counter left standing makes episode 3 report the run total."""
    b = FakeBackend(state="aborted")
    d = _at(_driver(b), (0.0, 0.0))
    d.step(np.array([5.0, 0.0]))
    assert (d.n_goals_sent, d.n_aborts) == (1, 1)
    d.reset()
    assert (d.n_goals_sent, d.n_aborts, d.n_resets, d.n_timeouts) == (0, 0, 0, 0)


def test_the_driver_records_what_it_last_returned():
    """`sim/ros2_env.py` compares the action it is handed against this to tell
    the mover's own action from one the FSM substituted afterwards."""
    d = _at(_driver(FakeBackend(state="active")), (0.0, 0.0))
    assert d.step(np.array([5.0, 0.0])).action == d.last_action == DRIVING
    d = _at(_driver(FakeBackend(state="succeeded")), (0.0, 0.0))
    d.step(np.array([5.0, 0.0]))
    assert d.last_action is None
