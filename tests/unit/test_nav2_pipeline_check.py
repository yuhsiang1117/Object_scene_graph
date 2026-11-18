"""End to end, without ROS: the goal the FSM picks is the goal Nav2 receives.

This is the property the whole layer exists to deliver, and the one with the
most places to go quietly wrong -- the frontier's metric goal, the mover's
hand-off, two coordinate conversions and a topic frame. Each is tested
separately elsewhere; here they are composed, because the composition is what
drives a real robot into a real wall.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.mapping.frontier import Frontier
from osg.planning.nav2_driver import Nav2Driver
from osg.ros2 import frames

from .test_nav2_driver import FakeBackend
from .test_nav_agent import _frame, make_agent, make_cfg


class RecordingTransport:
    """Just enough bridge to watch a goal arrive."""

    def __init__(self, state="active"):
        self.state = state
        self.goals = []

    def send_goal(self, x, y, yaw, frame_id):
        self.goals.append({"x": x, "y": y, "yaw": yaw, "frame_id": frame_id})
        return len(self.goals)

    def nav_status(self):
        return {"state": self.state, "goal_id": len(self.goals),
                "distance_remaining": 1.0}

    def cancel(self):
        pass


def _agent_with_nav2(backend=None, **cfg_over):
    backend = backend or FakeBackend()
    cfg = make_cfg(navigation="nav2", **cfg_over)
    agent = make_agent(cfg, pointnav=Nav2Driver(
        backend, stop_radius=cfg.agent.pointnav_stop_radius,
        goal_resend_m=cfg.ros2.goal_resend_m))
    return agent, backend


def _pursue(agent, centroid):
    from osg.agent.nav_agent import State

    agent.state = State.GOTO_FRONTIER
    agent._current_frontier = Frontier(
        id=1, centroid_xy=np.asarray(centroid, float),
        cells=np.zeros((0, 2), dtype=int), size=20,
    )


# ------------------------------------------ the FSM reaches the mover at all


def test_the_frontier_goal_the_fsm_chose_is_what_gets_posted():
    from osg.exploration.selector import frontier_goal_xy

    agent, backend = _agent_with_nav2()
    frame = _frame([0.0, 0.0])
    agent.pointnav.observe(frame)
    _pursue(agent, [9.0, 0.0])
    action = agent._follow_path(frame)

    expected = frontier_goal_xy(agent.exploration.current_frontier, agent.costmap,
                                agent.exploration.goal_prefer_free)
    assert len(backend.goals) == 1
    assert np.allclose(backend.goals[0][0], expected)
    assert action == "move_forward", "the loop still gets an action per tick"


def test_an_approach_goal_reaches_the_mover_through_the_other_seam():
    """`_follow_to` is the APPROACH and CLIMB path, and it applies the tighter
    arrival radius rather than the frontier one."""
    agent, backend = _agent_with_nav2()
    frame = _frame([0.0, 0.0])
    agent.pointnav.observe(frame)
    agent._follow_to(frame, np.array([4.0, 1.0]))
    assert len(backend.goals) == 1 and np.allclose(backend.goals[0][0], [4.0, 1.0])
    assert agent.stats.get("pointnav_moving", 0) == 1, "the reason is recorded"


def test_nav2_giving_up_retires_the_frontier_instead_of_pressing_into_it():
    """The FSM needed no new branch for this mover: an aborted goal arrives as
    `policy_stop`, which `pointnav_stop_means_blocked` already handles."""
    agent, _ = _agent_with_nav2(FakeBackend(state="aborted"),
                                pointnav_stop_means_blocked=True)
    frame = _frame([0.0, 0.0])
    agent.pointnav.observe(frame)
    _pursue(agent, [9.0, 0.0])
    assert agent._follow_path(frame) is None
    assert agent.stats.get("frontier_stub_block", 0) == 1


def test_arriving_ends_the_pursuit():
    agent, backend = _agent_with_nav2(FakeBackend(state="succeeded"))
    frame = _frame([0.0, 0.0])
    agent.pointnav.observe(frame)
    _pursue(agent, [9.0, 0.0])
    assert agent._follow_path(frame) is None
    assert agent.pointnav.goal_active is False


def test_a_whole_run_of_steps_posts_one_goal_not_one_per_step():
    """Re-posting an unchanged goal restarts Nav2's global planner; over a
    200-step leg that is 200 replans."""
    agent, backend = _agent_with_nav2()
    _pursue(agent, [9.0, 0.0])
    for i in range(10):
        frame = _frame([0.05 * i, 0.0], frame_id=i)
        agent.pointnav.observe(frame)
        agent._follow_path(frame)
    assert len(backend.goals) == 1


# ------------------------------------------ and arrives in ROS coordinates


def test_the_posted_goal_is_the_pipeline_goal_in_ros_coordinates():
    from osg.core.config import OSGConfig
    from osg.planning.nav2_backends import RosNav2Backend

    transport = RecordingTransport()
    cfg = OSGConfig()
    agent, _ = _agent_with_nav2()
    backend = RosNav2Backend(transport, cfg.ros2)
    driver = Nav2Driver(backend, stop_radius=0.9,
                        goal_resend_m=cfg.ros2.goal_resend_m)
    agent.pointnav = driver

    frame = _frame([1.0, 2.0])
    agent.pointnav.observe(frame)
    _pursue(agent, [9.0, -3.0])
    agent._follow_path(frame)

    from osg.exploration.selector import frontier_goal_xy
    goal = frontier_goal_xy(agent.exploration.current_frontier, agent.costmap,
                            agent.exploration.goal_prefer_free)
    x, y = frames.pipeline_xy_to_ros(goal)
    assert len(transport.goals) == 1
    sent = transport.goals[0]
    assert (sent["x"], sent["y"]) == pytest.approx((x, y))
    assert sent["frame_id"] == cfg.ros2.goal_frame == "map"
    # ... and driving there puts the camera back on the pipeline goal.
    assert np.allclose(frames.ros_xy_to_pipeline(sent["x"], sent["y"]), goal)
    assert backend.last_sent_ros["goal_id"] == 1


def test_the_goal_pose_faces_the_way_the_robot_travels():
    """Nav2 enforces the goal orientation, so it has to be the useful one: the
    camera arrives pointed at whatever the goal was chosen for."""
    from osg.core.config import OSGConfig
    from osg.planning.nav2_backends import RosNav2Backend

    transport = RecordingTransport()
    backend = RosNav2Backend(transport, OSGConfig().ros2)
    driver = Nav2Driver(backend, stop_radius=0.5)
    frame = _frame([0.0, 0.0])
    driver.observe(frame)
    driver.step(np.array([5.0, 0.0]))  # straight ahead in pipeline +x == ROS +x
    assert transport.goals[0]["yaw"] == pytest.approx(0.0)
