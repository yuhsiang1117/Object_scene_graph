"""The whole ROS-side contract, over a real socket, with no ROS installed.

Everything else in the ROS tests fakes one seam at a time. This one runs the
actual protocol -- `Listener`/`Client`, the array codec, `Transport`,
`Ros2Env`, `Nav2Driver` -- against `osg.ros2.loopback`, which is the fake robot
with its rclpy removed. What it cannot cover is rclpy itself; that is what
`fake_robot.py` and `scripts/ros2/check_pipeline.sh` are for.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from osg.core.config import OSGConfig
from osg.mapping.costmap import HEIGHT_AXIS, PLANE
from osg.planning.nav2_backends import RosNav2Backend
from osg.planning.nav2_driver import Nav2Driver
from osg.ros2 import frames
from osg.ros2.loopback import LoopbackRobot, serve
from osg.ros2.loopback import parse_args as robot_args
from osg.ros2.transport import Transport
from osg.sim.ros2_env import Ros2Env


@pytest.fixture
def robot_env():
    """A loopback robot on a thread, and an env connected to it."""
    from multiprocessing.connection import Listener

    # Port 0: let the OS pick, so parallel runs cannot collide.
    listener = Listener(("127.0.0.1", 0), authkey=b"test")
    args = robot_args(["--width", "32", "--height", "24", "--nav-speed", "10.0"])
    robot = LoopbackRobot(args)
    thread = threading.Thread(
        target=_serve_forever, args=(listener, robot), daemon=True)
    thread.start()

    cfg = OSGConfig()
    cfg.eval.mode = "ros2"
    cfg.agent.max_steps = 200
    cfg.ros2.rotate_deg = 0.0
    cfg.ros2.step_period_s = 0.0
    cfg.ros2.bridge_addr = f"{listener.address[0]}:{listener.address[1]}"
    cfg.ros2.authkey = "test"
    env = Ros2Env(cfg, transport=Transport(cfg.ros2, connect_timeout_s=5.0))
    try:
        yield env, robot
    finally:
        env.close()
        listener.close()


def _serve_forever(listener, robot):
    from osg.ros2.wire import serve_once

    try:
        while True:
            conn = listener.accept()
            try:
                while serve_once(conn, robot.handlers()):
                    pass
            finally:
                conn.close()
    except OSError:
        pass  # the listener closed: the test is over


def _driver(env):
    driver = Nav2Driver(RosNav2Backend(env.transport, env.cfg.ros2),
                        stop_radius=0.0, goal_resend_m=0.5)
    env.attach_driver(driver)
    return driver


def test_a_frame_survives_the_whole_path(robot_env):
    env, robot = robot_env
    frame = env.reset()
    assert frame.rgb.shape == (24, 32, 3) and frame.rgb.dtype == np.uint8
    assert frame.depth.dtype == np.float32
    assert (frame.depth > 0).any(), "the room should be visible"
    assert frames.looks_like_rotation(frame.T_wc)
    assert frame.camera_position[HEIGHT_AXIS] == pytest.approx(robot.camera_height)


def test_a_goal_reaches_the_navigator_in_ros_coordinates(robot_env):
    env, robot = robot_env
    frame = env.reset()
    driver = _driver(env)
    driver.observe(frame)
    goal = frame.camera_position[list(PLANE)] + np.array([1.0, -0.5])
    driver.step(goal)
    sent = env.transport.last_goal()
    assert (sent["x"], sent["y"]) == pytest.approx(frames.pipeline_xy_to_ros(goal))
    assert sent["frame_id"] == "map"


def test_driving_to_a_goal_puts_the_camera_where_the_pipeline_asked(robot_env):
    """The conversions have to be inverses of each other in practice, not only
    one at a time: a sign error anywhere in the loop drives to the mirror
    image of the intended point."""
    env, robot = robot_env
    frame = env.reset()
    driver = _driver(env)
    goal = frame.camera_position[list(PLANE)] + np.array([1.5, -1.0])
    for _ in range(60):
        driver.observe(frame)
        step = driver.step(goal)
        if step.reason == "arrived":
            break
        assert step.reason == "moving", step
        frame = env.step(step.action)
    else:
        pytest.fail("never arrived")
    reached = frame.camera_position[list(PLANE)]
    assert np.linalg.norm(reached - goal) < 0.35, f"asked {goal}, reached {reached}"


def test_a_refused_goal_comes_back_as_the_blocked_reason(robot_env):
    """What Nav2 does when it cannot plan, and what the FSM must hear: the
    reason `pointnav_stop_means_blocked` reads to retire a frontier."""
    env, robot = robot_env
    frame = env.reset()
    driver = _driver(env)
    driver.observe(frame)
    robot.args.abort_goals = True  # the real fake's --abort-goals

    step = driver.step(frame.camera_position[list(PLANE)] + np.array([2.0, 0.0]))
    assert step == (None, "policy_stop")
    assert driver.n_aborts == 1 and driver.goal_active is False


def test_the_discrete_actions_move_the_base_the_way_the_controller_expects(robot_env):
    """`turn_left` DECREASES `agent_heading` (planning/controller.py:82).
    Getting the sign wrong mirrors every turn while looking plausible."""
    from osg.planning.controller import agent_heading

    env, _ = robot_env
    frame = env.reset()
    before = agent_heading(frame.T_wc)
    frame = env.step("turn_left")
    delta = np.arctan2(np.sin(agent_heading(frame.T_wc) - before),
                       np.cos(agent_heading(frame.T_wc) - before))
    assert delta < 0, f"turn_left moved the heading by {np.degrees(delta):+.1f} deg"

    here = frame.camera_position[list(PLANE)].copy()
    frame = env.step("move_forward")
    moved = float(np.linalg.norm(frame.camera_position[list(PLANE)] - here))
    assert moved == pytest.approx(env.cfg.agent.forward_m, abs=0.02)


def test_the_floor_switch_travels_the_whole_way(robot_env):
    env, robot = robot_env
    frame = env.reset()
    seen = []
    env.on_floor_switch = seen.append
    ground = frame.camera_position[HEIGHT_AXIS]

    with robot._lock:
        robot._floor = 1  # what `/osg/floor` sets on the real bridge
    frame = env.step("turn_left")

    assert seen == [1] and env.floor_key == 1
    assert frame.camera_position[HEIGHT_AXIS] == pytest.approx(
        ground + env.cfg.floor.virtual_storey_m)


# ------------------------------------------- the two fakes must not drift apart


def test_both_fake_robots_share_one_camera_convention():
    """The loopback exists to localise a fault to the ROS half. It can only do
    that while the two agree on the world they simulate -- a private copy of
    the camera mount here would let the ROS-free check pass on a convention the
    bridge does not use."""
    from osg.ros2 import fake_robot, loopback

    assert loopback.camera_matrix is fake_robot.camera_matrix
    assert loopback.render_depth is fake_robot.render_depth
    assert loopback.ROOM_HALF_M is fake_robot.ROOM_HALF_M


def test_the_fake_camera_agrees_with_the_repos_heading_convention():
    """`turn_left` DECREASES `agent_heading` (planning/controller.py:82), and
    ROS yaw increases counter-clockwise. If the mount got those backwards every
    turn would mirror while still looking plausible."""
    from osg.planning.controller import agent_heading
    from osg.ros2.fake_robot import camera_matrix

    ahead = frames.ros_pose_to_pipeline(camera_matrix((0.0, 0.0, 0.0), 1.3))
    assert agent_heading(ahead) == pytest.approx(0.0)
    assert np.allclose(ahead[:3, 2], [1.0, 0.0, 0.0]), "camera z is forward"

    left = frames.ros_pose_to_pipeline(camera_matrix((0.0, 0.0, np.pi / 2), 1.3))
    assert np.degrees(agent_heading(left)) == pytest.approx(-90.0)
