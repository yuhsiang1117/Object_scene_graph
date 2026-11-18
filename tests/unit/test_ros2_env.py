"""The robot standing where habitat normally stands.

Everything here runs against a `FakeTransport` -- no ROS, no robot -- because
the two things `Ros2Env` actually decides are testable without either: who owns
the base on a given tick, and what height the pose is reported at.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from osg.core.config import OSGConfig
from osg.sim.ros2_env import Ros2Env


class FakeTransport:
    """The bridge's surface, scripted."""

    def __init__(self, camera_z=1.3, xy=(0.0, 0.0)):
        self.calls = []
        self.floor_queue = []
        self.cancels = 0
        self.pose = [float(xy[0]), float(xy[1]), float(camera_z)]
        self.stamp = 1000.0
        self.seq = 0

    def ping(self):
        self.calls.append(("ping",))
        return {"node": "fake"}

    def get_frame(self, after_seq=-1, timeout_s=5.0):
        self.calls.append(("get_frame", after_seq))
        self.stamp += 0.1
        self.seq += 1
        return {
            "seq": self.seq, "stamp": self.stamp,
            "rgb": np.zeros((4, 6, 3), np.uint8),
            "depth": np.full((4, 6), 1500, np.uint16),
            "depth_encoding": "16UC1",
            "K": np.array([3.0, 0.0, 3.0, 0.0, 3.0, 2.0, 0.0, 0.0, 1.0]),
            "width": 6, "height": 4,
            "T_map_cam": self._T(),
            "frame_id": "camera_color_optical_frame",
        }

    def _T(self):
        T = np.eye(4)
        T[:3, 3] = self.pose
        return T

    def send_goal(self, x, y, yaw, frame_id):
        self.calls.append(("send_goal", x, y, yaw, frame_id))
        return 1

    def nav_status(self):
        return {"state": "active", "goal_id": 1, "distance_remaining": 2.0}

    def cancel(self):
        self.cancels += 1
        self.calls.append(("cancel",))

    def execute(self, action, forward_m, turn_deg):
        self.calls.append(("execute", action, forward_m, turn_deg))
        return {"achieved": forward_m, "timed_out": False}

    def look(self, tilt_delta_deg):
        self.calls.append(("look", tilt_delta_deg))
        return {"tilt_deg": tilt_delta_deg}

    def pop_floor_switch(self):
        return self.floor_queue.pop(0) if self.floor_queue else None

    def last_goal(self):
        return None

    def close(self):
        self.calls.append(("close",))


class FakeDriver:
    """`Nav2Driver`'s ownership surface, scripted."""

    def __init__(self, stepped=False, goal_active=False, last_action="move_forward"):
        self.stepped, self.goal_active = stepped, goal_active
        # What the mover returned this tick. The env compares the action it is
        # handed against this to tell the mover's own action from one the FSM
        # substituted afterwards.
        self.last_action = last_action
        self.cancelled = 0

    def consume_tick(self):
        stepped, self.stepped = self.stepped, False
        return stepped, self.goal_active

    def mark_cancelled(self):
        self.cancelled += 1
        self.goal_active = False


def _env(transport=None, driver=None, **ros_over):
    cfg = OSGConfig()
    cfg.eval.mode = "ros2"
    cfg.agent.max_steps = 5
    cfg.ros2.rotate_deg = 0.0  # rotation has its own tests
    cfg.ros2.step_period_s = 0.0
    for k, v in ros_over.items():
        setattr(cfg.ros2, k, v)
    env = Ros2Env(cfg, transport=transport or FakeTransport())
    if driver is not None:
        env.attach_driver(driver)
    return env


def _kinds(transport):
    return [c[0] for c in transport.calls]


# ------------------------------------------------------------------- frames


def test_a_bridge_payload_becomes_the_frame_the_pipeline_speaks():
    t = FakeTransport(camera_z=1.3)
    frame = _env(t).reset()
    assert frame.rgb.shape == (4, 6, 3) and frame.rgb.dtype == np.uint8
    assert frame.depth.dtype == np.float32
    assert np.allclose(frame.depth, 1.5), "millimetres became metres"
    assert (frame.intrinsics.fx, frame.intrinsics.cx) == (3.0, 3.0)
    assert frame.timestamp > 0.0


def test_the_pose_arrives_in_the_pipelines_world():
    from osg.mapping.costmap import HEIGHT_AXIS, PLANE

    t = FakeTransport(camera_z=1.3, xy=(2.0, -1.0))
    frame = _env(t).reset()
    assert frame.camera_position[HEIGHT_AXIS] == pytest.approx(1.3)
    assert np.allclose(frame.camera_position[list(PLANE)], [2.0, 1.0])


def test_frames_are_numbered_so_the_object_layer_can_order_them():
    env = _env()
    assert [env.reset().frame_id, env.step("turn_left").frame_id] == [1, 2]


def test_the_camera_rotation_is_applied_on_the_way_in():
    t = FakeTransport()
    frame = _env(t, rotate_deg=90.0).reset()
    assert frame.rgb.shape[:2] == (6, 4), "the portrait mount is undone here"
    assert (frame.intrinsics.width, frame.intrinsics.height) == (4, 6)


# ---------------------------------------------------- who owns the base


def test_a_discrete_action_is_metered_on_the_base():
    t = FakeTransport()
    env = _env(t)
    env.reset()
    env.step("move_forward")
    assert ("execute", "move_forward", env.cfg.agent.forward_m,
            env.cfg.agent.turn_deg) in t.calls


def test_a_wait_touches_neither_the_wheels_nor_the_head():
    """WAIT_ACTION: the robot is standing at the stairs for the carry. One
    tick's pause and a fresh frame, no command to the base."""
    from osg.agent.state import WAIT_ACTION

    t = FakeTransport()
    env = _env(t)
    env.reset()
    before = _kinds(t)
    frame = env.step(WAIT_ACTION)
    assert frame is not None
    assert "execute" not in _kinds(t)[len(before):] and "look" not in _kinds(t)[len(before):]


def test_the_hold_ends_on_the_operators_switch_and_spends_no_step():
    """ros2.wait_for_switch: frames flow, the base is untouched, the switch is
    applied and recorded, and the episode's step budget is intact."""
    t = FakeTransport(camera_z=1.3)
    env = _env(t)
    heard = []
    env.on_floor_switch = heard.append
    env.reset()
    t.floor_queue.append(1)
    seen = []
    frame = env.wait_for_floor_switch(on_frame=seen.append)
    assert frame is not None and seen, "frames were delivered during the hold"
    assert env.floor_key == 1 and env.floor_switches == [(0, 1)]
    assert heard == [1], "the agent's floor policy was told"
    assert "execute" not in _kinds(t) and "look" not in _kinds(t)
    assert not env.episode_over


def test_a_look_goes_to_the_head_not_the_wheels():
    t = FakeTransport()
    env = _env(t, look_step_deg=30.0)
    env.reset()
    env.step("look_down")
    env.step("look_up")
    assert ("look", -30.0) in t.calls and ("look", 30.0) in t.calls
    assert "execute" not in _kinds(t)


def test_while_nav2_drives_the_placeholder_action_is_not_executed():
    """`Nav2Driver` returns "move_forward" to satisfy the control loop while
    the navigator steers. Executing it would fight Nav2 for the wheels."""
    t = FakeTransport()
    env = _env(t, driver=FakeDriver(stepped=True, goal_active=True))
    env.reset()
    env.step("move_forward")
    assert "execute" not in _kinds(t)
    assert "get_frame" in _kinds(t)


def test_the_fsm_taking_the_base_cancels_the_outstanding_goal_first():
    """The opening scan and the escape window issue actions without consulting
    any mover; a goal still outstanding would drive against them."""
    t = FakeTransport()
    driver = FakeDriver(stepped=False, goal_active=True)
    env = _env(t, driver=driver)
    env.reset()
    env.step("turn_left")
    order = _kinds(t)
    assert order.index("cancel") < order.index("execute")
    assert driver.cancelled == 1


def test_a_mover_that_has_arrived_does_not_cancel_anything():
    t = FakeTransport()
    env = _env(t, driver=FakeDriver(stepped=True, goal_active=False))
    env.reset()
    before = t.cancels
    env.step("move_forward")
    assert t.cancels == before and ("execute", "move_forward",
                                    env.cfg.agent.forward_m,
                                    env.cfg.agent.turn_deg) in t.calls


def test_stop_ends_the_run_and_releases_the_base():
    t = FakeTransport()
    env = _env(t, driver=FakeDriver(goal_active=True))
    env.reset()
    env.step("stop")
    assert env.episode_over is True
    assert t.cancels >= 1


def test_a_fresh_frame_is_demanded_after_a_command():
    """Acting on the frame that was current before the base moved is the
    failure that looks like a bad planner.

    Freshness is a sequence number, not a timestamp: image stamps are ROS time
    and the pipeline's clock is not, so under `use_sim_time` a wall-clock
    threshold could never be satisfied.
    """
    t = FakeTransport()
    env = _env(t)
    env.reset()
    env.step("move_forward")
    asked = [c[1] for c in t.calls if c[0] == "get_frame"]
    assert asked[-1] == 1, "must wait for a frame newer than the one acted on"


def test_an_action_the_fsm_substituted_takes_the_base_from_nav2():
    """`NavAgent.act` rewrites the action AFTER the mover ran -- the escape
    window turns a long run of forwards into a turn. That override is an FSM
    action and must reach the wheels, not be mistaken for the placeholder."""
    t = FakeTransport()
    driver = FakeDriver(stepped=True, goal_active=True, last_action="move_forward")
    env = _env(t, driver=driver)
    env.reset()
    env.step("turn_right")  # what ActionHistoryEscape substitutes
    order = _kinds(t)
    assert ("execute", "turn_right", env.cfg.agent.forward_m,
            env.cfg.agent.turn_deg) in t.calls
    assert order.index("cancel") < order.index("execute")


# ------------------------------------------------------------ the floor switch


def test_the_switch_lifts_the_pose_so_two_storeys_are_two_heights():
    """Nav2's map is 2D: both floors report the same z. Without the lift the
    two storeys share one costmap band and one `floor_of_height` answer."""
    from osg.mapping.costmap import HEIGHT_AXIS

    t = FakeTransport(camera_z=1.3)
    env = _env(t, driver=FakeDriver())
    ground = env.reset()
    t.floor_queue.append(1)
    upstairs = env.step("turn_left")
    assert ground.camera_position[HEIGHT_AXIS] == pytest.approx(1.3)
    assert upstairs.camera_position[HEIGHT_AXIS] == pytest.approx(
        1.3 + env.cfg.floor.virtual_storey_m)


def test_the_switch_reaches_the_agent_and_is_recorded():
    t = FakeTransport()
    env = _env(t, driver=FakeDriver())
    seen = []
    env.on_floor_switch = seen.append
    env.reset()
    t.floor_queue.append(1)
    env.step("turn_left")
    assert seen == [1] and env.floor_key == 1
    assert env.floor_switches == [(1, 1)]
    assert env.episode_metadata()["floor_switches"] == [(1, 1)]


def test_a_switch_drops_the_goal_chosen_on_the_floor_left_behind():
    t = FakeTransport()
    driver = FakeDriver(stepped=True, goal_active=True)
    env = _env(t, driver=driver)
    env.reset()
    t.floor_queue.append(1)
    env.step("move_forward")
    assert driver.cancelled == 1


def test_a_repeated_switch_to_the_same_floor_is_ignored():
    t = FakeTransport()
    env = _env(t, driver=FakeDriver())
    seen = []
    env.on_floor_switch = seen.append
    env.reset()
    t.floor_queue.extend([1, 1])
    env.step("turn_left")
    env.step("turn_left")
    assert seen == [1]


def test_reset_drains_a_switch_left_over_from_a_previous_run():
    """Run 2 must not begin by switching to the floor run 1 ended on."""
    t = FakeTransport()
    t.floor_queue.append(1)
    env = _env(t, driver=FakeDriver())
    seen = []
    env.on_floor_switch = seen.append
    env.reset()
    env.step("turn_left")
    assert seen == [] and env.floor_key == 0


# ------------------------------------------------------- the episode contract


def test_a_second_run_starts_on_the_ground_floor():
    """`reset` clears the storey as well as the log: a second run on the same
    env must not inherit the floor the previous one ended on."""
    t = FakeTransport()
    env = _env(t, driver=FakeDriver())
    env.reset()
    t.floor_queue.append(1)
    env.step("turn_left")
    assert env.floor_key == 1
    env.reset()
    assert env.floor_key == 0 and env.floor_switches == []


def test_the_run_ends_at_the_step_budget():
    env = _env()
    env.reset()
    for _ in range(int(env.cfg.agent.max_steps)):
        assert env.episode_over is False
        env.step("turn_left")
    assert env.episode_over is True


def test_the_record_can_be_written_without_a_habitat_episode():
    env = _env()
    env.reset()
    episode = env.current_episode
    assert episode.scene_id == env.cfg.ros2.map_tag
    assert env.target_category() == env.cfg.ros2.target
    assert episode.object_category == env.cfg.ros2.target


def test_nothing_is_scored_because_there_is_no_ground_truth():
    """A 0.0 success rate reported beside measured ones would be a fabricated
    number in summary.json."""
    env = _env()
    env.reset()
    metrics = env.metrics()
    # NaN in memory, because `eval/record.py` calls float() on all three; the
    # runner writes them out as null (scripts/run_robot.py:_json_safe), since a
    # bare NaN token is unreadable by any JSON parser outside Python.
    assert np.isnan(metrics["success"]) and np.isnan(metrics["spl"])
    assert env.attempt_scored(None, env.cfg) is False


def test_the_unmeasured_metrics_survive_the_trip_to_disk_as_null():
    from importlib.machinery import SourceFileLoader

    root = Path(__file__).resolve().parents[2]
    run_robot = SourceFileLoader(
        "run_robot", str(root / "scripts" / "run_robot.py")).load_module()

    env = _env()
    env.reset()
    text = json.dumps(run_robot._json_safe({"metrics": env.metrics(), "ok": [1.5]}))
    assert "NaN" not in text
    assert json.loads(text)["metrics"]["success"] is None


def test_closing_releases_the_base_and_the_socket():
    t = FakeTransport()
    env = _env(t)
    env.reset()
    env.close()
    assert "cancel" in _kinds(t) and "close" in _kinds(t)


def test_a_frame_gap_is_waited_out_then_fatal(capsys):
    """Carrying the robot to the other storey and relaunching slam_toolbox
    takes `map -> camera` away for minutes. A run waits through that, says
    so, and only fails once `frame_patience_s` is spent."""
    from osg.ros2.transport import BridgeUnavailable

    class GappyTransport(FakeTransport):
        def __init__(self, misses):
            super().__init__()
            self.misses = misses

        def get_frame(self, after_seq=-1, timeout_s=5.0):
            if self.misses > 0:
                self.misses -= 1
                self.calls.append(("get_frame", after_seq))
                raise BridgeUnavailable("no camera frame within 10.5s")
            return super().get_frame(after_seq, timeout_s)

    t = GappyTransport(misses=3)
    env = _env(t)
    env.cfg.ros2.frame_timeout_s = 10.0
    env.cfg.ros2.step_period_s = 0.5
    env.cfg.ros2.frame_patience_s = 60.0     # 3 misses = 31.5 s of gap: waited out
    env.reset()
    out = capsys.readouterr().out
    assert out.count("no camera frame for") == 3 and "frames are back" in out

    t = GappyTransport(misses=3)
    env = _env(t)
    env.cfg.ros2.frame_timeout_s = 10.0
    env.cfg.ros2.step_period_s = 0.5
    env.cfg.ros2.frame_patience_s = 20.0     # spent after the second miss
    with pytest.raises(BridgeUnavailable):
        env.reset()


def test_an_operator_waypoint_carries_its_own_heading_once():
    """RosNav2Backend.set_goal_yaw: the posed goal is sent facing the operator's
    yaw; any other goal, and the same goal sent again, faces the travel
    direction as before."""
    import math

    from osg.core.config import OSGConfig
    from osg.planning.nav2_backends import RosNav2Backend
    from osg.ros2.frames import ros_xy_to_pipeline

    t = FakeTransport()
    b = RosNav2Backend(t, OSGConfig().ros2)
    b._agent_xy = np.array([0.0, 0.0])
    goal = ros_xy_to_pipeline(-3.7773, 6.2081)
    b.set_goal_yaw(goal, math.radians(80.3))
    b.send_goal(goal)
    _, x, y, yaw, _ = [c for c in t.calls if c[0] == "send_goal"][-1]
    assert (round(x, 4), round(y, 4)) == (-3.7773, 6.2081)
    assert yaw == pytest.approx(math.radians(80.3))
    b.send_goal(goal)   # the heading was for one send
    _, _, _, yaw2, _ = [c for c in t.calls if c[0] == "send_goal"][-1]
    assert yaw2 == pytest.approx(math.atan2(6.2081, -3.7773))
    other = ros_xy_to_pipeline(1.0, 1.0)
    b.set_goal_yaw(goal, 0.0)
    b.send_goal(other)  # a different goal: not posed
    _, _, _, yaw3, _ = [c for c in t.calls if c[0] == "send_goal"][-1]
    assert yaw3 == pytest.approx(math.atan2(1.0, 1.0))
