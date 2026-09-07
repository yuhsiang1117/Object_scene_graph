"""ASCENT's carrot-waypoint stair traversal (ascent_policy.py:1075-1112).

The bearing is derived from a depth image and consumed by a mover, so a sign
error produces an agent that steers confidently into the wall opposite the
staircase while every intermediate value still looks reasonable. These tests pin
the convention against the same fixture `test_controller.py` uses.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.agent.nav_agent import State
from osg.core.types import CameraIntrinsics, FrameData
from osg.planning.controller import agent_heading

from .conftest import make_camera
from .test_nav_agent import make_agent, make_cfg

INTR = CameraIntrinsics.from_hfov(79.0, 640, 480)


def _frame_with_far_column(col, depth_near=1.0, depth_far=5.0):
    """A depth image that is uniformly near except for one far column."""
    d = np.full((480, 640), depth_near, dtype=np.float32)
    d[:, col] = depth_far
    T = make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0])  # facing world +x
    return FrameData(frame_id=0, rgb=np.zeros((480, 640, 3), np.uint8),
                     depth=d, T_wc=T, intrinsics=INTR)


def _agent(**over):
    return make_agent(make_cfg(climb_carrot=True, climb_carrot_m=0.8, **over))


# ------------------------------------------------------------------ bearing


def test_far_pixel_dead_ahead_gives_a_carrot_straight_ahead():
    a = _agent()
    f = _frame_with_far_column(320)  # image centre
    goal = a._carrot_goal(f, np.zeros(2))
    assert goal == pytest.approx([0.8, 0.0], abs=1e-6)


def test_far_pixel_on_the_right_steers_right():
    """Facing world +x, 'right' is +z (plane axis 1) -- the same fact
    test_controller.py pins by asserting a +z waypoint needs TURN_RIGHT."""
    a = _agent()
    goal = a._carrot_goal(_frame_with_far_column(639), np.zeros(2))
    assert goal[1] > 0.0, "a far pixel on the right must place the carrot to the right"
    assert np.linalg.norm(goal) == pytest.approx(0.8, abs=1e-6)


def test_far_pixel_on_the_left_steers_left():
    a = _agent()
    goal = a._carrot_goal(_frame_with_far_column(0), np.zeros(2))
    assert goal[1] < 0.0


def test_bearing_is_bounded_by_half_the_field_of_view():
    a = _agent()
    for col in (0, 639):
        goal = a._carrot_goal(_frame_with_far_column(col), np.zeros(2))
        bearing = abs(np.arctan2(goal[1], goal[0]))
        assert bearing <= np.radians(79.0) / 2 + 1e-6


def test_carrot_is_placed_relative_to_the_agent_not_the_origin():
    a = _agent()
    here = np.array([3.0, -2.0])
    goal = a._carrot_goal(_frame_with_far_column(320), here)
    assert goal == pytest.approx([3.8, -2.0], abs=1e-6)


def test_bearing_follows_the_agents_heading():
    """Same image, agent rotated 90 deg: the carrot must rotate with it."""
    a = _agent()
    d = np.full((480, 640), 1.0, dtype=np.float32)
    d[:, 320] = 5.0
    T = make_camera([0.0, 0.88, 0.0], [0.0, 0.88, 1.0])  # facing world +z
    f = FrameData(frame_id=0, rgb=np.zeros((480, 640, 3), np.uint8),
                  depth=d, T_wc=T, intrinsics=INTR)
    assert agent_heading(T) == pytest.approx(np.pi / 2)
    assert a._carrot_goal(f, np.zeros(2)) == pytest.approx([0.0, 0.8], abs=1e-6)


def test_degenerate_depth_yields_no_carrot():
    a = _agent()
    T = make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0])
    empty = FrameData(frame_id=0, rgb=np.zeros((1, 1, 3), np.uint8),
                      depth=np.zeros((0, 0), np.float32), T_wc=T, intrinsics=INTR)
    assert a._carrot_goal(empty, np.zeros(2)) is None


# ------------------------------------------------------------------ ratchet


def test_ratchet_keeps_the_carrot_closest_to_the_stair_end():
    a = _agent()
    a._climb_goal_xy = np.array([5.0, 0.0])  # the stair end
    a._carrot_xy = None

    first = a._update_carrot(_frame_with_far_column(320), np.zeros(2))  # ahead, toward the end
    assert first == pytest.approx([0.8, 0.0], abs=1e-6)

    # A bearing swinging hard right is FURTHER from the end -- keep the old one.
    kept = a._update_carrot(_frame_with_far_column(639), np.zeros(2))
    assert kept == pytest.approx(first, abs=1e-6)


def test_ratchet_accepts_a_carrot_that_closes_on_the_end():
    a = _agent()
    a._climb_goal_xy = np.array([5.0, 0.0])
    a._carrot_xy = np.array([0.0, 0.8])  # off to the side, far from the end
    better = a._update_carrot(_frame_with_far_column(320), np.zeros(2))
    assert better == pytest.approx([0.8, 0.0], abs=1e-6)


def test_disable_end_releases_the_ratchet():
    """Once the stall detector decides the recorded end is unreachable, the
    fresh bearing must win every time (ascent_policy.py:1099-1101)."""
    a = _agent()
    a._climb_goal_xy = np.array([5.0, 0.0])
    a._carrot_xy = np.array([0.8, 0.0])
    a._carrot_disable_end = True
    fresh = a._update_carrot(_frame_with_far_column(639), np.zeros(2))
    assert fresh[1] > 0.0


def test_ratchet_released_when_already_at_the_end():
    a = _agent()
    a._climb_goal_xy = np.array([0.2, 0.0])  # within 0.5 m of the agent
    a._carrot_xy = np.array([0.2, 0.0])
    fresh = a._update_carrot(_frame_with_far_column(639), np.zeros(2))
    assert fresh[1] > 0.0


# -------------------------------------------------------------- stall rule


def test_stall_counts_only_while_the_distance_is_not_changing():
    a = _agent()
    a._climb_centroid_xy = np.array([10.0, 0.0])
    # walking in: the distance changes every step, so nothing accumulates
    for i in range(40):
        assert a._carrot_stalled(np.array([float(i) * 0.25, 0.0])) is False
    assert a._carrot_disable_end is False


def test_stall_releases_the_ratchet_then_ends_the_climb():
    """Thresholds are 15 and 30 stalled steps. The first call only establishes
    the reference distance and counts nothing -- ASCENT's does the same, since
    its `_last_frontier_distance` starts at 0 and the opening comparison always
    exceeds 0.2 m -- so N calls leave a count of N-1."""
    a = _agent()
    a._climb_centroid_xy = np.array([10.0, 0.0])
    here = np.array([0.0, 0.0])
    for _ in range(16):  # count = 15, not yet past the threshold
        assert a._carrot_stalled(here) is False
    assert a._carrot_disable_end is False
    a._carrot_stalled(here)  # count = 16 > 15
    assert a._carrot_disable_end is True

    for _ in range(14):  # up to count = 30
        assert a._carrot_stalled(here) is False
    assert a._carrot_stalled(here) is True, "count 31 > 30 must end the climb"


# ---------------------------------------------------------- never gives up


def test_a_network_stop_becomes_a_forward_step():
    """ASCENT overrides STOP mid-climb (ascent_policy.py:1136-1139): on stairs a
    STOP usually means 'the treads fill my view', which is the one moment the
    agent must not stop."""
    from osg.planning.pointnav_driver import NavStep

    class _Driver:
        stop_radius = 0.9

        def observe(self, frame):
            pass

        def reset(self):
            pass

        def step(self, goal_xy, **kw):
            return NavStep(None, "policy_stop")

    a = make_agent(make_cfg(navigation="pointnav", climb_carrot=True), pointnav=_Driver())
    a._climb_goal_xy = np.array([5.0, 0.0])
    assert a._carrot_action(_frame_with_far_column(320), np.zeros(2)) == "move_forward"
    assert a.stats["climb_forced_forward"] == 1


def test_carrot_off_leaves_the_overshoot_goal_in_place():
    a = make_agent(make_cfg(climb_carrot=False))
    assert a._climb_carrot is False
