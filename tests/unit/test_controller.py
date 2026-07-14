from __future__ import annotations

import numpy as np

from osg.planning.controller import FORWARD, TURN_LEFT, TURN_RIGHT, WaypointController, agent_heading

from .conftest import make_camera


def test_forward_when_aligned():
    c = WaypointController()
    T = make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0])  # facing +x (plane axis 0)
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert c.act(T, path) == FORWARD


def test_turns_toward_waypoint():
    c = WaypointController()
    T = make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0])  # facing +x
    path_left = np.array([[0.0, 0.0], [0.0, 1.0]])  # target 90 deg left (plane axis 1)
    action = c.act(T, path_left)
    assert action in (TURN_LEFT, TURN_RIGHT)
    # Consistency: opposite side gives opposite turn
    path_right = np.array([[0.0, 0.0], [0.0, -1.0]])
    assert {c.act(T, path_left), c.act(T, path_right)} == {TURN_LEFT, TURN_RIGHT}


def test_arrival_returns_none():
    c = WaypointController()
    T = make_camera([2.0, 0.88, 0.0], [3.0, 0.88, 0.0])
    path = np.array([[1.9, 0.05]])
    assert c.act(T, path) is None


def test_stuck_detection_marks_cell():
    from osg.mapping.costmap import FREE, OCCUPIED, Costmap2D

    cm = Costmap2D(resolution=0.1, size_m=5.0)
    cm.grid[:, :] = FREE
    c = WaypointController(stuck_after=2)
    T = make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0])
    c.observe_progress(T, None, cm)
    # Two forward actions with no movement -> stuck, cell ahead blocked
    c.observe_progress(T, FORWARD, cm)
    c.observe_progress(T, FORWARD, cm)
    assert c.stuck
    ahead_rc = cm.world_to_grid(np.array([0.25, 0.0]))
    assert cm.grid[ahead_rc[0], ahead_rc[1]] == OCCUPIED
