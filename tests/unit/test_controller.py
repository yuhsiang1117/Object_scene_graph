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
    T = make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0])  # facing world +x
    # Facing +x with up +y: right = forward x up = +z (plane axis 1)
    path_pos_z = np.array([[0.0, 0.0], [0.0, 1.0]])
    assert c.act(T, path_pos_z) == TURN_RIGHT
    path_neg_z = np.array([[0.0, 0.0], [0.0, -1.0]])
    assert c.act(T, path_neg_z) == TURN_LEFT


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
    # P1b: a single ~0.1 m point left doorway-width bottlenecks bypassable
    # (an agent was observed frozen at one spot for 45 steps across 3
    # give-up cycles). The mark must now be a wider halo, not one point.
    farther_rc = cm.world_to_grid(np.array([0.6, 0.0]))
    assert cm.grid[farther_rc[0], farther_rc[1]] == OCCUPIED
    off_axis_rc = cm.world_to_grid(np.array([0.25, 0.3]))  # outside the old 0.2 m radius
    assert cm.grid[off_axis_rc[0], off_axis_rc[1]] == OCCUPIED
