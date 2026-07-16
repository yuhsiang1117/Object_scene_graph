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


def test_arrival_tolerance_is_overridable_per_call():
    """P1f: APPROACH's final segment passes a tighter arrival_tol_m than
    the 0.2 m default used for frontier/verify-view travel, so it keeps
    walking instead of declaring "close enough" prematurely."""
    c = WaypointController()
    T = make_camera([2.0, 0.88, 0.0], [3.0, 0.88, 0.0])  # facing +x
    path = np.array([[2.15, 0.0]])  # 0.15 m straight ahead: inside 0.2 m, outside 0.1 m

    assert c.act(T, path, arrival_tol_m=0.2) is None  # default: arrived
    assert c.act(T, path, arrival_tol_m=0.1) == FORWARD  # tighter: keep going


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


def test_stuck_marks_expire():
    """P1b regression: a stuck mark is a hypothesis, not sensed geometry, and
    costmap.update()'s raycast can never see past a forced-OCCUPIED cell to
    correct it on its own (the mark blocks the very ray that would disprove
    it). Without an expiry, one or two stuck events near a real doorway
    permanently sealed it — an agent was observed circling one room for an
    entire 500-step episode after two early stuck events near its only exit."""
    from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN, Costmap2D

    cm = Costmap2D(resolution=0.1, size_m=5.0)
    cm.grid[:, :] = FREE
    c = WaypointController(stuck_after=2, stuck_mark_expiry_steps=60)
    T = make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0])
    c.observe_progress(T, None, cm, step_count=0)
    c.observe_progress(T, FORWARD, cm, step_count=1)
    c.observe_progress(T, FORWARD, cm, step_count=2)  # stuck triggers here
    assert c.stuck
    ahead_rc = cm.world_to_grid(np.array([0.25, 0.0]))
    assert cm.grid[ahead_rc[0], ahead_rc[1]] == OCCUPIED

    # Well before expiry: still marked (only decay-checks, doesn't move time).
    c.observe_progress(T, None, cm, step_count=30)
    assert cm.grid[ahead_rc[0], ahead_rc[1]] == OCCUPIED

    # Past expiry (2 + 60 = 62): released back to UNKNOWN, not silently FREE
    # (we have no fresh evidence either way -- A* treats UNKNOWN as
    # traversable at a penalty, exactly "try again, but not for free").
    c.observe_progress(T, None, cm, step_count=65)
    assert cm.grid[ahead_rc[0], ahead_rc[1]] == UNKNOWN


def test_stuck_marks_do_not_clear_real_obstacles():
    """A cell that was independently sensed as OCCUPIED (a real wall) must
    not be affected by decay bookkeeping for an unrelated stuck event."""
    from osg.mapping.costmap import OCCUPIED, Costmap2D, FREE

    cm = Costmap2D(resolution=0.1, size_m=5.0)
    cm.grid[:, :] = FREE
    real_wall_rc = (10, 10)
    cm.grid[real_wall_rc] = OCCUPIED
    c = WaypointController(stuck_after=2, stuck_mark_expiry_steps=5)
    T = make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0])
    c.observe_progress(T, None, cm, step_count=0)
    c.observe_progress(T, FORWARD, cm, step_count=1)
    c.observe_progress(T, FORWARD, cm, step_count=2)
    c.observe_progress(T, None, cm, step_count=100)  # well past any expiry
    assert cm.grid[real_wall_rc] == OCCUPIED  # untouched: never in _stuck_cells
