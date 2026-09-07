from __future__ import annotations

import numpy as np
import pytest

from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN, Costmap2D
from osg.planning.planner import AStarPlanner


def _open_map() -> Costmap2D:
    cm = Costmap2D(resolution=0.1, size_m=10.0)
    cm.grid[:, :] = FREE
    return cm


def test_straight_line_cost():
    cm = _open_map()
    planner = AStarPlanner(inflate_radius_m=0.1)
    res = planner.plan(cm, np.array([0.0, 0.0]), np.array([2.0, 0.0]))
    assert res.success
    assert res.cost == pytest.approx(2.0, abs=0.4)


def test_routes_around_wall():
    cm = _open_map()
    # Wall across the middle with no gap in the corridor of interest
    mid = cm.world_to_grid(np.array([1.0, 0.0]))[0]
    cm.grid[mid, 20:80] = OCCUPIED
    planner = AStarPlanner(inflate_radius_m=0.1)
    res = planner.plan(cm, np.array([0.0, 0.0]), np.array([2.0, 0.0]))
    assert res.success
    assert res.cost > 2.5  # detour required


def test_unreachable_goal():
    cm = _open_map()
    goal_rc = cm.world_to_grid(np.array([2.0, 2.0]))
    r, c = goal_rc
    cm.grid[r - 5 : r + 6, c - 5 : c + 6] = OCCUPIED
    planner = AStarPlanner(inflate_radius_m=0.1, goal_tolerance_m=0.1)
    res = planner.plan(cm, np.array([0.0, 0.0]), np.array([2.0, 2.0]))
    assert not res.success


def test_unknown_traversable_with_penalty():
    """Goal inside unknown space must still be plannable (frontier goals)."""
    cm = _open_map()
    half = cm.grid.shape[1] // 2
    cm.grid[:, half:] = UNKNOWN
    planner = AStarPlanner(inflate_radius_m=0.1)
    goal = cm.grid_to_world(np.array([50, half + 10]))
    start = cm.grid_to_world(np.array([50, 10]))
    res = planner.plan(cm, start, goal)
    assert res.success


def test_per_call_goal_tolerance_overrides_default():
    """P1f: APPROACH needs a tighter stopping precision than frontier/
    verify-view travel without constructing a second planner instance."""
    cm = _open_map()
    planner = AStarPlanner(inflate_radius_m=0.05, goal_tolerance_m=0.3)
    goal = np.array([2.0, 0.0])
    start = np.array([0.0, 0.0])

    loose = planner.plan(cm, start, goal)  # uses the 0.3 m constructor default
    tight = planner.plan(cm, start, goal, goal_tolerance_m=0.05)

    assert loose.success and tight.success
    loose_final_gap = float(np.linalg.norm(loose.path[-1] - goal))
    tight_final_gap = float(np.linalg.norm(tight.path[-1] - goal))
    assert tight_final_gap < loose_final_gap
    assert tight_final_gap <= 0.15  # within ~1-2 cells of the 0.1 m grid
