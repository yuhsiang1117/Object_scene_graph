"""`agent.climb_route_start_near_m`: a route probe starts a climb only on the
staircase being pursued.

00821 (seed 42, cross 50006, mfv2_osg_mouth_walk): walking to the flight at
(-1.77, 1.43), the probe found a 0.5 m route up something step-like at (3, -0.5),
4 m away, and the climb stalled there at 0.0 m. The cells below are that
flight's first leg, read off the navmesh: west along z = 1.4 from x = -1.8.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from osg.agent.climb import ClimbPolicy
from osg.agent.floor_policy import FloorPolicy
from osg.mapping.stair_route import StairRoute

FLOOR = -3.47
LEG = np.array([[x, 1.4] for x in np.arange(-1.8, -3.7, -0.1)])


def _route(xy, heights):
    xy = np.asarray(xy, dtype=float)
    return StairRoute(xy=xy, heights=np.asarray(heights, dtype=float),
                      cost_m=np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))],
                      reaches_target=False, gain_m=float(max(heights) - heights[0]))


FURNITURE = _route([[1.46, -0.62], [2.0, -0.6], [2.5, -0.6], [3.0, -0.5]],
                   [FLOOR, FLOOR, FLOOR + 0.25, FLOOR + 0.5])
THE_FLIGHT = _route([[-0.9, 1.4], [-1.5, 1.4], [-1.8, 1.4], [-2.1, 1.4]],
                    [FLOOR, FLOOR, FLOOR + 0.36, FLOOR + 0.62])


def _climber(near_m, kind="flight_up", cells=LEG, goal=(-1.77, 1.43)):
    flight = SimpleNamespace(cells_xy=cells) if cells is not None else None
    return SimpleNamespace(
        cfg=SimpleNamespace(agent=SimpleNamespace(climb_route_start_near_m=near_m)),
        floors=SimpleNamespace(pursuit_kind=kind, pursuit_flight=flight),
        nav=SimpleNamespace(_goal_xy=np.asarray(goal, dtype=float)),
    )


def test_a_route_up_something_else_does_not_start_the_climb():
    c = _climber(1.5)
    assert not ClimbPolicy._route_on_pursued_stairs(c, FURNITURE, FLOOR)
    assert ClimbPolicy._route_on_pursued_stairs(c, THE_FLIGHT, FLOOR)


def test_off_by_default_as_measured():
    assert ClimbPolicy._route_on_pursued_stairs(_climber(0.0), FURNITURE, FLOOR)


def test_a_portal_pursuit_is_left_free():
    assert ClimbPolicy._route_on_pursued_stairs(_climber(1.5, kind="portal"), FURNITURE, FLOOR)
    assert ClimbPolicy._route_on_pursued_stairs(_climber(1.5, kind=None), FURNITURE, FLOOR)


def test_a_flight_whose_foot_is_misplaced_still_counts_by_its_cells():
    """00800: the flight's 'foot' (-8.92, -2.77) sits beside the middle of the
    treads; the real foot (-11.6, -3.5) is 2.7 m from it, but on the flight."""
    cells = np.array([[x, -3.5 - 0.27 * (x + 11.6)] for x in np.arange(-11.6, -7.3, 0.1)])
    route = _route([[-11.0, -3.0], [-11.6, -3.5], [-11.7, -3.8]], [0.16, 0.32, 0.56])
    c = _climber(1.5, cells=cells, goal=(-8.92, -2.77))
    assert ClimbPolicy._route_on_pursued_stairs(c, route, 0.16)


def test_without_cells_the_goal_decides():
    """A remembered stair: pass 1's arrival is a point, 0.3-1.0 m from the mouth."""
    c = _climber(1.5, kind="prior_stair", cells=None, goal=(-1.2, 1.5))
    assert ClimbPolicy._route_on_pursued_stairs(c, THE_FLIGHT, FLOOR)
    assert not ClimbPolicy._route_on_pursued_stairs(c, FURNITURE, FLOOR)


def test_pursuit_kind_reads_the_portal_log():
    f = SimpleNamespace(pursuing=True, portal_log=[(5, [0.0, 0.0], 0.4, 30)])
    assert FloorPolicy.pursuit_kind.fget(f) == "portal"
    f.portal_log.append((9, [1.0, 1.0], "flight_up", 594))
    assert FloorPolicy.pursuit_kind.fget(f) == "flight_up"
    f.pursuing = False
    assert FloorPolicy.pursuit_kind.fget(f) is None


def test_in_the_climb_the_goal_is_the_climbs_own():
    """Re-planning inside a climb (`_route_action`) judges against the goal the
    climb started for, whatever `nav._goal_xy` has become since."""
    from osg.agent.state import State

    c = _climber(1.5, kind="prior_stair", cells=None, goal=(30.0, 30.0))
    c.nav.state = State.CLIMB
    c._climb_goal_xy = np.array([-1.2, 1.5])
    assert ClimbPolicy._route_on_pursued_stairs(c, THE_FLIGHT, FLOOR)
    assert not ClimbPolicy._route_on_pursued_stairs(c, FURNITURE, FLOOR)
