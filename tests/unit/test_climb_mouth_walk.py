"""`agent.climb_route_mouth_first`: with no route onto the stairs in view from
the start floor, the climb walks to its own goal -- the stair mouth the pursuit
drove to -- before the far-end walk meant for landings.

00862 (seed 42, static 50007, mfv2_osg_mouth): both climbs began 1.2 m short of
the remembered mouth, saw no tread, and the far-end walk took the agent 4 m
along the upper floor.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from osg.agent.climb import ClimbPolicy

CAM = 0.88


def _climber(on=True, goal=(0.0, -1.2), floor_y=3.3):
    planned = []

    def plan(frame, agent_xy, direction, explore=False, toward_xy=None):
        planned.append((explore, None if toward_xy is None else tuple(toward_xy)))
        return SimpleNamespace(end_xy=np.asarray(toward_xy, float))

    c = SimpleNamespace(
        cfg=SimpleNamespace(agent=SimpleNamespace(climb_route_mouth_first=on, camera_height=CAM)),
        floors=SimpleNamespace(estimator=SimpleNamespace(height_of=lambda k: floor_y)),
        _climb_from_floor=0, _climb_goal_xy=None if goal is None else np.asarray(goal, float),
        _climb_direction=-1, _route_mouth_walked=False, stats={}, _plan_route=plan)
    return c, planned


def _frame(standing):
    return SimpleNamespace(camera_position=np.array([0.0, standing + CAM, 0.0]))


def test_from_the_floor_it_walks_to_the_mouth_once():
    c, planned = _climber()
    walk = ClimbPolicy._walk_to_mouth(c, _frame(3.3), np.array([0.0, 0.0]))
    assert walk is not None and planned == [(True, (0.0, -1.2))]
    assert c.stats["climb_route_mouth"] == 1
    assert ClimbPolicy._walk_to_mouth(c, _frame(3.3), np.array([0.0, 0.0])) is None


def test_off_by_default_as_measured():
    c, planned = _climber(on=False)
    assert ClimbPolicy._walk_to_mouth(c, _frame(3.3), np.array([0.0, 0.0])) is None
    assert planned == []


def test_not_from_the_treads_and_not_when_already_there():
    c, _ = _climber()
    assert ClimbPolicy._walk_to_mouth(c, _frame(2.9), np.array([0.0, 0.0])) is None   # 0.4 m down
    assert ClimbPolicy._walk_to_mouth(c, _frame(3.3), np.array([0.0, -0.8])) is None  # 0.4 m away
    c, _ = _climber(goal=None)
    assert ClimbPolicy._walk_to_mouth(c, _frame(3.3), np.array([0.0, 0.0])) is None
