"""`agent.climb_oracle`: the climb-only privileged follower (an ablation).

What is pinned is WHERE it is sent -- the far end of the flight being climbed,
at the storey the pursuit is for -- and that it gives the step back to the
sensor-only controller whenever the follower has nothing to say.
"""
from types import SimpleNamespace

import numpy as np

from osg.agent.climb import ClimbPolicy


class _Flight:
    kind = "up"
    n_cells = 3
    foot_xy = np.array([0.0, 0.0])
    top_xy = np.array([2.0, 0.5])
    heights = np.array([0.1, 1.0, 2.9])


def _climber(direction=1, target_y=3.0, answer="move_forward", flight=True):
    calls = []

    def fn(goal_xy, floor_y):
        calls.append((np.asarray(goal_xy, dtype=float).copy(), floor_y))
        return answer

    nav = SimpleNamespace(_climb_nav_fn=fn, _goal_floor_y_cache=target_y)
    climber = SimpleNamespace(
        nav=nav, stats={}, _climb_direction=direction,
        _climb_goal_xy=np.array([9.0, 9.0]),
        floors=SimpleNamespace(pursuit_flight=_Flight() if flight else None),
    )
    return climber, calls


def _act(climber):
    frame = SimpleNamespace(camera_position=np.zeros(3))
    return ClimbPolicy._oracle_action(climber, frame, np.zeros(2))


def test_without_the_ablation_the_controller_is_untouched():
    climber, _ = _climber()
    climber.nav._climb_nav_fn = None
    assert _act(climber) is None
    assert climber.stats == {}


def test_an_ascent_is_sent_to_the_top_at_the_target_storey():
    climber, calls = _climber(direction=1, target_y=3.0)
    assert _act(climber) == "move_forward"
    goal, floor_y = calls[-1]
    assert np.allclose(goal, [2.0, 0.5]) and floor_y == 3.0
    assert climber.stats["climb_oracle_steps"] == 1


def test_a_descent_is_sent_to_the_foot_at_its_own_height_when_no_storey_is_known():
    climber, calls = _climber(direction=-1, target_y=None)
    _act(climber)
    goal, floor_y = calls[-1]
    assert np.allclose(goal, [0.0, 0.0]) and floor_y == 0.1


def test_with_no_flight_the_pursuit_goal_is_used():
    climber, calls = _climber(flight=False)
    _act(climber)
    assert np.allclose(calls[-1][0], [9.0, 9.0])


def test_nothing_from_the_follower_hands_the_step_back():
    climber, _ = _climber(answer=None)
    assert _act(climber) is None
    assert climber.stats["climb_oracle_none"] == 1
