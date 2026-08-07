"""3D navmesh goals (osg.sim.habitat_env).

The bug these guard: `action_to_goal` / `is_reachable` used to substitute the
AGENT's current height into every 2D goal before snapping. A candidate one
storey up therefore snapped to whatever lies under the agent, `find_path`
failed, and `_check_candidates` blacklisted it -- so every cross-floor target
was discarded even though the navmesh connects the floors.

Stubs stand in for habitat's sim so these run without habitat_sim.
"""
import numpy as np
import pytest

from osg.sim.habitat_env import HabitatObjectNavEnv

GROUND_Y, UPPER_Y = 0.1, 2.9


class _StubPathfinder:
    """Navmesh with two storeys. snap_point returns the floor nearest the
    query height; a path exists between any two snapped points."""

    def __init__(self):
        self.snapped = []

    def snap_point(self, p):
        p = np.asarray(p, dtype=float)
        y = min((GROUND_Y, UPPER_Y), key=lambda f: abs(f - p[1]))
        out = np.array([p[0], y, p[2]], dtype=np.float32)
        self.snapped.append(out)
        return out

    def find_path(self, path):
        return True


class _StubSim:
    def __init__(self, agent_y=GROUND_Y):
        self.pathfinder = _StubPathfinder()
        self._agent_y = agent_y

    def get_agent_state(self):
        return type("AgentState", (), {"position": np.array([0.0, self._agent_y, 0.0])})()


def make_env(agent_y=GROUND_Y):
    env = HabitatObjectNavEnv.__new__(HabitatObjectNavEnv)  # bypass habitat init
    env.env = type("E", (), {"sim": _StubSim(agent_y)})()
    return env


# ------------------------------------------------------- backward compatibility


def test_2d_goal_still_uses_the_agent_height():
    """The legacy contract: no floor_y -> the agent's own height, unchanged."""
    env = make_env(agent_y=GROUND_Y)
    g = env._goal3d([1.0, 2.0])
    assert g.tolist() == pytest.approx([1.0, GROUND_Y, 2.0])


def test_2d_goal_on_an_upper_floor_agent():
    env = make_env(agent_y=UPPER_Y)
    assert env._goal3d([1.0, 2.0])[1] == pytest.approx(UPPER_Y)


# ----------------------------------------------------------------- 3D goals


def test_3d_goal_passes_through_untouched():
    env = make_env()
    assert env._goal3d([1.0, UPPER_Y, 2.0]).tolist() == pytest.approx([1.0, UPPER_Y, 2.0])


def test_floor_y_overrides_the_agent_height():
    env = make_env(agent_y=GROUND_Y)
    assert env._goal3d([1.0, 2.0], floor_y=UPPER_Y)[1] == pytest.approx(UPPER_Y)


def test_floor_y_of_zero_is_honoured_not_treated_as_missing():
    """0.0 is a legitimate floor height; `if floor_y:` would silently drop it."""
    env = make_env(agent_y=UPPER_Y)
    assert env._goal3d([1.0, 2.0], floor_y=0.0)[1] == pytest.approx(0.0)


# ------------------------------------------------------------- the real bug


def test_upstairs_goal_snaps_to_the_wrong_floor_without_floor_y():
    """Reproduces the defect: a 2D goal from a ground-floor agent snaps to the
    ground floor even when the target is upstairs."""
    env = make_env(agent_y=GROUND_Y)
    env.is_reachable([1.0, 2.0])
    assert env.env.sim.pathfinder.snapped[0][1] == pytest.approx(GROUND_Y)


def test_upstairs_goal_snaps_to_the_upper_floor_with_floor_y():
    env = make_env(agent_y=GROUND_Y)
    env.is_reachable([1.0, 2.0], floor_y=UPPER_Y)
    assert env.env.sim.pathfinder.snapped[0][1] == pytest.approx(UPPER_Y)


def test_is_reachable_accepts_a_full_3d_goal():
    env = make_env(agent_y=GROUND_Y)
    assert env.is_reachable([1.0, UPPER_Y, 2.0]) is True
    assert env.env.sim.pathfinder.snapped[0][1] == pytest.approx(UPPER_Y)


def test_unnavigable_goal_is_not_reachable():
    env = make_env()
    env.env.sim.pathfinder.snap_point = lambda p: np.array([np.nan] * 3, dtype=np.float32)
    assert env.is_reachable([1.0, 2.0]) is False
