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


# --------------------------------------------- why the follower stopped (S+2)
#
# action_to_goal returns None for four different situations and every caller has
# treated them alike: the snap failed, the follower raised, the follower stopped
# because the agent arrived, or the follower stopped because it will not go.
# Guessing between them was wrong twice -- once as "the goal snaps through a
# wall", once as "the object is on a disconnected navmesh island" -- and neither
# survived measurement.
#
# The counter is a PROPERTY rather than an __init__ field because
# YCBAuthoredNavEnv defines its own __init__ and never runs the parent's. The
# first version set it in __init__ and crashed a diagnostic run on
# AttributeError; this test is that bug.


class _StopFollower:
    """A follower that always says stop, like one refusing an unreachable goal."""

    def __init__(self, stop_action):
        self._stop = stop_action

    def get_next_action(self, goal):
        return self._stop


def _env_without_running_parent_init(cfg, agent_y=GROUND_Y):
    """Exactly what YCBAuthoredNavEnv does: build the object without the
    parent's __init__ ever running."""
    env = HabitatObjectNavEnv.__new__(HabitatObjectNavEnv)
    env.env = type("E", (), {"sim": _StubSim(agent_y)})()
    env._navmesh_goal_radius = float(cfg.agent.navmesh_goal_radius)
    env._follower = None
    return env


def test_the_counter_exists_even_when_the_parent_init_never_ran():
    from osg.core.config import OSGConfig

    env = _env_without_running_parent_init(OSGConfig())
    env.nav_reasons["probe"] += 1          # must not raise AttributeError
    assert env.nav_reasons["probe"] == 1


def test_a_stop_issued_from_across_the_room_is_recorded_as_a_refusal():
    from osg.core.config import OSGConfig

    cfg = OSGConfig()
    env = _env_without_running_parent_init(cfg)
    env._follower = _StopFollower(HabitatObjectNavEnv.ACTIONS["stop"])
    # a goal eight metres away, and the follower says stop
    assert env.action_to_goal(np.array([8.0, 0.0])) is None
    assert env.nav_reasons["nav_refused"] == 1
    assert env.nav_reasons["nav_arrived"] == 0


def test_a_stop_issued_at_the_goal_is_recorded_as_an_arrival():
    from osg.core.config import OSGConfig

    cfg = OSGConfig()
    env = _env_without_running_parent_init(cfg)
    env._follower = _StopFollower(HabitatObjectNavEnv.ACTIONS["stop"])
    assert env.action_to_goal(np.array([0.05, 0.0])) is None
    assert env.nav_reasons["nav_arrived"] == 1
    assert env.nav_reasons["nav_refused"] == 0
