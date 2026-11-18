"""Snapping navmesh queries onto the agent's own island (sim/habitat_env.py).

HM3D navmeshes carry furniture tops as tiny separate islands. A goal next to
a desk, snapped from the agent's height, lands on the desk-top island because
it is nearer in 3D than the floor beside the desk -- and every such goal then
reads as unreachable (00880's plate: 300 strikes on a track 0.05 m from the
object). Constrained to the agent's island, the same query snaps to the floor.
"""
from __future__ import annotations

import numpy as np

from osg.sim.habitat_env import HabitatObjectNavEnv


class _PF:
    """Two islands: the floor (0) and a desk top (1) that is nearer to the query."""

    def __init__(self):
        self.calls = []

    def snap_point(self, p, island_index=-1):
        self.calls.append(int(island_index))
        p = np.asarray(p, dtype=np.float32)
        near_desk = p[0] > 2.0
        if island_index == 0 or not near_desk:
            return np.array([p[0], 0.18, p[2] + (0.6 if near_desk else 0.0)], dtype=np.float32)  # the floor
        return np.array([p[0] + 0.2, 0.78, p[2]], dtype=np.float32)      # the desk top, nearer in 3D

    def get_island(self, p):
        return 1 if abs(float(np.asarray(p)[1]) - 0.78) < 1e-3 else 0

    def find_path(self, path):
        a = np.asarray(path.requested_start); b = np.asarray(path.requested_end)
        path.geodesic_distance = float(np.linalg.norm(a - b))
        return self.get_island(a) == self.get_island(b)


class _State:
    position = np.array([0.0, 0.18, 0.0], dtype=np.float32)


class _Sim:
    pathfinder = _PF()

    def get_agent_state(self):
        return _State()


def _env(snap_on_island: bool) -> HabitatObjectNavEnv:
    env = HabitatObjectNavEnv.__new__(HabitatObjectNavEnv)
    env.env = type("E", (), {"sim": _Sim()})()
    env.snap_on_agent_island = snap_on_island
    return env


def test_default_snap_lands_on_the_desk_top_and_reads_unreachable():
    env = _env(False)
    assert env.is_reachable(np.array([3.0, -2.5])) is False


def test_snapping_on_the_agents_island_finds_the_floor_beside_the_desk():
    env = _env(True)
    assert env.is_reachable(np.array([3.0, -2.5])) is True
    assert 0 in env.env.sim.pathfinder.calls  # the query was constrained to island 0


def test_a_nan_island_snap_falls_back_to_the_plain_snap():
    env = _env(True)

    class _PFNaN(_PF):
        def snap_point(self, p, island_index=-1):
            if island_index == 0:
                return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
            return super().snap_point(p, island_index)

    env.env.sim.pathfinder = _PFNaN()
    assert env.is_reachable(np.array([3.0, -2.5])) is False  # the plain snap is the desk top
