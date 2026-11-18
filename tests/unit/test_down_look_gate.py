"""The periodic look-down must not hunt for stairs in the middle of a bedroom.

`_down_look` is purely time-gated: every `down_look_every` steps the agent
pitches down, feeds whatever is in front of it to the stair detector, and
pitches back. Two steps, wherever it stands. Measured on 00800 (v13/v14):
6-16 look-downs an episode, none at the staircase, in a run whose stair map
was pasted from a prior ASCENT pass and already says where the stairs are.
"""
import numpy as np

from osg.agent.nav_agent import NavAgent
from osg.agent.stair_sense import StairSense
from osg.mapping.costmap import PLANE

from .test_climb import make_cfg


class _Frame:
    def __init__(self, xy):
        self.camera_position = np.array([xy[0], 1.0, -xy[1]], dtype=float)


def _agent(near_m):
    """The gate reads `cfg` and `costmap` and nothing else, so a stub carries
    them. NOTHING here may touch the NavAgent class itself -- every other test
    in the suite shares it."""
    from types import SimpleNamespace

    from osg.mapping.costmap import Costmap2D

    cfg = make_cfg()
    cfg.agent.down_look_every = 30
    cfg.agent.down_look_near_stairs_m = near_m
    return SimpleNamespace(cfg=cfg, stats={},
                           costmap=Costmap2D(resolution=0.05, size_m=20.0))


def _look(agent, xy):
    """`_down_look_here` called as a plain function on the stub."""
    return StairSense._down_look_here(agent, _Frame(xy))


def _stairs_at(agent, xy):
    cm = agent.costmap
    cm.stair_mask = np.zeros(cm.grid.shape, dtype=bool)
    rc = cm.world_to_grid(np.asarray(xy, float))
    cm.stair_mask[rc[0] - 5:rc[0] + 5, rc[1] - 5:rc[1] + 5] = True


def test_far_from_the_stair_map_the_look_down_is_skipped():
    agent = _agent(3.0)
    _stairs_at(agent, (0.0, 0.0))
    assert _look(agent, (8.0, 8.0)) is False


def test_near_the_stair_map_it_still_fires():
    agent = _agent(3.0)
    _stairs_at(agent, (0.0, 0.0))
    assert _look(agent, (1.0, 1.0)) is True


def test_with_no_stair_map_there_is_something_to_discover():
    """No prior, no confirmed stairs: the gate must not fire, because finding
    an unknown staircase is exactly what the look-down is for."""
    agent = _agent(3.0)
    assert _look(agent, (8.0, 8.0)) is True
    agent.costmap.stair_mask = np.zeros(agent.costmap.grid.shape, dtype=bool)
    assert _look(agent, (8.0, 8.0)) is True


def test_zero_keeps_the_shipped_behaviour():
    agent = _agent(0.0)
    _stairs_at(agent, (0.0, 0.0))
    assert _look(agent, (9.0, 9.0)) is True
