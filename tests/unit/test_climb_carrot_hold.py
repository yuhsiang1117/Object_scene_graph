"""The climber keeps the tread it is walking to.

`_flight_carrot` picked the nearest cell in a height band every step, so the
goal moved every step -- and `PointNavDriver` wipes its recurrent state
whenever the goal moves more than 0.1 m. A point-goal policy reset every step
turns toward the new goal, takes one action, and starts over. Measured
(outputs/mf5_pass2_v16 ep1): 300 steps for 2.56 m of descent, with NO
`climb_forced_forward` and no `climb_blocked_turn` in the whole climb, so the
mover always had an action and the agent was never pressed against anything.
The same production climb on the same pasted flight, started aligned with it,
covers 2.74 m in 50 steps.
"""
from types import SimpleNamespace

import numpy as np

from osg.agent.climb import ClimbPolicy
from osg.mapping.costmap import Costmap2D

from .test_climb import make_cfg


class _Flight:
    """A straight flight of treads climbing 0.15 m every 0.25 m."""

    kind = "up"

    def __init__(self, cm, n=20):
        self.cells_rc = np.array([cm.world_to_grid(np.array([0.25 * i, 0.0]))
                                  for i in range(n)])
        self.heights = np.array([0.15 * i for i in range(n)], dtype=float)
        self.foot_xy = np.array([0.0, 0.0])
        self.top_xy = np.array([0.25 * (n - 1), 0.0])

    @property
    def n_cells(self):
        return len(self.cells_rc)


def _agent(hold_m):
    cfg = make_cfg()
    cfg.agent.climb_flight_carrot = True
    cfg.agent.climb_carrot_min_ahead_m = 0.0
    cfg.agent.climb_carrot_hold_m = hold_m
    cfg.agent.camera_height = 0.88
    cm = Costmap2D(resolution=0.05, size_m=20.0, track_height=True)
    agent = SimpleNamespace(
        cfg=cfg, stats={}, costmap=cm,
        floors=SimpleNamespace(pursuit_flight=_Flight(cm)),
        _climb_direction=1, _flight_carrot_xy=None, _flight_carrot_h=0.0,
        _climb_blocked_carrots=[],
    )
    agent._hold_flight_carrot = lambda xy, h, i: ClimbPolicy._hold_flight_carrot(agent, xy, h, i)
    return agent


SLOPE = 0.15 / 0.25          # the fixture flight: 0.15 m up per 0.25 m along


def _carrot(agent, x, standing=None):
    """The carrot with the agent standing ON the flight at `x` -- so its own
    height rises as it advances, which is what moves the height band."""
    standing = SLOPE * x if standing is None else standing
    frame = SimpleNamespace(camera_position=np.array(
        [x, standing + agent.cfg.agent.camera_height, 0.0]))
    return ClimbPolicy._flight_carrot(agent, frame, np.array([x, 0.0]))


def test_without_the_hold_the_goal_moves_every_step():
    agent = _agent(0.0)
    goals = [_carrot(agent, x) for x in (0.0, 0.25, 0.5, 0.75, 1.0)]
    moves = [float(np.linalg.norm(b - a)) for a, b in zip(goals, goals[1:])]
    assert sum(m > 0.1 for m in moves) >= 2, (
        "the goal used to jump every step; this fixture no longer shows it")


def test_the_held_carrot_does_not_move_until_it_is_reached():
    agent = _agent(0.35)
    first = _carrot(agent, 0.0)
    for x in (0.1, 0.2, 0.3):
        assert np.allclose(_carrot(agent, x), first), "the carrot moved mid-approach"
    assert agent.stats["climb_carrot_held"] == 3


def test_reaching_it_picks_the_next_one():
    agent = _agent(0.35)
    first = _carrot(agent, 0.0)
    nxt = _carrot(agent, float(first[0]) - 0.2)        # within 0.35 m of it
    assert not np.allclose(nxt, first), "the carrot was never released"
    assert agent.stats.get("climb_carrot_repick", 0) >= 1


def test_climbing_past_it_also_releases_it():
    """On a staircase the agent gains height without closing much ground; a
    carrot already below it must not hold the climber back."""
    agent = _agent(0.35)
    first = _carrot(agent, 0.0)
    held_h = agent._flight_carrot_h
    later = _carrot(agent, 0.05, standing=held_h + 0.5)
    assert not np.allclose(later, first)


def test_the_hold_never_outlives_the_flight():
    """A relink is a new staircase; `_do_climb` clears the held carrot and the
    next call must choose from the new flight."""
    agent = _agent(0.35)
    _carrot(agent, 0.0)
    agent._flight_carrot_xy = None
    again = _carrot(agent, 0.1)
    assert again is not None


# ------------------------------------------- a tread the mover cannot reach

def test_a_blocked_tread_is_not_offered_again():
    """Holding a goal means holding an UNREACHABLE one too.

    Measured (outputs/mf5_pass2_v17 ep1, the descent): 107 forced-forwards and
    24 blocked turns inside a climb where v16 had none of either. The held
    tread was across the banister, the point-goal policy cannot route around
    it, and the climber kept re-committing to it.
    """
    agent = _agent(0.35)
    first = _carrot(agent, 0.0)
    agent._climb_blocked_carrots.append(np.asarray(first, dtype=float).copy())
    agent._flight_carrot_xy = None

    again = _carrot(agent, 0.0)
    assert again is not None, "the whole band was thrown away"
    assert float(np.linalg.norm(again - first)) > 0.25, "the same tread came back"


def test_blocking_never_empties_the_band():
    """If every remaining tread is blocked the climber still gets a goal --
    something to drive at beats nothing, and `_carrot_stalled` ends a climb
    that is genuinely going nowhere."""
    agent = _agent(0.35)
    goal = _carrot(agent, 0.0)
    cm = agent.costmap
    agent._climb_blocked_carrots = [
        cm.grid_to_world(rc.astype(float)) for rc in agent.floors.pursuit_flight.cells_rc]
    agent._flight_carrot_xy = None
    assert _carrot(agent, 0.0) is not None
