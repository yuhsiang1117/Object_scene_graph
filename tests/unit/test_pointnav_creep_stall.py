"""The creep must not press for ever at a goal it cannot reach.

`PointNavDriver.step` returns a blind `move_forward` between the arrival radius
and the creep radius -- no network, no obstacle test. That is fine for a goal
the agent can walk to and fatal for one it cannot: on the sensor arm 27 of 44
stranded approaches had a goal the navmesh calls non-navigable, habitat refused
every forward, and the press ran to the episode budget.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.planning.pointnav_driver import PointNavDriver


class _FakeDriver(PointNavDriver):
    """The stall logic without the 34 MB checkpoint or a GPU."""

    def __init__(self, **kw):
        self.stop_radius = float(kw.pop("stop_radius", 0.3))
        self.creep_stall_steps = int(kw.pop("creep_stall_steps", 0))
        self.creep_stall_eps = float(kw.pop("creep_stall_eps", 0.05))
        self.goal_change_m = 0.1
        self.last_rho = self.last_theta = None
        self.n_creep_stalls = 0
        self._rho = 1.0
        self.reset()

    # The parent's reset/_reset_creep are what we are testing; only the parts
    # that need torch are replaced.
    def reset(self):
        self._last_goal = None
        self.n_resets = 0
        self._started = False
        self._reset_creep()

    def _reset_recurrent(self):
        self._started = False

    def step(self, goal_xy, *, stop_radius=None, creep_below=0.0):
        from osg.planning.pointnav_driver import NavStep
        goal = np.asarray(goal_xy, dtype=float)
        if self._last_goal is None or np.linalg.norm(goal - self._last_goal) > self.goal_change_m:
            self._reset_recurrent()
            self._reset_creep()
            self.n_resets += 1
        self._last_goal = goal
        rho = self._rho
        radius = self.stop_radius if stop_radius is None else float(stop_radius)
        if rho < radius:
            return NavStep(None, "arrived")
        if creep_below > 0.0 and rho < creep_below:
            if self.creep_stall_steps > 0:
                if self._creep_best_rho is None or rho < self._creep_best_rho - self.creep_stall_eps:
                    self._creep_best_rho = rho
                    self._creep_stalled_steps = 0
                else:
                    self._creep_stalled_steps += 1
                    if self._creep_stalled_steps >= self.creep_stall_steps:
                        self.n_creep_stalls += 1
                        return NavStep(None, "creep_stalled")
            return NavStep("move_forward", "creep")
        return NavStep("move_forward", "moving")


def _press(driver, rho, n, goal=(1.0, 0.0)):
    """n creep steps at a fixed distance; returns the reasons in order."""
    out = []
    for _ in range(n):
        driver._rho = rho
        out.append(driver.step(np.array(goal), stop_radius=0.3, creep_below=1.0).reason)
    return out


def test_the_press_is_unconditional_when_the_window_is_zero():
    """Every measured ascent arm ran on this; the default must not change it."""
    d = _FakeDriver(creep_stall_steps=0)
    assert set(_press(d, 0.6, 50)) == {"creep"}
    assert d.n_creep_stalls == 0


def test_a_press_that_stops_closing_reports_arrival():
    d = _FakeDriver(creep_stall_steps=8)
    reasons = _press(d, 0.6, 20)
    assert reasons[0] == "creep", "the first step establishes the best distance"
    assert "creep_stalled" in reasons
    assert reasons.index("creep_stalled") == 8, "one call sets the best, the next eight fail to beat it"
    assert d.n_creep_stalls >= 1


def test_a_press_that_keeps_closing_is_never_stalled():
    d = _FakeDriver(creep_stall_steps=4)
    reasons = []
    for rho in np.arange(0.95, 0.35, -0.06):
        d._rho = float(rho)
        reasons.append(d.step(np.array([1.0, 0.0]), stop_radius=0.3, creep_below=1.0).reason)
    assert "creep_stalled" not in reasons
    assert d.n_creep_stalls == 0


def test_progress_under_the_epsilon_still_counts_as_stalled():
    """Sliding 1 cm per step against a wall is not progress."""
    d = _FakeDriver(creep_stall_steps=5, creep_stall_eps=0.05)
    reasons = []
    for i in range(20):
        d._rho = 0.60 - 0.01 * i
        reasons.append(d.step(np.array([1.0, 0.0]), stop_radius=0.3, creep_below=1.0).reason)
    assert "creep_stalled" in reasons


def test_a_new_goal_starts_a_new_pursuit():
    """Leaking the previous goal's best distance would stall the next one at once."""
    d = _FakeDriver(creep_stall_steps=3)
    _press(d, 0.6, 10, goal=(1.0, 0.0))
    assert d.n_creep_stalls >= 1
    before = d.n_creep_stalls
    first = d.step(np.array([5.0, 5.0]), stop_radius=0.3, creep_below=1.0)
    assert first.reason == "creep", "a fresh goal may not inherit a stall"
    assert d.n_creep_stalls == before


def test_arrival_still_wins_over_the_stall_test():
    d = _FakeDriver(creep_stall_steps=1)
    d._rho = 0.2
    assert d.step(np.array([1.0, 0.0]), stop_radius=0.3, creep_below=1.0).reason == "arrived"


def test_the_stall_never_fires_outside_the_creep_band():
    d = _FakeDriver(creep_stall_steps=2)
    reasons = [d.step(np.array([1.0, 0.0]), stop_radius=0.3, creep_below=0.0).reason
               for _ in range(10)]
    assert set(reasons) == {"moving"}
