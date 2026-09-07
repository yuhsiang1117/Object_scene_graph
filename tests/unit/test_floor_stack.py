"""Floor identification from observed standing heights."""
from __future__ import annotations

import numpy as np
import pytest

from osg.mapping.costmap import Costmap2D
from osg.mapping.floor_stack import FloorLayer, FloorStack

STOREY = 2.7


def _stack(band_m: float = 0.9, commit_steps: int = 4) -> FloorStack:
    def make_layer(key, floor_y):
        return FloorLayer(
            key=key, floor_y=floor_y, costmap=Costmap2D(resolution=0.1, size_m=10.0),
            planner=object(),
        )

    return FloorStack(make_layer, band_m=band_m, commit_steps=commit_steps)


def _feed(stack, ys):
    for i, y in enumerate(ys):
        stack.observe(y, step=i)


def test_single_floor_never_splits():
    """Sensor-level jitter and normal floor unevenness must not spawn floors."""
    stack = _stack()
    rng = np.random.default_rng(0)
    _feed(stack, rng.normal(0.0, 0.05, size=200))
    assert stack.n_floors() == 1


def test_new_floor_allocated_beyond_band():
    stack = _stack()
    _feed(stack, [0.0] * 5 + [STOREY] * 5)
    assert stack.n_floors() == 2
    assert stack.current().floor_y == pytest.approx(STOREY, abs=0.05)
    assert stack.order_of(stack.current().key) == 1


def test_hysteresis_ignores_a_single_spike():
    """One bad sample (or one step onto a landing) must not swap the active
    map: everything written that step would land on the wrong floor."""
    stack = _stack(commit_steps=4)
    _feed(stack, [0.0] * 5)
    ground = stack.current()
    stack.observe(STOREY, step=5)
    stack.observe(0.0, step=6)
    assert stack.current() is ground
    assert stack.switches == 0


def test_ramp_commits_exactly_once():
    stack = _stack(commit_steps=4)
    _feed(stack, [0.0] * 5 + list(np.linspace(0.0, STOREY, 12)) + [STOREY] * 10)
    assert stack.switches == 1
    assert stack.order_of(stack.current().key) == 1


def test_climbing_stairs_allocates_no_phantom_floor():
    """A staircase passes through heights far from BOTH real floors. Without a
    settle test those become a floor of their own -- one phantom layer part-way
    up, then the real one at the top, with the phantom left in n_floors, up()
    and down() forever."""
    stack = _stack(commit_steps=4)
    _feed(stack, [0.0] * 8)
    # ~0.17 m per forward step, a 30-35 deg staircase at forward_m=0.25.
    _feed(stack, list(np.arange(0.17, STOREY, 0.17)))
    _feed(stack, [STOREY] * 8)

    assert stack.n_floors() == 2, "a mid-staircase height became a floor"
    assert stack.switches == 1
    assert [round(l.floor_y, 1) for l in stack.layers()] == [0.0, round(STOREY, 1)]


def test_transit_holds_the_current_floor():
    """While between floors, `current` must not change: whatever is written
    that step would otherwise land on a floor the agent is not on."""
    stack = _stack()
    _feed(stack, [0.0] * 8)
    ground = stack.current()
    for y in (0.9, 1.2, 1.5, 1.8):  # unrecognised, and not settled
        assert stack.observe(y) is ground


def test_keys_stable_when_a_lower_floor_is_discovered():
    """order is recomputed by height; key is not. A basement found mid-episode
    must not renumber floors that tracks/frontiers already reference."""
    stack = _stack()
    _feed(stack, [0.0] * 6)
    ground_key = stack.current().key
    _feed(stack, [STOREY] * 6)
    upper_key = stack.current().key
    _feed(stack, [-STOREY] * 6)
    basement_key = stack.current().key

    assert {ground_key, upper_key, basement_key} == {0, 1, 2}
    assert stack.order_of(basement_key) == 0
    assert stack.order_of(ground_key) == 1
    assert stack.order_of(upper_key) == 2


def test_up_and_down_follow_height_not_key():
    stack = _stack()
    _feed(stack, [0.0] * 6)
    ground = stack.current()
    _feed(stack, [-STOREY] * 6)  # discovered second, but it is BELOW
    basement = stack.current()

    assert stack.up() is ground
    assert stack.down() is None
    _feed(stack, [0.0] * 6)
    assert stack.current() is ground
    assert stack.down() is basement


def test_floor_y_is_not_dragged_during_a_transition():
    """floor_y is refined only while settled on a floor. Averaging in
    mid-staircase heights would drift two floors toward each other until the
    band merges them."""
    stack = _stack()
    _feed(stack, [0.0] * 20)
    ground = stack.current()
    _feed(stack, list(np.linspace(0.1, STOREY, 15)))
    assert ground.floor_y == pytest.approx(0.0, abs=0.02)


def test_infinite_band_pins_to_one_floor():
    """How multi-floor support is switched off: one layer, whatever the
    heights, so behaviour is identical to the single-costmap agent."""
    stack = _stack(band_m=float("inf"))
    _feed(stack, [0.0] * 5 + [STOREY] * 5 + [-STOREY] * 5)
    assert stack.n_floors() == 1
    assert stack.switches == 0


def test_steps_on_floor_counts_per_floor():
    stack = _stack()
    _feed(stack, [0.0] * 10)
    ground = stack.current()
    _feed(stack, [STOREY] * 8)
    assert ground.steps_on_floor == 10 + 3  # 3 pre-commit steps still on ground
    assert stack.current().steps_on_floor == 5


def test_revisiting_a_floor_reuses_its_layer():
    stack = _stack()
    _feed(stack, [0.0] * 6)
    ground = stack.current()
    ground.costmap.grid[0, 0] = 42
    _feed(stack, [STOREY] * 6)
    _feed(stack, [0.0] * 6)

    assert stack.current() is ground, "returning to a floor must not rebuild it"
    assert stack.current().costmap.grid[0, 0] == 42
    assert stack.n_floors() == 2
    assert ground.visits == 2


def test_in_transit_only_between_floors():
    """in_transit is the frame-rejection condition: it must be false while
    standing anywhere on a known floor (including a step or ramp within it),
    and true only on the way between floors."""
    stack = _stack()
    _feed(stack, [0.0] * 8)
    assert not stack.in_transit()

    # A 0.42 m step is still this floor -- rejecting here blinds the agent.
    stack.observe(0.42)
    assert not stack.in_transit()

    for y in (1.2, 1.5, 1.8):  # genuinely between floors
        stack.observe(y)
        assert stack.in_transit()

    _feed(stack, [STOREY] * 6)
    assert not stack.in_transit()


def test_frozen_suspends_allocation_and_switching():
    """ASCENT changes its floor index in exactly one place -- when the agent
    leaves the staircase (map_controller.py:299). OSG clusters height
    continuously, which mid-flight both allocates a layer at a landing height
    and then switches to it: measured on the cross-floor split, climbs end after
    72 cm of gain against a 90 cm threshold, so the exit comes from the
    floor-changed branch, not the height one.
    """
    fs = _stack(band_m=0.9, commit_steps=2)
    ground = fs.observe(0.0, 0)

    # Frozen: a whole staircase's worth of heights changes nothing.
    for step, y in enumerate([0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8], start=1):
        assert fs.observe(y, step, frozen=True) is ground
    assert fs.n_floors() == 1
    assert fs.switches == 0

    # And the step is still charged to the floor the agent came from, so
    # steps_on_floor and everything keyed off it behave as before.
    assert ground.steps_on_floor == 8


def test_unfrozen_still_commits_the_arrival():
    """Freezing must not make a floor change impossible -- only defer it to
    when the climb ends."""
    fs = _stack(band_m=0.9, commit_steps=2)
    fs.observe(0.0, 0)
    for step, y in enumerate([1.0, 2.0, 3.0], start=1):
        fs.observe(y, step, frozen=True)
    assert fs.n_floors() == 1
    for step, y in enumerate([3.0, 3.0, 3.0, 3.0, 3.0], start=10):
        fs.observe(y, step)
    assert fs.n_floors() == 2, "the new storey is allocated once the climb ends"


def test_freeze_is_off_by_default():
    from osg.core.config import MappingConfig

    assert MappingConfig().freeze_floor_in_climb is False


def test_frozen_still_reports_being_in_transit():
    """in_transit() gates whether NavAgent writes a frame into the costmap, and
    frozen means "on a staircase" -- precisely when a frame belongs to no
    floor's map.

    An earlier version cleared _unassigned while frozen, so in_transit() said
    "settled" and mid-staircase geometry was projected against a stale floor_y.
    The comment on in_transit() warns about exactly that corruption. It cost 6.4
    points of SINGLE-floor SR (68.4% -> 62.0%) on episodes that only walk past a
    staircase, with six of them going from dtg ~0.03 m to 1.5-9.0 m.
    """
    fs = _stack(band_m=0.9, commit_steps=2)
    fs.observe(0.0, 0)
    assert not fs.in_transit(), "settled on the ground floor"

    for step, y in enumerate([0.4, 0.9, 1.4, 1.9], start=1):
        fs.observe(y, step, frozen=True)
        assert fs.in_transit(), "on the stairs, so the frame belongs to no map"

    for step, y in enumerate([1.9] * 4, start=10):
        fs.observe(y, step)
    assert not fs.in_transit(), "settled again once the climb ends"
