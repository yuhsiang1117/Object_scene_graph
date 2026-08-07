"""Cross-floor portals and the floor-switch gate (osg.mapping.portals)."""
import numpy as np
import pytest

from osg.mapping.costmap import FREE, Costmap2D
from osg.mapping.portals import FloorSwitchPolicy, find_portals

RES = 0.05


def blank():
    cm = Costmap2D(resolution=RES, size_m=20.0, track_height=True)
    cm.grid[:] = FREE
    return cm


def paint(cm, r0, r1, c0, c1, y):
    cm.height[r0:r1, c0:c1] = y


# ------------------------------------------------------------------- portals


def test_flat_floor_has_no_portal():
    cm = blank()
    paint(cm, 100, 200, 100, 200, 0.0)
    assert find_portals(cm, floor_y=0.0) == []


def test_upper_storey_seen_through_an_opening_is_a_portal():
    cm = blank()
    paint(cm, 100, 200, 100, 200, 0.0)
    paint(cm, 140, 160, 140, 160, 2.8)          # mezzanine visible above
    portals = find_portals(cm, floor_y=0.0)
    assert len(portals) == 1
    assert portals[0].going_up
    assert portals[0].target_y == pytest.approx(2.8, abs=0.05)


def test_lower_storey_is_a_portal_too():
    """Descending openings are invisible to the obstacle band, which is why the
    height layer records below the floor as well as above."""
    cm = blank()
    paint(cm, 100, 200, 100, 200, 0.0)
    paint(cm, 140, 160, 140, 160, -2.8)
    portals = find_portals(cm, floor_y=0.0)
    assert len(portals) == 1 and not portals[0].going_up
    assert portals[0].delta_y == pytest.approx(-2.8, abs=0.05)


def test_split_level_is_not_a_portal():
    """A 0.9 m step is a sunken room, not another storey."""
    cm = blank()
    paint(cm, 100, 200, 100, 200, 0.0)
    paint(cm, 140, 160, 140, 160, 0.9)
    assert find_portals(cm, floor_y=0.0) == []


def test_far_storey_is_rejected():
    cm = blank()
    paint(cm, 100, 200, 100, 200, 0.0)
    paint(cm, 140, 160, 140, 160, 8.0)          # two floors up an atrium
    assert find_portals(cm, floor_y=0.0) == []


def test_speckle_is_rejected():
    cm = blank()
    paint(cm, 100, 200, 100, 200, 0.0)
    cm.height[150, 150] = 2.8
    assert find_portals(cm, floor_y=0.0, min_cells=20) == []


def test_patches_of_one_opening_merge():
    cm = blank()
    paint(cm, 100, 200, 100, 200, 0.0)
    paint(cm, 140, 152, 140, 152, 2.8)
    paint(cm, 154, 166, 140, 152, 2.8)          # same stairwell, two glimpses
    assert len(find_portals(cm, floor_y=0.0, merge_m=2.0)) == 1


def test_up_and_down_portals_are_kept_separate():
    cm = blank()
    paint(cm, 100, 200, 100, 200, 0.0)
    paint(cm, 140, 160, 140, 160, 2.8)
    paint(cm, 140, 160, 300, 320, -2.8)
    portals = find_portals(cm, floor_y=0.0)
    assert {p.going_up for p in portals} == {True, False}


def test_portals_need_the_height_layer():
    cm = Costmap2D(resolution=RES)
    assert find_portals(cm, floor_y=0.0) == []


def test_portal_is_relative_to_the_current_floor():
    """Standing upstairs, the ground floor is the portal."""
    cm = blank()
    paint(cm, 100, 200, 100, 200, 2.8)
    paint(cm, 140, 160, 140, 160, 0.0)
    portals = find_portals(cm, floor_y=2.8)
    assert len(portals) == 1 and not portals[0].going_up


# -------------------------------------------------------------------- the gate


def policy(**kw):
    return FloorSwitchPolicy(max_steps=500, **kw)


def test_no_switch_while_a_near_frontier_remains():
    """ASCENT's condition: storeys are only considered when this one is done."""
    assert not policy().may_switch(step=100, best_path_cost=2.0)


def test_switch_allowed_when_everything_left_is_far():
    assert policy().may_switch(step=100, best_path_cost=12.0)


def test_switch_allowed_when_nothing_is_selectable():
    assert policy().may_switch(step=100, best_path_cost=None)


def test_no_switch_too_early():
    assert not policy().may_switch(step=10, best_path_cost=None)


def test_no_switch_too_late():
    """Late in the budget there is no time to recover from a wrong floor."""
    assert not policy().may_switch(step=400, best_path_cost=None)


def test_min_interval_prevents_oscillation():
    p = policy()
    assert p.may_switch(step=100, best_path_cost=None)
    p.note_switch(100)
    assert not p.may_switch(step=120, best_path_cost=None)
    assert p.may_switch(step=160, best_path_cost=None)


def test_gate_is_closed_by_default_thresholds_on_a_busy_floor():
    """Sanity: a normal mid-episode state with work left never switches."""
    p = policy()
    assert not any(p.may_switch(step=s, best_path_cost=1.5) for s in range(0, 350, 10))
