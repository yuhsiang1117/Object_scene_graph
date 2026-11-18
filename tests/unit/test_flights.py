"""A staircase read off the height layer, so the target is its foot.

Stage 4's detector asked which cells are steppable by their local height step
and fragmented every flight into riser edges, because a tread's interior is
flat. This asks which cells sit strictly between this floor and the next and
are contiguous, and keeps a run only if its heights span a metre: a table is a
plateau, a flight joins two storeys.

The detector's `stairs` mask, seen from below, put its target 2.7 m beside the
true foot on 00821, and twelve climbs steered at it rose 0.00 m. The lowest cell
of an intermediate-height run IS the foot.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.mapping.costmap import FREE, Costmap2D
from osg.mapping.stairs import find_flights

FLOOR_Y = 0.0


def _costmap(size_m=12.0, res=0.05):
    cm = Costmap2D(resolution=res, size_m=size_m, track_height=True)
    cm.grid[:] = FREE
    return cm


def _paint(cm, x0, x1, z0, z1, y):
    r0, c0 = cm.world_to_grid(np.array([x0, z0]))
    r1, c1 = cm.world_to_grid(np.array([x1, z1]))
    r0, r1 = sorted((int(r0), int(r1))); c0, c1 = sorted((int(c0), int(c1)))
    cm.height[r0:r1 + 1, c0:c1 + 1] = y


def _staircase(cm, x_foot=1.0, z=2.0, treads=12, tread_m=0.28, riser_m=0.17, width_m=1.0):
    """Treads rising along +x from `x_foot`, 1 m wide, contiguous."""
    for i in range(treads):
        x0 = x_foot + i * tread_m
        _paint(cm, x0, x0 + tread_m, z, z + width_m, FLOOR_Y + (i + 1) * riser_m)


def test_a_flight_is_found_and_its_foot_is_the_lowest_tread():
    cm = _costmap()
    _staircase(cm)
    flights = find_flights(cm, FLOOR_Y, new_level_m=2.6, min_span_m=1.0, min_cells=50)
    assert len(flights) == 1
    f = flights[0]
    assert f.kind == "up"
    assert f.span_m >= 1.0
    assert abs(f.foot_xy[0] - 1.0) < 0.35, "foot at the first tread, not mid-flight"
    assert abs(f.foot_xy[1] - 2.5) < 0.6
    assert f.top_xy[0] > f.foot_xy[0] + 2.0


def test_a_table_is_a_plateau_and_not_a_flight():
    cm = _costmap()
    _paint(cm, 1.0, 3.0, 2.0, 4.0, FLOOR_Y + 0.75)  # a 2x2 m table top
    assert find_flights(cm, FLOOR_Y, min_cells=50) == []


def test_a_descent_is_read_from_the_treads_below():
    cm = _costmap()
    for i in range(12):
        x0 = 1.0 + i * 0.28
        _paint(cm, x0, x0 + 0.28, 2.0, 3.0, FLOOR_Y - (i + 1) * 0.17)
    flights = find_flights(cm, FLOOR_Y, new_level_m=2.6, min_span_m=1.0, min_cells=50)
    assert len(flights) == 1 and flights[0].kind == "down"
    # The mouth on THIS floor is the highest tread, nearest the lip.
    assert abs(flights[0].foot_xy[0] - 1.0) < 0.35


def test_a_flight_on_another_storey_is_not_this_floors_flight():
    """Cells a whole storey up are the next floor, not treads."""
    cm = _costmap()
    _paint(cm, 1.0, 4.0, 2.0, 4.0, FLOOR_Y + 2.9)
    assert find_flights(cm, FLOOR_Y, new_level_m=2.6, min_cells=50) == []


def test_a_huge_sloped_region_is_capped():
    cm = _costmap(size_m=12.0)
    # a gentle ramp over the whole map: thousands of cells spanning a metre
    n = cm.grid.shape[0]
    cm.height[:] = np.linspace(0.3, 1.5, n)[:, None]
    assert find_flights(cm, FLOOR_Y, min_cells=50, max_cells=4000) == []


def test_nothing_without_a_height_layer():
    cm = Costmap2D(resolution=0.05, size_m=6.0, track_height=False)
    assert find_flights(cm, FLOOR_Y) == []


# -------------------------------------------- not stopping on a half-landing

def test_a_landing_does_not_become_a_storey_of_its_own():
    """Measured on 00808: a descent of 2.3 m out of 3.2 committed the landing at
    y=0.946 as a floor, which ended the climb one flight short. A landing is off
    every known level and roomy enough to walk 2.5 m across, so the
    horizontal-run route creates a level there."""
    from osg.mapping.floors import FloorEstimator

    def _walk(on_flight):
        est = FloorEstimator(camera_height=0.88, new_level_m=1.8, level_tol_m=0.35,
                             merge_m=0.6, min_dwell_steps=3, min_horizontal_run_m=2.5)
        est.update(0.88 + 2.86, step=0, xy=[0.0, 0.0])          # upstairs, floor 0
        for i in range(40):                                      # across a landing
            est.update(0.88 + 0.95, step=1 + i, xy=[0.1 * i, 0.0], on_flight=on_flight)
        return est

    assert len(_walk(False).levels) == 2, "the landing became a floor"
    held = _walk(True)
    assert len(held.levels) == 1
    assert held.suppressed_levels > 0


def test_arriving_on_a_floor_the_stack_knows_still_commits():
    """The suppression is about creating a NEW level. Reaching a storey the
    agent has already stood on is a real arrival and must still register."""
    from osg.mapping.floors import FloorEstimator

    est = FloorEstimator(camera_height=0.88, new_level_m=1.8, level_tol_m=0.35,
                         merge_m=0.6, min_dwell_steps=3, min_horizontal_run_m=2.5)
    est.update(0.88 + 0.0, step=0, xy=[0.0, 0.0])
    for i in range(30):                                # walk up to a second storey
        est.update(0.88 + 2.9, step=1 + i, xy=[0.1 * i, 0.0])
    assert len(est.levels) == 2
    upstairs = est.current
    for i in range(30):                                # come back down, on a flight
        est.update(0.88 + 0.0, step=40 + i, xy=[0.0, 0.1 * i], on_flight=True)
    assert est.current != upstairs, "returning to a known floor must still commit"


def test_a_landing_is_not_folded_into_the_floor_below():
    """`_refine` folds the samples nearest a level into its height. A landing is
    near enough to the floor below to be folded in: measured on 00808, a descent
    that stopped 0.89 m above floor 0 dragged floor 0's estimate up to meet it,
    so the agent "arrived" on a storey it was not standing on."""
    from osg.mapping.floors import FloorEstimator

    def _descend(on_flight):
        est = FloorEstimator(camera_height=0.88, new_level_m=1.8, level_tol_m=0.35,
                             merge_m=0.6, min_dwell_steps=3, min_horizontal_run_m=2.5)
        est.update(0.88 + 0.06, step=0, xy=[0.0, 0.0])      # floor 0, at 0.06
        for i in range(20):
            est.update(0.88 + 2.86, step=1 + i, xy=[0.1 * i, 0.0])   # up to floor 1
        for i in range(40):                                  # stop on a landing
            est.update(0.88 + 0.95, step=40 + i, xy=[0.1 * i, 1.0], on_flight=on_flight)
        return est

    drifted = _descend(False).levels[0]
    held = _descend(True).levels[0]
    assert held == pytest.approx(0.06, abs=0.05), "floor 0 moved to meet the landing"
    assert abs(drifted - 0.06) > abs(held - 0.06)


def _descent_seen_from_afar(cm, hidden=4):
    """Treads descending along +x from a lip at x = 1.0; the first `hidden`
    were never seen (behind the lip from where the agent stood)."""
    for i in range(hidden, 12):
        x0 = 1.0 + i * 0.28
        _paint(cm, x0, x0 + 0.28, 2.0, 3.0, FLOOR_Y - (i + 1) * 0.17)


def test_a_descent_seen_from_afar_has_its_mouth_down_the_flight():
    from osg.mapping.stairs import mouth_xy

    cm = _costmap()
    _descent_seen_from_afar(cm)
    (f,) = find_flights(cm, FLOOR_Y, new_level_m=2.6, min_span_m=1.0, min_cells=50)
    assert f.kind == "down"
    # The highest tread SEEN is 1.1 m down the flight: the pursuit goal the
    # step-3 arm drove to, under this storey.
    assert mouth_xy(f)[0] > 2.0


def test_the_descent_mouth_is_extrapolated_up_the_slope_to_the_floor():
    from osg.mapping.stairs import descent_mouth_xy

    cm = _costmap()
    _descent_seen_from_afar(cm)
    (f,) = find_flights(cm, FLOOR_Y, new_level_m=2.6, min_span_m=1.0, min_cells=50)
    m = descent_mouth_xy(f, FLOOR_Y, cm.resolution)
    assert m is not None
    assert abs(m[0] - 1.0) < 0.3, f"the lip is at x = 1.0, got {m}"
    assert abs(m[1] - 2.5) < 0.4


def test_no_extrapolated_mouth_for_a_flat_blob_or_an_ascent():
    from osg.mapping.stairs import Flight, descent_mouth_xy

    rc = np.array([[r, c] for r in range(10) for c in range(10)])
    flat = Flight(kind="down", cells_rc=rc, heights=np.full(len(rc), -1.0),
                  foot_xy=np.zeros(2), top_xy=np.zeros(2), span_m=0.0)
    assert descent_mouth_xy(flat, 0.0, 0.05) is None
    up = Flight(kind="up", cells_rc=rc, heights=rc[:, 0] * 0.03,
                foot_xy=np.zeros(2), top_xy=np.zeros(2), span_m=0.3)
    assert descent_mouth_xy(up, 0.0, 0.05) is None
