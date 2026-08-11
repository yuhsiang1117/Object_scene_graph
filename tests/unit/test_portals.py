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


# ------------------------------- semantic stair + descent portals (ASCENT-style)


from osg.mapping.portals import find_descent_portals, portals_from_stair_points
from osg.mapping.stairs import descent_points, stair_points_from_masks
from osg.core.types import CameraIntrinsics, Detection, FrameData


def _stair_cloud(x0=3.0, z0=0.0, n=400, rise=1.2):
    """Points along a flight climbing `rise` m over ~2 m of run."""
    t = np.linspace(0, 1, n)
    return np.stack([x0 + 2.0 * t, t * rise, np.full(n, z0)], axis=1)


def test_a_partially_seen_flight_extrapolates_to_a_storey():
    """From the bottom you see the first metre, not the landing -- so a 1.2 m
    visible rise must still aim a full storey up."""
    p = portals_from_stair_points(_stair_cloud(rise=1.2), floor_y=0.0)
    assert len(p) == 1 and p[0].going_up
    assert p[0].target_y == pytest.approx(2.8, abs=0.01)


def test_a_fully_seen_flight_uses_the_observed_top():
    p = portals_from_stair_points(_stair_cloud(rise=3.1), floor_y=0.0)
    assert p[0].target_y == pytest.approx(3.1, abs=0.01)


def test_a_descending_flight_aims_downward():
    p = portals_from_stair_points(_stair_cloud(rise=-1.0), floor_y=0.0)
    assert len(p) == 1 and not p[0].going_up
    assert p[0].target_y == pytest.approx(-2.8, abs=0.01)


def test_flat_floor_points_are_not_a_stair_portal():
    flat = np.stack([np.linspace(0, 2, 400), np.zeros(400), np.zeros(400)], axis=1)
    assert portals_from_stair_points(flat, floor_y=0.0) == []


def test_too_few_points_are_ignored():
    assert portals_from_stair_points(_stair_cloud(n=10), floor_y=0.0) == []


def test_two_separate_staircases_do_not_merge():
    both = np.concatenate([_stair_cloud(x0=3.0, z0=0.0), _stair_cloud(x0=3.0, z0=20.0)])
    assert len(portals_from_stair_points(both, floor_y=0.0)) == 2


def test_descent_portal_extrapolates_a_shallow_drop():
    """Only the first half-metre of a descending flight is visible from a few
    metres back, so the drop must not be trusted as measured."""
    pts = np.stack([np.full(200, 2.0), np.full(200, -0.5), np.zeros(200)], axis=1)
    p = find_descent_portals(pts, floor_y=0.0)
    assert len(p) == 1 and not p[0].going_up
    assert p[0].target_y == pytest.approx(-2.8, abs=0.01)


# ------------------------------------------------ mask -> points plumbing


def _frame_with_depth(d=2.0, size=64):
    T = np.eye(4)
    return FrameData(
        frame_id=0, rgb=np.zeros((size, size, 3), np.uint8),
        depth=np.full((size, size), d, np.float32), T_wc=T,
        intrinsics=CameraIntrinsics.from_hfov(79.0, size, size),
    )


def _det(label, mask, score=0.6):
    return Detection(label=label, score=score,
                     bbox_xyxy=np.array([0.0, 0.0, 1.0, 1.0]), mask=mask)


def test_only_stair_labels_contribute_points():
    f = _frame_with_depth()
    m = np.zeros((64, 64), bool); m[10:40, 10:40] = True
    assert stair_points_from_masks(f, [_det("chair", m)]).shape[0] == 0
    assert stair_points_from_masks(f, [_det("stairs", m)]).shape[0] > 0


def test_low_score_and_small_masks_are_rejected():
    f = _frame_with_depth()
    big = np.zeros((64, 64), bool); big[10:40, 10:40] = True
    small = np.zeros((64, 64), bool); small[10:12, 10:12] = True
    assert stair_points_from_masks(f, [_det("stairs", big, score=0.05)]).shape[0] == 0
    assert stair_points_from_masks(f, [_det("stairs", small)]).shape[0] == 0


def test_no_detections_is_empty_not_an_error():
    assert stair_points_from_masks(_frame_with_depth(), []).shape == (0, 3)


def test_descent_points_keeps_only_below_floor():
    """The occupancy band stops 0.3 m under the agent's feet; this is the only
    way a downward opening is ever seen."""
    f = _frame_with_depth()
    below = descent_points(f, floor_y=5.0, min_drop_m=0.4)   # everything is far below
    above = descent_points(f, floor_y=-5.0, min_drop_m=0.4)  # everything is above
    assert below.shape[0] > 0 and above.shape[0] == 0
