"""The flight the carrot follows must reach the next storey, not 1.8 m.

`find_flights` keeps cells within `new_level_m` of the floor. On 00800 the
storeys are 3.0 m apart and the ramp pasted from the ASCENT map spans all of
it, so the flight stopped at 1.6 m and the climb fell back to the depth-ray
carrot from there. Measured (outputs/mf5_pass2_v13): the ascent stalled at
+1.97 m, the descent at -1.51 m, both at the end of the truncated flight, and
the agent then pushed into the balustrade for the rest of its 200-step budget.
"""
import numpy as np

from osg.agent.floor_policy import FloorPolicy
from osg.mapping.costmap import Costmap2D
from osg.mapping.stairs import find_flights

from .test_climb import GROUND, UPPER, make_cfg


def _policy(*, levels, from_levels=True):
    cfg = make_cfg()
    cfg.floor.enabled = True
    cfg.floor.new_level_m = 1.8
    cfg.floor.flight_span_from_levels = from_levels
    policy = FloorPolicy(cfg, stats={})
    keys = [GROUND, UPPER][: len(levels)]
    policy.estimator._levels = dict(zip(keys, levels))
    policy.estimator.current = GROUND
    return policy


def _ramp(rise_m: float, floor_y: float = 0.0, *, mask: bool = True) -> Costmap2D:
    """A 1 m wide ramp, 2 m long, climbing `rise_m` from `floor_y`.

    `mask` stamps the stair map over it, the way `apply_to_costmap` does for a
    pasted ASCENT staircase.
    """
    cm = Costmap2D(resolution=0.05, size_m=10.0, track_height=True)
    cm.stair_mask = np.zeros(cm.grid.shape, dtype=bool)
    r0, c0 = 100, 100
    for i in range(40):
        for j in range(20):
            cm.grid[r0 + i, c0 + j] = 0
            cm.height[r0 + i, c0 + j] = floor_y + rise_m * (i + 1) / 41.0
            cm.stair_mask[r0 + i, c0 + j] = mask
    return cm


def _wide(cm, floor_y, span, **kw):
    return find_flights(cm, floor_y, new_level_m=1.8, min_span_m=1.0, min_cells=50,
                        wide_span_m=span, wide_mask=cm.stair_mask, **kw)


def test_the_span_is_the_gap_to_the_next_known_storey():
    assert _policy(levels=(0.0, 3.0)).flight_span_m(0.0) == 3.0
    assert _policy(levels=(3.0, 0.0)).flight_span_m(3.0) == 3.0


def test_the_span_never_shrinks_below_the_constant():
    # A level 0.2 m off the current one is the estimator's noise, not a storey.
    assert _policy(levels=(0.0, 0.2)).flight_span_m(0.0) == 1.8
    assert _policy(levels=(0.0, 1.2)).flight_span_m(0.0) == 1.8


def test_without_other_levels_or_the_flag_the_constant_stands():
    assert _policy(levels=(0.0,)).flight_span_m(0.0) == 1.8
    assert _policy(levels=(0.0, 3.0), from_levels=False).flight_span_m(0.0) == 1.8


def test_a_full_height_ramp_is_one_full_height_flight():
    cm = _ramp(3.0)
    cut = find_flights(cm, 0.0, new_level_m=1.8, min_span_m=1.0, min_cells=50)
    full = _wide(cm, 0.0, _policy(levels=(0.0, 3.0)).flight_span_m(0.0))
    assert len(cut) == 1 and len(full) == 1
    assert cut[0].heights.max() < 1.7, "the 1.8 m constant should truncate the ramp"
    assert full[0].heights.max() > 2.7, "the flight still stops half-way up"
    assert full[0].n_cells > cut[0].n_cells
    assert np.isclose(full[0].heights.min(), cut[0].heights.min(), atol=0.1)


def test_the_same_from_the_top_storey_going_down():
    cm = _ramp(-3.0, floor_y=3.0)
    full = _wide(cm, 3.0, _policy(levels=(3.0, 0.0)).flight_span_m(3.0))
    assert len(full) == 1 and full[0].kind == "down"
    assert full[0].heights.min() < 0.3


def test_only_the_stair_map_may_cross_the_constant():
    """The widening must not admit the storey below, seen over the banister.

    Measured (outputs/mf5_pass2_v14 ep1): with the band widened for every
    cell, the lower storey's furniture tops sit 2.0-2.8 m under the upper
    floor, are intermediate-height, touch the stairwell in projection, and
    merged with the staircase into a 1460-cell component whose mouth was 3 m
    from the real one. The agent pursued it for 392 steps and never climbed.
    """
    cm = _ramp(3.0, mask=False)
    full = _wide(cm, 0.0, 3.0)
    assert len(full) == 1, "unmasked cells were extended past the constant"
    assert full[0].heights.max() < 1.7

    # And with the mask, the same cells do extend.
    assert _wide(_ramp(3.0), 0.0, 3.0)[0].heights.max() > 2.7


def test_the_mouth_is_still_the_end_on_this_storey():
    """`foot_xy` is what the pursuit drives to. Ascending it is the lowest
    tread; descending it must be the HIGHEST -- the mouth on this storey, not
    the bottom of the stairs a floor below."""
    up = _wide(_ramp(3.0), 0.0, 3.0)[0]
    down = _wide(_ramp(-3.0, floor_y=3.0), 3.0, 3.0)[0]
    assert up.heights[np.argmin(np.linalg.norm(
        np.stack([_ramp(3.0).grid_to_world(rc.astype(float)) for rc in up.cells_rc])
        - up.foot_xy, axis=1))] < 0.5
    cm = _ramp(-3.0, floor_y=3.0)
    xy = np.stack([cm.grid_to_world(rc.astype(float)) for rc in down.cells_rc])
    assert down.heights[int(np.argmin(np.linalg.norm(xy - down.foot_xy, axis=1)))] > 2.5
