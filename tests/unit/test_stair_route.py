"""The stair route (`mapping/stair_route.py`): a walk through the height layer.

Synthetic height layers, each built so the answer is known: a straight flight,
a switchback whose two legs sit side by side a storey-half apart, furniture a
riser too tall to step onto, and a descent that must stop at the storey asked
for rather than run on to the next.
"""
from __future__ import annotations

import numpy as np

from osg.mapping.costmap import FREE, OCCUPIED, Costmap2D
from osg.mapping.stair_route import plan_stair_route


def _map():
    cm = Costmap2D(resolution=0.05, size_m=30.0, track_height=True, track_top=True)
    return cm


def _paint(cm, x0, x1, z0, z1, height, grid=FREE, top=False, floor=None):
    """Set a rectangle's height (a constant, or a function of (x, z)).

    With `floor`, the occupancy is what a floor-relative grid would hold:
    FREE within 0.15 m of that floor, OCCUPIED above it (a rising tread is an
    obstacle in the band), FREE below it (rays pass over a falling one).
    """
    r0, c0 = cm.world_to_grid(np.array([x0, z0]))
    r1, c1 = cm.world_to_grid(np.array([x1, z1]))
    rr, cc = np.mgrid[r0:r1, c0:c1]
    xz = cm.origin[None, None, :] + (np.stack([rr, cc], axis=-1) + 0.5) * cm.resolution
    h = height(xz[..., 0], xz[..., 1]) if callable(height) else np.full(rr.shape, float(height))
    (cm.height_top if top else cm.height)[r0:r1, c0:c1] = h
    if floor is not None:
        cm.grid[r0:r1, c0:c1] = np.where(h - float(floor) > 0.15, OCCUPIED, FREE)
    else:
        cm.grid[r0:r1, c0:c1] = grid


def _stairs(rise_per_m):
    """A stepped flight: 0.17 m risers every 0.28 m along x from x=0."""
    step_h, step_d = 0.17, 0.28
    return lambda x, z: np.floor(np.clip(x, 0, None) / step_d) * step_h * rise_per_m / (step_h / step_d)


def test_a_straight_flight_is_walked_to_the_top():
    cm = _map()
    _paint(cm, -4, 0, -2, 2, 0.0)                               # lower floor
    _paint(cm, 0, 5, -0.5, 0.5, lambda x, z: np.floor(x / 0.28) * 0.17, floor=0.0)
    _paint(cm, 5, 9, -2, 2, 3.06)                               # upper floor
    route = plan_stair_route(cm, floor_y=0.0, agent_xy=(-2.0, 0.0), standing_y=0.0,
                             direction=+1, target_y=3.06)
    assert route is not None and route.reaches_target
    assert route.xy[-1][0] > 5.0 and route.heights[-1] > 2.9
    assert np.all(np.abs(route.xy[:, 1]) < 0.6)                 # stayed on the treads
    assert 1.5 < route.leaves_floor_after_m(0.0) < 2.6          # walked ~2 m to the foot


def test_a_switchback_goes_round_its_landing():
    cm = _map()
    _paint(cm, -4, 0, -2, 1.0, 0.0)                              # lower floor, south of the stair
    _paint(cm, 0, 2.6, 0, 1.0, lambda x, z: np.minimum(np.floor(x / 0.28) * 0.17, 1.53), floor=0.0)
    _paint(cm, 2.6, 4.0, 0, 2.2, 1.53, floor=0.0)            # the landing
    # The second leg runs back west, BESIDE the first: 1.2-2.2 in z.
    _paint(cm, 0.08, 2.6, 1.2, 2.2, lambda x, z: 1.53 + np.minimum(
        np.floor((2.6 - x) / 0.28) * 0.17, 1.53), floor=0.0)
    _paint(cm, 0, 2.6, 1.0, 1.2, 0.0)                            # the gap between the legs: the floor below
    _paint(cm, -4, 0.08, 1.2, 4.0, 3.06)                         # upper floor, north-west
    route = plan_stair_route(cm, floor_y=0.0, agent_xy=(-1.0, 0.5), standing_y=0.0,
                             direction=+1, target_y=3.06)
    assert route is not None and route.reaches_target
    assert route.xy[:, 0].max() > 2.6                            # went round by the landing
    assert route.xy[-1][1] > 1.2 and abs(route.heights[-1] - 3.06) < 0.05


def test_furniture_a_riser_too_tall_is_not_a_staircase():
    cm = _map()
    _paint(cm, -4, 4, -4, 4, 0.0)
    _paint(cm, 1, 3, -1, 1, 0.45, grid=OCCUPIED)                 # a sofa
    _paint(cm, 1.5, 2.5, -0.5, 0.5, 1.6, grid=OCCUPIED)          # a cabinet on it
    assert plan_stair_route(cm, 0.0, (-2.0, 0.0), 0.0, +1, target_y=3.0) is None


def test_a_wrong_storey_height_still_ends_on_the_floor():
    cm = _map()
    _paint(cm, -4, 0, -2, 2, 3.06)
    _paint(cm, 0, 5, -0.5, 0.5, lambda x, z: 3.06 - np.minimum(np.floor(x / 0.28) * 0.17, 2.9),
           floor=3.06)
    _paint(cm, 5, 9, -2, 2, 0.16)
    # The prior map put this storey at 0.64 (00800): the floor, not a tread.
    route = plan_stair_route(cm, 3.06, (-2.0, 0.0), 3.06, -1, target_y=0.64)
    assert route is not None and route.reaches_target
    assert abs(route.heights[-1] - 0.16) < 0.05


def test_a_descent_stops_at_the_storey_asked_for():
    cm = _map()
    _paint(cm, -4, 0, -2, 2, 6.12)                               # top storey
    _paint(cm, 0, 5, -0.5, 0.5, lambda x, z: 6.12 - np.floor(x / 0.28) * 0.17, floor=6.12)
    _paint(cm, 5, 8, -2, 2, 3.06)                                # middle storey
    _paint(cm, 8, 13, -0.5, 0.5, lambda x, z: 3.06 - np.floor((x - 8) / 0.28) * 0.17, floor=6.12)
    _paint(cm, 13, 15, -2, 2, 0.0)                               # bottom storey
    route = plan_stair_route(cm, floor_y=6.12, agent_xy=(-2.0, 0.0), standing_y=6.12,
                             direction=-1, target_y=3.06, radius_m=20.0)
    assert route is not None and route.reaches_target
    assert abs(route.heights[-1] - 3.06) < 0.3 and route.xy[-1][0] < 8.5


def test_a_wall_on_this_floor_is_not_walked_through():
    cm = _map()
    _paint(cm, -4, 6, -3, 3, 0.0)
    _paint(cm, 1.0, 1.1, -3, 3, 0.0, grid=OCCUPIED)              # a wall at the floor, x = 1
    _paint(cm, 3, 8, 2.0, 3.0, lambda x, z: np.floor((x - 3) / 0.28) * 0.17, floor=0.0)
    assert plan_stair_route(cm, 0.0, (-2.0, 0.0), 0.0, +1, target_y=3.0) is None


def test_open_treads_seen_from_below_are_climbed_on_the_upper_level():
    """The minimum keeps the floor beneath open treads; the top level has them."""
    cm = _map()
    for top in (False, True):                                    # floors are in both layers
        _paint(cm, -4, 0, -2, 2, 0.0, top=top)
        _paint(cm, 5, 9, -2, 2, 3.06, top=top)
    _paint(cm, 0, 5, -0.5, 0.5, 0.0)                             # min: the floor seen UNDER open treads
    _paint(cm, 0, 5, -0.5, 0.5, lambda x, z: np.floor(x / 0.28) * 0.17, floor=0.0, top=True)
    route = plan_stair_route(cm, 0.0, (-2.0, 0.0), 0.0, +1, target_y=3.06)
    assert route is not None and route.heights[-1] > 2.9


def test_a_slab_over_a_descent_does_not_hide_its_treads():
    """From the storey above, the slab over the lower flight wins the top layer."""
    cm = _map()
    for top in (False, True):
        _paint(cm, -4, 0, -2, 2, 3.06, top=top)
        _paint(cm, 5, 9, -2, 2, 0.0, top=top)
    _paint(cm, 0, 5, -0.5, 0.5, lambda x, z: 3.06 - np.floor(x / 0.28) * 0.17, floor=3.06)
    _paint(cm, 0, 5, -0.5, 0.5, 3.06, top=True)                 # top: the slab seen over the treads
    _paint(cm, 0, 2, -0.5, 0.5, lambda x, z: 3.06 - np.floor(x / 0.28) * 0.17, top=True)
    route = plan_stair_route(cm, 3.06, (-2.0, 0.0), 3.06, -1, target_y=0.0)
    assert route is not None and route.reaches_target and abs(route.heights[-1]) < 0.05


def test_record_top_keeps_the_highest_point_below_the_camera():
    cm = _map()
    pts = np.array([[1.0, 0.0, 1.0], [1.0, 0.5, 1.0], [1.0, 2.0, 1.0]])
    cm.record_top(pts, cap_y=1.2)
    rc = cm.world_to_grid(np.array([1.0, 1.0]))
    assert abs(float(cm.height_top[rc[0], rc[1]]) - 0.5) < 1e-6
    cm.ensure_contains(np.array([40.0, 40.0]))                   # grows with the grid
    rc = cm.world_to_grid(np.array([1.0, 1.0]))
    assert cm.height_top.shape == cm.grid.shape
    assert abs(float(cm.height_top[rc[0], rc[1]]) - 0.5) < 1e-6


def _descent_from_the_treads(cm):
    """Standing 2.2 m down a flight that falls along +x from a lip at x=0,
    as the climb maps it: the treads in `height_top` only (the lowest layer is
    not recorded on a flight), the cells FREE where rays from the storey above
    passed over, the tread just ahead unseen, and the slab of the storey above
    overhanging the last treads in the lowest layer."""
    fall = lambda x, z: -np.floor(np.clip(x, 0, None) / 0.28 + 1) * 0.17      # noqa: E731
    _paint(cm, -3, 0, -1, 1, 0.0)                                             # storey above
    _paint(cm, 0, 5.0, -0.5, 0.5, fall, top=True)
    cm.grid[cm.world_to_grid(np.array([0.0, -0.5]))[0]:cm.world_to_grid(np.array([5.0, 0.5]))[0],
            cm.world_to_grid(np.array([0.0, -0.5]))[1]:cm.world_to_grid(np.array([5.0, 0.5]))[1]] = FREE
    _paint(cm, 5.0, 8.0, -1, 1, -3.06, top=True)                              # storey below
    _paint(cm, 3.8, 5.0, -0.5, 0.5, 0.0)                                      # the slab overhead
    r0, c0 = cm.world_to_grid(np.array([3.75, -0.5]))
    r1, c1 = cm.world_to_grid(np.array([3.95, 0.5]))
    cm.height_top[r0:r1, c0:c1] = np.nan                                      # unseen just ahead
    return 3.6, float(fall(3.6, 0.0))


def test_a_descent_from_the_treads_reaches_the_storey_below():
    cm = _map()
    x, standing = _descent_from_the_treads(cm)
    route = plan_stair_route(cm, floor_y=0.0, agent_xy=(x, 0.0), standing_y=standing,
                             direction=-1, target_y=-3.06, gap_fill_m=0.8)
    assert route is not None and route.reaches_target
    assert route.xy[-1][0] > 5.0 and abs(route.heights[-1] + 3.06) < 0.05


def test_without_the_stairs_handling_the_flight_ahead_is_paved_at_standing_height():
    cm = _map()
    x, standing = _descent_from_the_treads(cm)
    route = plan_stair_route(cm, floor_y=0.0, agent_xy=(x, 0.0), standing_y=standing,
                             direction=-1, target_y=-3.06)
    assert route is None or not route.reaches_target


def test_a_walk_toward_a_known_mouth_ends_beside_it_not_at_the_far_end():
    """`toward_xy`: the climb's own goal (a stair mouth) when no route onto the
    stairs is in view; without it, the farthest point at this level."""
    cm = _map()
    _paint(cm, -4, 4, -4, 4, 0.0)
    mouth = np.array([1.5, 0.5])
    to_mouth = plan_stair_route(cm, floor_y=0.0, agent_xy=(-1.0, 0.0), standing_y=0.0,
                                direction=-1, target_y=-3.0, explore=True, toward_xy=mouth)
    far = plan_stair_route(cm, floor_y=0.0, agent_xy=(-1.0, 0.0), standing_y=0.0,
                           direction=-1, target_y=-3.0, explore=True)
    assert to_mouth is not None and far is not None
    assert np.linalg.norm(to_mouth.end_xy - mouth) < 0.2
    assert np.linalg.norm(far.end_xy - mouth) > 1.0
    assert not to_mouth.reaches_target


def test_no_walk_once_at_the_mouth():
    cm = _map()
    _paint(cm, -4, 4, -4, 4, 0.0)
    assert plan_stair_route(cm, floor_y=0.0, agent_xy=(1.4, 0.5), standing_y=0.0, direction=-1,
                            target_y=-3.0, explore=True, toward_xy=(1.5, 0.5)) is None
