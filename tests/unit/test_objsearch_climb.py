"""S87: the on-stairs controller (`agent.search.climb_mount_m`,
`climb_heading_memory`, `climb_carrot_free`). Off, each is the reference."""
from __future__ import annotations

import numpy as np

from navigation.search.climb import HeadingMemory, free_carrot_offset

from .test_ascentnav_stairs import DEPTH, _paint_stairs, _px
from .test_objsearch_live import _agent


# ------------------------------------------------------------ climb_mount_m

def _on_the_flight_near_its_centroid(a):
    om = _paint_stairs(a, xy=(2.0, 0.0), r=20)               # a flight +-1 m about (2, 0)
    a.stairs.start_navigating(om, 1)
    for _ in range(2):
        a.stairs.pre_update(a, om, np.array([2.6, 0.0]), _px(a, (2.6, 0.0)))
    return om


def test_standing_on_the_flight_near_its_centroid_starts_the_climb():
    a = _agent(climb_mount_m=1.0)
    _on_the_flight_near_its_centroid(a)
    assert a.stairs.reach_stair and a.stairs.reach_stair_centroid
    assert a.stats["climb_mounted"] == 1


def test_without_the_flag_the_centroid_must_be_reached():
    a = _agent()
    _on_the_flight_near_its_centroid(a)
    assert a.stairs.reach_stair and not a.stairs.reach_stair_centroid


def test_off_the_flight_the_mount_rule_does_not_fire():
    a = _agent(climb_mount_m=1.0)
    om = _paint_stairs(a, xy=(2.0, 0.0), r=6)                # +-0.3 m: 2.6 is off it
    a.stairs.start_navigating(om, 1)
    a.stairs.reach_stair = True
    a.stairs.pre_update(a, om, np.array([2.6, 0.0]), _px(a, (2.6, 0.0)))
    assert not a.stairs.reach_stair_centroid


def _mounted(a):
    om = _on_the_flight_near_its_centroid(a)
    assert a.stairs.reach_stair_centroid and a.stairs.mounted_y is not None
    return om


def test_leaving_the_flight_at_the_mount_height_undoes_the_mount():
    a = _agent(climb_mount_m=1.0)
    om = _mounted(a)
    a.stairs.pre_update(a, om, np.array([3.6, 0.0]), _px(a, (3.6, 0.0)))   # off the mask, same height
    assert a.stairs.climbing and a.stairs.reach_stair and not a.stairs.reach_stair_centroid
    assert a.stats["climb_unmounted"] == 1 and "climb_ok" not in a.stats
    # one mount per attempt: back on the flight, the reference's 0.3 m rule again
    a.stairs.pre_update(a, om, np.array([2.6, 0.0]), _px(a, (2.6, 0.0)))
    assert not a.stairs.reach_stair_centroid and a.stats["climb_mounted"] == 1


def test_leaving_the_flight_having_climbed_is_an_arrival():
    a = _agent(climb_mount_m=1.0)
    a._floors.append(a._new_floor())                        # the storey the flight leads to
    om = _mounted(a)
    a._base_y = a.stairs.mounted_y + 0.6
    a.stairs.pre_update(a, om, np.array([3.6, 0.0]), _px(a, (3.6, 0.0)))
    assert not a.stairs.climbing and a.stats.get("climb_ok") == 1
    assert "climb_unmounted" not in a.stats


# --------------------------------------------------------- HeadingMemory

T30 = np.radians(30)


def test_it_steps_when_facing_the_open_heading_nearest_the_carrot():
    m = HeadingMemory()
    assert m.step([0, 0], 0.0, 0.0, 1, bearing=0.1) == "move_forward"


def test_it_turns_the_short_way_and_holds_its_choice_while_turning():
    m = HeadingMemory()
    assert m.step([0, 0], 0.0, 0.0, 1, bearing=2 * T30) == "turn_left"
    # the carrot, redrawn in the new view, now says right: the choice holds
    assert m.step([0, 0], T30, 0.0, 1, bearing=-2 * T30) == "turn_left"
    assert m.step([0, 0], 2 * T30, 0.0, 1, bearing=-2 * T30) == "move_forward"


def test_a_refused_step_closes_that_heading_at_that_spot_only():
    m = HeadingMemory()
    assert m.step([0, 0], 0.0, 0.0, 1, bearing=0.0) == "move_forward"
    # did not move: heading 0 is closed here, the next nearest is chosen
    a = m.step([0, 0], 0.0, 0.0, 1, bearing=0.0)
    assert a in ("turn_left", "turn_right") and m.refused == 1
    # one cell over, heading 0 is open again: a fresh choice there steps straight
    m.goal = None
    assert m.step([0.5, 0], 0.0, 0.0, 1, bearing=0.0) == "move_forward"
    assert m._bin(0.0) in m.closed[m._cell([0, 0])] and m._cell([0.5, 0]) not in m.closed


def test_a_step_down_on_the_way_up_turns_round_and_closes_it_at_both_spots():
    m = HeadingMemory()
    m.step([0, 0], 0.0, 0.0, 1, bearing=0.0)
    a = m.step([0.25, 0], 0.0, -0.15, 1, bearing=0.0)          # moved, but downhill
    assert m.wrong_way == 1 and a in ("turn_left", "turn_right")
    assert m.goal == m.n // 2                                  # the opposite heading, not the carrot's
    assert 0 in m.closed[m._cell([0, 0])] and 0 in m.closed[m._cell([0.25, 0])]
    m2 = HeadingMemory()
    m2.step([0, 0], 0.0, 0.0, 2, bearing=0.0)
    m2.step([0.25, 0], 0.0, -0.15, 2, bearing=0.0)             # downhill on the way down is right
    assert m2.wrong_way == 0


def test_a_step_that_climbed_keeps_its_heading_whatever_the_carrot_says():
    m = HeadingMemory()
    m.step([0, 0], 0.0, 0.0, 1, bearing=0.0)
    # up 0.18 m; the carrot, redrawn, now points 90 degrees left
    assert m.step([0.25, 0], 0.0, 0.18, 1, bearing=np.radians(90)) == "move_forward"
    assert m.kept == 1


def test_a_level_step_chooses_again_from_the_carrot():
    m = HeadingMemory()
    m.step([0, 0], 0.0, 0.0, 1, bearing=0.0)
    assert m.step([0.25, 0], 0.0, 0.0, 1, bearing=np.radians(90)) == "turn_left"
    assert m.kept == 0


def test_every_heading_refused_starts_the_spot_afresh():
    m = HeadingMemory()
    m.closed[m._cell([0, 0])] = set(range(m.n))
    assert m.step([0, 0], 0.0, 0.0, 1, bearing=0.0) == "move_forward"


# --------------------------------------------------- free_carrot_offset

def _scene(wall_x=None, far_cols=None, w=64, h=48):
    """A level camera 0.88 m up looking along +z (OpenCV axes = world axes
    with y flipped), depth 3 m everywhere unless a wall is in front."""
    fx = w / (2 * np.tan(np.radians(79) / 2))
    depth = np.full((h, w), 3.0, np.float32)
    if far_cols is not None:
        depth[:, far_cols] = 6.0
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, 1.0])                     # y down in the camera = y up in the world
    T[1, 3] = 0.88
    if wall_x is not None:                                     # a wall 0.5 m ahead over these columns
        depth[:, wall_x] = 0.5
    return depth, T, fx


def test_with_nothing_in_the_way_it_is_the_reference_farthest_pixel():
    depth, T, fx = _scene(far_cols=slice(48, 52))
    off = free_carrot_offset(depth, T, fx, fx, 0.0, np.radians(79))
    u = 50.0
    assert off is not None and abs(off - (u - 32) / 32 * np.radians(79) / 2) < np.radians(4)


def test_a_far_pixel_behind_a_near_wall_is_not_chosen():
    depth, T, fx = _scene(far_cols=slice(48, 52))
    depth[:, 40:64] = np.minimum(depth[:, 40:64], 0.5)         # right half walled off at 0.5 m
    depth[:, 48:52] = 6.0                                       # ...seen through a gap too narrow to pass
    depth[:, 44:48] = 0.5
    depth[:, 52:56] = 0.5
    off = free_carrot_offset(depth, T, fx, fx, 0.0, np.radians(79))
    assert off is not None and off < np.radians(5)              # not to the right


def test_everything_blocked_falls_back():
    depth, T, fx = _scene()
    depth[:] = 0.4
    assert free_carrot_offset(depth, T, fx, fx, 0.0, np.radians(79)) is None


# --------------------------------------------------------- in the agent

def test_flags_off_build_nothing():
    a = _agent()
    assert a.stairs.heading_memory is None and a.stairs.mount_m == 0 and not a._climb_carrot_free


def test_the_flight_is_walked_by_heading_when_asked():
    a = _agent(climb_heading_memory=True)
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a.stairs.start_navigating(om, 1)
    a.stairs.reach_stair = a.stairs.reach_stair_centroid = True
    action = a._climb_stair(DEPTH, np.array([2.0, 0.0]), 0.0, _px(a, (2.0, 0.0)))
    assert action == "move_forward" and a.stats["climb_heading_steps"] == 1
