"""`agent.search.*` hooked into `ascentnav`: the storey belief (M1) and the
climb judged by its height (M2b). `test_ascentnav_*.py` pin the off state; these
pin what each flag adds and that it adds nothing else."""
from __future__ import annotations

import numpy as np

from navigation.agent import AscentNavAgent
from osg.perception.detector import StubDetector

from .test_ascent_agent import _Driver
from .test_ascentnav_stairs import DEPTH, _paint_stairs, _px
from .test_nav_agent import _StubScorer


def _agent(driver=None, **search):
    from osg.core.config import OSGConfig
    from osg.exploration.async_scorer import AsyncScorer

    cfg = OSGConfig()
    cfg.agent.navigation = "pointnav"
    cfg.agent.initial_scan = False
    cfg.agent.stair_up_mode = "ascent"
    for k, v in search.items():
        setattr(cfg.agent.search, k, v)
    a = AscentNavAgent(cfg, StubDetector(), AsyncScorer(_StubScorer()),
                       None, "chair", pointnav=driver or _Driver())
    a.reset("chair")
    return a


def _with_frontier(a, xy=(-3.0, 0.0)):
    """One live frontier, so `_explore` gets past the no-frontier rule."""
    a.obstacle_map.frontiers = np.atleast_2d(np.asarray(xy, dtype=float))


def _searched(a, m2=60.0):
    """Mark `m2` of this storey explored."""
    om = a.obstacle_map
    side = int(round(np.sqrt(m2) * om.pixels_per_meter))
    om.explored_area[:side, :side] = True


def _climb_off_the_map(a, om, direction=1):
    """Start a climb, touch the centroid, walk off the stair map."""
    a.stairs.start_navigating(om, direction)
    a.stairs.pre_update(a, om, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))   # reached
    a.stairs.pre_update(a, om, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))   # centroid
    a.stairs.pre_update(a, om, np.array([9.0, 0.0]), _px(a, (9.0, 0.0)))   # off the map


# ---------------------------------------------------------------- M1

def test_the_belief_is_not_built_unless_asked_for():
    assert _agent()._storey_belief is None
    assert _agent(storey_belief=True)._storey_belief is not None


def test_a_searched_storey_with_stairs_to_an_unsearched_one_starts_the_climb():
    a = _agent(storey_belief=True)
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    a._floors[0]["standing_y_n"] = 10
    _with_frontier(a)
    _searched(a)
    action = a._explore(DEPTH, np.zeros(2), 0.0)
    assert a.stairs.climbing and a.stairs.direction == 1 and a._state == "climb"
    assert action == "move_forward"
    assert a.stats["storey_belief_climbs"] == 1 and a.stats["storey_belief_first_step"] == 0


def test_the_same_scene_without_the_flag_keeps_exploring():
    a = _agent()
    _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    a._floors[0]["standing_y_n"] = 10
    _with_frontier(a)
    _searched(a)
    a._explore(DEPTH, np.zeros(2), 0.0)
    assert not a.stairs.climbing


def test_a_storey_barely_searched_does_not_ask():
    a = _agent(storey_belief=True)
    _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    a._floors[0]["standing_y_n"] = 10
    _with_frontier(a)
    _searched(a, m2=1.0)
    a._explore(DEPTH, np.zeros(2), 0.0)
    assert not a.stairs.climbing


def test_no_request_without_a_staircase_or_with_too_few_steps_left():
    a = _agent(storey_belief=True)
    a._floors.append(a._new_floor())
    a._floors[0]["standing_y_n"] = 10
    _with_frontier(a)
    _searched(a)
    a._explore(DEPTH, np.zeros(2), 0.0)                    # no staircase on the map
    assert not a.stairs.climbing
    _paint_stairs(a, xy=(2.0, 0.0))
    a.step_count = a.max_steps - 49                        # 49 left, 50 needed
    a._explore(DEPTH, np.zeros(2), 0.0)
    assert not a.stairs.climbing
    a.step_count = a.max_steps - 50
    a._explore(DEPTH, np.zeros(2), 0.0)
    assert a.stairs.climbing


def test_a_new_episode_forgets_the_held_request():
    a = _agent(storey_belief=True)
    _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    a._floors[0]["standing_y_n"] = 10
    _with_frontier(a)
    _searched(a)
    a._explore(DEPTH, np.zeros(2), 0.0)
    assert a._storey_belief.log
    a.reset("chair")
    assert a._storey_belief.log == [] and a._storey_belief._requested is None


# ---------------------------------------------------------------- M2b

def test_off_the_stair_map_at_the_starting_height_is_not_an_arrival():
    a = _agent(verify_climb=True)
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    a._base_y = 0.1
    _climb_off_the_map(a, om)
    assert not a.stairs.climbing, "the attempt is over"
    assert a._floor_idx == 0, "and the storey has not changed"
    assert a.stats["climb_unverified"] == 1 and a.stats["climb_fail"] == 1
    assert "climb_ok" not in a.stats and "floor_switches" not in a.stats
    assert a._done_initializing is True, "no opening scan on a storey it never left"


def test_off_the_stair_map_half_a_metre_higher_is_ascents_arrival():
    a = _agent(verify_climb=True)
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    a._base_y = 0.1
    a.stairs.start_navigating(om, 1)
    a.stairs.pre_update(a, om, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))
    a.stairs.pre_update(a, om, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))
    a._base_y = 0.1 + 1.4                                  # a landing
    a.stairs.pre_update(a, om, np.array([9.0, 0.0]), _px(a, (9.0, 0.0)))
    assert a._floor_idx == 1 and a.stats["climb_ok"] == 1
    assert "climb_unverified" not in a.stats


def test_without_the_flag_the_same_walk_is_the_references_arrival():
    a = _agent()
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    a._base_y = 0.1
    _climb_off_the_map(a, om)
    assert a._floor_idx == 1 and a.stats["climb_ok"] == 1
    assert "climb_went_nowhere" not in a.stats and a.stairs.fails == {}


def test_two_attempts_that_went_nowhere_ban_the_direction_for_a_while():
    a = _agent(verify_climb=True, climb_fail_cap=2, climb_ban_steps=125)
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    _climb_off_the_map(a, om)
    assert not a.stairs.banned(a, 1)
    _climb_off_the_map(a, om)
    assert a.stairs.banned(a, 1) and not a.stairs.banned(a, 2)
    assert a.stats["climb_bans"] == 1 and a.stats["climb_went_nowhere"] == 2
    # Banned: the belief and the LLM's GO_UP are refused...
    assert a._navigate_stair_if_unexplored_floor(np.zeros(2), 0.0, "up") is None
    # ...a storey with nothing else left is not...
    assert a._navigate_stair_if_unexplored_floor(np.zeros(2), 0.0, "up", last_resort=True) is not None
    a.stairs.update_stair_state(om)
    # ...and it expires.
    a.step_count = 125
    assert not a.stairs.banned(a, 1)
    assert a._navigate_stair_if_unexplored_floor(np.zeros(2), 0.0, "up") is not None


def test_a_ban_belongs_to_the_storey_it_was_earned_on():
    a = _agent(verify_climb=True, climb_fail_cap=1, climb_ban_steps=125)
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    _climb_off_the_map(a, om)
    assert a.stairs.banned(a, 1)
    a._floor_idx = 1
    assert not a.stairs.banned(a, 1)


def test_a_banned_direction_is_not_entered_passively():
    a = _agent(verify_climb=True, climb_fail_cap=1, climb_ban_steps=125)
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    _climb_off_the_map(a, om)
    for _ in range(6):
        a.stairs.pre_update(a, om, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))
    assert not a.stairs.climbing
    a.step_count = 125
    for _ in range(6):
        a.stairs.pre_update(a, om, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))
    assert a.stairs.climbing


def test_a_flight_that_stalled_at_its_foot_counts_and_one_that_gained_height_does_not():
    a = _agent(climb_fail_cap=2)
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    for expect in (1, 2):
        a.stairs.start_navigating(om, 1)
        a.stairs.pre_update(a, om, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))
        om._climb_stair_paused_step = 30
        a._stairs_dispatch(DEPTH, np.array([2.0, 0.0]), 0.0, _px(a, (2.0, 0.0)), None)
        assert a.stats["climb_went_nowhere"] == expect
    assert a.stairs.banned(a, 1)

    b = _agent(climb_fail_cap=2)
    om = _paint_stairs(b, xy=(2.0, 0.0))
    b._floors.append(b._new_floor())
    b.stairs.start_navigating(om, 1)
    b.stairs.pre_update(b, om, np.array([2.0, 0.0]), _px(b, (2.0, 0.0)))
    b._base_y = 1.3                                         # stalled on a landing
    om._climb_stair_paused_step = 30
    b._stairs_dispatch(DEPTH, np.array([2.0, 0.0]), 0.0, _px(b, (2.0, 0.0)), None)
    assert "climb_went_nowhere" not in b.stats and b.stairs.fails == {}


def test_a_new_episode_clears_the_failures_and_the_bans():
    a = _agent(verify_climb=True, climb_fail_cap=1, climb_ban_steps=125)
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    _climb_off_the_map(a, om)
    assert a.stairs.banned_until
    a.reset("chair")
    assert a.stairs.banned_until == {} and a.stairs.fails == {} and a.stairs.attempt_y0 is None


# -------------------------------------------------------- the stair STOP guard

def test_a_pointnav_stop_on_the_way_to_the_stairs_ends_the_episode_only_in_the_reference():
    stop = _Driver(reason="stop", action=None)
    a = _agent(driver=stop)
    _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    assert a._navigate_stair_if_unexplored_floor(np.zeros(2), 0.0, "up") == "stop"

    b = _agent(driver=_Driver(reason="stop", action=None), stair_stop_guard=True)
    _paint_stairs(b, xy=(2.0, 0.0))
    b._floors.append(b._new_floor())
    assert b._navigate_stair_if_unexplored_floor(np.zeros(2), 0.0, "up") == "move_forward"
    assert b.stats["stair_stop_guarded"] == 1 and b.stairs.climbing


# ---------------------------------------------------------------- M2c

def _arrive_on_a_landing(a, om, rise=1.4, direction=1):
    """A climb that walks off the stair map `rise` metres from where it began."""
    y0 = a._base_y
    a.stairs.start_navigating(om, direction)
    a.stairs.pre_update(a, om, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))
    a.stairs.pre_update(a, om, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))
    a._base_y = y0 + (rise if direction == 1 else -rise)
    a.stairs.pre_update(a, om, np.array([9.0, 0.0]), _px(a, (9.0, 0.0)))


def test_an_arrival_short_of_a_storey_keeps_the_climb_going_and_refuses_the_way_back():
    a = _agent(climb_relay=True)
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    _arrive_on_a_landing(a, om)
    assert a._floor_idx == 1 and a.stairs.intent(a) == 1
    assert a.stats["climb_relay_landings"] == 1
    assert a.stairs.banned(a, 2) and not a.stairs.banned(a, 1)
    # The landing's own map has the flight just climbed as its way down...
    assert a.obstacle_map._has_down_stair
    assert a._navigate_stair_if_unexplored_floor(np.zeros(2), 0.0, "down") is None
    # ...which a storey with nothing else left may still take.
    a._floors[0]["obstacle"]._this_floor_explored = False
    assert a._navigate_stair_if_unexplored_floor(np.zeros(2), 0.0, "down", last_resort=True) is not None


def test_the_next_flight_is_taken_before_any_frontier():
    a = _agent(climb_relay=True)
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    _arrive_on_a_landing(a, om)
    _paint_stairs(a, xy=(2.0, 0.0))                        # the landing sees the flight on
    a._floors.append(a._new_floor())
    _with_frontier(a)
    a._explore(DEPTH, np.zeros(2), 0.0)
    assert a.stairs.climbing and a.stairs.direction == 1
    assert a.stats["climb_relay_continues"] == 1


def test_without_the_flag_a_landing_is_a_storey_like_any_other():
    a = _agent()
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    _arrive_on_a_landing(a, om)
    assert a._floor_idx == 1 and a.stairs.intent(a) == 0 and not a.stairs.banned(a, 2)
    assert not any(k.startswith("climb_relay") for k in a.stats)
    _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    _with_frontier(a)
    a._explore(DEPTH, np.zeros(2), 0.0)
    assert not a.stairs.climbing


def test_a_second_arrival_that_covers_the_storey_ends_the_intent():
    a = _agent(climb_relay=True)
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    _arrive_on_a_landing(a, om, rise=1.4)
    om2 = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    _arrive_on_a_landing(a, om2, rise=1.4)                 # 2.8 m from where the relay began
    assert a._floor_idx == 2 and a.stairs.intent(a) == 0
    assert a.stats["climb_relay_storeys"] == 1 and a.stats["climb_relay_landings"] == 1
    assert not a.stairs.banned(a, 2)


def test_one_climb_that_covers_a_storey_opens_no_intent():
    a = _agent(climb_relay=True)
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    _arrive_on_a_landing(a, om, rise=2.8)
    assert a.stairs.intent(a) == 0 and a.stats["climb_relay_storeys"] == 1


def test_the_intent_lapses_and_a_ban_on_the_way_on_ends_it():
    a = _agent(climb_relay=True, climb_ban_steps=125)
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    _arrive_on_a_landing(a, om)
    a.step_count = 124
    assert a.stairs.intent(a) == 1
    a.step_count = 125
    assert a.stairs.intent(a) == 0 and a.stats["climb_relay_lapsed"] == 1

    b = _agent(climb_relay=True, verify_climb=True, climb_fail_cap=1)
    om = _paint_stairs(b, xy=(2.0, 0.0))
    b._floors.append(b._new_floor())
    _arrive_on_a_landing(b, om)
    om2 = _paint_stairs(b, xy=(2.0, 0.0))
    b._floors.append(b._new_floor())
    _climb_off_the_map(b, om2)                             # the way on went nowhere: banned
    assert b.stairs.banned(b, 1) and b.stairs.intent(b) == 0
    assert not b.stairs.banned(b, 2), "so the way back is open again"


def test_a_stall_above_the_start_is_an_arrival_and_one_at_the_start_is_the_references():
    a = _agent(climb_relay=True)
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    a.stairs.start_navigating(om, 1)
    a.stairs.pre_update(a, om, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))
    a._base_y = 1.3
    om._climb_stair_paused_step = 30
    action = a._stairs_dispatch(DEPTH, np.array([2.0, 0.0]), 0.0, _px(a, (2.0, 0.0)), None)
    assert a._floor_idx == 1 and not a.stairs.climbing
    assert a.stats["climb_stall_arrivals"] == 1 and a.stats["climb_ok"] == 1
    assert a.stairs.intent(a) == 1
    assert action == "turn_left" and a._done_initializing is False, "the opening scan of a new storey"

    b = _agent(climb_relay=True)
    om = _paint_stairs(b, xy=(2.0, 0.0))
    b._floors.append(b._new_floor())
    b.stairs.start_navigating(om, 1)
    b.stairs.pre_update(b, om, np.array([2.0, 0.0]), _px(b, (2.0, 0.0)))
    om._climb_stair_paused_step = 30
    b._stairs_dispatch(DEPTH, np.array([2.0, 0.0]), 0.0, _px(b, (2.0, 0.0)), None)
    assert b._floor_idx == 0 and "climb_stall_arrivals" not in b.stats

    c = _agent()                                            # flag off: F5, whatever the height
    om = _paint_stairs(c, xy=(2.0, 0.0))
    c._floors.append(c._new_floor())
    c.stairs.start_navigating(om, 1)
    c.stairs.pre_update(c, om, np.array([2.0, 0.0]), _px(c, (2.0, 0.0)))
    c._base_y = 1.3
    om._climb_stair_paused_step = 30
    c._stairs_dispatch(DEPTH, np.array([2.0, 0.0]), 0.0, _px(c, (2.0, 0.0)), None)
    assert c._floor_idx == 0 and "climb_stall_arrivals" not in c.stats


def test_a_new_episode_forgets_the_intent():
    a = _agent(climb_relay=True)
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    _arrive_on_a_landing(a, om)
    a.reset("chair")
    assert a.stairs.intent_dir == 0 and a.stairs.intent(a) == 0


def test_a_dead_end_landing_is_left_by_the_flight_it_came_on_not_stopped_on():
    for relay, expect_stop in ((True, False), (False, True)):
        a = _agent(climb_relay=relay)
        om = _paint_stairs(a, xy=(2.0, 0.0))
        a._floors.append(a._new_floor())
        _arrive_on_a_landing(a, om)                        # up a flight onto a landing
        land = a.obstacle_map
        land.frontiers = np.zeros((0, 2))
        land._reinitialize_flag = True                     # its one stairwell re-scan is spent
        assert land._has_down_stair and land._explored_down_stair
        action = a._explore(DEPTH, np.zeros(2), 0.0)
        assert (action == "stop") == expect_stop
        if relay:
            assert a.stairs.climbing and a.stairs.direction == 2
            assert a.stats["climb_relay_dead_end"] == 1


def test_a_stall_arrival_scans_its_storey_without_sliding_back_down_the_flight():
    a = _agent(climb_relay=True)
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    a._floors[1]["obstacle"]._done_initializing = True      # a storey stood on before
    a.stairs.start_navigating(om, 1)
    a.stairs.pre_update(a, om, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))
    a._base_y = 1.3
    om._climb_stair_paused_step = 30
    action = a._stairs_dispatch(DEPTH, np.array([2.0, 0.0]), 0.0, _px(a, (2.0, 0.0)), None)
    assert a._floor_idx == 1 and action == "turn_left", "the opening scan runs anyway"
    assert a.stairs.on_arrival_flight
    land = a.obstacle_map
    land._down_stair_map = om._up_stair_map.copy()          # the flight it stalled on
    land._down_stair_frontiers = np.array([[2.0, 0.0]])
    land._has_down_stair = True
    a.stairs.intent_dir = 0                                 # even with no relay to refuse it
    for _ in range(6):
        a.stairs.pre_update(a, land, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))
    assert not a.stairs.climbing, "standing on the arrival flight is not stepping onto it"
    a.stairs.pre_update(a, land, np.array([9.0, 0.0]), _px(a, (9.0, 0.0)))   # off it
    assert not a.stairs.on_arrival_flight
    for _ in range(6):
        a.stairs.pre_update(a, land, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))
    assert a.stairs.climbing and a.stairs.direction == 2


def test_the_stair_stop_guard_is_spent_once_per_storey_and_direction():
    b = _agent(driver=_Driver(reason="stop", action=None), stair_stop_guard=True)
    _paint_stairs(b, xy=(2.0, 0.0))
    b._floors.append(b._new_floor())
    assert b._navigate_stair_if_unexplored_floor(np.zeros(2), 0.0, "up") == "move_forward"
    b.stairs.update_stair_state(b.obstacle_map)
    assert b._navigate_stair_if_unexplored_floor(np.zeros(2), 0.0, "up", last_resort=True) == "stop"
    assert b.stats["stair_stop_guarded"] == 1
    b.reset("chair")
    _paint_stairs(b, xy=(2.0, 0.0))
    b._floors.append(b._new_floor())
    assert b._navigate_stair_if_unexplored_floor(np.zeros(2), 0.0, "up") == "move_forward"
