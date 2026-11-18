"""The sensor-only climb, which did not exist.

`State.CLIMB` was in the enum, `_carrot_action`, `_on_a_staircase` and
`_left_the_stairs` were on `NavAgent`, and nothing ever assigned the state or the
attributes they read. The `ascent` policy dispatches on `State.CLIMB` to a
`_do_climb` that was never defined. So the OSG agent had no way up a staircase:
`portals.py` says a portal is "a place to walk toward, after which the navmesh
handles the climb", and the navmesh was removed on 2026-09-08.

Measured on 00821's cracker box: base, 55 switch attempts and 0.17 m of ascent;
navmesh restored, one attempt and the whole 3.6 m storey. These tests pin the
sensor-only replacement for that step: a pursuit that reaches the stairs becomes
a climb, the climb steers at the farthest depth, and it ends on a committed new
storey or on evidence that it is not working.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from osg.agent.nav_agent import NavAgent, State
from osg.exploration.async_scorer import AsyncScorer
from osg.exploration.scorer import NullScorer
from osg.perception.detector import StubDetector

from .conftest import make_frame
from .test_nav_agent import make_cfg

GROUND, UPPER = 0, 1


def _agent(enabled=True):
    cfg = make_cfg()
    cfg.floor.enabled = True
    cfg.floor.cross_floor = True
    cfg.mapping.multi_floor = True
    cfg.agent.climb_enabled = enabled
    cfg.agent.climb_max_steps = 20
    cfg.agent.stair_reach_m = 0.6
    agent = NavAgent(cfg, StubDetector(), AsyncScorer(NullScorer()), None, "bowl")
    est = agent.floors.estimator
    est._levels = {GROUND: 0.0, UPPER: 2.9}
    est.current = GROUND
    agent.floors.stack.layer(GROUND, step=0).floor_y = 0.0
    agent.floors.stack.layer(UPPER, step=0).floor_y = 2.9
    agent.floors.stack.current_id = GROUND
    return agent


def _pursuit(agent, goal_xy=(1.0, 2.0), target_y=2.9):
    """A floor pursuit in flight, as `_try_floor_switch` leaves one."""
    agent.state = State.GOTO_FRONTIER
    agent._goal_xy = np.asarray(goal_xy, dtype=float)
    agent._goal_floor_y_cache = target_y
    agent.floors.pursuing = True
    agent.floors._pursuit_goal_xy = np.asarray(goal_xy, dtype=float)
    agent._goto_deadline = 10 ** 9


def _frame(intrinsics, xz, y=0.0):
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = xz[0], y, xz[1]
    return make_frame(intrinsics, T)


# ----------------------------------------------------------------- entry

def test_arriving_at_the_stair_goal_starts_a_climb(intrinsics):
    agent = _agent()
    _pursuit(agent, goal_xy=(1.0, 2.0))
    frame = _frame(intrinsics, (1.2, 2.0), y=1.5)
    assert agent.climb._at_the_stairs(frame)
    agent.climb._start_climb(frame)
    assert agent.state is State.CLIMB
    assert agent.stats["climb_start"] == 1
    assert agent.climb._climb_direction == +1


def test_far_from_the_goal_and_off_any_stair_evidence_is_not_a_climb(intrinsics):
    agent = _agent()
    _pursuit(agent, goal_xy=(1.0, 2.0))
    assert not agent.climb._at_the_stairs(_frame(intrinsics, (5.0, 5.0), y=1.5))


def test_the_climb_is_off_by_default(intrinsics):
    agent = _agent(enabled=False)
    assert not bool(agent.cfg.agent.climb_enabled)


def test_a_descent_is_recognised_from_the_target_height(intrinsics):
    agent = _agent()
    agent.floors.estimator.current = UPPER
    agent.floors.stack.current_id = UPPER
    _pursuit(agent, goal_xy=(1.0, 2.0), target_y=0.0)
    agent.climb._start_climb(_frame(intrinsics, (1.0, 2.0), y=4.4))
    assert agent.climb._climb_direction == -1


# ------------------------------------------------------------------ body

def test_a_descent_tilts_the_camera_down_once_first(intrinsics):
    """ASCENT's phase 2: the carrot has to see the treads below, not the far
    wall."""
    agent = _agent()
    agent.floors.estimator.current = UPPER
    agent.floors.stack.current_id = UPPER
    _pursuit(agent, goal_xy=(1.0, 2.0), target_y=0.0)
    frame = _frame(intrinsics, (1.0, 2.0), y=4.4)
    agent.climb._start_climb(frame)
    assert agent.climb._do_climb(frame) == "look_down"
    assert agent.climb._climb_pitched
    assert agent.climb._do_climb(frame) != "look_down"


def test_the_climb_ends_when_a_new_storey_is_committed(intrinsics):
    agent = _agent()
    _pursuit(agent)
    frame = _frame(intrinsics, (1.0, 2.0), y=1.5)
    agent.climb._start_climb(frame)
    # The estimator commits the upper floor, as observe() does on the stairs.
    agent.floors.stack.current_id = UPPER
    agent.climb._do_climb(_frame(intrinsics, (1.0, 4.0), y=4.4))
    assert agent.state is State.EXPLORE
    assert agent.stats.get("climb_ok") == 1
    assert agent.stats.get("climb_end_new_floor") == 1


def test_the_climb_ends_on_budget_and_records_the_failure(intrinsics):
    """A climb that spends its budget without a new storey is evidence about
    the place, and `end_pursuit` hands it to the portal failure memory."""
    agent = _agent()
    agent.cfg.floor.portal_failure_memory = True
    _pursuit(agent, goal_xy=(1.0, 2.0))
    frame = _frame(intrinsics, (1.0, 2.0), y=1.5)
    agent.climb._start_climb(frame)
    for _ in range(25):
        if agent.state is not State.CLIMB:
            break
        agent.climb._do_climb(frame)
    assert agent.state is State.EXPLORE
    assert agent.stats.get("climb_fail") == 1
    assert agent.stats.get("climb_end_budget") == 1
    assert agent.floors._portal_failed_here([1.0, 2.0])


def test_height_gained_counts_even_if_the_pursuit_was_ended_elsewhere(intrinsics):
    """`pursuit_ok` can end a pursuit on its deadline while the agent is halfway
    up. Judge by height gained, not by what ended it."""
    agent = _agent()
    agent.cfg.floor.new_level_m = 1.8
    _pursuit(agent)
    agent.climb._start_climb(_frame(intrinsics, (1.0, 2.0), y=1.5))
    agent.floors.pursuing = False
    agent.climb._do_climb(_frame(intrinsics, (1.0, 3.0), y=1.5 + 1.2))
    assert agent.stats.get("climb_ok") == 1
    assert agent.stats.get("climb_end_height") == 1


def test_the_carrot_is_what_moves_the_agent(intrinsics):
    """No pointnav in the test harness, so the follower path is taken; the
    point is that a step in CLIMB is a motion command, not a stop."""
    agent = _agent()
    _pursuit(agent)
    frame = _frame(intrinsics, (1.0, 2.0), y=1.5)
    agent.climb._start_climb(frame)
    action = agent.climb._do_climb(frame)
    assert action in ("move_forward", "turn_left", "turn_right")


# ------------------------------------------------------------ targeting

def test_a_seen_staircase_is_chosen_over_a_portal():
    """`floor.climb_targets: stairs_first`. A `stairs` track the detector has
    corroborated is a place you can climb from; a portal is a place you can see
    the next floor from."""
    from osg.agent.floor_policy import FloorPolicy
    from osg.mapping.portals import FloorSwitchPolicy

    cfg = make_cfg()
    cfg.floor.enabled = True
    cfg.floor.cross_floor = True
    cfg.floor.climb_targets = "stairs_first"
    policy = FloorPolicy(cfg, stats={})
    policy.estimator._levels = {GROUND: 0.0, UPPER: 2.9}
    policy.estimator.current = GROUND
    policy.stack.layer(GROUND, step=0).floor_y = 0.0
    policy.stack.layer(UPPER, step=0).floor_y = 2.9
    policy.stack.current_id = GROUND
    policy.switch_policy = FloorSwitchPolicy(max_steps=500, no_switch_before=0,
                                             min_interval_steps=0)
    frame = SimpleNamespace(camera_position=np.array([0.0, 1.5, 0.0]))
    sg = SimpleNamespace(objects=[], rooms={})
    goal = policy.try_switch(
        frame, 100, best_path_cost=None, scene_graph=sg, target="bowl",
        reachable_fn=None, target_floor=UPPER,
        stair_xyz=[(np.array([3.0, 1.0]), "up"), (np.array([9.0, 9.0]), "up")],
    )
    assert goal is not None
    assert goal.goal_xy.tolist() == [3.0, 1.0], "nearest seen staircase, not the far one"
    assert goal.target_y == 2.9
    assert policy.stats.get("stair_track_switch_attempts") == 1


def test_a_flight_the_wrong_way_is_not_a_way_there():
    from osg.agent.floor_policy import FloorPolicy
    from osg.mapping.portals import FloorSwitchPolicy

    cfg = make_cfg()
    cfg.floor.enabled = True
    cfg.floor.cross_floor = True
    cfg.floor.climb_targets = "stairs_first"
    policy = FloorPolicy(cfg, stats={})
    policy.estimator._levels = {GROUND: 0.0, UPPER: 2.9}
    policy.estimator.current = GROUND
    policy.stack.layer(GROUND, step=0).floor_y = 0.0
    policy.stack.layer(UPPER, step=0).floor_y = 2.9
    policy.stack.current_id = GROUND
    policy.switch_policy = FloorSwitchPolicy(max_steps=500, no_switch_before=0,
                                             min_interval_steps=0)
    frame = SimpleNamespace(camera_position=np.array([0.0, 1.5, 0.0]))
    sg = SimpleNamespace(objects=[], rooms={})
    # The only known flight goes DOWN and the request is for the floor above.
    goal = policy.try_switch(
        frame, 100, best_path_cost=None, scene_graph=sg, target="bowl",
        reachable_fn=None, target_floor=UPPER, stair_xyz=[(np.array([3.0, 1.0]), "down")],
    )
    assert policy.stats.get("stair_track_switch_attempts") is None
    assert goal is None  # and no portal exists in an empty costmap either


def test_an_undirected_pursuit_takes_a_down_flight_downward():
    """With no storey requested, a `down` region names the floor below as the
    target, not the floor above."""
    from osg.agent.floor_policy import FloorPolicy
    from osg.mapping.portals import FloorSwitchPolicy

    cfg = make_cfg()
    cfg.floor.enabled = True
    cfg.floor.cross_floor = True
    cfg.floor.climb_targets = "stairs_first"
    policy = FloorPolicy(cfg, stats={})
    policy.estimator._levels = {0: 0.0, 1: 2.9, 2: -2.9}
    policy.estimator.current = 0
    for k, y in ((0, 0.0), (1, 2.9), (2, -2.9)):
        policy.stack.layer(k, step=0).floor_y = y
    policy.stack.current_id = 0
    policy.switch_policy = FloorSwitchPolicy(max_steps=500, no_switch_before=0,
                                             min_interval_steps=0)
    frame = SimpleNamespace(camera_position=np.array([0.0, 1.5, 0.0]))
    sg = SimpleNamespace(objects=[], rooms={})
    goal = policy.try_switch(
        frame, 100, best_path_cost=None, scene_graph=sg, target="bowl",
        reachable_fn=None, target_floor=None, stair_xyz=[(np.array([2.0, 2.0]), "down")],
    )
    assert goal is not None and goal.target_y == -2.9


# --------------------------------------------- the two v9 defects, pinned

def test_the_reach_is_never_inside_the_movers_own_stop_radius(intrinsics):
    """PointNav stops at 0.9 m and will not close further. Measured on 00821:
    82 pursuits of one stair target, the agent 0.87 m from it, zero climbs."""
    agent = _agent()
    agent.cfg.agent.stair_reach_m = 0.6
    agent.pointnav = SimpleNamespace(stop_radius=0.9)
    _pursuit(agent, goal_xy=(1.0, 2.0))
    assert agent.climb._at_the_stairs(_frame(intrinsics, (1.87, 2.0), y=1.5))


def test_a_pursuit_in_flight_is_not_reissued(intrinsics):
    """Each re-issue reset the progress clock, so a pursuit never ended and
    no failure was ever remembered."""
    agent = _agent()
    agent.cfg.floor.hold_pursuit = True
    _pursuit(agent, goal_xy=(1.0, 2.0))
    agent.floors._portal_step = agent.step_count
    agent.floors._portal_start_y = 1.5
    frame = _frame(intrinsics, (0.0, 0.0), y=1.5)
    assert agent._try_floor_switch(frame, None, target_floor=UPPER) is True
    assert agent.stats.get("floor_switch_reissue_suppressed") == 1
    assert agent.stats.get("floor_switch_attempts", 0) == 0


# ------------------------------------------------ v12: what the carrot aims at

def _stamp(agent, cells_xy, kind="up"):
    """Put detector stair evidence on the current floor's grid."""
    layer = agent.floor_layer
    from osg.mapping.stairs import StairDetector
    if agent.stair_detector is None:
        agent.stair_detector = StairDetector(resolution_m=layer.costmap.resolution, min_hits=1)
    agent.stair_detector._ensure_grids(layer)
    grid = layer.up_stair_hits if kind == "up" else layer.down_stair_hits
    for xy in cells_xy:
        r, c = layer.costmap.world_to_grid(np.asarray(xy, dtype=float))
        grid[r, c] = 5


def test_ascending_aims_at_the_farthest_stair_cell_in_reach(intrinsics):
    """The top of the visible flight, not the far wall."""
    agent = _agent()
    agent.cfg.agent.climb_cell_carrot = True
    _stamp(agent, [(1.5, 2.0), (2.0, 2.0), (2.5, 2.0), (9.0, 9.0)])
    _pursuit(agent, goal_xy=(1.0, 2.0))
    agent.climb._start_climb(_frame(intrinsics, (1.0, 2.0), y=1.5))
    goal = agent.climb._stair_cell_carrot(np.array([1.0, 2.0]))
    assert goal is not None
    assert abs(goal[0] - 2.5) < 0.06 and abs(goal[1] - 2.0) < 0.06, "the 9 m cell is out of reach"


def test_descending_aims_at_the_nearest_lip(intrinsics):
    agent = _agent()
    agent.cfg.agent.climb_cell_carrot = True
    agent.floors.estimator.current = UPPER
    agent.floors.stack.current_id = UPPER
    _stamp(agent, [(1.5, 2.0), (2.5, 2.0)], kind="down")
    _pursuit(agent, goal_xy=(1.0, 2.0), target_y=0.0)
    agent.climb._start_climb(_frame(intrinsics, (1.0, 2.0), y=4.4))
    goal = agent.climb._stair_cell_carrot(np.array([1.0, 2.0]))
    assert goal is not None and abs(goal[0] - 1.5) < 0.06


def test_without_stair_cells_the_depth_ray_is_the_fallback(intrinsics):
    agent = _agent()
    agent.cfg.agent.climb_cell_carrot = True
    _pursuit(agent)
    agent.climb._start_climb(_frame(intrinsics, (1.0, 2.0), y=1.5))
    assert agent.climb._stair_cell_carrot(np.array([1.0, 2.0])) is None


def test_a_run_of_blocked_forwards_becomes_a_turn(intrinsics):
    """Pressing into a wall 238 times gained 0.00 m. After N STOPs in a row the
    agent re-aims."""
    agent = _agent()
    agent.cfg.agent.climb_blocked_turn_after = 3
    agent.pointnav = SimpleNamespace(
        stop_radius=0.9,
        step=lambda goal, stop_radius=None: SimpleNamespace(action=None, reason="policy_stop"),
    )
    _pursuit(agent)
    frame = _frame(intrinsics, (1.0, 2.0), y=1.5)
    agent.climb._start_climb(frame)
    actions = [agent.climb._carrot_action(frame, np.array([1.0, 2.0])) for _ in range(3)]
    assert actions[:2] == ["move_forward", "move_forward"]
    assert actions[2] == "turn_left"
    assert agent.stats.get("climb_blocked_turn") == 1


# ------------------------------------------------- v13: the flight carrot

def _flight(agent, heights, xs, z=2.0):
    from osg.mapping.stairs import Flight
    cm = agent.costmap
    rc = np.stack([cm.world_to_grid(np.array([x, z], dtype=float)) for x in xs]).astype(int)
    return Flight(kind="up", cells_rc=rc, heights=np.asarray(heights, float),
                  foot_xy=np.array([xs[0], z]), top_xy=np.array([xs[-1], z]), span_m=float(max(heights) - min(heights)))


def test_the_flight_carrot_aims_at_the_next_tread(intrinsics):
    agent = _agent()
    agent.cfg.agent.climb_flight_carrot = True
    agent.cfg.agent.camera_height = 0.88
    xs = [1.0, 1.3, 1.6, 1.9, 2.2, 2.5]
    heights = [0.17, 0.34, 0.51, 0.68, 0.85, 1.02]
    _pursuit(agent, goal_xy=(1.0, 2.0))
    agent.floors.pursuit_flight = _flight(agent, heights, xs)
    frame = _frame(intrinsics, (1.0, 2.0), y=0.88)  # standing at floor level
    agent.climb._start_climb(frame)
    goal = agent.climb._flight_carrot(frame, np.array([1.0, 2.0]))
    assert goal is not None
    # 0.35-1.0 m above standing height 0.0: treads at 0.51..1.02; nearest is x=1.6
    assert abs(goal[0] - 1.6) < 0.06
    assert agent.stats.get("climb_flight_carrot") == 1


def test_at_the_top_the_highest_tread_is_the_goal(intrinsics):
    agent = _agent()
    agent.cfg.agent.climb_flight_carrot = True
    agent.cfg.agent.camera_height = 0.88
    xs = [1.0, 1.3, 1.6]
    heights = [0.17, 0.34, 0.51]
    _pursuit(agent, goal_xy=(1.0, 2.0))
    agent.floors.pursuit_flight = _flight(agent, heights, xs)
    frame = _frame(intrinsics, (1.3, 2.0), y=0.88 + 0.34)  # standing on the second tread
    agent.climb._start_climb(frame)
    goal = agent.climb._flight_carrot(frame, np.array([1.3, 2.0]))
    assert goal is not None and abs(goal[0] - 1.6) < 0.06
    assert agent.stats.get("climb_flight_carrot_top") == 1


def test_without_a_flight_the_carrot_declines(intrinsics):
    agent = _agent()
    agent.cfg.agent.climb_flight_carrot = True
    _pursuit(agent)
    frame = _frame(intrinsics, (1.0, 2.0), y=1.5)
    agent.climb._start_climb(frame)
    assert agent.climb._flight_carrot(frame, np.array([1.0, 2.0])) is None


# ------------------------------------------- v16: through the half-landing

def test_the_climb_picks_up_the_next_flight_from_a_landing(intrinsics):
    """One flight is not one storey. When the treads run out part-way the climb
    takes the next flight rather than handing back 2.3 m into a 3.2 m descent."""
    from osg.mapping.stairs import Flight

    agent = _agent()
    agent.cfg.agent.climb_relink_flights = True
    agent.cfg.agent.climb_flight_carrot = True
    agent.floors.estimator.current = UPPER
    agent.floors.stack.current_id = UPPER
    _pursuit(agent, goal_xy=(1.0, 2.0), target_y=0.0)
    frame = _frame(intrinsics, (1.0, 2.0), y=4.4)
    agent.climb._start_climb(frame)

    # Its only tread is ABOVE the agent: descending, there is nothing ahead on
    # this flight, which is what standing on the landing at its foot looks like.
    finished = Flight(kind="down", cells_rc=np.array([[10, 10]]), heights=np.array([4.0]),
                      foot_xy=np.array([9.0, 9.0]), top_xy=np.array([9.0, 9.0]), span_m=0.6)
    nxt = Flight(kind="down", cells_rc=np.array([[20, 20]]), heights=np.array([1.0]),
                 foot_xy=np.array([1.4, 2.0]), top_xy=np.array([2.0, 2.0]), span_m=0.6)
    agent.floors.pursuit_flight = finished
    agent.climb._relink_flight = lambda f, xy: (setattr(agent.floors, "pursuit_flight", nxt), True)[1]

    agent.climb._do_climb(_frame(intrinsics, (1.0, 2.0), y=4.4))
    assert agent.floors.pursuit_flight is nxt
    assert agent.stats.get("climb_relinked") == 1


def test_relinking_is_off_by_default(intrinsics):
    agent = _agent()
    assert not bool(agent.cfg.agent.climb_relink_flights)


def test_levels_are_suppressed_for_the_whole_climb_not_just_on_treads(intrinsics):
    """A half-landing is flat ground between two flights. Measured on 00808 the
    suppression fired on 7 frames of treads and the level was still created on
    the landing in between."""
    agent = _agent()
    agent.cfg.floor.no_level_on_flight = True
    _pursuit(agent)
    frame = _frame(intrinsics, (1.0, 2.0), y=1.5)
    assert not agent.floors.climbing
    agent.climb._start_climb(frame)
    assert agent.floors.climbing, "the whole climb suppresses, not only the treads"
    agent.climb._end_climb(False, "budget")
    assert not agent.floors.climbing


def test_a_storey_of_height_ends_the_climb_when_levels_are_suppressed(intrinsics):
    """With the estimator held back, nothing announces the new floor while the
    agent is on it, so the climb judges arrival by height gained."""
    agent = _agent()
    agent.cfg.floor.no_level_on_flight = True
    agent.cfg.floor.new_level_m = 1.8
    _pursuit(agent)
    agent.climb._start_climb(_frame(intrinsics, (1.0, 2.0), y=1.5))
    agent.climb._do_climb(_frame(intrinsics, (1.0, 4.0), y=1.5 + 1.9))
    assert agent.stats.get("climb_ok") == 1
    assert agent.stats.get("climb_end_storey_of_height") == 1
