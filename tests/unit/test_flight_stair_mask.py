"""Which flight the climber drives to, when several look like staircases.

`find_flights` asks the height layer for cells sitting between two storeys.
Furniture answers: a sofa back, a counter, a mezzanine lip. Selection was
`min(usable, key=distance to foot)`, so whatever is nearest wins -- and the
nearest intermediate-height blob is almost never the staircase.

Measured on 00800 cross_anchor_01 (outputs/mf5_pass2_v3), real staircase at
world (-8.1, -13.0):

    ep1  step 172  (-0.67, -4.73)   210 cells   ~10 m away, and pursued from
                                                the TOP storey as a flight_up
    ep2  step  68  (-3.32, -0.97)  1166 cells
    ep2  step 192  (-9.42, -3.32)   652 cells
    ep2  step 245  (-11.02, -3.82)  778 cells
    ep2  step 296  (-8.32, -12.67)  596 cells   <- the staircase, at last

Both episodes spent their climb on furniture and rose 0.00 m. The stair mask
carried over from the ASCENT pass is exactly the evidence that separates the
two, and nothing was reading it.
"""
from types import SimpleNamespace

import numpy as np

from osg.agent.floor_policy import FloorPolicy

from .test_climb import GROUND, UPPER, make_cfg


def _flight(kind, cells_rc, foot_xy):
    return SimpleNamespace(
        kind=kind, cells_rc=np.asarray(cells_rc), foot_xy=np.asarray(foot_xy, float),
        heights=np.linspace(0.3, 1.5, len(cells_rc)), n_cells=len(cells_rc),
    )


def _policy(*, levels=(0.0, 2.9), current=GROUND):
    cfg = make_cfg()
    cfg.floor.enabled = True
    cfg.floor.cross_floor = True
    cfg.floor.flights_prefer_stair_mask = True
    policy = FloorPolicy(cfg, stats={})
    keys = [GROUND, UPPER][: len(levels)]
    policy.estimator._levels = dict(zip(keys, levels))
    policy.estimator.current = current
    for key, height in zip(keys, levels):
        policy.stack.layer(key, step=0).floor_y = height
    policy.stack.current_id = current
    return policy


def test_the_stair_map_beats_the_nearer_sofa():
    policy = _policy()
    policy.costmap.stair_mask = np.zeros(policy.costmap.grid.shape, dtype=bool)
    policy.costmap.stair_mask[40:50, 40:50] = True
    sofa = _flight("up", [[5, 5], [5, 6], [6, 5]], foot_xy=[0.2, 0.2])
    stairs = _flight("up", [[42, 42], [43, 43], [44, 44]], foot_xy=[9.0, 9.0])
    picked = policy._best_flight([sofa, stairs], np.zeros(2))
    assert picked is stairs, "the near blob won again; the stair map was not read"
    assert policy.stats["flight_pick_corroborated"] == 1


def test_nearest_still_wins_among_corroborated_flights():
    policy = _policy()
    policy.costmap.stair_mask = np.zeros(policy.costmap.grid.shape, dtype=bool)
    policy.costmap.stair_mask[40:60, 40:60] = True
    near = _flight("up", [[42, 42]], foot_xy=[1.0, 0.0])
    far = _flight("up", [[55, 55]], foot_xy=[9.0, 0.0])
    assert policy._best_flight([near, far], np.zeros(2)) is near


def test_without_a_stair_map_nothing_is_corroborated_and_nearest_wins():
    """The mask is absent on a run with no prior map and before the detector
    has marked anything. That must degrade to the old behaviour, not to none."""
    policy = _policy()
    policy.costmap.stair_mask = None
    near = _flight("up", [[5, 5]], foot_xy=[1.0, 0.0])
    far = _flight("up", [[9, 9]], foot_xy=[9.0, 0.0])
    assert policy._best_flight([near, far], np.zeros(2)) is near
    assert policy.stats["flight_pick_uncorroborated"] == 1


def test_no_up_flight_from_the_topmost_storey():
    """ep1's error: on the top storey every flight_up is furniture, because
    there is no storey above for it to reach."""
    policy = _policy(current=UPPER)
    up = _flight("up", [[5, 5]], foot_xy=[1.0, 0.0])
    down = _flight("down", [[7, 7]], foot_xy=[8.0, 0.0])
    kept = policy._rank_flights([up, down], floor_y=2.9)
    assert kept == [down]
    assert policy.stats["flights_dropped_no_storey"] == 1


def test_one_known_storey_drops_nothing():
    """At episode start the estimator knows a single level, and every flight
    would look impossible -- which would remove the only way up."""
    policy = _policy(levels=(0.0,))
    up = _flight("up", [[5, 5]], foot_xy=[1.0, 0.0])
    assert policy._rank_flights([up], floor_y=0.0) == [up]
    assert "flights_dropped_no_storey" not in policy.stats


def test_the_filter_never_empties_the_pool():
    """If the direction test would drop everything, keep the flights: a wrong
    guess about storeys must not be able to strand the agent."""
    policy = _policy(current=UPPER)
    up = _flight("up", [[5, 5]], foot_xy=[1.0, 0.0])
    assert policy._rank_flights([up], floor_y=2.9) == [up]


# ------------------------------------------------------- climb direction

def test_an_up_flight_starts_an_up_climb_even_when_the_storeys_tie():
    """The measured bug (outputs/mf5_pass2_v5): flight kind `up`, `here_y` and
    `target_y` both 0.16, and `target_y > here_y` is False at equality -- so
    the tie fell through to -1 and the carrot hunted for treads BELOW an agent
    on the ground floor for 200 steps."""
    from osg.agent.nav_agent import State

    from .test_nav_agent import make_agent, make_cfg

    agent = make_agent(make_cfg(climb_enabled=True, climb_direction_from_flight=True))
    agent.floors.pursuit_flight = SimpleNamespace(kind="up")
    agent._goal_floor_y_cache = 0.163          # identical to the storey below
    agent.floors.estimator._levels = {0: 0.163}
    agent.floors.stack.current_id = 0
    agent._goal_xy = np.array([1.0, 1.0])
    frame = SimpleNamespace(camera_position=np.array([0.0, 1.5, 0.0]))
    agent.climb._start_climb(frame)
    assert agent.climb._climb_direction == 1, "an up flight must climb up"
    assert agent.stats["climb_start_up"] == 1
    assert agent.state is State.CLIMB


def test_a_down_flight_starts_a_down_climb():
    from .test_nav_agent import make_agent, make_cfg

    agent = make_agent(make_cfg(climb_enabled=True, climb_direction_from_flight=True))
    agent.floors.pursuit_flight = SimpleNamespace(kind="down")
    agent._goal_floor_y_cache = 3.163          # would say UP on heights alone
    agent.floors.estimator._levels = {0: 0.163}
    agent.floors.stack.current_id = 0
    agent._goal_xy = np.array([1.0, 1.0])
    agent.climb._start_climb(SimpleNamespace(camera_position=np.array([0.0, 1.5, 0.0])))
    assert agent.climb._climb_direction == -1
    assert agent.stats["climb_start_down"] == 1


def test_the_height_rule_is_untouched_when_the_flag_is_off():
    """Every arm measured before this existed must be unaffected."""
    from .test_nav_agent import make_agent, make_cfg

    agent = make_agent(make_cfg(climb_enabled=True))
    assert agent.cfg.agent.climb_direction_from_flight is False
    agent.floors.pursuit_flight = SimpleNamespace(kind="up")
    agent._goal_floor_y_cache = 0.163
    agent.floors.estimator._levels = {0: 0.163}
    agent.floors.stack.current_id = 0
    agent._goal_xy = np.array([1.0, 1.0])
    agent.climb._start_climb(SimpleNamespace(camera_position=np.array([0.0, 1.5, 0.0])))
    assert agent.climb._climb_direction == -1, "the shipped tie-goes-down behaviour"


# ---------------------------------------------- carrot: never underfoot

def _climbing_agent(**over):
    from .test_nav_agent import make_agent, make_cfg
    # `make_cfg` ships `climb_flight_carrot: False`, under which `_flight_carrot`
    # returns None before looking at a single cell.
    agent = make_agent(make_cfg(climb_enabled=True, climb_flight_carrot=True, **over))
    agent.climb._climb_direction = 1
    return agent


def _flight_at(agent, offsets_m, heights):
    """A flight whose cells sit at these horizontal offsets from the origin."""
    rc = np.asarray([agent.costmap.world_to_grid(np.array([d, 0.0])) for d in offsets_m])
    return SimpleNamespace(kind="up", cells_rc=rc, heights=np.asarray(heights, float),
                           n_cells=len(rc), foot_xy=np.zeros(2))


def test_shipped_carrot_takes_the_nearest_in_band_cell_even_underfoot():
    """The measured failure: a ramp says the cell at the agent's feet is
    0.4 m up, so it is the nearest in-band cell and the goal is 0 m away."""
    agent = _climbing_agent()
    agent.floors.pursuit_flight = _flight_at(agent, [0.05, 0.6, 1.2], [0.4, 0.7, 1.0])
    frame = SimpleNamespace(camera_position=np.array([0.0, 0.88, 0.0]))
    goal = agent.climb._flight_carrot(frame, np.zeros(2))
    assert float(np.linalg.norm(goal)) < 0.1


def test_min_ahead_skips_the_underfoot_cell_for_the_next_one():
    agent = _climbing_agent(climb_carrot_min_ahead_m=0.4)
    agent.floors.pursuit_flight = _flight_at(agent, [0.05, 0.6, 1.2], [0.4, 0.7, 1.0])
    frame = SimpleNamespace(camera_position=np.array([0.0, 0.88, 0.0]))
    goal = agent.climb._flight_carrot(frame, np.zeros(2))
    assert 0.5 < float(np.linalg.norm(goal)) < 0.7, "the 0.6 m cell, not the one underfoot"


def test_min_ahead_is_a_no_op_when_the_band_is_already_ahead():
    """With true heights the nearest in-band cell was 0.66-1.01 m ahead in
    every probe step; the flag must not change that choice."""
    a = _climbing_agent()
    b = _climbing_agent(climb_carrot_min_ahead_m=0.4)
    for agent in (a, b):
        agent.floors.pursuit_flight = _flight_at(agent, [0.7, 1.2, 1.8], [0.5, 0.8, 1.2])
    frame = SimpleNamespace(camera_position=np.array([0.0, 0.88, 0.0]))
    ga, gb = a.climb._flight_carrot(frame, np.zeros(2)), b.climb._flight_carrot(frame, np.zeros(2))
    assert np.allclose(ga, gb)


def test_when_every_in_band_cell_is_underfoot_the_carrot_aims_further_up():
    agent = _climbing_agent(climb_carrot_min_ahead_m=0.4)
    agent.floors.pursuit_flight = _flight_at(agent, [0.05, 0.1, 1.5], [0.4, 0.5, 1.6])
    frame = SimpleNamespace(camera_position=np.array([0.0, 0.88, 0.0]))
    goal = agent.climb._flight_carrot(frame, np.zeros(2))
    assert float(np.linalg.norm(goal)) > 1.0, "the far cell, via the fall-through"
    assert agent.stats["climb_carrot_underfoot"] == 1


# ------------------------------------------- climb: close the KNOWN gap

def _climb_in_progress(agent, dy: float):
    """A climb from a 0.0 m storey toward a 3.0 m one, `dy` metres up."""
    from osg.agent.nav_agent import State
    agent.floors.estimator._levels = {0: 0.0, 1: 3.0}
    agent.floors.estimator.current = 0
    agent.floors.stack.layer(0, step=0).floor_y = 0.0
    agent.floors.stack.layer(1, step=0).floor_y = 3.0
    agent.floors.stack.current_id = 0
    agent.cfg.floor.no_level_on_flight = True
    agent.cfg.floor.new_level_m = 1.8
    agent.floors.pursuit_flight = SimpleNamespace(kind="up", n_cells=0, cells_rc=np.zeros((0, 2), int),
                                                  heights=np.zeros(0), foot_xy=np.zeros(2))
    agent._goal_xy = np.array([1.0, 1.0]); agent._goal_floor_y_cache = 3.0
    agent.climb._start_climb(SimpleNamespace(camera_position=np.array([0.0, 0.88, 0.0])))
    agent.floors.pursuing = True
    frame = SimpleNamespace(camera_position=np.array([0.0, 0.88 + dy, 0.0]), depth=np.zeros((4, 4)),
                            T_wc=np.eye(4), intrinsics=SimpleNamespace(width=4, fx=2.0, cx=2.0))
    return frame, State


def test_shipped_rule_declares_a_storey_at_new_level_m():
    agent = _climbing_agent()
    frame, State = _climb_in_progress(agent, dy=1.85)
    agent.climb._do_climb(frame)
    assert agent.state is not State.CLIMB, "1.85 >= new_level_m 1.8: the shipped rule ends the climb"
    assert agent.stats.get("climb_ok") == 1


def test_with_a_known_gap_the_climb_continues_past_new_level_m():
    """v11 ep2: +1.83 m on a 3.0 m storey was called a storey. With the gap
    known it is 1.2 m short and the climb goes on."""
    agent = _climbing_agent(climb_to_target_storey_tol_m=0.3)
    frame, State = _climb_in_progress(agent, dy=1.85)
    agent.climb._do_climb(frame)
    assert agent.state is State.CLIMB


def test_with_a_known_gap_the_climb_ends_within_tolerance_of_it():
    agent = _climbing_agent(climb_to_target_storey_tol_m=0.3)
    frame, State = _climb_in_progress(agent, dy=2.75)
    agent.climb._do_climb(frame)
    assert agent.state is not State.CLIMB
    assert agent.stats.get("climb_ok") == 1
