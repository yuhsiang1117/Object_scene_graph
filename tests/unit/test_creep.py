"""The steered creep (planning/creep.py): turn toward the goal, sweep round a
refused forward, and stall only on forwards that do not close."""
import numpy as np

from osg.planning.creep import FORWARD, TURN_LEFT, TURN_RIGHT, SteeredCreep


def test_goal_ahead_walks_forward():
    c = SteeredCreep(steer_deg=20.0)
    assert c.step([1, 0], [0, 0], 0.8, np.radians(5.0), None) == (FORWARD, "creep")


def test_goal_off_the_heading_turns_toward_it():
    c = SteeredCreep(steer_deg=20.0)
    # theta is positive to the LEFT (pointnav_driver.rho_theta)
    assert c.step([1, 0], [0, 0], 0.8, np.radians(60.0), None)[0] == TURN_LEFT
    c = SteeredCreep(steer_deg=20.0)
    assert c.step([1, 0], [0, 0], 0.8, np.radians(-60.0), None)[0] == TURN_RIGHT


def test_refused_forward_sweeps_toward_the_goal_side_first():
    c = SteeredCreep(turn_deg=30.0, steer_deg=20.0)
    assert c.step([1, 0], [0, 0], 0.8, np.radians(10.0), None) == (FORWARD, "creep")
    # The forward went nowhere: try 30 degrees toward the goal's side (left),
    # committed with a forward.
    a1 = c.step([1, 0], [0, 0], 0.8, np.radians(10.0), FORWARD)
    assert a1 == (TURN_LEFT, "creep_sweep")
    # Turned 30 left: the goal now bears 20 degrees right.
    assert c.step([1, 0], [0, 0], 0.8, np.radians(-20.0), TURN_LEFT) == (FORWARD, "creep_sweep")
    # Refused again: the other side, 30 right of the goal's bearing -- 50 right
    # of this heading, two turns, then a forward.
    assert c.step([1, 0], [0, 0], 0.8, np.radians(-20.0), FORWARD) == (TURN_RIGHT, "creep_sweep")
    assert c.step([1, 0], [0, 0], 0.8, np.radians(10.0), TURN_RIGHT) == (TURN_RIGHT, "creep_sweep")
    assert c.step([1, 0], [0, 0], 0.8, np.radians(40.0), TURN_RIGHT) == (FORWARD, "creep_sweep")


def test_motion_ends_the_sweep():
    c = SteeredCreep(turn_deg=30.0, steer_deg=20.0)
    c.step([1, 0], [0, 0], 0.8, 0.0, None)
    c.step([1, 0], [0, 0], 0.8, 0.0, FORWARD)           # blocked: sweep starts
    c._queue = []
    # A forward that moved and closed on the goal: back to plain creeping.
    assert c.step([1, 0], [0.2, 0.0], 0.6, 0.0, FORWARD) == (FORWARD, "creep")


def test_stall_counts_only_forwards_that_do_not_close():
    c = SteeredCreep(turn_deg=30.0, steer_deg=20.0, stall_steps=3)
    c.step([1, 0], [0, 0], 0.8, np.radians(90.0), None)       # a turn
    for _ in range(5):                                         # turns do not count
        action, why = c.step([1, 0], [0, 0], 0.8, np.radians(90.0), TURN_LEFT)
        assert action == TURN_LEFT and why == "creep_turn"
    reasons = []
    for _ in range(3):
        c._queue = []
        reasons.append(c.step([1, 0], [0, 0], 0.8, 0.0, FORWARD)[1])
    assert reasons[-1] == "creep_stalled"
    assert c.n_stalls == 1


def test_a_new_goal_resets():
    c = SteeredCreep(turn_deg=30.0, steer_deg=20.0, stall_steps=2)
    c.step([1, 0], [0, 0], 0.8, 0.0, None)
    c.step([1, 0], [0, 0], 0.8, 0.0, FORWARD)
    # Another goal: no inherited stall count or sweep.
    assert c.step([0, 1], [0, 0], 0.9, 0.0, FORWARD) == (FORWARD, "creep")
