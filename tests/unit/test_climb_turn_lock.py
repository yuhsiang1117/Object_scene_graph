"""On a staircase, do not turn when a turn would overshoot.

One turn is `turn_deg` (30 degrees), so correcting a 10-degree heading error
costs 30 degrees and leaves 20 the other way, which the next step corrects
back. Measured on the climb traces (outputs/mf5_pass2_v18): 25% of turns on
the descent and 43% on the ascent immediately reversed the previous turn, and
the agent made 6.6 m of net progress along a 28.1 m path -- 8.9 m along 41.7 m
on the ascent. Three quarters of the motion undone.
"""
from types import SimpleNamespace

import numpy as np

from osg.agent.climb import ClimbPolicy

from .test_climb import make_cfg


def _agent(deadband=15.0, release=30.0, stuck_eps=0.0, cap=0, last_action="turn_left"):
    cfg = make_cfg()
    cfg.agent.climb_turn_deadband_deg = deadband
    cfg.agent.climb_turn_release_deg = release
    cfg.agent.climb_turn_stuck_eps_m = stuck_eps
    cfg.agent.climb_turn_suppress_max = cap
    return SimpleNamespace(cfg=cfg, stats={}, _climb_turn_locked=False,
                           _climb_suppressed_run=0, _climb_last_xy=None,
                           nav=SimpleNamespace(_last_action=last_action))


def _frame(heading_deg):
    """A pose facing `heading_deg` in the ground plane. `agent_heading` reads
    the camera's forward axis (OpenCV +z) projected onto (x, z)."""
    h = np.radians(heading_deg)
    T = np.eye(4)
    T[:3, 2] = [np.cos(h), 0.0, np.sin(h)]
    return SimpleNamespace(T_wc=T)


def _lock(agent, heading_deg, goal, action):
    return ClimbPolicy._climb_turn_lock(
        agent, _frame(heading_deg), np.zeros(2), np.asarray(goal, float), action)


def test_a_turn_inside_the_deadband_becomes_a_step_forward():
    agent = _agent()
    goal = [np.cos(np.radians(10)), np.sin(np.radians(10))]   # 10 deg off
    assert _lock(agent, 0.0, goal, "turn_left") == "move_forward"
    assert agent.stats["climb_turn_suppressed"] == 1


def test_a_real_misalignment_still_turns():
    agent = _agent()
    goal = [np.cos(np.radians(70)), np.sin(np.radians(70))]
    assert _lock(agent, 0.0, goal, "turn_left") == "turn_left"
    assert not agent._climb_turn_locked


def test_the_lock_holds_through_the_hysteresis_band():
    """Once forward, a 20-degree error must NOT reopen the turning -- that is
    the chatter the deadband alone leaves behind."""
    agent = _agent(deadband=15.0, release=30.0)
    near = [np.cos(np.radians(5)), np.sin(np.radians(5))]
    assert _lock(agent, 0.0, near, "turn_left") == "move_forward"
    mid = [np.cos(np.radians(20)), np.sin(np.radians(20))]
    assert _lock(agent, 0.0, mid, "turn_right") == "move_forward", "the lock let go too early"
    far = [np.cos(np.radians(50)), np.sin(np.radians(50))]
    assert _lock(agent, 0.0, far, "turn_right") == "turn_right"


def test_it_is_symmetric():
    agent = _agent()
    below = [np.cos(np.radians(-10)), np.sin(np.radians(-10))]
    assert _lock(agent, 0.0, below, "turn_right") == "move_forward"


def test_forward_is_never_touched_and_arms_the_lock():
    agent = _agent()
    assert _lock(agent, 0.0, [1.0, 0.0], "move_forward") == "move_forward"
    assert agent._climb_turn_locked


def test_zero_keeps_the_movers_own_turns():
    agent = _agent(deadband=0.0)
    goal = [np.cos(np.radians(2)), np.sin(np.radians(2))]
    assert _lock(agent, 0.0, goal, "turn_left") == "turn_left"


# ------------------------------------------- the lock has to know when to stop

def test_it_yields_when_forward_is_not_moving_the_agent():
    """Measured (outputs/mf5_pass2_v19): with the deadband alone, 328 of 332
    turns were suppressed and 340 forward actions produced 5.4 m of path. The
    mover's turns are not only alignment -- they are how it gets around
    things -- and the action alone does not say which."""
    agent = _agent(stuck_eps=0.05, last_action="move_forward")
    near = [np.cos(np.radians(5)), np.sin(np.radians(5))]
    agent._climb_last_xy = np.zeros(2)          # it was here last step too
    assert _lock(agent, 0.0, near, "turn_left") == "turn_left", "walked into the wall"
    assert agent.stats["climb_turn_yield_stuck"] == 1


def test_it_does_not_yield_while_the_agent_is_actually_moving():
    agent = _agent(stuck_eps=0.05, last_action="move_forward")
    near = [np.cos(np.radians(5)), np.sin(np.radians(5))]
    agent._climb_last_xy = np.array([-0.25, 0.0])   # a full step back
    assert _lock(agent, 0.0, near, "turn_left") == "move_forward"


def test_a_run_of_suppressions_is_capped():
    agent = _agent(cap=3)
    near = [np.cos(np.radians(5)), np.sin(np.radians(5))]
    outs = [_lock(agent, 0.0, near, "turn_left") for _ in range(5)]
    assert outs[:3] == ["move_forward"] * 3
    assert "turn_left" in outs[3:], "the lock never let a turn through"
    assert agent.stats["climb_turn_yield_run"] == 1


# ------------------------------------------------ face the flight, then climb

def _aligner(limit=20.0, aligned=False, turns=0, axis=(0.0, 1.0), flight=True):
    """The alignment reads the FLIGHT AXIS; the carrot is only the fallback."""
    cfg = make_cfg()
    cfg.agent.climb_align_first_deg = limit
    cfg.agent.climb_flight_carrot = True
    pursuit = None
    if flight:
        pursuit = SimpleNamespace(n_cells=40, foot_xy=np.zeros(2),
                                  top_xy=np.asarray(axis, dtype=float) * 3.0)
    return SimpleNamespace(cfg=cfg, stats={}, _climb_aligned=aligned,
                           _climb_align_turns=turns, _climb_direction=1,
                           floors=SimpleNamespace(pursuit_flight=pursuit),
                           _flight_carrot=lambda frame, xy: np.array([0.0, 1.0]))


def _align(agent, heading_deg):
    return ClimbPolicy._climb_align_action(agent, _frame(heading_deg), np.zeros(2))


def test_it_turns_in_place_until_it_faces_the_flight():
    """The probe climbs this flight in a third of a run's steps, and the one
    thing it does differently is start aligned with it."""
    agent = _aligner()
    assert _align(agent, 0.0) == "turn_left", "the tread is 90 degrees away"
    assert not agent._climb_aligned


def test_once_facing_it_the_climb_proceeds():
    agent = _aligner()
    assert _align(agent, 80.0) is None, "10 degrees off is facing it"
    assert agent._climb_aligned


def test_it_falls_back_to_the_carrot_without_a_flight():
    agent = _aligner(flight=False)
    assert _align(agent, 0.0) == "turn_left"


def test_a_descent_faces_the_other_way_along_the_same_flight():
    """`foot_xy` is the mouth on THIS storey, so a descent's axis is reversed."""
    agent = _aligner(axis=(0.0, 1.0))
    agent._climb_direction = -1
    assert _align(agent, -90.0) is None, "facing down the flight is aligned"


def test_it_can_never_spin_forever():
    agent = _aligner(turns=12)          # a full revolution at 30 degrees
    assert _align(agent, 0.0) is None
    assert agent.stats["climb_align_gave_up"] == 1


def test_zero_skips_the_alignment():
    agent = _aligner(limit=0.0)
    assert _align(agent, 0.0) is None
