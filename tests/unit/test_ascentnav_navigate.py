"""The commit -> STOP path of the ascentnav transcription (S71).

`Ascent_Policy._navigate` (`ascent_policy.py:927-990`) and the gate latch in
`Map_Controller._update_object_map_with_stair_and_person`
(`map_controller.py:771-776`). Every rule here was a divergence in the
previous port, and each is pinned to the reference line it was transcribed
from.
"""
from __future__ import annotations

import numpy as np
import pytest

from .test_ascentnav_stairs import DEPTH, _agent, _det, _wall_frame


def _committed(**over):
    """An agent with a target cloud right in front of it."""
    a = _agent(**over)
    a.object_map.clouds = {a.target: np.array([[0.8, 0.0, 0.5, 1.0]])}
    a.object_map._map[:] = 0
    a.obstacle_map._floor_num_steps = 60           # past the reinit window
    a.cur_dis_to_goal = 0.8
    return a


def test_arrival_stops_only_when_the_gate_is_latched():
    a = _committed()
    a.cur_dis_to_goal = 0.5
    a._double_check_goal = True
    assert a._navigate(np.zeros(2), 0.0, np.array([0.5, 0.0])) == "stop"
    assert a.approach_stop_reason == "nearest_point"


def test_an_ungated_arrival_burns_the_cloud_and_explores_on_the_same_step():
    """`:967-975`: not LEFT, not a turn -- the exploration policy acts now."""
    a = _committed()
    a.cur_dis_to_goal = 0.5
    a.object_map._map[300:310, 300:310] = 1
    a.obstacle_map.frontiers = np.array([[3.0, 0.0]])
    action = a._navigate(np.zeros(2), 0.0, np.array([0.5, 0.0]))
    assert action == "move_forward", "the explore policy drove at the frontier"
    assert a.object_map.clouds == {}
    assert a.object_map._disabled_object_map[300:310, 300:310].all()
    assert a._try_to_navigate is False and a._try_to_navigate_step == 0
    assert a.stats["give_up_unverified"] == 1


def test_the_gate_survives_a_failed_approach():
    """The reference never clears `_double_check_goal` on the failure path.
    The port re-armed its gate on every give-up, so approach two paid for
    approach one's refusal."""
    a = _committed()
    a._double_check_goal = True
    a.cur_dis_to_goal = 0.5
    a.stats["give_up_unverified"] = 0
    a._give_up_target("abandon", np.zeros(2), 0.0)
    assert a._double_check_goal is True


def test_the_stall_test_uses_the_previous_steps_distance():
    """`:962, :977`: `min_distance_xy` is overwritten every in-band step, so
    the second step inside the metre compares against the first, not against
    a running minimum. The first in-band step is always FORWARD."""
    a = _committed()
    a._double_check_goal = True
    a.cur_dis_to_goal = 0.8
    assert a._navigate(np.zeros(2), 0.0, np.array([0.8, 0.0])) == "move_forward"
    assert a.min_distance_xy == pytest.approx(0.8)
    a.cur_dis_to_goal = 0.75                       # closed 0.05: stalled
    assert a._navigate(np.zeros(2), 0.0, np.array([0.8, 0.0])) == "stop"


def test_progress_inside_the_metre_keeps_driving():
    a = _committed()
    a._double_check_goal = True
    a.cur_dis_to_goal = 0.95
    a._navigate(np.zeros(2), 0.0, np.array([0.9, 0.0]))
    a.cur_dis_to_goal = 0.75                       # closed 0.2
    assert a._navigate(np.zeros(2), 0.0, np.array([0.9, 0.0])) == "move_forward"
    assert a.min_distance_xy == pytest.approx(0.75)


def test_min_distance_is_not_reset_by_a_failure():
    """F11: only the episode reset touches it (`:216`)."""
    a = _committed()
    a.min_distance_xy = 0.7
    a.obstacle_map.frontiers = np.array([[3.0, 0.0]])
    a._give_up_target("unverified", np.zeros(2), 0.0)
    assert a.min_distance_xy == pytest.approx(0.7)


def test_a_policy_stop_far_from_the_cloud_is_an_episode_stop():
    """`:980, :990` -- the raw network action is returned; index 0 ends the
    episode. The port rewrote it to FORWARD."""
    from .test_ascent_agent import _Driver

    a = _committed(driver=_Driver(action=None))
    a.cur_dis_to_goal = 3.0
    assert a._navigate(np.zeros(2), 0.0, np.array([3.0, 0.0])) == "stop"
    assert a.stats["policy_stop_honoured"] == 1 and a._state == "done"


def test_the_abandon_counter_is_cumulative_and_checked_after_the_policy():
    """`:939, :981-989`: 100 navigate steps over the EPISODE, not per approach;
    tested after the mover ran, only on the far branch."""
    from .test_ascent_agent import _Driver

    a = _committed(driver=_Driver(action="turn_left"))
    a.cur_dis_to_goal = 3.0
    a.obstacle_map.frontiers = np.array([[3.0, 3.0]])
    a._try_to_navigate_step = 98
    assert a._navigate(np.zeros(2), 0.0, np.array([3.0, 0.0])) == "turn_left"   # 99
    a.obstacle_map._floor_num_steps = 60
    action = a._navigate(np.zeros(2), 0.0, np.array([3.0, 0.0]))                # 100 -> give up -> explore
    assert action == "turn_left", "the explore policy drove (the stub mover turns) on the same step"
    assert a.stats["give_up_abandon"] == 1 and a._try_to_navigate_step == 0
    assert a._state == "explore"


def test_the_gate_latches_only_on_a_step_after_navigation_began():
    """`map_controller.py:771-776`: the latch needs `try_to_navigate` set on a
    PRIOR dispatch, a target detection this frame, and the PREVIOUS step's
    cosine >= 0.15 (F1). Structurally, no STOP before the second navigate step."""
    from osg.perception.detector import StubDetector

    a = _agent()
    a._done_initializing = True
    a.obstacle_map._floor_num_steps = 5
    a._blip_cosine = 0.9
    det = StubDetector(); det.push([_det(score=0.95)]); a.detector = det
    f = _wall_frame([0, 0.88, 0], [1, 0.88, 0], range_m=2.0)
    a.act(f)                                       # exploring: not navigating yet
    assert a._double_check_goal is False
    a._try_to_navigate = True
    a.obstacle_map._floor_num_steps = 6
    a._blip_cosine = 0.9                           # what the PREVIOUS step's value map scored (F1)
    det.push([_det(score=0.95)])
    a.act(f)
    assert a._double_check_goal is True


def test_a_cosine_below_the_bar_never_latches():
    from osg.perception.detector import StubDetector

    a = _agent()
    a._done_initializing = True
    a._try_to_navigate = True
    a._blip_cosine = 0.1
    a.obstacle_map._floor_num_steps = 5
    det = StubDetector(); det.push([_det(score=0.95)]); a.detector = det
    a.act(_wall_frame([0, 0.88, 0], [1, 0.88, 0], range_m=2.0))
    assert a._double_check_goal is False


def test_every_target_detection_is_ingested():
    """A4: `map_controller.py:754-768` loops every detection with its own mask.
    The port took the argmax."""
    from osg.perception.detector import StubDetector

    a = _agent()
    a._done_initializing = True
    a.obstacle_map._floor_num_steps = 5
    calls = []
    a.object_map.update_map = lambda *args, **kw: calls.append(args[2].sum())
    det = StubDetector(); det.push([_det(score=0.9, box=(0, 0, 60, 60)),
                                    _det(score=0.85, box=(100, 100, 130, 130)),
                                    _det(score=0.81, box=(200, 200, 210, 210))])
    a.detector = det
    a.act(_wall_frame([0, 0.88, 0], [1, 0.88, 0], range_m=2.0))
    assert len(calls) == 3 and sorted(calls) == [100, 900, 3600]


def test_the_first_step_of_a_floor_ingests_nothing():
    """A11: `map_controller.py:734-735` skips the whole block while
    `_floor_num_steps == 0`."""
    from osg.perception.detector import StubDetector

    a = _agent()
    a._done_initializing = True
    det = StubDetector(); det.push([_det(score=0.9)]); a.detector = det
    a.act(_wall_frame([0, 0.88, 0], [1, 0.88, 0], range_m=2.0))
    assert not a.object_map.has_object(a.target)
    assert a.obstacle_map._floor_num_steps == 1


def test_no_stop_is_possible_during_the_opening_turns():
    """A13 / H1: 13 LEFTs (`_initialize`, `_initialize_step > 11`) before the
    goal is even consulted (`act` :567-577)."""
    a = _agent(initial_scan=True)
    a.object_map.clouds = {a.target: np.array([[0.5, 0.0, 0.5, 1.0]])}
    a._double_check_goal = True
    f = _wall_frame([0, 0.88, 0], [1, 0.88, 0])
    actions = [a.act(f) for _ in range(13)]
    assert actions == ["turn_left"] * 13
    assert a._done_initializing is True
