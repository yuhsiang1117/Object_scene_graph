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


class _Verifier:
    def __init__(self, verdict=True, boom=False):
        self.verdict, self.boom, self.calls = verdict, boom, []

    def verify_bbox(self, rgb, bbox, target):
        self.calls.append((None if rgb is None else rgb.shape, tuple(bbox), target))
        if self.boom:
            raise RuntimeError("nim down")
        return self.verdict


def _at_stop(verifier=None, **over):
    """An agent one step from a gated arrival STOP, with a detection in hand."""
    import numpy as np
    a = _committed(**over)
    a._double_check_goal = True
    a.cur_dis_to_goal = 0.5
    a.verifier = verifier
    a._last_dets = [_det(score=0.9)]
    a._last_rgb = np.zeros((480, 640, 3), np.uint8)
    a.obstacle_map.frontiers = np.array([[3.0, 0.0]])
    return a


def test_the_stop_verifier_is_off_unless_asked_for():
    """S75 is an ADDITION -- the reference has no verifier. The default arm must
    not call one even when a verifier object is handed to the agent."""
    v = _Verifier(verdict=False)
    a = _agent()                                    # verify_on_stop defaults False
    assert a.verifier is None
    a = _at_stop(verifier=None)
    assert a._navigate(np.zeros(2), 0.0, np.array([0.5, 0.0])) == "stop"
    assert v.calls == []


def test_a_refused_stop_becomes_a_give_up_and_the_agent_keeps_exploring():
    v = _Verifier(verdict=False)
    a = _at_stop(verifier=v, verify_on_stop=True)
    action = a._navigate(np.zeros(2), 0.0, np.array([0.5, 0.0]))
    assert action != "stop", "the refusal must not end the episode"
    assert a.stats["verify_refused"] == 1 and a.stats["verify_calls"] == 1
    assert a.stats["give_up_vlm_refused"] == 1
    assert a._try_to_navigate is False and a.object_map.clouds == {}
    assert len(v.calls) == 1 and v.calls[0][2] == a.target


def test_a_confirmed_stop_still_stops():
    v = _Verifier(verdict=True)
    a = _at_stop(verifier=v, verify_on_stop=True)
    assert a._navigate(np.zeros(2), 0.0, np.array([0.5, 0.0])) == "stop"
    assert a.stats["verify_calls"] == 1 and "verify_refused" not in a.stats


def test_a_broken_verifier_fails_open_and_says_so():
    """An unreachable VLM must degrade to the unverified behaviour, not refuse
    every stop -- but the counter has to show it, or the arm measures nothing."""
    v = _Verifier(boom=True)
    a = _at_stop(verifier=v, verify_on_stop=True)
    assert a._navigate(np.zeros(2), 0.0, np.array([0.5, 0.0])) == "stop"
    assert a.stats["verify_errors"] == 1


def test_no_detection_in_hand_means_no_call_and_no_block():
    a = _at_stop(verifier=_Verifier(verdict=False), verify_on_stop=True)
    a._last_dets = []
    assert a._navigate(np.zeros(2), 0.0, np.array([0.5, 0.0])) == "stop"
    assert a.stats["verify_no_view"] == 1 and "verify_calls" not in a.stats


def test_the_stored_view_is_the_best_look_not_the_last_one():
    """S76: the live gate is blind when the detection has left the frame. The
    stored view is the highest-scoring look at the cloud being stopped on."""
    import numpy as np
    v = _Verifier(verdict=True)
    a = _at_stop(verifier=v, verify_on_stop=True, verify_stop_view="stored")
    strong, weak = np.full((480, 640, 3), 7, np.uint8), np.full((480, 640, 3), 9, np.uint8)
    a._remember_best_view([_det(score=0.55)], weak)
    a._remember_best_view([_det(score=0.95, box=(10, 10, 60, 60))], strong)
    a._remember_best_view([_det(score=0.60)], weak)          # later, but worse
    a._last_dets, a._last_rgb = [], None                     # nothing in frame now
    assert a._navigate(np.zeros(2), 0.0, np.array([0.5, 0.0])) == "stop"
    assert a.stats["verify_calls"] == 1 and a.stats["verify_view_stored"] == 1
    assert v.calls[0][0] == strong.shape and v.calls[0][1] == (10, 10, 60, 60)


def test_the_live_gate_would_have_had_nothing_to_ask_about():
    """The same situation on S75's setting: no live detection, no call."""
    import numpy as np
    v = _Verifier(verdict=False)
    a = _at_stop(verifier=v, verify_on_stop=True)            # verify_stop_view defaults live
    a._remember_best_view([_det(score=0.95)], np.zeros((480, 640, 3), np.uint8))
    a._last_dets, a._last_rgb = [], None
    assert a._navigate(np.zeros(2), 0.0, np.array([0.5, 0.0])) == "stop"
    assert a.stats["verify_no_view"] == 1 and v.calls == []


def test_burning_the_cloud_drops_the_view_it_described():
    """A stored view must never outlive the cloud it was evidence for."""
    import numpy as np
    a = _at_stop(verifier=_Verifier(), verify_on_stop=True, verify_stop_view="stored")
    a._remember_best_view([_det(score=0.9)], np.zeros((480, 640, 3), np.uint8))
    a.obstacle_map.frontiers = np.array([[3.0, 0.0]])
    a._give_up_target("unverified", np.zeros(2), 0.0)
    assert a._best_view is None


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


# ------------------------------------------- a failed attempt under 3 attempts

def test_a_failed_attempt_burns_the_cloud_it_stopped_on():
    """DualMap's protocol grants three attempts; the reference has one. Without
    `rearm` the next act re-derives the same goal from the same cloud and STOPs
    again, so the episode loop (`eval/episode.py`) would spend all three on one
    object. The hook is ASCENT's own false-positive path (`:915-922`)."""
    a = _committed()
    a._double_check_goal = True
    a.object_map._map[300:310, 300:310] = 1
    a._state = "done"
    a.rearm(1000)
    assert a.object_map.clouds == {}
    assert not a.object_map.has_object(a.target)
    assert a.object_map._disabled_object_map[300:310, 300:310].all()
    assert not a.object_map._map.any()
    assert a._try_to_navigate is False and a._try_to_navigate_step == 0
    assert a._double_check_goal is True, "the latched gate carries over"
    assert a._state == "explore"
    assert a.stats["give_up_failed_attempt"] == 1
    assert a.giveup_log[-1][1] == "failed_attempt"


def test_with_place_disabling_the_burn_covers_the_radius_around_the_cloud():
    """`failed_attempt_disables_place` (the OSG arm's switch) with its radius,
    0.5 m = 10 cells at 20 px/m: a re-detection one cell over is not a new goal."""
    a = _committed()
    a.cfg.verification.failed_attempt_disables_place = True
    a.cfg.scene_graph.fp_disable_radius_m = 0.5
    a.object_map._map[300, 300] = 1
    a.rearm(1000)
    d = a.object_map._disabled_object_map
    assert d[300, 300] and d[300, 309] and d[309, 300]
    assert not d[300, 312] and not d[312, 300]
    b = _committed()
    b.cfg.verification.failed_attempt_disables_place = False
    b.object_map._map[300, 300] = 1
    b.rearm(1000)
    assert b.object_map._disabled_object_map.sum() == 1, "off: the reference burn only"


def test_the_episode_loop_reaches_the_ascent_rearm():
    """`rearm_after_failed_attempt` finds `rearm` by name; the ASCENT agent has
    no object layer, so this is the only thing it does for this policy."""
    from types import SimpleNamespace

    from osg.eval.attempts import rearm_after_failed_attempt

    a = _committed()
    a.object_map._map[300:310, 300:310] = 1
    cfg = SimpleNamespace(agent=SimpleNamespace(max_steps=1000),
                          verification=SimpleNamespace())
    rearm_after_failed_attempt(a, cfg)
    assert a.object_map.clouds == {}
    assert a.stats["give_up_failed_attempt"] == 1


def test_a_swapped_in_detector_does_not_join_ascents_stair_fusion():
    """With ASCENT's own stair detector present, a target detector that names
    "stairs" (YOLOE, `mfv2_ascent_yoloe`) must not add to the stair mask --
    D-FINE never could, so the reference fusion is GroundingDINO's alone."""
    from .test_ascentnav_stairs import _det

    class _Stair:
        def mask(self, rgb):
            return np.zeros((480, 640), bool)

    a = _committed()
    a.stair_detector = _Stair()
    raw = [_det(score=0.9, box=(0, 0, 60, 60), label="stairs")]
    m = a._stair_det_mask(raw, np.zeros((480, 640, 3), np.uint8))
    assert m is not None and not m.any()
    a.stair_detector = None
    m = a._stair_det_mask(raw, np.zeros((480, 640, 3), np.uint8))
    assert m[0:60, 0:60].all(), "without one, the detector's stairs still count"
