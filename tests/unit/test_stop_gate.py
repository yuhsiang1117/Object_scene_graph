"""The STOP gate (agent/stop_gate.py): a second recogniser decides whether the
agent may STOP on what it is looking at. Habitat-free; the recogniser is
stubbed, so these pin the decision logic, not a model."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from osg.agent.state import TURN_ACTION, State
from osg.agent.stop_gate import StopGate, background_for, crop_with_margin
from osg.objects.presence import PresenceState


class _Profiler:
    def timeit(self, name):
        import contextlib
        return contextlib.nullcontext()


class _Layer:
    def __init__(self, tracks):
        self._tracks = {t.id: t for t in tracks}
        self.presence_filter = SimpleNamespace(
            apply_reading=lambda track, present, r, q: setattr(
                track.presence, "log_odds", track.presence.log_odds - 1.0))

    def get(self, tid):
        return self._tracks.get(tid)


def _nav(mode="vlm", det=True, per_track=2, per_episode=4):
    track = SimpleNamespace(id=7, identity_rejections=0, presence=PresenceState(log_odds=3.0))
    cfg = SimpleNamespace(
        verification=SimpleNamespace(
            stop_gate=mode, stop_gate_max_per_track=per_track,
            stop_gate_max_per_episode=per_episode, stop_gate_recall=0.9, stop_gate_q=0.2,
            stop_gate_clip_margin=0.0),
        scene_graph=SimpleNamespace(presence=SimpleNamespace(min_presence=0.45)),
        detector=SimpleNamespace(vocabulary=["box", "lamp", "book"]),
    )
    nav = SimpleNamespace(
        cfg=cfg, stats={}, step_count=10, target="cracker box", _candidate_id=7,
        object_layer=_Layer([track]), profiler=_Profiler(), state=State.DONE,
        released=0, held=0,
    )
    nav._best_target_detection = lambda frame: (
        SimpleNamespace(label="cracker box", score=0.8, bbox_xyxy=np.array([10, 10, 50, 60]))
        if det else None)

    def release():
        nav.released += 1
        nav._candidate_id = None
        nav.state = State.EXPLORE

    def hold():
        nav.held += 1

    nav.release_candidate = release
    nav.hold_explore = hold
    return nav, track


def _gate(nav, verdict):
    gate = StopGate(nav)
    gate._judge = lambda frame, det: verdict
    return gate


def test_off_never_touches_a_stop():
    nav, _ = _nav(mode="off")
    assert _gate(nav, False).check(SimpleNamespace()) is None
    assert nav.stats == {}


def test_an_unknown_mode_fails_loudly():
    nav, _ = _nav(mode="maybe")
    with pytest.raises(ValueError):
        StopGate(nav)


def test_a_no_releases_the_candidate_without_an_attempt():
    nav, track = _nav()
    action = _gate(nav, False).check(SimpleNamespace())
    assert action == TURN_ACTION
    assert nav.released == 1 and nav.held == 1
    assert track.identity_rejections == 1
    # held just under the presence bar, as a failed attempt leaves it
    assert track.presence.p < 0.45
    assert nav.stats["stop_gate_reject"] == 1


def test_a_yes_lets_the_stop_stand():
    nav, track = _nav()
    assert _gate(nav, True).check(SimpleNamespace()) is None
    assert nav.released == 0 and track.identity_rejections == 0
    assert nav.stats["stop_gate_accept"] == 1


def test_an_error_is_no_information():
    nav, track = _nav()
    assert _gate(nav, None).check(SimpleNamespace()) is None
    assert track.identity_rejections == 0 and nav.stats["stop_gate_errors"] == 1


def test_a_stop_with_nothing_in_view_is_not_judged():
    """A stale-anchor stop, or the last metre of a path: the absence sensor has
    already spoken and there is no object in the frame to call wrong."""
    nav, _ = _nav(det=False)
    gate = _gate(nav, False)
    assert gate.check(SimpleNamespace()) is None
    assert nav.stats["stop_gate_nothing_to_judge"] == 1


def test_a_track_rejected_enough_is_let_stop():
    nav, track = _nav(per_track=1)
    gate = _gate(nav, False)
    assert gate.check(SimpleNamespace()) == TURN_ACTION
    nav._candidate_id = 7          # committed to the same track again
    assert gate.check(SimpleNamespace()) is None
    assert nav.stats["stop_gate_track_cap"] == 1 and track.identity_rejections == 1


def test_an_episode_rejects_a_bounded_number_of_stops():
    nav, _ = _nav(per_track=10, per_episode=2)
    gate = _gate(nav, False)
    for _ in range(2):
        nav._candidate_id = 7
        assert gate.check(SimpleNamespace()) == TURN_ACTION
    nav._candidate_id = 7
    assert gate.check(SimpleNamespace()) is None
    assert nav.stats["stop_gate_episode_cap"] == 1


def test_reset_clears_the_caps():
    nav, _ = _nav(per_track=1)
    gate = _gate(nav, False)
    gate.check(SimpleNamespace())
    gate.reset()
    nav._candidate_id = 7
    assert gate.check(SimpleNamespace()) == TURN_ACTION


def test_the_confusers_are_the_episodes_detector_vocabulary():
    # `box` is a whole-word part of the query and the detector drops it for
    # this episode, so it is not a confuser either.
    assert background_for("cracker box", ["box", "lamp", "book"]) == ["lamp", "book"]


def test_the_crop_grows_the_box_and_stays_in_the_frame():
    rgb = np.zeros((100, 200, 3), np.uint8)
    assert crop_with_margin(rgb, [50, 40, 90, 60]).shape[:2] == (32, 64)
    assert crop_with_margin(rgb, [0, 0, 200, 100]).shape[:2] == (100, 200)
