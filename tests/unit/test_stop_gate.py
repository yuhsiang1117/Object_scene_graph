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


def test_the_generic_confusers_name_no_benchmark_object_and_are_fixed():
    from osg.agent.stop_gate import confuser_vocabulary, generic_confusers
    from osg.core.config.ycb import YCB_TARGET_LABELS

    generic = generic_confusers()
    assert "bottle" in generic and "chair" in generic and len(generic) == len(set(generic))
    multi_word_targets = {v for v in YCB_TARGET_LABELS.values() if " " in v}
    assert not multi_word_targets & set(generic)
    cfg = SimpleNamespace(verification=SimpleNamespace(stop_gate_confusers="generic"),
                          detector=SimpleNamespace(vocabulary=["tin can", "lamp"]))
    assert confuser_vocabulary(cfg) == generic
    cfg.verification.stop_gate_confusers = "detector"
    assert confuser_vocabulary(cfg) == ["tin can", "lamp"]
    cfg.verification.stop_gate_confusers = "both"
    with pytest.raises(ValueError):
        confuser_vocabulary(cfg)


# --------------------------------------------- the approach's close view (v2)

def _nav_v2(view="approach", unseen="allow", det=False, from_prior=False, seen_live=True):
    nav, track = _nav(mode="clip", det=det)
    nav.cfg.verification.stop_gate_view = view
    nav.cfg.verification.stop_gate_unseen = unseen
    nav.cfg.verification.stop_gate_close_m = 1.5
    track.from_prior, track.seen_live = from_prior, seen_live
    return nav, track


def _sighting(score=0.6):
    return SimpleNamespace(label="cracker box", score=score, bbox_xyxy=np.array([5, 5, 40, 40]))


def _frame():
    return SimpleNamespace(rgb=np.zeros((60, 80, 3), np.uint8))


def test_with_nothing_in_frame_the_best_close_sighting_is_judged():
    """Most STOPs end where the object has left the frame (49 of 66 on
    Benchmark v2's seed 42): judge what the approach saw within reach."""
    nav, track = _nav_v2()
    gate = StopGate(nav)
    seen = []
    gate._judge = lambda rgb, det: seen.append((rgb.shape, det.score)) or False
    gate.note_sighting(_frame(), _sighting(0.5), 1.2)
    gate.note_sighting(_frame(), _sighting(0.7), 1.4)
    gate.note_sighting(_frame(), _sighting(0.4), 0.9)
    assert gate.check(SimpleNamespace()) == TURN_ACTION
    assert seen == [((60, 80, 3), 0.7)], "the most confident close view, once"
    assert gate.log[-1]["view"] == "approach"
    assert track.identity_rejections == 1


def test_a_far_sighting_is_not_a_close_view():
    nav, _ = _nav_v2()
    gate = _gate(nav, False)
    gate.note_sighting(_frame(), _sighting(0.9), 2.4)
    gate.note_sighting(_frame(), _sighting(0.9), None)
    assert gate.close_view is None
    assert gate.check(SimpleNamespace()) is None
    assert nav.stats["stop_gate_nothing_to_judge"] == 1


def test_a_new_approach_forgets_the_last_ones_view():
    nav, _ = _nav_v2()
    gate = _gate(nav, False)
    gate.note_sighting(_frame(), _sighting(), 1.0)
    gate.new_approach()
    assert gate.check(SimpleNamespace()) is None


def test_the_frame_mode_keeps_no_view():
    nav, _ = _nav_v2(view="frame")
    gate = _gate(nav, False)
    gate.note_sighting(_frame(), _sighting(), 1.0)
    assert gate.close_view is None


def test_a_live_detection_is_still_judged_first():
    nav, _ = _nav_v2(det=True)
    gate = StopGate(nav)
    scores = []
    gate._judge = lambda rgb, det: scores.append(det.score) or True
    gate.note_sighting(_frame(), _sighting(0.6), 1.0)
    assert gate.check(SimpleNamespace()) is None
    assert scores == [0.8] and gate.log[-1]["view"] == "frame"


def test_unseen_reject_releases_a_stop_nothing_close_backs():
    nav, track = _nav_v2(unseen="reject")
    gate = _gate(nav, True)
    assert gate.check(SimpleNamespace()) == TURN_ACTION
    assert nav.released == 1 and track.identity_rejections == 1
    assert nav.stats["stop_gate_unseen_reject"] == 1
    assert gate.log[-1]["view"] == "unseen"


def test_unseen_reject_lets_the_stale_anchor_gamble_through():
    nav, track = _nav_v2(unseen="reject", from_prior=True, seen_live=False)
    assert _gate(nav, False).check(SimpleNamespace()) is None
    assert nav.released == 0 and track.identity_rejections == 0


def test_unseen_reject_respects_the_episode_cap():
    nav, _ = _nav_v2(unseen="reject")
    nav.cfg.verification.stop_gate_max_per_episode = 1
    gate = _gate(nav, True)
    assert gate.check(SimpleNamespace()) == TURN_ACTION
    nav._candidate_id = 7
    assert gate.check(SimpleNamespace()) is None
    assert nav.stats["stop_gate_episode_cap"] == 1


def _prior_layer(nav, tracks):
    readings = []
    layer = SimpleNamespace(
        presence_filter=SimpleNamespace(
            apply_reading=lambda t, present, r, q: readings.append((t.id, present, r, q))),
        tracks=lambda include_blacklisted=False: list(tracks),
    )
    nav.object_layer = layer
    return readings


def _prior_track(tid, label="cracker box", from_prior=True, crop=True):
    return SimpleNamespace(id=tid, label=label, from_prior=from_prior,
                           best_crop=np.zeros((8, 8, 3), np.uint8) if crop else None)


def test_vetting_reads_each_target_labelled_prior_track_through_presence():
    nav, _ = _nav(mode="clip")
    nav.cfg.verification.stop_gate_vet_prior = True
    tracks = [_prior_track(1), _prior_track(2), _prior_track(3, label="lamp"),
              _prior_track(4, from_prior=False), _prior_track(5, crop=False)]
    readings = _prior_layer(nav, tracks)
    gate = StopGate(nav)
    verdicts = {1: True, 2: False}
    seen = []
    gate._clip_verdict = lambda crop: verdicts[seen.append(1) or len(seen)]
    out = gate.vet_prior()
    assert readings == [(1, True, 0.9, 0.2), (2, False, 0.9, 0.2)]
    assert (out["judged"], out["accepted"], out["rejected"], out["no_crop"]) == (2, 1, 1, 1)
    assert nav.stats["prior_vet_rejected"] == 1


def test_vetting_is_off_unless_asked_and_never_in_vlm_mode():
    for mode, on in (("clip", False), ("vlm", True)):
        nav, _ = _nav(mode=mode)
        nav.cfg.verification.stop_gate_vet_prior = on
        readings = _prior_layer(nav, [_prior_track(1)])
        assert StopGate(nav).vet_prior() is None and readings == []


def test_bad_view_or_unseen_values_fail_loudly():
    for kw in ({"view": "best"}, {"unseen": "maybe"}):
        nav, _ = _nav_v2(**kw)
        with pytest.raises(ValueError):
            StopGate(nav)
