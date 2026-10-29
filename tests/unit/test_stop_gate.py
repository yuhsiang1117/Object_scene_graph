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
    gate._judge = lambda frame, det, **kw: verdict
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
    gate._judge = lambda rgb, det, **kw: seen.append((rgb.shape, det.score)) or False
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
    gate._judge = lambda rgb, det, **kw: scores.append(det.score) or True
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
    gate._clip_verdict = lambda crop, **kw: verdicts[seen.append(1) or len(seen)]
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


def _live_nav(on=True, mode="clip"):
    nav, track = _nav(mode=mode)
    nav.cfg.verification.stop_gate_vet_live = on
    track.label = "cracker box"
    track.best_crop = np.zeros((40, 40, 3), np.uint8)
    track.best_cam_xy = np.array([1.0, 0.0])
    track.best_bbox_px = 1600.0
    nav.object_layer.center_of = lambda t: np.array([0.0, 0.5, 0.0])
    return nav, track


def test_a_live_track_is_judged_once_and_a_no_puts_it_under_the_bar():
    """`verification.stop_gate_vet_live`: the toy airplane on a handrail is a
    commit candidate; one reading through the presence filter, logged with the
    range its crop was taken from."""
    nav, track = _live_nav()
    gate = StopGate(nav)
    calls = []
    gate._clip_verdict = lambda crop, **kw: calls.append(1) or False
    track.presence.log_odds = -0.1                # p ~0.475, just over the 0.45 bar
    assert gate.vet_live(track) is False
    assert gate.vet_live(track) is True           # judged once: no second reading
    assert len(calls) == 1 and nav.stats["live_vet_rejected"] == 1
    assert nav.stats["live_vet_log"][0]["range_m"] == pytest.approx(1.0)


def test_a_yes_or_no_opinion_keeps_the_candidate():
    nav, track = _live_nav()
    gate = StopGate(nav)
    gate._clip_verdict = lambda crop, **kw: True
    assert gate.vet_live(track) is True
    nav, track = _live_nav()
    gate = StopGate(nav)
    gate._clip_verdict = lambda crop, **kw: None
    assert gate.vet_live(track) is True and "live_vet_log" not in nav.stats


def test_live_vetting_leaves_prior_proposal_and_unflagged_tracks_alone():
    for on, mode, attrs in ((False, "clip", {}), (True, "vlm", {}),
                            (True, "clip", {"from_prior": True, "seen_live": False}),
                            (True, "clip", {"proposal_only": True})):
        nav, track = _live_nav(on=on, mode=mode)
        for k, v in attrs.items():
            setattr(track, k, v)
        gate = StopGate(nav)
        gate._clip_verdict = lambda crop, **kw: pytest.fail("judged")
        assert gate.vet_live(track) is True


class _Encoder:
    """Features by hand: a crop carries its own unit feature in pixel (0, 0),
    channels 1-3 (axis 0: the target text's direction, axis 1: a confuser's,
    axis 2: appearance orthogonal to both)."""

    def encode_image(self, crop):
        return np.asarray(crop[0, 0, 1:4], dtype=float) if crop.shape[2] > 3 else None

    def encode_text(self, texts):
        return np.array([[1.0, 0.0, 0.0] if "cracker box" in t else [0.0, 1.0, 0.0]
                         for t in texts])


def _crop(feat):
    c = np.zeros((4, 4, 4), np.float32)
    c[0, 0, 1:4] = np.asarray(feat, dtype=float) / np.linalg.norm(feat)
    return c


def _exemplar_nav(mode, tau=0.79):
    nav, _ = _nav(mode="clip")
    nav.cfg.verification.stop_gate_exemplar = mode
    nav.cfg.verification.stop_gate_exemplar_tau = tau
    nav.region_proposer = SimpleNamespace(encoder=_Encoder())
    return nav


def _box(tid, feat, label="cracker box", from_prior=True):
    return SimpleNamespace(id=tid, label=label, from_prior=from_prior, best_crop=_crop(feat))


def test_the_exemplars_are_the_prior_crops_the_text_margin_accepts():
    """`verification.stop_gate_exemplar`: pass 1's pictures of the target, minus
    the ones the text margin calls something else, snapshotted at load."""
    nav = _exemplar_nav("log")
    tracks = [_box(1, [1, 0, 1]), _box(2, [0, 1, 0]), _box(3, [1, 0, 0], label="lamp"),
              _box(4, [1, 0, 0], from_prior=False)]
    _prior_layer(nav, tracks)
    gate = StopGate(nav)
    assert gate.vet_prior() is None          # vetting off: the snapshot is still taken
    assert [tid for tid, _ in gate._exemplar_set()] == [1]
    assert nav.stats["stop_gate_exemplars"] == 1
    tracks[0].best_crop = _crop([0, 1, 0])   # a later live crop does not move the set
    assert [tid for tid, _ in gate._exemplar_set()] == [1]


def test_log_mode_records_the_cosine_and_decides_by_the_text_margin():
    nav = _exemplar_nav("log")
    _prior_layer(nav, [_box(1, [1, 0, 1])])
    gate = StopGate(nav)
    # The text says yes, the appearance says it is not the instance: log keeps yes.
    assert gate._clip_verdict(_crop([1, 0, -0.5])) is True
    r = gate.last_reading
    assert r["n_exemplars"] == 1 and r["exemplar_cos"] < 0.79 and r["margin"] > 0


def test_on_mode_decides_by_the_exemplar_and_never_by_the_tracks_own_crop():
    nav = _exemplar_nav("on")
    _prior_layer(nav, [_box(1, [1, 0, 1])])
    gate = StopGate(nav)
    assert gate._clip_verdict(_crop([1, 0, -0.5])) is False      # the text margin passed it
    assert gate._clip_verdict(_crop([0.3, 0.4, 1.0])) is True    # the text margin failed it
    # Judging prior track 1 itself: no other exemplar, so the text margin decides.
    assert gate._clip_verdict(_crop([0.3, 0.4, 1.0]), exclude_id=1) is False
    assert "exemplar_cos" not in gate.last_reading


def test_without_exemplars_on_mode_is_the_text_margin():
    nav = _exemplar_nav("on")
    _prior_layer(nav, [])
    gate = StopGate(nav)
    assert gate._clip_verdict(_crop([1, 0, -0.5])) is True
    assert gate.last_reading == {"margin": pytest.approx(0.894, abs=1e-3), "n_exemplars": 0}


def test_a_bad_exemplar_mode_fails_loudly():
    with pytest.raises(ValueError):
        StopGate(_exemplar_nav("always"))


def test_a_live_no_is_identity_evidence_when_asked():
    """`stop_gate_vet_live_identity`: the detector re-detects a false positive
    every time it looks, so a presence reading alone does not keep it out."""
    for on, want in ((False, 0), (True, 1)):
        nav, track = _live_nav()
        nav.cfg.verification.stop_gate_vet_live_identity = on
        gate = StopGate(nav)
        gate._clip_verdict = lambda crop, **kw: False
        gate.vet_live(track)
        assert track.identity_rejections == want
    nav, track = _live_nav()
    nav.cfg.verification.stop_gate_vet_live_identity = True
    gate = StopGate(nav)
    gate._clip_verdict = lambda crop, **kw: True
    gate.vet_live(track)
    assert track.identity_rejections == 0


def test_revetting_judges_only_a_better_view_and_a_bounded_number_of_times():
    from osg.agent.stop_gate import MAX_LIVE_VETS

    nav, track = _live_nav()
    nav.cfg.verification.stop_gate_vet_live_revet = True
    track.best_score = 0.4
    gate = StopGate(nav)
    calls = []
    gate._clip_verdict = lambda crop, **kw: calls.append(track.best_score) or False
    gate.vet_live(track)
    gate.vet_live(track)                      # same crop: not judged again
    assert calls == [0.4]
    for i in range(MAX_LIVE_VETS + 2):        # every round a better detection
        track.best_score = 0.5 + 0.05 * i
        gate.vet_live(track)
    assert len(calls) == MAX_LIVE_VETS
    assert nav.stats["live_vet_revets"] == MAX_LIVE_VETS - 1
    assert [v["n"] for v in nav.stats["live_vet_log"]] == list(range(1, MAX_LIVE_VETS + 1))


def test_without_revetting_a_better_view_is_not_judged():
    nav, track = _live_nav()
    track.best_score = 0.4
    gate = StopGate(nav)
    calls = []
    gate._clip_verdict = lambda crop, **kw: calls.append(1) or True
    gate.vet_live(track)
    track.best_score = 0.9
    gate.vet_live(track)
    assert calls == [1]


def test_a_sticky_no_holds_until_a_better_view_says_yes():
    """`stop_gate_vet_live_sticky`: the detector's repeats of the same view
    raise presence, but only a better crop judged "yes" brings it back."""
    nav, track = _live_nav()
    nav.cfg.verification.stop_gate_vet_live_sticky = True
    nav.cfg.verification.stop_gate_vet_live_revet = True
    track.best_score = 0.4
    track.presence.log_odds = 3.0                 # well over the bar after one reading
    gate = StopGate(nav)
    answers = [False, True]
    gate._clip_verdict = lambda crop, **kw: answers.pop(0)
    assert gate.vet_live(track) is False          # held although presence is high
    track.presence.log_odds = 5.0                 # the detector re-detects it
    assert gate.vet_live(track) is False          # same view: the no stands
    assert nav.stats["live_vet_held_steps"] == 1
    track.best_score = 0.6                        # a better view, judged yes
    assert gate.vet_live(track) is True
    assert gate.vet_live(track) is True           # and the yes stands
    assert answers == []


def test_without_sticky_presence_decides_after_a_no():
    nav, track = _live_nav()
    track.presence.log_odds = 3.0
    gate = StopGate(nav)
    gate._clip_verdict = lambda crop, **kw: False
    assert gate.vet_live(track) is True           # still over the bar after one reading


def test_with_yes_final_a_better_view_never_takes_a_yes_back():
    """`stop_gate_vet_live_yes_final`: a revet is a held track's second chance;
    a "yes" stands (00821 pitcher: "yes" at margin 0.019, flipped at -0.021)."""
    nav, track = _live_nav()
    nav.cfg.verification.stop_gate_vet_live_sticky = True
    nav.cfg.verification.stop_gate_vet_live_revet = True
    nav.cfg.verification.stop_gate_vet_live_yes_final = True
    track.best_score = 0.4
    track.presence.log_odds = 3.0
    gate = StopGate(nav)
    answers = [True, False]
    gate._clip_verdict = lambda crop, **kw: answers.pop(0)
    assert gate.vet_live(track) is True
    track.best_score = 0.6                        # a better view: not judged again
    assert gate.vet_live(track) is True
    assert answers == [False] and nav.stats.get("live_vet_revets", 0) == 0


def test_with_yes_final_a_no_is_still_judged_again():
    nav, track = _live_nav()
    nav.cfg.verification.stop_gate_vet_live_sticky = True
    nav.cfg.verification.stop_gate_vet_live_revet = True
    nav.cfg.verification.stop_gate_vet_live_yes_final = True
    track.best_score = 0.4
    track.presence.log_odds = 3.0
    gate = StopGate(nav)
    answers = [False, True]
    gate._clip_verdict = lambda crop, **kw: answers.pop(0)
    assert gate.vet_live(track) is False
    track.best_score = 0.6
    assert gate.vet_live(track) is True
    assert answers == [] and nav.stats["live_vet_revets"] == 1


# ------------------------------------------- the ground-truth oracle (diagnostic)

def test_the_oracle_admits_the_object_and_where_it_was_and_nothing_else():
    from osg.eval.vet_oracle import VetOracle
    o = VetOracle([0.0, 0.8, 0.0], [5.0, 0.8, 5.0], radius_m=1.0)
    assert o.admits([0.9, 0.8, 0.0])                  # TRUE: within 1 m
    assert o.admits([5.0, 1.5, 5.6])                  # STALE: where pass 1 saw it
    assert not o.admits([1.2, 0.8, 0.0])              # past the radius
    assert not o.admits([0.0, 3.6, 0.0])              # the storey above
    assert not o.admits(None)
    assert VetOracle([0.0, 0.8, 0.0], None, 1.0).admits([0.0, 0.8, 0.5])
    with pytest.raises(ValueError):
        VetOracle(None, None, 1.0)


def _oracle_gate(mode, admits):
    nav, track = _live_nav()
    nav.cfg.verification.stop_gate_vet_oracle = mode
    gate = StopGate(nav)
    gate.oracle = SimpleNamespace(admits=lambda centre: admits)
    return nav, track, gate


def test_oracle_fp_holds_a_false_positive_for_the_episode_and_leaves_the_rest_to_the_arm():
    nav, track, gate = _oracle_gate("fp", False)
    gate._clip_verdict = lambda crop, **kw: pytest.fail("a held track is not judged")
    track.from_prior, track.seen_live = True, False   # prior and proposal tracks are held too
    assert gate.vet_live(track) is False and gate.vet_live(track) is False
    assert nav.stats["oracle_held_tracks"] == [7] and nav.stats["oracle_held_steps"] == 2
    nav, track, gate = _oracle_gate("fp", True)
    gate._clip_verdict = lambda crop, **kw: False     # the arm's own "no" still reads down
    track.presence.log_odds = -0.1
    assert gate.vet_live(track) is False and nav.stats["live_vet_rejected"] == 1


def test_oracle_full_replaces_the_recogniser_before_the_walk():
    nav, track, gate = _oracle_gate("full", True)
    gate._clip_verdict = lambda crop, **kw: pytest.fail("the oracle is the recogniser")
    gate.vet_live(track)
    assert nav.stats["live_vet_accepted"] == 1 and nav.stats["live_vet_log"][0]["oracle"] is True
    nav, _ = _nav(mode="clip")
    nav.cfg.verification.stop_gate_vet_prior = True
    nav.cfg.verification.stop_gate_vet_oracle = "full"
    tracks = [_prior_track(1), _prior_track(2)]
    readings = _prior_layer(nav, tracks)
    nav.object_layer.center_of = lambda t: np.array([float(t.id), 0.0, 0.0])
    gate = StopGate(nav)
    gate.oracle = SimpleNamespace(admits=lambda c: c[0] == 1.0)
    gate._clip_verdict = lambda crop, **kw: pytest.fail("the oracle is the recogniser")
    gate.vet_prior()
    assert readings == [(1, True, 0.9, 0.2), (2, False, 0.9, 0.2)]


def test_oracle_off_is_the_arm_and_on_without_a_layout_fails_loudly():
    nav, track = _live_nav()
    gate = StopGate(nav)                               # no stop_gate_vet_oracle key: off
    assert gate.oracle_admits(track) is True and "oracle_held_steps" not in nav.stats
    nav, track = _live_nav()
    nav.cfg.verification.stop_gate_vet_oracle = "fp"
    with pytest.raises(RuntimeError):
        StopGate(nav).vet_live(track)
    for retired in ("always", "refute", "refute_full"):   # refute*: v1, removed
        nav.cfg.verification.stop_gate_vet_oracle = retired
        with pytest.raises(ValueError):
            StopGate(nav)


def test_the_oracle_refutes_only_what_cannot_be_the_object_misplaced():
    from osg.eval.vet_oracle import VetOracle
    o = VetOracle([0.0, 0.8, 0.0], [5.0, 0.8, 5.0], radius_m=1.0)
    assert not o.refutes([2.0, 0.8, 0.0])             # past the radius, inside the band
    assert not o.refutes([5.0, 0.8, 7.4])             # near where pass 1 saw it
    assert o.refutes([2.6, 0.8, 0.0])                 # past the band
    assert o.refutes([0.0, 3.6, 0.0])                 # the storey above
    assert not o.refutes(None)                        # nothing to judge


def _refute_gate(mode, refutes):
    nav, track = _live_nav()
    nav.cfg.verification.stop_gate_vet_oracle = mode
    nav.cfg.scene_graph.presence.max_identity_rejections = 2
    gate = StopGate(nav)
    gate.oracle = SimpleNamespace(refutes=lambda centre: refutes,
                                  admits=lambda centre: pytest.fail("refute asks refutes"))
    return nav, track, gate


def test_oracle_sweep_retires_a_false_positive_at_candidacy_as_the_walk_would():
    """The "fp" hold LOST 11 episodes: a held false positive is never walked
    to, so its storey's floor-switch veto never lifts. Refuted, it is out of
    candidacy and the storey belief and under the bar -- and NOT absent: the
    walk to a false positive finds an object there, and a faked absence arms
    the twin rule (v1, s42 00878: the prior anchor struck off 690 times)."""
    nav, track, gate = _refute_gate("sweep", True)
    gate._clip_verdict = lambda crop, **kw: pytest.fail("a refuted track is not judged")
    assert gate.vet_live(track) is False
    assert track.identity_rejections == 2 and getattr(track, "absence_arrivals", 0) == 0
    assert track.presence.p < 0.45
    assert nav.stats["oracle_refuted_tracks"] == [7]
    assert nav.released == 1 and nav.held == 1        # it was the candidate: let go
    nav, track, gate = _refute_gate("sweep", False)
    gate._clip_verdict = lambda crop, **kw: False     # inside the band: the arm's recogniser
    track.presence.log_odds = -0.1
    assert gate.vet_live(track) is False and nav.stats["live_vet_rejected"] == 1
    assert track.identity_rejections == 0 and "oracle_refuted_tracks" not in nav.stats


def _sweep_gate(mode):
    nav, _ = _nav(mode="clip")
    nav.cfg.verification.stop_gate_vet_oracle = mode
    nav.cfg.scene_graph.presence.max_identity_rejections = 2
    tracks = [SimpleNamespace(id=tid, label=label, identity_rejections=0,
                              presence=PresenceState(log_odds=3.0), x=x)
              for tid, label, x in ((1, "cracker box", 0.0), (2, "cracker box", 9.0),
                                    (3, "lamp", 9.0), (4, "Cracker Box", 9.0))]
    nav.object_layer = SimpleNamespace(tracks=lambda: list(tracks),
                                       center_of=lambda t: np.array([t.x, 0.0, 0.0]))
    gate = StopGate(nav)
    gate.oracle = SimpleNamespace(refutes=lambda c: c[0] > 2.5,
                                  admits=lambda c: pytest.fail("sweep asks refutes"))
    return nav, tracks, gate


def test_the_sweep_retires_every_refuted_target_track_on_every_storey_once():
    """Retiring only the current storey's false positives (at candidacy) empties
    that storey's anchor while the others keep theirs: v1 s42 00873, the
    airplane's storey left at 44% coverage."""
    nav, tracks, gate = _sweep_gate("sweep")
    nav._candidate_id = None
    gate.oracle_sweep()
    assert nav.stats["oracle_refuted_tracks"] == [2, 4]   # the label, not the lamp
    assert [t.identity_rejections for t in tracks] == [0, 2, 0, 2]
    assert tracks[1].presence.p < 0.45 and nav.released == 0
    tracks[1].identity_rejections = 0                     # already retired: not again
    gate.oracle_sweep()
    assert nav.stats["oracle_refuted_tracks"] == [2, 4] and tracks[1].identity_rejections == 0


def test_the_sweep_is_a_no_op_in_every_other_mode():
    for mode in ("off", "fp", "full"):
        nav, tracks, gate = _sweep_gate(mode)
        gate.oracle_sweep()
        assert "oracle_refuted_tracks" not in nav.stats
        assert all(t.identity_rejections == 0 for t in tracks)
