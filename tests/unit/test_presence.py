"""Presence belief: positive AND negative evidence (docs/DYNAMIC_SCENES.md, Phase 1).

The defect these guard is the one DualMap has: an object leaves the map on a
timer, so "I turned away" and "it is gone" are the same transition. Here they
must be different -- and the difference is decided by depth, so most of these
tests are a synthetic depth map plus an ellipsoid and an assertion about the
exact size of the resulting belief step.

Habitat-free by construction.
"""
import math

import numpy as np
import pytest

from osg.core.types import CameraIntrinsics, Detection, FrameData
from osg.objects.association import ObjectTrack
from osg.objects.ellipsoid import Ellipsoid
from osg.objects.presence import Expectation, PresenceFilter, RecallModel

W, H = 320, 240
R_CONST, Q = 0.6, 0.05
NEG = math.log((1 - R_CONST) / (1 - Q))  # one clean miss
POS = math.log(R_CONST / Q)  # one sighting


def frame(depth_value=5.0, frame_id=0):
    """Camera at the origin looking down +z (OpenCV), depth constant."""
    return FrameData(
        frame_id=frame_id,
        rgb=np.zeros((H, W, 3), np.uint8),
        depth=np.full((H, W), depth_value, np.float32),
        T_wc=np.eye(4),
        intrinsics=CameraIntrinsics.from_hfov(90.0, W, H),
    )


def track(tid=1, center=(0.0, 0.0, 2.0), axes=(0.25, 0.25, 0.25), label="mug"):
    return ObjectTrack(
        id=tid,
        label=label,
        ellipsoid=Ellipsoid(center=np.array(center, float), axes=np.array(axes, float), R=np.eye(3)),
    )


def filt(**kw):
    kw.setdefault("min_area_px", 100.0)
    return PresenceFilter(recall=RecallModel(constant=R_CONST), q_false_alarm=Q, **kw)


def detection_over(tr, f, label=None):
    """A detection whose mask covers the track's projection."""
    proj = tr.ellipsoid.project(f.intrinsics.K(), f.T_cw)
    x1, y1, x2, y2 = proj.bbox().astype(int)
    mask = np.zeros((H, W), bool)
    mask[max(0, y1):y2, max(0, x1):x2] = True
    return Detection(label=label or tr.label, score=0.9,
                     bbox_xyxy=np.array([x1, y1, x2, y2], float), mask=mask)


# --------------------------------------------------------- the core channel


def test_expected_but_missing_is_negative_evidence_of_exactly_the_right_size():
    """Object mapped at 2 m; the depth map says 5 m everywhere, i.e. we are
    seeing straight through to the wall behind where it used to be."""
    tr, f, pf = track(), frame(5.0), filt()
    before = tr.presence.log_odds
    pf.update([tr], f, [])
    assert tr.presence.log_odds == pytest.approx(before + NEG, abs=1e-12)
    assert tr.presence.n_missed == 1


def test_occlusion_leaves_the_belief_untouched():
    """A surface 1 m in front of a 2 m object: E=0. THE test -- conflating this
    with absence is the whole failure mode being fixed."""
    tr, f, pf = track(), frame(1.0), filt()
    before = tr.presence.log_odds
    pf.update([tr], f, [])
    assert tr.presence.log_odds == before
    assert tr.presence.n_missed == 0
    assert pf.expectation(tr, f) is None


def test_a_sighting_is_positive_evidence():
    tr, f, pf = track(), frame(2.0), filt(l_clamp_pos=100.0)
    before = tr.presence.log_odds
    pf.update([tr], f, [detection_over(tr, f)])
    assert tr.presence.log_odds == pytest.approx(before + POS, abs=1e-12)
    assert tr.presence.last_seen_kf == f.frame_id


def test_zero_detection_keyframe_still_updates_beliefs():
    """The early-return regression: ObjectLayer.update() used to bail before any
    presence work when no detection cleared its quality gate."""
    from osg.objects.object_layer import ObjectLayer

    pf = filt()
    layer = ObjectLayer(presence_filter=pf, min_det_score=0.9, min_det_bbox_px=1e9)
    tr = track()
    layer._tracks[tr.id] = tr
    before = tr.presence.log_odds
    layer.update(frame(5.0), [])  # nothing at all to admit
    assert tr.presence.log_odds < before, "no belief update on a zero-detection frame"


def test_a_detection_too_small_to_admit_is_still_a_sighting():
    """A detection below the node-creation quality bar proves something is
    there; counting it as a miss would be actively wrong."""
    from osg.objects.object_layer import ObjectLayer

    pf = filt()
    layer = ObjectLayer(presence_filter=pf, min_det_score=0.99, min_det_bbox_px=1e9)
    tr = track()
    layer._tracks[tr.id] = tr
    f = frame(2.0)
    before = tr.presence.log_odds
    layer.update(f, [detection_over(tr, f)])  # admitted=[] but presence sees it
    assert tr.presence.log_odds > before


# ------------------------------------------------------------ the E=0 gates


def test_behind_the_camera_is_not_expected():
    tr, f, pf = track(center=(0.0, 0.0, -2.0)), frame(5.0), filt()
    assert pf.expectation(tr, f) is None


def test_out_of_range_is_not_expected():
    tr, f, pf = track(center=(0.0, 0.0, 12.0)), frame(20.0), filt(range_m=(0.4, 6.0))
    assert pf.expectation(tr, f) is None


def test_too_small_to_see_is_not_expected():
    """Expectation must share the admission threshold, or every distant object
    generates false negatives."""
    tr, f = track(axes=(0.01, 0.01, 0.01)), frame(5.0)
    assert filt(min_area_px=1500.0).expectation(tr, f) is None


def test_mostly_out_of_frame_is_not_expected():
    tr, f, pf = track(center=(2.6, 0.0, 2.0)), frame(5.0), filt()
    assert pf.expectation(tr, f) is None


def test_unreadable_depth_is_not_expected():
    """Habitat writes 0 for invalid depth; refusing to conclude is the only safe
    reading of a hole in the depth map."""
    tr, pf = track(), filt()
    f = frame(0.0)  # every sample invalid
    assert pf.expectation(tr, f) is None


# --------------------------------------------------- identity vs. existence


def test_a_relabelled_detection_still_counts_as_a_sighting():
    """Association gates on category, so a mug re-detected as a bowl would
    otherwise collapse the mug's belief on a RELABEL rather than a removal."""
    tr, f, pf = track(label="mug"), frame(2.0), filt()
    before = tr.presence.log_odds
    pf.update([tr], f, [detection_over(tr, f, label="bowl")])
    assert tr.presence.log_odds > before


# ------------------------------------------------------- analytic behaviour


def test_k_clean_negatives_cross_the_threshold_where_the_closed_form_says():
    """k >= (l0 - l*) / |dl|. Predicted, not a golden value."""
    tr, pf = track(), filt()
    l0 = tr.presence.log_odds
    l_star = math.log(0.1 / 0.9)
    k_expected = math.ceil((l0 - l_star) / abs(NEG))
    for i in range(k_expected - 1):
        pf.update([tr], frame(5.0, frame_id=i), [])
    assert tr.presence.p > 0.1, "crossed early"
    pf.update([tr], frame(5.0, frame_id=k_expected), [])
    assert tr.presence.p <= 0.1, "did not cross when predicted"


def test_the_clamp_keeps_a_disbelieved_object_resurrectable():
    """No absorbing states: 50 misses then one sighting must recover."""
    tr, pf = track(), filt()
    for i in range(50):
        pf.update([tr], frame(5.0, frame_id=i), [])
    assert tr.presence.p < 0.01
    assert tr.presence.log_odds >= -pf.l_clamp
    f = frame(2.0, frame_id=99)
    for i in range(4):
        pf.update([tr], f, [detection_over(tr, f)])
    assert tr.presence.p > 0.5, "clamped belief could not recover"


def test_beliefs_start_where_a_single_sighting_justifies():
    assert track().presence.p == pytest.approx(0.8176, abs=1e-3)


# ------------------------------------------------------------ recall model


def test_constant_recall_is_the_fallback_when_no_model_is_fitted():
    m = RecallModel.load("/nonexistent/path/model.json", constant=0.42)
    assert m(Expectation(area_px=5000, depth_m=2.0)) == pytest.approx(0.42)


def test_fitted_weights_make_recall_depend_on_the_view():
    """Bigger and nearer must be easier to detect, or the negative updates are
    mis-sized in exactly the situations that matter."""
    m = RecallModel(weights=[-2.0, 0.6, -0.4, 0.0])
    near_big = m(Expectation(area_px=40000, depth_m=1.0))
    far_small = m(Expectation(area_px=800, depth_m=5.0))
    assert near_big > far_small


def test_recall_is_clamped_away_from_certainty():
    """r=1 would make one miss infinitely conclusive."""
    m = RecallModel(weights=[50.0, 0.0, 0.0, 0.0], ceil=0.95)
    assert m(Expectation(area_px=10, depth_m=1.0)) == pytest.approx(0.95)


# ------------------------------------------------------------- integration


def test_candidates_rank_by_presence_weighted_score():
    from osg.objects.object_layer import ObjectLayer

    layer = ObjectLayer()
    strong, weak = track(1, label="chair"), track(2, label="chair")
    for t, score in ((strong, 0.9), (weak, 0.8)):
        t.best_score = score
        t.observations = [None] * 5
        layer._tracks[t.id] = t
    assert [t.id for t in layer.candidates("chair")] == [1, 2]

    strong.presence.log_odds = -4.0  # looked, not found
    assert [t.id for t in layer.candidates("chair")] == [2, 1]


def test_min_presence_can_drop_a_disproved_candidate_entirely():
    from osg.objects.object_layer import ObjectLayer

    layer = ObjectLayer()
    t = track(1, label="chair")
    t.best_score, t.observations = 0.9, [None] * 5
    layer._tracks[t.id] = t
    t.presence.log_odds = -4.0  # p ~ 0.018
    # Default 0.0 keeps every track eligible: presence acts through RANKING
    # unless a caller explicitly asks for a floor.
    assert layer.candidates("chair") == [t]
    assert layer.candidates("chair", min_presence=0.1) == []


# ------------------------------------------------- target/vocabulary collision


def test_generic_vocabulary_entry_that_swallows_the_target_is_dropped():
    """Measured on the YCB benchmark: target "cracker box" plus a generic "box"
    in the vocabulary made YOLOE label every sighting "box", so the target was
    mapped 3 times and proposable zero times."""
    from osg.agent.nav_agent import target_vocabulary

    vocab = target_vocabulary("cracker box", ["chair", "box", "table"])
    assert vocab[0] == "cracker box"
    assert "box" not in vocab
    assert "chair" in vocab and "table" in vocab


def test_the_target_is_not_listed_twice():
    from osg.agent.nav_agent import target_vocabulary

    vocab = target_vocabulary("chair", ["chair", "table"])
    assert vocab.count("chair") == 1


def test_unrelated_entries_survive_and_underscores_normalise():
    from osg.agent.nav_agent import target_vocabulary

    vocab = target_vocabulary("tv_monitor", ["sofa", "washing machine"])
    assert vocab[0] == "tv monitor"
    assert vocab[1:] == ["sofa", "washing machine"]


def test_a_word_that_merely_shares_a_substring_is_kept():
    """"boxer" is not a part of "cracker box" -- only whole-word sub-phrases
    compete for the same detection."""
    from osg.agent.nav_agent import target_vocabulary

    assert "boxer" in target_vocabulary("cracker box", ["boxer"])


# ------------------------------------------------------ ghosting: saturation


def test_belief_cannot_saturate_beyond_a_few_misses_of_recovery():
    """The defect this closes: a sighting is worth +2.5 and a miss only -0.9, so
    a symmetric clamp saturated after three sightings and then needed SEVEN
    clean misses to unwind. Measured on the benchmark, a bowl mapped from five
    observations stored p=0.9975, and the agent committed to it on step 1 of the
    next episode and stopped before the evidence could arrive."""
    tr, f, pf = track(), frame(2.0), filt()
    for i in range(20):
        pf.update([tr], frame(2.0, frame_id=i), [detection_over(tr, frame(2.0, frame_id=i))])
    assert tr.presence.log_odds == pytest.approx(pf.l_clamp_pos)

    misses = 0
    while tr.presence.p >= 0.5 and misses < 20:
        misses += 1
        pf.update([tr], frame(5.0, frame_id=100 + misses), [])
    assert misses <= 4, f"took {misses} clean misses to doubt a saturated belief"


def test_disbelief_keeps_the_deeper_floor():
    """An object known to be gone should stay gone -- the asymmetry only limits
    how CONFIDENT presence may become, not how firmly absence is held."""
    tr, pf = track(), filt()
    for i in range(40):
        pf.update([tr], frame(5.0, frame_id=i), [])
    assert tr.presence.log_odds == pytest.approx(-pf.l_clamp)


# --------------------------------------------- ghosting: who gets the credit


def test_a_detection_credits_one_track_not_every_overlapping_one():
    """A ghost 0.8 m from a live object still projects close enough to clear a
    permissive IoU gate. Crediting both keeps the ghost alive on the live
    object's evidence -- so the map never receives the negative evidence it is
    standing directly in front of."""
    ghost = track(1, center=(-0.4, 0.0, 2.0))
    live = track(2, center=(0.4, 0.0, 2.0))
    f = frame(2.0)
    pf = filt()
    before_ghost = ghost.presence.log_odds
    pf.update([ghost, live], f, [detection_over(live, f)])
    assert live.presence.log_odds > before_ghost, "the live object was not credited"
    assert ghost.presence.log_odds < before_ghost, "the ghost was credited with a neighbour's detection"


# ------------------------------------------------ C5: a second sensor's word


def test_a_vlm_miss_outweighs_a_detector_miss():
    """The payoff of writing this as a filter: fusion is free. A VLM with
    r=0.85, q=0.02 contributes log(0.15/0.98) per miss against the detector's
    log(0.5/0.95), so one trusted look is worth about two ordinary ones."""
    det_tr, vlm_tr, pf = track(1), track(2), filt()
    before = det_tr.presence.log_odds
    pf.apply_reading(det_tr, False, recall=0.5)
    pf.apply_reading(vlm_tr, False, recall=0.85, q=0.02)
    det_step = before - det_tr.presence.log_odds
    vlm_step = before - vlm_tr.presence.log_odds
    assert vlm_step > 2 * det_step * 0.9
    assert vlm_tr.presence.n_missed == 1


def test_an_external_sighting_is_positive_and_respects_the_clamp():
    tr, pf = track(), filt()
    tr.presence.log_odds = -5.0
    for _ in range(10):
        pf.apply_reading(tr, True, recall=0.85, q=0.02)
    assert tr.presence.log_odds == pytest.approx(pf.l_clamp_pos)


def test_a_reading_counts_as_evidence_the_belief_rests_on():
    tr, pf = track(), filt()
    pf.apply_reading(tr, False, recall=0.5)
    assert tr.presence.n_expected == 1
