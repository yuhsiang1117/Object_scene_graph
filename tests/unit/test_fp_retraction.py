"""Geometric false-positive retraction.

The dominant ObjectNav failure is walking a long way to a confidently-detected
object that is not there. A detection made at the very edge of the frame, or at
the far end of the depth range, is the cheapest kind to make and the most
likely to be wrong -- so it is treated as a hypothesis, and refuted if a later
clean, close look finds nothing of that category in the same place.
"""
from __future__ import annotations

import numpy as np

from osg.core.types import Detection
from osg.objects.object_layer import ObjectLayer

from .conftest import draw_ellipse_mask, make_camera, make_frame

W, H = 640, 480


def _det(label="chair", cx=320, cy=240, semi=(60, 40), score=0.8):
    mask = draw_ellipse_mask(H, W, (cx, cy), semi)
    return Detection(
        label=label, score=score,
        bbox_xyxy=np.array([cx - semi[0], cy - semi[1], cx + semi[0], cy + semi[1]],
                           dtype=float),
        mask=mask,
    )


def _layer():
    return ObjectLayer(min_det_score=0.0, min_det_bbox_px=0.0, max_range_m=5.0)


def _frame(intrinsics, depth=3.0, eye=(0.0, 0.88, 0.0), look=(0.0, 0.88, 3.0)):
    return make_frame(intrinsics, make_camera(list(eye), list(look)), depth_value=depth)


# ------------------------------------------------------------------- marginal


def test_edge_detection_is_marginal(intrinsics):
    layer = _layer()
    # Hard against the left edge, entirely within the left third.
    layer.update(_frame(intrinsics), [_det(cx=40, semi=(40, 40))])
    assert layer.tracks()[0].out_of_range


def test_centered_detection_is_not_marginal(intrinsics):
    layer = _layer()
    layer.update(_frame(intrinsics), [_det()])
    assert not layer.tracks()[0].out_of_range


def test_far_detection_is_marginal(intrinsics):
    """At 4.9 m of a 5 m range, depth is least reliable and the mask is tiny."""
    layer = _layer()
    layer.update(_frame(intrinsics, depth=4.9), [_det()])
    assert layer.tracks()[0].out_of_range


def test_a_clean_look_clears_the_doubt(intrinsics):
    layer = _layer()
    layer.update(_frame(intrinsics, depth=4.9), [_det()])
    track = layer.tracks()[0]
    assert track.out_of_range
    layer.update(_frame(intrinsics, depth=3.0), [_det()])
    assert not track.out_of_range


# ----------------------------------------------------------------- retraction


def test_retracted_when_in_view_and_absent(intrinsics):
    """The refutation case: the marginal track's position is dead ahead and
    close, and the detector reports nothing of that category."""
    layer = _layer()
    layer.update(_frame(intrinsics, depth=4.9), [_det()])
    track = layer.tracks()[0]

    n = layer.retract_unconfirmed(_frame(intrinsics), dets=[], half_range_m=5.0,
                                  fov_rad=np.radians(79.0))
    assert n == 1
    assert track.blacklisted and track.disabled
    assert layer.tracks() == []


def test_not_retracted_when_the_category_is_visible(intrinsics):
    """Seeing *a* chair means the hypothesis is not refuted -- the association
    may simply have gone to a different track."""
    layer = _layer()
    layer.update(_frame(intrinsics, depth=4.9), [_det()])
    n = layer.retract_unconfirmed(_frame(intrinsics), dets=[_det()],
                                  half_range_m=5.0, fov_rad=np.radians(79.0))
    assert n == 0


def test_not_retracted_when_behind_the_agent(intrinsics):
    layer = _layer()
    layer.update(_frame(intrinsics, depth=4.9), [_det()])
    behind = _frame(intrinsics, eye=(0.0, 0.88, 0.0), look=(0.0, 0.88, -3.0))
    n = layer.retract_unconfirmed(behind, dets=[], half_range_m=5.0,
                                  fov_rad=np.radians(79.0))
    assert n == 0


def test_not_retracted_when_too_far_to_judge(intrinsics):
    """Absence of evidence at long range is not evidence of absence."""
    layer = _layer()
    layer.update(_frame(intrinsics, depth=4.9), [_det()])
    n = layer.retract_unconfirmed(_frame(intrinsics), dets=[], half_range_m=1.0,
                                  fov_rad=np.radians(79.0))
    assert n == 0


def test_clean_tracks_are_never_retracted(intrinsics):
    """Only marginal tracks are hypotheses; a well-observed one is not dropped
    just because it is momentarily occluded."""
    layer = _layer()
    layer.update(_frame(intrinsics), [_det()])
    n = layer.retract_unconfirmed(_frame(intrinsics), dets=[], half_range_m=5.0,
                                  fov_rad=np.radians(79.0))
    assert n == 0
    assert len(layer.tracks()) == 1


def test_redetecting_a_retracted_object_does_not_resurrect_it(intrinsics):
    """Without this the agent re-commits to the same false positive every time
    it looks that way again."""
    layer = _layer()
    layer.update(_frame(intrinsics, depth=4.9), [_det()])
    layer.retract_unconfirmed(_frame(intrinsics), dets=[], half_range_m=5.0,
                              fov_rad=np.radians(79.0))
    assert layer.tracks() == []

    layer.update(_frame(intrinsics, depth=4.9), [_det()])
    assert layer.tracks() == [], "the retracted false positive came back"
    assert layer.candidates("chair", min_obs=1) == []


# ------------------------------------------------- one detector pass per step


def test_detector_runs_once_per_step(intrinsics):
    """The detector was being run twice on keyframe steps -- once for the object
    layer, once for the approach check. Per-step target detection would have
    made that three times. The cache must collapse them to one."""
    from osg.exploration.async_scorer import AsyncScorer
    from osg.perception.detector import StubDetector

    from .test_nav_agent import _StubScorer, make_cfg

    cfg = make_cfg()
    cfg.scene_graph.target_every_step = True
    detector = StubDetector()
    calls = {"n": 0}
    real_detect = detector.detect

    def counting_detect(rgb):
        calls["n"] += 1
        return real_detect(rgb)

    detector.detect = counting_detect
    from osg.agent.nav_agent import NavAgent

    agent = NavAgent(cfg, detector, AsyncScorer(_StubScorer()), None, "chair")
    agent.act(_frame(intrinsics))
    assert calls["n"] <= 1, f"detector ran {calls['n']} times in one step"


def test_every_action_is_already_a_keyframe():
    """Why scene_graph.target_every_step measured as a no-op (net -1 on dev50).

    The keyframe thresholds equal the action granularity exactly, so every
    forward step and every turn already crosses one -- detection runs on every
    step regardless of the flag. That is a coincidence of the defaults, not a
    design invariant: raise the keyframe thresholds and the flag silently starts
    mattering, which would make the measured no-op stop being true. Pinning the
    relationship so that change cannot happen quietly.
    """
    from osg.core.config import AgentConfig, SceneGraphConfig

    agent, sg = AgentConfig(), SceneGraphConfig()
    assert sg.keyframe_trans_m == agent.forward_m
    assert sg.keyframe_rot_deg == agent.turn_deg
