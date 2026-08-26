"""The admission deadlock, and the exemption that breaks it.

Measured over 96 episodes of condition L: 33% of the times the detector named
the target, the object layer discarded the detection. In ELEVEN episodes it
discarded every single naming -- so no track formed, so no candidate formed, so
the agent never approached, so the detection never got closer, bigger or more
confident. All eleven failed. A bowl was named 20 times at 3.26 m with a score
of 0.91 and a 608 px box, and the map refused all twenty.

Those eleven split across two independent gates:

  five on SCORE, and that half is a contradiction rather than a threshold.
  `detector.class_conf` lowers the DETECTOR to 0.20 for the pitcher, tin can,
  banana and red plate -- condition H, chosen from a 900-pose false-positive
  census -- and `scene_graph.min_det_score` 0.35 then discards everything those
  four classes gained in the 0.20-0.35 band. Their boxes were 5146, 5077, 3102,
  2808 and 1258 px: far above the size gate, thrown away on score alone.

  six on SIZE, on genuinely small and genuinely confident sightings.

Both flags default off, so this file pins the shipped behaviour too.
"""
from __future__ import annotations

import numpy as np

from osg.core.types import CameraIntrinsics, Detection, FrameData
from osg.objects.object_layer import ObjectLayer

K = CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0, width=640, height=480)


def _det(label, box, score):
    x1, y1, x2, y2 = box
    mask = np.zeros((480, 640), dtype=bool)
    mask[int(y1):int(y2), int(x1):int(x2)] = True
    return Detection(label=label, score=score,
                     bbox_xyxy=np.array([float(x1), float(y1), float(x2), float(y2)]),
                     mask=mask)


def _frame(depth=2.0):
    return FrameData(frame_id=1, rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                     depth=np.full((480, 640), depth, dtype=np.float32),
                     T_wc=np.eye(4), intrinsics=K)


# A bowl at 3.26 m: 608 px, score 0.91. Named twenty times, admitted none.
SMALL_AND_SURE = _det("bowl", (300, 220, 326, 243), 0.91)
# A tin can whose class the detector was deliberately loosened to 0.20 for.
BIG_AND_UNSURE = _det("tin can", (250, 180, 330, 244), 0.28)


def _layer(**kw):
    return ObjectLayer(min_det_score=0.35, min_det_bbox_px=1200.0, **kw)


# ------------------------------------------------------------ the deadlock

def test_the_size_gate_discards_a_small_confident_target():
    layer = _layer()
    layer.set_target("bowl")
    layer.update(_frame(), [SMALL_AND_SURE])
    assert layer.funnel["det_admitted"] == 0
    assert layer.tracks() == [], "no track, so no candidate, so no approach"


def test_the_score_gate_discards_a_large_target_the_detector_was_loosened_for():
    """detector.class_conf put this class at 0.20; min_det_score 0.35 undoes it."""
    layer = _layer()
    layer.set_target("tin can")
    assert BIG_AND_UNSURE.score >= 0.20, "the detector emitted it"
    layer.update(_frame(), [BIG_AND_UNSURE])
    assert layer.funnel["det_admitted"] == 0, "and the map threw it away"


# ------------------------------------------------------------- the exemption

def test_the_target_is_admitted_on_the_detectors_terms():
    for det, target in ((SMALL_AND_SURE, "bowl"), (BIG_AND_UNSURE, "tin can")):
        layer = _layer(target_bypasses_gates=True)
        layer.set_target(target)
        layer.update(_frame(), [det])
        assert layer.funnel["det_admitted"] == 1
        assert layer.funnel["target_bypassed"] == 1
        assert len(layer.tracks()) == 1, f"{target} now reaches the map"


def test_the_exemption_is_only_for_the_target():
    """Everything else the agent is not looking for still faces both gates --
    the whole reason they exist is a scene full of incidental furniture."""
    layer = _layer(target_bypasses_gates=True)
    layer.set_target("bowl")
    layer.update(_frame(), [_det("chair", (300, 220, 326, 243), 0.91)])
    assert layer.funnel["det_admitted"] == 0
    assert layer.tracks() == []


def test_the_exemption_does_nothing_without_a_target_set():
    layer = _layer(target_bypasses_gates=True)
    layer.update(_frame(), [SMALL_AND_SURE])
    assert layer.funnel["det_admitted"] == 0


def test_admission_is_not_candidacy():
    """A bypassed detection reaches the map. Whether it may become a GOAL is
    still decided by evidence, observation count and the identity channel."""
    layer = _layer(target_bypasses_gates=True)
    layer.set_target("bowl")
    layer.update(_frame(), [SMALL_AND_SURE])
    track = layer.tracks()[0]
    assert layer.candidates("bowl", min_obs=3, min_evidence=1.0) == [], (
        "one small sighting is admitted but is not yet a candidate"
    )
    assert track.n_obs == 1


# --------------------------------------------------------- the second gate

def test_the_candidate_size_gate_moves_the_deadlock_one_stage_later():
    layer = _layer(target_bypasses_gates=True)
    layer.set_target("bowl")
    layer.update(_frame(), [SMALL_AND_SURE])
    common = dict(min_obs=1, min_score=0.3, min_bbox_px=800, min_evidence=0.2)
    assert layer.candidates("bowl", **common) == [], (
        "608 px is under the 800 px candidate gate, so it still cannot be a goal"
    )
    assert layer.candidates("bowl", target_bypasses_bbox=True, **common), (
        "and it can only grow that box by being approached"
    )


# ---------------------------------------------- candidate ranking, measured
#
# Over the 170 within-episode pairs of K, L and M where a correct and a wrong
# BELIEVED track compete, the chance the key puts the correct one first:
#
#     best_score alone                   0.635
#     best_score * presence.p (shipped)  0.729
#     presence.p alone                   0.800
#
# Multiplying by detector confidence hurts, because a confident false positive
# is exactly a distant object that really does look like the target. Per episode
# with a real choice, the correct track is chosen 64/93 shipped, 73/93 by
# presence tie-broken on evidence.


def _track(tid, label, score, p, evidence, centre=(0.0, 0.5, 0.0)):
    from osg.objects.association import ObjectTrack
    from osg.objects.ellipsoid import Ellipsoid

    t = ObjectTrack(id=tid, label=label,
                    ellipsoid=Ellipsoid(center=np.array(centre),
                                        axes=np.array([0.1, 0.1, 0.1]), R=np.eye(3)))
    t.best_score, t.evidence = score, evidence
    t.best_bbox_px = 9000.0
    t.presence.log_odds = math.log(p / (1 - p))
    for _ in range(3):
        t.observations.append(None)
    return t


def _layer_with(tracks):
    layer = ObjectLayer()
    for t in tracks:
        layer._tracks[t.id] = t
    return layer


import math  # noqa: E402


GATES = dict(min_obs=1, min_score=0.0, min_bbox_px=0, min_evidence=0.0)


def test_a_confident_ghost_outranks_a_believed_track_today():
    """The shipped key. A false positive the detector is sure about beats the
    track the agent has actually been confirming."""
    ghost   = _track(1, "bowl", score=0.95, p=0.60, evidence=2.0)
    real    = _track(2, "bowl", score=0.40, p=0.93, evidence=5.0)
    layer = _layer_with([ghost, real])
    assert layer.candidates("bowl", **GATES)[0] is ghost


def test_ranking_by_belief_picks_the_track_the_agent_has_been_confirming():
    ghost   = _track(1, "bowl", score=0.95, p=0.60, evidence=2.0)
    real    = _track(2, "bowl", score=0.40, p=0.93, evidence=5.0)
    layer = _layer_with([ghost, real])
    assert layer.candidates("bowl", rank_by_presence=True, **GATES)[0] is real


def test_evidence_breaks_the_tie_when_belief_saturates():
    """Presence clamps, so believed tracks bunch at the top and the tie-break
    decides. Evidence beats score there too: 73/93 against 69/93."""
    thin  = _track(1, "bowl", score=0.95, p=0.93, evidence=0.5)
    solid = _track(2, "bowl", score=0.40, p=0.93, evidence=6.0)
    layer = _layer_with([thin, solid])
    assert layer.candidates("bowl", rank_by_presence=True, **GATES)[0] is solid


def test_the_shipped_ranking_is_unchanged_by_default():
    ghost = _track(1, "bowl", score=0.95, p=0.60, evidence=2.0)
    real  = _track(2, "bowl", score=0.40, p=0.93, evidence=5.0)
    layer = _layer_with([ghost, real])
    ranked = layer.candidates("bowl", **GATES)
    assert [t.id for t in ranked] == [1, 2]
