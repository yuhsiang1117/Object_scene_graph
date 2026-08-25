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
