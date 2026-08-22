"""Ground-truth visibility instrumentation (eval/runner.py).

The dynamic benchmark's dominant failure is that most episodes never perceive
the object at its new pose, and that has two causes with two different fixes --
the agent never pointed a camera there, or it did and the detection fell under
the gate. Nothing else in episodes.jsonl separates them. These tests pin the
geometry, because an instrument that is quietly wrong is worse than none.
"""
from __future__ import annotations

import types

import numpy as np

from osg.core.types import CameraIntrinsics, FrameData
from osg.eval.runner import _GroundTruthVisibility

K = CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0, width=640, height=480)


def _frame(depth_value: float, T_wc=None) -> FrameData:
    """A camera at the origin looking down +z (OpenCV convention)."""
    return FrameData(
        frame_id=0,
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        depth=np.full((480, 640), depth_value, dtype=np.float32),
        T_wc=np.eye(4) if T_wc is None else T_wc,
        intrinsics=K,
    )


def test_an_object_straight_ahead_and_unoccluded_counts_as_seen():
    v = _GroundTruthVisibility([0.0, 0.0, 2.0])
    v.observe(_frame(depth_value=2.0))
    assert v.in_view == 1
    assert v.close_frames == 1
    assert v.min_range_m == 2.0


def test_an_object_behind_the_camera_does_not():
    v = _GroundTruthVisibility([0.0, 0.0, -2.0])
    v.observe(_frame(depth_value=5.0))
    assert v.in_view == 0


def test_an_object_outside_the_image_does_not():
    # 3 m to the side at 2 m range is far outside a 640 px image at fx=320.
    v = _GroundTruthVisibility([3.0, 0.0, 2.0])
    v.observe(_frame(depth_value=5.0))
    assert v.in_view == 0


def test_a_wall_in_front_of_the_object_hides_it():
    """The point projects into the image, but the depth buffer says the ray
    stopped a metre short. Without this the instrument would count every object
    on the far side of a wall as looked at."""
    v = _GroundTruthVisibility([0.0, 0.0, 3.0])
    v.observe(_frame(depth_value=1.0))
    assert v.in_view == 0


def test_depth_landing_on_the_objects_own_front_face_still_counts():
    """Depth returns the front face and the projected point is the centre, so a
    small negative difference is the object itself, not an occluder."""
    v = _GroundTruthVisibility([0.0, 0.0, 2.0])
    v.observe(_frame(depth_value=1.9))
    assert v.in_view == 1


def test_invalid_depth_is_not_read_as_an_occluder():
    """0 means no return, not a surface at the camera."""
    v = _GroundTruthVisibility([0.0, 0.0, 2.0])
    v.observe(_frame(depth_value=0.0))
    assert v.in_view == 1


def test_range_is_tracked_and_far_sightings_are_flagged_separately():
    v = _GroundTruthVisibility([0.0, 0.0, 6.0])
    v.observe(_frame(depth_value=6.0))
    assert v.in_view == 1
    assert v.close_frames == 0, "6 m is in view but not at a range detection is likely"
    assert v.min_range_m == 6.0


def test_no_authored_target_is_a_no_op_rather_than_a_crash():
    v = _GroundTruthVisibility(None)
    v.observe(_frame(depth_value=2.0))
    assert v.fields()["gt_in_view_frames"] == 0
    assert v.fields()["gt_min_range_m"] is None


# ------------------------------------------------------------- keyframe half

class _Det:
    def __init__(self, label, score, bbox):
        self.label = label
        self.score = score
        self.bbox_xyxy = np.asarray(bbox, dtype=float)


def test_a_detection_covering_the_object_counts_as_seen_by_name():
    v = _GroundTruthVisibility([0.0, 0.0, 2.0])  # projects to the image centre
    v.observe_keyframe(_frame(2.0), [_Det("bowl", 0.7, [300, 220, 340, 260])], "bowl")
    assert v.kf_in_view == 1
    assert v.kf_detected == 1
    assert v.best_det_score == 0.7


def test_a_detection_of_a_different_class_does_not():
    v = _GroundTruthVisibility([0.0, 0.0, 2.0])
    v.observe_keyframe(_frame(2.0), [_Det("chair", 0.9, [300, 220, 340, 260])], "bowl")
    assert v.kf_in_view == 1
    assert v.kf_detected == 0


def test_a_detection_of_the_right_class_somewhere_else_does_not():
    """Otherwise a false positive across the room would be scored as having
    found the object, which is the exact confusion this instrument exists to
    avoid."""
    v = _GroundTruthVisibility([0.0, 0.0, 2.0])
    v.observe_keyframe(_frame(2.0), [_Det("bowl", 0.9, [10, 10, 60, 60])], "bowl")
    assert v.kf_detected == 0


def test_labels_are_normalised_on_both_sides():
    v = _GroundTruthVisibility([0.0, 0.0, 2.0])
    v.observe_keyframe(_frame(2.0), [_Det("Tomato_Soup_Can", 0.5, [300, 220, 340, 260])],
                       "tomato soup can")
    assert v.kf_detected == 1


def test_framing_is_bucketed_so_a_peripheral_sighting_is_distinguishable():
    """H could not tell an object clipped to the edge of a wide-FOV frame from
    one centred at the same range, and that is where its explanation ran out."""
    centred = _GroundTruthVisibility([0.0, 0.0, 2.0])
    centred.observe_keyframe(_frame(2.0), [], "bowl")
    assert centred.by_framing["close_centred"][0] == 1
    assert centred.by_framing["close_peripheral"][0] == 0

    # 1.7 m to the side at 2 m: inside the image, far off the optical axis.
    edge = _GroundTruthVisibility([1.7, 0.0, 2.0])
    edge.observe_keyframe(_frame(2.0), [], "bowl")
    assert edge.by_framing["close_peripheral"][0] == 1
    assert edge.best_offaxis > 0.6


def test_the_keyframe_half_shares_the_occlusion_test():
    v = _GroundTruthVisibility([0.0, 0.0, 3.0])
    v.observe_keyframe(_frame(1.0), [_Det("bowl", 0.9, [0, 0, 640, 480])], "bowl")
    assert v.kf_in_view == 0, "a wall in front must hide it here too"
    assert v.kf_detected == 0


def test_a_sliver_of_the_object_is_not_looked_at_it():
    """Occlusion used to be tested at the centre pixel alone, so an object nine
    tenths hidden behind a chair back -- with only its middle showing -- counted
    as a frame the agent looked at it. That understates in-situ recall by
    exactly the frames where the detector had no chance, and it is the end of
    the comparison against a probe that requires a real pixel count.

    Here the depth buffer puts a surface at 1.9 m everywhere except one pixel,
    so only the object's centre sample survives: 1 of 7, below the half the
    instrument now requires.
    """
    v = _GroundTruthVisibility([0.0, 0.0, 2.0])
    frame = _frame(depth_value=1.5)          # a wall at 1.5 m, object at 2.0 m
    frame.depth[240, 320] = 2.0              # ...with a peephole at the centre
    v.observe_keyframe(frame, [], "bowl")
    assert v.kf_in_view == 0


def test_a_mostly_visible_object_still_counts():
    v = _GroundTruthVisibility([0.0, 0.0, 2.0])
    frame = _frame(depth_value=2.0)
    frame.depth[200:220, 300:320] = 1.0      # a small occluder off to one side
    v.observe_keyframe(frame, [], "bowl")
    assert v.kf_in_view == 1
    assert v.fields()["gt_mean_visible_fraction"] == 1.0
