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
