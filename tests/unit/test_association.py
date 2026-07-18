from __future__ import annotations

import numpy as np
import pytest

from osg.core.types import Detection
from osg.objects.association import DataAssociator, ObjectTrack
from osg.objects.ellipsoid import Ellipsoid
from osg.objects.object_layer import ObjectLayer

from .conftest import draw_ellipse_mask, make_frame


def _det(label, center_px, semi_px, h=480, w=640, score=0.8):
    mask = draw_ellipse_mask(h, w, center_px, semi_px)
    x1, y1 = center_px[0] - semi_px[0], center_px[1] - semi_px[1]
    x2, y2 = center_px[0] + semi_px[0], center_px[1] + semi_px[1]
    return Detection(label=label, score=score, bbox_xyxy=np.array([x1, y1, x2, y2]), mask=mask)


def test_track_visible_immediately_on_creation(intrinsics):
    """P1i follow-up: unlike the earlier hard confirmed/tentative gate (which
    hid a track from tracks() until re-observed from far enough away, and
    was found to starve scene_graph.rebuild() of objects early in
    exploration), a track must be visible the moment it's created."""
    frame = make_frame(intrinsics, np.eye(4), depth_value=3.0)
    layer = ObjectLayer(confirm_baseline_m=0.15)
    layer.update(frame, [_det("chair", (320, 240), (60, 40))])
    tracks = layer.tracks()
    assert len(tracks) == 1
    assert tracks[0].n_obs == 1
    assert tracks[0].evidence == pytest.approx(0.8)  # first sighting: full det.score weight


def test_same_pose_redetection_discounts_evidence(intrinsics):
    """Two detections from (near-)identical camera poses still merge into
    one track (association still works, and it's still visible), but the
    second observation's evidence contribution is discounted -- no real
    parallax means it's still consistent with a one-off misdetection that
    just happened to repeat within the same dwell."""
    frame = make_frame(intrinsics, np.eye(4), depth_value=3.0)
    layer = ObjectLayer(confirm_baseline_m=0.15, repeat_view_discount=0.2)
    layer.update(frame, [_det("chair", (320, 240), (60, 40))])
    frame2 = make_frame(intrinsics, np.eye(4), depth_value=3.0, frame_id=1)
    layer.update(frame2, [_det("chair", (325, 238), (58, 42))])

    tracks = layer.tracks()
    assert len(tracks) == 1
    assert tracks[0].n_obs == 2
    # 0.8 (full, first sighting) + 0.8 * 0.2 (discounted repeat) = 0.96
    assert tracks[0].evidence == pytest.approx(0.8 + 0.8 * 0.2)


def test_different_pose_redetection_gets_full_evidence(intrinsics):
    """Re-observed from a pose far enough from the first sighting (here a
    pure dolly move along the optical axis, so the object stays centered in
    frame and association still matches) -> full evidence weight, no
    discount."""
    frame = make_frame(intrinsics, np.eye(4), depth_value=3.0)
    layer = ObjectLayer(confirm_baseline_m=0.15, repeat_view_discount=0.2)
    layer.update(frame, [_det("chair", (320, 240), (60, 40))])

    T2 = np.eye(4)
    T2[2, 3] = 0.2  # 0.2 m along world z, also the camera's forward axis
    frame2 = make_frame(intrinsics, T2, depth_value=3.0, frame_id=1)
    layer.update(frame2, [_det("chair", (320, 240), (60, 40))])

    tracks = layer.tracks()
    assert len(tracks) == 1
    assert tracks[0].n_obs == 2
    assert tracks[0].evidence == pytest.approx(0.8 + 0.8)  # both full weight


def test_different_label_new_track(intrinsics):
    frame = make_frame(intrinsics, np.eye(4), depth_value=3.0)
    layer = ObjectLayer()
    layer.update(frame, [_det("chair", (320, 240), (60, 40))])
    frame2 = make_frame(intrinsics, np.eye(4), depth_value=3.0, frame_id=1)
    layer.update(frame2, [_det("table", (320, 240), (60, 40))])
    labels = sorted(t.label for t in layer.tracks())
    assert labels == ["chair", "table"]


def test_depth_gate_rejects(intrinsics):
    """Same image position but very different depth -> separate objects."""
    associator = DataAssociator(depth_gate_m=0.5)
    e = Ellipsoid(center=np.array([0.0, 0.0, 3.0]), axes=np.array([0.4, 0.3, 0.35]), R=np.eye(3))
    track = ObjectTrack(id=0, label="chair", ellipsoid=e)
    frame = make_frame(intrinsics, np.eye(4), depth_value=6.0)  # twice as far
    det = _det("chair", (320, 240), (30, 20))
    matches = associator.associate([det], frame, [track])
    assert matches == [(0, None)]


def test_candidates_filtering(intrinsics):
    frame = make_frame(intrinsics, np.eye(4), depth_value=3.0)
    layer = ObjectLayer(confirm_baseline_m=0.15)
    layer.update(frame, [_det("bed", (320, 240), (80, 50))])
    assert layer.candidates("bed", min_obs=1)
    assert not layer.candidates("bed", min_obs=2)
    assert not layer.candidates("sofa", min_obs=1)
    layer.blacklist(layer.tracks()[0].id)
    assert not layer.candidates("bed", min_obs=1)


def test_candidates_evidence_gate(intrinsics):
    """min_evidence rejects a track that's only ever been glimpsed from one
    spot (discounted repeats), and accepts once genuine multi-view
    corroboration pushes evidence over the threshold."""
    frame = make_frame(intrinsics, np.eye(4), depth_value=3.0)
    layer = ObjectLayer(confirm_baseline_m=0.15, repeat_view_discount=0.2)
    layer.update(frame, [_det("bed", (320, 240), (80, 50))])
    frame2 = make_frame(intrinsics, np.eye(4), depth_value=3.0, frame_id=1)
    layer.update(frame2, [_det("bed", (325, 238), (78, 52))])  # same pose: discounted
    assert not layer.candidates("bed", min_evidence=1.5)

    T3 = np.eye(4)
    T3[2, 3] = 0.2
    frame3 = make_frame(intrinsics, T3, depth_value=3.0, frame_id=2)
    layer.update(frame3, [_det("bed", (320, 240), (80, 50))])  # new pose: full weight
    assert layer.candidates("bed", min_evidence=1.5)
