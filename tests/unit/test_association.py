from __future__ import annotations

import numpy as np

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


def test_same_pose_redetection_stays_unconfirmed(intrinsics):
    """A track is a hypothesis until re-observed with real parallax: two
    detections from (near-)identical camera poses still merge into one
    track (association still works), but it stays invisible to tracks()/
    candidates() -- no baseline means no corroborating evidence that this
    wasn't just a one-off misdetection repeated within the same dwell."""
    frame = make_frame(intrinsics, np.eye(4), depth_value=3.0)
    layer = ObjectLayer(confirm_baseline_m=0.15)
    layer.update(frame, [_det("chair", (320, 240), (60, 40))])
    assert len(layer.tracks()) == 0
    frame2 = make_frame(intrinsics, np.eye(4), depth_value=3.0, frame_id=1)
    layer.update(frame2, [_det("chair", (325, 238), (58, 42))])
    assert len(layer.tracks()) == 0
    unconfirmed = layer.tracks(include_unconfirmed=True)
    assert len(unconfirmed) == 1
    assert unconfirmed[0].n_obs == 2
    assert not unconfirmed[0].confirmed


def test_different_pose_redetection_confirms(intrinsics):
    """Re-observed from a pose far enough from the first sighting (here a
    pure dolly move along the optical axis, so the object stays centered in
    frame and association still matches) -> promoted to confirmed."""
    frame = make_frame(intrinsics, np.eye(4), depth_value=3.0)
    layer = ObjectLayer(confirm_baseline_m=0.15)
    layer.update(frame, [_det("chair", (320, 240), (60, 40))])
    assert len(layer.tracks()) == 0

    T2 = np.eye(4)
    T2[2, 3] = 0.2  # 0.2 m along world z, also the camera's forward axis
    frame2 = make_frame(intrinsics, T2, depth_value=3.0, frame_id=1)
    layer.update(frame2, [_det("chair", (320, 240), (60, 40))])

    tracks = layer.tracks()
    assert len(tracks) == 1
    assert tracks[0].n_obs == 2
    assert tracks[0].confirmed


def test_different_label_new_track(intrinsics):
    frame = make_frame(intrinsics, np.eye(4), depth_value=3.0)
    layer = ObjectLayer()
    layer.update(frame, [_det("chair", (320, 240), (60, 40))])
    frame2 = make_frame(intrinsics, np.eye(4), depth_value=3.0, frame_id=1)
    layer.update(frame2, [_det("table", (320, 240), (60, 40))])
    labels = sorted(t.label for t in layer.tracks(include_unconfirmed=True))
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
    assert not layer.candidates("bed", min_obs=1)  # not confirmed yet: one sighting only

    T2 = np.eye(4)
    T2[2, 3] = 0.2
    frame2 = make_frame(intrinsics, T2, depth_value=3.0, frame_id=1)
    layer.update(frame2, [_det("bed", (320, 240), (80, 50))])

    assert layer.candidates("bed", min_obs=1)
    assert not layer.candidates("bed", min_obs=3)
    assert not layer.candidates("sofa", min_obs=1)
    layer.blacklist(layer.tracks()[0].id)
    assert not layer.candidates("bed", min_obs=1)
