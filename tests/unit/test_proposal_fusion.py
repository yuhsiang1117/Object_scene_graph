"""Proposals are fused as appearance hypotheses, not as namings.

Measured on the full 107 (docs/history/REGION_FALSE_ADMISSION.md): a region the stage
admitted became an ordinary same-label track, won the candidate gate and spent
an attempt -- 18 trials lost, because the per-frame threshold admits on 13% of
frames with no object in view. The fix is not when the stage runs but how its
output is judged. This file pins that judgement.
"""
from __future__ import annotations

import numpy as np

from osg.core.types import CameraIntrinsics, Detection, FrameData
from osg.objects.object_layer import ObjectLayer

K = CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0, width=640, height=480)


def _det(label, box, score, source="detector", ft=None):
    x1, y1, x2, y2 = box
    mask = np.zeros((480, 640), dtype=bool)
    mask[int(y1):int(y2), int(x1):int(x2)] = True
    return Detection(label=label, score=score,
                     bbox_xyxy=np.array([float(x1), float(y1), float(x2), float(y2)]),
                     mask=mask, source=source, clip_ft=ft)


def _frame(frame_id, depth=2.0, x=0.0):
    T = np.eye(4); T[0, 3] = x
    return FrameData(frame_id=frame_id, rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                     depth=np.full((480, 640), depth, dtype=np.float32),
                     T_wc=T, intrinsics=K)


def _unit(*v):
    a = np.asarray(v, dtype=np.float32); return a / np.linalg.norm(a)


BOX = (280, 200, 360, 280)


def _layer():
    return ObjectLayer(min_det_score=0.0, min_det_bbox_px=0.0, confirm_baseline_m=0.0)


# ------------------------------------------------------------ the object layer

def test_a_proposal_counts_apart_from_a_naming_and_keeps_a_mean_feature():
    layer = _layer()
    layer.set_proposal_text(_unit(1, 0, 0))
    layer.update(_frame(1), [_det("bowl", BOX, 0.5, "proposal", _unit(1, 0, 0))])
    layer.update(_frame(2), [_det("bowl", BOX, 0.5, "proposal", _unit(0, 1, 0))])
    (t,) = layer.tracks(include_proposals=True)
    assert t.n_obs == 2 and t.n_proposal_obs == 2 and t.proposal_only
    # mean of (1,0,0) and (0,1,0), renormalised: cosine to (1,0,0) is 1/sqrt 2
    assert abs(t.proposal_sim - 0.7071) < 1e-3
    assert layer.funnel["proposal_obs"] == 2


def test_a_naming_at_the_same_spot_forms_its_own_track():
    layer = _layer()
    layer.set_proposal_text(_unit(1, 0, 0))
    layer.update(_frame(1), [_det("bowl", BOX, 0.5, "proposal", _unit(1, 0, 0))])
    layer.update(_frame(2), [_det("bowl", BOX, 0.9)])
    ts = layer.tracks(include_proposals=True)
    assert sorted(t.proposal_only for t in ts) == [False, True]
    assert all(t.n_obs == 1 for t in ts)


def test_without_a_query_text_the_cosine_stays_unset():
    layer = _layer()
    layer.update(_frame(1), [_det("bowl", BOX, 0.5, "proposal", _unit(1, 0, 0))])
    (t,) = layer.tracks(include_proposals=True)
    assert t.n_proposal_obs == 1 and t.proposal_sim == -1.0


def test_a_detector_only_map_is_untouched():
    layer = _layer()
    layer.update(_frame(1), [_det("bowl", BOX, 0.9)])
    layer.update(_frame(2), [_det("bowl", BOX, 0.9)])
    (t,) = layer.tracks()
    assert t.n_proposal_obs == 0 and not t.proposal_only and t.clip_ft is None
    assert layer.funnel["proposal_obs"] == 0


# ------------------------------------------------------------ the candidate gate

def _proposal_track(layer, n, sim_ft, text=_unit(1, 0, 0)):
    layer.set_proposal_text(text)
    for i in range(n):
        layer.update(_frame(i + 1), [_det("bowl", BOX, 0.5, "proposal", sim_ft)])


def test_a_proposal_only_track_needs_both_observations_and_the_cosine():
    layer = _layer()
    _proposal_track(layer, 3, _unit(1, 0.5, 0))          # cosine 0.894
    assert len(layer.candidates("bowl", min_obs=1)) == 1  # defaults: no bar
    assert layer.candidates("bowl", min_obs=1, proposal_min_obs=4) == []
    assert layer.candidates("bowl", min_obs=1, proposal_tau=0.95) == []
    assert len(layer.candidates("bowl", min_obs=1, proposal_min_obs=3, proposal_tau=0.85)) == 1


def test_commits_off_keeps_every_proposal_only_track_out():
    layer = _layer()
    _proposal_track(layer, 5, _unit(1, 0, 0))            # cosine 1.0
    assert layer.candidates("bowl", min_obs=1, proposal_commits=False) == []
    # ...and does not touch a named one
    layer.update(_frame(9), [_det("bowl", (10, 10, 90, 90), 0.9)])
    layer.update(_frame(10), [_det("bowl", (10, 10, 90, 90), 0.9)])
    out = layer.candidates("bowl", min_obs=1, proposal_commits=False)
    assert [t.proposal_only for t in out] == [False]


def test_a_named_track_always_ranks_ahead_of_a_proposal_only_one():
    layer = _layer()
    _proposal_track(layer, 6, _unit(1, 0, 0))            # strong: 6 obs, cosine 1.0
    layer.update(_frame(20), [_det("bowl", (10, 10, 90, 90), 0.4)])
    layer.update(_frame(21), [_det("bowl", (10, 10, 90, 90), 0.4)])
    for rank_by_presence in (False, True):
        out = layer.candidates("bowl", min_obs=1, rank_by_presence=rank_by_presence)
        assert [t.proposal_only for t in out] == [False, True]


def test_a_label_the_detector_never_used_is_still_not_a_candidate():
    layer = _layer()
    _proposal_track(layer, 3, _unit(1, 0, 0))
    assert layer.candidates("mug", min_obs=1) == []


# ------------------------------------------------------------ the stage's output

def test_the_proposal_detection_carries_its_feature_and_provenance():
    from osg.perception.region_proposer import RegionProposer

    class Cfg:
        imgsz = 64; conf = 0.1; iou = 0.5; tau = 0.0; admit_score = 0.5
        pad_frac = 0.25; min_area_px = 10.0; max_area_frac = 0.5; max_regions = 8
        device = "cpu"

    class T:
        def __init__(self, a): self.a = a
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.a
        def __getitem__(self, i): return T(self.a[i])   # masks are indexed on the device

    class Boxes:
        def __init__(self, a): self.xyxy = T(a)
        def __len__(self): return len(self.xyxy.a)

    class Masks:
        def __init__(self, a): self.data = T(a)

    class Result:
        def __init__(self, b, m): self.boxes = Boxes(b); self.masks = Masks(m)

    class Model:
        def predict(self, rgb, **kw):
            b = np.array([[10, 10, 30, 30], [40, 40, 60, 60]], dtype=float)
            m = np.zeros((2, 100, 100), dtype=bool); m[0, 10:30, 10:30] = True; m[1, 40:60, 40:60] = True
            return [Result(b, m)]

    class Enc:
        def text_feature(self, s): return _unit(1, 0, 0)
        def encode_images(self, crops): return np.stack([_unit(0, 1, 0), _unit(1, 0, 0)])

    rp = RegionProposer(Model(), Enc(), Cfg())
    rp.set_target("bowl")
    det = rp.propose(np.zeros((100, 100, 3), dtype=np.uint8))
    assert det is not None and det.source == "proposal"
    assert np.allclose(det.clip_ft, _unit(1, 0, 0))
    assert rp.last_score == 1.0
    assert det.bbox_xyxy[0] == 40.0     # the second region won


def test_every_keyframe_bypasses_the_episode_gates_but_not_the_cap():
    from osg.agent.nav_agent import NavAgent

    class Cfg:
        every_keyframe = True; require_never_named = True; unnamed_keyframes = 20
        max_per_episode = 2

    class RP:
        cfg = Cfg()

    a = NavAgent.__new__(NavAgent)
    a.region_proposer = RP(); a._region_admits = 0; a._region_kf = 0; a._region_named = True
    assert a._region_active()
    a._region_admits = 2
    assert not a._region_active()


# ------------------------------------------------------------ what a proposal may not touch

def test_a_proposal_beside_a_named_track_leaves_it_untouched():
    layer = _layer()
    layer.set_proposal_text(_unit(1, 0, 0))
    layer.update(_frame(1), [_det("tin can", BOX, 0.28)])                       # loosened class
    (t,) = layer.tracks()
    ev, best, cam, n = t.evidence, t.best_score, t.best_cam_xy.copy(), t.n_obs
    layer.update(_frame(2), [_det("tin can", BOX, 0.5, "proposal", _unit(1, 0, 0))])
    assert t.n_obs == n and t.n_proposal_obs == 0
    assert t.evidence == ev and t.best_score == best and np.allclose(t.best_cam_xy, cam)


def test_a_proposal_only_track_has_no_evidence_and_the_region_as_its_best_view():
    layer = _layer()
    layer.update(_frame(1), [_det("bowl", BOX, 0.5, "proposal", _unit(1, 0, 0))])
    (t,) = layer.tracks(include_proposals=True)
    assert t.evidence == 0.0 and t.best_score == 0.5
    assert t.best_cam_xy is not None and t.best_bbox_px > 0


def test_the_proposal_sensor_answers_only_for_a_proposal_only_track():
    from osg.agent.nav_agent import NavAgent

    class Track:
        def __init__(self, po): self.proposal_only = po

    class Layer:
        def __init__(self, tr): self.tr = tr
        def get(self, i): return self.tr

    a = NavAgent.__new__(NavAgent)
    a._candidate_id = None; a.object_layer = Layer(Track(True))
    assert not a._working_a_proposal_track()
    a._candidate_id = 3
    assert a._working_a_proposal_track()
    a.object_layer = Layer(Track(False))
    assert not a._working_a_proposal_track()


# ------------------------------------------------------------ two populations

def test_a_proposal_never_joins_a_named_track_and_a_naming_never_joins_a_proposal_track():
    layer = _layer()
    layer.set_proposal_text(_unit(1, 0, 0))
    layer.update(_frame(1), [_det("bowl", BOX, 0.9)])
    layer.update(_frame(2), [_det("bowl", BOX, 0.5, "proposal", _unit(1, 0, 0))])
    layer.update(_frame(3), [_det("bowl", BOX, 0.9)])
    layer.update(_frame(4), [_det("bowl", BOX, 0.5, "proposal", _unit(1, 0, 0))])
    named = [t for t in layer.tracks() if not t.proposal_only]
    props = [t for t in layer.tracks(include_proposals=True) if t.proposal_only]
    assert len(named) == 1 and named[0].n_obs == 2 and named[0].n_proposal_obs == 0
    assert len(props) == 1 and props[0].n_obs == 2 and props[0].n_proposal_obs == 2


def test_the_map_hides_proposal_only_tracks_unless_asked():
    layer = _layer()
    layer.update(_frame(1), [_det("bowl", BOX, 0.5, "proposal", _unit(1, 0, 0))])
    layer.update(_frame(2), [_det("mug", (10, 10, 90, 90), 0.9)])
    assert [t.label for t in layer.tracks()] == ["mug"]
    assert sorted(t.label for t in layer.tracks(include_proposals=True)) == ["bowl", "mug"]
    # ...but the candidate gate still judges them
    assert [t.label for t in layer.candidates("bowl", min_obs=1)] == ["bowl"]


def test_a_named_track_is_bit_identical_with_and_without_proposals_around_it():
    def build(with_props):
        layer = _layer()
        layer.set_proposal_text(_unit(1, 0, 0))
        for i in range(1, 7):
            dets = [_det("bowl", BOX, 0.7)]
            if with_props:
                dets.append(_det("bowl", (300, 210, 370, 290), 0.5, "proposal", _unit(1, 0, 0)))
            layer.update(_frame(i), dets)
        (t,) = [t for t in layer.tracks() if not t.proposal_only]
        return (t.n_obs, t.evidence, t.best_score, tuple(layer.center_of(t).round(6)),
                t.presence.log_odds, tuple(t.linked_ids) if hasattr(t, "linked_ids") else ())
    assert build(False) == build(True)


def test_proposal_tracks_take_no_draws_and_no_ids_from_the_named_map():
    from osg.objects.object_layer import PROPOSAL_ID_BASE
    rng = np.random.default_rng(7)

    def build(with_props):
        layer = _layer()
        layer.set_proposal_text(_unit(1, 0, 0))
        out = []
        for i in range(1, 9):
            depth = np.full((480, 640), 2.0, dtype=np.float32)
            depth += rng.random((480, 640)).astype(np.float32) * 0.5     # sampled depth matters now
            f = FrameData(frame_id=i, rgb=np.zeros((480, 640, 3), dtype=np.uint8), depth=depth,
                          T_wc=np.eye(4), intrinsics=K)
            dets = [_det("bowl", (40 * i, 100, 40 * i + 60, 160), 0.8)]        # a new bowl each frame
            if with_props:
                dets.insert(0, _det("bowl", (300, 300, 380, 380), 0.5, "proposal", _unit(1, 0, 0)))
            layer.update(f, dets)
        named = [t for t in layer.tracks() if not t.proposal_only]
        return [(t.id, tuple(np.round(t.ellipsoid.center, 6)), tuple(np.round(t.ellipsoid.axes, 6))) for t in named]

    rng = np.random.default_rng(7); a = build(False)
    rng = np.random.default_rng(7); b = build(True)
    assert a == b
    layer = _layer()
    layer.update(_frame(1), [_det("bowl", BOX, 0.5, "proposal", _unit(1, 0, 0))])
    (t,) = layer.tracks(include_proposals=True)
    assert t.id >= PROPOSAL_ID_BASE


def test_proposal_only_tracks_rank_by_their_mean_feature():
    layer = _layer()
    layer.set_proposal_text(_unit(1, 0, 0))
    for i in range(1, 5):                                            # weaker track first
        layer.update(_frame(i), [_det("bowl", (40, 40, 120, 120), 0.5, "proposal", _unit(1, 0.6, 0))])
    for i in range(5, 9):
        layer.update(_frame(i), [_det("bowl", (400, 300, 480, 380), 0.5, "proposal", _unit(1, 0.1, 0))])
    out = layer.candidates("bowl", min_obs=1, rank_by_presence=True)
    assert [round(t.proposal_sim, 2) for t in out] == [0.99, 0.86]


def test_the_detector_gates_do_not_apply_to_a_proposal_only_track():
    """Evidence, score and size are detector quantities; a proposal track has
    none and is judged on its own bar. The first fused run committed to zero
    proposal tracks in 49 episodes because of this."""
    layer = _layer()
    _proposal_track(layer, 4, _unit(1, 0.3, 0))                    # evidence 0.0, score 0.5
    out = layer.candidates("bowl", min_obs=2, min_evidence=1.0, min_score=0.6, min_bbox_px=1e9,
                           proposal_min_obs=4, proposal_tau=0.28)
    assert [t.proposal_only for t in out] == [True]
    # ...while a named track is still held to them
    layer.update(_frame(20), [_det("bowl", (10, 10, 90, 90), 0.4)])
    layer.update(_frame(21), [_det("bowl", (10, 10, 90, 90), 0.4)])
    out = layer.candidates("bowl", min_obs=2, min_evidence=1.0, min_score=0.6, min_bbox_px=1e9,
                           proposal_min_obs=4, proposal_tau=0.28)
    assert [t.proposal_only for t in out] == [True]


def test_a_per_class_bar_overrides_the_global_one_for_that_class_only():
    layer = _layer()
    _proposal_track(layer, 4, _unit(1, 0.5, 0))                     # cosine 0.894
    by = {"bowl": 0.95}
    assert layer.candidates("bowl", min_obs=1, proposal_tau=0.5, proposal_tau_by_class=by) == []
    assert len(layer.candidates("bowl", min_obs=1, proposal_tau=0.5, proposal_tau_by_class={"mug": 0.95})) == 1
