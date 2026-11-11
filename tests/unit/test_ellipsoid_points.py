"""scene_graph.ellipsoid: depth-point initialisation, buffer refit, 3D
association and the finalise merge, on a ray-cast scene with known boxes."""
from __future__ import annotations

import numpy as np
import pytest

from osg.core.config import OSGConfig
from osg.core.types import CameraIntrinsics
from osg.objects.ellipsoid import Ellipsoid
from osg.objects.merge import mutual_overlap, point_overlap
from osg.objects.point_buffer import VoxelBuffer, clean_points
from osg.pipeline.objects import build_object_layer

from .conftest import make_camera
from .synthetic_scene import Box, detections, frame_at

INTR = CameraIntrinsics(fx=240.0, fy=240.0, cx=200.0, cy=150.0, width=400, height=300)
CABINET = Box("cabinet", np.array([0.0, 0.5, 0.0]), np.array([0.5, 0.5, 0.25]))


def _ecfg(**kw):
    e = OSGConfig().scene_graph.ellipsoid
    for k, v in kw.items():
        setattr(e, k, v)
    return e


def _view(box, cam_pos, bleed=0):
    T = make_camera(np.asarray(cam_pos, float), box.center)
    frame, inst = frame_at([box], INTR, T)
    return frame, detections([box], inst, frame, bleed_px=bleed)[0]


def _extent_box(e: Ellipsoid):
    return np.array([e.world_extent(np.eye(3)[i]) for i in range(3)])


SQUARE = Box("cabinet", np.array([0.0, 0.5, 0.0]), np.array([0.4, 0.5, 0.4]))


def test_points_init_centre_survives_mask_bleed_the_moments_init_does_not():
    frame, det = _view(SQUARE, [0.0, 1.3, 2.5], bleed=3)
    moments = Ellipsoid.init_from_detection(det, frame, rng=np.random.default_rng(0))
    points = Ellipsoid.init_from_points(clean_points(det, frame, _ecfg()), frame, _ecfg())
    err_m = np.linalg.norm(moments.center - SQUARE.center)
    err_p = np.linalg.norm(points.center - SQUARE.center)
    assert err_p < 0.12
    assert err_p < err_m


def test_depth_axis_rule_is_a_guess_one_view_cannot_settle():
    """A 1.0 x 0.5 cabinet seen face-on. Barely above its top (1.3 m), no rule
    sees its depth and every one misplaces the centre by ~0.24 m. From 2.3 m
    the top surface is in view and "thickness" reads it, while "lateral" still
    assumes it is as deep as it is wide. Which rule wins on real objects at the
    robot's real camera height is measured (the depth_axis sweep,
    docs/history/AB_RESULTS.md), not assumed here."""
    def err(cam_y, rule):
        frame, det = _view(CABINET, [0.0, cam_y, 2.0])
        e = _ecfg(depth_axis=rule)
        ell = Ellipsoid.init_from_points(clean_points(det, frame, e), frame, e)
        return np.linalg.norm(ell.center - CABINET.center)
    assert err(1.3, "lateral") > 0.2 and err(1.3, "thickness") > 0.2
    assert err(2.3, "thickness") < 0.1 < err(2.3, "lateral")


def test_points_init_is_gravity_aligned_and_sized_like_the_box():
    frame, det = _view(CABINET, [0.0, 1.6, 2.5])           # looking down at it
    e = Ellipsoid.init_from_points(clean_points(det, frame, _ecfg()), frame, _ecfg())
    assert abs(e.R[:, 1] @ np.array([0.0, 1.0, 0.0])) == pytest.approx(1.0)   # up is up
    ext = _extent_box(e)
    assert ext[0] == pytest.approx(CABINET.half[0], rel=0.25)   # width, seen
    assert ext[1] == pytest.approx(CABINET.half[1], rel=0.30)   # height, seen


def test_buffer_refit_from_two_sides_recovers_depth():
    buf = VoxelBuffer(0.03)
    dirs = []
    ecfg = _ecfg()
    for pos in ([0.0, 1.3, 2.5], [2.5, 1.3, 0.2], [-2.0, 1.3, -1.5]):
        frame, det = _view(CABINET, pos)
        pts = clean_points(det, frame, ecfg)
        buf.add(pts.world)
        d = pts.world.mean(axis=0) - frame.camera_position
        dirs.append(np.array([d[0], d[2]]) / np.linalg.norm([d[0], d[2]]))
    e = Ellipsoid.fit_points(buf.points(), np.asarray(dirs), ecfg)
    ext = np.sort(_extent_box(e)[[0, 2]])
    assert ext[0] == pytest.approx(0.25, rel=0.35)
    assert ext[1] == pytest.approx(0.5, rel=0.25)
    assert np.linalg.norm(e.center - CABINET.center) < 0.1


def test_voxel_buffer_compress_and_roundtrip():
    rng = np.random.default_rng(0)
    buf = VoxelBuffer(0.02)
    buf.add(rng.uniform(-0.5, 0.5, size=(5000, 3)))
    n0 = len(buf)
    buf.compress(256)
    assert len(buf) <= 256 < n0
    back = VoxelBuffer.from_record(buf.to_record())
    assert np.allclose(np.sort(back.points(), axis=0), np.sort(buf.points(), axis=0))


def test_moved_object_does_not_overlap_its_ghost():
    """The ghost rule: half a metre away is a different place, not the same object."""
    mug = Box("mug", np.array([0.0, 0.8, 0.0]), np.array([0.05, 0.06, 0.05]))
    moved = Box("mug", np.array([0.5, 0.8, 0.0]), np.array([0.05, 0.06, 0.05]))
    layer = build_object_layer(_cfg(buffer=True, init="points"))
    for b in (mug, moved):
        frame, det = _view(b, [b.center[0], 1.3, 1.2])
        layer.update(frame, [det], floor_key=0)
    a, b = list(layer._tracks.values())
    assert mutual_overlap(a, b, 1.25) == 0.0


def _cfg(**kw):
    cfg = OSGConfig()
    cfg.scene_graph.min_det_bbox_px = 0.0
    for k, v in kw.items():
        setattr(cfg.scene_graph.ellipsoid, k, v)
    return cfg


def _orbit_tracks(cfg, box, n=10, bleed=3):
    layer = build_object_layer(cfg)
    for k in range(n):
        a = -0.6 + 1.2 * k / (n - 1)
        pos = np.array([2.6 * np.sin(a), 1.35, 2.6 * np.cos(a)])
        T = make_camera(pos, box.center)
        frame, inst = frame_at([box], INTR, T, frame_id=k)
        layer.update(frame, detections([box], inst, frame, bleed_px=bleed), floor_key=0)
    return layer


def test_default_group_changes_nothing():
    """`_geo` off: the layer is the layer as measured."""
    a = _orbit_tracks(OSGConfig(), CABINET)
    cfg = OSGConfig()
    cfg.scene_graph.ellipsoid.depth_weight = 999.0     # a knob nothing reads while mode is 2d
    b = _orbit_tracks(cfg, CABINET)
    assert not b._geo
    for ta, tb in zip(a._tracks.values(), b._tracks.values()):
        assert np.array_equal(ta.ellipsoid.center, tb.ellipsoid.center)
        assert np.array_equal(ta.ellipsoid.axes, tb.ellipsoid.axes)


def test_points_path_builds_one_accurate_track_over_an_orbit():
    layer = _orbit_tracks(_cfg(init="points", buffer=True, refine_mode="points",
                               assoc_3d_overlap=0.3), CABINET)
    tracks = list(layer._tracks.values())
    assert len(tracks) == 1
    assert np.linalg.norm(tracks[0].ellipsoid.center - CABINET.center) < 0.1


def test_point_overlap_counts_points_inside_the_inflated_ellipsoid():
    from osg.objects.association import ObjectTrack

    t = ObjectTrack(id=0, label="x", ellipsoid=Ellipsoid(np.zeros(3), np.ones(3) * 0.5, np.eye(3)))
    inside = np.array([[0.1, 0.0, 0.0], [0.0, 0.3, 0.0]])
    outside = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, -3.0]])
    assert point_overlap(t, np.vstack([inside, outside]), 1.0) == pytest.approx(0.5)


def test_voxel_overlap_ignores_an_oversized_ellipsoid():
    """Round 1's over-merge: an ellipsoid grown too big contains its
    same-label neighbour's points; the neighbour's surface is still nowhere
    near the track's own buffered surface."""
    from osg.objects.association import ObjectTrack

    buf = VoxelBuffer(0.03)
    rng = np.random.default_rng(0)
    buf.add(rng.uniform([-0.25, 0.0, -0.25], [0.25, 0.9, 0.25], size=(2000, 3)))
    t = ObjectTrack(id=0, label="chair", buffer=buf,
                    ellipsoid=Ellipsoid(np.array([0.5, 0.45, 0.0]), np.array([1.2, 0.6, 0.6]), np.eye(3)))
    neighbour = rng.uniform([0.75, 0.0, -0.25], [1.25, 0.9, 0.25], size=(500, 3))
    assert point_overlap(t, neighbour, 1.25) > 0.9
    assert point_overlap(t, neighbour, 1.25, mode="voxel") == 0.0
    own = rng.uniform([-0.2, 0.1, -0.2], [0.2, 0.8, 0.2], size=(300, 3))
    assert point_overlap(t, own, 1.25, mode="voxel") > 0.95


def test_assoc_3d_unique_leaves_an_ambiguous_detection_unmatched():
    from osg.objects.association import ObjectTrack
    from osg.objects.merge import associate_3d

    rng = np.random.default_rng(1)
    tracks = []
    for k, x0 in enumerate((0.0, 0.5)):
        buf = VoxelBuffer(0.03)
        buf.add(rng.uniform([x0, 0.0, 0.0], [x0 + 0.5, 0.5, 0.5], size=(2000, 3)))
        tracks.append(ObjectTrack(id=k, label="cabinet", buffer=buf,
                                  ellipsoid=Ellipsoid(np.array([x0 + 0.25, 0.25, 0.25]), np.ones(3) * 0.25, np.eye(3))))
    straddle = rng.uniform([0.3, 0.0, 0.0], [0.7, 0.5, 0.5], size=(400, 3))
    args = ([(0, "cabinet", straddle)], tracks, set(), 0.3, 1.25, False, "voxel")
    assert associate_3d(*args) != {}
    assert associate_3d(*args, unique=True) == {}


def test_refit_override_replaces_an_ellipsoid_that_misses_its_own_points():
    """A track whose ellipsoid sits 1 m from its buffered surface: the move
    limit refuses the refit, the override accepts it."""
    from osg.objects.association import ObjectTrack

    def refit(override):
        layer = build_object_layer(_cfg(init="points", buffer=True, refine_mode="points",
                                        refit_override_overlap=override))
        buf = VoxelBuffer(0.03)
        rng = np.random.default_rng(2)
        for _ in range(2):
            buf.add(rng.uniform([-0.25, 0.0, -0.25], [0.25, 0.8, 0.25], size=(800, 3)))
        buf.view_dirs = [np.array([0.0, -1.0]), np.array([1.0, 0.0])]
        t = ObjectTrack(id=0, label="cabinet", buffer=buf,
                        ellipsoid=Ellipsoid(np.array([0.0, 0.4, -1.0]), np.array([0.25, 0.4, 0.25]), np.eye(3)))
        return layer._refine_geo(t)
    assert refit(0.0) is None
    ell = refit(0.5)
    assert ell is not None and np.linalg.norm(ell.center - np.array([0.0, 0.4, 0.0])) < 0.1


def test_manhattan_frame_recovers_a_rotated_room():
    """The same depth image, its world frame turned 20 deg about up: the walls'
    dominant direction must come back as 20 deg (modulo 90)."""
    import math

    from osg.objects.point_buffer import ManhattanFrame

    frame, _ = frame_at([CABINET], INTR, make_camera(np.array([0.3, 1.3, 2.4]), CABINET.center))
    for turn in (0.0, 20.0):
        a = math.radians(turn)
        Ry = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])
        T = frame.T_wc.copy()
        T[:3, :3] = Ry @ T[:3, :3]
        mf = ManhattanFrame(stride=4)
        mf.add(type(frame)(**{**frame.__dict__, "T_wc": T}))
        got = math.degrees(mf.yaw) % 90.0
        assert min(abs(got - turn), 90.0 - abs(got - turn)) < 2.5


def test_snap_yaw_picks_the_nearest_of_four():
    import math

    from osg.objects.ellipsoid import snap_yaw

    y = math.radians(10.0)
    v = snap_yaw(np.array([math.cos(math.radians(95)), 0.0, math.sin(math.radians(95))]), y)
    assert math.degrees(math.atan2(v[2], v[0])) == pytest.approx(100.0)


def test_decoupled_geometry_moves_while_association_stays_put():
    """With a centre-move limit too tight for any refit, the coupled layer
    keeps every refit out; the decoupled one reports them as geometry and
    still associates exactly as the coupled one does."""
    def run(decouple):
        cfg = _cfg(init="points", buffer=True, refine_mode="points", depth_max_center_move_m=0.001,
                   decouple_geometry=decouple)
        return _orbit_tracks(cfg, CABINET)
    a, b = run(False), run(True)
    assert [len(t.observations) for t in a._tracks.values()] == [len(t.observations) for t in b._tracks.values()]
    moved = [t for t in b._tracks.values() if t.assoc_ellipsoid is not None]
    assert moved and b.funnel.get("geometry_decoupled", 0) > 0
    ta, tb = a._tracks[moved[0].id], moved[0]
    assert not np.allclose(ta.ellipsoid.center, tb.ellipsoid.center)
    assert np.allclose(ta.ellipsoid.center, tb.assoc_ellipsoid.center)


def test_assoc_2d_veto_drops_a_match_whose_points_are_elsewhere():
    from types import SimpleNamespace

    from osg.objects.association import ObjectTrack

    layer = build_object_layer(_cfg(init="points", buffer=True, assoc_2d_min_overlap=0.3))
    buf = VoxelBuffer(0.03)
    rng = np.random.default_rng(3)
    buf.add(rng.uniform([-0.25, 0.0, -0.25], [0.25, 0.9, 0.25], size=(1500, 3)))
    layer._tracks[0] = ObjectTrack(id=0, label="chair", buffer=buf,
                                   ellipsoid=Ellipsoid(np.array([0.0, 0.45, 0.0]), np.array([0.25, 0.45, 0.25]), np.eye(3)))
    here = SimpleNamespace(world=rng.uniform([-0.2, 0.1, -0.2], [0.2, 0.8, 0.2], size=(300, 3)))
    next_chair = SimpleNamespace(world=here.world + np.array([0.9, 0.0, 0.0]))
    dets = [SimpleNamespace(source="detector")] * 2
    out = layer._veto_2d([(0, 0), (1, 0)], dets, {0: here, 1: next_chair}, lambda d: False)
    assert out == [(0, 0), (1, None)]


def test_assoc_2d_reassign_hands_the_detection_to_the_neighbour_that_holds_it():
    from types import SimpleNamespace

    from osg.objects.association import ObjectTrack

    layer = build_object_layer(_cfg(init="points", buffer=True, overlap_mode="voxel",
                                    assoc_3d_overlap=0.3, assoc_2d_reassign=0.1))
    rng = np.random.default_rng(4)
    tracks = []
    for k, x0 in enumerate((0.0, 0.9)):
        buf = VoxelBuffer(0.03)
        buf.add(rng.uniform([x0 - 0.25, 0.0, -0.25], [x0 + 0.25, 0.9, 0.25], size=(1500, 3)))
        t = ObjectTrack(id=k, label="chair", buffer=buf,
                        ellipsoid=Ellipsoid(np.array([x0, 0.45, 0.0]), np.array([0.25, 0.45, 0.25]), np.eye(3)))
        layer._tracks[k] = t
        tracks.append(t)
    on_second = SimpleNamespace(world=rng.uniform([0.7, 0.1, -0.2], [1.1, 0.8, 0.2], size=(300, 3)))
    far = SimpleNamespace(world=on_second.world + np.array([5.0, 0.0, 0.0]))
    dets = [SimpleNamespace(source="detector")] * 2
    out = layer._reassign_2d([(0, 0), (1, 0)], dets, {0: on_second, 1: far}, tracks, lambda d: False)
    assert out[0] == (0, 1)          # the neighbour holds its points
    assert out[1] == (1, 0)          # nobody does: the match stands, no new track


# ------------------------------------------------- save-time consolidation


def _obs_track(tid, label, x0, frames, half=(0.25, 0.45, 0.25)):
    from osg.objects.association import ObjectTrack, Observation

    t = ObjectTrack(id=tid, label=label,
                    ellipsoid=Ellipsoid(np.array([x0, 0.45, 0.0]), np.array(half), np.eye(3)))
    t.observations = [Observation(frame_id=int(f), mu=np.zeros(2), cov=np.eye(2), K=np.eye(3),
                                  T_cw=np.eye(4), mean_depth=1.0) for f in frames]
    return t


def test_consolidate_folds_a_small_fragment_into_its_object():
    from osg.objects.merge import consolidate_minor

    big = _obs_track(0, "bed", 0.0, range(10))
    frag = _obs_track(1, "bed", 0.55, [20])            # AABBs 0.05 m apart
    far = _obs_track(2, "bed", 3.0, [21])               # another bed's fragment
    other = _obs_track(3, "chair", 0.5, [22])          # wrong label
    assert consolidate_minor([big, frag, far, other], 2, 0.2, 3.0, False) == [(1, 0)]


def test_consolidate_keeps_a_fragment_seen_in_the_same_keyframe():
    # Two masks of one label in one frame are two objects (adjacent cabinets).
    from osg.objects.merge import consolidate_minor

    big = _obs_track(0, "cabinet", 0.0, range(10))
    beside = _obs_track(1, "cabinet", 0.5, [3])
    assert consolidate_minor([big, beside], 2, 0.2, 3.0, False) == []


def test_consolidate_needs_a_clearly_bigger_track():
    from osg.objects.merge import consolidate_minor

    a = _obs_track(0, "bed", 0.0, range(4))
    b = _obs_track(1, "bed", 0.5, [10, 11])
    assert consolidate_minor([a, b], 2, 0.2, 3.0, False) == []      # 4 < 3 * 2
    assert consolidate_minor([a, b], 2, 0.2, 2.0, False) == [(1, 0)]


def test_layer_consolidate_merges_and_is_off_by_default():
    layer = build_object_layer(OSGConfig())
    layer._tracks = {0: _obs_track(0, "bed", 0.0, range(10)), 1: _obs_track(1, "bed", 0.5, [20])}
    assert layer.consolidate() == 0 and set(layer._tracks) == {0, 1}

    layer = build_object_layer(_cfg(consolidate_max_obs=2))
    layer._tracks = {0: _obs_track(0, "bed", 0.0, range(10)), 1: _obs_track(1, "bed", 0.5, [20])}
    assert layer.consolidate() == 1
    assert set(layer._tracks) == {0} and len(layer._tracks[0].observations) == 11
    assert layer.get(1) is layer._tracks[0]       # a held id answers as its keeper


def test_buffer_merge_across_voxel_sizes_keeps_points_where_they_are():
    # A snapshot's compressed summary (coarse voxels) absorbing a fresh 3 cm
    # buffer: the fresh keys must not be read as coarse indices.
    coarse = VoxelBuffer(0.03)
    coarse.add(np.random.default_rng(0).uniform([0, 0, 0], [0.5, 0.5, 0.5], size=(2000, 3)))
    coarse.compress(16)
    fresh = VoxelBuffer(0.03)
    near = np.array([[2.0, 0.2, 0.2], [2.01, 0.21, 0.2]])
    fresh.add(near)
    coarse.merge(fresh)
    assert coarse.contains(near, dilate=0).all()
    assert not coarse.contains(np.array([[2.0 * coarse.voxel_m / 0.03, 0.2, 0.2]]), dilate=0).any()
    pts = coarse.points()
    assert np.min(np.linalg.norm(pts - near.mean(axis=0), axis=1)) < 0.02
