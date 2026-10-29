"""The per-keyframe hot paths were rewritten for speed with the promise that every
value is the one the old code computed. These pin that promise bit for bit
against the straightforward versions they replaced."""
import numpy as np
import pytest

from osg.graph.scene_graph import _room_cells
from osg.objects.association import Observation, ObjectTrack
from osg.objects.ellipsoid import Ellipsoid
from osg.objects.linking import _near_pairs, relink
from osg.objects.optimization import residuals, view_constants


def _room_cells_loop(labels):
    out = []
    for rid in np.unique(labels):
        if rid == 0:
            continue
        mask = labels == rid
        out.append((int(rid), np.argwhere(mask).mean(axis=0), int(mask.sum())))
    return out


def test_room_cells_match_the_per_room_loop_exactly():
    rng = np.random.default_rng(0)
    for shape, k in (((400, 600), 12), ((37, 53), 3), ((1, 1), 1), ((50, 50), 0)):
        labels = rng.integers(0, k + 1, size=shape).astype(np.int32)
        got, want = _room_cells(labels), _room_cells_loop(labels)
        assert [g[0] for g in got] == [w[0] for w in want]
        for g, w in zip(got, want):
            assert g[2] == w[2]
            assert np.array_equal(g[1], w[1])      # bit-identical, not allclose


def _track(i, centre, frame_id):
    t = ObjectTrack(id=i, label="chair",
                    ellipsoid=Ellipsoid(center=np.asarray(centre, float), axes=np.ones(3) * 0.2,
                                        R=np.eye(3)))
    t.observations.append(Observation(frame_id=frame_id, mu=np.zeros(2), cov=np.eye(2),
                                      K=np.eye(3), T_cw=np.eye(4), mean_depth=1.0))
    return t


def test_near_pairs_are_the_pairs_the_scalar_loop_tests():
    rng = np.random.default_rng(1)
    centres = rng.uniform(-3, 3, size=(300, 3))
    centres[7] = np.nan                     # the scalar `d >= bar` is False for NaN
    group = [_track(i, c, 0) for i, c in enumerate(centres)]
    for bar in (0.4, 1.0):
        got = list(_near_pairs(group, bar, rows=64))
        want = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                d = np.linalg.norm(group[i].ellipsoid.center - group[j].ellipsoid.center)
                if not d >= bar:
                    want.append((i, j))
        assert got == want


def test_relink_components_unchanged_by_the_prefilter():
    rng = np.random.default_rng(2)
    tracks = [_track(i, c, int(f)) for i, (c, f) in
              enumerate(zip(rng.uniform(-4, 4, size=(200, 3)), rng.integers(0, 40, size=200)))]
    relink(tracks, link_dist_m=1.0, max_frame_gap=10)
    got = {t.id: set(t.linked_ids) for t in tracks}
    # The scalar union-find this replaced.
    parent = {t.id: t.id for t in tracks}

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for a in range(len(tracks)):
        for b in range(a + 1, len(tracks)):
            ta, tb = tracks[a], tracks[b]
            if np.linalg.norm(ta.ellipsoid.center - tb.ellipsoid.center) >= 1.0:
                continue
            if abs(ta.observations[-1].frame_id - tb.observations[-1].frame_id) > 10:
                continue
            parent[find(tb.id)] = find(ta.id)
    comps = {}
    for t in tracks:
        comps.setdefault(find(t.id), set()).add(t.id)
    want = {t.id: (comps[find(t.id)] - {t.id}) if len(comps[find(t.id)]) > 1 else set() for t in tracks}
    assert got == want


def test_residuals_with_view_constants_are_bit_identical():
    rng = np.random.default_rng(3)
    K = np.array([[776.0, 0, 640], [0, 776.0, 480], [0, 0, 1]])
    obs = []
    for k in range(8):
        T = np.eye(4)
        T[:3, 3] = [rng.uniform(-0.5, 0.5), rng.uniform(-0.2, 0.2), 3.0 + rng.uniform(0, 1)]
        A = rng.normal(size=(2, 2))
        obs.append(Observation(frame_id=k, mu=rng.uniform(300, 900, 2), cov=A @ A.T + np.eye(2) * 50,
                               K=K, T_cw=T, mean_depth=3.0))
    from osg.core.geometry import sqrtm_2x2_spd
    from osg.objects.optimization import _ellipsoid_from

    def residuals_per_view(params):          # the loop before view_constants existed
        e = _ellipsoid_from(params)
        res = []
        for o in obs:
            proj = e.project(o.K, o.T_cw)
            if proj is None:
                res.extend([1e3] * 5)
                continue
            dmu = proj.mu - o.mu
            ds = sqrtm_2x2_spd(proj.cov) - sqrtm_2x2_spd(o.cov)
            res.extend([dmu[0], dmu[1], ds[0, 0], ds[1, 1], ds[0, 1]])
        return np.asarray(res)

    for _ in range(20):
        params = np.concatenate([rng.normal(size=3) * 0.3, np.log(rng.uniform(0.05, 0.5, 3)),
                                 rng.normal(size=3) * 0.5])
        want = residuals_per_view(params)
        assert np.array_equal(residuals(params, obs), want)
        assert np.array_equal(residuals(params, obs, view_constants(obs)), want)


def test_batched_residuals_match_the_per_view_loop_with_views_that_fail():
    """Views behind the camera, or projecting to a conic `project` rejects,
    take the 1e3 penalty row in place, exactly as the loop gives it."""
    from osg.core.geometry import rotvec_to_matrix, sqrtm_2x2_spd
    from osg.objects.optimization import _ellipsoid_from

    rng = np.random.default_rng(4)
    K = np.array([[776.0, 0, 640], [0, 776.0, 480], [0, 0, 1]])
    n_failed = 0
    for _ in range(300):
        obs = []
        for k in range(int(rng.integers(3, 11))):
            T = np.eye(4)
            T[:3, :3] = rotvec_to_matrix(rng.normal(size=3) * 0.3)
            T[:3, 3] = [rng.uniform(-1, 1), rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 5.0)]
            A = rng.normal(size=(2, 2)) * rng.uniform(1, 40)
            obs.append(Observation(frame_id=k, mu=rng.uniform(0, 1280, 2), cov=A @ A.T + np.eye(2) * 5,
                                   K=K, T_cw=T, mean_depth=3.0))
        consts = view_constants(obs)
        for _ in range(4):
            params = np.concatenate([rng.normal(size=3) * 0.5, np.log(rng.uniform(0.01, 2.0, 3)),
                                     rng.normal(size=3)])
            e = _ellipsoid_from(params)
            want = []
            for o in obs:
                proj = e.project(o.K, o.T_cw)
                if proj is None:
                    want.extend([1e3] * 5)
                    n_failed += 1
                    continue
                dmu = proj.mu - o.mu
                ds = sqrtm_2x2_spd(proj.cov) - sqrtm_2x2_spd(o.cov)
                want.extend([dmu[0], dmu[1], ds[0, 0], ds[1, 1], ds[0, 1]])
            assert np.array_equal(residuals(params, obs, consts), np.asarray(want))
    assert n_failed > 100                     # the failing branch was exercised


def test_the_cached_mask_scan_is_the_scan_and_each_ellipse_is_fresh():
    from osg.core.geometry import det_ellipse, ellipse_from_mask, mask_pixels
    from osg.core.types import Detection

    rng = np.random.default_rng(5)
    for _ in range(50):
        mask = np.zeros((96, 128), bool)
        y0, x0 = rng.integers(0, 80), rng.integers(0, 110)
        mask[y0:y0 + rng.integers(1, 16), x0:x0 + rng.integers(1, 18)] = True
        mask |= rng.random(mask.shape) < 0.002          # stray pixels outside the blob
        det = Detection(label="mug", score=0.9, bbox_xyxy=np.zeros(4), mask=mask)
        ys, xs = mask_pixels(det)
        want_ys, want_xs = np.nonzero(mask)
        assert np.array_equal(ys, want_ys) and np.array_equal(xs, want_xs)
        a, b, want = det_ellipse(det), det_ellipse(det), ellipse_from_mask(mask)
        if want is None:
            assert a is None
            continue
        assert np.array_equal(a.mu, want.mu) and np.array_equal(a.cov, want.cov)
        assert a is not b and a.cov is not b.cov             # a fresh object, as before
        det.mask = np.zeros_like(mask)                       # a replaced mask is rescanned
        assert det_ellipse(det) is None and mask_pixels(det)[0].size == 0


# ------------------------------------------------ scene graph: geometry memo, room prefilter

def _rot(rng):
    a = rng.normal(size=3) * 0.4
    th = float(np.linalg.norm(a))
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]) / max(th, 1e-12)
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k


def test_member_geometry_is_the_direct_call_and_follows_in_place_edits():
    from osg.graph import containers as cm
    from osg.graph.scene_graph import SceneGraph, _AREA, _BOTTOM, _QUERY, _TOP

    rng = np.random.default_rng(3)
    g = SceneGraph()
    direct = {_TOP: cm.top_height, _AREA: lambda c, e: cm.footprint_area(e),
              _QUERY: cm.footprint_query, _BOTTOM: cm.bottom_height}

    def same(a, b):
        if isinstance(a, tuple):
            return all(np.array_equal(x, y) for x, y in zip(a, b))
        return (a is None and b is None) or a == b

    for _ in range(300):
        c = rng.normal(size=3)
        e = Ellipsoid(center=c.copy(), axes=rng.uniform(0.0, 1.5, 3), R=_rot(rng))
        for _ in range(2):                                # the second round is the memo's
            for w, f in direct.items():
                assert same(g._member_geometry(c, e, w), f(c, e))
        e.axes[1] *= 1.7                                  # refined in place
        for w, f in direct.items():
            assert same(g._member_geometry(c, e, w), f(c, e))
    q = g._member_geometry(c, Ellipsoid(center=c, axes=np.ones(3), R=np.eye(3)), _QUERY)
    assert not q[0].flags.writeable and not q[1].flags.writeable


def test_indexed_nearest_room_is_the_loop():
    from osg.graph.scene_graph import RoomNode, SceneGraph

    rng = np.random.default_rng(5)
    for _ in range(40):
        g = SceneGraph()
        for rid in range(1, int(rng.integers(1, 30))):
            xy = rng.uniform(-8, 8, 2)
            if rng.random() < 0.15 and g.rooms:           # a tie with an earlier room
                xy = next(iter(g.rooms.values())).centroid_xy.copy()
            if rng.random() < 0.05:
                xy = np.array([np.nan, 0.0])
            g.rooms[rid] = RoomNode(id=rid, centroid_xy=xy, floor=int(rng.integers(0, 2)))
        for _ in range(200):
            xy = rng.uniform(-10, 10, 2)
            if rng.random() < 0.1:                        # exactly on a centroid
                xy = g.rooms[int(rng.integers(1, len(g.rooms) + 1))].centroid_xy.copy() \
                    if g.rooms else xy
            for floor_id in (None, 0, 1):
                for max_dist in (3.0, 0.0, 50.0):
                    g._room_index = None
                    want = g._nearest_room(xy, max_dist=max_dist, floor_id=floor_id)
                    g._room_index = {}
                    assert g._nearest_room(xy, max_dist=max_dist, floor_id=floor_id) == want
        g._room_index = None


def test_rebuild_floor_unchanged_by_the_memo_and_the_index():
    """Five rebuilds over a layer whose ellipsoids are refined in place between
    them: every room, object and container field as the direct calls make it."""
    from osg.graph.scene_graph import _GEOMETRY, SceneGraph
    from osg.mapping.costmap import Costmap2D

    class _Layer:
        def __init__(self, tracks):
            self._t = tracks

        def tracks(self, include_blacklisted=False):
            return list(self._t)

        def center_of(self, track):
            return np.asarray(track.ellipsoid.center, dtype=float)

    rng = np.random.default_rng(11)
    tracks = []
    for tid in range(1, 90):
        label = str(rng.choice(["table", "bed", "counter", "mug", "bowl", "chair"]))
        big = label in ("table", "bed", "counter")
        c = np.array([rng.uniform(-6, 6), rng.uniform(0.3, 0.9) if big else rng.uniform(0.7, 1.2),
                      rng.uniform(-6, 6)])
        axes = rng.uniform(0.3, 1.2, 3) if big else rng.uniform(0.04, 0.15, 3)
        t = ObjectTrack(id=tid, label=label, ellipsoid=Ellipsoid(center=c, axes=axes, R=_rot(rng)))
        t.best_score = float(rng.uniform(0.3, 1.0))
        t.floor_key = 0
        for _ in range(int(rng.integers(1, 5))):
            t.observations.append(None)
        tracks.append(t)
    layer = _Layer(tracks)
    costmap = Costmap2D(resolution=0.1, size_m=20.0)
    labels = np.zeros(costmap.grid.shape, dtype=np.int32)
    h, w = labels.shape
    labels[: h // 2, : w // 2], labels[h // 2:, : w // 3], labels[: h // 3, w // 2:] = 1, 2, 3
    kw = dict(container_min_area_m2=0.05, container_min_obs=2, container_min_score=0.4,
              container_merge_m=1.0, container_merge_sigma=1.0)
    new, ref = SceneGraph(**kw), SceneGraph(**kw)
    ref._member_geometry = lambda c, e, what: _GEOMETRY[what](c, e)

    def ref_nearest(xy, max_dist, floor_id):
        saved, ref._room_index = ref._room_index, None
        try:
            return SceneGraph._nearest_room(ref, xy, max_dist=max_dist, floor_id=floor_id)
        finally:
            ref._room_index = saved
    ref._nearest_room_indexed = ref_nearest

    def dump(g):
        rooms = {k: (r.label, r.centroid_xy.tobytes(), r.n_cells, r.floor, list(r.container_ids))
                 for k, r in g.rooms.items()}
        objs = [(o.track_id, o.room_id, o.container_id, None if o.p_rel is None else o.p_rel.tobytes())
                for o in g.objects]
        conts = {k: (c.label, list(c.track_ids), c.center.tobytes(), c.top_h, c.area_m2, c.room_id,
                     list(c.object_ids)) for k, c in g.containers.items()}
        return rooms, objs, conts

    for round_ in range(5):
        for g in (new, ref):
            g.rebuild_floor(labels, costmap, layer, floor_key=0, floor_height=0.0)
        assert dump(new) == dump(ref), f"round {round_}"
        assert new._room_index is None
        for t in rng.choice(tracks, 15, replace=False):   # a refine between rebuilds
            t.ellipsoid.axes *= rng.uniform(0.8, 1.25, 3)
            t.ellipsoid.center[1] += rng.uniform(-0.05, 0.05)
    assert len(dump(new)[2]) > 0 and any(o[2] is not None for o in dump(new)[1])


def test_device_side_preprocess_is_the_transform_bit_for_bit():
    torch = pytest.importorskip("torch")
    transform = pytest.importorskip("open_clip.transform")
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    from osg.perception.feature_encoder import prepare_batch, split_preprocess

    rng = np.random.default_rng(3)
    crops = [rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
             for h, w in ((30, 40), (257, 90), (12, 300), (256, 256), (1, 1))]
    pool = ThreadPoolExecutor(max_workers=2)
    for pre in (   # MobileCLIP-S2's transform, and ViT-B/32's (mean / std, bicubic)
        transform.image_transform(256, is_train=False, mean=(0.0,) * 3, std=(1.0,) * 3,
                                  interpolation="bilinear"),
        transform.image_transform(224, is_train=False),
    ):
        split = split_preprocess(pre)
        assert split is not None
        want = torch.stack([pre(Image.fromarray(c).convert("RGB")) for c in crops])
        for p in (None, pool):
            got = prepare_batch(crops, pre, split, "cpu", p)
            assert got.dtype == want.dtype and got.shape == want.shape
            assert torch.equal(got, want)
        assert torch.equal(prepare_batch(crops[:1], pre, split, "cpu", pool), want[:1])
    assert split_preprocess(lambda im: im) is None      # not a Compose: the plain path
