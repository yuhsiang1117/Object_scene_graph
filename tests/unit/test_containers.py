"""Container (anchor) layer: floor -> room -> container -> object.

The gap these guard: the graph had no support relation, so "the mug is on THAT
table" was unrepresentable and every dynamic-scene mechanism that is defined
relative to a surface -- the presence filter's visibility check, the search
posterior over where a moved object went -- had nothing to attach to
(docs/DYNAMIC_SCENES.md, Phase 0).

Deliberately Habitat-free, like tests/unit/test_ycb_layouts.py: the whole layer
is decidable from ellipsoid geometry, so it must stay testable on a machine with
no simulator and no HM3D licence.
"""
import numpy as np
import pytest

from osg.graph import containers
from osg.graph.scene_graph import SceneGraph
from osg.graph.serialize import to_json, to_prompt_text
from osg.mapping.costmap import HEIGHT_AXIS, PLANE, Costmap2D
from osg.objects.ellipsoid import Ellipsoid


def ell(axes, R=None):
    return Ellipsoid(
        center=np.zeros(3),  # centre is supplied by the track/view, not the ellipsoid
        axes=np.asarray(axes, float),
        R=np.eye(3) if R is None else R,
    )


def box(center, half_extents, R=None):
    """A track-shaped fake: ellipsoid semi-axes = half extents in world axes."""
    e = ell(half_extents, R)
    e.center = np.asarray(center, float)
    return e


class _Track:
    def __init__(self, tid, label, center, half_extents=(0.1, 0.1, 0.1), R=None, linked=()):
        self.id, self.label = tid, label
        self.center = np.asarray(center, float)
        self.ellipsoid = box(center, half_extents, R)
        self.linked_ids = set(linked)
        self.n_obs, self.best_crop = 5, None


class _Layer:
    """Mirrors ObjectLayer's contract: center_of() averages linked components."""

    def __init__(self, tracks):
        self._t = {t.id: t for t in tracks}

    def tracks(self, include_blacklisted=False):
        return list(self._t.values())

    def center_of(self, t):
        ids = {t.id} | {i for i in t.linked_ids if i in self._t}
        return np.mean([self._t[i].center for i in sorted(ids)], axis=0)


def make(labels_value=1):
    cm = Costmap2D(resolution=0.05, size_m=20.0)
    labels = np.zeros(cm.grid.shape, dtype=np.int32)
    labels[:] = labels_value
    return cm, labels


def build(tracks):
    cm, labels = make()
    sg = SceneGraph()
    sg.rebuild(labels, cm, _Layer(tracks))
    return sg


# World is y-up (HEIGHT_AXIS=1, PLANE=(0,2)); a table top at 0.75 m means the
# ellipsoid centre sits at 0.75 - half_height.
def table(tid=1, xz=(1.0, 1.0), top=0.75, half=(0.6, 0.02, 0.4), label="table", linked=()):
    center = np.zeros(3)
    center[list(PLANE)] = xz
    center[HEIGHT_AXIS] = top - half[HEIGHT_AXIS]
    return _Track(tid, label, center, half, linked=linked)


def item(tid, xz, bottom, half=(0.05, 0.05, 0.05), label="box"):
    center = np.zeros(3)
    center[list(PLANE)] = xz
    center[HEIGHT_AXIS] = bottom + half[HEIGHT_AXIS]
    return _Track(tid, label, center, half)


# ------------------------------------------------------------ the relation


def test_object_on_a_table_gets_that_container():
    sg = build([table(1), item(2, (1.0, 1.0), bottom=0.75)])
    assert set(sg.containers) == {1}
    obj = next(o for o in sg.objects if o.track_id == 2)
    assert obj.container_id == 1
    assert sg.objects_on(1) == [obj]


def test_object_beside_the_table_is_on_nothing():
    """Right height, wrong footprint: 1 m to the side of a 0.6 m half-width."""
    sg = build([table(1), item(2, (2.4, 1.0), bottom=0.75)])
    obj = next(o for o in sg.objects if o.track_id == 2)
    assert obj.container_id is None


def test_object_on_the_floor_is_kept_not_deleted():
    """The anti-DualMap regression. utils/local_map_manager.py:316 deletes a
    high-mobility object with no supporting anchor; we keep it in the room."""
    sg = build([table(1), item(2, (5.0, 5.0), bottom=0.0)])
    obj = next((o for o in sg.objects if o.track_id == 2), None)
    assert obj is not None, "object on the floor was dropped from the graph"
    assert obj.container_id is None
    assert obj.room_id != 0


def test_relative_pose_is_container_relative():
    sg = build([table(1), item(2, (1.3, 1.2), bottom=0.75)])
    obj = next(o for o in sg.objects if o.track_id == 2)
    expected = obj.center - sg.containers[1].center  # R = I for these fakes
    assert obj.p_rel == pytest.approx(expected, abs=1e-9)


# ------------------------------------------------------------- membership


def test_sliver_table_is_rejected_by_area():
    """A mis-segmented sliver carrying an anchor label must not become one."""
    sg = build([table(1, half=(0.02, 0.02, 0.02))])
    assert sg.containers == {}


def test_table_top_out_of_reach_is_rejected():
    sg = build([table(1, top=1.9)])
    assert sg.containers == {}


def test_non_container_category_is_rejected():
    sg = build([table(1, label="picture")])
    assert sg.containers == {}


def test_tilted_ellipsoid_uses_the_true_world_top():
    """R is the camera rotation, refined -- never world-aligned, so reading
    axes[HEIGHT_AXIS] is wrong by however much the object is tilted. A table
    tilted 20 degrees really does present its top ~0.2 m above its centre; the
    naive reading would say 0.02 m and file it a fifth of a metre too low."""
    theta = np.radians(20.0)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])  # tilt within x-y
    half = (0.6, 0.02, 0.4)
    true_half_h = float(np.sqrt(half[0] ** 2 * s ** 2 + half[1] ** 2 * c ** 2))
    assert true_half_h > 10 * half[HEIGHT_AXIS], "test tilt too small to be a real check"

    center = np.zeros(3)
    center[list(PLANE)] = (1.0, 1.0)
    center[HEIGHT_AXIS] = 0.75 - true_half_h  # so the TRUE top sits at 0.75 m
    tr = _Track(1, "table", center, half, R=R)

    assert containers.top_height(tr.center, tr.ellipsoid) == pytest.approx(0.75, abs=1e-9)
    sg = build([tr])
    assert set(sg.containers) == {1}
    assert sg.containers[1].top_h == pytest.approx(0.75, abs=1e-9)


def test_world_extent_matches_the_analytic_support_function():
    e = ell((0.3, 0.1, 0.2))
    for axis, expected in zip(np.eye(3), (0.3, 0.1, 0.2)):
        assert e.world_extent(axis) == pytest.approx(expected, abs=1e-12)
    d = np.array([1.0, 1.0, 0.0])
    assert e.world_extent(d) == pytest.approx(np.sqrt((0.09 + 0.01) / 2), abs=1e-12)


def test_ground_footprint_area_matches_pi_a_b():
    e = ell((0.6, 0.02, 0.4))
    _, cov = e.ground_footprint(PLANE)
    assert np.pi * np.sqrt(np.linalg.det(cov)) == pytest.approx(np.pi * 0.6 * 0.4, abs=1e-9)


# ------------------------------------------------------- components, nesting


def test_linked_component_is_one_container():
    """An L-shaped sofa split across two ellipsoids is ONE anchor, named by the
    smallest member id -- the same component linking.relink() already builds."""
    a = table(3, xz=(1.0, 1.0), top=0.45, half=(0.6, 0.2, 0.4), label="sofa", linked=(7,))
    b = table(7, xz=(2.0, 1.0), top=0.45, half=(0.6, 0.2, 0.4), label="sofa", linked=(3,))
    sg = build([a, b])
    assert set(sg.containers) == {3}
    assert sorted(sg.containers[3].track_ids) == [3, 7]


def test_containers_do_not_nest():
    """A shelf standing on a counter stays a container in its own right and is
    never filed as an object resting on the counter."""
    counter = table(1, xz=(1.0, 1.0), top=0.9, half=(0.8, 0.05, 0.5), label="counter")
    shelf = table(2, xz=(1.0, 1.0), top=1.3, half=(0.3, 0.2, 0.2), label="shelf")
    sg = build([counter, shelf])
    assert set(sg.containers) == {1, 2}
    assert next(o for o in sg.objects if o.track_id == 2).container_id is None


def test_most_specific_surface_wins():
    """An object over both a wide counter and a small stool standing on it rests
    on the stool -- smallest footprint, not first found."""
    counter = table(1, xz=(1.0, 1.0), top=0.9, half=(1.2, 0.05, 0.9), label="counter")
    stool = table(2, xz=(1.0, 1.0), top=0.9, half=(0.15, 0.05, 0.15), label="stool")
    sg = build([counter, stool, item(3, (1.0, 1.0), bottom=0.9)])
    assert next(o for o in sg.objects if o.track_id == 3).container_id == 2


# ------------------------------------------------------------ integration


def test_rooms_index_their_containers():
    sg = build([table(1), item(2, (1.0, 1.0), bottom=0.75)])
    room = sg.rooms[next(iter(sg.rooms))]
    assert room.container_ids == [1]
    assert [c.id for c in sg.containers_in_room(room.id)] == [1]


def test_layer_is_a_noop_for_tracks_without_ellipsoids():
    """test_scene_graph_floors.py's fakes are pose-only; the layer must degrade
    rather than raise, exactly as `floors=None` does."""

    class _Bare:
        def __init__(self, tid, label, center):
            self.id, self.label = tid, label
            self.center = np.asarray(center, float)
            self.n_obs, self.best_crop = 5, None

    class _BareLayer:
        def __init__(self, t):
            self._t = t

        def tracks(self, include_blacklisted=False):
            return self._t

        def center_of(self, t):
            return t.center

    cm, labels = make()
    sg = SceneGraph()
    sg.rebuild(labels, cm, _BareLayer([_Bare(1, "table", [1.0, 0.7, 1.0])]))
    assert sg.containers == {}
    assert len(sg.objects) == 1


def test_prompt_text_is_untouched_by_the_container_layer():
    """Phase 0 is structure only: to_prompt_text feeds the LLM, so changing it
    would change behaviour in the same commit that changes the schema."""
    sg = build([table(1), item(2, (1.0, 1.0), bottom=0.75)])
    assert to_prompt_text(sg) == "Room 1: box x1 (near table), table x1 (near box)"


def test_to_json_carries_containers():
    sg = build([table(1), item(2, (1.0, 1.0), bottom=0.75)])
    d = to_json(sg)
    assert [c["id"] for c in d["containers"]] == [1]
    assert d["containers"][0]["object_ids"] == [2]
    assert d["containers"][0]["top_h"] == pytest.approx(0.75, abs=1e-6)
    obj = next(o for o in d["objects"] if o["track_id"] == 2)
    assert obj["container_id"] == 1 and obj["p_rel"] is not None


# ------------------------------------------------------- fast path == rule


def test_batched_support_matches_the_scalar_rule():
    """ShadowIndex exists only because the per-pair numpy overhead dominated the
    rebuild. It must decide exactly what `supports` decides, or the optimisation
    has quietly changed the map."""
    rng = np.random.default_rng(7)
    surfaces = []  # (cid, top, mu, inv, cov)
    for cid in range(12):
        top = float(rng.uniform(0.2, 1.4))
        mu = rng.uniform(-3, 3, size=2)
        axes = rng.uniform(0.05, 0.8, size=2)
        cov = np.diag(axes ** 2)
        surfaces.append((cid, top, mu, np.linalg.inv(cov), cov))

    index = containers.ShadowIndex.build([(c, t, mu, inv) for c, t, mu, inv, _ in surfaces])
    tol = containers.DEFAULT_SUPPORT_TOL_M

    for _ in range(400):
        bottom = float(rng.uniform(0.0, 1.6))
        xy = rng.uniform(-3.5, 3.5, size=2)
        fast = set(int(c) for c in index.supporting(bottom, xy, tol_m=tol))
        slow = {
            c
            for c, t, mu, inv, _ in surfaces
            if containers.supports(t, [(mu, inv)], bottom, xy, tol_m=tol)
        }
        assert fast == slow, f"batched and scalar disagree at bottom={bottom}, xy={xy}"


def test_shadow_index_handles_an_empty_map():
    index = containers.ShadowIndex.build([])
    assert index.supporting(0.75, np.zeros(2)).size == 0


# ------------------------------------------ linking must not merge a ghost


def test_linking_will_not_merge_an_object_with_its_own_past():
    """The in-anchor ghosting bug, measured: a bowl moved 0.80 m on the same
    table, the stale track and the fresh one were 0.80 m apart under
    link_dist_m=1.0, and relink merged them -- so object_center reported the
    midpoint, 0.42 m from either bowl, a place with no bowl at all. Neither
    observation can ever contradict that point."""
    from osg.objects.association import Observation, ObjectTrack
    from osg.objects.linking import object_center, relink

    def mk(tid, x, frame_id):
        t = ObjectTrack(id=tid, label="bowl",
                        ellipsoid=box([x, 0.8, 0.0], (0.08, 0.08, 0.08)))
        t.observations = [Observation(frame_id=frame_id, mu=np.zeros(2), cov=np.eye(2),
                                      K=np.eye(3), T_cw=np.eye(4), mean_depth=1.0)]
        return t

    ghost, fresh = mk(1, -0.4, -1_000_000), mk(2, 0.4, 12)
    tracks = [ghost, fresh]

    relink(tracks, link_dist_m=1.0)  # no window: the old behaviour
    assert ghost.linked_ids == {2}
    assert object_center(fresh, {t.id: t for t in tracks})[0] == pytest.approx(0.0)

    relink(tracks, link_dist_m=1.0, max_frame_gap=50)
    assert ghost.linked_ids == set() and fresh.linked_ids == set()
    assert object_center(fresh, {t.id: t for t in tracks})[0] == pytest.approx(0.4)


def test_linking_still_merges_fragments_seen_together():
    """The behaviour relink exists for: an L-shaped sofa split across two
    ellipsoids, both observed in the same frames."""
    from osg.objects.association import Observation, ObjectTrack
    from osg.objects.linking import relink

    def mk(tid, x, frame_id):
        t = ObjectTrack(id=tid, label="sofa",
                        ellipsoid=box([x, 0.5, 0.0], (0.4, 0.3, 0.4)))
        t.observations = [Observation(frame_id=frame_id, mu=np.zeros(2), cov=np.eye(2),
                                      K=np.eye(3), T_cw=np.eye(4), mean_depth=2.0)]
        return t

    a, b = mk(1, -0.3, 40), mk(2, 0.3, 42)
    relink([a, b], link_dist_m=1.0, max_frame_gap=50)
    assert a.linked_ids == {2} and b.linked_ids == {1}
