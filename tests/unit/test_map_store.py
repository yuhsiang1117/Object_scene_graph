"""Map snapshots: the map survives the episode that built it (Phase 2).

The dynamic benchmark is two passes -- explore the static layout, move the
objects, then navigate the moved world WITH THE OLD MAP. The staleness is the
experiment, so it is the snapshot that has to be faithful: a track that loses
its observations is a candidate the pipeline silently ignores, and a belief that
resets to 0.82 erases the evidence the whole comparison is about.

Habitat-free: a snapshot is JSON plus grids.
"""
import numpy as np
import pytest

from osg.graph.map_store import MapStoreError, apply_map, load_map, save_map
from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN, Costmap2D
from osg.objects.association import Observation, ObjectTrack
from osg.objects.ellipsoid import Ellipsoid
from osg.objects.object_layer import ObjectLayer


class _Layer:
    """Minimal stand-in for the FloorStack seam the agent exposes."""

    def __init__(self, costmap, room_labels=None):
        self.costmap = costmap
        self.room_labels = room_labels
        self._layers = {0: self}
        self.current = self


class _Agent:
    def __init__(self, object_layer, costmap, room_labels=None, scene_graph=None):
        self.object_layer = object_layer
        self._floor_stack = _Layer(costmap, room_labels)
        self.scene_graph = scene_graph or _NullGraph()

    @property
    def costmap(self):
        return self._floor_stack.costmap

    @property
    def _room_labels(self):
        return self._floor_stack.room_labels

    @_room_labels.setter
    def _room_labels(self, value):
        self._floor_stack.room_labels = value


class _NullGraph:
    def __init__(self):
        self.rebuilt = 0

    def rebuild(self, *a, **kw):
        self.rebuilt += 1


def track(tid=1, label="mug", center=(1.0, 0.8, 2.0), n_obs=3):
    t = ObjectTrack(
        id=tid, label=label,
        ellipsoid=Ellipsoid(center=np.array(center, float),
                            axes=np.array([0.1, 0.2, 0.3]),
                            R=np.eye(3)),
    )
    t.best_score = 0.77
    t.best_bbox_px = 4321.0
    t.evidence = 2.5
    t.best_cam_xy = np.array([0.5, 0.25])
    t.first_cam_xy = np.array([0.1, 0.2])
    for i in range(n_obs):
        t.observations.append(
            Observation(frame_id=i, mu=np.array([10.0 + i, 20.0]),
                        cov=np.eye(2) * 3.0, K=np.eye(3), T_cw=np.eye(4),
                        mean_depth=1.5 + i)
        )
    return t


def agent_with(tracks, *, size_m=20.0, room_labels=None):
    layer = ObjectLayer()
    for t in tracks:
        layer._tracks[t.id] = t
    layer._next_id = max((t.id for t in tracks), default=-1) + 1
    cm = Costmap2D(resolution=0.05, size_m=size_m)
    return _Agent(layer, cm, room_labels)


def roundtrip(agent, tmp_path, into=None):
    save_map(tmp_path / "m.json", agent, scene="s1", layout_id="static")
    blob = load_map(tmp_path / "m.json")
    target = into or agent_with([])
    apply_map(target, blob)
    return target


# ---------------------------------------------------------------- fidelity


def test_tracks_survive_with_geometry_and_score(tmp_path):
    out = roundtrip(agent_with([track(1), track(2, label="bowl")]), tmp_path)
    restored = {t.id: t for t in out.object_layer.tracks()}
    assert set(restored) == {1, 2}
    assert restored[2].label == "bowl"
    assert restored[1].ellipsoid.center == pytest.approx([1.0, 0.8, 2.0])
    assert restored[1].ellipsoid.axes == pytest.approx([0.1, 0.2, 0.3])
    assert restored[1].best_score == pytest.approx(0.77)


def test_observations_survive_so_candidates_still_qualify(tmp_path):
    """n_obs gates candidate selection; a snapshot that drops observations
    loads tracks the pipeline then refuses to propose."""
    out = roundtrip(agent_with([track(1, n_obs=4)]), tmp_path)
    restored = out.object_layer.get(1)
    assert restored.n_obs == 4
    assert restored.observations[2].mean_depth == pytest.approx(3.5)
    assert restored.observations[0].T_cw.shape == (4, 4)
    assert out.object_layer.candidates("mug", min_obs=3) == [restored]


def test_presence_belief_survives(tmp_path):
    """The whole experiment is about what the map still believes. Resetting the
    belief on load would erase exactly the thing being measured."""
    t = track(1)
    t.presence.log_odds = -3.5
    t.presence.n_missed = 7
    t.presence.last_seen_kf = 42
    out = roundtrip(agent_with([t]), tmp_path)
    restored = out.object_layer.get(1)
    assert restored.presence.log_odds == pytest.approx(-3.5)
    assert restored.presence.n_missed == 7
    assert restored.presence.p < 0.05


def test_blacklist_and_links_survive(tmp_path):
    a, b = track(1), track(2)
    a.blacklisted = True
    a.linked_ids, b.linked_ids = {2}, {1}
    out = roundtrip(agent_with([a, b]), tmp_path)
    assert out.object_layer.get(1).blacklisted is True
    assert out.object_layer.get(1).linked_ids == {2}
    assert [t.id for t in out.object_layer.tracks()] == [2], "blacklisted track proposed"


def test_next_track_id_does_not_collide_after_load(tmp_path):
    """A reused id would merge a new object into an old one's history."""
    out = roundtrip(agent_with([track(7)]), tmp_path)
    assert out.object_layer._next_id > 7


def test_occupancy_and_rooms_survive(tmp_path):
    src = agent_with([track(1)])
    src.costmap.grid[3, 4] = OCCUPIED
    src.costmap.grid[5, 6] = FREE
    labels = np.zeros(src.costmap.grid.shape, dtype=np.int32)
    labels[10:20, 10:20] = 3
    src._room_labels = labels
    out = roundtrip(src, tmp_path)
    assert out.costmap.grid[3, 4] == OCCUPIED
    assert out.costmap.grid[5, 6] == FREE
    assert out.costmap.grid[0, 0] == UNKNOWN
    assert out._room_labels[15, 15] == 3


def test_scene_graph_is_rebuilt_not_restored(tmp_path):
    """Derived structure must be recomputed, or a snapshot freezes yesterday's
    container rule into today's run."""
    src = agent_with([track(1)])
    src._room_labels = np.ones(src.costmap.grid.shape, dtype=np.int32)
    out = roundtrip(src, tmp_path)
    assert out.scene_graph.rebuilt == 1


# ------------------------------------------------------------------ guards


def test_multi_floor_save_is_refused_rather_than_lossy(tmp_path):
    src = agent_with([track(1)])
    src._floor_stack._layers[1] = object()
    with pytest.raises(MapStoreError, match="single-storey"):
        save_map(tmp_path / "m.json", src, scene="s1")


def test_a_grown_costmap_loads_into_a_fresh_one(tmp_path):
    """The costmap grows as the agent explores, so a snapshot is routinely
    larger than the 20 m grid a fresh agent starts with. Refusing that would
    reject every real mapping run."""
    src = agent_with([track(1)], size_m=40.0)
    src.costmap.grid[7, 8] = OCCUPIED
    dst = agent_with([], size_m=20.0)
    save_map(tmp_path / "m.json", src, scene="s1")
    apply_map(dst, load_map(tmp_path / "m.json"))
    assert dst.costmap.grid.shape == src.costmap.grid.shape
    assert dst.costmap.grid[7, 8] == OCCUPIED


def test_resolution_mismatch_is_refused(tmp_path):
    """Different resolution IS a real incompatibility: the same array indexes
    different world coordinates."""
    save_map(tmp_path / "m.json", agent_with([track(1)]), scene="s1")
    blob = load_map(tmp_path / "m.json")
    coarse = agent_with([])
    coarse.costmap.resolution = 0.1
    with pytest.raises(MapStoreError, match="resolution"):
        apply_map(coarse, blob)


def test_missing_snapshot_names_the_path(tmp_path):
    with pytest.raises(MapStoreError, match="no map snapshot"):
        load_map(tmp_path / "absent.json")


def test_schema_version_mismatch_is_refused(tmp_path):
    import json
    save_map(tmp_path / "m.json", agent_with([track(1)]), scene="s1")
    blob = json.loads((tmp_path / "m.json").read_text())
    blob["schema_version"] = 999
    (tmp_path / "m.json").write_text(json.dumps(blob))
    with pytest.raises(MapStoreError, match="schema"):
        load_map(tmp_path / "m.json")


# ------------------------------------------------------ ghosting on reload


def test_a_carried_belief_cannot_assert_certainty(tmp_path):
    """The map was built in another session and the world had every chance to
    change. A belief restored at p=0.998 needs seven clean misses to unwind, so
    the agent commits to a stale goal on step 1 and the episode ends before the
    evidence arrives."""
    t = track(1)
    t.presence.log_odds = 6.0
    out = roundtrip(agent_with([t]), tmp_path)
    assert out.object_layer.get(1).presence.log_odds == pytest.approx(1.5)


def test_disbelief_is_carried_across_unchanged(tmp_path):
    """An object already known to be gone has not become more likely by sitting
    in a file -- the cap limits confidence, not doubt."""
    t = track(1)
    t.presence.log_odds = -4.0
    out = roundtrip(agent_with([t]), tmp_path)
    assert out.object_layer.get(1).presence.log_odds == pytest.approx(-4.0)


def test_restored_observations_are_marked_as_a_previous_session(tmp_path):
    """relink must be able to tell "seen a moment ago" from "seen before the
    world changed"; raw frame ids restart each episode and would collide."""
    out = roundtrip(agent_with([track(1, n_obs=3)]), tmp_path)
    assert all(o.frame_id < 0 for o in out.object_layer.get(1).observations)
