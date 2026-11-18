"""Handing a stored ASCENT obstacle map to the OSG pipeline.

The shapes here are taken from a real pass, not invented: the first 00800
snapshot had FOUR storeys of which two were empty, because `_new_floor`
allocates a storey the moment a staircase is detected and the agent never
walked two of them. Floor 0 was one of the empty ones, so naive order-matching
handed a fresh agent a map with nothing in it and called that a success.
"""
from __future__ import annotations

import numpy as np
import pytest

from navigation.geometry import EpisodeAnchor
from navigation.mapping.map_store import save_obstacle_maps
from navigation.mapping.obstacle_map import ObstacleMap
from osg.eval.prior_map import load_obstacle_map
from osg.mapping.costmap import Costmap2D

SIZE, PPM = 400, 20


def _om(explored=None, occupied=None) -> ObstacleMap:
    om = ObstacleMap(min_height=0.61, max_height=0.88, agent_radius=0.18,
                     area_thresh=1.5, hole_area_thresh=100000, size=SIZE)
    if explored:
        r0, r1, c0, c1 = explored
        om.explored_area[r0:r1, c0:c1] = True
    if occupied:
        r0, r1, c0, c1 = occupied
        om._map[r0:r1, c0:c1] = True
    return om


class _SaveAgent:
    def __init__(self, floors, anchor, floor_idx=0):
        self._floors = [{"obstacle": om} for om in floors]
        self.anchor = anchor
        self._floor_idx = floor_idx


class _Layer:
    def __init__(self, floor_y, costmap):
        self.floor_y = floor_y
        self.costmap = costmap


class _Stack:
    def __init__(self, layers):
        self._layers = layers


class _LoadAgent:
    """Only what `load_obstacle_map` touches on a NavAgent."""

    def __init__(self, layers):
        self._floor_stack = _Stack(layers)


class _Cfg:
    class ycb:
        obstacle_map_in = ""
        obstacle_map_out = ""
        obstacle_map_overwrite = False
        obstacle_map_union = False
        seed_storeys_from_obstacle_map = False
        storey_seed_tol_m = 1.0


def _cfg(map_in: str, overwrite: bool = False, union: bool = False):
    cfg = _Cfg()
    cfg.ycb.obstacle_map_in = str(map_in)
    cfg.ycb.obstacle_map_out = ""
    cfg.ycb.obstacle_map_overwrite = overwrite
    cfg.ycb.obstacle_map_union = union
    cfg.ycb.seed_storeys_from_obstacle_map = False
    cfg.ycb.storey_seed_tol_m = 1.0
    return cfg


def _four_floors(tmp_path):
    """Two empty placeholders around two real storeys -- the 00800 shape."""
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    floors = [
        _om(),                                                  # 0: placeholder
        _om(explored=(150, 250, 150, 250), occupied=(160, 170, 160, 170)),
        _om(explored=(180, 230, 180, 230), occupied=(190, 195, 190, 195)),
        _om(),                                                  # 3: placeholder
    ]
    save_obstacle_maps(tmp_path / "00800-TEEsavR23oF.json",
                       _SaveAgent(floors, anchor, floor_idx=2),
                       scene="00800-TEEsavR23oF", layout_id="static")
    return tmp_path


def test_a_fresh_single_floor_agent_gets_a_mapped_storey(tmp_path):
    """The regression: floor 0 of that snapshot is EMPTY, and a fresh agent has
    exactly one layer. Matching by raw order loads nothing and says so."""
    maps = _four_floors(tmp_path)
    costmap = Costmap2D(resolution=1.0 / PPM, size_m=20.0)
    agent = _LoadAgent({0: _Layer(0.0, costmap)})

    note = load_obstacle_map(_cfg(maps), agent, "00800-TEEsavR23oF")

    assert note is not None
    assert note["stored_floors"] == 4
    assert note["mapped_floors"] == 2
    assert note["cells_written"] > 0, "a fresh agent was handed an empty map"
    assert note["matched_floors"][0]["ascent_floor"] == 1, "skipped the placeholder"
    assert (costmap.grid != -1).any()


def test_two_storeys_match_bottom_up(tmp_path):
    maps = _four_floors(tmp_path)
    lower = Costmap2D(resolution=1.0 / PPM, size_m=20.0)
    upper = Costmap2D(resolution=1.0 / PPM, size_m=20.0)
    # Deliberately out of key order: the match is by HEIGHT, not by key.
    agent = _LoadAgent({7: _Layer(3.1, upper), 2: _Layer(0.0, lower)})

    note = load_obstacle_map(_cfg(maps), agent, "00800-TEEsavR23oF")

    assert [p["ascent_floor"] for p in note["matched_floors"]] == [1, 2]
    assert [p["osg_floor"] for p in note["matched_floors"]] == [2, 7]
    assert all(p["cells"] > 0 for p in note["matched_floors"])


def test_no_configured_input_is_not_an_error(tmp_path):
    agent = _LoadAgent({0: _Layer(0.0, Costmap2D(resolution=0.05, size_m=20.0))})
    assert load_obstacle_map(_cfg(""), agent, "00800-TEEsavR23oF") is None


def test_a_missing_snapshot_is_loud(tmp_path):
    """Silently navigating without the map under test is the failure mode that
    produces a number nobody can interpret."""
    agent = _LoadAgent({0: _Layer(0.0, Costmap2D(resolution=0.05, size_m=20.0))})
    with pytest.raises(FileNotFoundError):
        load_obstacle_map(_cfg(tmp_path), agent, "00800-TEEsavR23oF")


def test_more_stored_storeys_than_the_agent_has_is_reported(tmp_path):
    maps = _four_floors(tmp_path)
    agent = _LoadAgent({0: _Layer(0.0, Costmap2D(resolution=1.0 / PPM, size_m=20.0))})

    note = load_obstacle_map(_cfg(maps), agent, "00800-TEEsavR23oF")

    assert note["unmatched_floors"] == 1
    assert len(note["matched_floors"]) == 1


# --------------------------------------------------------- which map is kept

def test_the_kept_map_is_the_one_that_walked_more_storeys(tmp_path):
    """Ranking on `len(_floors)` would keep the wrong episode.

    `_new_floor` allocates a storey the moment a staircase is DETECTED, so an
    episode that glimpsed three staircases and explored one room carries four
    entries while one that mapped the whole building carries two.
    """
    from osg.eval.prior_map import save_obstacle_map_for_scene

    class _Episode:
        episode_id = "0"
        scene_id = "data/scene_datasets/hm3d/val/00800-x/x.basis.glb"

    cfg = _cfg("")
    cfg.ycb.obstacle_map_out = str(tmp_path)
    episode = _Episode()

    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    # First: four storeys, three of them empty placeholders, one small room.
    glimpser = _SaveAgent([_om(explored=(100, 110, 100, 110)), _om(), _om(), _om()],
                          anchor)
    # Second: two storeys, both genuinely walked, far more of the building.
    mapper = _SaveAgent([_om(explored=(100, 300, 100, 300)),
                         _om(explored=(120, 260, 120, 260))], anchor)

    save_obstacle_map_for_scene(cfg, glimpser, episode)
    save_obstacle_map_for_scene(cfg, mapper, episode)

    from navigation.mapping.map_store import floor_summaries, load_obstacle_maps

    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1, f"expected one snapshot, got {written}"
    kept = [f for f in floor_summaries(
        load_obstacle_maps(written[0])) if f["explored_cells"] > 0]
    assert len(kept) == 2, "kept the episode that only saw staircases"
    assert sum(f["explored_cells"] for f in kept) > 40000


def test_a_policy_without_an_obstacle_map_writes_nothing(tmp_path):
    """`nav_agent` has no `_floors`; an empty snapshot loads silently."""
    from osg.eval.prior_map import save_obstacle_map_for_scene

    cfg = _cfg("")
    cfg.ycb.obstacle_map_out = str(tmp_path)

    class _NoFloors:
        pass

    class _Episode:
        episode_id = "0"
        scene_id = "data/scene_datasets/hm3d/val/00800-x/x.basis.glb"

    assert save_obstacle_map_for_scene(cfg, _NoFloors(), _Episode()) is None
    assert not list(tmp_path.glob("*.json"))


# ------------------------------------------------- the map's own diagnostic

def test_traj_on_map_is_recorded_but_does_not_gate_reuse(tmp_path):
    """`traj_on_map` is a coverage diagnostic, not a validity test.

    It reads like one, which is why this test exists. `explored_area` is what
    the agent SAW: obstacle_map.py:374 erases an agent-radius band around every
    obstacle each step, so a real pass scores far below 1.0 with a perfectly
    good map (measured on 00800: bowl 0.54, pitcher 0.11, and pitcher never
    climbed). A map must still load.
    """
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    om = _om(explored=(150, 250, 150, 250))
    om._camera_positions = [np.array([12.0, 12.0]) for _ in range(50)]  # all off it

    save_obstacle_maps(tmp_path / "00800-TEEsavR23oF.json",
                       _SaveAgent([om], anchor), scene="00800-TEEsavR23oF")
    agent = _LoadAgent({0: _Layer(0.0, Costmap2D(resolution=1.0 / PPM, size_m=20.0))})

    note = load_obstacle_map(_cfg(tmp_path), agent, "00800-TEEsavR23oF")

    assert note is not None, "a low traj_on_map must not block reuse"
    assert note["traj_on_map"] == [0.0], "but it must still be reported"
    assert note["cells_written"] > 0


def test_traj_on_map_is_one_when_the_path_is_on_the_explored_area(tmp_path):
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    om = _om(explored=(150, 250, 150, 250))
    om._camera_positions = [np.array([-1.0, 1.0]) for _ in range(50)]

    save_obstacle_maps(tmp_path / "00800-TEEsavR23oF.json",
                       _SaveAgent([om], anchor), scene="00800-TEEsavR23oF")
    from navigation.mapping.map_store import floor_summaries, load_obstacle_maps

    summary = floor_summaries(load_obstacle_maps(tmp_path / "00800-TEEsavR23oF.json"))
    assert summary[0]["traj_poses"] == 50
    assert summary[0]["traj_on_map"] == 1.0


def test_an_old_snapshot_without_the_diagnostic_still_loads(tmp_path):
    import json

    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    om = _om(explored=(150, 250, 150, 250))
    p = save_obstacle_maps(tmp_path / "00800-TEEsavR23oF.json",
                           _SaveAgent([om], anchor), scene="00800-TEEsavR23oF")
    blob = json.loads(p.read_text())
    for f in blob["floors"]:
        f.pop("traj_on_map", None); f.pop("traj_poses", None)
    p.write_text(json.dumps(blob))

    agent = _LoadAgent({0: _Layer(0.0, Costmap2D(resolution=1.0 / PPM, size_m=20.0))})
    note = load_obstacle_map(_cfg(tmp_path), agent, "00800-TEEsavR23oF")
    assert note["traj_on_map"] == [None]


def test_the_bigger_map_is_kept(tmp_path):
    """Retention ranks on CELLS, storeys only as a tie-break."""
    from osg.eval.prior_map import save_obstacle_map_for_scene
    from navigation.mapping.map_store import floor_summaries, load_obstacle_maps

    class _Episode:
        episode_id = "0"
        scene_id = "data/scene_datasets/hm3d/val/00800-x/x.basis.glb"

    cfg = _cfg("")
    cfg.ycb.obstacle_map_out = str(tmp_path)
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)

    small = _om(explored=(150, 250, 150, 250))
    big = _om(explored=(100, 400, 100, 400))
    save_obstacle_map_for_scene(cfg, _SaveAgent([small], anchor), _Episode())
    save_obstacle_map_for_scene(cfg, _SaveAgent([big], anchor), _Episode())

    kept = floor_summaries(load_obstacle_maps(next(tmp_path.glob("*.json"))))
    assert kept[0]["explored_cells"] == 300 * 300


def test_a_policy_without_an_obstacle_map_writes_nothing(tmp_path):
    """`nav_agent` has no `_floors`; an empty snapshot loads silently."""
    from osg.eval.prior_map import save_obstacle_map_for_scene

    cfg = _cfg("")
    cfg.ycb.obstacle_map_out = str(tmp_path)

    class _NoFloors:
        pass

    class _Episode:
        episode_id = "0"
        scene_id = "data/scene_datasets/hm3d/val/00800-x/x.basis.glb"

    assert save_obstacle_map_for_scene(cfg, _NoFloors(), _Episode()) is None
    assert not list(tmp_path.glob("*.json"))


def test_a_wide_one_storey_map_beats_a_tiny_multi_storey_one(tmp_path):
    """The 00821 regression.

    An episode that ran 215 steps over a 3.4 x 1.2 m box while touching four
    storeys beat episodes covering 10 x 28 m, because storeys were ranked
    before cells -- the scene's stored map came out at 2975 cells against
    30-36k for its neighbours. A storey stepped onto is not a storey mapped.
    """
    from osg.eval.prior_map import save_obstacle_map_for_scene
    from navigation.mapping.map_store import floor_summaries, load_obstacle_maps

    class _Episode:
        episode_id = "0"
        scene_id = "data/scene_datasets/hm3d/val/00821-x/x.basis.glb"

    cfg = _cfg("")
    cfg.ycb.obstacle_map_out = str(tmp_path)
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)

    # Four storeys, a few cells each: the episode that used to win.
    tiny = [_om(explored=(150 + 10 * i, 155 + 10 * i, 150, 158)) for i in range(4)]
    # One storey, the whole building.
    wide = [_om(explored=(100, 400, 100, 400))]

    save_obstacle_map_for_scene(cfg, _SaveAgent(tiny, anchor), _Episode())
    save_obstacle_map_for_scene(cfg, _SaveAgent(wide, anchor), _Episode())

    kept = [f for f in floor_summaries(load_obstacle_maps(next(tmp_path.glob("*.json"))))
            if f["explored_cells"] > 0]
    assert sum(f["explored_cells"] for f in kept) == 300 * 300, (
        "the tiny four-storey map displaced the one that mapped the building")


# ------------------------------------------------------ the union of a scene

def _two_storey_agent(explored_lower, explored_upper):
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    return _SaveAgent([_om(explored=explored_lower), _om(explored=explored_upper)],
                      anchor)


def _save_episode(tmp_path, tag, agent, union=True):
    from osg.eval.prior_map import save_obstacle_map_for_scene

    class _Episode:
        episode_id = tag
        scene_id = "data/scene_datasets/hm3d/val/00800-TEEsavR23oF/x.basis.glb"

    cfg = _cfg("", union=union)
    cfg.ycb.obstacle_map_out = str(tmp_path)
    return save_obstacle_map_for_scene(cfg, agent, _Episode())


def test_with_the_union_on_every_episode_keeps_its_own_snapshot(tmp_path):
    """The protocol runs one mapping episode per authored object and the
    writer keeps only the best. Five of six explorations are thrown away, and
    on 00800 the one kept leaves the episode-1 target 5.1 m outside the map."""
    _save_episode(tmp_path, "ep_a", _two_storey_agent((100, 200, 100, 200), (150, 200, 150, 200)))
    _save_episode(tmp_path, "ep_b", _two_storey_agent((200, 300, 200, 300), (200, 260, 200, 260)))
    assert len(list(tmp_path.glob("*.json"))) == 2, "an episode's map was discarded"


def test_the_union_pastes_every_snapshot_into_one_costmap(tmp_path):
    """Two explorations that saw DIFFERENT rooms must add up."""
    _save_episode(tmp_path, "ep_a", _two_storey_agent((100, 200, 100, 200), (150, 200, 150, 200)))
    _save_episode(tmp_path, "ep_b", _two_storey_agent((260, 340, 260, 340), (200, 260, 200, 260)))

    lower = Costmap2D(resolution=1.0 / PPM, size_m=40.0)
    upper = Costmap2D(resolution=1.0 / PPM, size_m=40.0)
    agent = _LoadAgent({0: _Layer(0.0, lower), 1: _Layer(3.1, upper)})
    note = load_obstacle_map(_cfg(tmp_path, union=True), agent, "scene")

    assert len(note["union_snapshots"]) == 2, note
    assert not note["union_skipped"]
    per_snapshot = {}
    for m in note["matched_floors"]:
        per_snapshot[m["from"]] = per_snapshot.get(m["from"], 0) + m["cells"]
    assert len(per_snapshot) == 2 and all(v > 0 for v in per_snapshot.values())
    known = int((lower.grid != -1).sum())
    assert known > max(per_snapshot.values()) * 0.9, "the second map overwrote the first"
    assert known > min(per_snapshot.values()) * 1.5, f"the union added nothing: {known}"


def test_a_snapshot_with_a_different_storey_count_is_skipped_not_guessed(tmp_path):
    """`ObstacleMap` records no world height and floors are matched BY ORDER,
    so a one-storey snapshot pasted onto a two-storey stack would land a whole
    floor on the wrong storey."""
    _save_episode(tmp_path, "ep_a", _two_storey_agent((100, 200, 100, 200), (150, 200, 150, 200)))
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    _save_episode(tmp_path, "ep_one", _SaveAgent([_om(explored=(260, 340, 260, 340))], anchor))

    lower = Costmap2D(resolution=1.0 / PPM, size_m=40.0)
    upper = Costmap2D(resolution=1.0 / PPM, size_m=40.0)
    agent = _LoadAgent({0: _Layer(0.0, lower), 1: _Layer(3.1, upper)})
    note = load_obstacle_map(_cfg(tmp_path, union=True), agent, "scene")

    assert len(note["union_snapshots"]) == 1
    assert [s["mapped_floors"] for s in note["union_skipped"]] == [1]
    assert note["union_skipped"][0]["why"] == "order-mismatch"


def test_without_the_flag_nothing_changes(tmp_path):
    """The union is opt-in: the best-of file is still what pass 2 reads."""
    maps = _four_floors(tmp_path)
    costmap = Costmap2D(resolution=1.0 / PPM, size_m=20.0)
    agent = _LoadAgent({0: _Layer(0.0, costmap)})
    note = load_obstacle_map(_cfg(maps), agent, "00800-TEEsavR23oF")
    assert note["union_snapshots"] == []
    assert note["cells_written"] > 0


def test_a_one_storey_snapshot_is_usable_once_it_records_its_height(tmp_path):
    """The reason `floor_y` exists: four of the six mapping episodes on 00800
    mapped a single storey each, and order-matching cannot place them."""
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    upper_only = _SaveAgent([_om(explored=(260, 340, 260, 340))], anchor)
    upper_only._floors[0].update(standing_y_sum=3.1 * 20, standing_y_n=20)
    both = _two_storey_agent((100, 200, 100, 200), (150, 200, 150, 200))
    for floor, y in zip(both._floors, (0.0, 3.1)):
        floor.update(standing_y_sum=y * 20, standing_y_n=20)
    _save_episode(tmp_path, "ep_two", both)
    _save_episode(tmp_path, "ep_upper", upper_only)

    lower = Costmap2D(resolution=1.0 / PPM, size_m=40.0)
    upper = Costmap2D(resolution=1.0 / PPM, size_m=40.0)
    agent = _LoadAgent({0: _Layer(0.0, lower), 1: _Layer(3.1, upper)})
    note = load_obstacle_map(_cfg(tmp_path, union=True), agent, "scene")

    assert not note["union_skipped"], note["union_skipped"]
    assert len(note["union_snapshots"]) == 2
    assert note["matched_by"] == ["height"]
    # the one-storey snapshot went to the UPPER layer, which is the whole point
    from_upper = [m for m in note["matched_floors"] if "ep_upper" in m["from"]]
    assert len(from_upper) == 1 and from_upper[0]["osg_floor"] == 1


def test_a_floor_no_storey_matches_is_left_out(tmp_path):
    """A snapshot floor 8 m off every known storey is not forced onto one."""
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    stray = _SaveAgent([_om(explored=(260, 340, 260, 340))], anchor)
    stray._floors[0].update(standing_y_sum=9.0 * 20, standing_y_n=20)
    _save_episode(tmp_path, "ep_stray", stray)

    lower = Costmap2D(resolution=1.0 / PPM, size_m=40.0)
    agent = _LoadAgent({0: _Layer(0.0, lower)})
    note = load_obstacle_map(_cfg(tmp_path, union=True), agent, "scene")
    assert note["cells_written"] == 0
    assert note["union_skipped"] and note["union_skipped"][0]["why"] == "height"


# ------------------------------------------- storeys the scene graph missed

def test_a_storey_the_scene_graph_never_saw_is_seeded_from_the_snapshots():
    """Measured on 00808: NO single mapping episode visited both storeys, so
    the scene-graph prior's upper "floor" is the staircase itself at 1.03 m
    when the storeys are 0.06 and 2.86. Pass 2 takes its storey heights from
    that stack, so the upper floor's occupancy would land 1.8 m low."""
    from osg.eval.prior_map import _seed_storeys

    class _Stack:
        def __init__(self, heights):
            self._layers = {i: _Layer(h, None) for i, h in enumerate(heights)}

        def set_height(self, key, floor_y):
            self._layers.setdefault(key, _Layer(0.0, None)).floor_y = float(floor_y)

    cfg = _cfg("")
    cfg.ycb.seed_storeys_from_obstacle_map = True
    cfg.ycb.storey_seed_tol_m = 1.0

    stack = _Stack([0.06])
    added, loose = _seed_storeys(cfg, stack, [])   # nothing to read
    assert added == [] and loose == []

    # what the snapshots say, without needing files: patch the reader
    import osg.eval.prior_map as pm
    import navigation.mapping.map_store as ms
    real = ms.floor_summaries
    ms.floor_summaries = lambda blob: [
        {"explored_cells": 100, "floor_y": 0.06}, {"explored_cells": 100, "floor_y": 2.86}]
    real_load = ms.load_obstacle_maps
    ms.load_obstacle_maps = lambda path: {}
    try:
        added, loose = _seed_storeys(cfg, stack, ["one.json"])
    finally:
        ms.floor_summaries, ms.load_obstacle_maps = real, real_load

    assert added == [2.86], f"the missing storey was not added: {added}"
    assert sorted(round(l.floor_y, 2) for l in stack._layers.values()) == [0.06, 2.86]


def test_seeding_never_moves_a_storey_the_scene_graph_did_map():
    """A mapped storey carries rooms, containers and tracks built at ITS
    height; the snapshot's number must not overwrite that."""
    from osg.eval.prior_map import _seed_storeys
    import navigation.mapping.map_store as ms

    class _Stack:
        def __init__(self, heights):
            self._layers = {i: _Layer(h, None) for i, h in enumerate(heights)}

        def set_height(self, key, floor_y):
            self._layers.setdefault(key, _Layer(0.0, None)).floor_y = float(floor_y)

    cfg = _cfg("")
    cfg.ycb.seed_storeys_from_obstacle_map = True
    cfg.ycb.storey_seed_tol_m = 1.0
    stack = _Stack([0.16, 3.16])
    real, real_load = ms.floor_summaries, ms.load_obstacle_maps
    ms.floor_summaries = lambda blob: [
        {"explored_cells": 100, "floor_y": 0.06}, {"explored_cells": 100, "floor_y": 3.0}]
    ms.load_obstacle_maps = lambda path: {}
    try:
        assert _seed_storeys(cfg, stack, ["one.json"])[0] == []
    finally:
        ms.floor_summaries, ms.load_obstacle_maps = real, real_load
    assert sorted(round(l.floor_y, 2) for l in stack._layers.values()) == [0.16, 3.16]


def test_off_by_default():
    from osg.eval.prior_map import _seed_storeys
    cfg = _cfg("")
    assert _seed_storeys(cfg, None, ["one.json"]) == ([], [])


def test_one_storey_read_as_several_is_clustered_back_together():
    """ASCENT allocates a floor per staircase it notices, so one storey
    arrives as several: on 00808 the upper one appears at 2.86, 3.07, 3.11 and
    3.26 across eight snapshots, with a landing at 0.86 besides."""
    import navigation.mapping.map_store as ms
    from osg.eval.prior_map import storeys_from_snapshots

    real, real_load = ms.floor_summaries, ms.load_obstacle_maps
    ms.load_obstacle_maps = lambda path: {}
    ms.floor_summaries = lambda blob: [
        {"explored_cells": 17289, "floor_y": 0.06},
        {"explored_cells": 10261, "floor_y": 0.86},
        {"explored_cells": 15201, "floor_y": 3.07},
        {"explored_cells": 11617, "floor_y": 2.86},
        {"explored_cells": 9062, "floor_y": 3.26},
        {"explored_cells": 14909, "floor_y": 3.11},
    ]
    try:
        storeys = storeys_from_snapshots(["one.json"])
    finally:
        ms.floor_summaries, ms.load_obstacle_maps = real, real_load

    assert len(storeys) == 2, f"expected two storeys, got {storeys}"
    assert 0.3 < storeys[0] < 0.5, storeys          # 0.06 and 0.86, area-weighted
    assert 3.0 < storeys[1] < 3.15, storeys


def test_a_storey_nothing_backs_is_left_out_of_the_paste():
    """The scene graph's mid-staircase 'storey' must not receive a floor:
    with it in the set, the lower storey's ramp is written 0.06 -> 1.03 and
    the climb tops out a metre up (7 climbs on 00808, none over 0.40 m)."""
    import navigation.mapping.map_store as ms
    from osg.eval.prior_map import _seed_storeys

    class _Stack:
        def __init__(self, heights):
            self._layers = {i: _Layer(h, None) for i, h in enumerate(heights)}

        def set_height(self, key, floor_y):
            self._layers.setdefault(key, _Layer(0.0, None)).floor_y = float(floor_y)

    cfg = _cfg("")
    cfg.ycb.seed_storeys_from_obstacle_map = True
    cfg.ycb.storey_seed_tol_m = 0.75
    stack = _Stack([0.06, 1.03])                    # what 00808's scene graph kept
    real, real_load = ms.floor_summaries, ms.load_obstacle_maps
    ms.load_obstacle_maps = lambda path: {}
    ms.floor_summaries = lambda blob: [
        {"explored_cells": 25000, "floor_y": 0.06},
        {"explored_cells": 15000, "floor_y": 3.07},
    ]
    try:
        added, loose = _seed_storeys(cfg, stack, ["one.json"])
    finally:
        ms.floor_summaries, ms.load_obstacle_maps = real, real_load

    assert added and 3.0 < added[0] < 3.15, added
    assert loose == [1], f"the mid-staircase storey was kept: {loose}"
