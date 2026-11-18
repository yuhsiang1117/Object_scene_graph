"""The offline paste and the in-run paste must write the same cells.

`load_obstacle_map` is what a run uses; `paste_scene` is what the coverage
audit, the map figure and the frame check use. They were separate code once,
and the separate one was WRONG: `navigation/mapping/map_store` writes its grids
transposed relative to its own points, so a second implementation put the walls
8-9 m from the truth and every geometric conclusion drawn before that was found
had to be retracted (docs/history/CROSS_ANCHOR_OBSTACLE_MAP.md).

So the offline path is now the in-run path with the costmaps passed in, and
this file is the lock: paste the same snapshot both ways, assert the grids are
identical cell for cell.
"""
from __future__ import annotations

import numpy as np

from navigation.geometry import EpisodeAnchor
from navigation.mapping.map_store import save_obstacle_maps
from navigation.mapping.obstacle_map import ObstacleMap
from osg.eval.prior_map import load_obstacle_map, paste_scene
from osg.mapping.costmap import Costmap2D

SIZE, PPM = 400, 20
SCENE = "00800-TEEsavR23oF"


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


def _cfg(map_in, union=False):
    cfg = _Cfg()
    cfg.ycb.obstacle_map_in = str(map_in)
    cfg.ycb.obstacle_map_out = ""
    cfg.ycb.obstacle_map_overwrite = False
    cfg.ycb.obstacle_map_union = union
    cfg.ycb.seed_storeys_from_obstacle_map = False
    cfg.ycb.storey_seed_tol_m = 1.0
    return cfg


def _write(tmp_path, name, floor_ys, *, episode=""):
    """A two-storey snapshot whose floors carry `floor_y`, as a real pass does."""
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    floors = [
        _om(explored=(150, 250, 150, 250), occupied=(160, 175, 160, 175)),
        _om(explored=(180, 240, 180, 240), occupied=(190, 200, 190, 200)),
    ]
    agent = _SaveAgent(floors, anchor, floor_idx=0)
    for record, y in zip(agent._floors, floor_ys):
        record["standing_y_sum"] = float(y)
        record["standing_y_n"] = 1
    save_obstacle_maps(tmp_path / name, agent, scene=SCENE,
                       layout_id="static", episode_id=episode)
    return tmp_path


HEIGHTS = [0.16, 3.16]


def _grids(costmaps):
    return [np.array(c.grid, copy=True) for c in costmaps]


def test_offline_paste_writes_the_same_cells_as_the_run(tmp_path):
    _write(tmp_path, f"{SCENE}.json", HEIGHTS)

    run_costmaps = [Costmap2D(resolution=1.0 / PPM, size_m=20.0) for _ in HEIGHTS]
    for costmap in run_costmaps:
        costmap.height = np.full(costmap.grid.shape, np.nan, dtype=np.float32)
    agent = _LoadAgent({i: _Layer(h, c)
                        for i, (h, c) in enumerate(zip(HEIGHTS, run_costmaps))})
    note = load_obstacle_map(_cfg(tmp_path), agent, SCENE)

    offline = paste_scene(str(tmp_path), SCENE, HEIGHTS, union=False,
                          size_m=20.0)

    assert note["cells_written"] > 0
    assert offline["cells_written"] == note["cells_written"]
    for run_grid, off in zip(_grids(run_costmaps), offline["costmaps"]):
        assert np.array_equal(run_grid, off.grid)


def test_the_stair_ramp_survives_the_offline_paste(tmp_path):
    """`_ramp_stair_heights` is a no-op without a height plane, and without the
    ramp `find_flights` sees no staircase at all."""
    _write(tmp_path, f"{SCENE}.json", HEIGHTS)
    offline = paste_scene(str(tmp_path), SCENE, HEIGHTS, union=False, size_m=20.0)
    for costmap in offline["costmaps"]:
        assert costmap.height is not None


def test_a_union_of_snapshots_is_pasted_and_reported(tmp_path):
    _write(tmp_path, f"{SCENE}__ep1.json", HEIGHTS, episode="ep1")
    _write(tmp_path, f"{SCENE}__ep2.json", HEIGHTS, episode="ep2")

    offline = paste_scene(str(tmp_path), SCENE, HEIGHTS, union=True, size_m=20.0)

    assert len(offline["paths"]) == 2
    assert len(offline["used"]) == 2
    assert offline["skipped"] == []
    assert {s["episode_id"] for s in offline["snapshots"]} == {"ep1", "ep2"}
    assert all(p["matched_by"] == "height" for p in offline["pasted"])


def test_a_missing_scene_returns_empty_rather_than_raising(tmp_path):
    """The audit runs against directories that may not hold every scene; a
    missing snapshot is a reportable fact, not an exception."""
    offline = paste_scene(str(tmp_path), SCENE, HEIGHTS, union=True)
    assert offline["paths"] == []
    assert offline["costmaps"] == []
    assert offline["cells_written"] == 0


def test_resolution_comes_from_the_snapshot(tmp_path):
    _write(tmp_path, f"{SCENE}.json", HEIGHTS)
    offline = paste_scene(str(tmp_path), SCENE, HEIGHTS, union=False, size_m=20.0)
    assert offline["resolution"] == 1.0 / PPM
    assert all(c.resolution == 1.0 / PPM for c in offline["costmaps"])
