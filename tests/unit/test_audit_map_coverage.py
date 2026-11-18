"""The coverage audit's decision rules, without a simulator.

Everything geometric in the audit goes through the production paste
(`osg.eval.prior_map.paste_scene`, pinned by test_paste_snapshots.py) and the
navmesh (pinned by test_navmesh_storeys.py). What is left, and what this file
covers, is the judgement: which storey an object stands on, what counts as
mapped, how far a target is from free space, and which thresholds trip.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN, Costmap2D

_spec = importlib.util.spec_from_file_location(
    "audit_map_coverage",
    Path(__file__).resolve().parents[2] / "scripts" / "audit_map_coverage.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


# --------------------------------------------------------------- storey_of

def test_a_target_on_a_counter_belongs_to_the_storey_it_stands_on():
    """The rule is the HIGHEST storey at or below the base, not the nearest.

    00878's storeys are 2.85 m apart and a kitchen counter is 0.92 m high, so a
    bottle on it is 1.93 m below the storey above and 0.92 above its own --
    nearest-height is right here. Make the gap smaller and it stops being right.
    """
    heights = [0.0, 2.2]          # 2.2 m apart; a 1.2 m counter is NEARER the top
    assert audit.storey_of(1.2, heights) == 0
    assert audit.storey_of(0.05, heights) == 0
    assert audit.storey_of(2.3, heights) == 1
    assert audit.storey_of(3.4, heights) == 1


def test_a_placement_a_little_below_its_own_floor_stays_on_it():
    assert audit.storey_of(-0.10, [0.0, 3.0], eps_m=0.15) == 0


def test_below_every_storey_falls_back_to_the_nearest():
    assert audit.storey_of(-5.0, [0.0, 3.0]) == 0


# ---------------------------------------------------------------- coverage

def _costmap(fill=None):
    costmap = Costmap2D(resolution=0.1, size_m=10.0)
    if fill is not None:
        costmap.grid[:, :] = fill
    return costmap


def test_coverage_separates_free_occupied_and_unknown():
    costmap = _costmap(UNKNOWN)
    costmap.grid[50:60, 50:60] = FREE
    costmap.grid[60:65, 50:60] = OCCUPIED
    free_pt = costmap.grid_to_world(np.array([55, 55]))
    occ_pt = costmap.grid_to_world(np.array([62, 55]))
    unk_pt = costmap.grid_to_world(np.array([10, 10]))

    out = audit.coverage(costmap, np.array([free_pt, free_pt, occ_pt, unk_pt]))

    assert out["n"] == 4
    assert out["free"] == pytest.approx(0.5)
    assert out["occupied"] == pytest.approx(0.25)
    assert out["unknown"] == pytest.approx(0.25)
    assert out["oob"] == 0.0


def test_points_outside_the_grid_are_counted_as_oob_not_unknown():
    """`apply_to_costmap` grows the destination, so oob should be 0 on a real
    map; a non-zero reading means the paste never reached that region and must
    not hide inside `unknown`."""
    costmap = _costmap(FREE)
    out = audit.coverage(costmap, np.array([[500.0, 500.0], [0.0, 0.0]]))
    assert out["oob"] == pytest.approx(0.5)
    assert out["free"] == pytest.approx(0.5)


def test_an_empty_sample_is_reported_rather_than_dividing_by_zero():
    assert audit.coverage(None, np.zeros((0, 2)))["n"] == 0


# ------------------------------------------------------------ free distance

def test_distance_to_free_is_real_for_a_target_far_outside_the_map():
    """The reason this is a distance transform and not `nearest_free_xy`:
    that helper searches a 1.5 m box and returns the query point UNCHANGED on
    failure, reporting 0.00 m for a target metres outside the map -- the exact
    reading the audit exists to make impossible."""
    costmap = _costmap(UNKNOWN)
    costmap.grid[50:55, 50:55] = FREE
    distances = audit.free_distance_m(costmap)

    near = costmap.world_to_grid(costmap.grid_to_world(np.array([52, 52])))
    far = costmap.world_to_grid(costmap.grid_to_world(np.array([90, 90])))
    assert distances[near[0], near[1]] == pytest.approx(0.0)
    assert distances[far[0], far[1]] > 3.0

    from osg.mapping.costmap import nearest_free_xy
    far_xy = costmap.grid_to_world(np.array([90, 90]))
    assert np.allclose(nearest_free_xy(costmap, far_xy), far_xy), (
        "if this ever stops being true, say so -- the docstring above is why "
        "the audit does not use it")


def test_a_map_with_no_free_cell_reports_infinity_not_zero():
    distances = audit.free_distance_m(_costmap(UNKNOWN))
    assert np.isinf(distances).all()


# ----------------------------------------------------------------- stairs

def test_the_ramp_span_is_reported_as_a_fraction_of_the_storey_gap():
    """A flight cut short is the measured failure this catches: a ramp that
    spanned 1.6 m of a 3.0 m gap left the climber with no waypoint past that
    point and it drove into the banister for 120 steps."""
    costmap = _costmap(UNKNOWN)
    costmap.stair_mask = np.zeros(costmap.grid.shape, dtype=bool)
    costmap.stair_mask[10:20, 10] = True
    costmap.height = np.full(costmap.grid.shape, np.nan, dtype=np.float32)
    costmap.height[10:20, 10] = np.linspace(0.0, 1.6, 10)

    report = audit.stair_report(costmap, storey_gap_m=3.0)

    assert report["cells"] == 10
    assert report["ramp_span_m"] == pytest.approx(1.6, abs=1e-3)
    assert report["ramp_span_frac"] == pytest.approx(1.6 / 3.0, abs=1e-3)


def test_no_stair_mask_is_zero_cells_and_no_span():
    report = audit.stair_report(_costmap(UNKNOWN), storey_gap_m=3.0)
    assert report["cells"] == 0 and report["ramp_span_m"] is None


# ------------------------------------------------------------------- gates

def _args(**over):
    base = dict(min_free_frac=0.60, max_occupied_frac=0.08,
                max_target_free_m=1.5, min_stair_cells=50,
                min_ramp_span_frac=0.60, require_graph_floor=True,
                storey_match_m=1.0, exempt_storeys={}, max_target_track_m=1.0)
    base.update(over)
    return SimpleNamespace(**base)


def _row(free=0.8, occupied=0.0, stair_cells=900, ramp_frac=1.0,
         graph_key=0, n_storeys=2, targets=()):
    return {
        "scene": "00800-TEEsavR23oF",
        "failures": [],
        "navmesh": {"storeys": [{}] * n_storeys},
        "graph": {"present": True},
        "storeys": [{
            "index": 0, "paste_storey": 0,
            "navigable": {"n": 100, "free": free, "occupied": occupied,
                          "unknown": 1 - free - occupied, "oob": 0.0},
            "stairs": {"cells": stair_cells, "ramp_span_m": 3.0,
                       "storey_gap_m": 3.0, "ramp_span_frac": ramp_frac},
            "graph": {"floor_key": graph_key},
        }],
        "targets": list(targets),
    }


def test_a_good_scene_trips_nothing():
    assert audit.gate(_row(), _args()) == []


def test_low_coverage_trips_the_free_gate():
    assert any("free" in f for f in audit.gate(_row(free=0.4), _args()))


def test_navigable_points_on_obstacles_are_reported_but_never_gated():
    """No threshold on `occupied` can do the job it was added for, and two
    were tried before that was understood.

    The transposed paste the log records read FREE 0.04 / OCC 0.04 -- its
    occupied fraction was LOWER than a healthy furnished storey's, because a
    map rotated away from the world stamps almost nothing where the floor
    actually is. So a cutoff loose enough to pass a correct map (0.040 at 500
    steps, 0.137 at 1500 on 00800's lower storey) cannot possibly fire at 0.04.
    What collapses under a transposition is FREE, and `min_free_frac` catches
    that. The real frame test is scripts/check_obstacle_map_reuse.py.
    """
    # a correct but heavily furnished storey: high occupied, healthy free
    assert audit.gate(_row(free=0.750, occupied=0.137), _args()) == []
    assert audit.gate(_row(free=0.876, occupied=0.040), _args()) == []
    # the transposed map: occupied is LOW, and free is what gives it away
    failures = audit.gate(_row(free=0.04, occupied=0.04), _args())
    assert any("free" in f for f in failures)
    assert not any("OBSTACLES" in f for f in failures)


def test_a_half_length_stair_ramp_trips_the_span_gate():
    failures = audit.gate(_row(ramp_frac=0.53), _args())
    assert any("ramp" in f for f in failures)


def test_a_storey_with_no_layer_and_nothing_to_seed_one_from_trips():
    row = _row(graph_key=None)
    row["union_heights"] = []
    assert any("scene-graph floor" in f for f in audit.gate(row, _args()))


def test_a_storey_the_obstacle_map_can_seed_a_layer_for_does_not_trip():
    """Pass 2 runs with `ycb.seed_storeys_from_obstacle_map`, and
    `_seed_storeys` adds a layer for any snapshot cluster the graph lacks. On
    00808 the graph's own floors are 1.03 and -0.43 -- neither is a real storey
    -- while the union clusters at 0.09 and 3.04 seed both correctly, so
    checking the graph as STORED asks a question pass 2 does not."""
    row = _row(graph_key=None)
    row["storeys"][0]["height"] = 3.26
    row["union_heights"] = [0.09, 3.04]
    assert audit.gate(row, _args()) == []


def test_a_single_storey_scene_is_not_asked_for_stairs():
    assert audit.gate(_row(stair_cells=0, ramp_frac=None, n_storeys=1),
                      _args()) == []


def test_a_target_outside_the_map_trips_and_names_the_worst():
    targets = [
        {"layout_id": "static", "handle": "003_cracker_box", "storey": 0,
         "dist_to_free_m": 0.2, "gate_ok": True},
        {"layout_id": "static", "handle": "072-a_toy_airplane", "storey": 0,
         "dist_to_free_m": 5.1, "gate_ok": False},
    ]
    failures = audit.gate(_row(targets=targets), _args())
    assert any("072-a_toy_airplane" in f for f in failures)


def test_an_exempt_storey_is_skipped_but_had_to_be_declared():
    """00808's basement holds no authored target in either layout. Skipping it
    is legitimate; skipping it SILENTLY is not, which is why the exemption is a
    CLI flag echoed in the header rather than a rule inside the code."""
    args = _args(exempt_storeys={"00800-TEEsavR23oF": {0}})
    assert audit.gate(_row(free=0.0, occupied=0.9), args) == []


# --------------------------------------------------------------- layout read

def test_layouts_are_read_raw_so_the_anchor_fields_survive(tmp_path):
    """`AuthoredObject` drops `anchor.floor_index`, `floor_height` and `top_y`,
    and the audit cross-checks its own storey rule against the first two."""
    scene = tmp_path / "00800-TEEsavR23oF"
    (scene / "dynamic_scene_config" / "cross_anchor").mkdir(parents=True)
    blob = {
        "id_handle_mapping": {"50001": "003_cracker_box"},
        "objects": [{"semantic_id": 50001, "translation": [1.0, 2.0, 3.0],
                     "rotation": [0, 0, 0, 1],
                     "anchor": {"object_id": "table_1", "floor_index": 0,
                                "floor_height": 0.16, "top_y": 0.9}}],
    }
    (scene / "static_scene_config.json").write_text(json.dumps(blob))
    (scene / "dynamic_scene_config" / "cross_anchor" / "layout_01.json"
     ).write_text(json.dumps(blob))

    out = audit.read_layouts([tmp_path], "00800-TEEsavR23oF",
                             ["static", "cross_anchor"], [1])

    assert [layout["_layout_id"] for layout in out] == ["static",
                                                        "cross_anchor_01"]
    anchor = out[0]["objects"][0]["anchor"]
    assert anchor["floor_index"] == 0 and anchor["top_y"] == 0.9


def test_a_scene_missing_from_a_root_is_not_an_error(tmp_path):
    assert audit.read_layouts([tmp_path], "00000-nope", ["static"], [1]) == []


def test_a_storey_the_agent_cannot_walk_to_is_out_of_scope_without_declaring_it():
    """00808's basement is 44.9 m2 of navigable area with 100% of it on an
    island disconnected from the start island. No mapping episode can reach it
    and no benchmark episode is scored on it, so gating on its coverage asks
    for something physically impossible -- and `--exempt-storey` did not cover
    it, because the "no pasted storey" failure is raised structurally, before
    the per-storey thresholds the flag skipped.
    """
    row = _row()
    row["storeys"][0]["unreachable"] = True
    row["storeys"][0]["navigable"] = {"n": 100, "free": 0.0, "occupied": 0.0,
                                      "unknown": 1.0, "oob": 0.0}
    assert audit.gate(row, _args()) == []


def test_a_reachable_storey_with_no_coverage_still_fails():
    """The escape hatch must not swallow the case it looks like."""
    row = _row(free=0.0)
    row["storeys"][0]["unreachable"] = False
    assert any("free" in f for f in audit.gate(row, _args()))


def _target(handle, in_prior, storey=0, d=0.3):
    return {"layout_id": "static", "handle": handle, "label": handle, "storey": storey,
            "gate_ok": True, "position": [0.0, 0.9, 0.0],
            "in_prior": in_prior, "prior_track_m": d if in_prior else None}


def test_a_static_target_missing_from_the_graph_trips_the_gate():
    """0 of 25 multi-floor search targets were in outputs/maps_p1500_osg at
    their static pose, and nothing failed: pass 2 had no stale belief to revise
    and the benchmark never exercised presence revision."""
    row = _row(targets=[_target("003_cracker_box", True), _target("024_bowl", False)])
    failures = audit.gate(row, _args())
    assert any("024_bowl" in f and "scene-graph" in f for f in failures)
    assert not any("003_cracker_box" in f for f in failures)


def test_the_graph_gate_can_be_switched_off_and_respects_exemptions():
    row = _row(targets=[_target("024_bowl", False, storey=0)])
    assert audit.gate(row, _args(max_target_track_m=0.0)) == []
    exempt = {"00800-TEEsavR23oF": (0,)}
    assert audit.gate(row, _args(exempt_storeys=exempt)) == []


def test_nearest_same_label_track_is_horizontal_and_same_storey():
    item = {"label": "bowl", "position": [0.0, 0.9, 0.0]}
    tracks = [
        {"label": "bowl", "center": [0.6, 0.85, 0.8]},     # 1.0 m away, same storey
        {"label": "bowl", "center": [0.1, 3.9, 0.0]},      # right above, a storey up
        {"label": "plate", "center": [0.0, 0.9, 0.0]},     # right there, wrong label
    ]
    assert abs(audit.nearest_same_label_m(item, tracks) - 1.0) < 1e-9
    assert audit.nearest_same_label_m(item, tracks[1:]) is None
