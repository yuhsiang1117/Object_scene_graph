"""Does moving a YCB object invalidate a prior? The decision rules, pinned.

The probe exists so that a layout edit does not cost a 14-minute mapping
episode per scene to price. Its answer rests on three geometric facts, and the
ones that are rules rather than measurements are asserted here.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN, Costmap2D

_spec = importlib.util.spec_from_file_location(
    "probe_remap_need",
    Path(__file__).resolve().parents[2] / "scripts" / "probe_remap_need.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

OBJECTS = Path("/habitat-data-collector/data/objects/ycb/configs")
_HAVE_YCB = OBJECTS.is_dir()


# ------------------------------------------------------------- the band test

def test_the_band_is_measured_from_the_storey_the_agent_stands_on():
    """ASCENT's 0.61-0.88 m is episodic z -- height above the agent's FEET --
    because `tf_camera_to_episodic` puts the camera at a constant
    `agent.camera_height`. Read as an absolute height it would put the whole
    upper storey of every scene outside the band."""
    # a can on a 0.75 m counter, on a storey whose floor is at 3.16 m
    assert probe.band_overlap_m(3.91, 4.01, 3.16, 0.61, 0.88) > 0
    # the same can read as an absolute height: nothing
    assert probe.band_overlap_m(3.91, 4.01, 0.0, 0.61, 0.88) == 0.0


def test_an_object_below_or_above_the_band_cannot_change_the_map():
    assert probe.band_overlap_m(0.20, 0.30, 0.0, 0.61, 0.88) == 0.0   # on the floor
    assert probe.band_overlap_m(1.20, 1.48, 0.0, 0.61, 0.88) == 0.0   # high shelf
    assert probe.band_overlap_m(0.70, 0.80, 0.0, 0.61, 0.88) == pytest.approx(0.10)


def test_a_tall_object_is_clipped_to_the_band_not_to_itself():
    """The overlap is what ASCENT could stamp, so it can never exceed the
    band's own 0.27 m."""
    assert probe.band_overlap_m(-5.0, 5.0, 0.0, 0.61, 0.88) == pytest.approx(0.27)


# -------------------------------------------------------------------- the ring

def _costmap(fill=UNKNOWN):
    costmap = Costmap2D(resolution=0.05, size_m=20.0)
    costmap.grid[:, :] = fill
    return costmap


def test_a_ring_of_furniture_explains_the_cell_under_the_object():
    """A YCB object is under 0.2 m across -- four cells -- while the table
    holding it is metres wide. A mostly-OCCUPIED annulus says the SUPPORT fills
    these cells, which the centre cell alone cannot say: in the static map the
    object's own stamp is already sitting there."""
    costmap = _costmap(FREE)
    centre = np.array([0.0, 0.0])
    rc = costmap.world_to_grid(centre)
    costmap.grid[rc[0] - 12:rc[0] + 13, rc[1] - 12:rc[1] + 13] = OCCUPIED

    assert probe.ring_occupied_frac(costmap, centre, 0.10, 0.30) > 0.9


def test_an_object_standing_alone_has_a_free_ring():
    """And the inner radius has to CLEAR the object, or the ring reads the
    object as its own support. At 0.10 m this exact case measured 0.22, which
    is why the default inner radius is 0.15 m: the widest YCB target here is
    0.164 m across, so its stamp reaches two cells at 0.05 m resolution."""
    costmap = _costmap(FREE)
    centre = np.array([0.0, 0.0])
    rc = costmap.world_to_grid(centre)
    costmap.grid[rc[0] - 2:rc[0] + 3, rc[1] - 2:rc[1] + 3] = OCCUPIED  # just it

    assert probe.ring_occupied_frac(costmap, centre, 0.15, 0.35) == pytest.approx(0.0)
    assert probe.ring_occupied_frac(costmap, centre, 0.10, 0.30) > 0.1


def test_unknown_cells_do_not_count_either_way():
    """An unmapped ring is no evidence about the furniture, and must not be
    read as evidence of its absence."""
    costmap = _costmap(UNKNOWN)
    assert probe.ring_occupied_frac(costmap, np.array([0.0, 0.0]), 0.10, 0.30) == 0.0


# ------------------------------------------------------------- object geometry

@pytest.mark.skipif(not _HAVE_YCB, reason="YCB object configs not mounted")
@pytest.mark.parametrize("handle,published", [
    ("003_cracker_box", 0.213),
    ("005_tomato_soup_can", 0.101),
    ("024_bowl", 0.055),
    ("011_banana", 0.036),
    ("002_master_chef_can", 0.139),
])
def test_the_aabb_matches_the_published_ycb_dimension(handle, published):
    """The asset's `(up, front)` rotation is load-bearing, not decorative.
    Without it the bowl measures 0.161 m instead of 0.055 and the banana 0.178
    instead of 0.036 -- either of which would put a flat object across ASCENT's
    whole 0.27 m band and turn every verdict into a false REBUILD."""
    lo, hi = probe.world_aabb([0, 0, 0], [0, 0, 0, 1], handle, OBJECTS)
    assert float(hi[1] - lo[1]) == pytest.approx(published, abs=0.008)


@pytest.mark.skipif(not _HAVE_YCB, reason="YCB object configs not mounted")
def test_the_base_is_snapped_to_the_authoring_top_y_when_it_is_recorded():
    """The authoring placed the object's base on that surface, so its number is
    exact by construction; the mesh AABB sits 0.00-0.18 m above it. Taking the
    base from the layout and only the HEIGHT from the mesh keeps the band test
    on the surface the object really rests on."""
    obj = {"translation": [0.0, 5.0, 0.0], "rotation": [0, 0, 0, 1],
           "anchor": {"top_y": 0.90}}
    y_lo, y_hi, source = probe.object_band(obj, "003_cracker_box", OBJECTS,
                                           pad_m=0.0)
    assert source == "top_y"
    assert y_lo == pytest.approx(0.90)
    assert y_hi - y_lo == pytest.approx(0.213, abs=0.008)


@pytest.mark.skipif(not _HAVE_YCB, reason="YCB object configs not mounted")
def test_without_top_y_it_falls_back_to_the_raw_aabb_and_says_so():
    """The released DualMap layouts carry no anchor block, so a verdict taken
    on them is visibly less exact."""
    obj = {"translation": [0.0, 1.0, 0.0], "rotation": [0, 0, 0, 1]}
    _, _, source = probe.object_band(obj, "003_cracker_box", OBJECTS, pad_m=0.0)
    assert source == "aabb"


def test_the_glb_reader_rejects_a_file_that_is_not_one(tmp_path):
    bogus = tmp_path / "x.glb"
    bogus.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(ValueError):
        probe.glb_bounds(bogus)


# -------------------------------------------------------------- track lookup

def test_a_track_is_found_by_its_own_centre_not_by_its_floor_key():
    """Measured on 00808: the prior's one YCB target track is filed under a
    phantom 1.03 m storey while its centre sits at y=0.56. Bucketing by
    `floor_key` would have reported the object as absent from its own storey."""
    tracks = [
        {"label": "cracker box", "center": [1.0, 0.56, 2.0], "floor_key": 1},
        {"label": "cracker box", "center": [9.0, 0.56, 9.0], "floor_key": 0},
        {"label": "bowl", "center": [1.0, 0.56, 2.0], "floor_key": 0},
    ]
    track, dist = probe.nearest_track(tracks, "cracker box", (1.2, 2.1), 0.56,
                                      [0.06, 3.0], 1.5)
    assert track is not None and dist < 0.5
    assert track["floor_key"] == 1, "found by geometry, despite the wrong key"


def test_a_label_with_no_track_reports_absent():
    track, dist = probe.nearest_track([], "cracker box", (0.0, 0.0), 0.0,
                                      [0.0], 1.5)
    assert track is None and dist is None


def test_a_ring_radius_is_rounded_to_cells_not_truncated():
    """0.15 / 0.05 is 2.9999... in binary floating point. Truncating gave a
    two-cell inner radius for a three-cell request, which put the ring back on
    top of the object it exists to exclude."""
    costmap = _costmap(FREE)
    centre = np.array([0.0, 0.0])
    rc = costmap.world_to_grid(centre)
    costmap.grid[rc[0] - 2:rc[0] + 3, rc[1] - 2:rc[1] + 3] = OCCUPIED

    assert probe.ring_occupied_frac(costmap, centre, 0.15, 0.35) == 0.0
