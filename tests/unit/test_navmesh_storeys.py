"""Storey heights read off a navmesh, without a simulator.

The peak finder (`navmesh_floor_heights`) is right about how MANY storeys a
scene has and wrong about WHERE they are: `get_topdown_view` is asked for the
slice at `y` with a 0.5 m tolerance, so each peak lands ~0.35 m above the floor
it names. Measured on 00800 the peaks are 0.513/3.513 where the authored floors
are 0.163/3.163 -- a third of a metre, which is more than anything downstream
matches storeys with.

`navmesh_storeys` therefore uses the peaks only to bucket an area-uniform point
sample and takes the median of each bucket. These tests plant a known floor and
assert the median recovers it where the peak does not.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.eval.floors import navmesh_path_for, navmesh_storeys

TRUE_HEIGHTS = (0.16, 3.16)
BIAS = 0.35  # what get_topdown_view's tolerance adds to every peak


class _StubPathFinder:
    """Enough of `habitat_sim.nav.PathFinder` for the height maths.

    `get_topdown_view` returns area that peaks BIAS above each true floor, which
    is the defect being corrected; `get_random_navigable_point` samples the
    floors themselves plus a staircase joining them.
    """

    def __init__(self, heights=TRUE_HEIGHTS, shares=(0.7, 0.3), stair_frac=0.02,
                 area=100.0, seed=0):
        self.heights = list(heights)
        self.shares = list(shares)
        self.stair_frac = stair_frac
        self.navigable_area = area
        self._rng = np.random.default_rng(seed)
        self.num_islands = 1

    def seed(self, value):
        self._rng = np.random.default_rng(int(value) % (2 ** 32))

    def build_navmesh_vertices(self):
        lo, hi = min(self.heights) - 0.2, max(self.heights) + 0.6
        return np.array([[0.0, lo, 0.0], [1.0, hi, 1.0]], dtype=np.float32)

    def get_topdown_view(self, eps, y):
        total = 0.0
        for height, share in zip(self.heights, self.shares):
            # a triangular bump centred BIAS above the floor
            total += share * max(0.0, 1.0 - abs(y - (height + BIAS)) / 0.25)
        return np.full((max(1, int(total * 1000)), 1), True)

    def get_random_navigable_point(self):
        u = self._rng.random()
        if u < self.stair_frac:  # on the staircase, between storeys
            lo, hi = self.heights[0], self.heights[-1]
            y = lo + self._rng.random() * (hi - lo)
        else:
            idx = int(self._rng.choice(len(self.heights), p=np.array(self.shares)
                                       / sum(self.shares)))
            y = self.heights[idx] + self._rng.normal(0.0, 0.01)
        return np.array([self._rng.normal(), y, self._rng.normal()],
                        dtype=np.float32)


def test_the_median_recovers_the_floor_the_peak_misses():
    pf = _StubPathFinder()
    result = navmesh_storeys(pf, n_points=6000, seed=1)

    assert len(result["storeys"]) == 2
    for storey, truth in zip(result["storeys"], TRUE_HEIGHTS):
        assert storey["height"] == pytest.approx(truth, abs=0.02), (
            "the median must land on the floor itself")
        assert storey["peak_height"] > truth + 0.2, (
            "the peak is the biased number this exists to correct")


def test_area_is_apportioned_by_the_point_count():
    pf = _StubPathFinder(shares=(0.7, 0.3), area=100.0, stair_frac=0.0)
    result = navmesh_storeys(pf, n_points=8000, seed=2)
    areas = [s["area_m2"] for s in result["storeys"]]

    assert sum(areas) == pytest.approx(100.0, abs=1.0)
    assert areas[0] / sum(areas) == pytest.approx(0.7, abs=0.03)


def test_staircase_points_belong_to_no_storey():
    """A tread is not floor. Counting it on either storey would inflate that
    storey's denominator with area the map is never expected to hold."""
    pf = _StubPathFinder(stair_frac=0.10)
    result = navmesh_storeys(pf, n_points=8000, seed=3)

    assert result["off_band_frac"] > 0.03
    counted = sum(s["n_points"] for s in result["storeys"])
    assert counted < 8000
    assert counted + round(result["off_band_frac"] * 8000) == pytest.approx(8000, abs=2)


def test_a_flat_scene_is_one_storey():
    pf = _StubPathFinder(heights=(0.2,), shares=(1.0,), stair_frac=0.0)
    result = navmesh_storeys(pf, n_points=3000, seed=4)

    assert len(result["storeys"]) == 1
    assert result["storeys"][0]["height"] == pytest.approx(0.2, abs=0.02)
    assert result["off_band_frac"] == pytest.approx(0.0, abs=1e-6)


def test_the_sample_is_reproducible_from_its_seed():
    a = navmesh_storeys(_StubPathFinder(), n_points=2000, seed=7)
    b = navmesh_storeys(_StubPathFinder(), n_points=2000, seed=7)
    assert [s["height"] for s in a["storeys"]] == [s["height"] for s in b["storeys"]]


def test_an_empty_navmesh_reports_no_storey_rather_than_raising():
    class _Empty(_StubPathFinder):
        def build_navmesh_vertices(self):
            return np.zeros((0, 3), dtype=np.float32)

    assert navmesh_storeys(_Empty())["storeys"] == []


def test_navmesh_path_for_finds_the_mesh_under_a_versioned_tree(tmp_path):
    scene = "00800-TEEsavR23oF"
    mesh = tmp_path / "hm3d" / "val" / scene / "TEEsavR23oF.basis.navmesh"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"")

    assert navmesh_path_for(scene, tmp_path) == mesh
    assert navmesh_path_for("00000-nope", tmp_path) is None
