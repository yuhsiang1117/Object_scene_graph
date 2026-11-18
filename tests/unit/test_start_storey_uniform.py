"""Benchmark v2 starts (sim/ycb_env.py `_starts_on_uniform_storeys`): the storey
is drawn uniformly, then a point on it, never on the stairs, never too close.
Habitat-free: the pathfinder and the geodesic are stand-ins."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import osg.sim.ycb_env as ycb_env
from osg.sim.ycb_layouts import YCBLayoutError


class _Pathfinder:
    """Area-uneven on purpose: 90% of points on the ground storey, 5% on a
    stair tread between storeys, 5% upstairs."""

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def seed(self, value):
        self.rng = np.random.default_rng(int(value))

    def get_random_navigable_point(self):
        u = self.rng.random()
        y = 0.1 if u < 0.90 else (1.6 if u < 0.95 else 3.1)
        return np.array([self.rng.uniform(-10, 10), y, self.rng.uniform(-10, 10)],
                        dtype=np.float32)


def _cfg(starts=200, min_geo=3.0):
    return SimpleNamespace(ycb=SimpleNamespace(
        seed=42, starts_per_target=starts, start_sample_attempts=2000,
        start_min_geodesic_m=min_geo, relocation_floor_tolerance_m=0.5,
    ))


@pytest.fixture
def straight_line(monkeypatch):
    def geodesic(pathfinder, start, goals):
        return min(float(np.linalg.norm(np.asarray(start) - np.asarray(g))) for g in goals)
    monkeypatch.setattr(ycb_env, "_geodesic_distance", geodesic)


def test_storeys_are_drawn_uniformly_not_by_area(straight_line):
    authored = SimpleNamespace(semantic_id=50001, handle="003_cracker_box")
    starts = ycb_env._starts_for_object(
        _Pathfinder(), [{"position": [0.0, 0.1, 0.0]}], authored, _cfg(),
        storey_heights=[0.1, 3.1],
    )
    upstairs = sum(1 for s in starts if abs(s["position"][1] - 3.1) < 0.5)
    assert 70 <= upstairs <= 130            # ~100 of 200, not the 5% area share
    assert all(abs(s["position"][1] - 1.6) > 0.5 for s in starts)    # no treads
    assert all(s["initial_geodesic_distance"] >= 3.0 for s in starts)
    assert {s["start_storey_y"] for s in starts} == {0.1, 3.1}


def test_an_unusable_storey_falls_through_to_the_next(straight_line):
    authored = SimpleNamespace(semantic_id=50002, handle="024_bowl")
    starts = ycb_env._starts_for_object(
        _Pathfinder(), [{"position": [0.0, 0.1, 0.0]}], authored, _cfg(starts=20),
        storey_heights=[0.1, 9.9],           # no point is ever at 9.9
    )
    assert all(s["start_storey_y"] == 0.1 for s in starts)


def test_no_usable_storey_is_an_error(straight_line):
    authored = SimpleNamespace(semantic_id=50003, handle="011_banana")
    with pytest.raises(YCBLayoutError):
        ycb_env._starts_for_object(
            _Pathfinder(), [{"position": [0.0, 0.1, 0.0]}], authored,
            _cfg(starts=1, min_geo=1000.0), storey_heights=[0.1, 3.1],
        )


def test_the_flag_is_part_of_the_manifest_cache_key():
    from osg.core.config import OSGConfig
    cfg = OSGConfig()
    layout = SimpleNamespace(scene_name="s", layout_id="static", layout_sha256="x")
    base = ycb_env.manifest_cache_payload(layout, cfg)
    cfg.ycb.start_storey_uniform = True
    cfg.ycb.start_on_target_floor = True
    changed = ycb_env.manifest_cache_payload(layout, cfg)
    assert base["start_storey_uniform"] is False
    assert changed["start_storey_uniform"] == "object_storeys"
    assert base["start_on_target_floor"] is False and changed["start_on_target_floor"] is True


def test_only_storeys_that_hold_an_object_are_drawn(monkeypatch):
    """00878's basement is navigable and holds nothing in any layout; it is out
    of the prior's scope, so no episode may start there."""
    import osg.eval.floors as floors

    monkeypatch.setattr(floors, "navmesh_storeys", lambda pf, **kw: {"storeys": [
        {"height": -2.8, "n_points": 900}, {"height": 0.04, "n_points": 2000},
        {"height": 2.8, "n_points": 1100}]})

    class _Snap:
        def snap_point(self, p):
            return np.array([p[0], 0.04 if p[1] < 2.0 else 2.8, p[2]], dtype=np.float32)

    def layout(*ys):
        return SimpleNamespace(objects=[SimpleNamespace(translation=[0.0, y, 0.0]) for y in ys])

    cfg = SimpleNamespace(ycb=SimpleNamespace(seed=42, relocation_floor_tolerance_m=0.5))
    assert ycb_env._benchmark_storeys(_Snap(), layout(0.9), layout(3.6), cfg) == [0.04, 2.8]
