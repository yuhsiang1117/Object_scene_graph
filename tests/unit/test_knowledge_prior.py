"""Statistical priors and the spatial score cache.

The room prior is the LLM-free half of ASCENT's frontier reasoning: a knowledge
graph over (object, room) turns objects already mapped near a frontier into a
room-type distribution, whose overlap with the goal's own room distribution
scores the frontier. No model call, and none of the id/async fragility that
made the LLM scorer measurably never influence a selection.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from osg.exploration.knowledge_prior import FloorPrior, KnowledgeGraph
from osg.exploration.score_cache import SpatialScoreCache

ROOMS = {
    "toilet": {"bathroom": 0.97, "bedroom": 0.02, "kitchen": 0.01},
    "bed": {"bedroom": 0.75, "office": 0.20, "bathroom": 0.05},
    "sofa": {"living_room": 0.60, "bedroom": 0.30, "kitchen": 0.10},
}


def _kg():
    return KnowledgeGraph(ROOMS)


# ------------------------------------------------------------------ room prior


def test_room_probabilities_are_normalised():
    d = _kg().room_probabilities("toilet")
    assert pytest.approx(sum(d.values())) == 1.0
    assert max(d, key=d.get) == "bathroom"


def test_unknown_category_yields_nothing():
    assert _kg().room_probabilities("aardvark") == {}


def test_synonym_and_case_are_normalised():
    kg = KnowledgeGraph({"tv_monitor": {"living_room": 1.0}})
    assert kg.room_probabilities("TV Monitor")["living_room"] == 1.0


def test_same_room_objects_score_higher_than_other_room_objects():
    """The whole point: a frontier surrounded by bathroom-ish things should
    look promising for a toilet and unpromising for a bed."""
    kg = _kg()
    near_bathroom = kg.affinity("toilet", ["toilet"])
    near_bedroom = kg.affinity("toilet", ["bed"])
    assert near_bathroom > near_bedroom


def test_affinity_is_none_without_evidence():
    """None means 'no evidence', not 'unpromising'. Confusing the two would
    permanently deprioritise unexplored regions, which by definition have no
    objects mapped near them yet -- exactly where the agent must go."""
    kg = _kg()
    assert kg.affinity("toilet", []) is None
    assert kg.affinity("toilet", ["aardvark", "unicorn"]) is None
    assert kg.affinity("aardvark", ["toilet"]) is None


def test_affinity_averages_over_nearby_objects():
    kg = _kg()
    only_bed = kg.affinity("bed", ["bed"])
    mixed = kg.affinity("bed", ["bed", "toilet"])
    assert only_bed > mixed > 0.0


def test_loads_the_generated_priors_if_present():
    """The committed subset must stay loadable and sane -- it is generated from
    ASCENT by scripts/make_priors.py and is easy to break silently."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data/priors/hm3d_room_prior.json"
    if not path.exists():
        pytest.skip("priors not generated")
    kg = KnowledgeGraph(json.loads(path.read_text()))
    assert max(kg.room_probabilities("toilet"), key=kg.room_probabilities("toilet").get) == "bathroom"
    assert max(kg.room_probabilities("bed"), key=kg.room_probabilities("bed").get) == "bedroom"


# ----------------------------------------------------------------- floor prior


def test_floor_prior_is_zero_indexed_and_normalised():
    fp = FloorPrior({"bed": {"3": {"1": 22.2, "2": 29.6, "3": 48.1}}})
    d = fp.probabilities("bed", 3)
    assert pytest.approx(sum(d.values())) == 1.0
    assert max(d, key=d.get) == 2, "table floors are 1-based, FloorStack order is 0-based"


def test_floor_prior_clamps_to_the_largest_table_entry():
    """A 6-storey scene should still get the shape of the distribution rather
    than nothing at all."""
    fp = FloorPrior({"bed": {"3": {"1": 20.0, "2": 30.0, "3": 50.0}}})
    assert fp.probabilities("bed", 6) == fp.probabilities("bed", 3)


def test_floor_prior_unknown_category():
    assert FloorPrior({}).probabilities("bed", 2) == {}


# ---------------------------------------------------------------- score cache


def test_score_is_found_by_position_not_id():
    """Frontier ids are reassigned on every extraction, which is why scores are
    keyed spatially at all."""
    c = SpatialScoreCache(radius_m=0.75, ttl_steps=60)
    c.put(np.array([1.0, 2.0]), 0.8, step=10)
    assert c.get(np.array([1.3, 2.1]), step=12) == 0.8


def test_score_not_found_beyond_the_radius():
    c = SpatialScoreCache(radius_m=0.75)
    c.put(np.array([1.0, 2.0]), 0.8, step=10)
    assert c.get(np.array([5.0, 5.0]), step=12) is None


def test_scores_expire():
    c = SpatialScoreCache(radius_m=0.75, ttl_steps=20)
    c.put(np.array([1.0, 2.0]), 0.8, step=10)
    assert c.get(np.array([1.0, 2.0]), step=25) == 0.8
    assert c.get(np.array([1.0, 2.0]), step=40) is None


def test_nearest_entry_wins():
    c = SpatialScoreCache(radius_m=1.0)
    c.put(np.array([1.0, 0.0]), 0.2, step=1)
    c.put(np.array([0.1, 0.0]), 0.9, step=1)
    assert c.get(np.array([0.0, 0.0]), step=2) == 0.9


def test_reset_clears():
    c = SpatialScoreCache()
    c.put(np.array([1.0, 2.0]), 0.8, step=1)
    c.reset()
    assert c.get(np.array([1.0, 2.0]), step=2) is None
