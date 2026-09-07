"""Places365 room classification, and the hole it exists to fill.

`RoomNode.label` had exactly one writer -- `LLMTextScorer.score` -- and every
measured arm runs a NullScorer, so labels were always None and the S10 ranker
described every frontier as "a unknown room". These tests cover the mapping
logic without a GPU, and the last one is the integration check that was missing:
that a label actually reaches the scene graph.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from osg.perception.room_classifier import load_categories, map_to_room

TABLE = json.loads(open("data/priors/place365_room_map.json").read())
MAPPING, ROOMS = TABLE["direct_mapping"], set(TABLE["reference_rooms"])


def _map(top):
    return map_to_room(top, MAPPING, ROOMS)


# ------------------------------------------------------------------- mapping


def test_first_mappable_class_in_rank_order_wins():
    """Rank order decides, not the mapping table's order: ASCENT breaks out of
    the loop at the first hit (utils.py:222-226)."""
    assert _map(["garage/indoor", "bathroom", "kitchen"]) == "garage"
    assert _map(["bathroom", "garage/indoor", "kitchen"]) == "bathroom"


def test_unmappable_classes_are_skipped_until_one_maps():
    assert _map(["attic", "basement", "shower", "kitchen"]) == "bathroom"


def test_falls_back_to_raw_top1_when_nothing_maps():
    """Deliberate, and it matters: an unmapped-but-real room name is still
    information for the LLM, where "unknown" is not (utils.py:229)."""
    assert _map(["attic", "basement", "balcony"]) == "attic"


def test_empty_input_is_handled():
    assert _map([]) == "unknown room"


@pytest.mark.parametrize(
    "place365,expected",
    [("shower", "bathroom"), ("hotel_room", "bedroom"), ("corridor", "hall"),
     ("television_room", "living_room"), ("home_office", "office"),
     ("playroom", "rec_room"), ("banquet_hall", "dining_room")],
)
def test_synonyms_reach_their_reference_room(place365, expected):
    assert _map([place365]) == expected


def test_every_mapping_target_is_a_reference_room():
    """The table is copied from ASCENT; a typo would silently produce a room
    type the knowledge-graph priors have no entry for."""
    assert set(MAPPING.values()) <= ROOMS


def test_reference_rooms_match_the_knowledge_graph():
    """The priors are keyed on these names. A mismatch means the prompt states
    probabilities for rooms the area descriptions can never mention."""
    from osg.exploration.knowledge_prior import KnowledgeGraph
    from pathlib import Path

    path = Path("data/priors/hm3d_room_prior.json")
    if not path.exists():
        pytest.skip("priors not generated")
    kg = KnowledgeGraph(json.loads(path.read_text()))
    known = {r for probs in kg._rooms.values() for r in probs} if hasattr(kg, "_rooms") else set()
    if not known:
        pytest.skip("knowledge graph exposes no room vocabulary")
    assert ROOMS & known, f"no overlap between {sorted(ROOMS)} and {sorted(known)}"


# ---------------------------------------------------------------- categories


def test_category_file_parses_to_365_clean_names():
    cats = load_categories("data/place365/categories_places365.txt")
    assert len(cats) == 365
    assert cats[0] == "airfield"
    assert not any(c.startswith("/") for c in cats), "letter prefix not stripped"
    assert not any(" " in c for c in cats), "class index not stripped"


def test_nested_categories_keep_their_subtype():
    """`garage/indoor` and `garage/outdoor` are distinct Places365 classes and
    both appear in the mapping table, so the slash must survive parsing."""
    cats = load_categories("data/place365/categories_places365.txt")
    assert "garage/indoor" in cats


def test_the_one_dead_key_in_ascents_table_is_still_dead():
    """`laundry_room` is a key in ASCENT's DIRECT_MAPPING (constants.py:283) but
    is not a Places365 class -- the dataset has `laundromat` and `utility_room`.
    So that entry can never fire, and a laundry is only recognised through
    `laundromat`.

    Kept rather than fixed: the table is a faithful copy, and quietly adding
    `utility_room` would make this port diverge from the baseline it is being
    compared against. Recorded here so the gap is a known one.
    """
    cats = set(load_categories("data/place365/categories_places365.txt"))
    dead = set(MAPPING) - cats
    assert dead == {"laundry_room"}, f"mapping keys drifted from Places365: {dead}"
    assert "laundromat" in cats and "utility_room" in cats


# ------------------------------------------------------- reaching the graph


class _StubClassifier:
    """Returns a scripted room name per call."""

    def __init__(self, names):
        self.names = list(names)
        self.n = 0

    def classify(self, rgb):
        name = self.names[min(self.n, len(self.names) - 1)]
        self.n += 1
        return name


def _agent_with(classifier):
    from osg.agent.nav_agent import NavAgent
    from osg.exploration.async_scorer import AsyncScorer
    from osg.perception.detector import StubDetector

    from .test_nav_agent import _StubScorer, make_cfg

    return NavAgent(make_cfg(), StubDetector(), AsyncScorer(_StubScorer()), None,
                    "chair", room_classifier=classifier)


def _label(agent, votes):
    """Drive _label_rooms with hand-placed votes over a two-room segmentation."""
    layer = agent.floors.current()
    grid = layer.costmap.grid
    labels = np.zeros(grid.shape, dtype=np.int32)
    mid = grid.shape[1] // 2
    labels[:, :mid] = 1
    labels[:, mid:] = 2
    layer.room_labels = labels
    agent.scene_graph.rebuild_floor(labels, layer.costmap, agent.object_layer,
                                    floor_key=layer.key)
    agent._room_votes = [
        (layer.costmap.grid_to_world(np.array(rc, float)), name) for rc, name in votes
    ]
    agent._label_rooms(layer)
    return agent.scene_graph.rooms


def test_labels_reach_the_scene_graph():
    """The integration check missing when S10 shipped. RoomNode.label had one
    writer -- the LLM scorer no measured arm enables -- so every area reached the
    ranker as "unknown room"."""
    agent = _agent_with(_StubClassifier(["bathroom"]))
    grid = agent.floors.current().costmap.grid
    mid = grid.shape[1] // 2
    rooms = _label(agent, [((10, mid // 2), "bathroom"), ((11, mid + 10), "kitchen")])
    assert {r.label for r in rooms.values()} == {"bathroom", "kitchen"}


def test_majority_vote_survives_one_bad_frame():
    """A frame taken through a doorway classifies the room beyond it; one such
    vote must not rename a room the agent has crossed repeatedly."""
    agent = _agent_with(_StubClassifier(["bedroom"]))
    grid = agent.floors.current().costmap.grid
    c = grid.shape[1] // 4
    votes = [((10 + i, c), "bedroom") for i in range(5)] + [((20, c), "garage")]
    rooms = _label(agent, votes)
    labelled = [r.label for r in rooms.values() if r.label]
    assert "bedroom" in labelled and "garage" not in labelled


def test_votes_outside_any_room_are_ignored():
    agent = _agent_with(_StubClassifier(["hall"]))
    layer = agent.floors.current()
    labels = np.zeros(layer.costmap.grid.shape, dtype=np.int32)
    labels[5:15, 5:15] = 1
    layer.room_labels = labels
    agent.scene_graph.rebuild_floor(labels, layer.costmap, agent.object_layer,
                                    floor_key=layer.key)
    agent._room_votes = [(layer.costmap.grid_to_world(np.array([50, 50], float)), "hall")]
    agent._label_rooms(layer)
    assert all(r.label is None for r in agent.scene_graph.rooms.values())


def test_no_classifier_leaves_labels_none():
    """The default, and the configuration every measured arm ran under."""
    agent = _agent_with(None)
    rooms = _label(agent, [])
    assert all(r.label is None for r in rooms.values())
