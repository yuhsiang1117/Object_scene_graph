"""The vendored `Ascent_LLM_Planner`: prompts, priors and reply parsing."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ascentnav.planner import AscentLLMPlanner, KnowledgeGraph


class _OM:
    _floor_num_steps = 7
    _this_floor_explored = False

    def __init__(self):
        self._each_step_rgb = {7: np.zeros((2, 2, 3), np.uint8)}
        self._disabled_frontiers = set()
        self._best_frontier_selection_count = {}
        self._finish_first_explore = False
        self._neighbor_search = False


class _OBJ:
    def __init__(self):
        self.each_step_rooms = {7: "bathroom"}
        self.each_step_objects = {7: ["shower", "towel"]}
        self.this_floor_rooms = {"bathroom"}
        self.this_floor_objects = {"shower"}


def _kg():
    return KnowledgeGraph({"toilet": {"bathroom": 0.97, "bedroom": 0.012},
                           "couch": {"living_room": 0.434}}, {"toilet", "couch", "bathroom"})


def _planner(llm=None):
    p = AscentLLMPlanner(llm=llm, knowledge_graph=_kg(),
                         floor_prior={"bed": {"2": {"1": 41.0, "2": 59.0}, "3": {"1": 22.2, "2": 29.6, "3": 48.1}}})
    p.frontier_step_list = [7]
    return p


def test_room_probabilities_match_the_references_rounding_and_fill():
    """`llm_planner.py:361-390`: every REFERENCE_ROOM present, missing ones 0.0,
    weights reported as percentages rounded to one decimal, synonyms widen."""
    p = _planner()
    r = p.get_room_probabilities("toilet")
    assert r["bathroom"] == 97.0 and r["bedroom"] == 1.2 and r["kitchen"] == 0.0
    assert p.get_room_probabilities("sofa")["living_room"] == 43.4, "sofa -> couch"
    assert p.get_room_probabilities("zebra") == {}


def test_floor_probabilities_clip_to_the_tables_largest_building():
    """`:392-429`."""
    p = _planner()
    assert p.get_floor_probabilities("bed", 2) == {1: 41.0, 2: 59.0}
    assert p.get_floor_probabilities("bed", 5) == {1: 22.2, 2: 29.6, 3: 48.1}
    assert p.get_floor_probabilities("plant", 3) == {1: 0.0, 2: 0.0, 3: 0.0}


def test_the_single_floor_prompt_is_the_references():
    p = _planner()
    prompt = p._prepare_single_floor_prompt("toilet", _OM(), _OBJ())
    assert prompt.startswith("You need to select the optimal area based on prior probabilistic data")
    assert '"Goal": "toilet"' in prompt
    assert '        "Bathroom": 97.0%' in prompt
    assert '        "Area 1": "a bathroom containing objects: shower, towel"' in prompt
    assert prompt.endswith("}")
    assert 'Example Response:\n{"Index": "1"' in prompt


def test_the_multi_floor_prompt_marks_explored_floors():
    p = _planner()
    p.floor_num = 2
    om0, om1 = _OM(), _OM()
    om1._this_floor_explored = True
    prompt = p._prepare_multiple_floor_prompt("bed", 0, [om0, om1], [_OBJ(), _OBJ()])
    assert '"Floor 1": 41.0%' in prompt and '"Floor 2": 59.0%' in prompt
    assert "Floor 1\": \"Current floor." in prompt
    assert "You do not need to explore this floor again" in prompt


@pytest.mark.parametrize("reply", ["not json", '{"Reason": "x"}', '{"Index": "none"}',
                                   '{"Index": "9"}', '{"Index": "0"}', "-1"])
def test_bad_replies_keep_the_value_ranking(reply):
    """`:303-357`: index 0 on anything that is not a valid 1-based index."""
    p = _planner(llm=lambda prompt: reply)
    idx = p.llm_analyze_single_floor("toilet", [0, 1, 2], _OM(), _OBJ())
    assert idx == 0


def test_a_valid_index_is_converted_to_the_candidate():
    p = _planner(llm=lambda prompt: '{"Index": "3", "Reason": "because"}')
    assert p.llm_analyze_single_floor("toilet", [4, 5, 6], _OM(), _OBJ()) == 6
    assert p.stats["rank_overrides"] == 1


def test_a_fenced_reply_is_still_parsed():
    p = _planner(llm=lambda prompt: '```json\n{"Index": "2"}\n```')
    assert p.llm_analyze_single_floor("toilet", [4, 5, 6], _OM(), _OBJ()) == 5


def test_an_llm_exception_is_the_references_minus_one():
    def boom(prompt):
        raise RuntimeError("down")

    p = _planner(llm=boom)
    assert p.llm_analyze_single_floor("toilet", [4, 5], _OM(), _OBJ()) == 4
    assert p.stats["rank_errors"] == 1


def test_no_llm_means_no_call():
    p = _planner(llm=None)
    assert p.llm_analyze_single_floor("toilet", [4, 5], _OM(), _OBJ()) == 4
    assert "llm_calls" not in p.stats


def test_the_knowledge_graph_loads_node_link_json(tmp_path):
    f = tmp_path / "kg.json"
    f.write_text(json.dumps({"nodes": [{"id": "a"}, {"id": "b"}],
                             "links": [{"source": "a", "target": "b", "weight": 0.5}]}))
    kg = KnowledgeGraph.load(str(f))
    assert "a" in kg and kg.has_edge("a", "b") and kg.weight("a", "b") == 0.5
    assert not kg.has_edge("b", "a")


def test_the_floor_decision_is_current_floor_on_any_failure():
    p = _planner()
    p.floor_num = 3
    assert p._extract_multiple_floor_decision("garbage", 1) == 2
    assert p._extract_multiple_floor_decision('{"Index": "7"}', 1) == 2
    assert p._extract_multiple_floor_decision('{"Index": "3", "Reason": "r"}', 1) == 3
