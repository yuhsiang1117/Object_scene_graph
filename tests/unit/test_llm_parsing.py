from __future__ import annotations

import numpy as np
import pytest

from osg.exploration.llm_scorer import LLMTextScorer
from osg.llm.client import extract_json
from osg.mapping.frontier import Frontier


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_prose():
    assert extract_json('Sure! Here you go:\n{"scores": {"1": 0.8}}\nHope that helps.') == {
        "scores": {"1": 0.8}
    }


def test_extract_json_failure():
    with pytest.raises(ValueError):
        extract_json("no json here")


def _frontiers(ids):
    return [Frontier(id=i, centroid_xy=np.zeros(2), cells=np.zeros((1, 2)), size=1) for i in ids]


def test_parse_scores_clamps_and_filters():
    resp = {"scores": {"1": 1.7, "2": -0.5, "3": "0.4", "junk": 0.9, "7": 0.5}}
    out = LLMTextScorer._parse_scores(resp, _frontiers([1, 2, 3]))
    assert out == {1: 1.0, 2: 0.0, 3: 0.4}  # id 7 not in subset, junk dropped


def test_parse_scores_without_wrapper():
    out = LLMTextScorer._parse_scores({"1": 0.6}, _frontiers([1]))
    assert out == {1: 0.6}
