"""ASCENT's forced-choice frontier ranker.

The properties that matter are that it never raises (a network hiccup must not
end an episode), that it degrades to the caller's own ranking on every failure
path, and that the prompt actually carries the two things the port exists for --
the priors as text and the areas described by room and nearby objects.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.exploration.ascent_ranker import AscentFrontierRanker, describe_area
from osg.exploration.knowledge_prior import KnowledgeGraph
from osg.graph.scene_graph import ObjectNodeView, RoomNode, SceneGraph
from osg.mapping.frontier import Frontier

ROOMS = {
    "toilet": {"bathroom": 0.9, "bedroom": 0.1},
    "bed": {"bedroom": 0.8, "living_room": 0.2},
}


class _StubClient:
    """Records the prompt and replies with a scripted answer."""

    def __init__(self, reply=None, raises=False):
        self.reply = reply if reply is not None else {"Index": "1", "Reason": "-"}
        self.raises = raises
        self.system = None
        self.user = None
        self.n = 0

    def chat(self, system, user, **kw):
        self.n += 1
        self.system, self.user = system, user
        if self.raises:
            raise RuntimeError("network down")
        return self.reply


def _f(fid: int, x: float, z: float = 0.0) -> Frontier:
    return Frontier(
        id=fid, centroid_xy=np.array([x, z], float),
        cells=np.zeros((0, 2), dtype=int), size=10,
    )


def _sg(rooms=(), objects=()) -> SceneGraph:
    sg = SceneGraph()
    sg.rooms = {r[0]: RoomNode(id=r[0], label=r[1], centroid_xy=np.array(r[2], float))
                for r in rooms}
    sg.objects = [
        ObjectNodeView(track_id=i, label=o[0], center=np.array(o[1], float),
                       room_id=o[2] if len(o) > 2 else 0, n_obs=3)
        for i, o in enumerate(objects)
    ]
    return sg


def _rank(client, kg=None, **kw):
    return AscentFrontierRanker(client, kg=kg, **kw)


# ------------------------------------------------------------------- choosing


def test_picks_the_index_the_model_returns():
    c = _StubClient({"Index": "2", "Reason": "x"})
    assert _rank(c).pick([_f(0, 1), _f(1, 2), _f(2, 3)], "toilet") == 1


def test_single_frontier_needs_no_call():
    c = _StubClient()
    assert _rank(c).pick([_f(0, 1)], "toilet") == 0
    assert c.n == 0, "one option is not a choice"


def test_only_topk_are_offered():
    """Eight frontiers in, three areas described. Counting inside the actual
    input only -- the few-shot example carries its own Area 1..3."""
    c = _StubClient()
    _rank(c, topk=3).pick([_f(i, float(i)) for i in range(8)], "toilet")
    body = c.user.split("Now answer question:")[1]
    assert [f'"Area {i}"' in body for i in (1, 2, 3, 4)] == [True, True, True, False]


def test_override_is_counted():
    """The diagnostic that says whether the model is doing anything: if it never
    disagrees with the value ranking, the calls are pure cost."""
    r = _rank(_StubClient({"Index": "3", "Reason": "x"}))
    r.pick([_f(0, 1), _f(1, 2), _f(2, 3)], "toilet")
    assert r.overrides == 1
    r2 = _rank(_StubClient({"Index": "1", "Reason": "x"}))
    r2.pick([_f(0, 1), _f(1, 2)], "toilet")
    assert r2.overrides == 0


# -------------------------------------------------------------- failure paths


@pytest.mark.parametrize(
    "reply", [{"Index": "0"}, {"Index": "9"}, {"Index": "banana"}, {}, {"Reason": "x"}]
)
def test_bad_replies_fall_back_to_the_value_ranking(reply):
    """ASCENT returns index 0 on every parse failure (llm_planner.py:296-300).
    Anything else would let one malformed reply steer the episode."""
    assert _rank(_StubClient(reply)).pick([_f(0, 1), _f(1, 2), _f(2, 3)], "toilet") == 0


def test_network_failure_does_not_raise():
    assert _rank(_StubClient(raises=True)).pick([_f(0, 1), _f(1, 2)], "toilet") == 0


# ------------------------------------------------------------------ the prompt


def test_priors_appear_as_percentages_in_descending_order():
    c = _StubClient()
    _rank(c, kg=KnowledgeGraph(ROOMS)).pick([_f(0, 1), _f(1, 2)], "toilet")
    body = c.user.split("Now answer question:")[1]
    assert '"Bathroom": 90.0%' in body
    assert body.index('"Bathroom"') < body.index('"Bedroom"')


def test_unknown_target_still_produces_a_valid_prompt():
    c = _StubClient()
    _rank(c, kg=KnowledgeGraph(ROOMS)).pick([_f(0, 1), _f(1, 2)], "aardvark")
    assert "Now answer question:" in c.user
    assert c.n == 1


def test_area_describes_room_and_nearby_objects():
    sg = _sg(rooms=[(1, "bathroom", [0.0, 0.0])],
             objects=[("shower", [0.5, 0.9, 0.0]), ("towel", [-0.5, 0.9, 0.0])])
    text = describe_area(_f(0, 0.0), sg, radius_m=3.0)
    assert text == "a bathroom containing objects: shower, towel"


def test_area_without_a_scene_graph_is_still_well_formed():
    assert describe_area(_f(0, 0.0), None, 3.0) == \
        "a unknown room containing objects: no visible objects"


def test_far_objects_are_excluded():
    sg = _sg(rooms=[(1, "bathroom", [0.0, 0.0])],
             objects=[("shower", [0.5, 0.9, 0.0]), ("car", [20.0, 0.9, 0.0])])
    assert "car" not in describe_area(_f(0, 0.0), sg, radius_m=3.0)


def test_object_list_is_capped_and_takes_the_nearest():
    sg = _sg(rooms=[(1, "kitchen", [0.0, 0.0])],
             objects=[(f"obj{i}", [0.1 * i, 0.9, 0.0]) for i in range(20)])
    text = describe_area(_f(0, 0.0), sg, radius_m=5.0, max_objects=3)
    assert text.count(",") == 2
    assert "obj19" not in text, "the cap should keep the closest, not the last seen"


def test_system_prompt_matches_ascent():
    c = _StubClient()
    _rank(c).pick([_f(0, 1), _f(1, 2)], "toilet")
    assert "advanced spatial reasoning" in c.system


def test_counters_reset_per_episode():
    """The runner builds one ranker and shares it across every episode, so
    without a reset the per-episode stats are a running total. Measured before
    the fix: 67 calls/episode reported against 3.5 actual."""
    r = _rank(_StubClient({"Index": "2", "Reason": "x"}))
    r.pick([_f(0, 1), _f(1, 2)], "toilet")
    assert (r.calls, r.overrides) == (1, 1)
    r.reset()
    assert (r.calls, r.overrides) == (0, 0)


def test_nav_agent_resets_the_ranker_each_episode():
    from osg.agent.nav_agent import NavAgent
    from osg.exploration.async_scorer import AsyncScorer
    from osg.perception.detector import StubDetector

    from .test_nav_agent import _StubScorer, make_cfg

    ranker = _rank(_StubClient())
    ranker.calls, ranker.overrides = 9, 4
    agent = NavAgent(make_cfg(), StubDetector(), AsyncScorer(_StubScorer()), None,
                     "chair", ranker=ranker)
    agent.reset("chair")
    assert (ranker.calls, ranker.overrides) == (0, 0)


def test_unlabelled_rooms_degrade_to_unknown_room():
    """The production configuration, and the reason the S10 measurement did not
    test what it claimed to.

    RoomNode.label is written in exactly one place -- LLMTextScorer.score
    (llm_scorer.py:81) -- and every measured arm runs exploration.frontier_text_scorer=disabled,
    which is a NullScorer. So labels are always None and every area reaches the
    model as "a unknown room containing objects: ...". The ranker was therefore
    given object lists with no room type at all, which is half of what the
    ASCENT prompt is built around.

    The earlier tests here passed labels in explicitly and so never covered this.
    """
    sg = _sg(rooms=[(1, None, [0.0, 0.0])],
             objects=[("shower", [0.5, 0.9, 0.0]), ("towel", [-0.5, 0.9, 0.0])])
    assert describe_area(_f(0, 0.0), sg, radius_m=3.0) == \
        "a unknown room containing objects: shower, towel"
