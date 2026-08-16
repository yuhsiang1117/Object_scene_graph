"""ASCENT coarse-to-fine reasoning (exploration/coarse_to_fine.py).

The tests that matter here are the GATES and the FAILURE PATHS, not the
prompting: the whole claim of the method is that the LLM is asked rarely, and
the whole risk of wiring a network call into the control loop is that an outage
silently changes the trajectory distribution and an A/B then measures the
outage rather than the method.
"""
from __future__ import annotations

import pytest

from osg.exploration.coarse_to_fine import (
    Area,
    CoarseToFinePlanner,
    FloorDesc,
    floor_priors,
    room_priors,
)


class FakeClient:
    """Records prompts and replays canned replies."""

    def __init__(self, replies=None, raises=False):
        self.replies = list(replies or [])
        self.raises = raises
        self.prompts = []

    def chat(self, system, user, **kw):
        self.prompts.append(user)
        if self.raises:
            raise RuntimeError("endpoint down")
        return self.replies.pop(0) if self.replies else {"Index": "1"}


# ------------------------------------------------------------------- priors

def test_room_priors_match_ascent_knowledge_graph():
    # Transcribed from their knowledge_graph.json; toilet is the sharpest row
    # and the one worth pinning.
    p = room_priors("toilet")
    assert p["bathroom"] == pytest.approx(97.0)
    assert max(p, key=p.get) == "bathroom"
    assert room_priors("TV_Monitor")["bedroom"] == pytest.approx(27.4)
    assert room_priors("nonsense") == {}


def test_floor_priors_indexing_and_fallback():
    # 1-indexed, bottom-up, summing to ~100 within a storey count.
    p = floor_priors("bed", 3)
    assert list(p) == [1, 2, 3]
    assert p[3] == pytest.approx(48.1)
    assert sum(p.values()) == pytest.approx(99.9, abs=0.2)
    # Their table stops at 4 storeys; taller buildings reuse the last row
    # rather than returning nothing.
    tall = floor_priors("bed", 6)
    assert len(tall) == 6 and tall[3] == pytest.approx(50.0)
    # A single-storey building has a degenerate answer, not a KeyError.
    assert floor_priors("bed", 1) == {1: 100.0}
    assert floor_priors("nonsense", 2) == {1: 0.0, 2: 0.0}


# -------------------------------------------------------------- fine gating

def _areas(n=3):
    return [Area(room="unknown room", objects=["chair"]) for _ in range(n)]


def test_area_not_asked_when_a_frontier_is_near():
    """The gate that makes this affordable: a frontier 2 m away is walked to,
    not reasoned about."""
    c = FakeClient()
    p = CoarseToFinePlanner(c, nearby_m=3.0)
    assert not p.should_ask_area(best_path_cost=2.0, n_areas=3)
    assert p.should_ask_area(best_path_cost=8.0, n_areas=3)
    # One candidate is not a choice.
    assert not p.should_ask_area(best_path_cost=8.0, n_areas=1)
    # An unreachable-but-far best still counts as "nothing near".
    assert p.should_ask_area(best_path_cost=None, n_areas=2)
    assert c.prompts == []


def test_area_choice_parses_one_based_index():
    p = CoarseToFinePlanner(FakeClient([{"Index": "2", "Reason": "bathroom"}]))
    assert p.choose_area("toilet", _areas(3)) == 1
    assert p.calls == 1


def test_log_records_what_the_llm_was_shown():
    """The index alone cannot tell you whether the method had a fair trial --
    a choice between three "unknown room / no visible objects" areas is a coin
    flip dressed as reasoning, and only the descriptions reveal that."""
    p = CoarseToFinePlanner(FakeClient([{"Index": "2"}]))
    p.choose_area("toilet", [Area("bathroom", ["sink"]), Area("kitchen", ["oven"])], step=7)
    step, kind, idx, shown = p.log[0]
    assert (step, kind, idx) == (7, "area", 1)
    assert shown == ["a bathroom containing objects: sink",
                     "a kitchen containing objects: oven"]


def test_area_prompt_carries_priors_and_objects():
    c = FakeClient([{"Index": "1"}])
    p = CoarseToFinePlanner(c)
    p.choose_area("toilet", [Area("bathroom", ["sink", "towel"]), Area("kitchen", ["oven"])])
    prompt = c.prompts[0]
    assert '"Goal": "toilet"' in prompt
    assert "97.0%" in prompt  # P(bathroom | toilet)
    assert "a bathroom containing objects: sink, towel" in prompt
    # Priors are listed most-likely first, as ASCENT does.
    assert prompt.index("Bathroom") < prompt.index("Rec room")


@pytest.mark.parametrize("reply", [
    {"Index": "0"},            # 1-based, so 0 is out of range
    {"Index": "9"},            # past the end
    {"Index": "banana"},
    {"Reason": "no index at all"},
    {},
])
def test_bad_area_reply_falls_back_to_geometric_best(reply):
    p = CoarseToFinePlanner(FakeClient([reply]))
    assert p.choose_area("toilet", _areas(3)) == 0


def test_llm_outage_does_not_change_the_choice():
    """An endpoint that starts refusing must leave behaviour exactly as it was,
    or the A/B measures the outage instead of the method."""
    p = CoarseToFinePlanner(FakeClient(raises=True))
    assert p.choose_area("toilet", _areas(3)) == 0
    assert p.errors == 1 and p.calls == 0


def test_no_client_is_inert():
    p = CoarseToFinePlanner(None)
    assert not p.should_ask_area(99.0, 5)
    assert p.choose_area("toilet", _areas(3)) == 0
    assert not p.should_ask_floor(500, 3, 400)


# ------------------------------------------------------------ coarse gating

def _floors(n=2, current=1):
    return [
        FloorDesc(index=i + 1, is_current=(i + 1 == current), rooms=[], objects=[])
        for i in range(n)
    ]


def test_floor_not_asked_before_the_current_one_is_searched():
    p = CoarseToFinePlanner(FakeClient(), floor_min_steps_on_floor=100, floor_ask_interval=60)
    assert not p.should_ask_floor(step=200, n_floors=2, steps_on_floor=40)
    assert p.should_ask_floor(step=200, n_floors=2, steps_on_floor=150)
    # A building with one known storey has no question to ask.
    assert not p.should_ask_floor(step=200, n_floors=1, steps_on_floor=150)


def test_floor_ask_interval_is_stamped_even_when_the_reply_is_useless():
    """Asking is what costs. A failed call must still start the interval, or a
    flapping endpoint gets asked every single selection."""
    p = CoarseToFinePlanner(FakeClient(raises=True), floor_ask_interval=60,
                            floor_min_steps_on_floor=0)
    assert p.should_ask_floor(step=100, n_floors=2, steps_on_floor=999)
    assert p.choose_floor("bed", _floors(2), current_index=1, step=100) == 1
    assert not p.should_ask_floor(step=130, n_floors=2, steps_on_floor=999)
    assert p.should_ask_floor(step=161, n_floors=2, steps_on_floor=999)


def test_floor_choice_can_veto_the_switch():
    """The half our co-occurrence table cannot express: priors.py judges the
    current floor alone and never compares storeys, so it can say "leave" but
    never "the one you are on is still the best bet"."""
    p = CoarseToFinePlanner(FakeClient([{"Index": "2", "Reason": "stay"}]))
    assert p.choose_floor("bed", _floors(3, current=2), current_index=2, step=1) == 2


def test_floor_description_marks_current_and_explored():
    c = FakeClient([{"Index": "2"}])
    p = CoarseToFinePlanner(c)
    p.choose_floor("bed", [
        FloorDesc(1, True, ["hall"], ["tv"], fully_explored=False),
        FloorDesc(2, False, [], [], fully_explored=True),
    ], current_index=1, step=1)
    prompt = c.prompts[0]
    assert "Current floor. There are room types: hall" in prompt
    assert "You do not need to explore this floor again" in prompt
    assert "unknown rooms" in prompt and "unknown objects" in prompt
    assert '"Floor 1": 41.0%' in prompt  # P(bed | 2 storeys, floor 1)
