"""Coarse level of the cascade: which storey is the target on?

The properties that matter are ASCENT's gating (a floor question is expensive
and only worth asking once this floor has been given a fair chance), that every
failure means "stay here" rather than acting on a malformed answer, and that the
decision translates into a *direction* the existing stair machinery can execute.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.exploration.floor_planner import FloorDecisionPlanner, describe_floor
from osg.graph.scene_graph import ObjectNodeView, RoomNode, SceneGraph


class _StubClient:
    def __init__(self, reply=None, raises=False):
        self.reply = reply if reply is not None else {"Index": "1", "Reason": "-"}
        self.raises = raises
        self.user = None
        self.n = 0

    def chat(self, system, user, **kw):
        self.n += 1
        self.user = user
        if self.raises:
            raise RuntimeError("network down")
        return self.reply


class _Layer:
    def __init__(self, key, floor_y, steps_on_floor=200, explored=False):
        self.key = key
        self.floor_y = floor_y
        self.steps_on_floor = steps_on_floor
        self.explored = explored


class _Floors:
    """Minimal FloorStack stand-in: layers ordered by height."""

    def __init__(self, layers, current_idx=0):
        self._layers = layers
        self._cur = current_idx

    def layers(self):
        return self._layers

    def n_floors(self):
        return len(self._layers)

    def current(self):
        return self._layers[self._cur]

    def order_of(self, key):
        return next(i for i, l in enumerate(self._layers) if l.key == key)


def _sg(rooms=(), objects=()):
    sg = SceneGraph()
    sg.rooms = {
        r[0]: RoomNode(id=r[0], label=r[1], centroid_xy=np.zeros(2),
                       n_cells=r[2] if len(r) > 2 else 10, floor=r[3] if len(r) > 3 else 0)
        for r in rooms
    }
    sg.objects = [
        ObjectNodeView(track_id=i, label=o[0], center=np.zeros(3), room_id=0,
                       n_obs=3, floor=o[1] if len(o) > 1 else 0)
        for i, o in enumerate(objects)
    ]
    return sg


def _planner(client, **kw):
    return FloorDecisionPlanner(client, **kw)


def _two_floors(**kw):
    return _Floors([_Layer(0, 0.0, **kw), _Layer(1, 3.0, **kw)])


# --------------------------------------------------------------------- gating


def test_never_asks_when_only_one_floor_could_exist():
    c = _StubClient()
    assert _planner(c).decide("bed", _Floors([_Layer(0, 0.0)]), _sg(), step=500) is None
    assert c.n == 0


def test_a_detected_staircase_is_enough_to_ask():
    """The design correction a smoke run forced. FloorStack allocates a storey
    only once the agent has STOOD at that height, so a question about which
    floor to search could only be asked after the climb it was meant to
    motivate -- 1 of 4 cross-floor episodes ever reached the model. ASCENT
    allocates on stair DETECTION (map_controller.py:530-537), so seeing a
    flight is enough.
    """
    c = _StubClient({"Index": "2", "Reason": "-"})
    one = _Floors([_Layer(0, 0.0)])
    assert _planner(c).decide("bed", one, _sg(), step=500) is None
    assert c.n == 0
    assert _planner(c).decide("bed", one, _sg(), step=500, has_up=True) == +1
    assert c.n == 1


def test_a_downward_staircase_offers_a_floor_below():
    """The implied floor goes UNDER the visited ones, so choosing it is -1."""
    c = _StubClient({"Index": "1", "Reason": "-"})  # first entry = the one below
    one = _Floors([_Layer(0, 0.0)])
    assert _planner(c).decide("bed", one, _sg(), step=500, has_down=True) == -1


def test_implied_floors_are_described_as_unmapped():
    c = _StubClient()
    _planner(c).decide("bed", _Floors([_Layer(0, 0.0)]), _sg(), step=500, has_up=True)
    body = c.user.split("Now answer question:")[1]
    assert body.count('"Floor ') == 3, "one visited floor, one implied, one priors key"
    assert "unknown rooms" in body


def test_waits_until_this_floor_has_had_a_fair_chance():
    """ASCENT's FLOOR_EXP_STEP_THRESHOLD (llm_planner.py:217): leaving a floor
    before exploring it is how an agent ping-pongs between storeys."""
    c = _StubClient()
    p = _planner(c, min_steps_on_floor=100)
    assert p.decide("bed", _two_floors(steps_on_floor=50), _sg(), step=500) is None
    assert c.n == 0
    assert p.decide("bed", _two_floors(steps_on_floor=150), _sg(), step=500) is not None


def test_rate_limited_between_asks():
    """MULTI_FLOOR_ASK_STEP_THRESHOLD -- a blocking call per selection round
    would dominate the step budget."""
    c = _StubClient({"Index": "2", "Reason": "-"})
    p = _planner(c, ask_every_steps=60)
    p.decide("bed", _two_floors(), _sg(), step=200)
    assert c.n == 1
    p.decide("bed", _two_floors(), _sg(), step=230)
    assert c.n == 1, "asked again inside the throttle window"
    p.decide("bed", _two_floors(), _sg(), step=300)
    assert c.n == 2


# ------------------------------------------------------------------- deciding


def test_returns_a_direction_not_a_floor_index():
    """The caller can only act on a direction -- the stair machinery takes one --
    and the candidate list mixes visited floors with stair-implied ones, where an
    absolute index is easy to get wrong."""
    up = _planner(_StubClient({"Index": "2", "Reason": "-"}))
    assert up.decide("bed", _two_floors(), _sg(), step=500) == +1
    stay = _planner(_StubClient({"Index": "1", "Reason": "-"}))
    assert stay.decide("bed", _two_floors(), _sg(), step=500) == 0


def test_counts_only_decisions_that_move():
    p = _planner(_StubClient({"Index": "1", "Reason": "-"}))
    p.decide("bed", _two_floors(), _sg(), step=500)
    assert (p.asks, p.moves) == (1, 0), "naming the current floor is not a move"

    p2 = _planner(_StubClient({"Index": "2", "Reason": "-"}))
    p2.decide("bed", _two_floors(), _sg(), step=500)
    assert (p2.asks, p2.moves) == (1, 1)


@pytest.mark.parametrize("reply", [{"Index": "0"}, {"Index": "9"}, {"Index": "x"}, {}])
def test_bad_replies_mean_stay_here(reply):
    """None is 'keep exploring this floor'. Acting on a malformed answer would
    send the agent up a staircase on the strength of a parse error."""
    assert _planner(_StubClient(reply)).decide("bed", _two_floors(), _sg(), step=500) is None


def test_network_failure_does_not_raise():
    assert _planner(_StubClient(raises=True)).decide("bed", _two_floors(), _sg(), 500) is None


def test_counters_reset_per_episode():
    p = _planner(_StubClient({"Index": "2", "Reason": "-"}))
    p.decide("bed", _two_floors(), _sg(), step=500)
    p.reset()
    assert (p.asks, p.moves) == (0, 0)
    # And the throttle resets too, or the first ask of a new episode is skipped.
    p.decide("bed", _two_floors(), _sg(), step=200)
    assert p.asks == 1


# ------------------------------------------------------------------ the prompt


def test_floor_is_described_through_its_rooms():
    """The extension over ASCENT, which aggregates a storey into two flat sets of
    strings with no room-object containment (object_point_cloud_map.py:44-45)."""
    sg = _sg(rooms=[(1, "bedroom", 50, 0), (2, "bathroom", 30, 0)],
             objects=[("bed", 0), ("towel", 0)])
    text = describe_floor(sg, floor_key=0, order=0, is_current=True, explored=False)
    assert "bedroom" in text and "bathroom" in text
    assert "bed" in text and "towel" in text
    assert text.startswith("Current floor")


def test_explored_floors_are_marked_so_they_can_be_ruled_out():
    """ASCENT appends this per floor (llm_planner.py:538-541). Without it the
    model can only rank storeys, never eliminate one."""
    text = describe_floor(_sg(), 0, 0, False, explored=True)
    assert "do not need to explore this floor again" in text


def test_unmapped_floor_says_so_rather_than_looking_empty():
    text = describe_floor(_sg(), 1, 1, False, False)
    assert "unknown rooms" in text and "unknown objects" in text


def test_priors_reach_the_prompt_as_percentages():
    from osg.exploration.knowledge_prior import FloorPrior, KnowledgeGraph

    c = _StubClient()
    p = _planner(
        c,
        floor_prior=FloorPrior({"bed": {"2": {"1": 20.0, "2": 80.0}}}),
        kg=KnowledgeGraph({"bed": {"bedroom": 0.8, "living_room": 0.2}}),
    )
    p.decide("bed", _two_floors(), _sg(), step=500)
    body = c.user.split("Now answer question:")[1]
    assert '"Floor 2": 80.0%' in body
    assert '"Bedroom": 80.0%' in body


def test_prompt_survives_missing_priors():
    c = _StubClient()
    _planner(c).decide("aardvark", _two_floors(), _sg(), step=500)
    assert "Now answer question:" in c.user
    assert c.n == 1


# ------------------------------------------------ the direction boost in NavAgent


def _agent(goal_dir=0):
    from osg.agent.nav_agent import NavAgent
    from osg.exploration.async_scorer import AsyncScorer
    from osg.perception.detector import StubDetector

    from .test_nav_agent import _StubScorer, make_cfg

    cfg = make_cfg()
    cfg.exploration.floor_llm_boost = 5.0
    agent = NavAgent(cfg, StubDetector(), AsyncScorer(_StubScorer()), None, "bed")
    agent.floors = _two_floors()
    agent.floors._cur = 0  # standing on the lower floor
    agent._floor_goal_dir = goal_dir
    return agent


def test_no_decision_leaves_stair_scores_untouched():
    a = _agent(goal_dir=0)
    assert a._floor_direction_boost("up") == 1.0
    assert a._floor_direction_boost("down") == 1.0


def test_wanting_a_higher_floor_favours_up_and_damps_down():
    a = _agent(goal_dir=1)
    assert a._floor_direction_boost("up") == 5.0
    assert a._floor_direction_boost("down") == pytest.approx(0.2)


def test_the_wrong_direction_is_damped_not_vetoed():
    """The floor decision is a guess from a partial map. A hard veto would
    strand the agent when it is wrong and the only staircase leads the other
    way."""
    a = _agent(goal_dir=1)
    assert a._floor_direction_boost("down") > 0.0


def test_deciding_to_stay_is_neutral():
    a = _agent(goal_dir=0)
    assert a._floor_direction_boost("up") == 1.0
    assert a._floor_direction_boost("down") == 1.0
