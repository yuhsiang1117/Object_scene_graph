"""The floor LLM, actually connected to something.

`FloorDecisionPlanner.decide` had no caller outside its own tests;
`_floor_goal_dir` was assigned 0 at construction and never again; and
`_floor_direction_boost` had no production caller at all. Its unit tests passed
because they set `_floor_goal_dir` by hand. So four presets claiming
`floor_llm: true` -- floor_llm, s71, ascent_llm, ascent_aligned -- measured
nothing in this agent.

These tests are about the WIRING, not the prompt: that a directed storey request
reaches the model, that its answer changes which floor the agent heads for, and
that every failure leaves the posterior's own answer standing.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from osg.agent.nav_agent import NavAgent
from osg.exploration.async_scorer import AsyncScorer
from osg.exploration.scorer import NullScorer
from osg.perception.detector import StubDetector

from .test_nav_agent import make_cfg

GROUND, UPPER = 0, 1


class _Planner:
    """Stands in for FloorDecisionPlanner: records the call, returns a direction."""

    def __init__(self, direction):
        self.direction = direction
        self.calls = []
        self.asks = 0
        self.moves = 0

    def decide(self, target, floors, sg, step, has_up=False, has_down=False):
        self.calls.append((target, step))
        self.asks += 1
        if isinstance(self.direction, Exception):
            raise self.direction
        if self.direction:
            self.moves += 1
        return self.direction


def _agent(direction, floor_llm=True):
    cfg = make_cfg()
    cfg.exploration.floor_llm = floor_llm
    agent = NavAgent(cfg, StubDetector(), AsyncScorer(NullScorer()), None, "bowl")
    agent.floor_planner = _Planner(direction)
    stack = agent.floors.stack
    stack.layer(GROUND, step=0).floor_y = 0.0
    stack.layer(UPPER, step=0).floor_y = 2.9
    stack.current_id = GROUND
    return agent


def test_a_directed_request_reaches_the_model():
    agent = _agent(direction=+1)
    assert agent._llm_floor_choice(UPPER) == UPPER
    assert agent.floor_planner.calls, "decide() was never called"
    assert agent.stats["floor_llm_asks"] == 1
    assert agent.stats.get("floor_llm_agreed") == 1


def test_the_model_can_override_the_posterior():
    """The whole point: a storey chosen by what the rooms are, not by how much
    furniture each floor happens to have."""
    agent = _agent(direction=-1)
    agent.floors.stack.current_id = UPPER
    assert agent._llm_floor_choice(UPPER) == GROUND
    assert agent.stats.get("floor_llm_override") == 1


def test_the_model_may_say_stay():
    agent = _agent(direction=0)
    assert agent._llm_floor_choice(UPPER) is False
    assert agent.stats.get("floor_llm_stay") == 1


def test_a_failed_call_leaves_the_posterior_answer_standing():
    agent = _agent(direction=None)
    assert agent._llm_floor_choice(UPPER) == UPPER
    assert agent.stats.get("floor_llm_no_answer") == 1


def test_an_undirected_round_never_asks():
    """No request means no question. The model is for choosing between storeys,
    not for deciding whether to leave one."""
    agent = _agent(direction=+1)
    assert agent._llm_floor_choice(None) is None
    assert not agent.floor_planner.calls


def test_the_flag_off_never_asks():
    agent = _agent(direction=+1, floor_llm=False)
    assert agent._llm_floor_choice(UPPER) == UPPER
    assert not agent.floor_planner.calls


def test_a_storey_the_stack_does_not_have_is_not_aimed_at():
    """`up()` returns None on the top floor. The request must survive."""
    agent = _agent(direction=+1)
    agent.floors.stack.current_id = UPPER
    assert agent._llm_floor_choice(GROUND) == GROUND
    assert agent.stats.get("floor_llm_no_such_floor") == 1


def test_the_direction_is_recorded_for_the_stair_boost():
    """`_floor_goal_dir` is what `_floor_direction_boost` reads, and it was
    never assigned. An arm that cannot be seen in a counter cannot be judged."""
    agent = _agent(direction=+1)
    assert agent._floor_goal_dir == 0
    agent._llm_floor_choice(UPPER)
    assert agent._floor_goal_dir == +1
    assert agent._floor_direction_boost("up") > agent._floor_direction_boost("down")
