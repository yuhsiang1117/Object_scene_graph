"""A failed attempt must buy one real exploration round, not just EXPLORE.

`rearm` has always set `state = State.EXPLORE`, which reads like "the agent
goes back to exploring". It does not follow. `_act_inner` runs
`candidates.check` BEFORE the EXPLORE branch on every step, so the next
same-label track commits immediately and the agent is back in APPROACH within
the same step -- and the exploration ROUND, which is the only call site of
`floor_switch` and therefore of `_try_floor_switch`, is separately rate-limited
to one run per `exploration.select_every` steps.

Measured on 00800 cross_anchor_01 (outputs/mf5_pass2_v2): attempt 1 fails at
step 70, the next commit lands at step 74, and those four EXPLORE steps sit
inside the 5-step rate limit. `frontier_select_log` and `search_log_events` are
EMPTY across 180 and 362 steps; `flights_seen` and `portals_seen` never appear.
All three attempts went to same-label tracks on the starting storey while the
goal was on the other one.

`agent.explore_after_failed_attempt_steps` holds the commit open for a window
and forces the round to run now. Default 0 keeps every measured arm unchanged.
"""
from __future__ import annotations

from osg.agent.nav_agent import State

from .test_nav_agent import _frame_at, make_agent, make_cfg


def _count_candidate_checks(agent):
    """Replace the candidate check with a counter, keeping its signature."""
    calls = []
    agent.candidates.check = lambda *a, **k: calls.append(agent.step_count)
    return calls


def test_the_hold_is_off_by_default():
    """Every arm measured before this existed must be unaffected."""
    agent = make_agent()
    assert agent.cfg.agent.explore_after_failed_attempt_steps == 0
    agent.rearm(agent.cfg.agent.max_steps)
    assert agent._explore_hold_until == 0
    calls = _count_candidate_checks(agent)
    agent.state = State.EXPLORE
    agent.act(_frame_at((0.0, 0.0)))
    assert calls, "the shipped behaviour checks candidates on the next step"


def test_the_hold_suppresses_the_commit_for_its_window_and_then_stops():
    agent = make_agent(make_cfg(explore_after_failed_attempt_steps=3))
    agent.rearm(agent.cfg.agent.max_steps)
    assert agent._explore_hold_until == agent.step_count + 3
    calls = _count_candidate_checks(agent)
    for _ in range(3):
        agent.state = State.EXPLORE
        agent.act(_frame_at((0.0, 0.0)))
    assert not calls, "a commit pre-empted the round the hold exists to allow"
    assert agent.stats["explore_hold_steps"] == 3
    agent.state = State.EXPLORE
    agent.act(_frame_at((0.0, 0.0)))
    assert calls, "the hold is a hold, not a ban -- candidates must resume"


def test_the_hold_drops_the_selection_rate_limit():
    """The window is only worth holding if a round actually runs inside it.
    `select` fires once per `select_every` steps, and the gap between a failed
    attempt and the next commit is shorter than that."""
    agent = make_agent(make_cfg(explore_after_failed_attempt_steps=10))
    agent.exploration._last_select_step = 10_000  # a round just ran
    agent.rearm(agent.cfg.agent.max_steps)
    assert agent.exploration._last_select_step < 0, (
        "the next select is still rate-limited, so the hold buys nothing"
    )


def test_the_hold_does_not_survive_the_episode():
    agent = make_agent(make_cfg(explore_after_failed_attempt_steps=25))
    agent.rearm(agent.cfg.agent.max_steps)
    assert agent._explore_hold_until > 0
    agent.reset("chair")
    assert agent._explore_hold_until == 0
