"""A storey that will not let the agent leave is a storey to search.

Measured (outputs/mf5_pass2_v18, the cracker box): the agent climbed to the
CORRECT storey at step 161, searched three containers, and from step 333 to
841 made six descent attempts, every one ending in
`portal_end_no_vertical_progress`. That is 508 steps, half the episode, spent
trying to leave the storey the target was on -- while selecting a frontier
three times in the whole 1000 steps.
"""
from types import SimpleNamespace

import numpy as np

from osg.agent.nav_agent import NavAgent


def _agent(limit, failures, ban_steps=0, step=100):
    return SimpleNamespace(
        cfg=SimpleNamespace(agent=SimpleNamespace(
            max_failed_switches_per_storey=limit, switch_ban_steps=ban_steps)),
        stats={},
        step_count=step,
        floors=SimpleNamespace(current_id=1),
        exploration=SimpleNamespace(forced_floor=3, requested_floor=3),
        _failed_switches_by_floor=dict(failures),
        _switch_banned_at={},
    )


def _note(agent):
    NavAgent._note_failed_switch(agent)


def _banned(agent):
    return NavAgent._switches_exhausted_here(agent)


def test_failures_are_counted_per_storey():
    agent = _agent(2, {})
    _note(agent); _note(agent)
    assert agent._failed_switches_by_floor == {1: 2}
    agent.floors.current_id = 0
    _note(agent)
    assert agent._failed_switches_by_floor == {1: 2, 0: 1}


def test_under_the_limit_the_switch_is_still_offered():
    """The gate must not fire before the storey has actually refused."""
    agent = _agent(2, {1: 1})
    assert _banned(agent) is False
    assert agent.exploration.forced_floor == 3, "nothing should have been cleared yet"


def test_at_the_limit_the_request_is_dropped_too():
    """Banning the switch is not enough: the posterior keeps asking, so the
    standing request has to be cleared or every round re-raises it."""
    agent = _agent(2, {1: 2})
    assert _banned(agent) is True
    assert agent.stats["floor_switch_banned_after_failures"] == 1
    assert agent.exploration.forced_floor is None
    assert agent.exploration.requested_floor is None


def test_another_storey_is_unaffected():
    """The ban is per storey: the agent that cannot leave the upper one must
    still be able to leave the lower one."""
    agent = _agent(2, {1: 2})
    assert _banned(agent) is True
    agent.floors.current_id = 0
    assert _banned(agent) is False


def test_zero_means_keep_asking():
    agent = _agent(0, {1: 9})
    assert _banned(agent) is False
    assert agent.exploration.forced_floor == 3


def test_the_ban_is_served_not_permanent():
    """A permanent ban is too absolute: in outputs/mf5_pass2_v19 the cracker
    box failed twice early, hit the limit, and could then never reach the
    target storey at all -- banned 16 times, `goal_floor_reached` false."""
    agent = _agent(2, {1: 2}, ban_steps=250, step=100)
    assert _banned(agent) is True
    assert agent._switch_banned_at[1] == 100

    agent.step_count = 300                     # 200 steps later, still serving
    assert _banned(agent) is True

    agent.step_count = 351                     # 251 steps later
    assert _banned(agent) is False, "the ban never lifted"
    assert agent.stats["floor_switch_ban_lifted"] == 1
    assert agent._failed_switches_by_floor[1] == 0, "the count must start again"
