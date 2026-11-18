"""Two decisions the cross-anchor episodes needed and did not get.

Measured on 00800 cross_anchor_01 (outputs/mf5_pass2_v6), toy airplane, agent
on the upper storey, target one storey down and never once in view:

  * three attempts went to three "toy airplane" tracks on the upper storey --
    a ceiling fixture and its kin, detector score up to 0.82, 0 presence
    events for the label in 200 steps because a false positive IS present;
  * `floor_target_evidence` added _TARGET_PRESENT for those believed fakes,
    and `may_switch` refused to leave "a floor that has the thing on it";
  * the one switch that got through (step 172, via the prior-stairs override)
    was abandoned nine steps later when another same-floor track committed.

`floor_disproved_after_failed_attempts` turns failed attempts into evidence
about the STOREY: at N the storey's target tracks stop vetoing, and the nearest
other storey is requested directly. `protect_floor_switch` keeps a switch that
has been decided from being pre-empted by anything but a live, close, confident
candidate. Both default off.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from osg.agent.nav_agent import State
from osg.objects.association import ObjectTrack
from osg.objects.ellipsoid import Ellipsoid

from .test_nav_agent import _frame_at, make_agent, make_cfg


def _two_storeys(agent, current=0):
    agent.floors.estimator._levels = {0: 0.0, 1: 2.9}
    agent.floors.estimator.current = current
    agent.floors.stack.layer(0, step=0).floor_y = 0.0
    agent.floors.stack.layer(1, step=0).floor_y = 2.9
    agent.floors.stack.current_id = current


def _fail_attempts(agent, n):
    for _ in range(n):
        agent.rearm(agent.cfg.agent.max_steps)


# ------------------------------------------------------- storey disproved

def test_off_by_default_nothing_is_disproved():
    agent = make_agent()
    _two_storeys(agent)
    assert agent.cfg.agent.floor_disproved_after_failed_attempts == 0
    _fail_attempts(agent, 3)
    assert not agent._storey_disproved(0)
    assert agent.exploration.forced_floor is None
    assert agent.exploration.requested_floor is None


def test_two_failed_attempts_disprove_the_storey_and_request_the_other():
    agent = make_agent(make_cfg(floor_disproved_after_failed_attempts=2))
    _two_storeys(agent, current=0)
    _fail_attempts(agent, 1)
    assert not agent._storey_disproved(0), "one failure is not a verdict"
    _fail_attempts(agent, 1)
    assert agent._storey_disproved(0)
    assert agent.exploration.forced_floor == 1
    assert agent.exploration.requested_floor == 1
    assert agent.stats["floors_disproved"] == 1
    assert agent.stats["floor_disproved_to"] == 1


def test_the_nearest_other_storey_is_chosen():
    agent = make_agent(make_cfg(floor_disproved_after_failed_attempts=1))
    agent.floors.estimator._levels = {0: 0.0, 1: 2.9, 2: 6.0}
    agent.floors.estimator.current = 2
    for key, h in ((0, 0.0), (1, 2.9), (2, 6.0)):
        agent.floors.stack.layer(key, step=0).floor_y = h
    agent.floors.stack.current_id = 2
    _fail_attempts(agent, 1)
    assert agent.exploration.forced_floor == 1, "6.0 -> 2.9 is nearer than 6.0 -> 0.0"


def test_a_disproved_storey_is_never_requested_again():
    agent = make_agent(make_cfg(floor_disproved_after_failed_attempts=1))
    _two_storeys(agent, current=0)
    _fail_attempts(agent, 1)                       # 0 disproved -> request 1
    agent.floors.stack.current_id = 1
    agent.floors.estimator.current = 1
    _fail_attempts(agent, 1)                       # 1 disproved -> nowhere left
    assert agent._storey_disproved(1)
    assert agent.stats["floor_disproved_no_other"] == 1
    assert agent.exploration.forced_floor == 1, "left as it was; nothing better to say"


def test_one_known_storey_has_nowhere_to_send_the_agent():
    agent = make_agent(make_cfg(floor_disproved_after_failed_attempts=1))
    _fail_attempts(agent, 1)
    assert agent.stats.get("floor_disproved_no_other") == 1
    assert agent.exploration.forced_floor is None


def test_the_veto_is_lifted_on_a_disproved_storey_only():
    """`floor_target_evidence` gates _TARGET_PRESENT on `presence_of`; on a
    disproved storey every same-label track must read as not believed."""
    agent = make_agent(make_cfg(floor_disproved_after_failed_attempts=1))
    agent.cfg.exploration.floor_evidence_by_presence = True
    _two_storeys(agent, current=0)
    from osg.agent.nav_agent import NavAgent

    before = agent._presence_for_floor_evidence()
    assert getattr(before, "__func__", None) is NavAgent._track_still_believed
    _fail_attempts(agent, 1)
    lifted = agent._presence_for_floor_evidence()
    assert getattr(lifted, "__func__", None) is not NavAgent._track_still_believed
    assert lifted(12345) is False
    assert agent.stats["floor_veto_lifted"] == 1


def test_the_counts_do_not_survive_the_episode():
    agent = make_agent(make_cfg(floor_disproved_after_failed_attempts=1))
    _two_storeys(agent)
    _fail_attempts(agent, 1)
    assert agent._storey_disproved(0)
    agent.reset("chair")
    assert not agent._storey_disproved(0)
    assert agent._failed_attempts_by_floor == {}


def test_select_surface_honours_the_forced_storey(intrinsics):
    """The posterior abstains by design (`mean` + margin) and the anchor hold
    holds; a storey disproved by failed attempts outranks both."""
    from .test_cross_floor_search import _agent, _world

    agent = _agent()
    world = _world(agent, intrinsics, floor_key=9)     # standing upstairs
    agent.exploration.forced_floor = 4
    assert agent.exploration._select_surface(world, None) is None
    assert agent.exploration.requested_floor == 4
    assert agent.exploration.stats["floor_forced_by_failed_attempts"] == 1


# ------------------------------------------------ floor switch protection

def _arm_agent(**agent_overrides):
    """An agent with the cross-anchor arm's CANDIDATE gates.

    The shipped `verification.min_score` is 0.7, above the protection's 0.6
    bar, so at plain defaults a "weak" candidate cannot exist -- it is filtered
    before the protection ever sees it. The arm this mechanism is measured on
    (`mf5_osg_on_ascent_map`) admits at 0.3 / one observation / evidence 0.2
    and bypasses the size gate, and those are the gates the fixture must clear.
    """
    cfg = make_cfg(**agent_overrides)
    cfg.verification.min_obs = 1
    cfg.verification.min_score = 0.3
    cfg.verification.min_evidence = 0.2
    cfg.verification.target_bypasses_bbox_gate = True
    return make_agent(cfg)


def _weak_score(agent) -> float:
    """Clears the arm's candidate score gate, fails the protection's."""
    floor = float(agent.cfg.verification.min_score)
    ceiling = float(agent.cfg.agent.protect_floor_switch_min_score)
    assert floor < ceiling, "the test needs room between the two bars"
    return floor + 0.5 * (ceiling - floor)


def _candidate(agent, xy, score, live=True, track_id=7):
    """A track that clears every candidate gate, at `xy`, seen live or not."""
    track = ObjectTrack(
        id=track_id, label=agent.target,
        ellipsoid=Ellipsoid(center=np.array([xy[0], 0.8, xy[1]]),
                            axes=np.array([0.1, 0.1, 0.1]), R=np.eye(3)),
    )
    track.observations.append(SimpleNamespace(frame_id=3 if live else -1))
    if "n_obs" in ObjectTrack.__dataclass_fields__:
        track.n_obs = 1
    track.best_score = score
    track.best_bbox_px = 50_000.0            # the size gate, if it is on
    track.evidence = max(1.0, float(agent.cfg.verification.min_evidence))
    track.presence.log_odds = 1.5
    track.floor_key = int(agent.floors.current_id)
    agent.object_layer._tracks[track.id] = track
    return track


def _act_once(agent):
    agent.state = State.EXPLORE
    agent.act(_frame_at((0.0, 0.0)))


def test_off_by_default_any_candidate_preempts_a_pursuit():
    agent = _arm_agent()
    agent.floors.pursuing = True
    far_weak = _candidate(agent, (6.0, 6.0), score=_weak_score(agent))
    _act_once(agent)
    assert agent._candidate_id == far_weak.id


def test_a_far_or_weak_candidate_cannot_preempt_a_pursuit():
    agent = _arm_agent(protect_floor_switch=True)
    agent.floors.pursuing = True
    far_weak = _candidate(agent, (6.0, 6.0), score=_weak_score(agent))
    _act_once(agent)
    assert agent._candidate_id is None
    assert agent.stats["pursuit_preempt_blocked"] == 1
    assert not far_weak.blacklisted, "not a verdict: the track is untouched"


def test_a_live_close_confident_candidate_still_preempts():
    agent = _arm_agent(protect_floor_switch=True)
    agent.floors.pursuing = True
    near_strong = _candidate(agent, (0.8, 0.0), score=0.9)
    _act_once(agent)
    assert agent._candidate_id == near_strong.id
    assert agent.stats["pursuit_preempt_allowed"] == 1


def test_a_prior_track_cannot_preempt_even_if_close():
    """`seen_live` is the third condition: a snapshot track at a stale pose is
    exactly the thing a floor switch is leaving behind."""
    agent = _arm_agent(protect_floor_switch=True)
    agent.floors.pursuing = True
    _candidate(agent, (0.8, 0.0), score=0.9, live=False)
    _act_once(agent)
    assert agent._candidate_id is None


def test_without_a_pursuit_the_protection_is_inert():
    agent = _arm_agent(protect_floor_switch=True)
    agent.floors.pursuing = False
    far_weak = _candidate(agent, (6.0, 6.0), score=_weak_score(agent))
    _act_once(agent)
    assert agent._candidate_id == far_weak.id
    assert "pursuit_preempt_blocked" not in agent.stats


# ---------------------------------- the verdict is honoured downstream

def test_no_candidate_preempts_a_switch_away_from_a_disproved_storey():
    """v7 ep1: the ceiling-fixture false positive (score 0.82) passed the
    live/close/confident test as the agent walked under it toward the stairs,
    and took the last attempt. On a disproved storey nothing pre-empts."""
    agent = _arm_agent(protect_floor_switch=True, floor_disproved_after_failed_attempts=1)
    _two_storeys(agent, current=0)
    _fail_attempts(agent, 1)                       # storey 0 disproved
    agent.floors.pursuing = True
    near_strong = _candidate(agent, (0.8, 0.0), score=0.9)
    _act_once(agent)
    assert agent._candidate_id is None
    # The verdict is now applied before the pursuit test is reached: on a
    # disproved storey the check is skipped outright, pursuit or not.
    assert agent.stats["candidates_skipped_disproved"] == 1
    assert not near_strong.blacklisted


def test_the_floor_llm_is_not_asked_to_overrule_a_disproved_storey():
    """v7 ep1 step 158: asked, the model said stay because the scene graph
    "explicitly lists 'toy airplane'" on this floor -- the false positives."""
    agent = make_agent(make_cfg(floor_disproved_after_failed_attempts=1))
    agent.cfg.exploration.floor_llm = True
    _two_storeys(agent, current=0)

    class _Planner:
        asks = 0
        def decide(self, *_a, **_k):
            self.asks += 1
            return 0                          # "stay"

    agent.floor_planner = _Planner()
    _fail_attempts(agent, 1)
    assert agent._llm_floor_choice(1) == 1, "the request stands"
    assert agent.floor_planner.asks == 0, "the model was not consulted"
    assert agent.stats["floor_llm_skipped_disproved"] == 1


def test_the_floor_llm_is_still_asked_on_a_storey_not_disproved():
    agent = make_agent(make_cfg(floor_disproved_after_failed_attempts=1))
    agent.cfg.exploration.floor_llm = True
    _two_storeys(agent, current=0)

    class _Planner:
        asks = moves = blocked_throttle = blocked_too_soon = blocked_one_floor = 0
        last_reason = ""
        def decide(self, *_a, **_k):
            self.asks += 1
            return 0

    agent.floor_planner = _Planner()
    assert agent._llm_floor_choice(1) is False, "stay, as the model said"
    assert agent.floor_planner.asks == 1


def test_no_candidate_commits_on_a_disproved_storey_even_without_a_pursuit():
    """v11 ep1: the climb ended on budget, `pursuing` went False, and the next
    step committed to an upper-storey fake from the stairs. The verdict is
    about the storey, not about whether a switch happens to be in flight."""
    agent = _arm_agent(floor_disproved_after_failed_attempts=1)
    _two_storeys(agent, current=0)
    _fail_attempts(agent, 1)                       # storey 0 disproved
    agent.floors.pursuing = False
    near_strong = _candidate(agent, (0.8, 0.0), score=0.9)
    _act_once(agent)
    assert agent._candidate_id is None
    assert agent.stats["candidates_skipped_disproved"] == 1
    assert not near_strong.blacklisted


def test_candidates_resume_on_a_storey_that_is_not_disproved():
    agent = _arm_agent(floor_disproved_after_failed_attempts=1)
    _two_storeys(agent, current=0)
    _fail_attempts(agent, 1)                       # storey 0 disproved, 1 is not
    agent.floors.stack.current_id = 1
    agent.floors.estimator.current = 1
    track = _candidate(agent, (0.8, 0.0), score=0.9)
    _act_once(agent)
    assert agent._candidate_id == track.id


def test_the_posterior_never_selects_a_disproved_storey(intrinsics):
    """v12 ep1: disproved the upper storey, climbed down, and the next round
    sent it straight back up."""
    from .test_cross_floor_search import _agent, _world

    agent = _agent()
    world = _world(agent, intrinsics, floor_key=4)          # downstairs now
    agent.exploration.disproved_floors.add(9)               # upstairs is disproved
    # With upstairs excluded the round plans on THIS storey, which the
    # cross-floor fixture forbids on purpose; allow it, with nothing reachable.
    agent.exploration.viewpoint_planner = SimpleNamespace(approach_viewpoint=lambda *a: None)
    agent.exploration.planner = SimpleNamespace(plan=lambda *a: SimpleNamespace(success=False, cost=None))
    agent.exploration._select_surface(world, None)
    assert agent.exploration.requested_floor != 9
    assert agent.exploration.stats.get("floor_posterior_disproved_excluded") == 1


def test_the_agent_and_the_strategy_share_one_verdict():
    agent = make_agent(make_cfg(floor_disproved_after_failed_attempts=1))
    _two_storeys(agent, current=0)
    _fail_attempts(agent, 1)
    assert 0 in agent.exploration.disproved_floors
    agent.reset("chair")
    assert not agent.exploration.disproved_floors, "the shared set is cleared with the episode"


def test_an_empty_storey_is_unmapped_not_disproved(intrinsics):
    """v12 ep1 step 390: just arrived downstairs, no containers here yet,
    `floor_mass={'0': 11.8}` -- and the round requested the storey it had
    just left. An empty storey is somewhere to explore, not somewhere to leave."""
    from osg.graph.scene_graph import ContainerNode, FloorNode
    from .test_cross_floor_search import _agent, _world

    agent = _agent()
    # only the UPPER storey (9) has containers; the agent stands on 4
    agent.scene_graph.containers = {
        2: ContainerNode(2, "table", [2], np.array([1.0, 3.45, 2.0]), 3.45, 0.6, floor=9),
        3: ContainerNode(3, "counter", [3], np.array([2.0, 3.45, 2.0]), 3.45, 0.6, floor=9),
    }
    world = _world(agent, intrinsics, floor_key=4)          # step 12, arrived at 0
    agent.cfg.exploration.empty_storey_settle_steps = 60
    agent.exploration.viewpoint_planner = SimpleNamespace(approach_viewpoint=lambda *a: None)
    agent.exploration.planner = SimpleNamespace(plan=lambda *a: SimpleNamespace(success=False, cost=None))
    agent.exploration._select_surface(world, None)
    assert agent.exploration.requested_floor is None, "an empty storey must not trigger a switch"
    assert agent.exploration.stats["floor_posterior_empty_here"] == 1


def test_after_the_settle_window_an_empty_storey_is_left_as_before(intrinsics):
    """The shipped rule (test_floor_anchor_order): with nothing at all here,
    anywhere else is better -- once the agent has actually been here."""
    from osg.graph.scene_graph import ContainerNode
    from .test_cross_floor_search import _agent, _world

    agent = _agent()
    agent.scene_graph.containers = {
        2: ContainerNode(2, "table", [2], np.array([1.0, 3.45, 2.0]), 3.45, 0.6, floor=9),
    }
    agent.cfg.exploration.empty_storey_settle_steps = 10
    world = _world(agent, intrinsics, floor_key=4)          # step 12 > 10
    agent.exploration._select_surface(world, None)
    assert agent.exploration.requested_floor == 9


def test_when_every_storey_with_mass_is_disproved_the_agent_stays(intrinsics):
    from osg.graph.scene_graph import ContainerNode
    from .test_cross_floor_search import _agent, _world

    agent = _agent()
    agent.scene_graph.containers = {
        2: ContainerNode(2, "table", [2], np.array([1.0, 3.45, 2.0]), 3.45, 0.6, floor=9),
    }
    agent.exploration.disproved_floors.add(9)
    world = _world(agent, intrinsics, floor_key=4)
    agent.exploration.viewpoint_planner = SimpleNamespace(approach_viewpoint=lambda *a: None)
    agent.exploration.planner = SimpleNamespace(plan=lambda *a: SimpleNamespace(success=False, cost=None))
    agent.exploration._select_surface(world, None)
    assert agent.exploration.requested_floor is None
