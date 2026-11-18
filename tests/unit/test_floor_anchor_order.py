"""The order of the two systems: test the stale anchor, then choose a storey.

`_select_surface` aggregates container mass per floor and asks FloorPolicy for
the argmax. That mass is affinity x proximity over MAPPED SURFACES, so with a
flat proximity prior it reduces to "which storey has more furniture" -- a fact
about the prior map, available on the first selection round, and independent of
anything the agent has observed this episode.

Measured on outputs/osg_authored_15 (30 episodes, two storeys, presence and the
search posterior both on): the first cross-floor request came at step 13 in 17
of 30 episodes, including 10 of the 19 whose object never left the start floor.
405 requests produced 12 directed switch attempts and 3 arrivals on the goal
floor, while the presence filter spoke 4 times in 30 episodes. The agent was
leaving before it had been anywhere.

These tests pin the ordering: while a target-labelled track on this floor is
still believed AND has not been visited, the storey question is held.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from osg.agent.nav_agent import NavAgent
from osg.exploration.async_scorer import AsyncScorer
from osg.exploration.scorer import NullScorer
from osg.exploration.strategy import WorldView
from osg.graph.scene_graph import ContainerNode, FloorNode
from osg.perception.detector import StubDetector

from .conftest import make_frame
from .test_nav_agent import make_cfg

HERE, AWAY = 4, 9


def _track(tid, label, p, centre, floor_key, absence_arrivals=0):
    from osg.objects.association import ObjectTrack
    from osg.objects.ellipsoid import Ellipsoid

    track = ObjectTrack(
        id=tid, label=label,
        ellipsoid=Ellipsoid(center=np.array(centre, dtype=float),
                            axes=np.array([0.1, 0.1, 0.1]), R=np.eye(3)),
    )
    track.best_score, track.evidence = 0.9, 1.0
    track.best_bbox_px = 9000.0
    track.presence.log_odds = math.log(p / (1 - p))
    track.floor_key = floor_key
    track.absence_arrivals = absence_arrivals
    for _ in range(3):
        track.observations.append(None)
    return track


def _agent(anchor_gate: bool):
    cfg = make_cfg()
    cfg.exploration.search_posterior = True
    cfg.exploration.search_floor_requires_anchor_test = anchor_gate
    agent = NavAgent(cfg, StubDetector(), AsyncScorer(NullScorer()), None, "bowl")
    agent.exploration.planner = SimpleNamespace(plan=lambda *a: SimpleNamespace(success=False, cost=0.0))
    agent.exploration.viewpoint_planner = SimpleNamespace(
        approach_viewpoint=lambda *a: None
    )
    agent.scene_graph.floors = {HERE: FloorNode(HERE, 0.0), AWAY: FloorNode(AWAY, 2.7)}
    # The away storey carries more mapped surface mass, which is the whole of
    # the floor argmax's evidence.
    agent.scene_graph.containers = {
        1: ContainerNode(1, "table", [1], np.array([1.0, 0.75, 2.0]), 0.75, 0.6, floor=HERE),
        2: ContainerNode(2, "table", [2], np.array([1.0, 3.45, 2.0]), 3.45, 0.6, floor=AWAY),
        3: ContainerNode(3, "counter", [3], np.array([2.0, 3.45, 2.0]), 3.45, 0.6, floor=AWAY),
    }
    return agent


def _world(agent, intrinsics, floor_key=HERE):
    return WorldView(
        frame=make_frame(intrinsics, np.eye(4)),
        step=12,
        agent_xy=np.zeros(2),
        costmap=agent.costmap,
        scene_graph=agent.scene_graph,
        object_layer=agent.object_layer,
        keyframes=agent.keyframes,
        target="bowl",
        goal_xy=None,
        floor_id=floor_key,
    )


def _put(agent, track):
    agent.object_layer._tracks[track.id] = track


# ------------------------------------------------------------------ the gate

def test_a_believed_anchor_on_this_floor_holds_the_storey_request(intrinsics):
    """The defect this exists to prevent: leaving at step 12 while the prior map
    still says the bowl is on the floor the agent is standing on."""
    agent = _agent(anchor_gate=True)
    _put(agent, _track(7, "bowl", p=0.82, centre=(1.0, 0.8, 2.0), floor_key=HERE))
    world = _world(agent, intrinsics)

    agent.exploration._select_surface(world, None)

    assert agent.exploration.requested_floor is None
    assert agent.stats.get("cross_floor_request_held") == 1


def test_without_the_gate_the_same_state_leaves_immediately(intrinsics):
    """The shipped behaviour, unchanged when the flag is off."""
    agent = _agent(anchor_gate=False)
    _put(agent, _track(7, "bowl", p=0.82, centre=(1.0, 0.8, 2.0), floor_key=HERE))
    world = _world(agent, intrinsics)

    assert agent.exploration._select_surface(world, None) is None
    assert agent.exploration.requested_floor == AWAY
    assert "cross_floor_request_held" not in agent.stats


def test_an_arrival_that_found_nothing_releases_the_request(intrinsics):
    """The agent went to the mapped pose and the object was gone. That is the
    test the gate was waiting for, and it is the ARRIVAL, not the belief."""
    agent = _agent(anchor_gate=True)
    _put(agent, _track(7, "bowl", p=0.82, centre=(1.0, 0.8, 2.0), floor_key=HERE,
                       absence_arrivals=1))
    world = _world(agent, intrinsics)

    assert agent.exploration._select_surface(world, None) is None
    assert agent.exploration.requested_floor == AWAY


def test_a_belief_below_the_candidate_bar_releases_the_request(intrinsics):
    """Presence can also fall below the bar without a dedicated arrival, e.g.
    from repeated expected misses. Candidates stop being proposed at
    `min_presence`, and the floor gate uses the same number so the two cannot
    disagree about one track."""
    agent = _agent(anchor_gate=True)
    _put(agent, _track(7, "bowl", p=0.10, centre=(1.0, 0.8, 2.0), floor_key=HERE))
    world = _world(agent, intrinsics)

    assert agent.exploration._select_surface(world, None) is None
    assert agent.exploration.requested_floor == AWAY


def test_an_anchor_on_the_other_floor_does_not_hold_anything(intrinsics):
    """Only a believed anchor HERE is a reason to stay here."""
    agent = _agent(anchor_gate=True)
    _put(agent, _track(7, "bowl", p=0.82, centre=(1.0, 3.5, 2.0), floor_key=AWAY))
    world = _world(agent, intrinsics)

    assert agent.exploration._select_surface(world, None) is None
    assert agent.exploration.requested_floor == AWAY


def test_nothing_target_labelled_mapped_leaves_the_argmax_alone(intrinsics):
    """With no anchor at all the storey argmax is the only opinion available,
    and it should stand -- this is the never-seen-it case, not the stale one."""
    agent = _agent(anchor_gate=True)
    _put(agent, _track(7, "chair", p=0.82, centre=(1.0, 0.8, 2.0), floor_key=HERE))
    world = _world(agent, intrinsics)

    assert agent.exploration._select_surface(world, None) is None
    assert agent.exploration.requested_floor == AWAY


def test_holding_the_request_still_searches_this_floor(intrinsics):
    """A held round must not be a wasted round: it falls through to same-floor
    selection, so the agent goes to the surface the map points at."""
    agent = _agent(anchor_gate=True)
    _put(agent, _track(7, "bowl", p=0.82, centre=(1.0, 0.8, 2.0), floor_key=HERE))
    world = _world(agent, intrinsics)
    planned = []
    agent.exploration.planner = SimpleNamespace(
        plan=lambda costmap, start, goal: planned.append(np.asarray(goal))
        or SimpleNamespace(success=True, cost=2.0)
    )
    agent.exploration.viewpoint_planner = SimpleNamespace(
        approach_viewpoint=lambda goal, costmap: goal
    )

    surface = agent.exploration._select_surface(world, None)

    assert surface is not None, "the held round produced no same-floor surface"
    assert surface.ref_id == 1, "planned for a surface that is not on this floor"
    assert planned, "no same-floor path was ever costed"


# -------------------------------------- a request that cannot be acted on

def test_an_unreachable_storey_still_leaves_this_floor_searchable(intrinsics):
    """`try_switch` finds no portal 97% of the time it is asked. While the
    request stands, the container posterior must not be off on the floor the
    agent is actually standing on."""
    agent = _agent(anchor_gate=False)
    agent.cfg.exploration.search_surface_when_floor_unreachable = True
    world = _world(agent, intrinsics)
    agent.exploration.planner = SimpleNamespace(
        plan=lambda costmap, start, goal: SimpleNamespace(success=True, cost=2.0)
    )
    agent.exploration.viewpoint_planner = SimpleNamespace(
        approach_viewpoint=lambda goal, costmap: goal
    )

    # The storey is wanted and no portal exists: the switch callback says False.
    choice = agent.exploration.select(world, lambda cost, floor=None: False)

    assert agent.stats.get("floor_unreachable_fallback", 0) >= 1
    assert choice is not None and choice.kind == "surface"
    assert choice.container_id == 1, "fell back to a surface off this floor"


def test_the_fallback_is_off_by_default(intrinsics):
    """Without the flag the standing request still suppresses the search, which
    is the shipped behaviour."""
    agent = _agent(anchor_gate=False)
    world = _world(agent, intrinsics)
    agent.exploration.planner = SimpleNamespace(
        plan=lambda costmap, start, goal: SimpleNamespace(success=True, cost=2.0)
    )
    agent.exploration.viewpoint_planner = SimpleNamespace(
        approach_viewpoint=lambda goal, costmap: goal
    )

    agent.exploration.select(world, lambda cost, floor=None: False)

    assert "floor_unreachable_fallback" not in agent.stats


def test_a_successful_switch_never_reaches_the_fallback(intrinsics):
    """When the agent really is leaving, this floor's surfaces are not the
    question and the round must end."""
    agent = _agent(anchor_gate=False)
    agent.cfg.exploration.search_surface_when_floor_unreachable = True
    world = _world(agent, intrinsics)

    assert agent.exploration.select(world, lambda cost, floor=None: True) is None
    assert "floor_unreachable_fallback" not in agent.stats


# ------------------------------------------- what a storey's mass should mean

def _two_floor_world(agent, intrinsics, n_here, n_there):
    """A ground floor with many surfaces and an upper floor with few, all of
    equal quality -- the shape 00808's prior map actually has."""
    from osg.graph.scene_graph import ContainerNode
    containers = {}
    cid = 1
    for _ in range(n_here):
        containers[cid] = ContainerNode(cid, "table", [cid],
                                        np.array([1.0 + cid * 0.5, 0.75, 2.0]),
                                        0.75, 0.6, floor=HERE)
        cid += 1
    for _ in range(n_there):
        containers[cid] = ContainerNode(cid, "table", [cid],
                                        np.array([1.0 + cid * 0.5, 3.45, 2.0]),
                                        3.45, 0.6, floor=AWAY)
        cid += 1
    agent.scene_graph.containers = containers
    return _world(agent, intrinsics)


def test_summing_mass_prefers_whichever_floor_has_more_furniture(intrinsics):
    """The shipped rule. Both floors hold identical tables; one just has more."""
    agent = _agent(anchor_gate=False)
    world = _two_floor_world(agent, intrinsics, n_here=3, n_there=9)
    agent.exploration._select_surface(world, None)
    assert agent.exploration.requested_floor == AWAY


def test_the_mean_sees_two_floors_of_equal_quality_as_equal(intrinsics):
    """Same map, scored scale-free: identical tables are identical evidence
    however many of them a floor happens to hold, so there is no request."""
    agent = _agent(anchor_gate=False)
    agent.cfg.exploration.floor_mass_rule = "mean"
    world = _two_floor_world(agent, intrinsics, n_here=3, n_there=9)
    agent.exploration._select_surface(world, None)
    assert agent.exploration.requested_floor is None


def test_the_margin_refuses_a_switch_the_rule_would_have_made(intrinsics):
    """Nine surfaces against three is a 3x sum, and still not worth the stairs
    if the bar is set at 5x. The counter names the decision."""
    agent = _agent(anchor_gate=False)
    agent.cfg.exploration.floor_mass_margin = 5.0
    world = _two_floor_world(agent, intrinsics, n_here=3, n_there=9)
    agent.exploration._select_surface(world, None)
    assert agent.exploration.requested_floor is None
    assert agent.stats.get("floor_mass_no_opinion", 0) >= 1


def test_a_decisively_better_storey_still_wins(intrinsics):
    """The margin must not weld the agent to its floor: with nothing at all
    here, anywhere else is better."""
    agent = _agent(anchor_gate=False)
    agent.cfg.exploration.floor_mass_rule = "mean"
    agent.cfg.exploration.floor_mass_margin = 1.15
    world = _two_floor_world(agent, intrinsics, n_here=0, n_there=9)
    agent.exploration._select_surface(world, None)
    assert agent.exploration.requested_floor == AWAY


def test_the_rule_and_margin_are_off_by_default(intrinsics):
    agent = _agent(anchor_gate=False)
    assert agent.cfg.exploration.floor_mass_rule == "sum"
    assert agent.cfg.exploration.floor_mass_margin == 0.0
    world = _two_floor_world(agent, intrinsics, n_here=3, n_there=9)
    agent.exploration._select_surface(world, None)
    assert agent.exploration.requested_floor == AWAY
    assert "floor_mass_no_opinion" not in agent.stats
