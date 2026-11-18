"""The close look (agent/close_look.py): the frame a decision rests on.

Two entry points, each behind its own flag, and the invariants an A/B needs:
off by default and inert; the look before absence runs before the sensor and
re-aims on a detection; the opportunistic look fires on an affording surface
in view, once per surface, under a per-episode cap; the hold ends on the
heading it started from; a commit pre-empting a look counts as a detection.
"""
from __future__ import annotations

import numpy as np

from osg.agent.close_look import HOLD_PATTERN
from osg.agent.nav_agent import State
from osg.exploration.strategy import container_in_view
from osg.graph.scene_graph import ContainerNode
from osg.objects.association import ObjectTrack
from osg.objects.ellipsoid import Ellipsoid
from osg.planning.controller import TURN_LEFT, TURN_RIGHT

from .test_nav_agent import _det, _frame, make_agent, make_cfg


def _track(tid, centre, label="chair"):
    return ObjectTrack(
        id=tid, label=label,
        ellipsoid=Ellipsoid(center=np.asarray(centre, dtype=float),
                            axes=np.array([0.2, 0.2, 0.2]), R=np.eye(3)),
    )


def _silent_arrival_agent(**over):
    """An agent whose approach to track 1 is about to end with the detector silent."""
    cfg = make_cfg(approach_to_viewpoint=False, approach_navigable_goal=False, **over)
    agent = make_agent(cfg, target="chair")
    track = _track(1, [3.0, 0.5, 0.0])
    agent.object_layer._tracks[1] = track
    agent._candidate_id = 1
    agent.approach.start(np.array([3.0, 0.0]), agent_xy=np.zeros(2))
    agent.approach.steps_left = 0  # the deadline branch fires on the next step
    agent._goto_deadline = 10_000
    return agent, track


def _container(cid=40, centre=(2.0, 0.6, 0.0), label="table"):
    return ContainerNode(id=cid, label=label, track_ids=[], center=np.asarray(centre, float),
                         top_h=0.75, area_m2=1.0)


# ------------------------------------------------------------------ defaults

def test_off_by_default_and_inert():
    cfg = make_cfg()
    assert cfg.agent.close_look_before_absence is False
    assert cfg.agent.close_look_opportunistic is False
    agent, _ = _silent_arrival_agent()
    agent.approach.step(_frame([0.0, 0.0]))
    assert agent.state in (State.DONE, State.EXPLORE)
    assert agent.close_look.log == []
    assert "close_look_started" not in agent.stats


def test_hold_pattern_returns_to_the_starting_heading():
    assert len(HOLD_PATTERN) == 4
    assert HOLD_PATTERN.count(TURN_LEFT) == HOLD_PATTERN.count(TURN_RIGHT)


def test_container_in_view_is_range_frame_and_occlusion_gated():
    frame = _frame([0.0, 0.0])  # camera at the origin looking down +x
    assert container_in_view(frame, _container(centre=(2.0, 0.6, 0.0)), 2.5) is not None
    assert container_in_view(frame, _container(centre=(3.5, 0.6, 0.0)), 2.5) is None  # too far
    assert container_in_view(frame, _container(centre=(-2.0, 0.6, 0.0)), 2.5) is None  # behind
    frame.depth[:, :] = 0.5  # a wall half a metre in front of the camera
    assert container_in_view(frame, _container(centre=(2.0, 0.6, 0.0)), 2.5) is None


# --------------------------------------------------------- before absence

def test_silent_arrival_looks_before_the_absence_sensor():
    agent, _ = _silent_arrival_agent(close_look_before_absence=True)
    action = agent.approach.step(_frame([0.0, 0.0]))
    assert action is not None
    assert agent.state is State.CLOSE_LOOK
    entry = agent.close_look.log[-1]
    assert entry["resume"] == "absence" and entry["reason"] == "deadline"
    assert entry["container_id"] == -1  # keyed by the negative track id
    assert agent.stats["close_look_started"] == 1
    assert agent.stats["close_look_absence"] == 1
    # The look aims at the 1.5 m ring around the track, not at the track.
    assert abs(np.linalg.norm(np.asarray(entry["goal_xy"]) - np.array([3.0, 0.0])) - 1.7) < 0.3


def test_silent_look_ends_and_hands_back_to_the_absence_path():
    agent, _ = _silent_arrival_agent(close_look_before_absence=True,
                                     close_look_max_steps=2, close_look_face_turns=0,
                                     close_look_hold_steps=4)
    agent.approach.step(_frame([0.0, 0.0]))
    actions = []
    for i in range(12):
        if agent.state is not State.CLOSE_LOOK:
            break
        actions.append(agent.act(_frame([0.0, 0.0], frame_id=i + 1)))
    assert agent.state is not State.CLOSE_LOOK, actions
    entry = agent.close_look.log[-1]
    assert entry["detected"] is False and entry["steps"] >= 4
    assert agent.stats["close_look_silent"] == 1
    # The four hold turns were the pattern, so the heading is where it was;
    # what follows them is the absence path re-approaching the ring.
    pattern = list(HOLD_PATTERN)
    assert any(actions[i:i + 4] == pattern for i in range(len(actions) - 3)), actions
    assert agent.stats["close_look_stop_kept"] == 1
    # A second silent arrival at the same track does not look again.
    assert agent.close_look.before_absence(_frame([0.0, 0.0]), "path_consumed") is None


def test_a_detection_during_the_look_reaims_the_approach():
    agent, _ = _silent_arrival_agent(close_look_before_absence=True)
    agent.approach.step(_frame([0.0, 0.0]))
    assert agent.state is State.CLOSE_LOOK
    # The look runs the detector on its own frame, as the approach does.
    agent.detector.push([_det("chair", (300, 300))])
    agent.close_look.step(_frame([0.0, 0.0], frame_id=1))
    entry = agent.close_look.log[-1]
    assert entry["detected"] is True
    assert agent.stats["close_look_reapproach"] == 1
    assert agent._candidate_id == 1
    assert agent.state in (State.APPROACH, State.DONE)


# ------------------------------------------------------------- opportunistic

def _explore_agent_with_surface(**over):
    cfg = make_cfg(close_look_opportunistic=True, **over)
    agent = make_agent(cfg, target="chair")
    agent.state = State.EXPLORE
    agent.scene_graph.containers = {40: _container(40)}
    return agent


def test_an_affording_surface_in_view_earns_a_look():
    agent = _explore_agent_with_surface()
    frame = _frame([0.0, 0.0])
    assert agent.close_look.maybe_opportunistic(frame) is True
    assert agent.state is State.CLOSE_LOOK
    entry = agent.close_look.log[-1]
    assert entry["resume"] == "explore" and entry["container_id"] == 40
    assert entry["label"] == "table" and 1.5 < entry["trigger_range_m"] < 2.5
    assert agent.stats["close_look_explore"] == 1


def test_a_surface_is_looked_at_once_and_the_cap_holds():
    agent = _explore_agent_with_surface(close_look_max_steps=1, close_look_face_turns=0,
                                        close_look_hold_steps=0)
    frame = _frame([0.0, 0.0])
    assert agent.close_look.maybe_opportunistic(frame) is True
    for i in range(6):
        if agent.state is not State.CLOSE_LOOK:
            break
        agent.act(_frame([0.0, 0.0], frame_id=i + 1))
    assert agent.state is not State.CLOSE_LOOK
    assert 40 in agent.close_look.looked and 40 in agent.exploration.inspected
    looks = [e for e in agent.exploration.search_log_events if e.get("close_look")]
    assert len(looks) == 1 and looks[0]["detected"] is False
    assert looks[0]["belief_factor"] is not None  # silence retired belief
    # Same surface again: no second look.
    agent.state = State.EXPLORE
    agent.scene_graph.containers = {40: _container(40)}
    assert agent.close_look.maybe_opportunistic(frame) is False
    # A fresh surface under a spent cap: no look either.
    agent.cfg.agent.close_look_max_per_episode = 1
    agent.scene_graph.containers = {41: _container(41, centre=(1.5, 0.6, 0.3))}
    assert agent.close_look.maybe_opportunistic(frame) is False


def test_a_surface_the_search_is_driving_to_is_left_to_the_search():
    agent = _explore_agent_with_surface()
    agent.exploration.search_container = 40
    assert agent.close_look.maybe_opportunistic(_frame([0.0, 0.0])) is False


def test_a_surface_that_cannot_hold_the_target_is_ignored():
    agent = _explore_agent_with_surface()
    agent.scene_graph.containers = {40: ContainerNode(
        id=40, label="shelf", track_ids=[], center=np.array([2.0, 2.0, 0.0]),
        top_h=1.9, area_m2=1.0,
    )}
    assert agent.close_look.maybe_opportunistic(_frame([0.0, 0.0])) is False


def test_a_commit_that_preempts_the_look_counts_as_a_detection():
    agent = _explore_agent_with_surface()
    assert agent.close_look.maybe_opportunistic(_frame([0.0, 0.0])) is True
    agent.state = State.APPROACH  # the candidate path committed mid-look
    agent.close_look.interrupted()
    assert agent.close_look.active is False
    entry = agent.close_look.log[-1]
    assert entry["detected"] is True and entry["pre_empted"] is True
    looks = [e for e in agent.exploration.search_log_events if e.get("close_look")]
    assert looks[-1]["belief_factor"] is None  # a found target retires nothing


def test_the_interrupted_pursuit_is_resumed():
    agent = _explore_agent_with_surface(close_look_max_steps=1, close_look_face_turns=0,
                                        close_look_hold_steps=0)
    agent.state = State.GOTO_FRONTIER
    agent._goal_xy = np.array([4.0, 4.0])
    assert agent.close_look.maybe_opportunistic(_frame([0.0, 0.0])) is True
    for i in range(6):
        if agent.state is not State.CLOSE_LOOK:
            break
        agent.act(_frame([0.0, 0.0], frame_id=i + 1))
    assert agent.state in (State.GOTO_FRONTIER, State.EXPLORE)
    if agent.state is State.GOTO_FRONTIER:
        assert np.allclose(agent._goal_xy, [4.0, 4.0])


def test_rearm_aborts_a_look_in_progress():
    agent = _explore_agent_with_surface()
    assert agent.close_look.maybe_opportunistic(_frame([0.0, 0.0])) is True
    agent.rearm(100)
    assert agent.close_look.active is False
    assert agent.close_look.log[-1]["aborted"] is True
    assert agent.state is State.EXPLORE


# ------------------------------------------------------------ by belief

def test_by_belief_prefers_the_believed_surface_over_the_nearest():
    agent = _explore_agent_with_surface(close_look_by_belief=True, close_look_min_belief=0.5)
    agent.scene_graph.containers = {
        40: _container(40, centre=(1.5, 0.6, 0.0)),   # nearer, barely believed
        41: _container(41, centre=(2.2, 0.6, 0.3)),   # farther, the best surface on the floor
    }
    agent.exploration.surface_beliefs = lambda world: {40: 0.2, 41: 1.0}
    assert agent.close_look.maybe_opportunistic(_frame([0.0, 0.0])) is True
    entry = agent.close_look.log[-1]
    assert entry["container_id"] == 41 and entry["belief"] == 1.0
    assert agent.stats["close_look_below_belief"] == 1


def test_by_belief_spends_nothing_on_a_surface_below_the_bar():
    agent = _explore_agent_with_surface(close_look_by_belief=True, close_look_min_belief=0.5)
    agent.exploration.surface_beliefs = lambda world: {40: 0.3}
    assert agent.close_look.maybe_opportunistic(_frame([0.0, 0.0])) is False
    assert agent.close_look.log == []
    assert agent.stats["close_look_below_belief"] == 1


def test_nearest_first_is_unchanged_when_by_belief_is_off():
    agent = _explore_agent_with_surface()
    agent.scene_graph.containers = {
        40: _container(40, centre=(1.5, 0.6, 0.0)),
        41: _container(41, centre=(2.2, 0.6, 0.3)),
    }
    agent.exploration.surface_beliefs = lambda world: {40: 0.2, 41: 1.0}
    assert agent.close_look.maybe_opportunistic(_frame([0.0, 0.0])) is True
    assert agent.close_look.log[-1]["container_id"] == 40
    assert agent.close_look.log[-1]["belief"] is None
