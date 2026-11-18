"""The stale-anchor policy (core/config/verification.py): a prior-map track
that this episode has never seen gets one STOP instead of an absence verdict,
and once a prior-map track of the label is refuted, its unseen twins from the
same static pass stop being candidates."""
from __future__ import annotations

import numpy as np

from osg.agent.nav_agent import State
from osg.objects.association import Observation, ObjectTrack
from osg.objects.ellipsoid import Ellipsoid

from .test_nav_agent import _frame, make_agent, make_cfg


def _obs(frame_id: int) -> Observation:
    return Observation(frame_id=frame_id, mu=np.zeros(2), cov=np.eye(2), K=np.eye(3),
                       T_cw=np.eye(4), mean_depth=1.0)


def _track(tid, centre, *, prior: bool, live: bool, label="chair", score=0.9):
    t = ObjectTrack(
        id=tid, label=label,
        ellipsoid=Ellipsoid(center=np.asarray(centre, dtype=float),
                            axes=np.array([0.2, 0.2, 0.2]), R=np.eye(3)),
    )
    t.from_prior = prior
    t.observations = [_obs(-1_000_000 + 5)] if prior else []
    if live:
        t.observations.append(_obs(12))
    t.best_score = score
    return t


def test_seen_live_reads_the_frame_ids():
    assert _track(1, [1, 0, 0], prior=True, live=False).seen_live is False
    assert _track(2, [1, 0, 0], prior=True, live=True).seen_live is True
    assert _track(3, [1, 0, 0], prior=False, live=True).seen_live is True


def _silent_arrival(track, **over):
    cfg = make_cfg(approach_to_viewpoint=False, approach_navigable_goal=False)
    for k, v in over.items():
        setattr(cfg.verification, k, v)
    agent = make_agent(cfg, target="chair")
    agent.object_layer._tracks[track.id] = track
    agent._candidate_id = track.id
    agent.approach.start(np.array([3.0, 0.0]), agent_xy=np.zeros(2))
    agent.approach.steps_left = 0
    agent._goto_deadline = 10_000
    return agent


def test_off_by_default():
    cfg = make_cfg()
    assert cfg.verification.stop_at_stale_anchor_once is False
    assert cfg.verification.retire_stale_twins_after_absence is False


def test_first_silent_arrival_at_a_stale_anchor_stops_once():
    agent = _silent_arrival(_track(1, [3.0, 0.5, 0.0], prior=True, live=False),
                            stop_at_stale_anchor_once=True)
    action = agent.approach.step(_frame([0.0, 0.0]))
    assert action == "stop" and agent.state is State.DONE
    assert agent.stats["stale_anchor_stop"] == 1
    # The second silent arrival goes to the absence sensor as before.
    agent.rearm(100)
    agent._candidate_id = 1
    agent.approach.start(np.array([3.0, 0.0]), agent_xy=np.zeros(2))
    agent.approach.steps_left = 0
    agent.approach.step(_frame([0.0, 0.0]))
    assert agent.stats["stale_anchor_stop"] == 1


def test_a_track_seen_live_gets_no_free_stop():
    agent = _silent_arrival(_track(1, [3.0, 0.5, 0.0], prior=True, live=True),
                            stop_at_stale_anchor_once=True)
    agent.approach.step(_frame([0.0, 0.0]))
    assert "stale_anchor_stop" not in agent.stats


def test_a_live_track_gets_no_free_stop():
    agent = _silent_arrival(_track(1, [3.0, 0.5, 0.0], prior=False, live=True),
                            stop_at_stale_anchor_once=True)
    agent.approach.step(_frame([0.0, 0.0]))
    assert "stale_anchor_stop" not in agent.stats


def _agent_with_twins(refuted: bool, **over):
    cfg = make_cfg()
    cfg.verification.min_obs = 0
    for k, v in over.items():
        setattr(cfg.verification, k, v)
    agent = make_agent(cfg, target="chair")
    anchor = _track(1, [3.0, 0.5, 0.0], prior=True, live=False)
    if refuted:
        anchor.absence_arrivals = 1
    else:
        anchor.identity_rejections = 1  # an unreachable strike is not a refutation
        anchor.presence.log_odds = -3.0  # under the bar, as after a refutation
    twin = _track(2, [8.0, 0.5, 2.0], prior=True, live=False, score=0.8)
    live = _track(3, [5.0, 0.5, -2.0], prior=False, live=True, score=0.7)
    for t in (anchor, twin, live):
        agent.object_layer._tracks[t.id] = t
    return agent


def test_unseen_prior_twins_are_retired_once_the_anchor_is_refuted():
    agent = _agent_with_twins(refuted=True, retire_stale_twins_after_absence=True)
    cands = agent.object_layer.candidates("chair", min_obs=0, min_score=0.0, min_bbox_px=0,
                                          min_evidence=0.0, min_presence=0.0)
    kept = agent.candidates._without_stale_twins(cands)
    assert [t.id for t in kept] == [3]
    assert agent.stats["stale_twins_retired"] >= 1


def test_nothing_is_retired_before_a_refutation_or_with_the_flag_off():
    agent = _agent_with_twins(refuted=False, retire_stale_twins_after_absence=True)
    cands = agent.object_layer.candidates("chair", min_obs=0, min_score=0.0, min_bbox_px=0,
                                          min_evidence=0.0, min_presence=0.0)
    assert len(agent.candidates._without_stale_twins(cands)) == len(cands)
    agent = _agent_with_twins(refuted=True)
    agent.candidates.check(np.zeros(2))
    assert "stale_twins_retired" not in agent.stats


def test_the_stop_granted_inside_the_look_is_taken_on_the_next_ring_arrival():
    """With the look before absence on, the silent look ends on the 1.5 m
    ring; the stop must be taken back at the tight ring, not spent there."""
    from osg.agent.nav_agent import State
    track = _track(1, [3.0, 0.5, 0.0], prior=True, live=False)
    cfg = make_cfg(approach_to_viewpoint=False, approach_navigable_goal=False,
                   close_look_before_absence=True, close_look_max_steps=1,
                   close_look_face_turns=0, close_look_hold_steps=0)
    cfg.verification.stop_at_stale_anchor_once = True
    agent = make_agent(cfg, target="chair")
    agent.object_layer._tracks[1] = track
    agent._candidate_id = 1
    agent.approach.start(np.array([3.0, 0.0]), agent_xy=np.zeros(2))
    agent.approach.steps_left = 0
    agent._goto_deadline = 10_000
    agent.approach.step(_frame([0.0, 0.0]))          # silent arrival -> look
    assert agent.state is State.CLOSE_LOOK
    for i in range(6):                                # the look ends silent
        if agent.state is not State.CLOSE_LOOK:
            break
        agent.act(_frame([0.0, 0.0], frame_id=i + 1))
    assert agent.stats["stale_anchor_stop"] == 1
    assert agent._stale_stop_pending is True and agent.state is State.APPROACH
    agent.approach.steps_left = 0                     # arrive at the ring again
    action = agent.approach.step(_frame([0.0, 0.0], frame_id=20))
    assert action == "stop" and agent.state is State.DONE
    assert agent._stale_stop_pending is False


def test_the_nearest_free_stop_walks_to_the_track_first():
    from osg.agent.nav_agent import State
    track = _track(1, [3.0, 0.5, 0.0], prior=True, live=False)
    agent = _silent_arrival(track, stop_at_stale_anchor_once=True, stale_stop_at_nearest_free=True)
    action = agent.approach.step(_frame([0.0, 0.0]))
    assert action != "stop" and agent.state is State.APPROACH
    assert agent.stats["stale_stop_reaimed"] == 1 and agent._stale_stop_pending is True
    assert np.linalg.norm(agent._goal_xy - np.array([3.0, 0.0])) < 0.6  # at the track, not on a ring
    agent.approach.steps_left = 0                     # arrive there
    action = agent.approach.step(_frame([2.9, 0.0], frame_id=5))
    assert action == "stop" and agent.state is State.DONE


def test_a_failed_attempt_disables_every_fragment_at_the_place():
    from types import SimpleNamespace
    from osg.eval.attempts import rearm_after_failed_attempt
    cfg = make_cfg()
    cfg.verification.failed_attempt_disables_place = True
    agent = make_agent(cfg, target="chair")
    ghost = [_track(1, [3.0, 0.5, 0.0], prior=True, live=True),
             _track(2, [3.15, 0.5, 0.1], prior=False, live=True),
             _track(3, [3.3, 0.5, 0.0], prior=False, live=True)]
    far = _track(4, [8.0, 0.5, 0.0], prior=True, live=False)
    for t in ghost + [far]:
        agent.object_layer._tracks[t.id] = t
    agent._candidate_id = 1
    rearm_after_failed_attempt(agent, cfg)
    assert all(t.blacklisted and t.disabled for t in ghost)
    assert not far.blacklisted
    assert agent.stats["failed_attempt_place_disabled"] == 3
    assert ghost[0].failed_attempts == 1


def test_nearest_free_reachability_rescues_an_object_with_no_ring_pose():
    """A candidate whose ring poses are all occupied and whose own position is
    off the navmesh is reachable if the nearest free cell is."""
    cfg = make_cfg(reachable_via_viewpoint=True, reachable_via_nearest_free=True)
    agent = make_agent(cfg, target="chair")
    asked = []
    def reachable(xy, floor_y=None):
        asked.append(np.asarray(xy, dtype=float).copy())
        return len(asked) >= 2  # the viewpoint says no, the nearest free cell says yes
    agent._reachable_fn = reachable
    track = _track(1, [3.0, 0.5, 0.0], prior=False, live=True)
    agent.object_layer._tracks[1] = track
    assert agent.candidates._reachable(track, np.array([3.0, 0.0]), np.array([3.0, 0.5, 0.0])) is True
    assert agent.stats["reachable_via_nearest_free"] == 1
    cfg.agent.reachable_via_nearest_free = False
    asked.clear()
    agent._reachable_fn = lambda xy, floor_y=None: False
    assert agent.candidates._reachable(track, np.array([3.0, 0.0]), np.array([3.0, 0.5, 0.0])) is False


def test_a_failed_stale_stop_schedules_a_relook_from_elsewhere():
    from osg.agent.nav_agent import State
    track = _track(1, [3.0, 0.5, 0.0], prior=True, live=False)
    agent = _silent_arrival(track, stop_at_stale_anchor_once=True, relook_after_stale_stop=True)
    assert agent.approach.step(_frame([0.0, 0.0])) == "stop"      # the stale stop
    agent._agent_xy = np.array([2.4, 0.0])                        # where it stood
    agent.rearm(100)                                              # the attempt failed
    assert agent._relook is not None and agent.stats["relook_scheduled"] == 1
    action = agent.act(_frame([2.4, 0.0], frame_id=3))            # EXPLORE opens with the look
    assert action is not None and agent.state is State.CLOSE_LOOK
    entry = agent.close_look.log[-1]
    assert entry["reason"] == "relook"
    assert np.linalg.norm(np.asarray(entry["goal_xy"]) - np.array([2.4, 0.0])) >= 1.0
    assert agent.stats["relook_started"] == 1 and agent._relook is None


def test_no_relook_without_a_stale_stop_or_with_the_flag_off():
    track = _track(1, [3.0, 0.5, 0.0], prior=True, live=False)
    agent = _silent_arrival(track, relook_after_stale_stop=True)   # no stale stop taken
    agent.rearm(100)
    assert agent._relook is None
    agent = _silent_arrival(track, stop_at_stale_anchor_once=True)  # flag off
    agent.approach.step(_frame([0.0, 0.0]))
    agent.rearm(100)
    assert agent._relook is None
