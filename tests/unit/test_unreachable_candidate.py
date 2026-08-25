"""What happens to a candidate the agent believes it cannot reach.

Two defects, one decision. `_check_candidates` asked Habitat whether the
OBJECT'S OWN POSITION was on the agent's navmesh component, and blacklisted the
track when the answer was no.

The question is wrong. `_start_approach` never drives to the object's position
-- it drives to a viewpoint on the ring, because "a tabletop object's centre is
an occupied cell inside the furniture". Measured with no detector involved over
the 36 authored target poses of scene 00829: the object's own position is off
the navmesh in 6 of them, and in ALL SIX an authored viewpoint is reachable. The
benchmark defines success as standing at such a viewpoint, so those episodes are
solvable by construction.

And the answer must not be permanent. Blacklisting is absorbing, which is the
one thing this pipeline says everywhere else that no state may be -- the absence
path, the map loader and the attempt protocol each had a blacklist removed for
that reason. On condition K, `unreachable_skip` fired in 18 of 53 failing
episodes against 2 of 43 successful ones.

Both are off by default, so this file also pins the shipped behaviour.
"""
from __future__ import annotations

import numpy as np

from osg.agent.nav_agent import NavAgent
from osg.exploration.async_scorer import AsyncScorer
from osg.exploration.scorer import NullScorer
from osg.mapping.costmap import FREE
from osg.objects.association import ObjectTrack
from osg.objects.ellipsoid import Ellipsoid
from osg.perception.detector import StubDetector

from .test_nav_agent import make_cfg

OBJ = np.array([2.0, 0.6, 1.0])


def _agent(*, reachable, **flags):
    """An agent with one mapped `chair` track, and a navmesh that answers
    `reachable(xy)` for any query."""
    cfg = make_cfg()
    cfg.agent.use_habitat_navmesh = True
    cfg.verification.min_obs = 1
    cfg.verification.min_score = 0.0
    cfg.verification.min_bbox_px = 0
    for dotted, value in flags.items():
        group, field = dotted.split(".")
        setattr(getattr(cfg, group), field, value)
    asked: list = []

    def reachable_fn(xy, floor_y=None):
        asked.append(np.asarray(xy, dtype=float).copy())
        return reachable(np.asarray(xy, dtype=float))

    agent = NavAgent(
        cfg, StubDetector(), AsyncScorer(NullScorer()), None, "chair",
        nav_fn=lambda goal, floor_y=None: "move_forward",
        reachable_fn=reachable_fn,
    )
    agent.costmap.grid[:, :] = FREE
    track = ObjectTrack(
        id=1, label="chair",
        ellipsoid=Ellipsoid(center=OBJ.copy(), axes=np.array([0.15, 0.15, 0.15]),
                            R=np.eye(3)),
    )
    track.n_obs_override = None
    track.evidence = 5.0
    track.best_score = 0.9
    track.best_bbox_px = 9000.0
    for _ in range(3):
        track.observations.append(None)
    agent.object_layer._tracks[1] = track
    return agent, track, asked


def _nothing_reachable(xy):
    return False


def _only_away_from_the_object(xy):
    """The navmesh a tabletop object actually sits on: the object's own (x, z)
    is inside the furniture, anything a ring-radius away is standable."""
    return bool(np.linalg.norm(xy - OBJ[[0, 2]]) > 0.5)


# ------------------------------------------------------------ shipped behaviour

def test_by_default_an_unreachable_candidate_is_struck_off_for_good():
    agent, track, _ = _agent(reachable=_nothing_reachable)
    agent.candidates.check()
    assert track.blacklisted, "this is the shipped behaviour and it is absorbing"
    assert agent.stats["unreachable_skip"] == 1
    assert agent._candidate_id is None


def test_by_default_the_question_is_asked_about_the_object_itself():
    agent, _, asked = _agent(reachable=_nothing_reachable)
    agent.candidates.check()
    assert asked, "the navmesh was consulted"
    assert np.allclose(asked[0], OBJ[[0, 2]]), (
        "shipped behaviour asks about the object's own position"
    )


# ---------------------------------------------------------------- the two fixes

def test_a_non_absorbing_verdict_leaves_the_track_in_the_map():
    agent, track, _ = _agent(
        reachable=_nothing_reachable, **{"verification.unreachable_is_absorbing": False}
    )
    agent.candidates.check()
    assert not track.blacklisted, "one failed reach is evidence, not a verdict"
    assert track.identity_rejections == 1
    assert agent.stats["unreachable_skip"] == 1


def test_two_non_absorbing_verdicts_do_retire_it():
    """Non-absorbing must not mean a livelock: the identity channel already
    carries the retirement, at max_identity_rejections."""
    agent, track, _ = _agent(
        reachable=_nothing_reachable, **{"verification.unreachable_is_absorbing": False}
    )
    for _ in range(2):
        agent.candidates.check()
    assert track.identity_rejections == 2
    assert not agent.object_layer.candidates("chair", min_obs=1,
                                             max_identity_rejections=2), (
        "two failures retire it, and it is still recoverable by a later sighting"
    )


def test_asking_about_the_viewpoint_rescues_a_tabletop_object():
    agent, track, asked = _agent(
        reachable=_only_away_from_the_object,
        **{"agent.reachable_via_viewpoint": True},
    )
    agent.candidates.check()
    assert not track.blacklisted
    assert agent.stats.get("unreachable_skip", 0) == 0, (
        "a viewpoint is reachable, so the candidate is not out of reach"
    )
    assert any(np.linalg.norm(a - OBJ[[0, 2]]) > 0.5 for a in asked), (
        "the navmesh must have been asked about a pose off the object"
    )


def test_a_genuinely_sealed_target_is_still_rejected():
    """The mechanism exists for real cases -- a target behind a closed door --
    and must keep working when neither the object nor any viewpoint is reachable."""
    agent, track, _ = _agent(
        reachable=_nothing_reachable,
        **{"agent.reachable_via_viewpoint": True},
    )
    agent.candidates.check()
    assert agent.stats["unreachable_skip"] == 1
    assert track.blacklisted, "absorbing is still the default"
