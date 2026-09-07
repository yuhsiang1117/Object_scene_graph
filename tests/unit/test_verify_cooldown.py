"""A VLM rejection is evidence about a view, not a verdict on an object (S38).

Measured failure it fixes: 3 of 22 episodes ended within 0.5 m of the goal
having never issued STOP -- two of them at 3-5 cm -- because their only
candidate had been permanently blacklisted by a single rejection made from
across the room. ASCENT's equivalent gate is re-evaluated every step until it
passes and only abandons a target at close range.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.objects.association import ObjectTrack
from osg.objects.ellipsoid import Ellipsoid
from osg.objects.object_layer import ObjectLayer


def _layer():
    layer = ObjectLayer()
    t = ObjectTrack(id=0, label="chair",
                    ellipsoid=Ellipsoid(center=np.array([3.0, 0.5, 4.0]),
                                        axes=np.full(3, 0.2), R=np.eye(3)))
    t.evidence, t.best_score, t.best_bbox_px = 10.0, 0.9, 50_000.0
    layer._tracks[0] = t
    return layer, t


def _cands(layer, step=None):
    return layer.candidates("chair", min_obs=0, step=step)


def test_suppression_expires():
    layer, _ = _layer()
    assert [t.id for t in _cands(layer, step=0)] == [0]
    layer.suppress(0, until_step=100)
    assert _cands(layer, step=50) == [], "still inside the cooldown"
    assert [t.id for t in _cands(layer, step=100)] == [0], "cooldown is exclusive"
    assert [t.id for t in _cands(layer, step=150)] == [0]


def test_suppression_never_shortens_an_existing_one():
    layer, _ = _layer()
    layer.suppress(0, until_step=200)
    layer.suppress(0, until_step=50)
    assert _cands(layer, step=100) == []


def test_suppression_is_not_blacklisting():
    layer, t = _layer()
    layer.suppress(0, until_step=100)
    assert t.blacklisted is False, "a rejected view must not retire the object"


def test_blacklist_still_permanent():
    layer, _ = _layer()
    layer.blacklist(0)
    for step in (0, 1_000, 10_000):
        assert _cands(layer, step=step) == []


def test_no_step_means_no_suppression_filter():
    """Callers that do not track steps keep the old semantics exactly."""
    layer, _ = _layer()
    layer.suppress(0, until_step=10_000)
    assert [t.id for t in _cands(layer, step=None)] == [0]


# ------------------------------------------------------- through the agent


def _agent_with_candidate(cooldown):
    from .test_nav_agent import make_agent, make_cfg

    # `costmap` navigation: the cooldown is about the verifier, not the mover,
    # and this keeps the fixture from having to stand up a PointNav driver.
    cfg = make_cfg()
    cfg.verification.reject_cooldown_steps = cooldown
    # The hand-built track has no Observations, so the real min_obs gate would
    # filter it out before the verifier is ever consulted.
    cfg.verification.min_obs = 0
    a = make_agent(cfg)
    layer, t = _layer()
    a.object_layer._tracks[0] = t
    a.verifier = type("V", (), {"verify": lambda self, *_a: False,
                                "n_calls": 0, "n_errors": 0})()
    a._direct_approach = True
    return a, t


def test_agent_rejection_is_temporary_with_a_cooldown():
    a, t = _agent_with_candidate(cooldown=100)
    a.step_count = 10
    a._check_candidates()
    assert a.stats.get("verify_reject") == 1
    assert t.blacklisted is False
    assert t.suppressed_until == 110


def test_agent_rejection_is_permanent_without_one():
    """Default behaviour, which every pre-S38 number was measured on."""
    a, t = _agent_with_candidate(cooldown=0)
    a.step_count = 10
    a._check_candidates()
    assert a.stats.get("verify_reject") == 1
    assert t.blacklisted is True
    assert t.suppressed_until == 0
