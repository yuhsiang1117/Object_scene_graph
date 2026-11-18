"""What a failed navigation attempt does to the candidate it was aimed at.

DualMap allows a query several navigation attempts, and matching that means the
map must survive between them. What must NOT survive is a permanent verdict:
C1's premise is that no state is absorbing, and this is the third place that
premise had to be enforced after the absence path and the map loader.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from osg.eval.attempts import rearm_after_failed_attempt
from osg.objects.association import ObjectTrack
from osg.objects.ellipsoid import Ellipsoid
from osg.objects.object_layer import ObjectLayer
from osg.objects.presence import PresenceFilter


MIN_PRESENCE = 0.45


def _cfg(min_presence=MIN_PRESENCE):
    return SimpleNamespace(
        agent=SimpleNamespace(max_steps=500),
        verification=SimpleNamespace(vlm_recall=0.9, vlm_q=0.2),
        scene_graph=SimpleNamespace(presence=SimpleNamespace(min_presence=min_presence)),
    )


def _agent(log_odds, *, with_filter=True):
    layer = ObjectLayer(presence_filter=PresenceFilter() if with_filter else None)
    track = ObjectTrack(
        id=1, label="cracker box",
        ellipsoid=Ellipsoid(center=np.array([1.0, 0.8, 2.0]),
                            axes=np.array([0.1, 0.1, 0.1]), R=np.eye(3)),
    )
    track.presence.log_odds = log_odds
    layer._tracks[1] = track
    # What a failed attempt is WORTH is the protocol's question and is what
    # these tests are about; putting the agent back into EXPLORE is the agent's
    # own `rearm`, covered in test_nav_agent.py. Recorded here so the protocol
    # is still shown to hand off.
    agent = SimpleNamespace(object_layer=layer, _candidate_id=1, stats={}, rearmed=[])
    agent.rearm = agent.rearmed.append
    return agent, track


@pytest.mark.parametrize("log_odds", [1.5, 3.0])
def test_a_failed_attempt_puts_the_candidate_under_the_bar_but_leaves_it_in_the_map(log_odds):
    """1.5 is a track restored from a snapshot, 3.0 one detected this episode at
    the positive clamp. Both must end below `min_presence` or the next attempt
    simply repeats the candidate that just failed."""
    agent, track = _agent(log_odds)
    rearm_after_failed_attempt(agent, _cfg())
    assert track.blacklisted is False, "a failed attempt is not a permanent verdict"
    assert track.presence.p < MIN_PRESENCE
    assert agent.object_layer.get(1) is track
    assert not agent.object_layer.candidates(
        "cracker box", min_obs=0, min_presence=MIN_PRESENCE
    ), "the next attempt must choose something else"


def test_one_later_detection_brings_the_candidate_back():
    """The whole difference from a blacklist. Measured cost of getting this
    wrong: three cracker box episodes finished 0.67-0.95 m from the goal with
    429 steps unspent, unable to stop, because the only track that could have
    been the answer had been struck off."""
    agent, track = _agent(1.5)
    rearm_after_failed_attempt(agent, _cfg())
    assert track.presence.p < MIN_PRESENCE

    agent.object_layer.presence_filter.apply_reading(track, True, 0.6, 0.05)
    assert track.presence.p > MIN_PRESENCE


def test_without_a_presence_filter_the_blacklist_is_still_the_fallback():
    """The C1-off ablation has no belief to lower, and something still has to
    stop the next attempt repeating this candidate."""
    agent, track = _agent(1.5, with_filter=False)
    rearm_after_failed_attempt(agent, _cfg())
    assert track.blacklisted is True


def test_a_failed_attempt_also_counts_as_identity_evidence():
    """Presence and identity are different questions and the belief can only
    carry one of them. A false positive is an object that really is there, so
    every look that disproves it as the target re-detects it as an object and
    restores the belief the clamp just lowered."""
    agent, track = _agent(3.0)
    rearm_after_failed_attempt(agent, _cfg())
    assert track.identity_rejections == 1

    # A detection undoes the belief step, exactly as it should ...
    agent.object_layer.presence_filter.apply_reading(track, True, 0.6, 0.05)
    assert track.presence.p > MIN_PRESENCE
    # ... and the identity channel still holds the candidate back.
    assert not agent.object_layer.candidates(
        "cracker box", min_obs=0, min_presence=MIN_PRESENCE, max_identity_rejections=1
    )
    assert agent.object_layer.candidates(
        "cracker box", min_obs=0, min_presence=MIN_PRESENCE, max_identity_rejections=0
    ), "0 must disable the gate"


def test_identity_rejections_do_not_survive_a_map_reload():
    """Episode-scoped, like the blacklist: 'I walked over there twice today' is
    not a fact about tomorrow's world."""
    from osg.graph.map_store import apply_map

    agent, track = _agent(1.5)
    track.identity_rejections = 5
    blob = {"tracks": [{
        "id": 1, "label": "cracker box", "center": [1.0, 0.8, 2.0],
        "axes": [0.1, 0.1, 0.1], "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "observations": [], "best_score": 0.7, "best_bbox_px": 900.0,
        "blacklisted": False, "linked_ids": [], "refined_at_obs": 0,
        "evidence": 1.0, "presence": {"log_odds": 1.5, "last_seen_kf": 1,
                                      "last_absent_kf": None, "n_expected": 1, "n_missed": 0},
    }], "next_track_id": 2, "resolution": agent.object_layer and 0.05}
    holder = SimpleNamespace(object_layer=agent.object_layer,
                             costmap=SimpleNamespace(resolution=0.05))
    try:
        apply_map(holder, {k: v for k, v in blob.items() if k != "resolution"})
    except Exception:
        pass  # the costmap half of apply_map needs a real agent; the tracks half ran
    restored = agent.object_layer.get(1)
    assert restored is not None and restored.identity_rejections == 0
