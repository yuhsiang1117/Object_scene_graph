"""`exploration.storey_hold_follows_candidacy` (objects/retirement.py): a track the
candidate gate has struck off -- a stale twin, an identity-retired false positive --
holds no storey through the storey belief, the anchor test or the floor gate's
target bonus. Off, the three readers are exactly what they were."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from osg.exploration.storey_belief import target_tracks
from osg.objects.retirement import Retirement, identity_retired, is_stale_twin

from .test_stale_anchor import _track
from .test_nav_agent import make_agent, make_cfg


def _agent(follows: bool, refuted: bool = True, twins: bool = True):
    """s44 00821 in miniature: the prior anchor (1) found empty, its twin (2) never
    seen live and still believed, both on this storey."""
    cfg = make_cfg()
    cfg.verification.retire_stale_twins_after_absence = twins
    cfg.exploration.search_floor_requires_anchor_test = True
    cfg.exploration.floor_evidence_by_presence = True
    cfg.exploration.storey_hold_follows_candidacy = follows
    agent = make_agent(cfg, target="chair")
    anchor = _track(1, [3.0, 0.5, 0.0], prior=True, live=False)
    anchor.absence_arrivals = 1 if refuted else 0
    twin = _track(2, [8.0, 0.5, 2.0], prior=True, live=False)
    twin.presence.log_odds = 3.0
    for t in (anchor, twin):
        agent.object_layer._tracks[t.id] = t
    world = SimpleNamespace(object_layer=agent.object_layer, target="chair",
                            floor_id=agent.floors.current_id)
    return agent, world, anchor, twin


def _readers(agent, world):
    """(tracks in the storey belief, anchor test, floor-gate verdict on the twin)."""
    strategy = agent.exploration
    belief = target_tracks(agent.object_layer, "chair", max_rejections=2,
                           retirement=strategy._retirement(world))
    believed = agent._presence_for_floor_evidence()
    return len(belief), strategy._anchor_untested(world), believed(2)


def test_off_a_stale_twin_still_holds_its_storey():
    agent, world, _, _ = _agent(follows=False)
    assert _readers(agent, world) == (2, True, True)
    assert "anchor_test_retired_skipped" not in agent.exploration.stats
    assert "floor_bonus_retired_skipped" not in agent.stats


def test_on_a_stale_twin_holds_nothing():
    agent, world, _, _ = _agent(follows=True)
    n, untested, believed = _readers(agent, world)
    # The refuted anchor is a never-seen prior track too: the gate strikes both off.
    assert (n, untested, believed) == (0, False, False)
    assert agent.exploration.stats["anchor_test_retired_skipped"] == 1   # the twin; the anchor is tested
    assert agent.stats["floor_bonus_retired_skipped"] == 1
    cands = agent.object_layer.candidates("chair", min_obs=0, min_presence=0.0)
    assert agent.candidates._without_stale_twins(cands) == []        # the gate's own verdict


def test_on_a_twin_still_holds_before_any_refutation_or_without_the_twin_rule():
    for refuted, twins in ((False, True), (True, False)):
        agent, world, _, _ = _agent(follows=True, refuted=refuted, twins=twins)
        n, untested, believed = _readers(agent, world)
        assert untested and believed and n == 2


def test_on_an_identity_retired_false_positive_no_longer_forbids_leaving():
    """The detector pumps a false positive's presence straight back, so presence
    alone kept the anchor test and the target bonus closed after the gate had
    retired it."""
    for follows in (False, True):
        agent, world, anchor, twin = _agent(follows=follows, refuted=False)
        del agent.object_layer._tracks[1]
        twin.observations.append(twin.observations[0])   # seen: not a twin
        twin.from_prior = False
        twin.identity_rejections = 2
        n, untested, believed = _readers(agent, world)
        assert n == 0                                   # the storey belief already agreed
        assert (untested, believed) == ((False, False) if follows else (True, True))


def test_the_helpers_read_the_counters_the_gate_reads():
    t = SimpleNamespace(identity_rejections=2, from_prior=True, seen_live=False)
    assert identity_retired(t, 2) and not identity_retired(t, 3) and not identity_retired(t, 0)
    assert is_stale_twin(t) and not is_stale_twin(SimpleNamespace(from_prior=True, seen_live=True))
    layer = SimpleNamespace(tracks=lambda include_blacklisted=False: [
        SimpleNamespace(label="Toy Airplane", from_prior=True, absence_arrivals=0, failed_attempts=1)])
    assert Retirement(layer, "toy_airplane", 2, True).twins_armed
    assert not Retirement(layer, "toy airplane", 2, False).twins_armed
    assert not Retirement(layer, "bowl", 2, True).twins_armed
    np.testing.assert_equal(Retirement(layer, "toy airplane", 2, True).retired(t), True)
