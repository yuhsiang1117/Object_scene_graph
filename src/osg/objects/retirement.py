"""When has the candidate gate given up on a target-labelled track?

Four places ask whether a target-labelled track still argues for its storey: the
candidate gate (`agent/candidate.py`), the storey belief
(`exploration/storey_belief.target_tracks`), the anchor test
(`ExplorationStrategy._anchor_untested`) and the floor gate's target bonus
(`NavAgent._track_still_believed`). Each read its own subset of the refutation
counters, and where they disagreed a track the agent would never walk to again
still forbade leaving its storey:

* a STALE TWIN -- a prior-map track of the label never seen live, once another
  one has been refuted in place -- is no candidate, but it kept its full weight
  in the storey belief. Measured on osg_landing_vet_live s44 00821
  cross_anchor_01 (toy airplane, moved up a storey): the prior anchor was found
  empty at step 40, its twin (p 0.95 after the prior vet's "yes") was struck off
  765 times and held the start storey's anchor at 0.93-0.97 to step 1000;
* a track the identity channel has retired (`max_identity_rejections`) left the
  storey belief, but the anchor test and the target bonus read presence and
  absence arrivals alone, and the detector pumps a false positive's presence
  straight back.

`exploration.storey_hold_follows_candidacy` makes the three storey readers ask
the gate's question through these helpers. `candidate.py` uses them with or
without it, so the twin rule itself is unchanged.

Measured, it LOSES (osg_landing_vet_live_follows, 3 seeds: 131 vs 136): a
stale twin may be the object itself, where the refuted prior track was a pass-1
false positive, and dropped from the belief it is never looked at. So it stays
off; the sweep oracle (`verification.stop_gate_vet_oracle: sweep`) is what
needs it, for the tracks the oracle retires.
"""
from __future__ import annotations


def label_key(label) -> str:
    """The candidate gate's label test (`ObjectLayer.candidates`)."""
    return str(label).lower().replace(" ", "_")


def prior_refuted_in_place(tracks, target: str) -> bool:
    """Has a prior-map track of the label been refuted at its pose -- an absence
    arrival or a failed attempt at it? What arms the twin rule."""
    want = label_key(target)
    return any(
        bool(getattr(t, "from_prior", False))
        and (int(getattr(t, "absence_arrivals", 0)) > 0
             or int(getattr(t, "failed_attempts", 0)) > 0)
        for t in tracks
        if label_key(t.label) == want
    )


def is_stale_twin(track) -> bool:
    """A prior-map track this episode has never seen: a ghost of the same
    static pass once the twin rule is armed."""
    return bool(getattr(track, "from_prior", False)) and not bool(getattr(track, "seen_live", False))


def identity_retired(track, max_identity_rejections: int) -> bool:
    """Retired by the identity channel, as `ObjectLayer.candidates` retires it
    (0 disables the limit there, so here too)."""
    limit = int(max_identity_rejections)
    return bool(limit) and int(getattr(track, "identity_rejections", 0) or 0) >= limit


class Retirement:
    """The gate's verdict on each target-labelled track, for one decision.

    Built once per decision from the whole layer, blacklisted tracks included,
    as `CandidatePolicy._without_stale_twins` builds it; then `retired(t)` is
    a lookup.
    """

    def __init__(self, object_layer, target: str, max_identity_rejections: int,
                 twins_retired: bool) -> None:
        self.max_identity_rejections = int(max_identity_rejections)
        self.n_twins_dropped = 0           # by `storey_belief.target_tracks`, for the stats
        self.twins_armed = bool(twins_retired) and prior_refuted_in_place(
            object_layer.tracks(include_blacklisted=True), target)

    def twin(self, track) -> bool:
        return self.twins_armed and is_stale_twin(track)

    def retired(self, track) -> bool:
        return identity_retired(track, self.max_identity_rejections) or self.twin(track)
