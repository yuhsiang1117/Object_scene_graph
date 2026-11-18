# Dominant failure mode of each ablated variant

Drafted for the results-section TODO ("please add a short discussion of the
dominant failure mode of each ablated variant, e.g. repeated returns to the
stale location without revision").

**Provenance, stated first because it constrains what may be claimed.** Every
number below is measured, but on the **DualMap dynamic benchmark** (in_anchor /
cross_anchor over 00829/00848/00880), not on the multi-floor cross-anchor set.
We do not have a per-variant ablation sweep on the multi-floor scenes. The
mechanisms are the same and the failure modes transfer in kind, but the
sentences below should either cite the dynamic benchmark explicitly or the
sweep should be run before they are stated of the multi-floor result.

---

## Draft paragraph for the paper

Each ablation fails in a characteristic and separately observable way, and the
three failure modes are not variations of one another.

**Without presence-belief revision the agent returns to the stale location and
commits there.** With a map built before the object moved, the dominant mode is
not wandering but confident error: the agent commits to the ghost track at the
old pose at step 1 with p = 0.98 and stops on arrival, ending the episode at
step 64 where the same query on a freshly built map succeeds at step 363
(`ghost_rate` 1.0, `belief_latency.flip_rate` 0.0). The stale map is not merely
unhelpful, it is actively harmful: it converts a success into a *confident*
failure, and the episode terminates before any negative evidence can accumulate.
Two compounding causes were measured. A saturated belief cannot be argued with —
a sighting is worth +2.5 log-odds against a miss's −0.9, so three sightings
saturate a symmetric clamp and seven clean misses are then needed to unwind it
(the ghost was stored at p = 0.9975 from five observations). And same-label
linking merged the object with its own past: a bowl displaced 0.80 m was unioned
with its ghost, so the reported centre sat 0.42 m from either bowl — a place
where no bowl is, and which no observation can contradict.

**Without the identity channel the agent revisits the same wrong object
indefinitely.** Presence cannot express this failure, because a false positive is
an object that really is there: every look that disproves it as the *target* also
re-detects it as an *object* and pins its belief at the positive clamp. With the
channel off, a single episode committed to the same wrong track **251 times in
500 steps**, its belief held at the clamp through 250 absence readings. Enabling
it reduces repeat commits to the same wrong track from 251 to 6 and removes
livelocked episodes entirely (7 → 0). The failure is therefore a livelock, not a
misallocation of search: the agent is neither lost nor exploring, it is
re-answering the same query with the same wrong answer.

**Without belief-guided surface scoring the agent explores geometrically instead
of searching where the object plausibly went.** Mapped support surfaces no longer
compete with geometric frontiers, so the relocation's destination is reached only
incidentally. Scored offline against the true destination over 114 relocations,
the share of cases in which the true surface appears in the top 5 — what one
episode can afford to inspect — falls from 36/114 to 12/114 when proximity is
dropped after absence. A related scale failure is worth reporting because it is
silent rather than noisy: anchoring the candidate set at the wrong mass switched
the search line off altogether, taking a pilot from 27 surface inspections over
six episodes to one, with no error and no change in any success metric that would
have revealed it.

These three modes are separable in the traces — a stale-location commit, a
repeat-commit livelock, and an absent search line — which is what allows
presence-belief revision and belief-guided surface scoring to be identified as
the mechanisms behind the improvement under dynamic relocation, rather than
inferred from the aggregate rate alone.

---

## Why the baseline is not a fourth row of this table

On the multi-floor cross-anchor set the transcribed ASCENT baseline scores 0/25.
That number should **not** be presented as a navigation result, because the
failure is upstream of navigation: ASCENT drives on D-FINE, a closed-set COCO
detector, and none of the eight YCB targets in that set has a COCO name. Across
the 25 episodes the target was in view for **91 frames and detected 0 times**,
and the agent issued **0 STOPs** — it never commits because it never detects.
Doubling the step budget from 500 to 1000 doubled goal-storey arrivals (3 → 6)
and left STOPs at exactly 0, which rules out the budget as the cause.

On DualMap's released benchmark, whose queries include ordinary scene objects,
the same baseline *does* terminate — 107/107 trials issued a STOP — and scores
2/107. Its two successes are `chair` and `decoration`; every YCB target scores
zero, including the four COCO-nameable ones (banana 0/6, bowl 0/11, mug 0/6,
scissors 0/16). So the honest statement is narrower than "limited vocabulary":
the baseline navigates to furniture-scale scene objects and fails on small placed
objects whether or not its detector has a word for them. (107 of 186 trials at
the time of writing; the remainder are in progress.)
