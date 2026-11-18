# DualMap's route, transplanted: what happened and why

The probe (`docs/history/FEATURE_MEMORY_PROBE.md`) ended by saying that threshold-based
admission is refuted but DualMap's actual mechanism -- argmax over accumulated
objects near a reached anchor -- was untested. This is that test.

## What was built

`feature_memory.local_pick_on_arrival`. On arriving at a searched surface with no
candidate from the label path, rank the live tracks near it by cosine between the
query text and their accumulated CLIP image feature, and commit to the best.
Argmax, never a threshold: `sorted_candidates[0]`, as in DualMap's
`find_best_candidate_with_inquiry`. The pick is a proposal and not a verdict --
reachability, the VLM, the identity channel, the absence sensor and the attempt
protocol judge it exactly as they judge a candidate the label proposed, and
nothing here can stop an approach on a cosine.

Three arms, all off the shipped `island_close` configuration, all on the 24
dynamic mug and scissors trials (`data/splits/dualmap_mug_scissors.json`):

* `_feat` -- the pick, on cosine alone. DualMap's rule.
* `_featp` -- fused with the presence belief, which DualMap has no equivalent of:
  its memory of a failed candidate is an ignore list discarded when the query
  ends, so its argmax cannot say "I have walked past that and never seen it
  again". Score becomes `exp(-beta * (best - sim)) * presence.p`, never the
  cosine as a probability.
* `_featsz` -- the pick bounded to tracks small enough to be the object.

## Result

| slice | base | `_feat` | `_featp` | `_featsz` |
|---|---:|---:|---:|---:|
| mug in-anchor | 3/3 | 1/3 | 1/3 | 0/3 |
| mug cross-anchor | 2/3 | 3/3 | 2/3 | 2/3 |
| scissors in-anchor | 2/9 | 3/9 | 0/9 | 1/9 |
| scissors cross-anchor | 1/9 | 1/9 | 2/9 | 1/9 |
| **total** | **8/24** | **8/24** | **5/24** | **4/24** |

None of the three helps. The mechanism is not inert -- 44 picks over 24 trials,
every one committed, 835 tracks considered -- so this is a refutation and not an
untested knob. The presence fusion changed the answer 10 times and lost three
trials doing it. `_feat` is a flat null on a subset whose noise floor is about
three trials; the other two are below it.

The size bound did not rescue it either, and its counters say why: it rejected
2333 tracks as too large and still chose a lamp 9 times, a pillow 8 and a bed 4.
The bound is on the track's own mapped ellipsoid, and our ellipsoids for large
furniture are frequently small fragments of it -- a lamp fitted from a few
distant views is well under 0.6 m across. Filtering on mapped geometry cannot
separate "a small object" from "a small piece of a big one".

## Why, and it is not the matcher

Two measurements say the same thing from different ends.

**Our search does not stand where the object is.** Counting every surface the
search arrived at, and how many were within 2 m of the object:

| query | surfaces arrived at | within 2 m | picks fired | picks within 2 m |
|---|---:|---:|---:|---:|
| mug | 19 | 1 | 11 | **0** |
| scissors | 92 | 10 | 30 | 3 |

DualMap's global text match commits to `dining table` on all 21 of its mug
trials across three seeds and ends 0.33-0.94 m from the mug. Its local inquiry
then only has to choose among the things on that table. Transplanting the local
half into a search that arrives somewhere else just commits to whatever is
nearest, and the object is not there to be chosen.

**What it picks is furniture.** Of 44 picks: pillow 13, lamp 5, bench 4, picture
4, chair 4 -- 37 of 44 furniture-sized. This is not the cosine misbehaving; a
crop of a bed genuinely is a better match for "a photo of a mug" than forty
pixels of the real mug beside it. DualMap never meets the problem because its
detector proposes the small object: its mug arrives labelled `speaker`, a tight
crop, and its local map is full of object-shaped things. Ours mostly proposes
the furniture the object rests on -- on the 160 dumped mug frames a box small
enough to BE the mug covers it on only 39, and one covering the pixel at all on
84, the rest being cabinets and beds whose boxes happen to contain it.

`_featsz` tried to address the second with a bound from mapped geometry alone
(`local_max_extent_m: 0.6`; every movable query here is under 0.3 m across). It
could never address the first, and it did not manage the second either: 4/24.

## What this says about the gap

The chain is region proposal, then feature match, then anchor choice. DualMap
wins the mug at the first link, not the third: because its detector emits a tight
region on the object, its features describe the object, so its global anchor
inherits a mug-like feature and its local argmax has a mug-shaped thing to find.
Our features describe the furniture, so every stage downstream inherits that.

The matching method was never the difference. Adding it changes nothing on its
own, which is what these arms measure.

## Safety of the branch

With `feature_memory.enabled: false` the shipped `island_close` arm reproduces
exactly on this code -- 6 of 6 re-run trials identical in steps, success and
final distance to the millimetre (`outputs/osg_feat_identity`). All three flags
default off, `pytest tests/unit` is green at 920, and the flattened config diff
is additions under `feature_memory.*` with nothing moved.

## Recommendation

Do not run these arms on the full 107; a mechanism that is null on the 24 trials
it was aimed at, with its counters showing it firing 44 times and choosing
furniture, has already answered. Keep the code behind its flags as the
measurement, not as a configuration to adopt.

The lever the evidence points at is the first link in the chain: a region
proposal stage that emits the small object, which is what DualMap's detector
gives it for free. The class-agnostic proposal replay in
`scripts/legacy/offline_appearance_proposals.py` reached 98.1% plausible proposal
recall with FastSAM-s where our detector reaches 39/160 on the mug, so the
offline evidence for that link already exists; what is unknown is its cost in
the live loop, which `docs/history/APPEARANCE_REID_AND_CANDIDATE_BUDGET_PLAN.md` never
budgeted.
