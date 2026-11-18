# Appearance Re-identification and Distinct-Candidate Budget Plan

## Goal

Improve dynamic-object search on both the released DualMap layouts and our
authored layouts without changing the benchmark into an anchor-only lookup.
The first two changes to evaluate are:

1. appearance-based re-identification of a stale mapped object in live RGB;
2. an episode-level budget on distinct failed candidate tracks.

The frozen reference run is `outputs/osg_dualmap_tightring`. It contains 107
dynamic trials: 24/54 in-anchor successes (44.4%) and 15/53 cross-anchor
successes (28.3%). Its failure funnel is dominated by search and targeting,
not final standoff distance: 18 targets were never in view, 30 were visible but
never named by the detector, and 17 more were lost after naming or commitment.
The run also committed frequently to wrong locations (53/97 in-anchor commits
and 97/116 cross-anchor commits) and exhausted the step budget in 47 trials.

This document defines offline gates before either behavior is enabled in the
agent. Offline results are diagnostics and do not constitute a new SR/SPL
result.

## Option 1: Appearance-based live re-identification

### Intended runtime behavior

At map creation time, retain several high-quality appearance embeddings for
each object track, together with their crops and observation metadata. At
dynamic-search time, score class-agnostic live regions against the stale
target's stored appearance prototypes. A strong appearance match may admit or
boost a candidate even when the open-vocabulary detector does not emit the
target class.

The initial controlled implementation should use the same MobileCLIP image
encoder as DualMap so that the comparison is not confounded by another
backbone. The locally staged `mobileclip_blt.ts` is a text-only export, however,
so the first offline diagnostic uses the complete local OpenAI CLIP ViT-B/32
checkpoint in `data/clip/ViT-B-32.pt`. This tests the mechanism, not the final
backbone choice. A runtime A/B must repeat the gate with a complete MobileCLIP
image checkpoint before claiming a controlled DualMap comparison. A later A/B
may compare a correspondence-oriented representation such as DINO. Multiple
prototypes are required because one static crop is fragile under viewpoint,
scale, and occlusion changes.

Runtime matching must obey these constraints:

- no ground-truth target position, projected pixel, relocation metadata, or
  semantic ID is available to the matcher;
- proposals are class-agnostic and originate from the current RGB frame;
- similarity is used with geometric consistency, floor scope, temporal
  confirmation, and existing presence/search evidence;
- appearance can recover a missed class label, but it cannot bypass collision,
  reachability, or final visual verification;
- thresholds are fixed on a development split and not tuned per scene, target,
  or layout.

### Offline gate

The frozen `gt_dump` frames provide the target's projected pixel only for
analysis. The first offline experiment therefore measures two different
questions and reports them separately:

1. **Representation gate (oracle region):** compare static-map target crops to
   live crops centered on the ground-truth projected target. This asks whether
   the stored appearance can survive the visual domain change. It does not test
   proposal generation.
2. **Retrieval gate (image regions):** rank a target-containing crop among
   same-frame multi-scale distractor regions. The dump stores a projected
   centre but no target box or mask; `gt_px` is a detector mask covering that
   point and must not be treated as target size. The positive crop is therefore
   centred by ground truth and sized from the known YCB mesh extent, camera
   focal length, and recorded range. This estimates whether a class-agnostic
   region stage can surface the target, but remains an upper-bound diagnostic,
   not deployable navigation evidence.

Report cosine-similarity distributions, pairwise AUC, top-1/top-5 retrieval,
median positive rank, and results by split and target class. Report the
`seen-never-named` subset separately because it is the direct opportunity for
appearance recovery. Also report reference coverage: trials without a usable
static target crop cannot benefit from this design as specified.

Proceed to an online implementation only if live positives separate from
distractors materially above chance and the result is not carried by one target
class. The next required gate is then a saved-proposal replay using proposals
generated without ground truth.

## Option 2: Distinct failed-candidate budget

### Intended runtime behavior

The current identity-rejection limit is per track. An episode can consequently
spend most of its budget approaching several different wrong tracks. Add an
episode-level set of distinct candidates that have produced conclusive failure
evidence.

A track enters the set only after one of these observable outcomes:

- visual verification rejects its identity with adequate target visibility;
- inspection produces confident negative presence evidence;
- the candidate is reached and the target is absent under the existing
  arrival/inspection contract.

Navigation uncertainty alone does not count as an identity failure. In
particular, `path_consumed`, planner failure, or an interrupted approach must
remain retryable or be handled by the navigation-failure policy. Relinked track
IDs count as one physical candidate, and candidate identity remains floor
scoped.

After the configured number of distinct failures, the agent stops selecting new
object-track hypotheses temporarily and reserves the remaining episode budget
for container surfaces, high-posterior anchors, and frontier coverage. A newly
observed high-confidence target or appearance match can still interrupt this
search mode.

### Offline gate

Replay the chronological candidate and goal logs from the frozen run for
budgets of 1, 2, 3, and 4. Two analyses must remain distinct:

1. **Observable replay:** trigger only from recorded rejection/absence events
   that the online policy could have known. Measure trigger coverage, trigger
   step, remaining steps, subsequent commits, and repeat pressure.
2. **Oracle waste audit:** use authored destination geometry after the run to
   label committed tracks as correct or wrong. Estimate how many wrong commits
   and steps occur after the Nth distinct wrong candidate. This provides an
   upper bound on recoverable waste; it is not a valid runtime decision rule and
   cannot predict converted successes.

Do not convert saved steps into hypothetical successes. A budget is promising
only if it fires early in budget-exhausted failures, avoids firing before most
successful target commitments, and leaves enough steps for a meaningful search
phase. The report must show false-trigger risk on successful episodes and break
results down by in-anchor and cross-anchor layouts.

## Integration sequence

1. Run both offline gates against the immutable tight-ring output and archive
   JSON plus Markdown reports under
   `outputs/osg_dualmap_tightring/offline_improvements/`.
2. If appearance passes the representation gate, add offline class-agnostic
   proposal generation and rerun retrieval without oracle proposal geometry.
3. Implement telemetry before behavior: appearance scores/prototype IDs,
   proposal ranks, physical candidate IDs, conclusive-failure reasons, budget
   transitions, and post-budget search allocation.
4. Add deterministic unit tests for prototype persistence, similarity scoring,
   floor scoping, relinking, conclusive versus navigation failures, and reset at
   episode boundaries.
5. Run a paired online A/B with four arms: baseline, appearance only, candidate
   budget only, and both. Keep episode IDs, maps, seeds, detector, VLM endpoint,
   and stopping rules identical.
6. Select thresholds on a development subset, freeze them, then report the held
   out in-anchor/cross-anchor SR, SPL, failure funnel, travel, and mechanism
   counters. Repeat the chosen configuration on our authored 15-scene data.

## Success criteria and safeguards

The combined change is accepted only if it improves held-out SR without losing
target-admission, presence, prior-map-load, inspection, or floor telemetry. It
must not increase anchor-only stops, use layout ground truth at runtime, decay
tracks across floors, or approach stale coordinates from another floor.

The desired 70/50 SR remains a program target, not an expected outcome from
these two mechanisms alone. Appearance re-identification addresses the
visible-but-unnamed failures; the distinct-candidate budget addresses wasted
targeting and preserves time for the separate search-coverage problem.

## Reproduction

The scripts accept the frozen run directory as an explicit argument, never
modify episode records or maps, and write only derived reports beneath
`outputs/`:

```bash
python scripts/legacy/offline_appearance_retrieval.py \
  --run outputs/osg_dualmap_tightring \
  --maps outputs/maps_v5 \
  --checkpoint data/clip/ViT-B-32.pt \
  --device cuda

python scripts/legacy/offline_candidate_budget.py \
  --run outputs/osg_dualmap_tightring

python scripts/legacy/offline_appearance_proposals.py \
  --run outputs/osg_dualmap_tightring \
  --maps outputs/maps_v5 \
  --checkpoint data/clip/ViT-B-32.pt \
  --proposal-model outputs/dualmap_comparison/vendor/DualMap/model/FastSAM-s.pt \
  --device cuda
```

## First offline result (2026-09-09)

The reports are
`outputs/osg_dualmap_tightring/offline_improvements/APPEARANCE_RETRIEVAL.md`,
`outputs/osg_dualmap_tightring/offline_improvements/APPEARANCE_PROPOSALS.md`,
and
`outputs/osg_dualmap_tightring/offline_improvements/CANDIDATE_BUDGET.md`, with
machine-readable JSON beside each.

### Option 1 result: promising, proceed to proposal replay

The static-map reference gate covered 78/107 trials. On 427 sampled live frames
from 62 episodes, the correct reference achieved 80.4% pairwise AUC, 43.8%
top-1 retrieval, and 58.8% top-5 retrieval among about 50 regions per frame;
chance top-1 was 2.0%. A same-scene wrong-object reference control fell to
59.2% AUC and 11.6% top-1, so most of the measured lift is identity-specific
rather than target-centre salience alone.

On the directly relevant visible-but-never-named subset with a valid static
reference, the result was 83.5% AUC, 53.7% top-1, and 67.2% top-5 over 67
sampled frames from 12 episodes. Performance is strongly class-dependent:
banana, soup can, and pitcher are strong; bowl is useful; cracker box is weak;
and plate is close to unusable. The five uncovered scene/target pairs are all
scissors pairs plus the `00848` mug and `00880` cracker box. Consequently only
12 of the 30 visible-but-never-named failures currently have a valid stored
reference.

This passes the representation gate, not the runtime gate. The next offline
test replaced the oracle-centred crop with class-agnostic FastSAM-s proposals
created from RGB alone. Ground truth was consulted only after proposal
generation to decide whether a mask covered the projected target centre. A
second evaluation-only size check rejected boxes larger than 16 times the
square of the known YCB extent projected at the recorded range.

Across the same 427 frames, plausible proposal recall was 98.1%. Correct-target
appearance ranking reached 42.2% end-to-end top-1 and 61.1% end-to-end top-5;
the same-scene wrong-reference control reached 11.1% and 29.5%. On the 67
sampled visible-but-never-named frames with a static reference, end-to-end
top-1 was 38.8% and top-5 was 65.7%. This passes the first query-free proposal
gate and supports a guarded online pilot.

It is still not a complete runtime validation. The dump has no true target
mask, so a centre-covering FastSAM mask can still be a compact piece of the
supporting surface. The pilot should save ranked crops, require temporal and 3D
consistency, and use appearance as candidate admission/boost evidence rather
than an immediate STOP decision. It should also preserve more than one static
prototype and improve static reference coverage; otherwise appearance
re-identification will not touch the scissors-heavy part of the failure
funnel.

### Option 2 result: do not choose a cap from current logs

All 60 entries in `candidate_reject_log` are `unreachable`, which is navigation
uncertainty and was correctly excluded from the distinct identity-failure set.
The remaining observable proxy is an evaluator-confirmed failed attempt linked
to its most recent committed candidate.

A budget of one distinct failed candidate triggers in 62/107 episodes,
including 21/39 eventual successes, so it is too aggressive. A budget of two
triggers in 24/107 episodes and only 7/39 successes, but it covers 0/47
budget-exhausted failures: those failures usually never produce a second
conclusive attempt before time expires. Budgets of three and four never fire.
The oracle audit confirms that wrong-target pressure exists, but it cannot
supply a deployable trigger.

Do not enable option 2 from this replay. First log candidate lifecycle events
that distinguish visually confirmed identity/absence failures from
`path_consumed`, planner failure, and reachability uncertainty. Then repeat the
observable replay. In parallel, a pre-approach ranking/diversity rule may be
more relevant than a post-failure cap because the budget-exhausted cases are
often stuck before accumulating two confirmed failures.
