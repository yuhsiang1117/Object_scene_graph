# Fusing the proposal stage: appearance hypotheses, not namings

The class-agnostic proposal stage (FastSAM-s regions ranked by MobileCLIP-S2
against the query phrase) lost 18 trials on the full 107 when its output was a
same-label `Detection` (`docs/history/REGION_FALSE_ADMISSION.md`): the per-frame bar
admits on 13% of frames with no object in view, and nothing downstream could
tell a proposed bowl from a detected one. The direction from the user was to
keep the stage in the full pipeline, aligned with DualMap, and change how it
is fused.

## The fusion

DualMap's matching unit is an object's feature accumulated over views, not a
frame's argmax. So:

* The stage runs on **every keyframe** the detector did not name the target on.
* A region past the per-frame bar (tau 0.24) enters the map as a **proposal
  observation** -- `Detection.source = "proposal"`, carrying its region
  feature -- and the track keeps the **running-mean feature** and its cosine
  to the query phrase (`proposal_sim`).
* Proposals form a **second population**. They associate only with
  proposal-only tracks; the detector's tracks never see them; the presence
  filter sees neither proposal tracks nor proposal detections; proposal-only
  tracks are never linked; they draw from their own random generator and take
  ids from 1,000,000 up; `tracks()` hides them unless asked, so the scene
  graph, the search anchor and the prior map are exactly the detector's map.
* A **proposal-only track** may become a candidate only behind every named
  track, ranked among its own kind by `proposal_sim`, and only past
  `commit_min_obs` observations and `commit_tau` on the mean feature.
* The proposal sensor answers `_best_target_detection` **only while the agent
  works a proposal-only track**; a named track gets the detector.

## Verified by identity

`_sensor_v4_observe` (commits off) must reproduce `_sensor_v2` trial for
trial. It took four rounds to get there, each found by that check:

| round | trials diverging | the leak |
|---|---:|---|
| 1 | 7/14 | the proposal sensor stopped and steered NAMED approaches; proposal score displaced the best view |
| 2 | 5/15 | proposals associated into named tracks, lent presence, linked, became the search anchor |
| 3 | 3/19 | shared random generator (ellipsoid depth sample) and shared id counter |
| 4 | 1/43 | `in_anchor__0118__soup_can`: identical through step 201, then the detector's own scores differ on identical frames (0.596 -> 0.527). Rerun alone on an idle GPU, the observe arm reproduces v2 exactly (269 steps, 0.647 m): not a leak but run-to-run GPU numerics under two co-resident workers. That is the identity check's floor. |

Final: 21/22 on the phantom split, 21/21 on the perception split.

## The bar, set on tracks against the authored positions

Two observation runs, 43 episodes (37 with authored positions): true = a
proposal-only track within 1.0 m of the object.

| population | n | n_proposal_obs p50/p90 | proposal_sim p10/p50/p90 |
|---|---:|---:|---:|
| true (perception trials) | 14 | 4/12 | 0.284/0.294/0.325 |
| true (named trials) | 15 | 3/15 | 0.251/0.302/0.321 |
| phantom | 110 | 1/5 | 0.242/0.253/0.280 |

| min_obs | tau | episodes with a phantom past the bar | perception episodes whose true track passes |
|---:|---:|---:|---:|
| 4 | 0.26 | 9/37 | 8/11 |
| **4** | **0.28** | **5/37** | **8/11** |
| 4 | 0.29 | 3/37 | 5/11 |
| 4 | 0.30 | 1/37 | 3/11 |
| 5 | 0.28 | 4/37 | 6/11 |

Chosen: **min_obs 4, tau 0.28**. A phantom past the bar costs one attempt of
three, not the trial, it must be re-seen four times on the same region, and
where a true track and a phantom both pass the true one outranks it by
feature (measured: 0.325 vs 0.286 in the same trial). Scissors reaches 22
observations at 0.330 on the one trial where it was in view long enough --
the encoder can read it once the region is tight and the views accumulate.

The per-frame bar (0.24) now only gates OBSERVATIONS. Lowering it would let
the mean accumulate on weaker objects at the price of more phantom tracks that
never pass; that is a separate arm.

Arm: `dualmap_protocol_osg_sensor_v4_fuse`. Judge it on
`goal_commit_log[].proposal_only` and the paired 21 perception trials before SR;
the benchmark's noise floor is +-3.
