# Rerunning DualMap on its own released benchmark

## What the official repository provides

Upstream: `https://github.com/Eku127/DualMap.git`, commit `157235e`, used unmodified
(`git status` clean). Dataset: the authors' released `HM3D_collect` (scenes
`00829-QaLdnwvtxbs`, `00848-ziup5kvtCCR`, `00880-Nfvxx8J5NCo`), each shipping
`static_scene_config.json`, `dynamic_scene_config/{in_anchor,cross_anchor}/`,
`global_map/`, `class_bbox.json` and `class_num.json`.

Two facts determine everything below.

1. **Neither official repository computes navigation metrics.** Grepping DualMap
   and `habitat-data-collector` for `spl|success_rate|geodesic|path_length`
   returns nothing. `evaluation/` holds only semantic-segmentation code, which
   is the mapping table, not the navigation table.
2. **The documented navigation workflow is manual.** `resources/doc/app_simulation.md`
   has the operator edit `config/actions.yaml` (`get_goal_mode: inquiry`,
   `inquiry_sentence`, `calculate_path: true`), watch Rerun/RViz, and judge the
   outcome by eye: "If no local path is planned, the navigation attempt is
   considered failed."

So following the guide reproduces the *demo*, not the numbers. Measuring SR and
SPL over 186 trials × 3 seeds requires an external harness. Ours
(`scripts/run_dualmap_released_native.py`) imports the released DualMap and
collector packages and drives them through their own public interfaces
(`actions.yaml`, `core.calculate_path`, `core.trigger_find_next`); it replaces
ROS 2 only as transport. No upstream source file is modified.

## Metrics

DualMap defines SR as "the percentage of queries in which the agent stops within
1 meter of the queried object", and for dynamic scenes "success further requires
finding the target within three attempts". That is the criterion implemented here.

**DualMap does not publish SPL.** Paper Table II and Appendix Tables IX/X report
success only — Tables IX/X are literally binary `Type | Success` columns. SPL in
our tables is measured by the harness against Habitat's geodesic shortest path
and has no published counterpart to validate against.

Published SR (paper Table II):

| Split | 00829 | 00848 | 00880 | Trials | Avg. |
|---|---:|---:|---:|---:|---:|
| Static | 73.1% | 69.2% | 69.2% | 78 | 70.5% |
| In-anchor | 66.7% | 66.7% | 61.1% | 54 | 64.8% |
| Cross-anchor | 55.6% | 61.1% | 64.7% | 53 | 60.3% |

## Protocol reconstruction

* **Dynamic splits** come straight from the released layout JSONs: 6 queried
  objects × 3 layouts × 3 scenes, minus the cracker-box trial that the appendix
  does not report for `00880` `0128-2`. 54 in-anchor + 53 cross-anchor = 107,
  matching the paper.
* **Static split** uses the query lists in Appendix Table VIII. Those total
  **79** (26 + 27 + 26), while Table II reports **78** trials, and the per-scene
  percentages (73.1% = 19/26, 69.2% = 18/26) imply 26 per scene. The paper does
  not say which `00848` query was dropped, so all 79 are run and the discrepancy
  is reported rather than silently resolved.
* **Targets.** YCB objects use the released poses. HM3DSem classes use the
  instance boxes in `class_bbox.json`, which the release ships "for evaluation".

## Deviations from the harness's previous behaviour, and why

Three changes were needed. Each is in our harness, not in DualMap.

1. **A false match no longer ends the trial.** DualMap clears
   `ignore_global_obj_list` whenever a local path is planned (`dualmap/core.py`),
   so after a false match it has forgotten which anchors it already visited and
   the next global plan re-selects the one just rejected. The harness previously
   terminated the trial there, which contradicts the paper's three-attempt rule
   and the guide's `trigger_find_next` ("very useful in dynamic object
   navigation"). The harness now restores the anchors already tried and requests
   a fresh global plan. Only DualMap's public state is set.
2. **Distances are measured to the object, not its centroid.** `class_bbox.json`
   gives full box extents. A bed is 2.33 m across, so its centre is more than a
   metre from any navigable floor and "within 1 m" would be unsatisfiable by
   construction. Distances are now taken to the box. YCB targets keep a zero
   extent, so the dynamic splits are mathematically unchanged.
3. **The collector seed is a run parameter**, so the agent start pose can be
   varied across seeds. The official protocol has no seed: the layouts are fixed
   files, and seeding is our addition to estimate run-to-run variance.

## Results

Full tables: `outputs/dualmap_official_bench/RESULTS.md`. Three seeds (12, 13, 14),
186 trials each; 556 of 558 completed, 2 lost to an upstream crash.

| Split | SR (ours, 3 seeds) | SR (published) | SPL (ours) |
|---|---:|---:|---:|
| Static | 68.1% ± 1.1 | 70.5% | 0.327 ± 0.044 |
| In-anchor | 62.3% ± 2.8 | 64.8% | 0.431 ± 0.019 |
| Cross-anchor | 30.2% ± 0.0 | 60.4% | 0.088 ± 0.019 |

Static and in-anchor reproduce within ~2.5 points. **Cross-anchor does not
reproduce**: 30.2% against a published 60.4%, and the seed-to-seed standard
deviation is 0.0 — 16/53 on every seed. That is a systematic difference, not
sampling noise, and it survives the false-match fix that lifted cross-anchor
from 22.6% to 30.2%.

What was ruled out for cross-anchor, with evidence:

* Not a missing candidate: all 53 trials produce a global plan every seed.
* Not an unwalked local path: 6+ keyframes always remain after one is planned.
* Not a truncated retry budget: 42/53 trials now use all three attempts.
* Not a broken online update: the preloaded abstract map for `00829` holds 21
  objects, 9 of them carrying `related_objs`, and candidates do change across
  attempts in 21/53 trials.
* Not a similarity ceiling artefact: scores top out at 0.640 in every split,
  including in-anchor, which is MobileCLIP's natural cosine range.

What remains is candidate *ranking* after relocation: 30 of 37 cross-anchor
failures end more than 3 m from the target, so the agent commits to wrong
anchors rather than narrowly missing correct ones. The release publishes no
per-trial start poses, so the authors' exact observation trajectories cannot be
reconstructed to close this gap.

## Two upstream crashes

`00848 static kettle` and `00880 static painting` raise `ValueError: zero-size
array to reduction operation minimum` at `utils/object_detector.py:1792`
(`update_bbox` via `overlap_check`), an empty-mask edge case entirely inside
DualMap. The failures are seed-dependent — seed 12 ran `kettle` cleanly. They
are excluded from the denominators and listed in the results rather than scored.

## Where the local path points

Full tables: `outputs/dualmap_official_bench/LOCAL_PATH_ANALYSIS.md`, regenerated by
`scripts/analyze_dualmap_local_paths.py`. Saved paths are in DualMap's z-up frame;
Habitat `(x, z)` is `(path_x, -path_y)`, verified by the global path's first point
coinciding with the agent pose and its last point with the local path's first point.

| Split | Local paths planned | Goal within 1 m of object | Median goal error |
|---|---:|---:|---:|
| Static | 189 | 63.5% | 0.57 m |
| In-anchor | 104 | 67.3% | 0.65 m |
| Cross-anchor | 98 | 38.8% | 2.81 m |

Cross-anchor's failure is **localisation, not control**. When the goal is right the
agent nearly always reaches it (33/38 cross-anchor, 67/70 in-anchor); the problem is
that 61% of cross-anchor local paths aim at the wrong object, a median 2.81 m away.
The chain is: wrong anchor selected globally, so the local map holds the wrong
objects, so the best local match is a different instance.

Position-correct SR — crediting success only when DualMap actually localised the
object and the agent reached it — is **50.2% static, 41.4% in-anchor, 20.8% cross-anchor**.

This makes the headline SR *generous* rather than harsh. Of 48 cross-anchor successes,
only 33 involved correct localisation; 14 had no local path at all (the agent reached
the anchor and the object happened to lie within 1 m), and 1 succeeded with a wrong goal.

It also rules out the reconciliation with the published number. Counting any planned
local path gives cross-anchor 61.6%, close to the published 60.4% — but only 38 of
those 98 paths point at the real object. Requiring correctness drops it to 23.9%,
*below* our 30.2% headline. Every criterion that checks where the object actually is
lands in the 20-30% band:

| Criterion (cross-anchor) | SR |
|---|---:|
| Any local path planned | 61.6% |
| Agent within 1 m of object (headline) | 30.2% |
| Local-path goal correct | 23.9% |
| Local-path goal correct and reached | 20.8% |
