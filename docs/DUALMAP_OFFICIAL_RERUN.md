# Rerunning DualMap on its own released benchmark

DualMap, run unmodified on the dataset its authors released, because the published
numbers cannot be reproduced without an external harness. This is the source of the
30.2% cross-anchor figure the paper sets beside its own (Table I), and of what the
benchmark can and cannot resolve. Our own pipeline runs the same 186 trials from the
same start poses under the same rule, so the comparison is paired trial by trial;
where its numbers are is the last section.

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

## Results: DualMap

Full tables: `outputs/dualmap_official_bench/RESULTS.md`, aggregated across seeds by
`scripts/report_dualmap_official.py`. Three seeds (12, 13, 14),
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

## What the benchmark can resolve

This section asks a prior question — whether the benchmark's rule can resolve what it
claims to — and it needs no system at all. `scripts/analyze_benchmark_validity.py`
writes the tables as `BENCHMARK_VALIDITY.md`.

Take a do-nothing agent: it ignores the dynamic change entirely, walks to where the
object sat in the *static* scene, and stops. It performs no perception, no search
and no re-detection. Score it under the benchmark's own 1 m rule against the
object's *new* position.

| | in-anchor | cross-anchor |
|---|---:|---:|
| Median object displacement | **0.70 m** | 5.56 m |
| Do-nothing agent's SR | **74.1%** | 9.4% |
| DualMap (measured, seed 12) | 64.8% | 30.2% |
| Ours, `dualmap_protocol_osg` (2026-09-09, superseded) | 33.3% | 22.6% |

In-anchor moves objects less far than its own success tolerance, so doing nothing
outscores both real systems. That condition cannot separate a system that
re-detects a moved object from one that never noticed it moved, and an in-anchor SR
should not be read as evidence about dynamic-scene handling. Cross-anchor, whose
objects move a median 5.56 m, is the informative half.

Six trials move the object by less than a centimetre and are labelled dynamic
anyway — including **all three cross-anchor `mug` trials**, so `mug` contributes no
cross-anchor evidence at all. Separately, 102 of the 107 dynamic trials have exactly
one valid instance, so a wrong stop is a wrong *place* rather than a sibling
instance; the exception is `mug` in 00848, which the release gives two instances
sharing one semantic id.

## What DualMap's success rate is made of

A claim worth stating carefully, because the obvious version of it is false.

**DualMap does not replay stale positions.** Its saved local paths record where it
planned to go. Restricted to the 88 planned goals across three seeds whose object
moved more than 1.5 m — so that "aimed at the new position" and "aimed at the old
one" are distinguishable — **30 aimed at the new position and 1 at the old**. When
DualMap plans a local path it has genuinely re-detected the object, including
objects our own detector never sees. Any claim that it navigates to where the object
used to be is not supported.

What is true is that its success rule credits arrival without identification. A
DualMap trial can score having never planned a local path at all: its abstract map
text-matches a piece of furniture, the agent walks to that anchor, and the anchor
happens to lie within a metre of the object. Over three seeds:

| Split | Anchor-only successes | What they stopped at |
|---|---:|---|
| Static | 36 | `countertop` ×7, `cabinet` ×5, `shower wall` ×5, `couch` ×4, … |
| In-anchor | 34 | `couch` ×10, `kitchen island` ×9, `dining table` ×6, `desk` ×5, `bed` ×4 |
| Cross-anchor | 14 | `bed` ×9, `kitchen island` ×3, `dining table` ×1, `couch` ×1 |

All six of 00848's `scissors` trials are this pattern — `candidate: bed`, no local
path ever planned, three of them scored. The anchor score is identical across
layouts (`desk` at 0.5651295544750305 for `plate` in 0116/0117/0118), which is what
you expect from a static text match against the prior map: it does not depend on
where the object went.

Scoring success only when the system localised the object gives, over three seeds,
static 68.1% → 51.1%, in-anchor 62.3% → 43.2%, cross-anchor 30.2% → 23.9%.

## What a paired comparison cannot share

| | Ours | DualMap |
|---|---|---|
| Camera height | 0.88 m | 1.5 m |
| Sensor | 1280x960, 79 deg H / 63 deg V | 1200x680, 90 deg H / 59 deg V |
| Motion | discrete, 0.25 m / 30 deg | continuous path follower |
| Prior map | our own pass over the static scene: 682 / 800 / 618 tracks in `maps_released`, the maps behind the paper's run (689 / 343 / 520 in September's `maps_v5`) | the released `global_map/`: 21 / 31 / 35 objects |

The prior-map row is the largest asymmetry and it favours us: each system was
given one pass over the same static scene and preloads whatever its own pipeline
produces, and ours produces an order of magnitude more objects. It is stated
because a reader who does not know it will read the static split wrongly -- our
prior is far richer and still answers fewer queries, because it is labelled from a
fixed detector vocabulary rather than retrievable by text (mechanism 1 in
`docs/history/DUALMAP_PROTOCOL_EARLY_ARMS.md`).

The embodiment difference was measured rather than assumed. DualMap's start poses
land on our navmesh with a median snap offset of 0.000 m (max 0.063 m), and the
shortest-success-path optimum recomputed on our navmesh differs from DualMap's by
a median of 0.111 m. The pairing is real, not approximate.

**Query strings are not shared, deliberately.** The benchmark asks for an object;
which string to hand a text-conditioned detector is part of the system. DualMap
asks MobileCLIP for `pitcher`; the probe behind this repository's
`YCB_TARGET_LABELS` measured `pitcher` at 0.00 on that asset at every resolution
and `blue plastic pitcher` at 0.71. Three of the eight YCB queries are renamed for
our detector -- `soup can` to `tin can`, `plate` to `red plate`, `pitcher` to
`blue plastic pitcher` -- and the other five and all 60 HM3DSem class queries pass
through unchanged. The mapping already existed in the repository before the first
of these runs; every trial records both strings.


## Our pipeline on this benchmark

The harness above is shared: `src/osg/eval/dualmap_release.py` defines the trial list, what
each query points at and the success rule, and both `scripts/run_dualmap_released_native.py`
and our `dualmap_protocol` eval mode (`src/osg/sim/dualmap_env.py`) import it, so the two
systems are scored by the same code. Each of our trials starts where that trial's DualMap run
started, read from its own `result.json`, so the comparison is paired trial by trial.
`tests/unit/test_dualmap_release.py` recomputes the target geometry and every per-attempt
distance of the DualMap trials on disk from that module.

Our numbers on it:

- the paper's arm, `dualmap_osg_unified`: `docs/PAPER.md` Table I and Appendix B; whether it
  re-derives today: `docs/CROSS_ANCHOR_REPRODUCTION.md`;
- the September arms before it (`dualmap_protocol_osg` at 37.6% overall, then the tight
  ring), their gap analysis against DualMap, and the rescoring of the old authored-benchmark
  headline: `docs/history/DUALMAP_PROTOCOL_EARLY_ARMS.md`.
