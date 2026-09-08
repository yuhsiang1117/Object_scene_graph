# A/B results log — closing the SR gap to ASCENT

Running record of every staged change, its measured effect, and the decision
taken. This is both the working log and the source material for the paper's
ablation table.

**Target.** HM3D ObjectNav v1 val, `success_distance=0.1`, 500 steps,
`allow_sliding=false`, sensor-only. ASCENT (arXiv:2505.23019) scores **63%**
under exactly these conditions. Our starting point was ~49% measured under
*looser* conditions (`success_distance=0.18`, 100 episodes, habitat's ground
truth navmesh), so the real gap is larger than the headline numbers suggest.

## How to read this log

Every A/B runs **50 paired episodes** (`dev50`, or `dev50_mf` for the
multi-floor stages). At that size SR has a ~7 percentage-point standard error,
so **the SR delta on its own is not evidence**. Two things are:

1. **Paired flips** — both arms run identical episode ids, so each episode is
   its own control. Report `gained` / `lost` / `net`.
2. **Mechanism metrics** — per-episode counters that move on nearly every
   episode and therefore resolve at n=50 where SR cannot.

Decision rule:

| `net` | decision |
|---|---|
| `>= +3` | real effect — keep, enable by default |
| `-2 .. +2` | noise at this sample size — keep the code, leave the flag **off**, decide on the final full-split run |
| `<= -3` | regression — revert |

A stage whose mechanism metric moved as designed but whose SR did not is
recorded as "mechanism correct, underpowered" and kept.

**The 50-episode `net` values are not additive into an SR claim.** Only the
final full-split run (`+experiment=ascent_matched`) produces a quotable SR.

Reproduce any row with:

```bash
python scripts/run_eval.py +experiment=ascent_matched +profile=<report|laptop> \
    eval=<dev50|dev50_mf> <the flag under test>
python scripts/compare_runs.py outputs/<baseline>/ outputs/<treatment>/
```

`compare_runs.py` refuses to interpret two runs whose `summary.json` protocol
fingerprints differ, which is the guard against accidentally comparing across
an evaluation-protocol change.

## Stage log

### S0 — protocol alignment, splits, instrumentation

No algorithmic change. Establishes the honest baseline everything else is
measured against.

Three settings did not match ASCENT and were corrected (the rest already did,
by inheriting the same `benchmark/nav/objectnav/objectnav_hm3d.yaml`):

| setting | was | now | source |
|---|---|---|---|
| `success_distance` | 0.18 | **0.10** | `habitat/task/objectnav.yaml:41` |
| `iterator_options.shuffle` | True (habitat default) | **False** | `ascent/experiments/eval_ascent_hm3d.yaml:21` |
| `iterator_options.max_scene_repeat_steps` | 10000 (habitat default) | **50000** | `ascent/experiments/eval_ascent_hm3d.yaml:20` |

Also fixed, and load-bearing for every later A/B:

- **`episode_id` is not unique.** habitat's ObjectNav loader assigns it per
  per-scene content file (`object_nav_dataset.py`: `episode.episode_id = str(i)`
  inside the per-file loop), so every scene contains an episode `"0"`. The old
  `eval.episode_ids` filter matched on the bare id, which would have selected
  one episode per scene from all 20 scenes instead of the intended subset — and
  it filtered *after* `env.reset()`, costing a scene load per skipped episode.
  Ids are now scene-qualified (`<scene>:<id>`) and the dataset is filtered
  up-front. Guarded by `tests/unit/test_eval_uid.py`.
- `scripts/analyze_stages.py` hardcoded `SUCCESS_DIST = 0.18`; it now reads the
  run's own `success_distance` from `summary.json`.
- `tests/unit/test_nav_agent.py::make_cfg` was missing
  `exploration.los_visibility_penalty` and `continuity_weight`, so two tests
  were failing on `main` before any of this work. Fixture completed; suite is
  green (75 passed).

**Splits** (`scripts/make_dev_split.py`, seed 42, fixed for the whole project —
regenerating them would break cross-stage comparability):

| split | content |
|---|---|
| `dev50` | 50 episodes, 2–3 from each of the 20 val scenes |
| `dev50_mf` | 50 episodes whose **goal is on another floor**, 3–4 from each of the 13 scenes containing such episodes |

#### Measured: only 19% of episodes require a floor change

The first version of `dev50_mf` selected episodes from *scenes* whose goal
view-points span more than a storey. Checked against ground truth, only
**15 of its 50** episodes actually needed a floor change — multi-storey houses
are full of episodes that start and end on the same floor. Across the whole v1
val split the start-to-goal height gap has a **median of 0.00 m**, and only
**376/2000 (19%)** exceed 1.0 m.

Selecting per *episode* gives 50/50 cross-floor (median gap 2.80 m) — 3.3× the
statistical power for the same 50 episodes. It also drops `ziup5kvtCCR`, whose
1.17 m scene span was a false positive, leaving 13 scenes, which matches the
earlier investigation's independently-derived "13 multi-floor scenes".

**This bounds the multi-floor work.** Perfect cross-floor navigation can only
move 19% of episodes, so the achievable gain is `0.19 × (target − current
cross-floor SR)` — around **+10 SR points** even if cross-floor SR went from
~10% to ~60%. The earlier framing (multi-floor 27.7% vs single-floor 68.6%)
was a *scene-level* split, so its 27.7% is mostly same-floor episodes in large
houses, not failures to climb. Multi-floor alone will not close the gap to 63%;
the same-floor majority has to improve too.

`compare_runs.py --split multi` keys on this ground-truth label. It previously
used the agent's observed trajectory span, which reads ~0 for an episode where
the agent *should* have climbed but never did — hiding precisely the failures
under test.

| A/B | status |
|---|---|
| `full_v1_navmesh` (0.18) vs `ascent_matched` (0.10) | **not run — superseded** |

Superseded rather than queued: every measurement since has run at the aligned
protocol, so `dev50` at 52% (geometric) / 56% (value map) already *is* the
honest baseline this comparison was meant to establish. Re-scoring the old
0.18 run would only quantify how much of the historical 49% was threshold
slack, which changes no decision. (It remains cheap if the paper needs it:
`success_distance` affects only the metric, not behaviour, so one run scores
both thresholds by re-scoring `distance_to_goal` offline.)

### S1 — cross-floor frame rejection

`mapping.floor_reject_m` (default 0 = off) drops a depth frame from the costmap
once the agent stands more than that far from the floor the map was started on.

Measured mechanism: a single frame taken a storey up, looking somewhere the
lower floor had not mapped, writes **~3700 cells** of the upper floor onto the
lower floor's grid. Nothing about the height band filters it — the costmap has
no idea the frame came from another storey (`tests/unit/test_floor_reject.py`
pins both the pollution and its suppression).

| A/B | status |
|---|---|
| `dev50_mf` × `floor_reject_m` ∈ {0.0, 0.4} | **not run — superseded by S2** |

Never measured on its own, and no longer worth it: with `multi_floor` on, the
rejection condition is `FloorStack.in_transit()` and `floor_reject_m` is only
the fallback for the single-map agent. The probe below is why.

> **Probe run (4 cross-floor episodes, flag on): it never fired.**
> `y_range_m = 0.00` and `frames_off_plane = 0` on every episode — the agent
> never left the starting floor at all. So on cross-floor episodes the loss is
> not map pollution; it is that **nothing drives the agent upstairs**: frontiers
> come from the 2D costmap, which only ever proposes same-floor goals.
>
> S1 remains worth keeping (it is a few lines and protects against transient
> stair-landing excursions), but the plan's +3–6 SR estimate for it looks
> unsupported, and the multi-floor gain is concentrated in **S3** (stair
> detection + a CLIMB state), not S1/S2.

### S2 — `FloorStack` (per-floor maps)

`mapping.multi_floor` (default off) gives each storey its own costmap, planner,
room segmentation and object tracks, selected by clustering the agent's
standing height. With the flag off the stack is pinned to a single layer and
behaviour is unchanged — verified byte-identical (same trajectories, steps and
`distance_to_goal`) on 3 episodes.

Three design points that only showed up once built, each now pinned by a test:

- **A staircase would allocate a phantom floor.** It passes through heights far
  from both real floors, so nearest-floor allocation creates a layer part-way
  up and another at the top. Allocation had to become hysteretic too: an
  unrecognised height becomes a floor only once the agent is vertically *still*
  there. A stair rises ~0.17 m/step and never settles.
- **`floor_reject_m` and `floor_band_m` formed a dead zone.** The S1 threshold
  must be smaller than a storey, so on one episode a **0.42 m step within a
  floor** tripped it and **471 of 500 frames were dropped** — the agent mapped
  nothing for 94% of the episode while `FloorStack` correctly considered it the
  same floor. Rejection now keys on `FloorStack.in_transit()` (matching no
  known floor); the same episode drops 0 frames. The height threshold survives
  only as the `multi_floor=off` fallback.
- **`key` vs `order`.** Discovering a basement mid-episode reorders floors, so
  everything persistent references the allocation key, not the height rank.

| A/B | status |
|---|---|
| `dev50_mf` × `multi_floor` ∈ {false, true} | **not run in isolation** |

Both arms of the S3 stair A/B had `multi_floor` on, so its standalone effect was
never separated from stair frontiers'. That is deliberate: without stair
frontiers the agent never leaves the floor (measured), so per-floor maps have
nothing to be per-floor about. The pair is only meaningful together.

> Expected to be **net 0 on its own**: per-floor maps are correct but inert
> until something proposes an off-floor goal. On the 3 probe episodes
> `n_floors` stayed 1 and `floor_switches` 0 — the agent never left the
> starting floor. S2 is the substrate S3 needs, not a standalone win.

### S3 — stair detection + `CLIMB`

#### Measurement first: which stair signal actually works

`scripts/measure_stair_recall.py` derives ground truth from the navmesh rather
than from annotations: on a cross-floor episode the geodesic path from start to
goal *must* cross a staircase, so path waypoints where the height climbs steeply
mark real stairs. Poses are rendered by teleporting the camera, so the agent's
own behaviour cannot bias the sample. 12 episodes, 117 stair poses, 25 flat
controls:

| signal | up-stairs (37) | down-stairs (80) | control FP (25) |
|---|---|---|---|
| YOLOE `stairs` | **24%** | 0% | **0%** |
| below-floor geometry | 0% | **50%** | **0%** |
| rise-vs-distance "ramp", no detector | 32% | 0% | **28%** ❌ |

Conclusions, all of which changed the design:

- **YOLOE cannot carry up-stair detection alone** (24% per pose) but its
  precision is perfect, and every mask it fired on passed the geometric gate
  (median slope +0.87, i.e. a genuine flight). Keep it as a sparse, trustworthy
  signal; do not gate a floor transition on it firing.
- **Down stairs are solved by geometry** — points below the standing floor,
  50% per pose at 0% false positive, no model at all.
- **The detector-free "ramp" test was rejected.** It fires on flat control
  poses (28%) nearly as often as on stairs, so it is noise. It lives on only in
  the measurement script, marked as a rejected candidate, so the negative
  result stays reproducible.

Per-pose rates understate per-staircase recall: `StairDetector` accumulates
hits across frames, and the agent passes many poses near a staircase.

#### Probe (6 cross-floor episodes — a probe, not an A/B)

| arm | SR | note |
|---|---|---|
| `stair_prior=0` (S2 only) | **0/6** | stairs never selected; the agent never leaves the ground floor |
| `stair_prior=0.6`, no CLIMB | **1/6** | reaches the target but circuitously (207 steps, SPL 0.17) |
| `stair_prior=0.6` + CLIMB | **2/6** | same success in 51 steps, SPL 0.77 |

This is the first configuration in which the agent **changes floor and finds
the target** — episodes that stopped 4.5 m and 3.0 m away now stop at 0.02 m and
0.05 m.

Two honest caveats:

- **n=6 cannot support a conclusion** (`compare_runs` verdict: NOISE). The real
  measurement is `dev50_mf` × {`stair_prior` 0, 0.6}, pending.
- **`CLIMB` does not perform the climb.** `climb_ok` stayed 0: the floor change
  always completed a dozen steps *after* the state exited, during ordinary
  frontier navigation. The overshoot goal repositions the agent into the
  stairwell; the navmesh follower does the ascent. Removing CLIMB costs a
  success, so it earns its place — but as an approach behaviour, not a
  traversal. It also had to be made to fail fast: it initially spun in place
  for exactly the 20-step stall window after the follower had already given up.

#### A/B: `dev50_mf` (50 paired cross-floor episodes)

| | `stair_prior=0` | `stair_prior=0.6` |
|---|---|---|
| SR | 4.0% | **10.0%** |
| gained / lost / **net** | — | 3 / 0 / **+3** |
| SPL | 0.033 | 0.058 |
| mean distance to goal | 12.25 m | 11.14 m |
| episodes that changed floor | 5/50 | **13/50** |
| floor switches per episode | 0.10 | 0.40 |
| trajectory height span | 0.27 m | 0.51 m |

**Verdict: real effect** (net ≥ +3, nothing lost). Enabled by default — inert
unless `multi_floor` is also on, since without per-floor maps no stair detector
is constructed.

Two things the numbers say clearly:

- **Changing floor is necessary here.** All 5 successes in the treatment arm
  changed floor; the baseline manages 4% because a handful of episodes have a
  same-floor instance of the target category.
- **The climb itself is the bottleneck, not finding the stairs.** 178 climb
  attempts across 50 episodes produced **5 floor changes — 2.8%**. Median
  distance to goal is still 12.6 m and only 5/50 episodes end within 1 m. So
  stair *detection* now works well enough to be selected 3.6 times per episode,
  and stair *traversal* almost always fails.

That 2.8% is the highest-value thing to fix next, well ahead of any further
stage: the mechanism is already firing, it just does not complete.

#### Follow-up: fixing the repeat attempts did NOT help (net −1)

Diagnosis of the 178 attempts found three things:

1. **The centroid blacklist never worked.** The failure path recorded
   `_climb_goal_xy` — the overshoot point, `stair_overshoot_m` (1.5 m) beyond
   the centroid — while the filter compares against new detections' centroids
   with a 1.0 m radius. It could never match. A plain bug.
2. **134 of the 178 attempts entered and exited `CLIMB` within a single step**,
   because the navmesh had nowhere to snap the overshoot goal (for a down
   staircase it can sit over the void).
3. Two episodes produced 131 attempts between them, re-finding the same
   stairwell every selection round for 500 steps.

Fixes: record the centroid; fall back from the overshoot to the staircase
itself before giving up; retire a failed staircase by its cells plus a margin.

| | before | after |
|---|---|---|
| `climb_fail` per episode | 3.46 | **0.38** |
| `climb_ok` per episode | 0.10 | 0.12 |
| floor switches per episode | 0.40 | 0.40 |
| SR | 10.0% | **8.0%** |
| gained / lost / net | — | 0 / 1 / **−1** |

**The mechanism worked and SR got slightly worse.** The reading: those repeat
attempts were nearly free — most failed within a single step — so removing them
bought nothing, while retiring cells permanently discards real staircases the
agent might have used from a better approach angle. A free non-problem traded
for a real cost.

Kept: the centroid fix and the overshoot fallback (both unambiguously correct;
`climb_ok` ticked up). Made opt-in: `exploration.stair_retire_cells`, since the
aggressive retirement is what measured negative.

**What this rules out.** Climb *attempts* are not the bottleneck. Floor switches
sat at 0.40/episode across both arms regardless of attempt count.

#### Decomposition (`scripts/analyze_climb.py`)

| | before | after |
|---|---|---|
| attempts | 178 | 26 |
| per-attempt success | 2.8% | **23.1%** |
| **INSTANT** (never lasted one step) | **75.3%** | **0%** |
| episodes that attempted a climb | 16 | 16 |
| episodes that switched floor | 13 | 13 |

The fix worked, decisively: instant failures — attempts where navigation had
nowhere to snap the goal and the state was entered and left inside a single
step — went from 134 to zero, and the per-attempt success rate improved 8×.

**And it changed nothing about which episodes get upstairs.** Both arms:
16 episodes attempt a climb, 13 of them switch floor. The earlier net −1 was
one episode's worth of noise on top of an unchanged distribution.

**The real bottleneck is stair recall, not traversal.** Every one of these 50
episodes requires a floor change, and only **16 ever attempt one** — but 13/16
(81%) of the attempts that happen do get the agent to another floor. Traversal
works; finding the staircase is what fails. That points straight back at the
detection measurement (up-stairs 24% per pose, down-stairs 50%), and makes
recall — not climb tuning — the next lever.

A smaller secondary finding: 5 episodes gained real height (median 0.65 m) but
never committed a floor change. If that persists it points at `FloorStack`'s
band/commit thresholds rather than at navigation.

#### …and stair recall is not the bottleneck either

Splitting the 50 cross-floor episodes by whether they ever attempted a climb:

| | attempted (16) | **never attempted (34)** |
|---|---|---|
| steps (median) | 459 | **108** |
| hit the 500-step cap | 8/16 | 6/34 |
| **committed to a target and stopped** | 8/16 | **28/34** |
| frontier selections (median) | 10 | **3** |
| distance to goal (median) | 8.2 m | 13.4 m |

The 34 episodes that never attempt a climb are **not** running out of budget
looking for stairs. They stop after a median of **108 steps and 3 frontier
selections**, having committed to something on the starting floor, and end
13.4 m from the goal.

Every one of these episodes is cross-floor, meaning *every annotated instance of
the target category is on another floor*. HM3D annotates all instances, so
whatever the agent walked to is almost certainly a **misdetection**. The
committed track had a median of 3 observations at detector score 0.63 — just
over the admission gates — and the categories are exactly the two weakest in the
historical per-category table: **bed (12) and tv_monitor (8)** of 28, the classic
sofa-as-bed and picture-as-monitor confusions.

**So the multi-floor loss is mostly premature commitment to a misdetection, not
stair detection and not traversal.** That is precisely what the forced-choice
VLM verifier exists to catch.

#### A/B: the verifier, on cross-floor episodes

`dev50_mf`, 50 paired episodes, value map on (weight 4), multi-floor on,
`verification` ∈ {`off`, `nim`}. The first measurement in this log to use the
VLM at all — every earlier run had no API key.

**The targeted quantity moved decisively:**

| | off | nim |
|---|---|---|
| **committed without ever attempting a climb** | **27/50** | **12/50** |
| median steps of those that still did | 57 | 134 |
| verifier calls / rejects | — | 82 / **55 (67%)** |

**And the whole predicted chain followed:**

| | off | nim |
|---|---|---|
| episodes attempting a climb | 14 | **22** |
| episodes switching floor | 12 | **17** |
| switched **and** succeeded | 4 | **6** |
| per-attempt climb success | 19.0% | **27.8%** |
| distance to goal | 11.58 m | 10.95 m |
| SPL | 0.054 | 0.068 |
| **SR** | 8.0% | **12.0%** (gained 2, lost 0) |

Reject a downstairs misdetection → the agent keeps exploring instead of stopping
→ it finds the stairs → it climbs → it reaches the real, upstairs goal. Floor
changes now include **bed (4)**, the category most often falsely committed to
before.

net +2 sits in the noise band, so by the letter of the rule the flag stays as it
is — but this is the log's clearest "mechanism correct, underpowered" case:
gained 2, lost 0, the targeted metric more than halved, and seven downstream
metrics all moving the predicted way. The repo default (`verification: nim` in
`config.yaml`) is already on; it was every *measurement* here that deviated by
running `verification=off`, and that should stop.

Cost: 1.6 VLM calls and ~100 extra steps per episode (the agent explores instead
of stopping early). SPL still improved, so the successes are not bought with
wandering.

### S5 — geometric FP retraction + per-step target detection

Two flags, both off by default, both aimed at the same-floor majority (81% of
episodes) that the multi-floor work cannot reach.

- **`scene_graph.fp_retraction`.** A track born from a detection hard against an
  image edge or at the far end of the depth range is treated as a hypothesis.
  When the agent later has that position well inside its view cone at close
  range and sees nothing of that category, the hypothesis is refuted and the
  track retracted; the position and label are remembered so re-detection does
  not resurrect it. Costs nothing beyond detections already computed, and needs
  no VLM — which is the point, since verification provably cannot help with
  false positives that are category-correct.
- **`scene_graph.target_every_step`.** Keyframes are 0.25 m / 30° apart, so a
  target glimpsed while crossing a doorway is missed. Target-category
  detections now reach the object layer every step. Deliberately target-only:
  the full set would add ~500 near-duplicate observations per episode and
  inflate `evidence`, invalidating the calibrated `min_evidence` /
  `confirm_baseline_m` thresholds.

Neither costs an extra forward pass — the detector was already running twice on
some steps, and a one-slot cache collapses every consumer in a step to one call
(pinned by a test).

#### A/B: `dev50`, 50 paired episodes, on top of value map + VLM verifier

Baseline deliberately includes the verifier, so the question is whether free
geometry adds anything on top of the paid VLM.

| | base | `+fp_retraction` | `+target_every_step` |
|---|---|---|---|
| SR | **64.0%** | 56.0% | 62.0% |
| gained / lost / **net** | — | 0 / 4 / **−4** | 0 / 1 / **−1** |
| SPL | 0.339 | 0.303 | 0.326 |
| retractions / episode | — | **16.1** | — |

**`fp_retraction` is a regression and stays off.** 16 retractions per episode is
the tell: one in-cone frame where the detector happens to miss is enough to
disbelieve a track for good, and detectors miss frames constantly. The mechanism
is right — it is ASCENT's — but "not re-detected once" is far too weak a
refutation. If revisited, it needs a run of consecutive in-cone misses, and it
has to earn its place against the VLM verifier, which already covers this
failure mode and is what gets the baseline to 64%.

**`target_every_step` cannot do anything, by construction.** 45 of 50 episodes
were byte-identical and `steps_to_first_candidate` was unchanged to three
decimals. The reason: `keyframe_trans_m` (0.25) equals `forward_m` (0.25) and
`keyframe_rot_deg` (30) equals `turn_deg` (30), so **every action already
triggers a keyframe** and per-step detection was happening anyway. The
"keyframe" abstraction is vacuous at this action granularity. The flag only
becomes meaningful if the keyframe thresholds are raised.

### S6 — nearest-point terminal stop

`agent.terminal_rule="nearest_point"` (default `"depth"`) accumulates the
target's masked depth points into a bounded surface cloud and stops when the
**nearest surface point** is within `terminal_stop_m`, or when closing stalls.

The motivation is a mismatch: HM3D scores distance to a view_point, and view
points are tiled around the surface, while median mask depth measures to the
middle of whatever the mask covers. A 2 m sofa and a chair therefore stop at
very different distances from their near edge — consistent with the documented
near-miss mode where episodes stall at dtg 0.107–0.147 m.

The stall clause covers being blocked by the object itself or by furniture in
front of it, and is self-limiting only because the distance is recomputed from
the live pose each step. The cloud spans the linked component, so an L-shaped
sofa is measured to its near end; it is accumulated for the episode target only,
bounding memory by one category.

A latent bug surfaced while testing: `if self._candidate_id` is falsy for track
id **0**, so the rule was silently disabled for the first object mapped in an
episode.

#### A/B: `dev50` — a severe regression, caused by a bug

| | `depth` | `nearest_point` |
|---|---|---|
| SR | **64.0%** | **32.0%** |
| gained / lost / **net** | — | 1 / 17 / **−16** |
| SPL | 0.339 | 0.175 |
| median distance to goal | **0.04 m** | 1.09 m |
| stopped by the stall clause | — | **18/50** |

**The rule was not wrong; the stall test was.** Approaching an object means
turning to face it, and a turn leaves the distance to it unchanged. Counting
that as "failed to close" fired the stall on the first turn of nearly every
approach, stopping the agent a metre out — visible directly in the median
distance to goal going from 0.04 m to 1.09 m.

Fixed: the stall test now only counts steps in which the agent actually moved,
and requires a run of them (`terminal_stall_steps`, default 3) rather than a
single non-improving step, since an oblique approach barely changes the distance
to the nearest surface. Pinned by
`tests/unit/test_terminal_stop.py::test_turning_never_counts_as_a_stall`.

#### A/B: `dev50` — re-measured after the fix, still a regression

| | `depth` | `nearest_point` 0.6 | `nearest_point` 0.4 |
|---|---|---|---|
| SR | **64.0%** | 40.0% | 48.0% |
| gained / lost / **net** | — | 1 / 13 / **−12** | 1 / 9 / **−8** |
| SPL | 0.339 | 0.201 | 0.246 |
| stopped by the stall clause | — | **0/50** | **1/50** |
| median dtg among surface stops | 0.042 (depth) | 1.008 | 0.401 |
| surface stops landing ≤ 0.1 m | 27/37 | 15/38 | 14/31 |

The fix worked — the stall clause is now essentially never the reason an episode
ends. The rule itself still loses, and the *shape* of the loss says why.

`steps_to_first_candidate` is identical to the baseline (77.787, n=47 in both),
so exploration is untouched and the entire loss is in the terminal phase. Every
lost episode stops **early** (steps 90→78, 43→22, 62→42) at a **large** dtg —
0.04 m becomes 1.0–2.3 m.

Crucially the result is **bimodal**: at `stop_m=0.4`, 14 of 31 surface stops land
within 0.1 m of the goal and the rest scatter out to 2.3 m. A threshold set too
large would shift the whole distribution; it cannot split it. Bimodality means
outliers.

The cause is the statistic. `nearest_point_dist_xy` took `.min()` over a cloud
accumulated across hundreds of frames of mask noise and pose drift, spanning
linked tracks — the least robust statistic available, where the rule it competes
with uses a *median*. One stray backprojected pixel a metre in front of the
object ends the approach there.

Sweeping `terminal_stop_m` corroborates this rather than fixing it: 0.4 beats 0.6
(−8 vs −12) because a tighter threshold is harder for a stray point to trigger,
not because the threshold was the problem.

Added `agent.terminal_percentile` (default `0.0` = the exact min = ASCENT's
behaviour, so earlier numbers stay reproducible) to test the diagnosis directly.
Pinned by `test_one_stray_point_hijacks_the_minimum` and
`test_percentile_still_reports_the_near_surface` — the guard must drop strays
without degenerating into "distance to the object's middle", which is the
median-depth rule it is trying to beat.

#### A/B: `dev50` — the percentile guard confirms the diagnosis

| arm | statistic | `stop_m` | SR | net | median dtg among surface stops | ≤ 0.1 m |
|---|---|---|---|---|---|---|
| `s50_base` | median depth | — | **64.0%** | — | 0.042 | 27/37 |
| `s51_term04` | `min` | 0.4 | 48.0% | −8 | 0.401 | 14/31 |
| `s52_p5_stop04` | **5th pct** | 0.4 | **66.0%** | **+1** | **0.060** | 15/22 |
| `s51_term06` | `min` | 0.6 | 40.0% | −12 | 1.008 | 15/38 |
| `s52_p5_stop06` | **5th pct** | 0.6 | 58.0% | −3 | **0.044** | 20/32 |

The outlier hypothesis holds. Changing only the statistic — same thresholds,
same everything else — moves the median distance to goal among surface stops
from 1.008 m to 0.044 m at `stop_m=0.6`, and from 0.401 m to 0.060 m at 0.4.
Precision rises from 45% to 68% of surface stops landing within 0.1 m.

The rule also fires *less often* once guarded (22 and 32 stops, down from 30 and
38), because a guarded distance is larger and the threshold is crossed later;
the remainder fall through to `path_consumed`, which is the pre-existing
behaviour.

**But the fixed rule only matches median mask depth, it does not beat it.**
net +1 is inside the noise band, so by the S0 decision rule the flag stays off.
The single lost episode is not a terminal failure either: it ends at dtg 8.8 m
via `path_consumed`, meaning the agent went somewhere else entirely — a
trajectory divergence downstream of an earlier different stop, not the rule
misfiring.

`terminal_percentile` now defaults to **5.0** rather than 0.0, so that opting
into `terminal_rule="nearest_point"` gets the version that works. Pass
`terminal_percentile=0` to reproduce the ASCENT-faithful arms above.

**Still off by default.** Median mask depth remains the best terminal rule
measured, at 27/37 stops within 0.1 m. What this sequence bought is not SR, but
the knowledge that the surface rule's failure was a statistics bug rather than
evidence against the idea — worth knowing before the full-split run, where the
two rules may not tie.

### S4 — semantic value map

`exploration=value` paints each frame's image-text similarity to
`"Seems like there is a {target} ahead."` over the ground that frame observed,
and ranks frontiers by it instead of by a flat prior — the mechanism ASCENT and
VLFM have and geometric exploration lacks.

Design notes worth keeping:

- **Confidence is angular, not radial.** A surface seen down the optical axis is
  observed well at 1 m or 4 m; one at the frame edge is observed poorly at any
  range. Fusion keeps each cell's value from its most confident observation, so
  a glancing sweep past a doorway never overwrites a head-on look into the room.
- **The observed region is built in world coordinates** from the camera's own
  axes, *not* by rasterising a local patch and rotating it into place as ASCENT
  does. That construction is exactly where a sign error yields a mirrored or
  rotated map — plausible on a heatmap, and steering the agent away from the
  target. The acceptance test asserts the cone lands in front of the camera and
  nothing behind it, for headings all the way round.
- `value_weight` exists because the geometric boosts are multiplicative and
  would otherwise swamp a value range of a few hundredths; `value_argmax` skips
  the divide-by-path-cost so ASCENT's ranking can be compared directly.

**Deployment note.** The nav container has no outbound network, and
`data/weights` is a docker *named volume* the host cannot write to. The CLIP
checkpoint therefore has to be staged host-side into the bind-mounted
`data/clip`:

```bash
python scripts/download_weights.py --clip     # on the HOST, ~350 MB
```

Until that runs, `exploration=value` will fail at model load. Everything else is
implemented and unit-tested; the end-to-end run with a real CLIP is the one part
not yet exercised.

#### A/B: `dev50` (50 paired episodes, representative split)

| | `sweep` | `value` |
|---|---|---|
| SR | 52.0% | 52.0% |
| gained / lost / **net** | — | 2 / 2 / **0** |
| `steps_to_first_candidate` | 86.9 | 87.7 |
| steps | 151.4 | 145.2 |
| CLIP calls / episode | 0 | 145 |

**Inert — and the arithmetic says it had to be.** The value map ran on every
step and painted the map correctly; it simply cannot reach the argmax at this
parameterisation:

```
geometric boosts   (1 + 2·gain/gmax) · (1 + 2·align)   spans 1 → 9      (900%)
value term         base = cosine ** value_weight       spans 1 → 1.2     (20%)
```

CLIP cosines on *maximally* different images (the red/blue smoke test) differ by
0.08; real indoor frames differ far less, so `base` covers roughly 0.20–0.24. At
`value_weight = 1` the geometric terms out-range the semantic one by about 7:1,
so the value can only break near-ties.

This measures **one parameterisation, not the mechanism**. So the knobs that
exist precisely for this were swept.

#### The sweep: the value map works, once it can compete

`dev50`, same 50 paired episodes, `knowledge_prior` off throughout, all compared
against `sweep` (no value map):

| arm | SR | net | steps→1st candidate | steps | SPL |
|---|---|---|---|---|---|
| `sweep` | 52.0% | — | 86.9 | 151.4 | 0.281 |
| `value_weight=1` | 52.0% | 0 | 87.7 (+0.8) | 145.2 | 0.283 |
| **`value_weight=4`** | 56.0% | +2 | **77.8 (−9.1)** | **127.8 (−23.6)** | **0.320** |
| **`value_weight=8`** | **58.0%** | **+3** | 83.4 (−3.4) | 133.4 (−18.0) | 0.312 |
| `value_argmax` | 52.0% | 0 | 76.8 (−10.0) | 152.1 | 0.311 |

The designated primary metric — steps to first candidate — **does not move at
weight 1 and moves substantially at weight ≥ 4**, exactly as the arithmetic
predicted. Both weight 4 and weight 8 improve SR, total steps, SPL and time to
first candidate together; they differ from each other by a single episode.

**Enabled by default at `value_weight=4`.** The decision does not rest on one
arm crossing the net ≥ +3 line (only weight 8 does, by one episode) but on four
independent metrics moving consistently at two different weights, with a
mechanism that predicts the weight-1 null result in advance. Weight 4 is chosen
over 8 on the efficiency metrics, which are more stable at n=50.

`value_argmax` is a genuine curiosity: dropping the divide-by-path-cost finds
targets fastest of all (−10.0 steps to first candidate, SPL +0.030) but converts
none of it into SR and leaves total steps unchanged — it goes to the right
places without finishing. Left off.

> Judge primarily on `steps_to_first_candidate` — the quantity the value map
> directly moves, and far more stable than SR at n=50. It did not move.

### S7 — knowledge-graph frontier re-ranking (LLM-free)

ASCENT asks a language model to name the room a frontier sits in, then puts that
in a prompt. The same information is already implicit in the objects mapped
nearby — its knowledge graph is a joint distribution over (object, room) — so
`exploration.knowledge_prior` scores a frontier directly:

```
affinity(f) = Σ_r  P(r | objects near f) · P(r | goal)
```

A frontier surrounded by a shower and a towel scores high for `toilet` and low
for `bed`, with no model call. That also sidesteps the reason the LLM scorer was
measured to never influence a selection: its results arrived asynchronously,
keyed on frontier ids that are reassigned on every extraction.
`SpatialScoreCache` fixes that properly for any future asynchronous scorer by
keying on position.

Design point worth keeping: affinity returns **None, not 0**, when nothing is
known nearby. Treating "no evidence" as "unpromising" would permanently
deprioritise unexplored regions — which by definition have no mapped objects and
are exactly where the agent must go.

`scripts/make_priors.py` subsets ASCENT's 2 MB networkx graph and xlsx floor
table into a few KB of plain JSON, so neither networkx nor pandas is needed.
Extracted values are sane: toilet→bathroom 0.97, bed→bedroom 0.75,
sofa→living_room 0.51, beds 48% on the top floor of a three-storey house.

#### A/B: `dev50` (50 paired episodes)

| | `value` | `value` + `knowledge_prior` |
|---|---|---|
| SR | 52.0% | 50.0% |
| gained / lost / **net** | — | 3 / 4 / **−1** |
| steps | 145.2 | 156.7 |

Noise, and subject to the same swamping as the value map: the affinity enters
as `1 + w·(aff − 0.5)`, a factor of at most 1.5, against geometric boosts
spanning 9×. Left off.

Worth separating from the value-map result, though: this one *changed* outcomes
(3 gained, 4 lost) rather than doing nothing, so the prior is reaching
selection — it is just not reaching it usefully at this weight, on a split where
86% of episodes never leave one floor and the target is usually in the first or
second room.

**Not implemented:** the LLM ranker itself (top-k prompt, SSIM dedup of frontier
context images, nearby-frontier shortcut). The prior above is its LLM-free core;
whether a model call adds anything on top is an open question, and the earlier
measurement on single-floor said it did not.

### S9 — ASCENT's frontier extraction and selection

A line-by-line comparison of the two frontier paths found three semantic
differences that had never been measured. Each is ported behind a flag.

| | OSG | ASCENT |
|---|---|---|
| **F1** what the size filter measures | the frontier's own cell count, `frontier_min_cells=8` ≈ 0.4 m of boundary | the **adjacent unexplored area**, `area_thresh=1.5 m²` |
| **F2** distance | `util = score / path_cost` | never divides; value argmax, distance only as a 3 m shortcut |
| **F3** commitment | none; re-selects every 5 steps, `continuity_weight` as a soft proxy | force frontier + retire after 20 rounds without closing + retire after 20 repeat selections |

Checked and found **identical**, so not ported: the value reduction. Both take
the median within a 0.5 m radius — ASCENT's `pixel_value_within_radius` defaults
to `reduction="median"`. Only the window shape (box vs circle) and the
no-observation sentinel (`0.0` vs `-1`) differ, and both sort unobserved
frontiers last either way.

#### S9a — contour extraction

`exploration.extractor="contour"`. Ports `frontier_detection.py` function for
function: absorb small unexplored pockets into the explored mask, contour the
result, split the contour where it stops bordering unexplored space, take each
arc's length-weighted midpoint.

**The prediction in the plan was wrong, and in the opposite direction.** It
expected *fewer* frontiers, since small pockets are filtered. Measured on dev50:
**9.83 frontiers per selection round**, against 6.08 for WFD on a 4-episode
probe — substantially more.

The cause is a map-semantics difference the plan listed as a risk and which
turned out to dominate: ASCENT contours a fog-of-war **cone**, which is smooth,
while OSG's explored region comes from per-point Bresenham raycasting and has a
ragged boundary. A ragged boundary yields a longer contour that walls interrupt
more often, so it splits into more, shorter arcs. The area filter only removes
*isolated* pockets; it does nothing about short arcs bordering a large
unexplored region. The filter itself is verified working in unit tests (0.36 and
1.0 m² pockets dropped at a 1.5 m² threshold, 4.0 and 9.0 m² kept).

| | `wfd` | `contour` |
|---|---|---|
| SR | **64.0%** | **64.0%** |
| gained / lost / **net** | — | 3 / 3 / **0** |
| SPL | 0.339 | 0.308 |
| frontiers per round | 6.08 (4-ep probe) | 9.83 |

`steps_to_first_candidate` appears to improve by 8.9 steps, but that is an
artifact worth recording: two episodes saw a candidate under WFD and **none**
under contour, so they drop out of the treatment mean. Re-paired over the 45
episodes where both arms saw a candidate, the median delta is **+0.0** (17
faster, 13 slower, 15 identical) — noise. The two dropouts are a mild negative
that the mean was hiding.

All six flips are large swings (dtg 7.33→0.05, 0.03→9.43), i.e. trajectory
divergence from a different early choice rather than a systematic effect.

**Verdict: noise, flag off.** The port is faithful; the method simply does not
transfer without ASCENT's smoother explored-region semantics.

#### S9b — ASCENT selection

`exploration.selector="ascent"`. Value argmax, 3 m nearby shortcut, and
position-keyed frontier retirement. The LLM branch is deliberately not ported —
S7 established the knowledge-graph prior is the part that reaches selection, and
a synchronous call per round would dominate a 50-episode A/B.

One marked deviation: selection falls through to the next candidate when
planning fails. ASCENT hands its waypoint to a learned PointNav policy that
makes progress toward almost anything; OSG plans with A*, so an unreachable
frontier would be re-picked every round until the repeat counter retires it 20
rounds later, burning ~100 steps.

#### A/B: all five arms on `dev50`

| arm | extract | rank | commit | SR | net | SPL | steps | frontiers/round | mean size | timeouts | never saw target |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `s50_base` | wfd | utility | — | **64.0%** | +0 | **0.339** | 164.9 | 6.44\* | 34.0\* | 6 | 3 |
| `s9_contour` | contour | utility | — | 64.0% | +0 | 0.308 | 166.3 | 9.83 | 20.0 | 6 | 5 |
| `s9_sel` | wfd | **ascent** | yes | 58.0% | **−3** | 0.295 | 183.0 | 6.44 | 34.0 | 8 | **7** |
| `s9_full` | contour | ascent | yes | 60.0% | −2 | 0.288 | 193.8 | 10.31 | 22.7 | 8 | 6 |
| `s9_commit` | wfd | utility | **yes** | 68.0% | +2 | 0.334 | 163.5 | 6.44 | 34.0 | 5 | 3 |

\* read off `s9_sel`, which uses the same WFD extractor; the baseline predates
these counters.

**F1 — extraction (net 0).** The finer/slivers question resolves cleanly:
6.44 frontiers × 34.0 cells = 219 cells of boundary for WFD, 9.83 × 20.0 = 197
for contour. Nearly the same boundary, subdivided about 50% more finely. Not
slivers, and not more coverage. SR is unchanged and SPL slightly worse.

**F2 — ranking (net −3, the only real regression).** Every mechanism metric
agrees on the cause: steps +11%, timeouts 6→8, and episodes that never see the
target at all 3→7. Three of the six losses are 500-step timeouts with dtg 14.68,
5.52 and 9.45 and no candidate ever committed to. This is the cost the plan
pre-registered: ASCENT absorbs travel with a learned PointNav policy, OSG walks
it with A*, so removing the path-cost division buys longer routes the step
budget cannot pay for. **F2 does not transfer.**

**F3 — commitment: measured, and the measurement does not support it.** The
arm scores the best SR of the five (68%, gained 2, lost 0), and it would have
been easy to report that as a win. It is not one:

* `frontier_retired` totals **3 across all 50 episodes**, in a single episode.
* **46 of 50 episodes are bit-identical to the baseline.**
* Of the 4 that diverge, **none retired anything** — including both gains.

With nothing retired the code path is identical by construction: `blocked` is
unchanged and `observe()` only mutates private state. So the divergence is not
the mechanism.

### The A/B noise floor (a methodological caveat on this whole log)

Chasing the above found something that affects every result here. The four
diverging episodes are the ones that lean hardest on the hosted VLM verifier —
mean 3.25 verifier calls against 1.26 for the bit-identical ones — and the same
episodes flip across arms whose mechanisms have nothing to do with each other:

| episode | flipped in | baseline |
|---|---|---|
| `Dd4bFSTQ8gi:42` | 3 of 4 arms | 357 steps, dtg 1.00 |
| `qyAac8rV8Zk:96` | 3 of 4 arms | 500 steps, dtg 1.85 |
| `QaLdnwvtxbs:89` | 2 of 4 arms | 99 steps, dtg 0.04 |

`qyAac8rV8Zk:96` "improves" in the contour arm (500→118), the ASCENT-ranking arm
(500→170) and the commitment arm (500→174). Three unrelated mechanisms do not
all rescue the same episode; a nondeterministic verifier does.

`verification=nim` and `llm=nim` are hosted model calls with no seed, so runs are
not reproducible. Measured directly, with two comparisons in which **nothing
real changed**:

| comparison | bit-identical | diverged | **flipped (net)** |
|---|---|---|---|
| `s50_base` vs a replicate of itself (identical config and code) | 48/50 | 2 | **0** |
| `s50_base` vs `s9_commit` (mechanism provably inert: 0 retirements in every diverged episode) | 46/50 | 4 | **+2** |

So the floor on `net` is **0 to +2** over two samples. That **confirms the
decision rule's ±2 noise band rather than undermining it** — an earlier draft of
this section overstated the problem by guessing the floor could reach +3 before
measuring it. What is fair to say is that net = +3 clears the floor only
narrowly.

The source is the **VLM verifier**, and only it. Every episode that diverged had
made verifier calls, and every episode that made none was bit-identical in every
comparison — a call is a necessary condition for divergence.

One correction to an earlier reading of this: `llm_calls` in `episodes.jsonl`
counts `AsyncScorer.n_calls`, and every arm here runs `exploration.scorer=
nearest`, so those are `NullScorer` invocations — local, free and deterministic,
not network requests. Its movement (3→6, 23→8) is a *consequence* of a
trajectory diverging, not a cause. The verifier is the one non-deterministic
component in the loop.

That also means the text LLM has never been exercised in any arm of this log,
which is why nobody noticed that `nvidia/nvidia-nemotron-nano-9b-v2` was retired
on 2026-08-26 (see S10).

**Consequence for the earlier conclusions.** Two arms in this log reached exactly
+3, and both were already argued on mechanism metrics rather than on SR:
`stair_prior=0.6` moved floor-changing episodes from 5/50 to 13/50, and the value
map moved four metrics together across two weights while predicting the weight-1
null in advance — its chosen default, `value_weight=4`, is itself only +2. So
nothing needs revising, but the point is now demonstrated rather than assumed:
**at n=50 the mechanism metrics are the evidence and SR is corroboration, not the
other way round.**

**Verdict: all three flags off.** F2 is a genuine regression with a mechanism
explanation. F1 is a faithful port that does not transfer, at net 0. F3 was
never actually exercised — its arm is one of the two inert comparisons used to
measure the noise floor above, so it is untested rather than tested-and-neutral.
Retirement needs a scenario that triggers it (frontiers repeatedly chosen and
not reached) before it can be judged; on `dev50` with the navmesh follower, that
scenario arises three times in fifty episodes.

### S10 — ASCENT's LLM usage

Three ports from `ascent/llm_planner.py`, each behind a flag: a **forced choice**
among the top-3 by value (`{"Index","Reason"}`) instead of a 0–1 rating of up to
eight frontiers; the **room-to-goal priors written into the prompt** rather than
applied as a multiplier outside it; and a **synchronous** call, because the async
scorer's results are keyed by `Frontier.id` and almost never reached a decision.
`temperature` also drops from 0.1 to 0, as ASCENT runs at 0 with
`do_sample=False` (`model_api/qwen25_out.py:62-66`).

Known deviation, and not a small one: areas are described from YOLOE labels and
the Voronoi room segmenter, where ASCENT uses RAM tags and a Place365 scene
classifier (`map_controller.py:805-830`). The prompt *shape* is ASCENT's; the
content comes from different models.

#### The dead endpoint

`nvidia/nvidia-nemotron-nano-9b-v2` was retired on 2026-08-26 and answers 410.
**No earlier result is affected**: every arm ran `exploration.frontier_text_scorer=disabled` (then named
`exploration.scorer=nearest`), so
no text request was ever sent — which is exactly why it went unnoticed. The
nemotron-3 successors 404 for this key and `llama-3.3-70b`, `llama-3.1-8b` and
`phi-4-mini` are also 410, so the text model moves to the one text-capable
endpoint this account can reach: the same vision model the verifier uses.

#### A/B: `dev50`

| arm | SR | net | SPL | steps | fps |
|---|---|---|---|---|---|
| `s50_base` (temp 0.1) | **64.0%** | — | **0.339** | 164.9 | 6.27 |
| `s10_temp0_a` | 66.0% | +1 | 0.340 | 166.2 | 6.2 |
| `s10_ranker` | 64.0% | **0** (4 gained, 4 lost) | 0.317 | 174.8 | **2.90** |

**The determinism audit is the more useful result, and it failed.**

| two runs, identical config and code | bit-identical | flips |
|---|---|---|
| temperature **0.1** | 48/50 | 0 |
| temperature **0.0** | **49/50** | 0 |

One episode. Temperature 0 does **not** make the hosted verifier reproducible, so
the noise floor documented above is not sampling noise — it is provider-side
(batching, routing, kernel non-determinism). This was the outcome S10a was
supposed to rule in or out, and it rules it out: **there is no config change on
this side that buys reproducibility.** Any future claim resting on a net of ±2
still cannot be resolved at n=50.

**The ranker mechanically works and changes nothing.** 175 calls over 50
episodes (3.5 per episode, 42/50 episodes used it), 0 unreachable picks, and an
**override rate of 49%** — the model disagrees with the CLIP value ranking almost
half the time. That is a stronger result than the old async scorer, which was
shown to be structurally incapable of influencing a decision. Here it does
influence them, on half of all calls, and the outcome is a wash: 4 gained, 4
lost, SPL 0.339→0.317, steps +6%, and 2.2× slower wall-clock from the
synchronous calls.

Three of the four losses end 6–11 m from the goal after diverging early; two of
the four gains are 500-step rescues.

> **Correction — this arm did not test what it was built to test.**
> `RoomNode.label` is written in exactly one place, `LLMTextScorer.score`
> (`llm_scorer.py:81`), and every arm in this log runs
> `exploration.scorer=nearest`, which is a `NullScorer`. Room labels are
> therefore always `None`, and `describe_area` falls back to `"unknown room"`.
> Every area reached the model as `"a unknown room containing objects: <YOLOE
> labels>"`.
>
> So the 49% override rate and the net 0 describe a model choosing between
> **object lists with no room type at all** — half of what the ASCENT prompt is
> built around, and the half the room-to-goal priors are supposed to connect to.
> The priors were in the prompt; nothing in the area descriptions could be
> matched against them.
>
> The correct conclusion is narrower: **an object-list judgement is no better
> than the CLIP value map here.** Whether a room-typed one would be is
> unmeasured. Pinned by
> `test_ascent_ranker.py::test_unlabelled_rooms_degrade_to_unknown_room`; the
> earlier tests passed labels in explicitly and so never covered the production
> configuration.

**Both flags off.** `temperature=0` is kept as the default anyway — it costs
nothing, matches ASCENT, and removing sampling from a forced-choice task is
right on principle even though it did not buy the reproducibility it was meant
to.

An instrumentation bug found here and fixed: the runner builds one ranker and
shares it across episodes, so `rank_calls` was cumulative and read back as
per-episode it overstated usage 20× (67/episode against 3.5 actual). Pinned by
`test_counters_reset_per_episode`.

### S11 — room typing, and the hierarchy question

S10 concluded nothing because every area reached the model as `"unknown room"`.
This closes that hole and re-runs the test properly.

#### S11b′ — the segmenter was collapsing the graph to one room

Before anything could be built on the room level, it had to exist. Measured with
the new `rooms_total` counter: **median 1 room per episode**, 5/12 with more than
one, and both episodes that ran the full 500 steps segmenting into 1–2 rooms.

`scripts/measure_room_seg.py` sweeps the segmenter over saved costmaps
(`eval.save_costmap`, ~40 KB each) rather than re-running an episode per setting:

| `erode_iters` | median rooms | >1 room | 500-step episodes with ≥3 rooms |
|---|---|---|---|
| **12** (was default) | **2.0** | 7/12 | **0/2** |
| 10 | 4.0 | 9/12 | 0/2 |
| 8 | 5.5 | 12/12 | 1/2 |
| **6** (new default) | **5.5** | 11/12 | **2/2** |
| 4 | 7.0 | 12/12 | 2/2 |

`min_room_cells` over 30–120 barely moved anything; erosion is the whole story. A
partially explored costmap's free space is a narrow region carved along the
trajectory, so 0.6 m of erosion destroys every core but the widest and the
regrow step assigns the entire map to it.

6 rather than 4 because **there is no ground-truth room count**: nothing
distinguishes "correctly found 7 rooms" from "over-segmented one room into 7", so
this takes the largest erosion that still clears the bar. The bar itself
("a 500-step episode should segment into ≥3 rooms") is a proxy for not
under-segmenting, not an accuracy measure.

Also exposed `erode_iters` and `min_room_cells` in the config. They were
hardcoded in the segmenter while the two knobs that *are* in `SceneGraphConfig`,
`room_min_radius_m` and `room_door_width_m`, are accepted by its constructor and
then ignored.

End to end: `rooms_total` median 1.0 → **3.0**, max 4 → 9, episodes with more
than one room 5/12 → **12/12**.

#### Are the labels any good?

Checked before drawing conclusions from the A/B, per the plan's own rule.
Aggregate over 12 episodes, 34 labelled rooms:

`hall` 29% · `bedroom` 24% · `living_room` 21% · `bathroom` 9% · `kitchen` 6% ·
`laundry_room`/`garage` 3% each — a plausible distribution for HM3D houses, and
**94% (32/34) map into the ten reference rooms**. The two that do not (`sky`,
`butchers_shop`) come from the deliberate top-1 fallback.

So the classifier is not the problem, and the A/B below is about the idea rather
than about Places365.

#### A/B: `dev50`

| arm | rooms | SR | net | SPL | steps | ranker calls | **override rate** |
|---|---|---|---|---|---|---|---|
| `s50_base` | — | **64.0%** | — | **0.339** | 164.9 | — | — |
| `s10_ranker` | none (`"unknown room"`) | 64.0% | 0 | 0.317 | 174.8 | 175 | 49% |
| `s11_rooms` | typed | 66.0% | +1 | 0.340 | 166.2 | — | — |
| `s11_ranker` | typed | **58.0%** | **−3** | 0.293 | 173.7 | 172 | **66%** |

**Room typing alone is inert, as it should be** (net +1, inside the measured
noise floor): with `scorer=nearest`, `knowledge_prior=false` and `ranker=none`,
nothing reads `RoomNode.label`. The arm exists to confirm the labels arrive
without changing behaviour, and it does.

**Room typing makes the LLM ranker worse.** The information reaches the model and
demonstrably changes its decisions — overrides of the CLIP value ranking go from
49% to **66%** — and the changed decisions are worse: net 0 → **−3**, SPL 0.317 →
0.293. Of the 7 losses, five turn a near-miss into a far one (dtg 0.04 → 6.35,
0.04 → 10.29, 0.01 → 2.34) and two land just outside the 0.1 m success radius
(0.18, 0.60).

This is the conclusion S10 could not reach. Stated precisely: **on `dev50`, a
room-typed judgement over YOLOE object labels is worse than the CLIP value map at
choosing between frontiers, and giving the model more room context makes it
override the value map more often and lose more.**

Caveats, all measurable and none resolved: label coverage is 61% (the agent can
only vote for rooms it has physically stood in, so rooms seen through a doorway
stay unlabelled); Places365 accuracy on HM3D renders is plausible but has no
ground truth; and net −3 sits exactly on the revert threshold at n=50.

**All flags off.** `room_classifier` defaults to `"none"`; the segmenter fix is
kept unconditionally, since a graph with one room per house was wrong regardless
of what consumes it.

#### A note on the counter semantics

`rank_calls` is not comparable across S10 and S11 without care: in S10 the runner
shared one ranker across episodes so the value was cumulative, and the per-episode
reset landed between the two. 175 (S10, cumulative total) against 172 (S11, summed
per-episode) is a like-for-like comparison; the raw maxima, 175 and 12, are not.

### S12 — the coarse level: which storey is the target on?

S11 showed a room-typed LLM choosing between *frontiers* is a regression. But 86%
of `dev50` episodes never change floor, so that experiment used the model for the
one job ASCENT does **not** give it. ASCENT asks about floors
(`llm_planner.py:216-236`) and, when the answer is elsewhere, never asks the fine
question at all. This gives it that job, on the cross-floor split.

The floor is described through OSG's floor→room→object graph rather than
ASCENT's two flat per-storey string sets — the extension, not a port. The
decision sets a **direction**; the existing stair machinery executes it, with the
wrong direction damped by 1/boost rather than vetoed.

#### A/B: `dev50_mf` (50 paired cross-floor episodes)

| | geometric | + floor LLM |
|---|---|---|
| SR | 10.0% | 12.0% |
| gained / lost / **net** | — | 1 / 0 / **+1** |
| SPL | 0.058 | 0.066 |
| floor switches / episode | 0.50 | 0.48 |
| **episodes bit-identical to baseline** | — | **45/50** |

The mechanism is **not** inert: **99 questions asked** across 23/50 episodes, and
**69 of them decided to change floor**. It fires often, decides confidently, and
changes almost nothing.

#### Why: the question is degenerate here

| visited floors | episodes |
|---|---|
| 1 | 32 |
| 2 | 16 |
| 3 | 2 |

The choice is almost always between **exactly two** candidates — the current
floor and one neighbour, often only implied by a detected staircase. Every
`dev50_mf` episode requires a floor change by construction, and the question is
gated on `steps_on_floor ≥ 100`.

That gate is the problem. It is the **same condition** that switches on the
geometric prior: `stair_prior 0.6 × stair_explored_boost 3.0` fires exactly when
`layer.explored`, i.e. `steps_on_floor ≥ floor_exp_steps = 100`. So both
mechanisms activate at the same moment and point the same way. The model is
asked a two-way question with one plausible answer, at precisely the moment the
heuristic already produces that answer, and 70% of the time it agrees.

**Verdict: noise, flag off.** Not because floor reasoning is useless, but because
on two-storey houses it is redundant with a multiplier that already encodes
"this floor is exhausted, take the stairs". It could only pay off where the
choice is genuinely ambiguous — three or more storeys, which is 2/50 episodes
here.

#### A prediction that was wrong, and why

Before running this I predicted the arm would be inert, blocked by stair recall,
on the strength of an 8-episode probe that showed 5 asks and 88 blocks for "no
second floor seen". The full run shows **99 asks**. The probe happened to cover
episodes that ended early or never found a staircase, and a rate estimated from
8 episodes was simply not a rate. The conclusion it pointed at — "the bottleneck
is stair detection" — is also not what the 50-episode data says: 20/50 episodes
attempt a climb and 18/50 switch floor, so stairs are found often enough for the
question to be live. Recorded because the wrong diagnosis was the more
comfortable one: it would have blamed a known bottleneck instead of the design.

#### An audit bug this run exposed

`compare_runs` reported `algorithm config: IDENTICAL` for two arms whose entire
difference was `floor_llm`. The fingerprint block predated the flag, so the flag
under test was the one field not recorded — and the tool printed "any difference
below is run-to-run noise, not an effect", which is a false reassurance rather
than a missing one. Fixed, and
`test_eval_uid.py::test_algorithm_fingerprint_covers_every_flag_an_ab_can_switch`
now fails if a behavioural config field is added without extending the block.

### S13 — the aligned run, and what it exposed

100 episodes: 20 val scenes × episodes 0–4 (`eval=scenes20_ep0to4`), 21 of them
cross-floor. Every ported ASCENT mechanism on together
(`+experiment=ascent_aligned`), including the path-cost removal. ASCENT is
reported at **0.70** on this set.

| | SR | SPL |
|---|---|---|
| `ascent_aligned` | **55.0%** | 0.271 |
| + `verification.min_score=0.70` | **58.0%** | — |
| ASCENT | **70%** | — |

**This is not a like-for-like number.** It drives habitat's navmesh — ground-truth
geometry — where ASCENT walks the whole way on a learned PointNav policy. The gap
is therefore a *lower bound* on the real one.

#### Where the 45 failures go

| | |
|---|---|
| **committed to something, ended >1 m from the goal** | **30 (67%)** |
| never committed to anything | 9 |
| committed, ended ≤1 m away (near miss) | 6 |

And the decisive measurement — in those 30, the agent **reached what it aimed
at**:

| | |
|---|---|
| final position → the object it committed to | median **0.43 m** (21/26 within 1 m) |
| final position → the nearest real goal | median **7.48 m** |

Navigation is not the problem. **The agent commits to a false positive and walks
to it.** HM3D scores success against a view point of *any* instance of the
category, so ending 7.5 m away means the thing it walked to was not an instance
at all.

They are committed on thin evidence:

| | far-fails | successes |
|---|---|---|
| detection score at commit | **0.698** | 0.853 |
| observations of the track | **4** | 9 |
| `n_obs ≤ 3` (i.e. exactly at `min_obs`) | 16/30 | 7/55 |

The VLM verifier ran on **all 30** and accepted them (44% reject rate overall,
1.5 calls/episode). Geometric FP retraction was off — it measured net −4 on
`dev50` and was disabled.

#### Raising the commit gate

`verification.min_score` 0.45 → 0.70, chosen because it separated the two
populations best in the post-hoc scan (blocking 57% of far-fails against 11% of
successes).

| | 0.45 | 0.70 |
|---|---|---|
| SR | 55.0% | **58.0%** (7 gained, 4 lost, **net +3**) |
| **far-commit failures** | **30** | **23** |
| never committed | 9 | 12 |
| single-floor (n=79) | 63.3% | **68.4%** |
| cross-floor (n=21) | 23.8% | **19.0%** |

The mechanism moved as predicted, and the flips confirm it: **every gain has the
committed track's score rising from 0.40–0.57 to 0.79–0.97** — the agent skipped
a weak detection and later committed to a strong one. Two of the three losses
never committed at all.

Note the post-hoc 57% was a separation statistic, not a forecast: blocking a bad
commit only converts to a success if the agent then finds the real object, and
30 → 23 is what that actually bought.

Cross-floor got *worse* (23.8% → 19.0%), which fits: those episodes have less
opportunity to accumulate a confident detection before the budget runs out, so a
stricter gate costs them more than it saves.

#### What is still open

At 0.70 the failure profile is 42 = 12 never-saw + **23 far-commit** + 7 near.
False-positive commitment is still the largest single bucket, and cross-floor at
19% (n=21) against 68% single-floor is a separate and larger problem. The
remaining ~12 points to ASCENT are roughly "false positives" + "cross-floor",
and neither is solved.

#### The audit bug, twice

`compare_runs` again reported `algorithm config: IDENTICAL` for two arms whose
only difference was `verification.min_score`. The test added after the first
occurrence checked `ExplorationConfig` plus a hand-kept list, and
`VerificationConfig` was not in it — the same failure one config class over. The
test now enumerates every overridable dataclass rather than a list someone has
to remember to extend.

### S14a — the stair-recall measurement, and what it caught

The cross-floor decomposition of the aligned run put the whole problem in one
place: **7/21 episodes covered ≥80% of the start-to-goal height gap and 4 of
them succeeded (57%); the other 14 succeeded 0 times.** Getting to the goal's
storey is the entire game, and 17/21 of those episodes need to go *up* — the
branch that depends on the detector, where down-stairs are pure geometry.

ASCENT tilts its camera to find stairs (`ascent_policy.py:534-542`, LOOK_UP when
the stair class appears in the upper half of the frame within 2 m) and OSG never
pitches at all. That suggested the 24% up-stair recall was a *viewpoint* problem:
standing at the foot of a flight, the treads recede upward out of a level frame.
`measure_stair_recall.py` now renders every pose at each `--pitch-deg`, with a
sign check (looking up must raise the mean height of the backprojected points).

#### Both hypotheses were wrong, in opposite directions

| | YOLOE-11s @ 512 | YOLOE-11l @ 640 |
|---|---|---|
| up-stairs, pitch **0°** | **0%** (0/21) | **19%** (4/21), control FP 0% |
| up-stairs, pitch **+30°** | 0% | **0%** |
| down-stairs geometry, 0° | 45% | 50%, control FP 0% |
| `stairs` in the label histogram | **0 of 580** | **8 of 584** |

**Pitching up makes it worse, not better.** Stairs are not out of frame; tilting
turns them out of it. So the look-up probe is not the fix, and neither is a
dedicated segmentation model — the recall is there at level pitch, on a bigger
detector.

#### The finding underneath: every number in this log ran on the small detector

`ascent_matched.yaml` says in its own header: *"It deliberately does not pick a
detector … Always stack a profile, and say which one produced a number when
quoting it."* No profile was ever stacked. The container had only
`yoloe-11s-seg.pt`, and `/workspace/data/weights` is a named docker volume that
**shadows** the repo's `data/weights`, so a file placed in the repo is invisible
at that path.

It stayed invisible because the fingerprint records `cfg.detector.name`, which is
`"yoloe"` for *both* the 11s@512 and 11l@640 configs — they share `base_yoloe`.
That is the **third** time the fingerprint missed the variable under test
(`floor_llm`, then `verification.min_score`, now the detector).
`detector_weights` and `detector_imgsz` are now recorded and covered by the test.

**What this does and does not invalidate.** Every paired A/B stands: both arms
ran the same detector, so the `net` values are unaffected. What is affected is
every *absolute* number — 55%/58% on 100 episodes is a 28 MB model at 512 px —
and the attribution of the gap to ASCENT. The false-positive commitment
diagnosed in S13 and the cross-floor collapse are both plausibly symptoms of the
same upstream cause rather than two independent problems.

### S15 — the 2×2, and where the gap actually is

Four 100-episode runs on `scenes20_ep0to4`, crossing the detector against the
candidate commit gate. ASCENT is at **0.70** on this set.

| | gate 0.35 | gate 0.70 |
|---|---|---|
| **YOLOE-11s @ 512** | 55.0% | **58.0%** |
| **YOLOE-11l @ 640** | 55.0% | 57.0% |

The gate is worth +3 / +2 consistently. **The detector is worth nothing** — 0 at
the loose gate, −1 at the tight one, and paired at the same gate it is 8 gained
against 8 lost. Per-category redistributes heavily for that net of zero (bed
54.5%→31.8%, tv_monitor 25%→41.7%, sofa 50%→62.5%), which is a warning against
reading anything into one category at n≈15.

Two things this refutes, both mine:

* **Per-pose stair recall does not survive accumulation.** S14a measured
  up-stair recall going 0% → 19% with the bigger detector. Climb attempts went
  5 → 4. `StairDetector` needs `min_hits=3` frames on the same cells and
  `min_cells=25`; 19% per pose does not get there. The lever is the accumulation
  threshold, not the detector.
* **The false positives are not small-model artifacts.** far-commit failures are
  30 → 32 across a 2.5× larger detector at the same gate. Whatever the agent is
  walking to, a better detector still detects it.

#### The entire remaining gap is cross-floor

| | single-floor (n=79) | cross-floor (n=21) |
|---|---|---|
| best run (11s, gate 0.70) | **68.4%** | **19.0%** |

| | SR |
|---|---|
| actual | 58.0% |
| **if cross-floor matched single-floor** | **68.4%** |
| ASCENT | 70% |

**Single-floor is already at parity.** 68.4% against ASCENT's 70% on the whole
set is within the noise of n=79. The 12-point deficit is 10.4 points of
cross-floor and about 1.6 of everything else.

That retires a framing used earlier in this log. The false-positive commitment
diagnosed in S13 is real and the gate that addresses it is the only change with
a positive effect — but it is not *the gap*, because it costs single-floor and
cross-floor alike and single-floor is already where it needs to be.

#### What the cross-floor bottleneck is, measured

Of 21 cross-floor episodes in the best run: **4–5 ever attempt a climb**, 14
never get within 80% of the goal's height, and **every one of those 14 fails**.
Of the 7 that do get there, 4 succeed — 57%, close to the single-floor rate. The
funnel is entirely upstream of navigation: the agent cannot find the stairs.

17 of the 21 need to go **up**, which is the branch that depends on the detector;
down-stairs are pure geometry at 50% recall and 0% false positives. And the
detector is not the lever, per above. So the next thing to test is
`StairDetector.min_hits` / `min_cells` — whether a 19% per-pose signal can be
turned into a component at all by relaxing accumulation, and what that costs in
false staircases.

#### A configuration trap worth stating plainly

`VerificationConfig.min_score` was raised to 0.70 in the dataclass and the change
was inert: `configs/verification/nim.yaml` sets `min_score: 0.35`, and that file
loads on every run. The gate quoted as "0.45 → 0.70" in S13 was really 0.35 →
0.70, and `min_obs` was 2 rather than 3, so the claim that "16 of 30 far-fails
sat exactly at the `min_obs` boundary" was wrong in its specifics though right in
direction. Fixed in the yaml, and
`test_composed_config_matches_the_tuned_defaults` now composes the real config
and asserts the knobs whose numbers appear in this document.

### S16 — stair accumulation: the bottleneck moves

S15 left one untested lever. Per-POSE up-stair recall improved 0% → 19% with a
2.5× larger detector and produced *no* extra climb attempts, so the loss had to
be in `StairDetector`'s accumulation — `min_hits` frames on the same cells, and
a component of at least `min_cells`.

#### The offline sweep

`scripts/measure_stair_accumulation.py` walks each cross-floor episode's
geodesic path, folds every frame in as NavAgent does, and re-thresholds the
accumulated grids. This measures **per-STAIRCASE** recall, which is what gates
behaviour, rather than per-pose detection. 38 cross-floor episodes:

| `min_hits` | `min_cells` | staircases found | false components |
|---|---|---|---|
| **3** | 25 | **8/38 (21%)** | 38 ← was the default |
| 2 | 25 | 12/38 (32%) | 37 |
| **1** | **25** | **14/38 (37%)** | **26** |
| 1 | 10 | 16/38 (42%) | 36 |

**1 dominates 3 on both axes** — more staircases *and* fewer spurious
components. Counter-intuitive until the mechanism is clear: keeping more cells
lets a flight merge into one component instead of fragmenting into pieces that
each fail `min_cells`, and the merged component is the one containing the real
staircase.

*A 10-episode read of this same sweep said `min_hits=2` was best (5/10 vs 2/10).
It was noise, and at 38 episodes the ordering inverts.* The cheap version of the
measurement would have picked the wrong value — the second time in this log that
an 8–12 episode probe pointed the wrong way.

#### A/B: 100 episodes, `min_hits` 3 → 1

| | 3 | 1 |
|---|---|---|
| SR | 58.0% | 59.0% (net +1) |
| single-floor (n=79) | 68.4% | **68.4%** — no cost |
| cross-floor (n=21) | 19.0% | **23.8%** |
| episodes attempting a climb | 4/21 | **9/21** |
| climb attempts | 5 | **16** |
| **climb success rate** | 40% (2/5) | **56% (9/16)** |
| reached the goal's storey | 7 | 6 |

The mechanism converted at the detection end and the *success rate of a climb
went up*, so the extra detections are real staircases rather than noise. SR
barely moved because the bottleneck relocated rather than lifted.

#### Where cross-floor now fails

| | of 21 |
|---|---|
| never detect a staircase at all (`y_range` 0.00, 0 attempts) | **9** |
| climb, but thrash between floors | **~7** |
| succeed | 5 |

The thrashing is new and severe, and only visible now that climbing happens at
all: `mL8ThkuaVTM:0` records **18 floor switches** in one episode and
`XB4GS9ShBRE:0` records 12 — both time out having climbed repeatedly without
settling. Timeouts are unchanged at 9/21, and cross-floor still burns 319 steps
on average against 183 for single-floor.

So the next lever is **retiring a staircase that has been climbed without
progress**, which is ASCENT's `_disabled_stair_map`. OSG has `stair_retire_cells`
for this and it measured net −1 — but that was at `min_hits=3`, when only 5
climbs happened in the entire split and there was almost nothing to retire. It
is worth re-measuring now that there are 16.

### S17–S19 — why climbs ended early, and the fix

Three measurements on the 21 cross-floor episodes of `scenes20_ep0to4`
(`eval=scenes20_crossfloor`, which carries the whole cross-floor signal:
reaching the goal's storey predicted success 5/6, never reaching it predicted
failure 15/15).

#### S17: thrashing is not the cause

`stair_retire_cells` re-measured at `min_hits=1`, where there are finally climbs
to retire. **The mechanism works and the hypothesis was still wrong**: floor
switches 49 → 33 (one episode 18 → 6), and cross-floor SR unchanged at 23.8%,
reached-the-goal-storey unchanged at 6. Net −1 overall, from one lost
single-floor episode. Reducing thrashing by a third changed nothing about
arriving, so thrashing was a symptom.

#### S18: the instrumentation that corrected my own diagnosis

I had argued from indirect evidence — partial climbs stalling at 50–75% of a
2.7–3.2 m storey, and the successes all having 1.15–1.60 m gaps — that
`floor_gap_min_m = 0.9` was firing at a mid-flight landing.

Logging the actual height gain when CLIMB declares success refuted it in one
number: **72 cm, against a 90 cm threshold.** The height branch *cannot* fire at
72 cm. Climbs were ending through the other disjunct,
`layer.key != self._climb_from_key` — **the floor stack committing a floor
change mid-flight.**

That also explains why porting ASCENT's topological exit rule
(`is_robot_in_stair_map_fast`, `map_controller.py:181-215`) bought only net +1:
it modifies `climbed`, and `climbed` was never what fired.

Both failure modes were then confirmed directly: 3 of 21 episodes allocate a
phantom floor (one reaches **5**), and `mL8ThkuaVTM:0` records **18 switches
between "two" floors while covering 1.34 m of height** — impossible for two real
storeys 2.66 m apart, so a layer had been allocated at a landing.

#### S19: freeze the floor stack during a climb

`mapping.freeze_floor_in_climb` suspends allocation and switching while CLIMB is
active, matching ASCENT's structure, where the floor index moves in exactly one
place — on leaving the stairs (`map_controller.py:299`).

| | SR | reached goal storey | **exit gain** | switches | timeouts |
|---|---|---|---|---|---|
| height (baseline) | 23.8% | 7/21 | **72 cm** | 51 | 11 |
| topological only | 28.6% | 8/21 | 61 cm | 54 | 9 |
| **freeze + topological** | **28.6%** | **10/21** | **98 cm** | 44 | 8 |

The exit gain crossing 90 cm is the mechanism confirming itself: climbs now end
through the height branch as designed. **Reached-the-goal-storey 7 → 10** is the
largest movement in that metric so far, and `XB4GS9ShBRE:0` is the clean case —
switches 8 → 2, climbed 64% → 107%, fail → success.

Two things to read carefully:

* **`climb_ok` collapses 9 → 1 and this is not a regression.** It was being
  credited by the floor-stack branch, i.e. it was counting the bug. Arrivals
  went up while `climb_ok` went down, so the counter was never measuring what
  its name suggests.
* **Phantom floors get worse, 3 → 6 episodes.** Freezing defers allocation to
  after the climb, and the agent sometimes settles on a landing then. The net is
  still positive, but this is the cost.

Cross-floor is now **28.6%** against 19.0% at the start of this thread, with
reached-the-goal-storey nearly halving the deficit. It is still far from the
68% single-floor rate.

#### What is next, and why

`mL8ThkuaVTM:0` is unchanged at 18 switches — predicted before the run, because
it records 18 switches against **1** climb attempt, so its oscillation happens
outside the CLIMB state entirely: the agent walks up and down during ordinary
frontier navigation. Gating on the state machine cannot reach that.

The generalisation is to freeze on **being on a staircase** rather than on being
in CLIMB, which `_left_the_stairs` already computes — but it currently only
knows the cells of the climb in progress, so it would need the floor's whole
stair mask (`layer.up_stair_hits` / `down_stair_hits`), which is what ASCENT
tests against.

#### S20: freeze on being on a staircase, not on being in CLIMB

The generalisation ASCENT actually implements — its floor index moves in exactly
one place, when the agent leaves the stairs. `_on_a_staircase` ports
`is_robot_in_stair_map_fast` over the floor's whole stair mask at the detector's
own `min_hits`, rather than over one climb's component.

| arm | SR | reached | exit gain | **switches** | phantom floors | timeouts |
|---|---|---|---|---|---|---|
| height (baseline) | 23.8% | 7/21 | 72 cm | 51 | 3 | 11 |
| freeze in CLIMB | 28.6% | 10/21 | 98 cm | 44 | 6 | 8 |
| **freeze on stairs** | **38.1%** | 8/21 | **114 cm** | **17** | **3** | 9 |

**gained 3, lost 0** against the baseline — the only strictly-dominating change
in the cross-floor thread.

The mechanism is unambiguous in the per-episode data:

| | S18 | S19 | S20 |
|---|---|---|---|
| `mL8ThkuaVTM:0` switches / floors | 18 / 2 | 18 / 2 | **0 / 1** |
| `cvZr5TUy5C5:4` switches / floors | 4 / **5** | 4 / **5** | **1 / 2** |
| `XB4GS9ShBRE:0` | fail | fail | **success** |

`mL8ThkuaVTM:0` is the case that motivated the generalisation — 18 switches
against one climb attempt, untouched by the CLIMB-scoped freeze exactly as
predicted before that run, and completely resolved by the stair-scoped one. The
five-floor episode collapses to two. Phantom floors return to 3 after S19 had
pushed them to 6.

Exit gain reaches 114 cm, so climbs now end well past the 90 cm threshold rather
than at 72 cm through the floor-changed branch.

**One cost, and it is the predicted one.** `reached` falls 10 → 8: freezing
defers a legitimate arrival until the agent walks clear of the stairs, so some
episodes end still nominally on the lower floor. SR rose anyway, because the
episodes that do arrive no longer burn their budget oscillating. If this becomes
the binding constraint, the lever is `stair_exit_m`, not abandoning the freeze.

**Caveat on size.** 38.1% is 8/21. `gained 3 lost 0` is a clean signal at that
size but the interval is wide; this wants confirming on the full 100 before it
is quoted as a headline.

#### ⚠️ S21–S22: the confirmation failed, and it retracts most of S18–S20

Run on the full 100 with every validated change on, twice (the second after a
bug fix described below):

| | overall | single-floor (79) | cross-floor (21) |
|---|---|---|---|
| baseline (`min_hits=1`) | **59.0%** | **68.4%** | 23.8% |
| + freeze + topological | 55.0% | 62.0% | 28.6% |
| + the same, after the fix | 54.0% | 62.0% | 23.8% |

**Net −4 and −5. The change is rejected.**

Then the measurement that matters most here — **the same 21 cross-floor
episodes, the same config, two runs**:

| | cross-floor SR |
|---|---|
| `s20` (run on the 21-episode split) | **38.1%** (8/21) |
| `s22` (the same 21, inside the 100) | **23.8%** (5/21) |

3 episodes disagree, 7 of 21 diverge, 14 are bit-identical. **The run-to-run
noise floor on this subset is ±3 episodes, i.e. ±14 percentage points — larger
than every effect measured on it.**

So the S18 → S19 → S20 progression (23.8% → 28.6% → 38.1%) is **not
established**. Those numbers are inside the variance of the split they were
taken on. The 21-episode subset is fine for reading *mechanism counters* —
switches, exit gain, phantom floors, all of which moved by factors, not by
noise — but it cannot resolve SR at all, and it was used to claim SR movement.

What survives:

* **Mechanism, confirmed and reproducible.** Exit gain 72 → ~110 cm across
  both full runs, floor switches 49 → 15, phantom floors contained. The freeze
  does exactly what it was built to do.
* **Cost, confirmed and reproducible.** Single-floor 68.4% → 62.0% in *both*
  100-episode runs. On a 79/21 split that cost dominates any cross-floor gain.
* **Benefit, not reproducible.** Cross-floor 28.6% then 23.8%.

#### A wrong diagnosis, and what it cost

The first 100-run's regression was blamed on the freeze clearing `_unassigned`,
which `in_transit()` reads and which gates whether a frame is written to the
costmap — so frozen frames were being projected against a stale `floor_y`. The
reasoning was sound and the test pinning it is worth keeping (frozen *means* on
a staircase *means* the frame belongs to no map).

**It was not the cause.** Single-floor came back at 62.0%, unchanged, and
cross-floor fell. Fixing it swapped corrupted geometry for dropped frames and
cost the same either way — which is the failure mode the code already documents
for `floor_reject_m` ("471 of 500 steps dropped … blinding the agent").

#### The methodological lesson

Three rounds of iteration were run on a 21-episode split whose noise exceeds the
effects being chased, and a narrative was built across them. The earlier
discipline in this log — mechanism counters over SR, paired comparisons, a
measured noise floor — was in place precisely to prevent this, and was not
applied to the new split. **A subset chosen to make iteration cheap needs its
own noise floor measured before its SR is read**, exactly as `dev50` did.

### S23 — the gap is single-floor, and it is object localisation

#### A correction to S15

S15 concluded "single-floor is already at parity, 68.4% against ASCENT's 70%".
That compared OSG's **same-floor** rate against ASCENT's **overall** rate — a
mistake that should have been caught when it was written. ASCENT's same-floor SR
on these 100 episodes is **80.8%**.

Re-attributing with the right number (79 same-floor / 21 cross-floor):

| | OSG | ASCENT | deficit |
|---|---|---|---|
| same-floor (79) | 68.4% | **80.8%** | **−9.8 pts** |
| cross-floor (21) | 23.8% | ~29.5%\* | −1.2 pts |
| overall | 59.0% | 70.0% | −11 pts |

\* implied: 79 × 0.808 + 21 × *x* = 70 ⇒ *x* ≈ 0.295.

**89% of the gap is same-floor**, the opposite of what S15 said. ASCENT's
cross-floor rate is also poor (~30%), so cross-floor was never where the
difference lived.

#### Where the 25 same-floor failures go

| | n | share | median dtg |
|---|---|---|---|
| **committed, ended >1 m away** | **14** | **56%** | 6.66 m |
| committed, ended ≤1 m (near miss) | 6 | 24% | 0.32 m |
| never committed | 5 | 20% | 9.23 m |

Ten of these must succeed to reach 80.8%.

**Raising the commit gate cannot do it.** The surviving far-commit failures carry
a *higher* detection score and *more* observations than they used to — score
median 0.781, `n_obs` 12, against successes at 0.846 and 10. They are confident,
well-corroborated detections.

#### What they actually are

Distance from the committed object to the nearest **annotated instance** of its
category:

| | n |
|---|---|
| within 2 m — a real instance | 5/12 |
| 2–5 m | 3/12 |
| beyond 5 m — a false positive | 4/12 |

And for six of the fourteen, the agent finishes **euclideanly within 2 m of a
valid view point but geodesically 2–20× further**:

| episode | target | euclid | geodesic | ratio |
|---|---|---|---|---|
| `6s7QHgap2fW:4` | plant | **0.33 m** | 6.64 m | **20×** |
| `6s7QHgap2fW:1` | plant | **0.38 m** | 6.91 m | **18×** |
| `qyAac8rV8Zk:2` | toilet | 1.26 m | 9.03 m | 7× |
| `ziup5kvtCCR:3` | bed | 1.64 m | 6.69 m | 4× |

The agent stops 33 cm from a valid view point, **on the other side of a wall**.
Four of the six stop with `path_consumed` — they navigated to the goal they
published and arrived. So the goal itself was on the wrong side, and the terminal
rule is not what let them through.

#### The mechanism, and it separates perfectly

In navmesh mode `_start_approach` publishes the **object centre** and lets
habitat snap it to the nearest standable point — nearest *euclideanly*. So a
centre that lands near or across a wall snaps to the agent's side.

Localisation error of the committed object:

| | median | `> 1 m` |
|---|---|---|
| successes (54) | **0.30 m** | 6/54 (11%) |
| far-commit failures (12) | **3.29 m** | **12/12 (100%)** |

**Every far-commit failure has the committed object misplaced by more than a
metre; 89% of successes have it inside one.** Commit range barely differs
(median 1.95 m vs 1.54 m), so this is not simply "detected from too far".

So the same-floor deficit is not a detection-confidence problem and not a
navigation problem. It is **object localisation**: the agent knows *what* it
found and roughly *where*, and the roughly is worth 3.3 m — enough to put the
approach goal through a wall.

Two sub-populations worth separating before fixing anything: the ~5 that are a
real instance misplaced by 1.3–1.8 m (an ellipsoid-centre problem — mask bleed
onto background through a doorway, or refine drift), and the ~4 beyond 5 m
(genuine false positives, which localisation cannot explain).

### S25 — would ASCENT's navigation fix the same-floor gap? No, and here is why

ASCENT aims at the **observed surface point nearest the agent**
(`object_point_cloud_map.py:127-130`, `_get_closest_point` at `:225`); OSG aims
at the **fitted ellipsoid centre** (`nav_agent.py:1552`). Since S23 found every
far-commit failure has its committed object misplaced by >1 m, the obvious
question is whether ASCENT's choice of target is the fix.

Both were recorded at commit time over 100 episodes and scored against ground
truth. SR was 59.0%, identical to the run without the instrumentation, so
nothing changed behaviourally.

**Measured against the object centre, the cloud point looks worse** — successes
0.30 m for the centre against 1.07 m for the cloud. That comparison is biased:
the nearest cloud point sits on the object's near *surface* while the annotation
is its *centre*, so roughly half an object diameter is baked in.

Scored against the nearest **view point** — where the agent actually has to
stand — the bias is gone:

| | centre | cloud | cloud closer |
|---|---|---|---|
| successes (51) | 0.35 m | 0.31 m | 31/51 |
| far-commit failures (14) | 2.80 m | **1.65 m** | 11/14 |

| within 1 m of a view point | centre | cloud |
|---|---|---|
| successes | **51/51** | 45/51 |
| far-commit failures | 3/14 | **5/14** |

**Switching the aim point wholesale is a net loss**: it rescues 2 more failures
and pushes 6 successes out. So the answer to "does ASCENT's navigation fix this"
is **no** — and S8 (replacing the navmesh with a sensor-only planner) is further
still from the problem, since it changes the *mover* while these failures arrive
at the goal they published.

#### What the per-episode data actually shows

Five of the fourteen already have the centre **within ~1.1 m of a valid view
point**:

| episode | target | centre → view point | cloud → view point |
|---|---|---|---|
| `6s7QHgap2fW:4` | plant | **0.44 m** | 0.21 m |
| `TEEsavR23oF:2` | sofa | **0.50 m** | 0.04 m |
| `6s7QHgap2fW:1` | plant | **0.65 m** | 0.31 m |
| `qyAac8rV8Zk:2` | toilet | 1.03 m | 1.01 m |
| `zt1RVoi7PcG:4` | tv_monitor | 1.09 m | 1.08 m |

**For these the target was never the problem.** The agent aimed half a metre
from a valid view point and still finished 6.6 m away — because that half metre
is *euclidean* and the route is *geodesic*. `6s7QHgap2fW:4` ends 0.33 m from a
view point in a straight line and 6.64 m around: a wall. In navmesh mode
`_start_approach` publishes the object position and habitat snaps it to the
nearest standable point **euclideanly**, so it snaps to the agent's side.

The other seven have centre *and* cloud beyond 2 m of any view point — the cloud
is on the wrong object, so no aiming rule reaches them.

#### The same-floor deficit, decomposed

Of the 25 same-floor failures (79 episodes, 68.4% against ASCENT's 80.8%):

| | n | what would fix it |
|---|---|---|
| far-commit: **wrong side of a wall** | **5** | goal selection by geodesic reachability, not euclidean snap |
| far-commit: **genuine false positive** | **7** | detector/verifier precision — 3 are `tv_monitor` |
| far-commit: ambiguous (2–3.5 m) | 2 | — |
| near miss (≤1 m) | 6 | terminal precision |
| never committed | 5 | exploration |

Three hypotheses are now closed by measurement rather than argument: it is not
the mover (S8), not the aim point (this section), and not the commit gate (S23 —
the survivors carry *higher* scores than successes).

### S8 — sensor-only navigation

Every number above this line drives on habitat's ground-truth navmesh, which is
privileged information, and that was the single largest caveat on the whole log.
It is now **implemented and switchable**; the measurement is pending.

#### The two channels, and what replaced each

`runner.py` hands `NavAgent` exactly two simulator handles, and only when the
resolved mode is `navmesh`:

| privileged channel | what it gave away | sensor-only replacement |
|---|---|---|
| `nav_fn` = `env.action_to_goal` | `ShortestPathFollower` on the GT navmesh — a perfect map of everything walkable, including rooms never seen | `agent.navigation=pointnav`: a frozen PointNav ResNet reading `(rho, theta)` + a 224×224 depth image |
| `reachable_fn` = `env.is_reachable` | GT geodesic connectivity, used to blacklist targets on disconnected islands before committing (`unreachable_skip`) | `approach_abandon_steps`: commit, and give up after 100 steps of not arriving |

Nothing else changes hands. Pose still comes from the simulator, which is fair —
ASCENT reads GPS+compass the same way (`ascent_policy.py:229-232`).

#### Why PointNav and not the costmap planner

This was read off ASCENT's source rather than its paper. **ASCENT has no planner
at all** — no A\*, no FMM, no waypoints on flat ground. Every goal, frontier or
object, is converted to `(rho, theta)` and handed to a frozen PointNav ResNet
policy whose action is executed directly (`_pointnav` at `ascent_policy.py:837`,
`_navigate` at `:876`, wrapper at `pointnav_policy.py:51`). A grep for
`pathfinder|geodesic|_sim\.` across `ascent_policy.py`, `map_controller.py`,
`llm_planner.py`, `pointnav_policy.py` and `mapping/*.py` returns zero hits;
ASCENT's only navmesh use is in `utils.py:337-413`, inside the
`MultiFloorTopDownMap` *measure* that renders the output video.

So "replace the navmesh with the costmap planner" would have measured a
different system. `pointnav` is ASCENT's own mover, on the same checkpoint
(`pointnav_weights.pth`), which is what makes the resulting number belong in the
same column as its 63%.

The checkpoint needed one correction on the way in. VLFM's own non-habitat
loader builds the **Spot/continuous** policy — a `GaussianNet(512, 2)` head over
linear and angular velocity — while `pointnav_weights.pth` is the **discrete**
habitat policy (`action_distribution.linear.weight` is `(4, 512)`,
`net.prev_action_embedding.weight` is `(5, 32)`, i.e. `Embedding(4 + 1, 32)`).
Loaded through that path the encoder loads and the action head stays randomly
initialised, filtered out as "unused keys". `planning/pointnav/discrete_policy.py`
is the matching head, so all 80 tensors load strictly; see that directory's
README.

#### The two mechanisms that came with it

Neither is optional in the way a tuning knob is. A reactive mover with no global
plan has two deadlock modes, and the machinery the costmap arm handles them with
does not survive the switch:

- **`escape_window: 30`** — ASCENT's action-history override
  (`ascent_policy.py:595-606`): thirty consecutive turns forces a forward, thirty
  consecutive forwards forces a right turn. It replaces the costmap's stuck
  detector (`controller.observe_progress`), which needs a map to mark.
- **`frontier_stick_rule: closing`** — and this one is a new rule, not the
  retune it was planned as. OSG retires a frontier when the **agent has not
  moved** (0.2 m over 15 steps); ASCENT retires it when the **distance to the
  frontier has not changed** (0.3 m over 20 steps, `llm_planner.py:239-257`,
  `constants.py:234-235`). Those catch different failures. The displacement test
  catches a motionless push against something the map cannot see, which is how
  the costmap and navmesh arms get stuck. A reactive mover does not freeze — it
  orbits, moving every step and never arriving, and the displacement test never
  fires on it.

  Measured, not argued: a pointnav smoke run before this was added held a single
  frontier from step 85 to step 500, moving throughout, with `frontier_give_up`
  reading 1 for the whole episode. `test_navigation_mode.py` pins both halves —
  the displacement rule provably misses a synthetic orbit that the closing rule
  catches. The old rule stays the default, so no pre-S8 number moves.

#### Deviations from ASCENT, and why

1. **The terminal STOP is exempt from the escape guard.** ASCENT overrides
   whatever action came back, so a thirty-turn history can flip the agent's own
   deliberate STOP and the episode never ends. The exemption is for
   `State.DONE` only; every other action still goes through the guard.
2. **Stair locomotion is unchanged.** `CLIMB` keeps its current mover, so the
   cross-floor result is not confounded by a second change landing in the same
   run. ASCENT's carrot-waypoint climb (`ascent_policy.py:1075-1112`) is a
   separate port.
3. **The spatial rejection reuses `_disabled_pts`.** ASCENT filters the
   rejected *cells* (`_disabled_object_map`, `object_point_cloud_map.py:102`)
   rather than the track id, so a false positive cannot return under a new id —
   the weakness S29 flagged. OSG already had that mechanism, feeding
   `retract_unconfirmed`; `disable_target` is the same two lines, so an
   abandoned target is killed at track birth rather than filtered at every
   query.
4. **`approach_abandon_steps` defaults OFF.** It is only asked when there is no
   oracle, which is true of `costmap` as well as `pointnav` — and in costmap
   mode it lands on the same step as the existing approach deadline and would
   convert its stop-where-you-are into a return-to-exploring. That is a real
   behaviour change to an arm every pre-S8 number was measured on, so the
   sensor presets opt in rather than it becoming a default.

#### Two things a 3-episode smoke caught that no unit test would have

Both are interactions between the new mover and code written assuming the old
one, and both are the kind that degrade a result quietly rather than crashing.

**The arrival tolerance was tighter than the mover's stop radius.** A frontier
counts as genuinely reached only within `_frontier_reach_m` = 0.5 m; outside it,
a mover reporting "arrived" is assumed to have handed back a degenerate stub
path, and the frontier is blacklisted. pointnav reports arrival at its own
`pointnav_stop_radius` of 0.9 m — so *every* genuine arrival was being read as a
stub. Measured: `frontier_stub_block` 8 / 3 / 1 across three episodes, each one
retiring a frontier the agent had actually reached. `_frontier_reach_m` is now
raised to the mover's stop radius.

**The mover was reloading its checkpoint every episode.** `NavAgent` is
constructed per episode; the driver was constructed with it. It is now built
once per run and shared, like the detector and scorer, with `NavAgent.reset`
clearing the recurrent state so nothing carries over.

#### Reading the result

`+experiment=ascent_sensor` against `+experiment=ascent_aligned` is the pair:
identical perception, mapping, exploration, ranking and terminal rule, differing
only in the mover and the reachability oracle. It is **not** a single-variable
A/B — the escape guard and the abandon rule change with it, because a
sensor-only agent without them is not a weaker ASCENT but a deadlocked one.

`+experiment=final_sensor` is the same thing on the full v1 split: the number for
the `Final results` table below.

#### A/B: `scenes20_ep0to4`, 100 paired episodes

The pair is `+experiment=ascent_aligned` (= the S27 `graph` arm, `outputs/s27_graph`)
against `+experiment=ascent_sensor` (`outputs/s8_pointnav`). Verified rather than
assumed: re-evaluating `run_eval`'s own `"algorithm"` dict against a composed
config reproduces `s27_graph`'s fingerprint with **zero** mismatches, and the
only keys that differ between the two arms are the seven navigation ones.
`outputs/s8_sensor` is the same split on the from-scratch costmap planner, so
all three movers are paired on the same 100 episodes.

| mover | privileged? | SR | SPL | mean steps | same-floor (79) | cross-floor (21) | timeouts |
|---|---|---|---|---|---|---|---|
| navmesh | **yes** | **63.0%** | 0.292 | 206.9 | 73.4% | 23.8% | 14 |
| pointnav (ASCENT's) | no | **33.0%** | 0.129 | 358.7 | 39.2% | 9.5% | 57 |
| costmap A\*/Voronoi | no | 23.0% | 0.087 | 362.9 | 27.8% | 4.8% | 53 |

pointnav vs navmesh: wins 5, loses 35, **net −30**, McNemar exact two-sided
**p < 0.0001** (40 discordant). This is not a noise-floor result.

**Removing the privilege costs 30 points.** ASCENT's mover recovers 10 of them
relative to the costmap planner, which is worth having and is why this is the
sensor-only default — but it does not close the gap.

#### The mechanism is the step budget, not goal selection

| counter (100 eps) | navmesh | pointnav |
|---|---|---|
| episodes hitting 500 steps | 14 | **57** |
| `frontier_give_up` | 14 | **134** |
| `approach_abandon` / `unreachable_skip` | 3 | **56** |
| climb attempts (ok / fail) | 22 (8 / 14) | **1 (1 / 0)** |
| floor switches | 55 | 13 |

Terminal stops tell the same story: the navmesh arm ends 84 episodes by arriving
somewhere (`path_consumed` 46, `nearest_point` 38) and 14 by timeout; the
pointnav arm ends 44 by arriving and **56 by timeout**.

So the agent is not choosing worse goals — it is failing to *reach* them inside
500 steps. `frontier_give_up` firing ~10× as often is the closing rule correctly
detecting that, and `approach_abandon` firing 56 times against the oracle's 3
genuine rejections means the 100-step approach budget is absorbing a great deal
of the episode rather than acting as a rare backstop.

The cross-floor collapse is downstream of the same thing: **one** climb attempt
against the navmesh arm's 22. The agent is not declining staircases, it is
running out of budget before it gets to one.

#### What this does and does not establish

It establishes the cost of the two privileged channels under this repo's
perception stack, at p < 0.0001, with the mover being ASCENT's own.

It does **not** reproduce ASCENT's published 63% sensor-only, and should not be
read as evidence against that number. Four substitutions from
`ascent_aligned.yaml` remain in force (CLIP for BLIP-2 ITM, YOLOE over a 41-word
vocabulary for RAM++, and one text model for Qwen2.5-7B), and ASCENT's stair
locomotion — the carrot-waypoint climb at `ascent_policy.py:1075-1112` — was
deliberately not ported, which is exactly where the 21 cross-floor episodes are
lost. The honest reading is that the mover is now the right one and the
remaining gap lives elsewhere.

Note the coincidence that `s27_graph` scores 63.0% on the navmesh, numerically
equal to ASCENT's sensor-only 63%. That is a coincidence between a privileged
run on 100 episodes and a sensor-only run on ~2000; the two are not comparable.

#### Next, in order of expected value

1. **Budget accounting.** 57% of episodes time out. Measure where the steps go
   (explore vs approach vs abandoned approach) before tuning anything —
   `approach_abandon_steps` at 100 is ASCENT's constant but ASCENT's mover has
   its obstacle map informing frontier choice differently.
2. **Port the carrot-waypoint climb.** One climb attempt in 100 episodes is not
   a tuning problem.
3. Only then re-test on the full v1 split for the `Final results` table.

### S30 — why the sensor-only arm loses 30 points

S8 measured the cost of removing the navmesh (63.0% → 33.0%). This decomposes it
and fixes three divergences from ASCENT. Every fix is config-gated; every
pre-S30 default is unchanged.

#### It is not goal selection, it is the step budget

Integrating `state_log` per FSM state over the two runs:

| | navmesh (`s27_graph`) | pointnav (`s8_pointnav`) |
|---|---|---|
| total steps | 20 691 | 35 867 |
| in `GOTO_FRONTIER` | 15 491 | 25 085 |
| in `APPROACH` | **1 510** | **7 084** |
| episodes hitting 500 steps | 14 | **57** |
| `frontier_give_up` | 14 | **134** |
| `approach_abandon` / `unreachable_skip` | 3 | **56** |
| climb attempts (ok/fail) | 22 (8/14) | 1 (1/0) |

Of the 35 episodes navmesh wins and pointnav loses, **30 timed out**, 25 entered
`APPROACH`, and **12 finished within 0.5 m** of a view point against a 0.1 m
success radius. Terminal stops: navmesh ends 84 episodes by arriving somewhere
(`path_consumed` 46, `nearest_point` 38) and 14 by timeout; pointnav ends 44 by
arriving and **56 by timeout**. The agent reaches its goals and does not stop.

#### Fix 1 — the terminal stop was gated on a live detection

`_do_approach` evaluated the terminal rule only inside `if det is not None:`.
With `terminal_rule: nearest_point` the rule is `_nearest_point_stop` →
`object_layer.nearest_point_dist_xy`, which reads the **accumulated surface
cloud**: a live detection was never one of its inputs.

ASCENT asks unconditionally, every step — `ascent_policy.py:434` recomputes
`cur_dis_to_goal` from the target cloud (`map_controller.py:845-866`), and
`:910-911` tests it with no reference to a detection.

Navmesh mode hid this: `ShortestPathFollower` reports arrival, which becomes
`path_consumed` → `DONE` → STOP for 46 of 100 episodes. A sensor-only mover has
no arrival signal — inside 1 m the creep returns `move_forward` forever — so the
approach could only end by seeing the target again or timing out. **28 episodes
entered `APPROACH`, closed to a median 0.67 m of their goal, and never issued
STOP**; 11 of them held a target surface cloud, which is exactly the input the
rule needs. Navmesh wins 75% of those same 28.

Flag: `agent.terminal_requires_detection` (default True = old behaviour).

#### Fix 2 — a PointNav STOP retired the frontier

ASCENT treats a network STOP on an *explore* frontier as noise: it overwrites the
action with `MOVE_FORWARD` and keeps the target (`ascent_policy.py:705-711`). It
disables on a network STOP only in the two **stair** paths (`:810-815`,
`:1011-1014`). The port did the opposite — one spurious STOP cost a whole pursuit
plus a 100-step block. The code comment asserting otherwise cited those very
lines; it was wrong and is corrected.

Underneath was a design fault: the driver returned a bare `None` for both
"arrived" and "network said stop", disambiguated downstream only against
`_frontier_reach_m`, which is pinned to the same 0.9 m — so the two cases met
exactly at `rho == 0.9`. The driver now returns a `NavStep(action, reason)`.

Flag: `agent.pointnav_stop_means_blocked` (default True = old behaviour).

#### Fix 3 — frontiers were vetoed by an A\* plan the mover discards

`_select_new_frontier` would not enter `GOTO_FRONTIER` unless `_plan_to`
succeeded, and `ascent_selector._first_plannable` filtered the same way — but in
pointnav mode `_follow_path` takes the driver branch and never reads
`_current_path`. A costmap planner was vetoing frontiers for a mover that does
not use the costmap. ASCENT gates on nothing (`ascent_policy.py:705`).

Implemented as `StraightLinePlanner` — the absence of a planner in the shape the
selectors already expect — so neither selector needed a new parameter. Paired
with a cheap per-step check that releases a frontier whose goal no longer borders
UNKNOWN space, i.e. one explored away mid-pursuit: ASCENT never faces that
because it rebuilds and re-picks its frontier list every step
(`map_controller.py:528`, `ascent_policy.py:684`), while OSG commits.

Flag: `agent.frontier_reachability_gate` (default True; forced True whenever the
planner actually drives).

#### ⚠️ Endpoint drift: A/Bs separated by days are not clean

Re-running `ascent_aligned` on the first 3 episodes after these changes,
episodes 1 and 2 reproduced `s27_graph` **byte-identically** — steps, `llm_calls`,
`value_calls`, `steps_to_first_candidate`, full `state_log`. Episode 0 did not.

It is not the code. At the divergence point (step 13, the first frontier
selection) all runs share the identical agent pose `[7.19, 4.89]` and the
identical 9 candidate frontiers, so costmap, extraction and candidate set are
untouched. Only the LLM ranker's forced choice differs: `[3.58, 3.28]` on
2026-09-03 versus `[8.33, 1.06]` on three separate runs on 2026-09-05.

The hosted NIM endpoint answers this prompt differently than it did two days
earlier. **Consequence for every A/B in this log that compares runs taken on
different days:** part of the delta is endpoint drift. It is symmetric noise, so
it inflates discordant pairs and costs statistical power rather than biasing the
sign — a significant McNemar result survives it, a null one is ambiguous and
needs a contemporaneous baseline.

#### Not aligned, deliberately

- **Obstacle band.** ASCENT uses `[0.61, 0.88]` m (`VLFMConfig`,
  `base_objectnav_policy.py:385-386`, not overridden by its yaml); OSG uses
  `[0.15, 1.5]`. Narrowing toward `[0.15, 0.88]` previously regressed SR
  40% → 28.6% (`config.py:626-630`), so it is left alone.
- **Ranker cadence** (ASCENT every step, OSG every 20), **`look_up`/`look_down`**
  (OSG never emits them), and **the per-floor re-scan after a climb** (ASCENT
  re-initialises 12 turns, OSG does not). All real; none is the binding
  constraint while 57% of episodes time out.

#### A/B: `scenes20_ep0to4`, paired against `outputs/s8_pointnav`

Run B = `+experiment=ascent_sensor_fix1` (Fix 1 alone).
Run C = `+experiment=ascent_sensor` (all three).

**Prediction for Run B, stated before running it:** 33.0% → ≈44% (+11 — the
episodes that entered `APPROACH` holding a target cloud and never stopped);
`APPROACH` steps 7 084 → ~3 000; `approach_abandon` 56 → ~20. **Under +5 and the
diagnosis is wrong.**

#### Run B measured: Fix 1 alone, 33.0% → 42.0%

| | baseline (`s8_pointnav`) | Run B (`s30_fix1`) | navmesh (`s27_graph`) |
|---|---|---|---|
| SR | 33.0% | **42.0%** | 63.0% |
| SPL | 0.129 | 0.182 | 0.292 |
| mean steps | 358.7 | 315.1 | 206.9 |

Paired: wins 13, loses 4, **net +9**, McNemar exact two-sided **p = 0.049**.

The point of the prediction was that it could be wrong. It was not — and it held
on the *mechanism*, not just the headline:

| | predicted | measured |
|---|---|---|
| SR | ≈44% | 42.0% |
| `APPROACH` steps | 7 084 → ~3 000 | 7 084 → **3 949** |
| `approach_abandon` | 56 → ~20 | 56 → **23** |
| timeouts | — | 57 → **43** |
| `GOTO_FRONTIER` steps | unchanged (Fix 1 is not an exploration fix) | 25 085 → 24 025 |

Terminal stops moved exactly where the diagnosis said they would: `timeout` 56 →
43, `nearest_point` 43 → 49, and `nearest_point_stalled` 1 → **8** — that last is
the "cannot get closer" clause finally firing, which it can only do once the rule
is allowed to run without a live detection.

**The cost, stated plainly.** The 13 gains stop at a median distance-to-goal of
**0.030 m**. Of the 4 regressions, one is an unrelated trajectory change (it ends
0.04 m from the goal but times out); the other **three are genuine premature
stops** — the blind rule fired on a cloud that was misplaced, ending 1.5-7.3 m
out. That is the real trade of removing visual confirmation from the terminal
decision, and it is what ASCENT's `_double_check_goal` gate exists to catch.
`verification.approach_recheck` is OSG's port of that gate and is currently off
(S26/S29); re-testing it on top of Fix 1 is now the obvious follow-up, because
the failure it targets has gone from hypothetical to three measured episodes.

#### Run C measured: fixes 2 and 3 are null

| | baseline | Run B (Fix 1) | Run C (all three) | navmesh |
|---|---|---|---|---|
| SR | 33.0% | **42.0%** | 41.0% | 63.0% |
| SPL | 0.129 | 0.182 | 0.191 | 0.292 |

Run C against Run B: wins 6, loses 7, **net −1**, McNemar **p = 1.00**. Noise.

The pre-registered criterion was "if SR moves less than +3 over Run B, report
Fix 2/3 as null rather than folding them in". It did not, so they are not folded
in: **`ascent_sensor` now carries Fix 1 only**, and
`+experiment=ascent_sensor_all3` preserves the faithful three-fix arm.

**This is a null about the world, not a no-op.** Both fixes demonstrably fired —
`frontier_consumed` 213, `pointnav_stop_forced_forward` 27, `frontier_stub_block`
8 → 0. The prediction attached to them was simply wrong, and in the informative
direction:

| | predicted | measured |
|---|---|---|
| `GOTO_FRONTIER` steps | 25 085 → under 20 000 | 24 025 → **24 058** (flat) |
| `frontier_give_up` | 134 → under 60 | 127 → **168** (rose) |

So: not retiring a frontier on a spurious network STOP, and not gating frontiers
on an A\* plan the mover discards, does **not** buy exploration budget back.
`frontier_consumed` firing 213 times says frontiers are routinely explored away
mid-pursuit — releasing them just sends the agent back to pick another, which is
why `frontier_give_up` went up rather than down.

**What this rules out.** The remaining sensor-only gap is not the frontier
retirement policy. Exploration still burns 24k of 31k steps against the navmesh
arm's 15.5k, and that is now the largest unexplained block. The obstacle band
(deliberately not changed, see above) and the per-step re-ranking cadence are the
two untested candidates left on the exploration side.

### S31 — ASCENT's carrot-waypoint stair traversal

S30 left the cross-floor collapse untouched and said so: on `scenes20_ep0to4` the
pointnav arm made **one climb attempt in 100 episodes** against the navmesh arm's
22, while 21 of those 100 episodes need a floor change. This ports the mechanism
ASCENT uses to get up a flight.

#### What OSG did, and why it cannot work behind a reactive mover

`_enter_climb` aims at a fixed point `stair_overshoot_m` (1.5 m) past the
staircase centroid, along the direction the agent approached from. For a
straight flight that is fine. For a stairwell that turns — which is most of them
— it is a straight line through a wall. The navmesh hid this: it simply routed
around the wall to the far side. A mover that only sees depth cannot.

#### What ASCENT does instead

Re-aim every step at the **farthest thing in the depth image**
(`ascent_policy.py:1075-1112`). On a flight the treads and side walls are close
and the far end is not, so the maximum-depth bearing points along the well.
ASCENT averages every pixel at the maximum depth, converts the mean column to an
angle off boresight, and places a waypoint 0.8 m along it:

```python
normalized_u = np.clip((u - self._cx) / self._cx, -1, 1)
angle_offset = normalized_u * (self._camera_fov / 2)
target_heading = heading - angle_offset
```

Two guards come with it:

- **A ratchet** (`:1104-1121`). A single-frame bearing is noisy and on a landing
  can swing back the way the agent came, so the previous waypoint is kept unless
  the new one is closer to the recorded stair end. Released when there is
  nothing to ratchet against, when the agent is already at the end, or when the
  stall detector gives up on the end (`_disable_end`).
- **The climb cannot be ended by a navigation verdict** (`:1136-1139`,
  `:1055-1058`). A network STOP becomes a forward step. On stairs a STOP usually
  means "the treads fill my view", which is the one moment the agent must not
  stop. Ending is left to a stall counter measured on **distance to the
  staircase, not height** (`:1036-1046`) — a mid-flight landing gains no height
  for several steps, and a height rule cuts the climb off exactly there. Past 15
  stalled steps the ratchet releases; past 30 the climb is abandoned.

#### The port

`agent.climb_carrot` (default **off** — it replaces the overshoot goal outright,
and every cross-floor number so far was measured on that goal), with
`climb_carrot_m: 0.8`. Preset: `+experiment=ascent_sensor_carrot`, which differs
from `ascent_sensor` in that one key and nothing else.

`_climb_goal_xy` stands in for ASCENT's `_up/_down_stair_end` in the ratchet —
both are a point placed past the flight along the approach direction. The L1
comparison in map pixels becomes L2 in metres, which orders candidates
identically without a grid to quantise to.

**One sign is inverted on purpose.** ASCENT subtracts `angle_offset` because its
heading is CCW-positive; OSG's ground-plane heading is CW-positive (`turn_left`
*decreases* `agent_heading`, which `tests/unit/test_controller.py` asserts), so
the port adds. This is the same class of error that would have made the S8 mover
mirror every turn, and it is pinned the same way: `tests/unit/test_climb_carrot.py`
checks that a far pixel on the right places the carrot to the right, that the
bearing rotates with the agent, and that it never exceeds half the field of view.

**Not ported, deliberately:** ASCENT's phase 1 (pointnav to the stair centroid
until the network stops, `:1046-1058`) is subsumed — OSG only enters `CLIMB`
once already within `stair_reach_m` (0.6 m) of the centroid. ASCENT's phase 2
tilts the camera down for descents (`:1069-1073`); OSG never emits
`look_up`/`look_down`, so there is no pitch to correct. That omission is a known
gap for *descending* stairs specifically.

#### Measured: correct, and unmeasurable on this split

`+experiment=ascent_sensor_carrot` vs `ascent_sensor`, 100 paired episodes
(`outputs/s31_carrot` vs `outputs/s30_fix1`):

| | Fix 1 | + carrot | navmesh |
|---|---|---|---|
| SR | 42.0% | 44.0% | 63.0% |
| SPL | 0.182 | 0.179 | 0.292 |
| **cross-floor SR (21 eps)** | **9.5%** | **9.5%** | 23.8% |
| `climb_attempt` | 1 | **1** | 22 |

wins 2, loses 0, net +2, **p = 0.50**.

**The +2 is not the carrot.** `CLIMB` was entered in exactly one episode of 100
(`mL8ThkuaVTM:0`), and the two episodes that flipped to success
(`Dd4bFSTQ8gi:1`, `p53SfW6mjZe:1`) are **not** that episode — the carrot code
never ran in either. Both are among the four episodes Run B had regressed on, so
they are the same LLM-endpoint noise flipping back. Cross-floor SR is identical
to the digit in both arms.

**The port is not what failed; the premise was.** The carrot improves stair
*traversal*, and traversal is not the binding constraint — **entry** is. The
agent reaches `GOTO_FRONTIER` in 21 of 21 cross-floor episodes but reaches a
staircase in 1. Meanwhile the navmesh arm enters `CLIMB` 22 times at a median
step of 137 (p25 = 83), so those staircases are reachable early and are not
merely being crowded out by the step budget — though budget is clearly a
compounding factor, with the sensor-only arm averaging 445 steps and timing out
in 17 of 21 cross-floor episodes against the navmesh arm's 315 and 9.

Kept switchable and **off by default**: it is a faithful port with unit-tested
geometry, and it becomes testable the moment stair entry works. Nothing about
this result says the mechanism is wrong — only that this split cannot see it.

**What to fix next, and what not to.** Not traversal. The open question is why a
depth-only agent selects and reaches stair frontiers 1/21 times where the navmesh
agent manages 22 — stair detection, stair-frontier ranking, and the interaction
with the 500-step budget, in that order. Note also the un-ported phase 2
(`look_up`/`look_down` for descents, `ascent_policy.py:1069-1073`): OSG never
tilts the camera, and `scripts/measure_stair_recall.py:374` already concluded
offline that a look-down probe fixes a viewpoint-limited stair recall of 24%.
That is the most concrete lead.

### S32 — the down-look stair probe, and a rejected signal worth reopening

S31 ended with a lead: OSG never tilts its camera, and `measure_stair_recall.py`
carried an untested branch. That lead was half wrong and needed correcting before
anything was built.

**What was already settled.** S14a measured the UP probe and killed it: up-stair
YOLOE recall fell from 19% at level pitch to 0% at +30 degrees. The line at
`measure_stair_recall.py:374` that reads like a conclusion is a *decision rule*,
written before the run. Nothing here tilts up to search, and a unit test now
asserts that `look_up` is only ever emitted to undo a `look_down`.

**What was genuinely untested.** The probe only ever rendered `[0, +30]`. Down
stairs are a different signal entirely — pure geometry
(`mapping/stairs.py:_below_floor_points`), no detector — and the question for
them is not classification but framing: a level 79-degree frustum at 0.88 m stops
covering floor about two metres out, so a stairwell beyond that is never sampled.

#### Measured: 40 cross-floor episodes, 443 poses, three pitches

Sign check passed (mean back-projected height +1.57 m level, +2.42 m up,
+0.92 m down), so the pitch is applied in the direction the labels claim.

| signal | pose set | 0° | **−30°** | +30° |
|---|---|---|---|---|
| below-floor geometry | stair_down (n=131) | 49.6% | **59.5%** | 0.0% |
| below-floor geometry | control (n=62) | 0.0% | **3.2%** | 0.0% |
| YOLOE `stairs` | stair_up (n=250) | 10.4% | 3.2% | 7.2% |
| ramp geometry | stair_up | 46.4% | 28.4% | 0.4% |
| ramp geometry | control | 24.2% | **0.0%** | 6.5% |

**The down probe works, modestly.** Down-stair recall rises 49.6% → 59.5% for
3.2 points of control false-positive. Real, and in the predicted direction.

**It is also confirmed that pitch does not help up-stairs**, in either
direction — 10.4% level against 7.2% up and 3.2% down. S14a reproduces.

#### The result that was not being looked for

`_ramp_geometry` is a **rejected** candidate in this repo, and its own docstring
says why: *"32% recall at 28% false-positive on flat control poses, i.e. it fires
on ordinary corridors nearly as often as on stairs."* That rejection was measured
at level pitch only.

From a down-tilted frame it reads **28.4% recall at 0.0% control false-positive**
(0/62). Same signal, same scenes, an operating point that is not comparable to
the one it was rejected at — a corridor floor tilts into view and is flat, while a
flight still rises.

That matters more than the down probe does. Up-stairs are **17 of the 21**
cross-floor episodes on `scenes20_ep0to4`, and the only up-stair signal OSG has
today is YOLOE at 10.4%. A detector-free geometric signal at 28% recall and no
measured false positives would be the largest available change to stair entry —
which S31 identified as the actual cross-floor bottleneck (`CLIMB` entered in
1 of 21 cross-floor episodes, against the navmesh arm's 22 of 100).

It is not implemented here: reopening a rejected detector is a different change
from tilting a camera, and it deserves its own stage.

#### Up-stairs: what OSG has, and what ASCENT has

Read off both sources, since this is what any further stair work has to close.

| | OSG | ASCENT |
|---|---|---|
| semantic signal | YOLOE open-vocab, `STAIR_LABELS` (`mapping/stairs.py:31`), one model | **RedNet** MPCAT40 segmentation, `STAIR_CLASS_ID = 17`, run every step (`ascent_policy.py:151, 424`) **AND** GroundingDINO prompted `"stair ."` (`map_controller.py:700-704`) |
| how they combine | detector mask → geometric gate (rise 0.35 m, slope 0.30, span 0.6 m) | `fusion_stair_mask = stair_mask & (seg_mask == STAIR_CLASS_ID)`, gated on ≥20 stair pixels (`obstacle_map.py:520-524`) — two independent detectors intersected |
| up vs down | geometry only: below-floor points, or the slope gate | **the sign of the camera pitch** (`obstacle_map.py:534-542`): same pixels become `_up_stair_map` or `_down_stair_map` |
| stairs in the map | ordinary obstacles in the `[0.15, 1.5]` band | dedicated `_up/_down_stair_map`, written into the obstacle map and **excluded from agent-radius dilation** |
| entering a climb | deliberate only — select a stair frontier, arrive within `stair_reach_m` | that, **plus passive entry**: standing on stair cells for N steps triggers a climb (`map_controller.py:626-671`) |
| down-stairs | below-floor points from a level frame | that, plus the inverted-depth trick and an explicit look-down mode (`obstacle_map.py:547-558`, `ascent_policy.py:800`) |

**OSG's own pre-registered rule already picked the answer.** S14a wrote: if
up-stair recall rises with pitch it is a viewpoint problem (S14b, cheap); if it
does not, *"the detector is the problem and only a dedicated segmentation model
will move it (S14c, RedNet, ~200 MB)"*. Measured above: 10.4% level, 7.2% up,
3.2% down. Pitch does not help — so by that rule the answer is S14c, and ASCENT
is running exactly the model S14c names.

**Detection is not the whole gap, and the runs separate the two halves.** The
navmesh arm carries the *identical* stair detector and reaches 23.8% cross-floor
with 22 climb attempts; the sensor-only arm reaches 9.5% with 1. So there are two
independent deficits: a **detection** deficit shared by both arms (YOLOE vs
RedNet), and a **reaching/entry** deficit that only the sensor-only arm has,
where ASCENT holds passive entry, `_get_close_to_stair` with its own stall
handling, and the carrot traversal already ported in S31.

**Not a measurement of ASCENT.** Its README carries no results table; the 63% is
an aggregate on HM3D v1 val taken from the paper, and no cross-floor breakdown
for ASCENT exists here. The table above is a source comparison, not a claim
about which system scores better on cross-floor episodes.

#### Shipped, and left off

`agent.down_look_every` (0 = off). One probe is `look_down`, observe, `look_up`;
the agent never moves while tilted, so the mover's depth input stays level. The
tilted frame is handed to the stair detector directly rather than through
`_on_keyframe`, because whether a 30-degree tilt counts as a keyframe depends on
`keyframe_rot_deg` also being 30 — the probe would otherwise work by coincidence.

Pose needs no bookkeeping: `sim/habitat_env.py` reads `T_wc` from the sensor's
own state, so a tilted frame back-projects correctly. ASCENT has to track
`_pitch_angle` by hand and fold it into its transform (`ascent_policy.py:237`).

**Default off, and no A/B run, deliberately.** The gain is +10 points of recall
on a signal that governs down-stairs, which are 4 of the 21 cross-floor episodes
— an upper bound of roughly 4 episodes in 100, before any of it has to convert
into a success. The paired noise floor on this split is wider than that: the S31
carrot arm moved 2 episodes with 2 discordant pairs. A 100-episode A/B is
**underpowered to detect the effect this probe can produce**, so running one
would buy an uninterpretable number for 3.5 hours of compute. It becomes worth
measuring bundled with a change to up-stair entry, where the effect is large
enough to see.

### S33 — ASCENT's up-stair detector: 5x the recall, zero extra climbs

S32 ended by naming RedNet as the thing S14a's rule pointed at. This ports it,
measures it in isolation, and runs the A/B. The detection result is large. The
navigation result is nothing, and that is the finding.

#### What ASCENT does, and what it is worth here

```python
# ascent/mapping/obstacle_map.py:520-524
if np.any(stair_mask) > 0 and np.sum(seg_mask == STAIR_CLASS_ID) > 20:
    fusion_stair_mask = stair_mask & (seg_mask == STAIR_CLASS_ID)
```

RedNet's MPCAT40 stair class (id 17, run every step, `ascent_policy.py:151, 424`)
intersected with GroundingDINO prompted `"stair ."`
(`map_controller.py:700-704`), gated on ≥20 segmenter pixels, projected into the
stair map with no geometric check.

Measured on 250 stair poses from the cross-floor split, control false-positive in
brackets:

| signal | up-stair recall |
|---|---|
| YOLOE `stairs` + geometric gate (what OSG ran) | 10% [0%] |
| **ASCENT's exact fusion, RedNet AND YOLOE** | **10% [0%]** |
| RedNet through OSG's geometric gate | 27% [2%] |
| **RedNet alone** | **54% [3%]** |

**The faithful port is worth nothing here, structurally.** An intersection cannot
beat its weaker input, and ASCENT's second opinion is GroundingDINO, which this
repo does not run. All 26 YOLOE firings already sit inside RedNet's 134, so the
AND discards 81% of what RedNet found. All three are available as
`agent.stair_up_mode` (`detector` | `ascent` | `rednet`); the preset uses
`rednet`.

#### A/B: `scenes20_ep0to4`, 100 paired episodes

`+experiment=ascent_sensor_rednet` against `ascent_sensor` (`outputs/s33_rednet`
vs `outputs/s30_fix1`), differing in exactly two fingerprint keys.

| | Fix 1 | + RedNet | navmesh |
|---|---|---|---|
| SR | 42.0% | **42.0%** | 63.0% |
| SPL | 0.182 | 0.171 | 0.292 |
| cross-floor SR (21) | 9.5% | 4.8% | 23.8% |
| **`climb_attempt`** | **1** | **1** | 22 |
| `floor_switches` | 13 | 14 | 55 |

wins 2, loses 2, **net 0**, McNemar **p = 1.00**.

**Five times the up-stair recall produced exactly zero additional climb
attempts.** That is a clean negative and it is the third independent
confirmation of S31: the cross-floor bottleneck is not detection.

#### Why the probe result did not transfer

The probe teleports the camera to poses **on the geodesic path, facing along
it** — i.e. standing near a staircase looking at it. It answers "if the agent
were there, would it see the stairs", and RedNet's answer is much better than
YOLOE's. It does not answer "does the agent ever get there", and the A/B says it
does not: with 57 of 100 episodes previously timing out and 24k of 31k steps
spent in `GOTO_FRONTIER`, the agent rarely occupies the poses the probe measures.

A recall number measured on curated poses is an upper bound on what a detector
swap can buy, not an estimate of it. That is worth remembering for the next
detector question in this log.

#### Where the chain is now instrumented

Detection → stair component → stair frontier offered → frontier selected →
agent within `stair_reach_m` → `CLIMB`. Nothing recorded which link drops the
gain, so `stair_rounds`, `stair_frontiers_seen` and `stair_frontier_selected`
are now in the episode stats. The next cross-floor run will say whether
staircases are never offered, offered and never chosen (they carry
`stair_prior` 0.6 against value-map scores), or chosen and never reached.

Kept off by default: 626 MB and ~30% of control-loop throughput for a measured
net zero.

### S34 — ASCENT's exploration cadence: ported half, and it cost 7 episodes

#### The measured difference

Per frontier pursuit, over the same 100 paired episodes (`s30_fix1` vs
`s27_graph`):

| | navmesh | pointnav |
|---|---|---|
| median planned cost | 3.67 m | 3.23 m |
| median steps spent | 16 | **23** |
| detour factor | 1.35 | **1.67** |
| p90 steps | 52 | **92** |
| closes distance at | 0.076 m/step | **0.046 m/step** |
| moved AWAY from the goal | 0% | **9%** |

Pursuits start from the same place (median 2.72 vs 2.77 m out) and close a
similar fraction of it, so the frontier CHOICES are not what differs. The mover
is slower — and ASCENT shares that mover, so that half is not an OSG/ASCENT gap
at all.

What OSG adds is commitment. ASCENT rebuilds its frontier list
(`map_controller.py:528`) and re-runs the whole selection
(`ascent_policy.py:684`) every step; its "commitment" is bookkeeping inside the
selector, not an FSM state. OSG holds one target for a median 23 steps, chosen
from a map two dozen steps out of date, behind a mover closing 4.6 cm/step.

#### A/B: reselect + carrot + RedNet, 100 paired episodes

`+experiment=ascent_sensor_full` vs `ascent_sensor` (`outputs/s34_full` vs
`outputs/s30_fix1`):

| | Fix 1 | + reselect + stairs | navmesh |
|---|---|---|---|
| SR | **42.0%** | 35.0% | 63.0% |
| SPL | 0.182 | 0.160 | 0.292 |
| cross-floor (21) | 9.5% | 4.8% | 23.8% |
| `GOTO_FRONTIER` steps | 24 025 | **24 411** | 15 491 |
| timeouts | 43 | **56** | 14 |
| `frontier_switch` | 0 | **4 002** | 0 |
| `frontier_give_up` | 127 | 21 | 14 |
| `climb_attempt` | 1 | **7** | 22 |

wins 9, loses 16, net −7, McNemar p = 0.23 — directionally worse, not
statistically established, but the mechanism is not ambiguous.

#### What went wrong: the cadence was ported without its damping

**4 002 target switches across 100 episodes — 40 per episode.** The agent
reconsiders, picks a different frontier, walks a few steps, reconsiders again.
`frontier_give_up` collapses from 127 to 21 precisely because pursuits are
switched away before the stall rule can accumulate its 20 steps.

And it bought nothing: `GOTO_FRONTIER` steps went 24 025 → 24 411, flat. Cutting
commitment was supposed to reduce time spent walking at stale targets; it did
not, and timeouts rose 43 → 56.

ASCENT's re-selection is safe because of the other half of its design — the
sticky counter and `_force_frontier` in `_handle_frontier_stick_and_disable`
(`llm_planner.py:239-271`), which re-forces a previously chosen frontier and
retires one selected too many times. In OSG that bookkeeping is
`FrontierCommitState`, and it is effectively bypassed on this arm: `observe()` is
called only `if self.commit_state is not None and not self._ascent_rank`, and the
preset runs `selector: ascent`, so `_ascent_rank` is True. Re-selection cadence
without commitment bookkeeping is half a mechanism, and the missing half is the
one that stops thrashing.

**`reselect_every` stays 0 by default.** Re-testing it means porting
`_force_frontier` first, not tuning the interval.

#### The one thing that did move: the stair chain

`climb_attempt` **1 → 7**, with `stair_frontier_selected` at 1 230. The
instrumentation added after S33 also localised where the stair gain had been
disappearing — from a single smoke episode:

```
stair_rounds 149   stair_frontiers_seen 386   stair_frontier_selected 66
climb_attempt 0
```

Staircases are detected, offered in every selection round, and win 66
selections — and the agent still never gets within `stair_reach_m` (0.6 m) of
one. So the cross-floor chain breaks at the **last** link, reaching, not at
detection (S33) or ranking. That is now three stages agreeing.

Even at 7 attempts, cross-floor SR did not improve (9.5% → 4.8%, one episode,
noise). Reaching staircases more often is necessary and evidently not
sufficient.

#### Cost note

RedNet is free at this cadence (median control fps 2.64 against 2.68 without
it). Per-step selection is not: `select_every: 1` costs 3.3x the control loop
(0.81 fps, ~11 h for 100 episodes), because OSG's selection re-runs contour
extraction, stair extraction and ranking while ASCENT gets its frontier list as
a by-product of the obstacle-map update it already performs. This arm therefore
reconsidered every 5 steps, not every 1 — a forced deviation, and one that makes
the thrashing result a lower bound on what the full cadence would have done.

### S35 — the alignment inventory, and a reframing that retires four stages

#### Where the gap actually is

ASCENT is at **70%** on `scenes20_ep0to4`. OSG's navmesh arm is at 63.0% and the
best sensor-only arm at 42.0%. Splitting the 21-episode sensor-only deficit by
floor (`s30_fix1` against `s27_graph`, same 100 episodes):

| | n | navmesh | sensor-only | lost |
|---|---|---|---|---|
| **same-floor** | 79 | 73.4% | 50.6% | **18 episodes** |
| cross-floor | 21 | 23.8% | 9.5% | **3 episodes** |

**Eighteen of the twenty-one lost episodes are same-floor.** S31 through S34
spent four stages on stairs, which is three of them. That was a
misallocation, and the decomposition above should have been run first.

And the same-floor losses are overwhelmingly a budget problem, not a
perception or a commitment problem — of 23 gross same-floor losses:

| bucket | n |
|---|---|
| timed out holding a candidate | **11** |
| never found a candidate | **7** |
| stopped >1 m from the goal | 3 |
| stopped, near miss (<=1 m) | 2 |

18 of 23 are "ran out of steps", which matches the mover measurement from S34:
the PointNav policy closes distance at 0.046 m/step against the navmesh
follower's 0.076.

#### The alignment inventory

Everything that still differs from ASCENT, and its status:

| deviation | OSG | ASCENT | status |
|---|---|---|---|
| mover | PointNav ResNet | same | **aligned** (S8) |
| terminal stop | map-based, every step | same | **aligned** (S30, +9 episodes) |
| initial scan, camera, action space, protocol | — | — | **aligned** (S0/S8) |
| **obstacle band** | [0.15, 1.5] | **[0.61, 0.88]** | **OPEN — S35** |
| value map backend | CLIP ViT-B/32 | BLIP-2 ITM | open; `lavis` not installed |
| object tags for prompts | YOLOE, 41 words | RAM++ | open; weights ARE on disk |
| text model | llama-3.2-11b-vision | Qwen2.5-7B | open; needs local serving |
| detector | YOLOE-11s @ 512 | D-FINE + GroundingDINO | **measured worth 0** (S15) |
| stair detection | YOLOE `stairs` | RedNet MPCAT40 | ported; **null** (S33) |
| stair traversal | fixed overshoot | carrot waypoint | ported; **null** (S31) |
| exploration cadence | commit until exit | per-step + damping | half-ported; **negative** (S34) |

Four of the remaining differences have now been ported and measured, and three
of those were null or worse. The detector — the intuitive candidate — was
measured worth zero back in S15. What is left that is both open and free is the
obstacle band.

#### Why the band is worth testing now, when it was not before

`MappingConfig` records an A/B where narrowing "the FULL band to the old
[0.15, 0.88] regressed SR 40% -> 28.6%", and S32 cited it as reason to leave the
band alone. Two things weaken that:

1. **It tested the wrong half.** [0.15, 0.88] keeps the conservative 0.15 floor
   and drops the ceiling — the worst of both, still marking every low object
   while missing tall ones. ASCENT's band raises the **floor** to 0.61, and that
   half has never been measured.
2. **It was measured on an arm where the costmap drives.** Under
   `navigation=pointnav` it does not: `_follow_path` takes the driver branch and
   never reads `_current_path`, and obstacle avoidance is the PointNav policy
   reacting to raw depth. The band now only shapes frontier extraction — exactly
   the role it plays in ASCENT, where the same policy does the driving.

A conservative band inflates into doorways and shrinks the free component that
frontier goals are placed on. Opening it should shorten pursuits, which is what
the 18 budget-bound losses need.

**Prediction:** mean steps 315 -> under 290, timeouts 43 -> under 35. If steps do
not fall, the band is not what costs the budget and this is a null whatever SR
does.

**Measured:** _pending_ (`+experiment=ascent_sensor_band`, `outputs/s35_band`).

### S36 — how ASCENT navigates, line by line, against OSG

Read off both sources. Everything here is a per-step control-flow difference, not
a parameter.

#### ASCENT: one stateless dispatch, recomputed from scratch every step

`ascent_policy.py:385-637`. There is no FSM. Every step rebuilds every input and
re-decides:

```
act(obs):
  perception   RedNet seg + GDINO/D-FINE/SAM detections + RAM tags   (:411-431)
  maps         obstacle (per floor), value (BLIP-2), object cloud    (:433)
  distance     cur_dis_to_goal = |nearest cloud point - agent|       (:434)
  goal         _get_target_object_location()                          (:441)
  dispatch     climbing? -> _climb_stair / _get_close_to_stair
               not initialised? -> _initialize (12x turn_left)
               goal is None -> _explore
               else -> _navigate(goal, stop=True)
  override     30-step action history (:595-606)
  deadline     force STOP at max_steps-1 (:609)
```

`_explore` (`:648`) reselects the frontier every step; `_navigate` (`:876`)
re-aims at the object every step.

#### OSG: an explicit FSM with committed states

```
INIT -(12 turns)-> EXPLORE <-> GOTO_FRONTIER -> APPROACH -> DONE
                       \-> GOTO_VERIFY_VIEW -> VERIFYING     \-> CLIMB
```

`EXPLORE` selects a frontier (throttled to once per 5 steps) and hands control to
`GOTO_FRONTIER`, which drives at that frontier until it arrives or stalls.
`APPROACH` drives at a goal fixed when the target was committed.

#### The four differences that are structural

| | ASCENT | OSG |
|---|---|---|
| **object goal** | recomputed **every step** from the cloud, with hysteresis (`get_best_object`, object_point_cloud_map.py:127-150: ignore <0.1 m moves, or <0.5 m while >2 m away) | fixed once at `_start_approach`; never updated |
| **frontier goal** | reselected every step (`ascent_policy.py:684`) | committed until arrival or stall — median 23 steps |
| **arrival at a frontier** | no such concept; every call passes `stop=False`, making the `rho < stop_radius` branch inert (`:869-872`) | explicit, and it is what ends the pursuit |
| **verification** | a continuous gate: BLIP-2 cosine >= 0.15 sets `_double_check_goal` (`map_controller.py:770-776`), which the stop test then requires | a separate `GOTO_VERIFY_VIEW` -> `VERIFYING` detour with its own VLM call |

The frontier-goal difference was tested in S34 and made things worse, because
ASCENT's cadence is safe only with the damping (`_force_frontier`) that OSG
bypasses. **The object-goal difference has not been tested**, and it is the one
that lines up with the measured failure.

#### Why the static approach goal is the next thing to fix

S35's decomposition: 18 of the 21 sensor-only losses are same-floor, and of 23
gross same-floor losses, **11 timed out while holding a candidate**. Those
episodes found the object and then failed to reach it inside the budget.

OSG aims at the object's position as it was understood at commit time and never
re-aims, while the surface cloud keeps accumulating underneath it. ASCENT re-aims
every step at the nearest point of the current cloud. Since S30 the *stop* test
already reads the live cloud (`terminal_rule: nearest_point`) — so OSG is in the
odd position of deciding when to stop from fresh evidence while still driving at
a stale point.

The hysteresis is the part worth copying carefully rather than just re-aiming:
without it the goal jitters with every mask update and the mover's recurrent
state is reset on every move over 0.1 m (`pointnav_driver._reset_recurrent`),
which would cost more than the staleness does.

### S37 — ASCENT's control flow as a whole, not one mechanism at a time

#### Why a whole policy

On `scenes20_ep0to4` ASCENT scores **70% sensor-only**; OSG scores 63% *with* a
ground-truth navmesh and 42% sensor-only. ASCENT's navigation beats OSG's even
when OSG is allowed to cheat, so the gap is the pipeline rather than its
settings.

(The "63% = 63%" match noted earlier in this log is a coincidence between OSG's
privileged run on 100 episodes and ASCENT's published number on the full
~2000-episode val. On the same split the figures are 63.0% and 70%.)

Four ASCENT mechanisms were ported into `NavAgent` one at a time and three
measured null or worse — S31 carrot, S33 RedNet, S34 cadence. S36 explains the
pattern: the differences are structural. ASCENT has no state machine, while
`NavAgent` commits to a frontier for a median 23 steps and to an approach goal
for the whole approach. A mechanism lifted out of the first structure and
dropped into the second loses the context that made it work — S34 being the
clearest case, where the re-selection cadence without `_force_frontier` damping
thrashed 4 002 times and cost 7 episodes.

#### The port

`agent/ascent_agent.py` subclasses `NavAgent` and overrides exactly one method,
`_dispatch`, which `_act_inner` was split to expose. Perception, mapping, floor
stack, value map, object layer, stair detector, PointNav driver and escape guard
are inherited unchanged, so `+experiment=ascent_policy` differs from
`ascent_sensor` in exactly one fingerprint key.

| | NavAgent | AscentAgent |
|---|---|---|
| control | FSM with committed states | re-decides from the maps every step |
| object goal | fixed at commit | re-aimed every step, ASCENT's hysteresis (<0.1 m, or <0.5 m while >2 m out) |
| frontier | committed ~23 steps | re-chosen every round, **with `FrontierCommitState` fed on every choice** |
| network STOP | ends the pursuit | overwritten with a forward step, exploring and approaching alike |

Deviations, each keeping a measured OSG win: the candidate gate stays (+3,
S13/S15) where ASCENT commits to any cloud of the class; the VLM verifier stays
(largest win in the log) where ASCENT gates on a BLIP-2 cosine, which needs
`lavis`; and the terminal stop and 100-step abandon are *called* rather than
rewritten, since S30 already ported them — so both arms decide terminal
questions identically.

#### A correction the tests caught

The port originally carried a comment claiming the opening 12-turn scan yields
to a target found during it. A test failed, and `ascent_policy.py:566-576` shows
ASCENT orders it the other way: `not done_initializing` precedes
`elif goal is None`, so the scan finishes first. The code was right and the
comment was wrong. The ordering is now pinned by a test, since the opposite is
the intuitive guess and would be a silent divergence.

#### A 0/3 smoke that meant nothing

The first `val_mini` smoke returned 0/3 with three 500-step timeouts, which
looked like a broken port. Running the **baseline on the same three episodes**
returned 0/3 as well, ending *further* out on all three (dtg 20.9-22.6 m against
the port's 15.5-20.6 m). Those episodes are simply hard, and `val_mini` is a
different split from the one every number here is measured on. A smoke on
non-comparable episodes cannot condemn an arm; only the paired control could
say so.

Also worth recording: `ascent_frontier_retired` firing 6-16 times per episode is
not a counter bug. `observe` keeps a cumulative reference distance, so an agent
closing at 0.046 m/step crosses the 0.3 m reset after ~7 steps and never reaches
the 20-step retirement. Frontiers being retired means the agent is genuinely not
closing — the same thing the timeouts report.

#### S38 — the run was killed: a verifier rejection was permanent

22 episodes in, the arm looked like it was collapsing partway through the split.
It was not, and the check that showed it is the same one that mattered in the
smoke: run the BASELINE on the same episodes.

`scenes20_ep0to4` is ordered by scene, five episodes each, so "fails from
episode 12" is "fails from scene 3":

| scene | eps | ascent_policy | ascent_sensor | navmesh |
|---|---|---|---|---|
| 4ok3usBNeis | 5 | 3 | 5 | 5 |
| 5cdEh9F2hJL | 5 | 4 | 3 | 4 |
| 6s7QHgap2fW | 5 | 1 | 1 | 2 |
| DYehNKdT76V | 5 | 0 | 1 | 3 |
| Dd4bFSTQ8gi | 2 | 0 | 0 | 1 |

Both sensor-only arms fall over on the same scenes while the navmesh arm keeps
scoring, so the sensor-only deficit is concentrated by scene rather than spread
across the split — worth knowing on its own.

**But the check turned up a real defect, and it is the opposite of what the
numbers suggested.** Three of 22 episodes ended within 0.5 m of the goal having
never issued STOP (baseline 1, navmesh 0), all three carrying `verify_reject: 1`:

| episode | ascent_policy | ascent_sensor |
|---|---|---|
| DYehNKdT76V:3 | **0.03 m**, no stop | 4.78 m |
| Dd4bFSTQ8gi:1 | **0.05 m**, no stop | 3.00 m |

On two of the three the new control flow navigated to within 5 cm where the
baseline finished 3-4.8 m out. **It navigates better and is then denied the
stop.** The defect surfaces more here precisely because the agent arrives more
often, which means the 8/22 it was scoring was a floor, not a measurement.

The mechanism, `nav_agent.py:2181-2188`: a VLM rejection called
`object_layer.blacklist(track.id)` — permanent. With the only candidate retired,
the terminal rule never runs again, and the agent explores forever while
standing on the goal.

ASCENT does not work that way. `_double_check_goal` is a **retryable gate**
re-evaluated every step until the score passes (`map_controller.py:770-776`),
and a target is abandoned only at close range where the view is best
(`ascent_policy.py:910-922`). A rejection is evidence about a VIEW, not a
verdict on an object.

`verification.reject_cooldown_steps` (0 = the old permanent behaviour) sets the
track aside instead, so it can be re-judged from closer. Both arms are being
re-run with it at 100, since the change moves the baseline too.

**Measured:** _pending_ (`outputs/s38_sensor` and `outputs/s38_policy`).

### S39 — ASCENT's whole pipeline, reimplemented under `src/ascentnav`

S30-S38 ported ASCENT one mechanism at a time onto OSG's maps and moved the
sensor-only arm 33.0% -> 42.0%. S37 showed the remaining gap was not any single
mechanism: OSG's maps differ from ASCENT's **in kind**, not in parameters. So
this stage stops porting and reimplements the pipeline whole, on ASCENT's own
maps, vendored from the reference implementation:

* `ObstacleMap` — a maintained navigable map with a fog-of-war reveal over a
  `[0.61, 0.88]` obstacle band and `detect_frontier_waypoints`, against OSG's
  per-frame depth raycast over `[0.15, 1.5]` with contour frontiers.
* `ValueMap`, `ObjectPointCloudMap` — likewise verbatim.

Three substitutions, each for something that cannot run in this environment and
each already a documented deviation: YOLOE for D-FINE + GroundingDINO +
MobileSAM (S15 measured the detector as worth zero here), CLIP for BLIP-2 ITM,
the hosted client for a local Qwen2.5-7B. The object scene graph is built from
the same object cloud and is read-only with respect to every navigation
decision.

**Measured**, `scenes20_ep0to4`, 100 paired episodes:

| arm | SR | SPL | steps | timeouts | same-floor | cross-floor |
|---|---|---|---|---|---|---|
| navmesh `s27_graph` (privileged) | 63.0% | 0.292 | 207 | 14 | 73.4% | 23.8% |
| pointnav baseline `s8_pointnav` | 33.0% | 0.129 | 359 | 57 | 39.2% | 9.5% |
| best port `s38_sensor` | 42.0% | 0.196 | 314 | 44 | 50.6% | 9.5% |
| **`s39_ascentnav`** | **55.0%** | **0.284** | 191 | 16 | **69.6%** | **0.0%** |

Against `s38_sensor`: 21 wins, 8 losses, **McNemar exact p = 0.024**. Validated
before reporting: 100 unique uids set-identical to the split, protocol matched,
no success recorded outside 0.1 m, median success 96 steps at 0.037 m.

The same-floor number is the finding. 69.6% against the privileged navmesh
arm's 73.4% says a sensor-only mover on ASCENT's maps is within 4 points of
ground-truth path planning — the 30-point loss S8 measured was never about the
mover.

#### Four bugs that made the first four episodes fail, and what they cost

Worth recording because three of the four were silent:

1. **No `detector.set_vocabulary`.** YOLOE is open-vocabulary; without the call
   it has no classes and detects nothing. Writing the agent from scratch rather
   than subclassing `NavAgent` dropped the line at `nav_agent.py:479`. Symptom:
   `steps_to_first_candidate: None` in a scene the baseline sweeps 5/5.
2. **`mask.astype(bool)`** into `_extract_object_cloud`, which does `mask * 255`
   and hands the result to `cv2.erode` — which rejects the int64 a bool array
   promotes to. ASCENT's masks arrive from `ObjectDetections` already uint8.
3. **`get_best_object` given a 3D position** where it wants 2D.
4. **`verify_target()` written and never called** — `give_up_unverified: 18`.

And one parameter: the frontier `stop_radius` was 0.9 where ASCENT passes
`stop=False`, which makes its `rho < stop_radius` branch inert
(`ascent_policy.py:869-872`). There is no such thing as arriving at a frontier.
At 0.9 the driver reported "arrived" for anything within 0.9 m and each became a
blind forced-forward: **189 of 500 steps** on the first smoke. At 0.0: none.

### S41 — the stair machinery, and the 0.0% it was hiding

`s39_ascentnav` scored **0.0% on all 21 cross-floor episodes** — worse than the
port it beat overall (9.5%) and worse than the baseline. The cause was not a
tuning failure: the vendored `ObstacleMap` was being handed **zeros for both
stair masks**, so `_up_stair_map` / `_down_stair_map` never populated, no stair
frontier was ever published, and every mechanism downstream of them — all of it
already vendored and correct — was unreachable code.

Added in `src/ascentnav/stairs.py` + the agent's climb branch:

* the fused stair input (detector mask AND RedNet MPCAT40, gated at >20 px,
  routed into the up- or down-stair map by the sign of the camera pitch);
* a **per-floor map stack** — ASCENT keeps one obstacle/value/object triple per
  storey and swaps the active triple on a transition (`map_controller.py:67-92,
  253-257`). One map collapses the storeys onto each other;
* `_do_climb`: approach -> footprint contact -> centroid -> ratcheted carrot,
  with a network STOP authoritative on the approach (`:1011-1015`) but forced
  forward mid-flight (`:1136-1139`), because a STOP on a staircase usually means
  the treads fill the view;
* `_link_stair_to_new_floor`: the climbed flight is handed to the arrival storey
  with its ends swapped and already marked explored
  (`map_controller.py:563-596`). Without it a floor that runs out of frontiers
  climbs straight back down, then up again.

**One deliberate divergence, on measured evidence.** ASCENT's rule is
`stair_mask AND rednet` where `stair_mask` is GroundingDINO. This repo has no
GroundingDINO, and S33 measured the substitute over 250 stair poses from this
split: YOLOE `stairs` 10% recall, RedNet alone 54%, and the AND of the two 10% —
all 26 YOLOE firings already sit inside RedNet's 134. An intersection cannot
beat its weaker input, so keeping ASCENT's AND would import GroundingDINO's job
without GroundingDINO and cap stair recall at 10%. The default unions them
(`stair_up_mode: rednet`), which makes the map's own AND a no-op and leaves
RedNet deciding; `stair_up_mode: ascent` restores the strict fusion.

**Measured**, `scenes20_crossfloor` (the 21 cross-floor episodes of the same
split), paired against `s39_ascentnav` on the same episodes:

| arm | SR | SPL | mean steps | timeouts | floor switches |
|---|---|---|---|---|---|
| `s39_ascentnav` | 0.0% | 0.000 | 323 | 11 | 0 |
| navmesh `s27_graph` (privileged) | 23.8% | — | — | — | 7 |
| **`s41_stairs_cf`** | **28.6%** | 0.090 | 283 | 7 | 4 |

**6 wins, 0 losses**, exact binomial p = 0.031. The sensor-only arm now beats
the ground-truth-navmesh arm on the episodes that need a floor change.

Two things the counters say that the SR does not. 30 climbs were attempted and
only 4 completed by the topological rule (off the stair map having reached its
centroid) — yet two of the six wins changed storey by 1.4 m and 1.9 m with
`climb_ok: 0`, i.e. the climb finished and the 30-step stall counter retired it
anyway. The abandon threshold is firing on successful climbs, so there is
headroom here that costs nothing to test. And `wcojb4TFT35:3` attempted 8 climbs
on a 0.2 m height range without ever looping: the disable-and-remember path
works.

**100-episode run**, `scenes20_ep0to4`, paired against `s39_ascentnav`
(`outputs/s41_stairs`, 100 unique uids set-identical to the split, no success
recorded outside 0.1 m, median success 102 steps at 0.037 m):

| arm | SR | SPL | steps | timeouts | same-floor (78) | cross-floor (22) |
|---|---|---|---|---|---|---|
| navmesh `s27_graph` (privileged) | 63.0% | 0.292 | 207 | 14 | 73.4% | 23.8% |
| `s39_ascentnav` | 55.0% | 0.284 | 191 | 16 | 70.5% | 0.0% |
| **`s41_stairs`** | **58.0%** | 0.285 | 184 | **12** | 66.7% | **27.3%** |

7 wins, 4 losses, net +3, **McNemar exact p = 0.55** — the headline is not
significant, and saying otherwise at n=100 would be reading noise. The
mechanism, however, separates cleanly:

* All 7 wins but one are cross-floor, all carry climb attempts, and the
  cross-floor cell moves 0.0% -> 27.3% (6-0 on that subset, p = 0.031), past the
  privileged navmesh arm's 23.8%.
* **All 4 losses are same-floor, and 3 of them recorded zero stair pixels and
  zero stair-segmentation frames** — the machinery provably never ran in them,
  so it cannot be what changed them. The remaining source of run-to-run
  variation is the hosted LLM ranker, which is sampled, not deterministic.
* Same-floor episodes that *did* paint stair pixels score 65.9% (n=41) against
  67.6% (n=37) for those that did not. Painting stairs into the navigable map
  costs nothing measurable.
* Only 2 of 78 same-floor episodes ever started a climb, so the "stairs distract
  a single-floor agent" failure mode did not materialise.

The claim this run supports is therefore the narrow one: **the stair machinery
recovers the cross-floor cell from zero without a measurable same-floor cost**,
and the 3-point headline movement is within noise.


### S42 — why `wcojb4TFT35:3` never moves: a reactive mover in a 1 m pocket

The debug video shows the agent seeing the staircase repeatedly and going
nowhere for 500 steps. The trace and four simulator probes localise it, and the
answer is none of the three things it looks like.

**Not the detector.** RedNet saw stairs on 84 steps, the up-stair map reached
942 px, and a published up-stair frontier existed on **496 of 499 steps**. The
agent entered climb state 8 times for 246 steps and never once reached the foot
of the stairs -- closest approach to the stair centroid **4.80 m**, against the
~0.2 m the footprint test needs.

**The body is stuck, not the decision.** Total path **2.95 m over 499 steps**,
inside a 0.4 x 0.5 m box, with **132 steps where a forward was commanded and
nothing happened** (no yaw change, no displacement, no tilt) against 351 turns
and 16 steps of actual motion.

**The episode is solvable and the pocket is escapable.** The start is navigable,
on island 1 of 7, with `distance_to_closest_obstacle` = **0.001 m** -- it is
touching geometry. The target is reachable: geodesic **7.89 m** (three of the
four goal instances are on other islands, one is not). Habitat's own
`ShortestPathFollower` from the same start reaches within 0.85 m of the goal in
**53 steps, 8.51 m of path, zero blocked forwards**, and leaves a 1 m ball
around the start in 13 steps. Probing the wedge pose directly: all 12 headings
move on the FIRST step (0.18-0.30 m), but 12 consecutive forwards (3.0 m
commanded) buy at most **0.96 m**, and 0.18 m in 8 of the 12 headings. The free
space is a pocket about a metre across with one narrow exit.

**Not contact-stop.** Re-running the episode with `allow_sliding=true` -- off
protocol, as a probe -- drops blocked forwards 132 -> 20 and raises path length
2.95 -> **14.63 m**, and the bounding box stays **0.5 x 0.5 m**. It slides back
and forth inside the same pocket. Disabling sliding is not what pins it.

**Not goal churn.** The other candidate was the recurrent state being wiped: the
driver resets the LSTM whenever the goal moves >0.1 m (ASCENT's rule,
`ascent_policy.py:849-853`), and this agent re-selects a frontier every step. It
does not happen -- **20 resets in 484 mover steps**, 21 distinct goals, the goal
held constant for tens of steps at a time.

**What it actually is.** A reactive policy in a local minimum. The commanded
goal sits a median **5.09 m** away through the pocket wall, and escaping needs
travel AWAY from the goal down a narrow exit. PointNav has no map and no
planner, so it presses toward the goal; the privileged follower escapes because
it plans on the navmesh.

**And nothing in the stack rescues it.** Three guards exist and all three are
blind to this:

* `ActionHistoryEscape` fires only when the last 30 actions are ALL turns or ALL
  forwards (`escape.py:52-58`). The real stream alternates, so it never fires.
  It also reads COMMANDED actions -- a forward that moved 0 m is indistinguishable
  from one that moved 0.25 m.
* The stair approach stall rule fires correctly, every 30 steps, 8 times -- and
  retires the STAIRCASE. The agent picks another stair frontier from the same
  pocket. `frontier_give_up` fired 5 times with the same effect.
* The obstacle map records the pocket, but nothing in `ascentnav` consults a map
  for motion. That is ASCENT's architecture, not an omission in the port.

The missing piece is a guard on REALISED DISPLACEMENT rather than on the action
stream: forward commanded, <1 cm moved, N times in a window -> a deliberate
escape (turn away, commit forward for k steps). OSG's `NavAgent` already carries
that machinery (`frontier_stick_rule: displacement`, the costmap controller's
`observe_progress`); `ascentnav` uses neither.

Footnote worth keeping: this episode never needed the stairs. The reachable bed
is 7.89 m away on the starting floor, and the agent spent 246 of 500 steps
trying to climb to a storey it did not need.

### S43 — the stair region was painted at max_depth, not at its own range

The purple stair region on the debug maps sat along the right bearing at the
wrong distance. S41 had blamed pitch routing; that was wrong. Synthetic geometry
-- a fronto-parallel wall at a known range with a known patch marked as stairs
-- pins it exactly:

| camera pose | centroid error before | after |
|---|---|---|
| level, centre patch | 2.00 m | 0.004 m |
| level, off-centre patch | 2.30 m | 0.004 m |
| translated + rotated 90 deg | 2.00 m | 0.004 m |
| pitched down 30 deg | 1.75 m | 0.004 m |

A 3.0 m staircase was painted at 5.0 m; a 2.0 m one, also at 5.0 m. The
displacement was radial and scaled as `max_depth / true_depth`.

**Cause** (`obstacle_map.py:519-540`):

```python
fusion_stair_mask = stair_mask & stair_map        # uint8 & bool -> uint8
stair_depth = np.full_like(depth, max_depth)
stair_depth[fusion_stair_mask] = scaled_depth[fusion_stair_mask]
```

`stair_mask` arrives as uint8 here -- ASCENT's comes from GroundingDINO as bool
-- and `uint8 & bool` promotes to uint8, which turns that assignment into
INTEGER ROW indexing rather than boolean masking: rows 0-1 are clobbered and
every stair pixel keeps `max_depth`. `np.where` inside `get_point_cloud` treats
uint8 as nonzero, so the pixel SELECTION was right the whole time and only the
range was wrong. That is why it read as a calibration shift rather than as
garbage, and why the agent still climbed sometimes: the bearing was correct, so
walking at it eventually arrived.

The down-stair map was never affected -- it comes from the pitch-independent
inverted-depth path -- which is exactly the 0.14 m vs 1.54 m accuracy gap S41
measured and misattributed.

**Measured on the 10-episode viz split** (`outputs/s48_viz_fixed` against
`outputs/s44_viz`, same episodes, same config apart from the fix):

| | pre-fix | post-fix |
|---|---|---|
| SR | 6/10 | 6/10 |
| climbs attempted | 16 | **11** |
| climbs completed | 1 | **4** |
| floor switches | 1 | **4** |
| conversion | 6% | **36%** |

And on `mL8ThkuaVTM:2`, where the painted centroid can be compared against where
the agent actually changed height:

| | pre-fix | post-fix |
|---|---|---|
| up-stair centroid error | 1.54 m | 0.63 m |
| up-stair centroid **wander** | 7.9 x 4.4 m | **0.3 x 0.2 m** |
| episode length | 418 steps | **196 steps** (SPL 0.31 -> 0.58) |

The wander is the number that matters: a staircase does not move, and before the
fix its estimate swept an 8 x 4 m box as the agent walked. Cross-floor episodes
finish roughly twice as fast (`p53SfW6mjZe:0` 209 -> 110 steps, SPL 0.28 ->
0.55); the four same-floor episodes are bit-identical, as they should be.

Three regression tests: painted at true range, uint8 and bool masks agreeing,
and a 2 m and a 3.5 m wall landing in different places -- the specific signature
of the old bug.

**This invalidates the stair numbers in S41 and the in-flight full-split run**
(`outputs/s47_full_v1`), both of which ran with the defect.

### S44 — down-stairs: the detector was fine, the direction preference was not

A strict descent split (`configs/eval/downstairs5.yaml` -- every view-point of
every goal instance more than a metre BELOW the start, so the episode cannot be
solved by climbing) exposed a clean failure:

**Before: 148 climb steps across the split, every one an ascent.** Detection was
not the problem -- the down-stair map was often LARGER than the up map (1257 px
vs 690; 1219 vs 537) and a down frontier was published on most steps.

Three causes, all in the port rather than in ASCENT:

1. **The camera never tilted** (pitch 0.0 on every step of all five episodes).
   ASCENT's direction disambiguation IS the pitch: `update_map` routes fused
   stair pixels by `agent_pitch_angle >= 0`, so at level pitch the fused writer
   and the drop-off writer both fire on the same pixels and one staircase is
   filed as both up and down.
2. **`_maybe_start_climb` preferred up unconditionally**, returning on the first
   viable direction. With the up map essentially never empty at level pitch,
   direction 2 was unreachable in practice.
3. **`_look_for_downstair` was not ported.** The map raises
   `_look_for_downstair_flag` when it holds down-stair pixels that never grew
   into a frontier (`obstacle_map.py:737-739`) and nothing read it.

**Implemented:** the probe (`ascent_policy.py:623-658`), camera levelling when
not on stairs (`:556-563` -- without it a single probe leaves the camera down
and relabels every later staircase), and a direction tie-break using ASCENT's
own image-space discriminator, `check_stairs_in_upper_50_percent`
(`ascent/utils.py:163`): treads you must climb project into the top half of the
frame, treads you must descend do not. Up-first is kept whenever only one
direction is available.

**Measured** (`outputs/s50_down_strict` -> `outputs/s51_down_fixed`, same 5
episodes, same config apart from these three changes):

| | before | after |
|---|---|---|
| SR | 1/5 | **2/5** |
| UP climb steps | 148 | 73 |
| DOWN climb steps | **0** | **67** |
| probe steps | 0 | 6 |
| steps with the camera tilted | 0 | 23 |

The decisive episode is `qyAac8rV8Zk:62`, whose goal is 1.52 m below the start:

| | before | after |
|---|---|---|
| outcome | fail, 500 steps | **success, 102 steps** |
| net height change | **+1.50 m** | **-1.70 m** |
| distance to goal | 11.87 m | 0.03 m |
| climb steps | 75 up, 0 down | 0 up, 18 down |

It was climbing the wrong way, completing the ascent, and timing out on the
wrong storey. Now it descends.

**Not uniformly better, and worth recording as such.** `q3zU7Yy5E5s:96` gained
49 down-climb steps and got WORSE on time (392 -> 500 steps, a timeout), though
it ends closer (dtg 12.98 -> 11.00). `XB4GS9ShBRE:43` improved slightly
(249 -> 220 steps, dtg 11.44 -> 11.32) and still fails. `6s7QHgap2fW:50` is
untouched by any of this: it stops after 35 steps on a false-positive commit
16.4 m from the goal, which is the S13 commit-gate failure, not a stair one.

n=5 measures nothing on its own; the mechanism counters (148/0 -> 73/67) are
what this run establishes, not the SR.

### S45 — the down-stair region was placed by what is visible THROUGH the hole

S43 fixed the up-stair range. The down-stair region was still displaced, and for
a completely different reason: it is not written by the fused detector at all but
by the vendored inverted-depth trick (`obstacle_map.py:561-573`), which mirrors
depth about `(max+min)/2`, keeps rays whose true range exceeds 3.5 m, and paints
whatever lands below the floor plane **at the mirrored range**.

Synthetic geometry -- a floor that stops at a known distance, with the lower
floor visible 3.8 m away through the hole:

| true lip | painted at (old) |
|---|---|
| 1.0 m | 1.70 m |
| 1.5 m | 1.70 m |
| 2.0 m | nothing |
| 2.5 m | nothing |
| 3.0 m | nothing |

1.70 m is `5.5 - 3.8`: the position was set by the range of the surface seen
THROUGH the hole, not by where the hole is. Two different lips landed in the same
cell, and anything at 2 m or beyond was invisible -- the below-ground test can
only fire in a narrow band of ray angles (true range 3.5-4.1 m, bottom rows of
the frame). ASCENT's own comment concedes the trick is weak for short flights.

**Replaced with the geometric test.** Every pixel is a ray with a known
direction; a downward ray must meet the floor plane at a known forward distance;
if the measured depth runs half a metre past that, the floor is missing along
that ray, and the lip is where it should have been. Measured on the same
geometry, the marked region now begins at the lip:

| true lip | marked from | on an unbroken floor | tilted 30 deg down |
|---|---|---|---|
| 1.5 m | 1.50 m | nothing | — |
| 2.0 m | 2.00 m | nothing | 2.00 m |
| 2.5 m | 2.50 m | nothing | — |
| 3.0 m | 3.00 m | nothing | — |

One guard is load-bearing: a return at the sensor's far clip is "nothing came
back", not "the floor is missing". Without it every near-horizon ray in a room
wider than `max_depth` reads as a drop-off -- the first cut of this change grew
the down-stair map from ~1.2k cells to ~21k, i.e. the whole room, and the agent
chased it (`qyAac8rV8Zk:62` regressed from success to a 6 m miss). With the
guard the map stays at hundreds to a few thousand cells.

**Measured on the strict descent split** (5 episodes; n=5 measures nothing on
its own, the mechanism counters do):

| | SR | UP / DOWN climb steps | total steps |
|---|---|---|---|
| S44 baseline | 1/5 | 148 / 0 | 1249 |
| + direction preference (S44) | 2/5 | 73 / 67 | 930 |
| **+ lip fix** | **2/5** | 73 / 53 | **682** |

SR does not move beyond what the direction preference already bought, but the
split finishes in **45% fewer steps** than the baseline, and the two episodes
that were merely slow get much faster: `XB4GS9ShBRE:43` 249 -> 123 steps,
`q3zU7Yy5E5s:96` 392 -> 338. Against S44's own numbers `q3zU7Yy5E5s:96` ends
farther out (11.00 -> 15.98 m) while taking fewer steps, so this is not a
uniform win either.

Known limitation, unchanged: a stairwell whose visible return is beyond
`max_depth` still cannot be detected by this path.

### S46 — the drop-off marking filled the whole void, not the edge

S45 moved the down-stair marking from a mirrored depth to the geometric
missing-floor test. It put the near edge in exactly the right place, and then
kept going: every ray past the lip also misses the floor, so the marked region
was the entire VISIBLE VOID -- on synthetic geometry a band from the lip out to
3.30 m, and in episodes a blob averaging 2296 cells (5.7 m^2) and peaking at
8808 (22 m^2), spilling across the lower floor and out through whatever the
stairwell overlooks.

That matters beyond tidiness: the stair frontier is the centroid of the largest
component, so a void-shaped region aims the agent at the middle of the hole
rather than at the lip, and `robot_on_stairs` -- a footprint test against the
same map -- only fires once the agent is over the drop.

**Fix:** mark one point per image column, the NEAREST missing-floor sample along
that bearing, and only for columns with a real run of missing pixels (8) rather
than a single noisy one. That is the lip, and it cannot spread into the void.

| true lip | marked band (before) | marked band (after) |
|---|---|---|
| 1.5 m | 1.50 .. 3.30 m | **1.50 .. 1.50 m** |
| 2.0 m | 2.00 .. 3.30 m | **2.00 .. 2.00 m** |
| 2.5 m | 2.50 .. 3.30 m | **2.50 .. 2.50 m** |
| 3.0 m | 3.00 .. 3.30 m | **3.00 .. 3.00 m** |

Still nothing on an unbroken floor, still pitch-invariant, and the per-frame
cost drops from thousands of points to at most one per column.

**Measured on the 10-episode strict descent split:**

| | SR | UP / DOWN climb steps | mean down-stair cells |
|---|---|---|---|
| whole void (S45) | 3/10 | 0 / 194 | 2296 |
| **lip only** | **4/10** | 0 / **309** | **1273** |

`q3zU7Yy5E5s:9` flips to a success (dtg 12.82 -> 0.06 m), and `XB4GS9ShBRE:13`
gets from 11.40 m to **4.90 m** of its goal while still failing. Total steps rise
1377 -> 1768, which is the right direction: episodes that used to stop early on
the starting floor now go down.

The residual cells are not bleed. A one-pixel lip curve is thickened by
ASCENT's own `MORPH_CLOSE` with the agent-radius kernel (`obstacle_map.py:673`),
which is what makes the lip wide enough to stand on, and the curve accumulates
as the agent moves and sees more of the edge.

### S47 — the same-floor gap is one failure, and it is not navigation

`s47_full_v1` (2000 episodes, full v1 val, sensor-only, pre-stair-fixes) scored
**52.15% SR / 0.270 SPL**, split **62.2% same-floor (n=1589)** and 13.4%
cross-floor (n=411). ASCENT reports **72.6% same-floor** on the same split, so
the same-floor gap is 10.4 points -- about 165 episodes.

**Every one of those 165 could come from a single failure mode.** Taxonomy of
the 601 same-floor failures:

| | n | share |
|---|---|---|
| committed, stopped, and was wrong | **509** | **85%** |
| ran out of steps | 92 | 15% |
| ended some other way | 0 | 0% |

Of the 509 stops, only **40** were near misses within 1 m. **373 were more than
3 m from any instance of the category** -- median **6.51 m**, p75 9.76 m, p90
13.06 m. HM3D scores success against a view-point of ANY instance, so ending
6.5 m out does not mean a bad approach; it means the thing the agent walked to
was not an instance at all.

**It is a commit failure, not a navigation failure**, and three measurements say
so:

* Far-commits happen EARLY. First `approach` at median step **22**, against 45
  for successes; 47% commit inside the first 20 steps against 27%.
* They are cheap in time and fatal anyway. Median 11 steps walking to the false
  positive, median episode length **69 steps of a 500-step budget**. The agent
  is not running out of anything -- it stops, and in ObjectNav STOP is
  irreversible.
* The VLM verifier does not separate them. It rejects at least once in **46% of
  successes** and **37% of far-commits** -- pointing the wrong way, with
  ~2 calls per episode either way. As a gate on commit correctness it is noise.

**Arithmetic.** Far-commits are 23.5% of all same-floor episodes. Converting 44%
of them into episodes that keep exploring and eventually succeed closes the
entire 10.4-point gap; converting half would put same-floor at 73.9%, just past
ASCENT's 72.6%.

**What is missing is the commit gate.** OSG's `NavAgent` requires a track to
clear `verification.min_score` 0.70, `min_obs` 2, `min_bbox_px` 1200 and
`min_evidence` 0.5 before it will walk to it (`object_layer.py:367`, called from
`nav_agent.py:2145-2151`). All four are SET in this preset and `ascentnav` reads
none of them: it writes any detection above the detector's own `conf: 0.3` into
the object cloud and treats a cloud as a goal. S13 measured the `min_score`
raise alone at net +3 episodes on 100 navmesh episodes, for exactly this failure
-- there, 30 far-commit failures reached what they aimed at (median 0.43 m)
while sitting a median 7.48 m from any real goal.

This run cannot say WHICH threshold would have blocked which commit:
`cand_best_score` and `cand_n_obs` come from OSG's object layer, and
`ascentnav`'s view of it returns no tracks, so both are null for all 2000
episodes. Instrumenting the commit (score, observation count, bbox at commit)
is the prerequisite for calibrating the gate rather than guessing it.

### S48 — the commit gate: prediction met, SR null, and the reason is instructive

S47 predicted the same-floor gap was false-positive commits. The gate
(`agent.commit_gate`: detection score >= 0.70, bbox >= 1200 px, and 2 accepted
sightings before a cloud counts as a goal) was pre-registered with a falsifiable
prediction: *far-commits fall by at least a third, and SR rises; if far-commits
fall while SR does not, the blocked episodes were failing for another reason and
the gate is a null.*

**100 paired episodes on `scenes20_ep0to4`, one fingerprint field apart:**

| | SR | SPL | steps | timeouts | same | cross | far-commits |
|---|---|---|---|---|---|---|---|
| `s56_fixed100` (control) | 56.0% | 0.285 | 199 | 17 | 65.4% | 22.7% | 22 |
| `s57_gate100` (gate) | 54.0% | 0.258 | 270 | **34** | 65.4% | 13.6% | **8** |

Far-commits fell **22 -> 8 (-64%)**, well past the pre-registered third. SR did
not follow: 9 wins, 11 losses, **net -2, McNemar p = 0.82**. By the
pre-registration this is a **null**, and the mechanism says why.

**Following the 22 episodes that far-committed without the gate:**

| with the gate they | n |
|---|---|
| became a success | **7** |
| ran out of steps | **11** |
| far-committed anyway | 4 |

The gate does exactly what it was built to do -- 7 of the 22 recover -- and then
the same conservatism costs 11 episodes that used to succeed (5 of them
timeouts). Mean steps 199 -> 270 and timeouts 17 -> 34: withholding a goal until
a second sighting at 0.70 leaves the agent exploring, and the budget runs out.

**The lesson is not "commits are fine".** Blocking a bad commit does not produce
a good one: half the blocked episodes simply never found the target. The failure
S47 measured is real, but it is not one bad decision away from a success -- the
agent that commits at step 22 to the wrong sofa mostly has not seen the right
one either.

**Calibration, not abandonment, is the next move.** These thresholds are OSG's,
tuned for `NavAgent`, whose track layer accumulates evidence across frames with
a different detector pipeline. Ported wholesale onto an agent that has no track
layer they are too strict. The obvious cheaper variants -- `min_obs` alone with
no score raise, or `min_score` at 0.5 -- are one flag each, and the counters
needed to choose between them (how many blocks were score failures vs bbox
failures) are not yet split apart.

#### S48b — the stair fixes at n=100: mechanism yes, SR no

The same control run is also the first n=100 measurement of S43-S46 (stair
projection, direction preference, `_look_for_downstair`, lip marking), against
`s41_stairs` which predates all four:

| | SR | climb attempts | completed | conversion | floor switches |
|---|---|---|---|---|---|
| `s41_stairs` | 58.0% | 37 | 4 | 11% | 4 |
| `s56_fixed100` | 56.0% | 53 | **16** | **30%** | **16** |

Climb conversion nearly triples and floor switches quadruple -- the fixes do
what the synthetic geometry said they would. SR moves -2 (6 wins, 8 losses,
p = 0.79): a null. The cross-floor cell is 22 episodes here, so it cannot
resolve a change of this size; the full-split re-run is what would.

### S49 — the agent walks past the target, and the detector is why

S48 left a puzzle: blocking bad commits recovered 7 episodes and cost 11. S49
asks what those episodes were doing instead, using the 2000-episode run plus the
dataset's own goal view-points.

**Most same-floor failures reach the goal region and leave.** Taking each
episode's logged explore positions and measuring the closest approach to any
goal view-point:

| | successes | failures |
|---|---|---|
| p25 | 0.04 m | **0.18 m** |
| p50 | 0.61 m | 1.86 m |
| never within 3 m | 12% | 37% |

A quarter of same-floor FAILURES pass within 18 cm of a view-point of the
target. And of the 318 failures whose explore track came within 3 m, **265
passed the goal BEFORE committing elsewhere**, a median 27 steps before.

**Why: the detector fires on about one in nine of the frames where it should.**
16 of those episodes were re-run with per-step pose and detection logging, and
scored against the view-points offline:

| criterion | steps with a goal view-point in frame | detector fired | rate | rate when NOT in frame |
|---|---|---|---|---|
| < 5 m, full FOV | 2694 | 284 | **10.5%** | 1.2% |
| < 3 m, central 40 deg | 2013 | 237 | **11.8%** | 3.3% |

The control is what makes this readable: 3-9x more firing when a view-point is
in frame than when none is, so the proxy carries real signal -- and the absolute
rate is ~12%. A view-point is a standing position rather than the object itself,
so occlusion and objects behind the agent mean 12% is a LOWER bound on true
recall; it is not a measurement of YOLOE's accuracy on a clean crop. It is a
measurement of how often this pipeline notices the target while walking past it.

**And that calibrates S48's null exactly.** Scoring those detections by whether
a view-point was in frame:

| `min_score` | keeps of likely-TRUE | keeps of likely-FALSE |
|---|---|---|
| 0.50 | 69.2% | 45.0% |
| **0.60** | **49.4%** | **13.3%** |
| 0.70 (S48) | **32.5%** | 5.0% |

At 0.70 the gate throws away two thirds of the real sightings. Combined with
~12% per-step recall and `min_obs` = 2, a correct commit needs roughly
0.12 x 0.325 = 4% per step, twice -- which is why 11 previously-successful
episodes turned into timeouts. The gate was not wrong in kind, it was set for a
detector with better recall than this one.

**Consequence for the gap.** S15 measured YOLOE-11s vs 11l as worth nothing and
concluded "the detector is worth zero". That conclusion was about FALSE
positives -- far-commits went 30 -> 32 across a 2.5x larger model -- and it does
not cover recall on the true object, which nothing had measured until now. The
two findings are compatible: a bigger YOLOE does not stop the agent walking to
the wrong sofa, and the reason the agent needs a sofa at all is that it did not
see the right one.

Next arm: `+experiment=ascentnav_gate60`, the same gate at the knee of the
curve, pre-registered in that file.

### S50 — the agent is facing the wrong way, and looking around does not pay

S49 measured ~12% detection recall while traversing. S50 asks whether that is
the detector's fault, and the answer is no.

**Three detector configurations, same measurement** (steps with a goal
view-point within 3 m and in the central 40 degrees, on the 16-episode
diagnostic split):

| detector | recall | control (nothing in frame) |
|---|---|---|
| YOLOE-11s @ 512, 42-class vocabulary | **11.8%** | 3.3% |
| YOLOE-11l @ 640, 42-class vocabulary | **8.4%** | 1.5% |
| YOLOE-11s @ 512, target-only prompt | **11.8%** | 4.0% |

The larger model is WORSE -- it fires less on everything -- and prompting it with
the target alone changes nothing. That closes the detector as a lever, and it
does so for the cost of two 16-episode diagnostics rather than two 100-episode
A/Bs.

**The detector is fine when it is actually looking.** In the last 15 steps of
successful episodes, with the object close, centred and being approached, recall
is **66%** (59 of 89 in-frame steps). The geometry proxy is therefore sound --
66% against 14% is not a measurement artefact -- and the low traverse number is
about FRAMING.

**Decomposing the 2809 steps spent within 3 m of the target object:**

| | share |
|---|---|
| object OUTSIDE the 79-degree FOV -- facing the wrong way | **62%** |
| in frame, not detected | 33% |
| detected | 5% |

A 79-degree camera bolted to the direction of travel sees a fifth of a room, and
after the opening scan nothing makes the agent look around again.

**So: scan on arriving at a frontier** (`agent.scan_on_arrival: 12`, at most
once per 1.5 m cell -- ASCENT's own `_initialize` scan applied at every vantage
point instead of only on entering a floor). Pre-registered: SR above the
control's 56.0%.

| | SR | SPL | steps | timeouts | same | cross |
|---|---|---|---|---|---|---|
| `s56_fixed100` control | 56.0% | 0.285 | 199 | 17 | 65.4% | 22.7% |
| `s63_scan100` | 54.0% | 0.284 | 190 | **11** | 62.8% | 22.7% |

4 wins, 6 losses, net -2, p = 0.754. **Prediction NOT met -- a null.** 288 scans
consumed 3297 steps, **17% of every step taken**, and bought nothing:
`steps_to_first_candidate` got WORSE (median 54 -> 63). The looking is paid for
out of forward progress at par, exactly the failure mode the preset named.

#### Four nulls in a row, and what that actually means

| change | mechanism moved? | SR effect | p |
|---|---|---|---|
| stair fixes S43-S46 | yes: climb conversion 11% -> 30% | -2 | 0.79 |
| commit gate @ 0.70 | yes: far-commits 22 -> 8 | -2 | 0.82 |
| commit gate @ 0.60 | yes: far-commits 22 -> 15 | +0 | 1.00 |
| scan on arrival | yes: 288 scans, 17% of steps | -2 | 0.75 |

Every mechanism does its job locally and none moves SR. Before reading that as
"none of this matters", look at what these A/Bs can resolve. They produce 10-20
discordant pairs; with 10 discordant pairs the smallest detectable effect at
p < 0.05 is **9 wins against 1 loss, a net of +8 episodes**. An intervention
worth a genuine +2 or +3 is INVISIBLE at n=100 -- it cannot be distinguished
from these results no matter how many times it is run.

The gap to ASCENT is 11 points at n=2000. The mechanisms above plausibly carry
1-3 points each. **The 100-episode split is the wrong instrument for them**, and
running a fifth arm on it would be spending an hour to learn nothing again. The
next measurement that can actually settle any of this is the full 2000-episode
split against `outputs/s47_full_v1` (52.15%), which has 20x the paired power.

### S26 — ASCENT's dense approach re-check

S23/S25 closed the mover, the aim point and the commit gate as explanations for
the same-floor deficit. What was left was the one ASCENT mechanism OSG had never
run: a **continuous** re-check of the target during the approach.

**What ASCENT actually does.** Every step it scores the live frame against the
target prompt and latches a flag once the score clears 0.15
(`map_controller.py:770-776`). The score is not an extra model call — it is the
same cosine that drives the value map (`map_controller.py:562`). At the stop
moment, if the flag never latched it wipes the object clouds, marks the region
disabled and returns to exploring (`ascent_policy.py:915-922`).

OSG already had a terminal re-check (`verification.terminal`) but it had never
been switched on, and it is a *single* expensive VLM call on one frame rather
than a free dense signal over the whole approach. This stage ports the dense
version, reading OSG's own value-map score the same way ASCENT reads its own.

Three findings came out of the smoke tests, before any SR was measured.

**1. ASCENT's 0.15 is not transferable.** It is a BLIP-2 ITM cosine; OSG's value
map is CLIP, whose cosines occupy a different range. Three true positives on a
4-episode smoke scored 0.237, 0.241, 0.249 — every one of them far above 0.15,
so copying the constant would have produced a gate that accepts everything and
an A/B that correctly reported "no change". The threshold has to be calibrated
against OSG's own distribution (`scripts/calibrate_approach_recheck.py`).

**2. Gating only the terminal rule is not the same mechanism.** ASCENT's
approach has exactly one STOP (`ascent_policy.py:913`) and it is gated:
too far returns a pointnav action, still closing returns a forced
`MOVE_FORWARD`, and the 100-step timeout returns to exploring. Nothing can stop
on an unvouched target. OSG's `_do_approach` had four STOP exits and only the
terminal one was gated. With the threshold forced to 0.99 — reject everything —
all 12 rejections over 4 episodes were absorbed by `path_consumed`, which
stopped the agent 0.02–0.05 m from where the gated exit would have. SR stayed
4/4. **The gate rejected every target and the agent stopped on them anyway.**

After gating all four exits the same forced-reject smoke gives SR 0/4 with
`approach_stop_reason=None` — the agent never calls STOP, including one episode
where it stands 0.04 m from the goal. That is the correct control: the
mechanism is now decisive in both directions, which is what makes the A/B
readable at all.

Pinned by `test_every_approach_exit_is_gated`, which fails if a STOP exit is
ever added without a gate.

**3. Two deliberate deviations, both documented in code.**

| | ASCENT | OSG | why |
|---|---|---|---|
| latch scope | per **episode** (`map_controller.py:177`) | per **approach** | ASCENT's latch means one vouched target vouches for every later one in the episode; per-approach matches the intent |
| rejection scope | spatial — `_disabled_object_map` filters every future cloud point in those cells (`object_point_cloud_map.py:102`) | per-track `blacklist(id)` | OSG has no spatial disable; a rejected object can be re-detected as a fresh track. Strictly weaker; only worth building if the threshold calibrates |

**Measurement.** `verification.approach_recheck_thresh=0.0` records
`approach_recheck_max` at every stop while rejecting nothing, so the calibration
run is behaviourally identical to the baseline and doubles as the paired
control.

**Result: the signal is not there.** `s26_calib`, 100 episodes,
`approach_recheck=true thresh=0.0` — the gate ran on 82 episodes, passed 82,
rejected 0, and left SR at 57.0% against the baseline's 59.0% (2 points, inside
the LLM-nondeterminism noise for an identical code path; `recheck_no_obs=0`
confirms the score ran on every gated stop).

| outcome | n | p25 | median | p75 |
|---|---|---|---|---|
| successes | 56 | 0.232 | **0.241** | 0.254 |
| far-commit failures | 20 | 0.232 | **0.249** | 0.257 |

The failures score *higher* than the successes. **AUC = 0.479** — over all
(success, far-commit) pairs the success scores higher less than half the time,
so there is no ordering to cut and no threshold can help. The best cell in the
sweep rejects 1 far-commit at zero cost to successes: net +1 on 100 episodes,
inside the ±2 noise band.

**Why, mechanically.** The value-map score is the whole frame against
"Seems like there is a {target} ahead." Standing 0.4 m from anything in a
bedroom, the frame is dominated by room context, and CLIP scores "there is a bed
ahead" high whether or not the object in front of the agent is a bed. It is a
**scene-level** score. OSG's far-commits are not "approached something that
looks nothing like a bed" — per the S23 decomposition they are 5 wrong-side-of-a-wall,
7 genuine false positives and 2 ambiguous, i.e. mostly *"approached something
that really does look like a bed, in a bedroom"*. A scene-level score cannot
separate those in principle.

This also reframes ASCENT's 0.15: on the BLIP-2 ITM scale it is a **low bar**,
a safety net against gross mismatch rather than a discriminator. OSG's commits
all sit at 0.20–0.29, so the equivalent bar rejects nothing.

**What does separate.** Asking the same AUC question of every recorded
per-episode signal, on the same 57 successes and 20 far-commits:

| signal | AUC |
|---|---|
| `cand_best_score` (YOLOE detection score) | **0.700** |
| `cand_n_obs` | 0.577 |
| `verify_calls` | 0.472 |
| `approach_recheck_max` (CLIP whole-image) | 0.471 |
| `steps_to_first_candidate` | 0.356 (inverted: successes commit *earlier*) |

The discriminative signal is the detector score — which OSG already exploits;
raising `verification.min_score` 0.45 → 0.70 on exactly that basis is S16.

**Scope of the claim.** This says the ported mechanism carries no information
*with CLIP as the scorer*. It does not say ASCENT's gate is useless in ASCENT:
substituting CLIP for BLIP-2 ITM is one of the four forced deviations recorded
in `configs/experiment/ascent_aligned.yaml`, and this is the first stage where
that substitution is load-bearing. The obvious follow-up is an **instance-level**
score — the detection crop rather than the whole frame — which is what would
test the mechanism rather than the room context.

Verdict: mechanism correct and verified decisive in both directions, signal
absent. `approach_recheck` stays **off** by default. No armed A/B was run: the
calibration bounds its best case at +1 episode. The scorer swap is the open
question — see S29 below.

### S27 — frontier descriptions from the frame that revealed them

**All evaluations from here run on the remote GPU box**, not locally:
`ssh -p 2201 seanchen@140.114.58.148`, checkout at
`~/Warehouse/Object_scene_graph`, same container and image. Verified equivalent
before any measurement — `eval=scenes20_ep0to4 num_episodes=2` reproduces the
local run episode for episode (spl 0.725/78 steps, 0.626/48 steps).

**The gap this closes.** Both systems describe each frontier to a *text* LLM;
neither feeds a frontier image to a model (ASCENT's image-to-LLM call is
commented out at `model_api/qwen25_out.py:105`, and its per-frontier RGB is used
only for SSIM dedup and visualisation). The difference is where the words come
from:

| | source of "a bedroom containing objects: bed, nightstand" |
|---|---|
| ASCENT | RAM++ and Places365 run on **the frame that first revealed the frontier** (`map_controller.py:800-830` → `llm_planner.py:418-419, 445`) |
| OSG (before) | a spatial query against the accumulated scene graph around the frontier centroid (`ascent_ranker.py:58-85`) |

A frontier is a frontier *because what lies beyond it is unknown*, so "objects
already mapped near this point" tends to describe the room the agent is standing
in rather than the opening it is being asked to choose.

`exploration.frontier_desc: graph|frame` selects the source; `frame` binds each
frontier to a keyframe and reads that frame's Places365 room plus that frame's
YOLOE labels. Per the plan, YOLOE stands in for RAM++ for now so that this stage
changes exactly one thing; the tagger is a plain list of strings, so RAM++ drops
in without touching the frontier logic.

**Two things the smoke test caught, both of which would have made the A/B
meaningless.**

1. **Binding to "the latest keyframe" describes wherever the agent was facing.**
   ASCENT cannot have this bug: it stores the RGB inside the function that
   detects the frontier from that very frame (`obstacle_map.py:421-431`). OSG
   runs keyframes on a movement threshold and frontier extraction on a step
   interval, so they routinely differ. First smoke: three different frontiers in
   one ranking call received the identical string *"a garage containing objects:
   bench, chair, lamp, sofa"* — nothing for the ranker to choose between. Fixed
   by binding to the most recent frame that actually had the frontier **in
   view** (FOV + range test); after the fix the same call produced garage /
   bedroom / office / hall.

2. **Binding is not the same as changing the prompt.** If Places365 and the
   scene graph agree, the LLM reads an identical prompt and the A/B measures
   nothing while looking like a null result. `desc_frame`/`desc_differs` now
   count it: on the 4-episode smoke, **24 of 24** descriptions differed.

Faithfulness checks: OSG's `classify` already takes the top-k Places365 classes
and maps them (`room_classifier.py:110-115`), the same shape as ASCENT's
`extract_room_categories` — including falling back to the top-1 raw class when
nothing maps, which is why implausible labels like "garage" for a sofa-and-lamp
room appear in both.

Known gap, not ported: ASCENT skips a frontier whose image is SSIM > 0.75
similar to one already listed (`llm_planner.py:197-206`). OSG has no such dedup
(`ssim_thresh` appears only in a stale test exemption list and does not exist in
`config.py`).

**The baseline arm produced a finding of its own, and it is bigger than this
stage.** `s27_graph` is the same algorithm as `s16_minhits1` and `s26_calib` --
`approach_recheck` defaults off, `frontier_desc` defaults to `graph`, the new
counters are inert unless the store exists -- so the three differ only in LLM
nondeterminism:

| run | overall | single-floor |
|---|---|---|
| `s16_minhits1` | 59.0% | 68.4% |
| `s26_calib` | 57.0% | 65.8% |
| `s27_graph` | **63.0%** | **73.4%** |

**A 6-point spread on 100 episodes with the algorithm held fixed.** That is far
larger than the ±2 this log has been assuming from the dev50 paired work, and it
has a direct consequence: the S15/S23 framing of "OSG same-floor 68.4% against
ASCENT's 80.8%, so 89% of the gap is same-floor" rested on a single draw. The
same configuration measured 73.4% here, which puts the same-floor deficit at
−7.4 rather than −12.4.

Nothing in the stage log that rests on a **paired** `net` is affected -- pairing
is exactly the defence against this. What is affected is every sentence that
compares a bare SR against ASCENT's published number. Those should be read as
±3 at best, and the attribution of the gap between same-floor and cross-floor is
much softer than S15 claimed.

**Result: net −4, a regression.** 100 paired episodes, `s27_graph` vs
`s27_frame`, fingerprint diff exactly `frontier_desc: 'graph' -> 'frame'`.

| | baseline (graph) | treatment (frame) |
|---|---|---|
| SR | 63.0% | 59.0% |
| single-floor | 58/79 | 54/79 — **net −4** |
| cross-floor | 5/21 | 5/21 — net 0 |
| mean steps | 207 | 237 |
| `rank_calls` / `rank_overrides` | 454 / 254 | 521 / 304 |

The mechanism was at full strength, so the number is attributable: **1539 of
1545** descriptions read differently from the graph version, over 1680 bound
frontiers.

**How it fails.** Four of the seven lost episodes ran the full 500 steps and
ended 3.8–14.4 m from the goal; in the baseline the same episodes finished in
214–298 steps at 0.03–0.08 m. The agent was told to go the wrong way and never
recovered. Mean steps +15% and the LLM overriding the value ranking more often
(56.0% → 58.3%) are the same story.

**Why ASCENT's source is worse *here*.** My first explanation for this was
wrong and is corrected below; the original claim was that OSG aggregates
Places365 over many views with position-keyed voting, so replacing an aggregate
with one noisy sample threw that away. A 10-episode probe dumping both
descriptions side by side (98 pairs, `OSG_DEBUG_DESC=1`) says otherwise:

| | |
|---|---|
| room differs | 88.8% |
| objects differ | 95.9% (mean Jaccard 0.36) |
| identical | 0% |
| **graph says "unknown room"** | **66%** |
| frame says "unknown room" | 0% |

Two thirds of the time the scene graph has **no room label at all**. So the
change is not "good aggregate → noisy sample"; it is overwhelmingly
**"honest *I don't know* → a confident guess"**, and the guess is often wrong
(Places365 calling a bench/chair/lamp/sofa room "a garage"). The seventeen most
common disagreements are all `<some room> -> unknown room`.

That reframes the negative result. The ranker was not being deprived of a better
aggregate; it was being given assertions where it previously got an explicit
absence, and acting on them. An LLM told "a garage" will steer away from a
bedroom target; told "unknown room" it falls back on the object list and the
value ranking.

So this is not "the port failed" -- the port is faithful and verified active --
but neither is it "OSG's aggregate beats ASCENT's frame". It is that a wrong
label is worse than no label, and OSG's room segmentation abstains far more
often than ASCENT's classifier does.

Verdict: `frontier_desc` stays **`graph`** (the default). The `frame` path and
`FrontierSemantics` are kept -- they are the plumbing the RAM++ follow-up
would need, and the counters make any future version auditable.

**Caveat on the threshold.** The repo's `net <= -3 -> revert` rule was
calibrated on dev50. Two same-config 100-episode runs on the same machine
(`s16_minhits1` vs `s26_calib`) flip only **2** episodes; this A/B flipped
**10**, asymmetrically. That churn ratio, not the net alone, is what makes −4
readable here.

**Local and remote are not interchangeable.** `s26_calib` (local) against
`s27_graph` (remote) is the *same algorithm* and pairs at net **+6, gained 6
lost 0** -- a systematic offset, not noise, most likely GPU-dependent numerics
in the detector. Any A/B must have both arms on the same machine. Both S28 arms
did.

### S28 — is RAM++ worth adding? No, and the reason is not its quality

Before spending a 5.6 GB download and an integration on RAM++, the question
worth asking was whether the channel RAM++ replaces is the one that hurt.
S27 moved the room and object halves at the same time, so it could not say.

`exploration.frontier_desc=frame_objects` takes **only** the object list from
the frame and leaves the room to the graph. Against the same `s27_graph`
baseline, 100 paired episodes:

| arm | channels moved | net |
|---|---|---|
| S27 `frame` | room + objects | −4 |
| S28 `frame_objects` | **objects only** | **−5** (63.0% → 58.0%, gained 3 lost 8) |

The object channel alone accounts for the whole regression.

**Why this settles RAM++ without running it.** Both arms use the *same
detector*. `graph` objects are YOLOE detections accumulated into object-layer
tracks; `frame_objects` objects are YOLOE detections from one frame. The tagger
is identical and only the aggregation differs, so the −5 isolates
**aggregation**, not **tag quality** — and swapping YOLOE for RAM++ changes tag
quality while leaving aggregation exactly as it is.

For RAM++ to rescue this, single-frame RAM++ would have to beat accumulated
YOLOE. Possible, but that is a much weaker bet than "RAM++ is a better tagger",
and it is not what the S27 result suggested.

Verdict: **do not port RAM++.** The frame-sourced description is not held back
by its tagger; it is held back by being one frame.

What the two stages jointly establish is narrower and more useful than "ASCENT's
mechanism does not transfer": for frontier description, **accumulating evidence
across views beats any single view, whichever model labels it**. OSG already
accumulates; ASCENT cannot, having no persistent object or room representation.
That is a place where OSG's architecture is ahead, and the S27/S28 pair is the
evidence.

Open, and more promising than either: **why does adding information make the
ranker worse?** `rank_overrides` rose 56.0% → 58.3% in S27, and each override
discards a geometric ranking that already integrates the value map, information
gain and path cost. The suspect is the LLM over-reacting to descriptions rather
than the descriptions themselves.

## Planned — not yet run

### S29 — re-test the approach re-check with BLIP-2 ITM

S26 left exactly one thing unresolved. It measured that *CLIP's whole-image
score* carries no information about whether a commit is correct (AUC 0.479). It
did **not** measure ASCENT's mechanism, because ASCENT scores with BLIP-2 ITM
and OSG substituted CLIP — one of the four forced deviations in
`configs/experiment/ascent_aligned.yaml`, and S26 is the first stage where that
substitution is load-bearing rather than incidental.

Two hypotheses remain live and S26 cannot tell them apart:

1. **The scorer is the problem.** BLIP-2 ITM is trained for image-text
   *matching* rather than retrieval-style alignment, and may be far more
   sensitive to whether the specific object is present than CLIP is. Then the
   port works once the scorer matches.
2. **The framing is the problem.** Any whole-image score is scene-level, so it
   is dominated by room context and cannot separate "a bed" from "a
   bed-like thing in a bedroom" whatever the model. Then ASCENT's gate is a
   safety net against gross mismatch, worth little on OSG's failure mix.

Hypothesis 2 is separable from the scorer choice and is the cheaper test: score
the **detection crop** instead of the whole frame, using the CLIP model already
loaded. If crop-level CLIP separates, the framing was the problem and BLIP-2 is
not required; if it does not, run BLIP-2 to settle hypothesis 1.

**All the machinery is already in place.** `verification.approach_recheck` and
the four gated exits are implemented, tested and merged; the only change is what
`_update_value_map` feeds into `self._approach_itm_max`.

Protocol, unchanged from S26 so the results are directly comparable:

1. Run 100 episodes with `approach_recheck=true thresh=0.0` — behaviourally
   identical to the baseline, records `approach_recheck_max` at every stop.
2. `python scripts/calibrate_approach_recheck.py outputs/<run>/`.
3. **Gate on AUC before spending an hour on an A/B.** S26's AUC of 0.479 bounded
   the best achievable at +1 episode; that check is what made the armed run
   unnecessary. Only run the paired A/B if AUC is meaningfully above 0.5.

If a threshold does calibrate, the next thing to port is ASCENT's **spatial**
rejection: `_disabled_object_map` filters every future cloud point in the
rejected cells (`object_point_cloud_map.py:102`), whereas OSG's
`blacklist(track_id)` lets the same object return as a fresh track. The S26
forced-reject smoke showed one episode rejecting six times, which is that
weakness showing up.

## Where things stand

| split | config | SR | SPL |
|---|---|---|---|
| `dev50` (representative, 50 eps) | geometric only, navmesh, no verify | 52.0% | 0.281 |
| `dev50` | + value map (weight 4) | **56.0%** | 0.320 |
| `dev50_mf` (cross-floor, 50 eps) | value map + multi-floor + stairs, no verify | 8.0% | 0.054 |
| `dev50_mf` | + VLM verifier | **12.0%** | 0.068 |
| ASCENT (published, full v1 val) | sensor-only | 63% | — |

`dev50` at the aligned protocol sits at 52% ± ~7 (n=50), against ASCENT's 63%
on the full split. Those are not yet comparable — one is 50 episodes with
habitat's navmesh, the other 2000 episodes sensor-only — but the gap is smaller
than the historical 42%-at-0.18 figure suggested.

19% of episodes need a floor change and score ~10%; the rest score ~52%. So the
remaining work splits cleanly: **stair recall** for the 19% (see the S3
decomposition — traversal already works, 13 of the 16 episodes that attempt a
climb complete it), and everything else for the 81%.

## Final results

| config | SR | SPL | notes |
|---|---|---|---|
| ASCENT (published) | 63% | — | sensor-only, v1 val |
| `final_sensor` | _pending_ | | sensor-only — the comparable number, on the full split |
| **`ascentnav` + stairs on `scenes20_ep0to4`** | **58.0%** | **0.285** | sensor-only, 100 eps — S41; the current best arm, and what `final_sensor` now composes to |
| `ascentnav` on `scenes20_ep0to4` | 55.0% | 0.284 | sensor-only, 100 eps — S39, no stair machinery (0.0% cross-floor) |
| `ascent_sensor` on `scenes20_ep0to4` | 42.0% | 0.196 | sensor-only, 100 eps — the S30-S38 port chain at its best |
| `ascent_sensor` (S8 baseline) on `scenes20_ep0to4` | 33.0% | 0.129 | sensor-only, 100 eps — S8 above; its navmesh pair scores 63.0% |
| `final_navmesh` | _pending_ | | uses habitat's ground-truth navmesh; **not** comparable to ASCENT |

`final_sensor` inherits `ascentnav`, so it is exactly the `outputs/s41_stairs`
configuration on the full v1 val split. The port chain (`ascent_sensor` and its
`ascent_sensor_*` variants) is kept reachable by name — every S30-S38 number is
reproducible — but it is no longer what the headline preset composes to.
