# SR-Gap Investigation (2026-07)

Why the from-scratch `osg` rewrite scored **~18–19% SR** on HM3D ObjectNav
while the previous ROS2 stack (`/workspace/ObjectSceneGraph_old`) scored
**54%** — what was measured, what was tried, and what actually moved the
number. Findings are saved as reproducible runs under `outputs/` and driven by
the `scripts/analyze_*.py` tools.

## TL;DR

- The gap is **not** a config/dataset artifact and **not** a perception-quality
  problem. Matching the old setup (dataset, success threshold, detector,
  embodiment, LLM) lifts SR from ~10% to ~18–19% but no further.
- **Biggest single factor: the 2D map collapses multi-floor scenes.**
  Single-floor SR is **~40%** vs **~13%** on multi-floor scenes.
- On single-floor scenes (where the map assumption holds), the remaining
  failures are **exploration coverage** (~23% never find the target) and the
  agent **committing to real target-category objects that are not the
  annotated goal** (HM3D v0.1 annotation incompleteness + premature
  commitment) — verified by looking at the actual detections.
- **Four interventions that did NOT move SR** (all confirmed by A/B): VLM
  verification (pre-approach crop, terminal close-up, and whole-image+box),
  navigable approach goal, and deferred commitment. Verification in any form is
  a dead end because the false positives are *category-correct* real objects.
- The **one** intervention that improved its target metric: the give-up
  **region-escape** halved no-find episodes (8→4) — but SR stayed flat at n=35
  (over-fencing cost as much as it gained). See open levers below.

## Method

- **Matched-config runs**: `configs/experiment/matched_*.yaml` reproduce the
  old system's evaluation (ObjectNav **v1** episodes / HM3D-semantics v0.1,
  `success_distance=0.18`, YOLOE-11l @640px, camera height 0.88 m, agent radius
  0.18, 500 steps, NVIDIA NIM LLM) so differences are attributable, not guessed.
- **Single-variable A/Bs**: every prototype is gated behind a config flag
  (default = current behavior) and compared against a baseline with the *same
  seed and episodes*; only the one variable changes.
- **Single-episode caveat**: a single episode (e.g. `ep21`) is **not** a valid
  test — it succeeds via one specific trajectory, and *any* exploration change
  perturbs that path and flips the outcome. Evaluate on the full 35-episode
  single-floor set (or 200-episode full set) where perturbations average out.
- **Runs are NOT reproducible while the VLM verifier is on** (measured
  2026-08). Two runs of *identical code and config* diverged on 2 of 4
  episodes (`verify_reject` 3 vs 1; ep0 5 steps/dtg 8.28 vs 240 steps/dtg
  9.45). The hosted NIM verifier is the only nondeterministic component —
  `exploration=sweep` uses `NullScorer`, so with `verification=off` the same
  four episodes reproduce **exactly** (same steps, same `final_xy`).

  This undercuts the same-seed premise above: a small-n A/B with
  `verification=nim` cannot distinguish a code change from VLM noise. Any A/B
  meant to prove two configs *equivalent* must run `verification=off`, where
  the comparison is exact. A/Bs measuring an SR *difference* can keep the
  verifier but need n large enough to swamp the flips — several of the
  small-n verification A/Bs in the table below are weaker evidence than their
  episode counts suggest.

## Diagnostic tools (`scripts/`)

| Tool | Question it answers |
|---|---|
| `analyze_stages.py` | Furthest state each episode reached → explore-fail vs approach-fail split. |
| `analyze_localization.py` | Where the agent STOPPED vs GT goal objects/viewpoints (approach-geometry vs perception miss). |
| `analyze_trackloc.py` | The mapped track center vs GT (requires track-center logging): navigation vs false-positive vs ellipsoid-drift. |
| `analyze_approach.py` | Why the terminal approach failed (planner_no_path vs controller_arrived; goal-cell status; min distance to goal). |

Per-episode logging that feeds these lives in `eval/runner.py`
(`target_obj_xy`, `cand_best_cam_xy`, `approach_diag`, `verify_calls`, …) and
`agent/nav_agent.py` (`state_log`, `giveup_log`, `approach_bbox_log`).

Set `eval.debug_frames=true` to write a per-step debug video per episode
(`viz/debug/ep<ID>.mp4`): live RGB + YOLOE segmentation | top-down costmap with
the agent, trajectory, chosen frontier and planned path.

## What the decomposition found

**Full v1 200-episode matched run (verification off, SR 18.5%):**
- `FAIL_explore` 28% — never mapped a target candidate.
- `FAIL_approach` 53% — reached APPROACH but stopped in the wrong place.
- Of the approach failures (track-center split): **navigation** ~36% (target
  correctly mapped, agent stopped ~2 m short of its viewpoint), **false-
  positive detection** ~38% (committed to a category-correct object far from
  any real goal), **ellipsoid localization drift 0%** (the 3D localization is
  accurate when the detection is real).

**Single-floor 35-episode matched run (SR ~40%):**
- `FAIL_explore` 23% (8/35): all time out at 500 steps with 4–11 give-ups
  clustered within ~1.3 m, ending ~1.5–2 m from a goal viewpoint — the agent
  gets **stuck reaching frontiers and thrashes**, not under-covers.
- `FAIL_approach` (13/35): **69% are false positives** — inspecting the saved
  keyframes, YOLOE correctly detects **real** TVs/sofas that are simply not in
  the HM3D goal annotations (identical in v1 and v2), so reaching them fails.
  Worst categories: `tv_monitor` (1/9), `toilet` (1/5).

## Interventions and their effect

| Intervention | Config | Effect | Verdict |
|---|---|---|---|
| Pre-approach VLM verify (crop) | `verification=nim` | 17.5% vs 18.5% | no effect (97% accept) |
| Terminal-view VLM verify | `verification=nim_terminal` | 17.5% vs 19.5% | no effect / slight loss |
| Navigable approach goal | `agent.approach_navigable_goal=true` | reached-goal 5.7%→61.5%, **SR flat** | fixes a real bug, not the SR lever |
| Deferred commitment | (reverted) | 34.3% vs 40% | **refuted** — early commits are mostly correct |
| Give-up region-escape | (reverted) | **no-find 8→4**, SR flat | moved its metric; over-fencing offset it |
| Persistent same-frontier block | (reverted) | breaks retry the agent needs | **refuted** |
| Whole-image + red-box verifier | (`verifier.py`, active when verify on) | replaces crop input for the VLM | implemented; folded into the choice A/B below |
| Center-then-verify | `verification.center_before_verify` | frame target before VLM call | implemented; folded into the choice A/B below |
| Forced-choice verification (pick category, not yes/no) | `verification.choice_mode=true` + `verification=nim` | single-floor 37.1% vs 40% off (net −1); ~96% accept, ~1 reject/26 | **no SR effect** — 5th verify variant to fail; single-floor FPs are category-correct real non-goal objects the VLM rightly accepts |
| Obstacle band → old `[0.15, 0.88]` | `mapping.obstacle_low_m/high_m` | single-floor SR ~40% → **28.6%**, no-find 8→10 | **regressed** — the 0.88 m ceiling, not the lower bound, was the culprit (reverted the ceiling) |
| Obstacle band lower bound `0.1 → 0.15` (ceiling 1.5) | `mapping.obstacle_low_m=0.15` (current default) | single-floor SR **40%** (= `[0.1,1.5]`) | neutral at scale; safe to keep |
| LOS-visibility down-weighting | `exploration.los_visibility_penalty=0.5` | 28.6% vs 28.6% (0 gained/0 lost); changed 13/35 trajectories | no SR effect at 0.5 |
| Hit-count occupancy (2-hit corroboration) | `mapping.occ_hit_threshold=2` (reverted) | SR 42.9% → **20.0%** (net −8) | **regressed** — corroboration weakens real obstacles in noise-free sim |
| Context/co-occurrence commitment gate | (reverted) | 46.0% → 48.0% (tight) → 45.0% (loose); dtg>3m 22 → 21 → 22; explore-fail 20 → 20 → **23** | **refuted** — the agent commits before the room is mapped, so there is almost no context to consult; loosening barely raised the firing rate (22 → 24) and cost SR/SPL |
| Speckle filter (clear <3-cell OCCUPIED components) | `mapping.speckle_min_cells=3` | SR 40% → **28.6%** (net −4); no-find 7→7 (no help); 32/35 trajectories changed | **regressed** — in noise-free sim it removes real thin/edge geometry, not noise |

### ASCENT's coarse-to-fine LLM reasoning makes this pipeline worse (2026-08, refuted)

Ported faithfully from `ascent/llm_planner.py` as `exploration/coarse_to_fine.py`:
an LLM picks the **storey** (per-floor room/object summaries + HM3D-train floor
priors, may answer "stay"), then the **area** (top-3 frontiers described by room
type + mapped objects, asked only when nothing is within 3 m). Both ASCENT prior
tables transcribed rather than imported. 100 paired episodes:

| | baseline | + coarse-to-fine |
|---|---|---|
| SR | **51.0%** | **44.0%** |
| SPL | 0.240 | 0.227 |
| cross-floor (24) | 20.8% | 8.3% |
| explore-fail | 18 | **22** |
| wrong-object (>3 m) | 25 | 24 |

2 gained / 9 lost, McNemar p = 0.065.

**The mechanism was healthy, which is what makes this a real result** and not a
repeat of the endpoint outage that killed the first attempt (see below): 157
calls, **zero errors**, **2.49 calls/episode** — inside ASCENT's reported
2.0-2.7 — and the LLM kept the geometric best 55.9% of the time where uniform
choice over 3 options would be 33%. It was discriminating, not guessing.

The damage lands exactly where it acted:

| | n | baseline | +ctf |
|---|---|---|---|
| episodes the LLM changed | 58 | 41.4% | **32.8%** |
| episodes it was inert | 42 | 27 succ | 25 succ |

The inert half moves by 2 — that is this pipeline's VLM-verifier noise floor.
Seven of the nine losses are on episodes the LLM steered, and five of the nine
became **explore-failures**: the agent never committed to anything and ran out
of budget.

**Why it hurts here and not in ASCENT.** Our objective carries a momentum term
(`continuity_weight=2.0`), the single largest exploration win in this project's
history (+8.5 SR). ASCENT's fine step overrides the geometric argmax with a pick
that ignores momentum *and* distance, so every override interrupts the sweep.
ASCENT can afford this because it has no momentum term to break — its value map
is direction-agnostic. Note where the two semantic-authority experiments land:
the cascade gave 44.6%, this gives 44.0%.

**The obvious confound was measured and does not explain it.** ASCENT's areas
carry a Places365 room type; ours do not. Over 91 area descriptions logged on a
15-episode debug run:

- **0% carried a room label** — the Places365 gap is total
- **100% carried objects**, mean **9.8 objects** per area
- **0 of 32 decisions** had identical options; mean pairwise Jaccard between the
  option object-sets was 0.48, with 17% of pairs clearly disjoint

So the model was given rich, distinguishable object context and still degraded
the trajectory. The missing room label is a real deviation, but it is not the
reason: the object list is the discriminative part and it was present.

**Caveat, held to this repo's own standard.** The README warns not to trust
A/Bs at n=100, because with a nondeterministic verifier nothing under ~1000
episodes resolves less than about 5 points — and this is a 7-point delta at
n=100 with p = 0.065. Two things make it stronger than the raw n suggests but
neither makes it conclusive: the comparison is **paired** on identical episodes,
and the 42 episodes the LLM never touched act as an **internal control**,
flipping by 2 where the 58 it steered flipped by 7 against. The claim this
supports is "no evidence of benefit, and a clear signal of harm concentrated
where it acted" — not a precise −7.0. A confirming run would need ~500+
episodes; that was judged not worth the compute given the direction.

**Methodological note worth keeping.** The first attempt at this A/B was
scrapped: the hosted endpoint began stalling and every call burned
`timeout_s`x3 (~6 min) before falling back to the geometric choice. It would
have completed and reported a null indistinguishable from baseline *because it
would have been baseline*. Any A/B on a network-dependent component must record
per-episode call and error counts and be checked before the SR is read.
`ctf_calls`/`ctf_errors` in `episodes.jsonl` exist for this.

### Frontier stickiness: the 40% revisit rate was not a defect (2026-08, reverted)

ASCENT disables a frontier after 20 steps in which the **distance to it** has not
fallen by 0.3 m, and disables any location selected 20 times. Ours only gives up
after 15 steps without 0.2 m of *movement*, so an agent circling a room — moving
fine, approaching nothing — is never caught. Both ASCENT mechanisms were
implemented and measured on 500 paired episodes:

| | baseline | + stickiness |
|---|---|---|
| SR | 44.6% | 45.4% (26 gained / 22 lost, McNemar p = 0.67) |
| **revisit rate** | **47%** | **46%** |
| steps to success | 160 | 167 |
| timeouts | 113 | 118 |

It fired **266 stick-disables + 54 repeat-disables against 108 give-ups** — three
times the reach of the existing net — and changed nothing, *including the metric
that motivated it*.

**The premise was wrong.** "40% of selections land within 1.5 m of a frontier
already chosen" was read as thrashing. But as the agent advances the frontier
boundary recedes, so re-selecting nearby is usually the correct continuation of
the same pursuit — walked partway, boundary moved, carry on. Disabling those is
why steps-to-success got *worse*. A secondary mismatch: blocks are keyed within
0.6 m while the metric counts 1.5 m, so many counted "revisits" were never
blockable.

Lesson for future diagnostics here: a high rate of *similar* consecutive choices
is not by itself evidence of thrash in a frontier explorer. Distinguish
"re-selected the same place and never got closer" from "the boundary moved".

### A semantic value map cannot steer this selector (2026-08, refuted)

VLFM/ASCENT rank frontiers by an image-text value map. Implemented with CLIP
ViT-B/32, measured over ~1500 episodes, reverted. Every candidate explanation was
eliminated:

| hypothesis | verdict |
|---|---|
| encoder too weak | **no** — CLIP separates target-in-view from absent at AUC 0.865 on 120 labelled pairs; BLIP-2 is no better (0.874 contrastive, 0.783 ITM head) |
| spatial attribution smeared | **fixed** — per-bearing depth clip + range falloff |
| swamped by `path_cost` | **no** — 12.6% of selections change at weight 2 |
| the changed decisions help | **no** — 47 gained / 39 lost on 500 paired episodes, McNemar p = 0.45 |

A counterfactual sweep over 1969 selections (which frontier would weight *w*
have chosen at this exact state?) shows influence saturating at **24% even at a
64× weight**:

```
weight   0.5    1.0    2.0    4.0    8.0   16.0   32.0   64.0
flips   0.0%   6.7%  12.6%  17.6%  21.0%  22.7%  23.7%  24.1%
```

In the other 76% of selections one frontier dominates on geometry — nearer, or
the only reachable option — and no semantic weight overturns it.

**That ceiling was then tested directly, and it is NOT the cause.** ASCENT's
selection shape was implemented (`exploration.selection_mode=cascade`): take the
nearest frontier when one is within 3 m, otherwise rank purely by value —
distance as a gate rather than a divisor, which gives the semantic prior full
authority exactly when a long trip is being committed to. On the same 500 paired
episodes:

| | SR | vs baseline | McNemar |
|---|---|---|---|
| baseline (`utility`) | 44.6% | — | — |
| value map, `utility` | 46.2% | +47 / −39 | p = 0.45 |
| value map, `cascade` | **44.6%** | **+48 / −48** | **p = 1.00** |

Full authority changed **96 episodes' outcomes for exactly zero net effect**. So
"the objective was in the way" is refuted as an explanation: lifting the ceiling
changes nothing.

**The signal is good and still does not help, because it answers the wrong
question.** A value map tells you which ROOM TYPE you are heading toward. Neither
of our failure modes is a room-choice problem: wrong-object commitment (24.6% of
episodes) is the agent reaching a plausible room and committing to a real but
un-annotated instance — room-level guidance cannot separate those and may reach
them faster — and explore-failures (18.6%, concentrated in `toilet` at 32.2%) are
reachability behind closed doors, not direction-choice.

A value map is the right tool for "I do not know which way to go". This agent
mostly knows which way to go and then picks the wrong object when it arrives.

**This also revises the LLM-scoring reading.** The byte-identical trajectories
were originally read as "the LLM adds nothing", then as "the objective leaves no
room". Both are now doubtful: the simplest account consistent with all three
experiments is that frontier-level semantic guidance of any kind — symbolic,
visual, weighted or gated — does not address what this pipeline actually gets
wrong.

`scripts/itm_discrimination.py` is kept — it answers "does encoder X tell
target-in-view from target-absent?" offline, for any future candidate.

### Context priors cannot gate object commitment (2026-08, refuted)

The largest remaining loss is committing to the wrong object: of 34 approach
failures on the best full-v1 run, **22 end more than 3 m from any goal** — the
agent walks confidently to entirely the wrong thing. The VLM verifier cannot
catch these (78 rejections and they still get through) because they are
*category-correct wrong instances*. The hypothesis was that **context** could:
a toilet surrounded by a sofa and a TV is not the bathroom's toilet. Implemented
as a co-occurrence gate over the scene graph (LLM-free), rejecting a candidate
when N other objects are mapped nearby and none belong with the target.

| arm | SR | SPL | ctx rejects | dtg>3m | explore-fail |
|---|---|---|---|---|---|
| no gate | 46.0 | 0.228 | 0 | 22 | 20 |
| gate, ≥3 neighbours within 3 m | 48.0 | 0.226 | 22 | 21 | 20 |
| gate, ≥2 neighbours within 4 m | 45.0 | 0.207 | 24 | 22 | **23** |

**Refuted, and the reason is structural rather than a tuning failure.** Loosening
the thresholds barely changed the firing rate (22 → 24 rejections) where roughly
double was expected. The binding constraint is not the neighbour count: it is
that **the agent commits before the surrounding room is mapped at all**. A
candidate is committed as soon as its track clears `min_obs`, so at the moment
of the decision there is almost no context to consult. Forty-six rejections
across both arms moved the target metric by nothing (22 → 21 → 22), and the
looser arm converted genuine targets into never-committed episodes
(explore-fail 20 → 23), costing SR and SPL.

This is also consistent with the annotation ceiling measured earlier: ~69% of
approach failures are real objects that simply are not the annotated goal, and
an unannotated toilet in a real bathroom has *perfect* context.

Do not re-attempt as a commitment gate. The one variant not tested is gating at
the **terminal STOP** decision rather than at commitment — by then the room has
been mapped and the context actually exists.

**Costmap-noise mitigation is a dead end here.** Sim depth is noise-free, so incomplete-mesh speckle is too rare to justify any obstacle-removal; both hit-count and speckle cost more real geometry than they filter. Keep the hard-write costmap.

## Habitat-navmesh navigation (old-stack alignment)

Replaced the from-scratch costmap planner+controller with Habitat's
`ShortestPathFollower` (drive to a goal point on the navmesh; navigate to the
object then STOP, like the old `/goal_object`), gated by
`agent.use_habitat_navmesh`. Perception/scene-graph/frontier-selection unchanged.

- Navigation is fixed as intended: single-floor **no-find 8→1, give-ups
  1.9→0.2, plan_fail 16→0**. SPL back to ~0.17.
- **A terminal bug surfaced first**: the navmesh drives the full distance to the
  object but inherited the costmap-era `approach_max_steps=12` (~3 m) cap, so
  the approach was cut off 2-4 m short *while still seeing the target* (21/21
  deadline failures were tracking the target at stop). Fixed with
  `agent.navmesh_approach_steps=200` (drive to the object, generous deadline):
  SR **11.4% → 31.4%**, deadline-fails 21→7.
- Net single-floor: navmesh **31.4%** vs costmap **40%** (≈ noise at n=35) but
  far more efficient navigation. Remaining failures: 19/24 at dtg>2m = the agent
  efficiently reaches a **false-positive / wrong-instance** object and stops on
  it. **Key insight: with navigation fixed, SR is gated by object-commitment
  decisions** — the costmap's inefficiency (stuck/give-up) had been accidentally
  masking bad commits by forcing more exploration.
- Scope: fixes navigation failures; does NOT fix multi-floor exploration
  (frontiers still come from the 2D costmap).
- **Unreachable-target bug + fix**: a target seen dead-ahead but in a sealed
  room on a DISCONNECTED navmesh island (closed door / step in the mesh) is
  visible-but-unreachable (`find_path` geodesic=inf). The agent committed and
  STOPped there (`path_consumed`), failing with budget to spare (e.g. ep5:
  stopped 9.24 m from goal on a non-goal toilet in a sealed bathroom). Fix:
  `env.is_reachable` (pathfinder.find_path) check in `_check_candidates` --
  blacklist a target not on the agent's navmesh component and keep exploring.
  ep5: FAIL(dtg 9.24) -> SUCCESS(dtg 0.05); single-floor SR 31.4% -> 34.3%.
- Progression (single-floor): costmap 40% | navmesh raw 11.4% | +approach-cap
  fix 31.4% | +reachability fix 34.3%. Remaining 19/23 failures at dtg>2m are
  reachable-but-wrong-object (false-positive / wrong-instance) commits -- the
  object-commitment decision problem, not navigation.
- **VLM verify FINALLY helps once navigation is reliable** (revises the earlier
  "verification is a dead end"): adding the forced-choice VLM gate to the
  navmesh commit path (verify candidate before approaching; reject -> blacklist
  + keep exploring) lifts single-floor **34.3% -> 42.9%** (SPL 0.226) -- the
  FIRST config to beat the 40% costmap baseline. Reject rate 38/69 (55%),
  catching real detector mislabels (bed->"sofa", chair->"tv monitor"). Nuance:
  verification did nothing for the COSTMAP system because stuck-ness already
  masked bad commits; in the navmesh system a bad commit efficiently reaches
  the wrong object, so rejecting it + continuing to explore recovers the
  episode. Navmesh navigation and the VLM FP-gate are complementary.
- VLM debugging: `eval.debug_frames=true` with a verifier also dumps every VLM
  call's input image (whole frame + red box, as sent) and response to
  `verify_debug/` (image + index.jsonl).

## Approach-oscillation bug + fix (navmesh)

The navmesh fixed frontier-reaching stuck, but a SECOND stuck remained: episodes
committed to a target then oscillated in APPROACH for 200+ steps until deadline
(e.g. ep0 chair FAIL 230 steps; ep1 chair 240 steps). Cause: the costmap-era
retreat-on-lost-detection logic (for LOS occlusion) fires in navmesh mode every
time the target leaves the camera FOV as the follower TURNS the agent along the
path -> approach -> lose detection -> retreat -> re-detect -> ... The target
stayed detected at >=1.5 m so the depth-stop never fired either. Fix: gate the
retreat off when use_habitat_navmesh (the navmesh already knows the path).
Result single-floor: SR **34.3% -> 42.9%** (+3, 0 lost), deadline-stops 7->0,
mean-steps 175->131, SPL 0.177->0.252. ep0 chair FAIL(230)->SUCCESS(51 steps).

**Best LLM-free config = 42.9%**: navmesh nav + nearest geometric exploration +
retreat-fix, NO LLM scoring, NO VLM verify -- matches navmesh+VLM-verify (42.9%)
and beats the costmap baseline (40%).

## Continuous-sweep exploration (momentum bonus) -- best config so far

Greedy per-selection argmax picks the globally-best frontier each time, which can
be behind/across the map from the last one -> ping-pong -> ~30 steps/trip and the
step budget burned on backtracking. Fix: a CONTINUITY/momentum bonus in
select_frontier -- score *= (1 + continuity_weight * align), align in [0,1] = how
far AHEAD of the agent's current heading the frontier lies (behind -> 0 bonus).
The agent finishes its current direction before reversing = a continuous sweep.
Gated by exploration.continuity_weight (0=off); heading from camera-forward; LLM-free.
A/B (navmesh, nearest, no verify): SR **42.9% -> 51.4%** (+3 net, gained 4/lost 1),
explore-steps 114->99, frontier-trips 4.3->3.7, steps-to-goal on shared successes
97->82 (-15%), SPL 0.252->0.275. Config: `exploration=sweep`.

Single-floor progression (navmesh, LLM-free): +approach-cap 31.4% | +reachability
34.3% | +retreat-fix 42.9% | +continuity **51.4%** -- approaching old-stack 54%.

## LLM frontier guidance is redundant (geometric nearest matches it exactly)

A/B of LLM-guided (`llm_text`) vs geometric NEAREST exploration (`exploration=nearest`,
`NullScorer` -> select_frontier uses unscored_prior for all -> utility =
(1 + info_gain*gain)/path_cost = nearest + largest exploration range), navmesh,
no verify: **identical** -- SR 34.3% (12/35), SPL 0.177, no-find 2, mean-steps
175, and ALL 35 episodes byte-identical (0 different steps/final_xy). The LLM
arm made 63 NIM scoring calls that NEVER changed a frontier selection (async
scorer drops in-flight requests -> stale; unscored_prior + info_gain_weight make
the geometric term dominate the argmax). => on single-floor the LLM adds cost +
a NIM dependency for zero benefit; the geometric heuristic is strictly better
(same result, LLM-free, faster). Config: `exploration=nearest`.

## Best result so far (full v1, 100 eps: navmesh + sweep + VLM verify)

`+experiment=full_v1_navmesh` (navmesh nav + reachability/retreat fixes + sweep
exploration (nearest+momentum, info_gain=2) + forced-choice VLM verify): full v1
**SR 42.0%** (42/100), SPL 0.203 -- vs ~18-19% at the start of the investigation.
Split: single-floor **68.6%** (24/35, exceeds the old stack's 54%), multi-floor
**27.7%** (18/65). VLM verify 161 calls / 83 rejects (52%) / 0 err. Per-cat: sofa
8/11, toilet 11/22, chair 10/23, bed 7/21, tv_monitor 5/17, plant 1/6.
=> single-floor is essentially solved; MULTI-FLOOR (2D costmap collapse) is now
the dominant loss, dragging the full-v1 number from ~68% to 42%.

## Future work (prioritized)

1. **Multi-floor mapping** (biggest lever, 27.7%->~68%): 2.5D/per-floor
   occupancy; floor-aware / navmesh-based frontier extraction (navigation is
   already multi-floor-capable, only exploration is stuck in the 2D costmap);
   detect stairs as cross-floor frontiers.
2. **Instance selection**: VLM catches mislabels but not real non-goal instances
   (wrong-instance chairs); add room/context priors or defer commitment.
3. **Category perception**: plant (1/6), tv_monitor (5/17) weakest; per-category
   tuning; wall-vs-monitor TV as a soft prior (HM3D counts both).
4. **Terminal precision**: near-misses reach dtg~0 but don't STOP in time; target
   a GT-viewpoint standoff, tune navmesh goal_radius / depth-stop.
5. **Validation/scale**: full v1 val (10/scene or 2000 eps) for a clean vs-54%
   comparison; multi-view/temporal VLM verify.
6. **Selective LLM exploration**: redundant on single-floor, but semantic
   room/floor guidance may help the large multi-floor scenes -- test only there.

## Open levers (the ones still worth pursuing)

1. **Multi-floor mapping** — the single largest factor (13% vs 40%). Needs
   multi-level mapping or floor-aware goal filtering; the 2D costmap cannot
   represent stairs.
2. **Exploration coverage** — 23% of single-floor episodes never find the
   target; the agent gets stuck reaching frontiers. The give-up region-escape
   halved this but needs a variant that doesn't over-fence (the frontiers the
   agent gets stuck on are often the through-door ones it actually needs).
3. **False-positive suppression** — mostly a benchmark-annotation ceiling
   (real objects that aren't goals); the addressable part is instance
   selection and per-category detection tuning (`tv_monitor`, `toilet`).

Dead ends (do not re-attempt): any form of VLM verification as an SR lever,
ellipsoid-localization tuning, persistent/regional give-up blocking,
**semantic value maps for frontier ranking** (refuted 2026-08 — the encoder is
fine; `score / path_cost` caps any semantic prior at 24% of decisions, so rework
the objective before retrying), **context/co-occurrence priors as a commitment
gate** (refuted 2026-08 -- the
agent commits before the surrounding room is mapped, so there is no context to
consult; see the section above), and **geometric stair detection via a per-cell
height gradient** (flat tread interiors fragment a staircase into disconnected
riser strips -- see docs/MULTI_FLOOR.md).
