# Multi-Floor ObjectNav — Literature Survey and Design (2026-08)

Why `osg` loses most of its SR on multi-floor scenes, what the 2024–2026
literature does about it, and the design we adopt.
中文的实现结构说明见 **[MULTI_FLOOR_CN.md](MULTI_FLOOR_CN.md)**。 Companion to
**[INVESTIGATION.md](INVESTIGATION.md)**, whose future-work item #1 this
addresses.

## Why this matters here

Full HM3D v1 SR is **42.0%**, but that averages two very different regimes
([INVESTIGATION.md](INVESTIGATION.md), "Best result so far"):

| split | episodes | SR |
|---|---|---|
| single-floor | 35 | **68.6%** (above the old ROS stack's 54%) |
| multi-floor | 65 | **27.7%** |

Single-floor is essentially solved. Multi-floor is the dominant remaining loss.
The cause is not perception and not navigation — Habitat's navmesh already
traverses stairs (`sim/habitat_env.py`, `ShortestPathFollower`). It is that
**mapping and exploration live in one 2D plane**:

- `agent/nav_agent.py` latches `_floor_y` once from the first frame's camera
  height and never re-estimates it; there is exactly one `Costmap2D` per episode.
- `mapping/costmap.py` drops points outside `[floor_y+0.15, floor_y+1.5)`, so an
  upper floor is never mapped — while `_raycast_batch` hard-writes `OCCUPIED`
  and never clears it, so geometry from different levels collides in the same
  cells and a **staircase's treads stamp the only cross-floor connection as a
  wall**.
- `mapping/frontier.py` restricts frontiers to the robot's 2D-connected `FREE`
  component, so another floor can never produce a frontier.
- `sim/habitat_env.py` (`action_to_goal`, `is_reachable`) substitutes the
  agent's **current** `pos[1]` into every 2D goal before `snap_point` /
  `find_path`. So the navmesh's multi-floor ability is unreachable through the
  API — and the reachability gate added for the sealed-bathroom bug actively
  **blacklists cross-floor targets** as unreachable.
- `graph/scene_graph.py` has no floor node and discards `center[1]`.

## The benchmark is mostly multi-floor

ASCENT measures the split directly: **65.0% of HM3D val episodes and 54.7% of
MP3D episodes are in multi-level buildings**, with **28.1% / 19.5% requiring an
explicit floor transition**. HM3D v1 val has a named 429-episode multi-floor
subset ("HM3D Mul"). Our own 65/100 multi-floor share on full v1 matches — this
is structural to the benchmark, not an artifact of our scene selection.

The sharpest number in the literature: on HM3D episodes whose start and goal are
on **different** floors, ASCENT reaches **33.3% SR where VLFM reaches 0.4%**. A
2D-collapsed map does not degrade gracefully across floors; it fails almost
completely.

### Measured on our own splits

`scripts/scene_floors.py` classifies the episode datasets we actually run
(no simulator needed). A scene is multi-floor when its goal view points cluster
at more than one height; an episode is **cross_floor** when *no* goal instance
of the target category sits on the agent's starting floor, i.e. the stairs are
mandatory.

| split | scenes (single / multi) | episodes | cross_floor |
|---|---|---|---|
| ObjectNav **v2** val | 10 / 26 | 1000 | **0 (0.0%)** |
| ObjectNav **v1** val | 6 / 14 | 2000 | **411 (20.6%)** |

Two things follow, and they matter for where the work goes:

1. **On v2, no episode requires a floor transition at all.** Every one of the 6
   categories has an instance on the start floor. So the entire multi-floor SR
   loss on v2 is the 2D map *corrupting the floor the agent is already on* —
   upstairs geometry stamping `OCCUPIED` into downstairs cells and vice versa.
   Per-floor costmaps fix this without any stair traversal.
2. **On v1 — the split our headline 42.0% / 68.6% / 27.7% numbers come from —
   20.6% of episodes are genuinely cross-floor**, in the same range as ASCENT's
   reported 28.1%. So roughly a fifth of the multi-floor loss needs real stair
   traversal and the other four fifths need clean per-floor maps.

The 10-scene list in `configs/eval/hm3d_val_single_floor.yaml` reproduces
exactly (v2-derived). One caveat it does not capture: `ziup5kvtCCR` is a single
floor under v2 annotations but its v1 goal view points sit at two heights 1.17 m
apart (a split level), so the "single-floor" preset is marginally impure when
run against v1 episodes.

### Two independent floor signals, and they agree

`scene_floors.py --scenes-dir ...` adds a second, stronger scene-level signal:
floor heights read straight off the **Habitat navmesh**, by slicing it at 5 cm
intervals and taking the navigable *area* per slice, then keeping local maxima.
It loads into a bare `PathFinder` — no `Simulator`, no GPU, ~1 s/scene.

Two details matter for anyone reimplementing it:

- **Area, not vertex density.** Vertex counts are biased by mesh tessellation.
- **A gap-cluster over navmesh vertex heights does not work at all.** A
  navigable staircase contributes vertices continuously across the whole
  vertical range, so gap-clustering reports *one* floor for any scene whose
  stairs are on the navmesh (verified: `y9hTuugGdiq`, 3 real floors, all 2070
  vertices in one cluster).

On v2 val the two signals agree on the single/multi classification for **all 36
scenes**, and on the exact floor count for 31. The five disagreements are
informative rather than errors: on `wcojb4TFT35` the navmesh finds a third floor
that holds **no goal objects** (invisible to goal-view-point clustering), while
on `p53SfW6mjZe` / `yr17PDCnDDW` / `7MXmsvcQjpJ` the goal clustering over-splits
across split-levels closer than a storey.

This gives a per-scene floor-count ground truth to validate the online floor
estimator against — which is the main risk in the design below.

## The four systems that matter

| Method | ID | Floor representation | Stair handling | HM3D SR/SPL | MP3D SR/SPL |
|---|---|---|---|---|---|
| **MFNP** | [2409.10906](https://arxiv.org/html/2409.10906) | One semantic map + a dedicated **stair channel** | Semantic seg locates stairs; entrance dilated and marked obstacle after a transition to prevent backtracking; map re-init after 200 failed steps | 58.3 / 26.7 | 41.1 / 15.4 |
| **ASCENT** | [2505.23019](https://arxiv.org/abs/2505.23019) · [code](https://github.com/Zeying-Gong/ascent) | **List of per-floor BEV maps** (obstacle + value) | Stair-aware obstacle map: stairs **re-labelled traversable** once detected; bidirectional detection incl. a look-down probe; cross-floor topology with recorded start/end points | **65.4 / 33.5** | **44.5 / 15.5** |
| **ZONDA** | [2607.21025](https://arxiv.org/html/2607.21025) | **Height-difference traversable map** (0.1 m grid) | `Δh = max\|h̄ − h̄_nbr\|` over 8-neighbours; passable if `Δh < H_agent`; stair candidate needs label conf ≥ δ_stair **and** `Δh < H_agent`; 360° panorama → VLM at the candidate | 66.5 / 33.0 | 48.2 / 21.5 |
| **TravExplorer** | [2605.19958](https://arxiv.org/html/2605.19958v1) | **Unified 3D volumetric map**, adaptive traversability layer (geometric ray-cast on support planes, semantic fallback for stairs/ramps) | Active downward camera pitching for frontiers outside the forward FOV; probabilistic multi-frame stair-instance accumulation; foothold-guided 3D search | **70.0 / 37.2** (48.7 on multi-floor eps) | 48.8 / 21.1 |

**HOV-SG** ([RSS 2024](https://arxiv.org/abs/2403.17846),
[code](https://github.com/hovsg/HOV-SG)) is the scene-graph reference rather
than a navigation policy: an explicit **floor → room → object** open-vocabulary
hierarchy traversed with a cross-floor Voronoi graph, at ~75% the representation
size of a dense open-vocab map. Its floor segmentation is the cheapest usable
recipe:

> Discretize height at **0.01 m** and build a 1D histogram of point density over
> the vertical axis. Peaks correspond to floors and ceilings. Detect local
> maxima within **±0.2 m**, keep only those exceeding **90% of the global
> maximum**, and merge duplicate responses with **DBSCAN**.

One correction from implementing it (`mapping/floors.py`): **the 90% threshold
does not transfer.** HOV-SG runs offline on a complete, uniformly reconstructed
cloud where every floor plane accumulates a comparable bin count. Ours is
incremental and view-dependent, and even two *identically sized* synthetic modes
differ by ~14% from sampling noise alone — at 0.9 the second floor is silently
dropped. We use 0.2. (The DBSCAN merge is a 1D gap-merge, which is what DBSCAN
reduces to at `min_samples=1` in one dimension — no sklearn needed.)

## What the field agrees on

1. **Do not collapse floors into one plane.** Every method keeps them separate —
   a list of BEV maps (ASCENT), a height-difference layer (ZONDA), or a true
   volumetric map (TravExplorer). This is exactly our failure mode.
2. **Stairs are traversable, not obstacles.** ASCENT explicitly re-labels
   detected stairs traversable. Our costmap does the opposite: treads fall in
   `[0.15, 1.5)` and become a permanent wall.
3. **Descending stairs need active perception** — they are invisible from a
   horizontal viewpoint. ASCENT flags depth returns **below −0.2 m** relative to
   the floor plane, then issues `LOOK_DOWN` and approaches to confirm;
   TravExplorer pitches the camera down for the same reason. We already have
   `look_down` in `habitat_env.ACTIONS` and never use it.
4. **Floor switching must be gated, not greedy.** ASCENT reasons about floors
   only when no *near* frontier exists, with a minimum interval of `T/10` steps
   between transitions. MFNP scores the decision from timestep, object-category
   diversity, exploration stagnation and an LLM prior, and bans stair use in the
   first 150 and last 200 steps. Unconstrained switching thrashes.
5. **LLM calls belong at the floor-choice level, not per step.** ASCENT's
   coarse-to-fine split cuts LLM calls **>90%** (2.0–2.7 vs 35–149 per episode)
   at higher SR. This is a live revision of our own finding: LLM frontier
   scoring was byte-for-byte redundant with geometric nearest **on
   single-floor** ([INVESTIGATION.md](INVESTIGATION.md)), but floor-level
   semantic guidance is a different question and remains untested here.

## Read against `osg`

- We are closest to **ASCENT**. A BEV occupancy grid per floor is a small,
  faithful change to `Costmap2D`, and its stair-aware obstacle map maps directly
  onto our `mapping.obstacle_low_m` / `obstacle_high_m` banding.
- **ZONDA's `Δh < H_agent`** test is a free geometric confirmation signal
  needing no model — a good fuser for our existing YOLOE `stairs` detections.
- **TravExplorer** has the strongest numbers but requires rewriting `mapping`,
  `frontier`, `room_seg` and `planning`. Not warranted while the cheap 80% is
  unclaimed.
- **HOV-SG's** floor → room → object hierarchy is what our advertised
  "building → room → object" graph should actually be.

**Latent asset:** `"stairs"` is already in `DEFAULT_VOCABULARY`
(`core/config.py`), so stair instances are *already* detected and localized as
3D ellipsoid tracks in the object layer — they are simply never consumed by
mapping or exploration. Stair detection costs us no new model and no new
weights.

## Adopted design

**Per-floor 2.5D costmap stack** (ASCENT) with the floor index derived online
from a height histogram (HOV-SG), and **stair detection from the existing YOLOE
`stairs` vocabulary entry** fused with ZONDA's geometric `Δh` check.

Staged, each gated behind a config flag defaulting to current behaviour and
validated by a single-variable A/B, per the house rules in
[INVESTIGATION.md](INVESTIGATION.md):

## Results

**Headline (full v1 val, 2000 episodes, `verification=nim`):** SR **49.8% ±2.2**,
SPL 0.218. Single-floor scenes 60.9%, multi-floor scenes 43.8%, **cross-floor
episodes 18.2%** — up from 0.0% before this work.

The per-stage table below is on a 100-episode subset (5 episodes/scene) and is
kept for the progression it shows, but **that subset is not representative**: it
read 48.0% where the 2000-episode truth is 49.8%, single-floor 68.6% vs 60.9%,
multi-floor 36.9% vs 43.8%. Treat the per-stage deltas as directional only.

### Per-stage progression (100-episode subset — see caveat above)

| run | SR | SPL | single-floor | multi-floor | cross-floor |
|---|---|---|---|---|---|
| baseline (all flags off) | 40.0 | 0.200 | 68.6 | 24.6 | **0.0** |
| + floors, per-floor maps, 3D goals | 42.0 | 0.211 | 68.6 | 27.7 | **0.0** |
| + category-timed switching | 44.0 | 0.217 | 71.4 | 29.2 | **0.0** |
| + frontier cost to free cell | **46.0** | 0.215 | 71.4 | **32.3** | **4.2** |

### The frontier goal: cost and drive are separate decisions

`select_frontier` ranks candidates with the **costmap planner** even in navmesh
mode, and `frontier_goal_xy` returned an `UNKNOWN` cell -- which the planner
frequently cannot reach, collapsing whole selection rounds (`select_none` 128
per 100 episodes at baseline). Pointing everything at the free-snapped centroid
fixes that (`select_none` -> 2) but **costs coverage**: the agent then stops at
the edge of known space instead of pushing into the frontier, so single-floor
explore-failures went 0 -> 3, mean steps 136 -> 165, and single-floor SR fell
71.4 -> 60.0 (5 episodes lost, 1 gained -- mechanistic, not verifier noise).

Splitting them keeps both: rank against the free centroid so the planner
succeeds, keep DRIVING to the frontier cell so coverage is unaffected. The two
points are a cell or two apart, so ranking barely shifts.
`exploration.frontier_cost_free_cell` does this and is the recommended setting;
`frontier_goal_free_cell` (moving the drive goal too) should stay off.

| | baseline | v4 | v5 |
|---|---|---|---|
| switch attempts | 0 | 66 | 19 |
| committed transitions | 2 | 5 | **11** (7 eps) |
| attempt → transition | — | 8% | **58%** |
| first switch at step | 315 | 109 | 115 |
| budget left after | 185 | 207 | **265** |
| phantom floors | 0 | 0 | **0** |
| transitions on single-floor scenes | 0 | 0 | **0** |

**Read this carefully before treating +4 SR as a win.** On the 35 single-floor
episodes — where the flags provably change nothing (0 transitions, 0 phantom
floors, `n_floors_seen` never exceeds 1) — **three episodes still flipped**
against baseline, two up and one down. That is the VLM verifier's
nondeterminism, and it sets the noise floor: a 4-episode total change at n=100
is inside it. SPL (0.200 → 0.217) and mean steps (223 → 210) are aggregate and
more trustworthy than the SR deltas.

**Cross-floor SR is 0.0% in all three runs — 0 of 24, unchanged.** Every part of
the mechanism now works: portals are found, the gate fires early and selectively,
the navmesh routes over the stairs, floors commit, and 11 transitions land across
7 episodes. Three cross-floor episodes did cross, and all three still failed
(dtg 2.9, 11.3, 14.0 m). Searching a fresh, empty floor within the remaining
budget is the unsolved part, and it is an exploration-efficiency problem rather
than a floor-logic one.

## Stages

| stage | change | flag | status |
|---|---|---|---|
| 0 | log agent `y`, floor class, stair tracks; `per_floor_class` metric; `scripts/scene_floors.py` | — | **done** |
| 1 | `mapping/floors.py` — `FloorEstimator` (height trace + hysteresis) | `floor.enabled` | **done** (estimate-only) |
| 2 | `mapping/floor_stack.py` — one `Costmap2D` / room-seg / frontier set per floor | `floor.per_floor_costmap` | **done** |
| 3 | 3D navmesh goals: stop substituting the agent's `pos[1]` | `agent.navmesh_3d_goals` | **done** |
| 4 | stair detection, stairs re-labelled traversable, cross-floor transitions | `floor.stairs` | **built, does not work** |
| 5 | floor-aware exploration (portals + ASCENT gate) | `floor.cross_floor` | **partial** |
| 6 | `FloorNode` in the scene graph; `to_prompt_text` grouped by floor | `floor.enabled` | **done**\* |

\* Object→floor assignment (from `center[1]`, previously discarded), `FloorNode`,
and floor-grouped serialization are in. Per-floor *room segmentation* still
waits on Stage 2 — while one costmap is shared, a room is a 2D region that
cannot be split by height, so a room takes the storey most of its objects are on.

Stage 3 is independent of the rest and worth running first as a cheap probe: it
alone may recover a slice of the 27.7%, since the reachability gate at
`nav_agent.py:578` currently blacklists every cross-floor target.

Three implementation notes that are easy to get wrong:

- **The floor signal should be the agent's own height trace, not the depth
  cloud.** `cam_y − camera_height` is a direct, per-step, noise-free reading of
  the floor the agent is *standing on*, and the agent only ever stands on
  floors. A depth-cloud histogram also peaks on ceilings and table tops. Keep
  the point-cloud histogram as an optional secondary source whose only job is
  pre-registering a floor that has been *seen* but not visited.
- **Re-labelling stairs `FREE` is not enough on its own.** `_raycast_batch`
  re-stamps `OCCUPIED` from the very next frame, so the stair mask must also
  suppress the obstacle-endpoint write — otherwise the relabelling is a no-op.
  Scope the mask strictly (semantic track **and** `Δh` **and** a minimum
  component size): every previous attempt to soften the costmap's obstacle
  writes regressed SR ([INVESTIGATION.md](INVESTIGATION.md)).
- **Back-project once per frame.** `FloorEstimator` and `Costmap2D.update` both
  want the same points; `backproject` at stride 4 dominates mapping cost, so
  pass the points in rather than computing them twice.

**Regression gate for the whole effort:** with the flags on, an episode that
only ever visits one storey must behave **identically**, not merely similarly.
Run the gate with **`verification=off`** — the hosted VLM verifier makes runs
irreproducible (see [INVESTIGATION.md](INVESTIGATION.md), "Method"), so an
equivalence A/B with it on cannot tell a real change from VLM noise.

**Gate result (35 single-floor episodes, `verification=off`, flags off vs on):
35/35 byte-identical** — same steps, same `final_xy`, same success. SR 57.1%
and SPL 0.300 in both arms. The estimator saw exactly one floor on every
episode and committed zero transitions. (57.1% rather than 68.6% because the
gate runs `verification=off`; the VLM verifier is worth ~11 points here. Both
arms are affected equally, which is the point — the gate tests equivalence,
not absolute SR.)

On the 4-episode multi-floor probe the same flags turned the one floor-crossing
episode from FAIL (dtg 2.60, 500 steps) into **SUCCESS** (dtg 0.04, 423 steps).

One latent risk the gate surfaced: `floor_y_drift` — how far the estimated
floor height strays from the value the old code latched on frame 1 — was 0 on
32 of 35 episodes but reached **0.20 m** on one (`5cdEh9F2hJL_ep2`). That
shifts the obstacle band by the same amount. It changed nothing here, but it
is not structurally guaranteed to be harmless; `floor_y_drift` is logged per
episode so a future regression can be traced to it.

### Stage 5: the decision is fixed, the follow-through is not

`mapping/portals.py` replaces stair detection with **portals** -- patches of the
height layer sitting a storey away from the current floor. Wherever another
level is visible (up a stairwell, over a mezzanine, down an opening) those cells
record a surface ~2.8 m away, which is somewhere to head for; the navmesh walks
the stairs. This needs no working stair detector, and it sees *descending*
portals, which a forward-facing obstacle band cannot.

A/B on the 5 most cross-floor-heavy v1 scenes (25 episodes, `verification=nim`
because `verification=off` makes the agent commit to a wrong object within ~20
steps and never explore long enough to test anything):

| | base (Stage 2) | + `cross_floor` |
|---|---|---|
| SR, all 25 | 32.0% | 32.0% |
| SR, cross-floor (n=13) | 0.0% | 7.7% |
| SR, same-floor (n=12) | 66.7% | 58.3% |
| switch attempts | 0 | **29**, in 16/25 episodes |
| committed transitions | 0 | **1** |

**The gate works; the climb needed two more fixes.** v1 found portals reliably
(140 seen, deltas 1.85–3.72 m — real storey gaps) and made the agent *try* in 16
of 25 episodes against never trying before, but only 1 of 29 attempts reached
another storey. Two causes, both now fixed:

1. **The climb was interrupted, not failed.** Three episodes climbed ~1.6 m and
   turned around. While `on_stairs` the floor id is frozen, so the costmap the
   agent sees is still the floor *below* — exploration picked a downstairs
   frontier and walked it back down. The portal goal is now held while the agent
   is making vertical progress or is `on_stairs`, and the give-up net judges a
   portal pursuit on **vertical** progress (a switchback staircase barely moves
   in x/z while climbing fine).
2. **One threshold could not do both jobs.** `new_level_m = 1.8` had to reject
   landings (measured up to 1.4 m) *and* accept a climb stalling at 1.6 m —
   unsatisfiable. A storey can now also be committed by **horizontal room**:
   displacement from where the agent first stood at that height, since a 1.2 m
   landing pins you within 1.2 m however long you stay. Displacement, not path
   length — path length is racked up by pacing on the spot.

| | base | v1 | v2 (+commit) | v3 (+patience) |
|---|---|---|---|---|
| SR all / cross / same | 32.0 / 0.0 / 66.7 | 32.0 / 7.7 / 58.3 | 32.0 / 7.7 / 58.3 | 32.0 / 7.7 / 58.3 |
| switch attempts | 0 | 29 | 28 | 24 |
| **committed transitions** | **0** | **1** | **4** | **4** |
| pursuits ending "arrived" | 0 | 0 | 4 | 4 |
| ending "no vertical progress" | 0 | 0 | 21 | 9 |

**Every climb that starts now completes** (arrivals == transitions == 4).
Widening the patience window cut stalled pursuits 21 → 9 but added no arrivals,
so patience is not the remaining constraint.

**SR did not move.** 32.0% in all four arms; the cross-floor +1 and same-floor
−1 are single episodes under a nondeterministic verifier. What changed is
mechanical: floor transitions went from impossible to routine.

### The route bug (v4) — mechanism fixed, outcome not

The `no_vertical_progress` failures were a **bug**, not geometry. `_follow_path`
chose the navmesh snap height from the *state*:

```python
None if self.state == State.GOTO_FRONTIER else self._goal_floor_y_cache
```

True when written in Stage 3 (frontier goals are on the agent's own floor),
false from Stage 5 on, when portal pursuits began reusing `GOTO_FRONTIER` as
their driving state. Every portal goal therefore had its target height
discarded and its (x, z) snapped at the agent's *current* height — routing the
agent to a point directly beneath the mezzanine, where it arrived, gained no
height, and gave up. Extra patience did not help because it was patiently
walking to the wrong place. `is_reachable` *did* pass the target height, so the
portal was judged reachable and then driven to incorrectly; that inconsistency
is what disguised it as a geometry problem. Keyed on the goal
(`_portal_active`) now, with a regression test covering both directions.

| | base | v1 | v3 | **v4 (route fix)** |
|---|---|---|---|---|
| committed transitions | 0 | 1 | 4 | **20** |
| episodes changing floor | 0 | 1 | 3 | **11 / 25** |
| pursuits "arrived" | 0 | 0 | 4 | **20** |
| "no vertical progress" | 0 | 0 | 9 | **5** |
| mean vertical travel | 0.22 m | 0.46 | 0.45 | **1.29 m** |
| SR all / cross / same | 32.0 / 0.0 / 66.7 | 32.0 / 7.7 / 58.3 | 32.0 / 7.7 / 58.3 | 28.0 / 7.7 / 50.0 |

The transitions land where they should — **10 of the 11 episodes that changed
floor are cross-floor episodes** — and one cross-floor episode now succeeds *via*
a transition (`bxsVRursffK_ep10`: crossed at step 80, 186 steps left, success).

**SR still did not improve, and slipped by one episode.** Two measured reasons,
both about *when* rather than *whether*:

- **Transitions come too late.** Mean 167 steps remain after the first
  transition and **5 of 11 have under 100 left** — not enough to search a fresh,
  empty floor. This is inherent to the ASCENT gate: "no near frontier" only
  becomes true once the current floor is largely exhausted, which is past
  half the budget.
- **Some episodes thrash**, taking 4–5 transitions and paying the travel cost
  each time.

So the next lever is the switch *timing* and hysteresis, not the mechanism:
decide to change floor while budget remains (e.g. on evidence the target
category is absent from this floor rather than on frontier exhaustion), and damp
repeat crossings.

### Stage 4 is a negative result — read this before extending it

The geometric detector in `mapping/stairs.py` is implemented and tested, but it
**does not detect staircases**, and `floor.stairs` stays default-off.

The criterion was "a cell is steppable when `min_dh < Δh < climb_limit`", with
connected steppable cells forming a stair region. Measured behaviour:

| `stair_min_rise_m` | result |
|---|---|
| 0.3 | Fires in **28 of 35 single-floor episodes**, relabelling 43k cells. Every region rose only 0.33–0.75 m — thresholds, sloped floor, low furniture. Not staircases. |
| 1.0 | Fires **nowhere**, including on an episode where the agent demonstrably crosses floors. |

No threshold sits in between, because the failure is structural, not numeric.
A real staircase has **flat treads**: the interior of each tread has `Δh ≈ 0`,
falls below `min_dh`, and is excluded — leaving only riser edges as thin
disconnected strips. Measured on a synthetic staircase with 0.28 m treads and
0.17 m risers: **9 fragments, largest 200 cells, 0 regions detected**. The same
surface as a continuous *ramp* gives 1 component of 1500 cells and is detected
fine. `test_discrete_treads_are_detected` is an `xfail` pinning exactly this.

The deeper error is that ZONDA uses `Δh < H_agent` as a **traversability**
filter (flat floor passes) and relies on the *semantic* label to pick out
stairs. This implementation inverted that, using `Δh` as the detector itself.
Fixing it means either segmenting passable regions by height variation over a
sliding window, or restoring the semantic label to a primary role — which our
own measurement says is only available on 15% of multi-floor episodes.

**Before investing in that, note that the payoff is smaller than the plan
assumed.** In the best config (`use_habitat_navmesh`) the costmap does **not**
drive motion — `_follow_path` delegates to the navmesh, and the agent already
traverses stairs successfully with Stage 2 alone (probe ep3: FAIL → SUCCESS,
one committed floor transition, zero stair cells relabelled). The costmap only
affects frontier extraction and path *cost*. So a working stair mask would help
the agent *notice* the staircase as explorable space, not walk up it.

What Stage 4 did land, and is worth keeping: the per-cell height layer
(`Costmap2D.track_height`), the `Δh` primitive, the `stair_mask` obstacle-stamp
exemption (without which any FREE relabel is undone by the next frame), and
`StairEdge` records on `FloorStack`.

### An LLM floor choice does not beat the co-occurrence table (2026-08, refuted)

The one place `INVESTIGATION.md` predicted the retired LLM lever might still pay
was *floor-level* guidance, and ASCENT agrees — its coarse step asks a 7B model
which storey to search. Implemented as `floor.llm_floor_choice` (see
`exploration/coarse_to_fine.py`): the model sees each storey's room types,
mapped objects, an explored flag, and HM3D-train floor priors, and answers with
a storey — including the option to **stay**, which `graph/priors.py` structurally
cannot express because it judges the current floor alone and never compares
storeys.

It fired 18 times across the 24 cross-floor episodes of a 100-episode run,
moving 11 and vetoing 7, with zero call errors. **Cross-floor SR went 20.8% →
8.3%** (5 successes → 2). That is three episodes and well inside noise at n=24,
so read it as "no evidence of benefit", not as a measured harm — but it was run
alongside the fine-grained area choice, which cost 7 SR points overall, and
nothing here argues for keeping either. Full accounting in
[ASCENT_GAP.md](ASCENT_GAP.md) and [INVESTIGATION.md](INVESTIGATION.md).

The cheap conclusion stands: the binding constraint on cross-floor episodes is
still **seeing a portal at all** (14 of 24 episodes never do), not choosing
correctly between storeys once several are known. Reasoning better about a floor
you cannot detect buys nothing.

### Biggest risk

**A multi-flight staircase landing looks exactly like a new floor** to any
dwell-based height estimator: it is a genuine, sustained height plateau midway
between two storeys. If the estimator registers one, the map splits mid-stairs,
that "floor" has no reachable frontiers, and the episode stalls — strictly worse
than today's collapse, which at least leaves one connected (if wrong) map.

Mitigation is measurement before commitment: Stage 1 ships in an
estimate-and-log-only mode, so `n_floors_seen` can be checked against the
per-scene navmesh ground truth above across all 100 v1 episodes *before* any
behaviour depends on it. If landings do turn out to be the failure mode, require
a new level to be both >1.5 m from every known level **and** reached via a
completed stair traversal — i.e. only staircases create floors.

Second risk: Stage 4 assumes the YOLOE `stairs` class actually fires on HM3D
staircases at useful range. Nothing in this repo has ever measured that, which
is why Stage 0 logs `n_stair_tracks` / `stair_tracks` per episode. **Check that
before building Stage 4** — if it yields well under one track per multi-floor
episode, drop the semantic half and raise the geometric bar instead.

## References

- ASCENT — *Stairway to Success: Zero-Shot Floor-Aware Object-Goal Navigation via LLM-Driven Coarse-to-Fine Exploration*, [arXiv:2505.23019](https://arxiv.org/abs/2505.23019) · [project](https://zeying-gong.github.io/projects/ascent/) · [code](https://github.com/Zeying-Gong/ascent)
- MFNP — *Multi-Floor Zero-Shot Object Navigation Policy*, [arXiv:2409.10906](https://arxiv.org/html/2409.10906)
- ZONDA — *Zero-shot Object Navigation with Dynamic Avoidance in Multi-floor Environments*, [arXiv:2607.21025](https://arxiv.org/html/2607.21025)
- TravExplorer — *Cross-Floor Embodied Exploration via Traversability-Aware 3-D Planning*, [arXiv:2605.19958](https://arxiv.org/html/2605.19958v1)
- HOV-SG — *Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation*, RSS 2024, [arXiv:2403.17846](https://arxiv.org/abs/2403.17846) · [code](https://github.com/hovsg/HOV-SG)
- *3D Scene Graphs: Open Challenges and Future Directions*, [arXiv:2606.19383](https://arxiv.org/abs/2606.19383)
