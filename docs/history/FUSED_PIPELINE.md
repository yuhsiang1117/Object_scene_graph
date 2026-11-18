# One pipeline for a moved object in a house with stairs

The stale-map machinery and the multi-storey machinery were built separately and
both work. Run together they are not one system: they answer the same question
at the same time, and the one with no information answers first.

This document records what was wrong, what changed, and what each change is
worth. Every flag defaults to the previous behaviour, so the baseline stays
reproducible and each arm is an explicit assertion.

## The question each mechanism is for

A dynamic episode has one shape. The agent starts with a prior map built over
the static layout. The object has since been moved, sometimes to another storey.
The agent has to work out that the map is stale, and then where to look instead.

Three mechanisms exist to answer that, and each is competent at one part:

- **Presence** answers *is it still where the map put it?* It is a log-odds Bayes
  filter per track, tested by driving to the pose and looking.
- **The container posterior** answers *which surface on this floor?* It ranks
  mapped support surfaces by affinity and proximity over path cost.
- **The floor policy** answers *which storey?* It finds portals and drives to
  them.

The ordering follows from what each one knows. Until the anchor has been tested,
the prior map is the only evidence anyone has, and it names a floor. Once the
anchor fails, that evidence is spent and the storey is genuinely open.

## What was actually happening

Measured on `outputs/osg_authored_15`, 30 episodes over two multi-storey scenes,
with presence, the posterior and the floor stack all switched on.

| observation | value |
|---|---:|
| first cross-floor request at step 13 | 17 of 30 episodes |
| ... in episodes whose object never changed floor | 10 of 19 |
| cross-floor requests raised | 405 |
| directed switch attempts they produced | 12 |
| episodes that reached the object's floor | 3 of 11 |
| absence checks in the whole run | 4 |

The floor argmax fired on the first selection round and kept firing. The
presence filter, the mechanism that knows the object moved, spoke four times in
thirty episodes.

Four defects sat underneath that, none of which announced itself.

**Containers did not exist above the ground floor.** `container_top_h_m` is a
band above the floor, 0.2 to 1.4 m. `top_height` returns an absolute world
height. On the ground floor these agree; on 00808's upper storey, at y = 2.86,
every table top measures about 3.6 m and is rejected. That floor holds 367
mapped tracks and produced zero containers against the ground floor's 76. So the
dynamic search could only ever work downstairs, and the per-floor mass that
chooses a storey had an entry for the ground floor alone.

**A storey's score was a count of its furniture.** The mass is a sum over
surfaces. With the flat proximity prior this line uses, the two floors' *mean*
candidate mass agrees to within half a percent while their sums differ by 2.4x.
The priors carry no information about which storey; the sum turned that tie into
a standing preference for the bigger floor.

**A request that could not be executed vetoed the current floor.** When the
posterior wants another storey, `_select_surface` returns None and the round
falls through to a frontier. `try_switch` needs a portal already visible, and
97% of the time there is none, so the request simply stood, and with it the
container posterior stayed off on the floor the agent was actually on.

**The staircase was in the map and nobody read it.** `save_map` records every
committed floor transition in `connectivity`, and `apply_map` restores it into
`FloorStack.stair_edges`. Nothing consumed it. Of the eight cross-floor episodes
that never reached the object's floor, four never saw a portal and never
attempted a switch at all.

One more thing was silently not happening. The nav pass runs
`layout_types=[in_anchor,cross_anchor]`, which kept the static layout out of
discovery, so the relocation source was `None` and every relocation field is
null in all 30 episodes. The record could not say which floor the object came
from, and `start_on_prior_floor` -- this benchmark's headline, "every
deterministic start is sampled on the object's prior floor" -- had no floor to
require. Starts were sampled anywhere, and `floor_class` was an accident of
sampling.

## The pipeline

Each step is a flag, listed with the preset that first asserts it.

1. **Test the anchor first** (`exploration.search_floor_requires_anchor_test`,
   `ycb_authored_15_fused`). A cross-floor request is held while a
   target-labelled track on this floor is still believed and unvisited. The held
   round falls through to same-floor selection, so it is not a wasted round.
   Released by an absence arrival, or by presence falling below the same
   `min_presence` at which candidates stop being proposed -- the same number, so
   the floor question and the candidate question cannot disagree about one
   track. Releasing on decay as well as on arrival is deliberate: a belief that
   decays below the candidate bar means the agent will never be sent there, and
   a gate that waited for an arrival that can no longer happen would deadlock.

2. **Ask presence what a mapped instance means** (`floor_evidence_by_presence`).
   `floor_target_evidence` gives a mapped instance of the target category a
   bonus that makes the switch gate absolute. On a stale map that instance is
   precisely the object that moved. Context furniture stays ungated: it does not
   move, and its presence decays from ordinary missed expectations.

3. **Do not let a wish veto this floor**
   (`search_surface_when_floor_unreachable`, `..._v2`). A failed switch falls
   back to same-floor surface selection instead of to frontiers alone.

4. **Drive to the staircase you already walked** (`floor.use_prior_stairs`,
   `..._v3`). A remembered mouth is the fallback when no portal is visible.
   `StairEdge` records both ends of a traversal, so an edge touching the current
   floor names a point on this floor whichever way it was walked.

5. **Give the storey question to the model** (`exploration.floor_llm`,
   `..._v4`). See below.

6. **Fix what a floor's surfaces are and what its score means**
   (`scene_graph.containers_floor_relative`, `exploration.floor_mass_rule: mean`,
   `floor_mass_margin`, `..._v5`).

## Where the LLM fits

The fused pipeline creates exactly one moment per episode at which "which storey
is it on now?" is both open and worth a call: the agent has driven to the pose
the stale map named, the absence sensor has said the object is gone, and the
storey request is finally released. That is the question the model gets.

It is not the question it used to get. `FloorDecisionPlanner.decide` had no
caller outside its own tests, `_floor_goal_dir` was assigned zero once and never
again, and `_floor_direction_boost` had no production caller at all. Its unit
tests passed because they set `_floor_goal_dir` by hand. Four presets claiming
`floor_llm: true` therefore measured nothing, and any conclusion drawn from them
about whether the model helps choose floors is a null over an untested knob.

The call is synchronous, happens only on a directed request, and every failure
-- no answer, a malformed one, a storey the stack has never allocated -- leaves
the posterior's own answer standing. Counters: `floor_llm_asks`, `_agreed`,
`_override`, `_stay`, `_no_answer`, `_no_such_floor`.

This placement matters more than the prompt. ASCENT asks every 60 steps from
step 100, which spends calls while the agent is still walking to a pose the map
is confident about. Here the trigger is the absence, so the model is asked when
its answer can change what happens next.

## Reading the results

The benchmark's noise floor is about three trials, so a swing of that size in
success rate says nothing on its own. Report mechanism counters first:

- `cross_floor_request_held` -- the ordering fired.
- `floor_mass_no_opinion` -- the posterior declined to choose a storey.
- `prior_stair_switch_attempts` -- a remembered staircase was driven to.
- `directed_floor_switch_attempts` and `traj_y_range` -- whether the agent
  actually climbed. On the baseline these separate the outcome perfectly: every
  episode that reached the object's floor used a directed switch, and every one
  that did not used none.
- `floor_llm_asks` -- the model was actually consulted.

An arm whose directed count falls has probably lost the floor even if its
success rate looks flat.

`scripts/legacy/run_fusion_ab.sh` runs the arms paired, and
`scripts/legacy/report_fusion_ab.py` compares them episode for episode.

## The merged perception layer (b94fa81, d8afa65)

`experiment/sr-70-50` brought seven commits: a class-agnostic FastSAM +
MobileCLIP proposal stage for objects the detector will not name, an
extent-aware container merge, and a fix stopping the absence sensor re-asking
the detector that failed. Both features default off, so every arm above is
unchanged; `build_region_proposer` returns None when disabled and FastSAM is
never imported, so there is no VRAM cost either.

### The two container fixes are complementary

`containers_floor_relative` (ours) makes upstairs containers exist at all;
`container_merge_sigma` (theirs) collapses several fragments of one physical
surface into one anchor. Rebuilt from the prior maps:

| scene | absolute band (old) | floor-relative | + sigma 2.5 |
|---|---|---|---|
| 00808 | 132, all f0 | 197 = f0 135 + f1 62 | 145 = f0 97 + f1 48 |
| 00800 | 62, **all f1** | 149 = f0 84 + f1 65 | 115 = f0 63 + f1 52 |
| 00810 | 74 = f0 72 + f1 2 | 85 = f0 74 + f1 11 | 71 = f0 60 + f1 11 |
| 00821 | 136, all f0 | 138, all f0 | 106, all f0 |

The merge removes about a quarter of the nodes while the floor-1 *share* barely
moves (00808 31.5% -> 33.1%, 00800 43.6% -> 45.2%), so it shortens the surface
queue without disturbing the container-mass floor argmax that `_select_surface`
depends on. 00800 is the sharpest evidence for the original bug: under the
absolute band every container landed on floor 1 and the ground floor had none.

The merge runs per storey -- candidates are filtered by `floor_key` before
`_merge_pass` -- so it cannot fuse a table on floor 0 with the one directly
above it.

`scripts/render_scene_graph_multifloor.py` had independently worked around the
same bug at the figure level, shifting `container_top_h_m` per storey by hand.
Running both implementations over four scenes gives identical containers -- same
ids, centres, heights and labels, 569 of them -- which is a useful check on the
pipeline fix. That workaround is now redundant.

### A caution before enabling the proposal stage here

The absence fix makes the agent *less* willing to retire a track: the close look
falls through to the proposal stage instead of trusting a detector that could
not name the object. On the perception subset that is worth +1 trial and cuts
silent close looks from 13 to 8.

The cross-floor blocker is the opposite failure. Nine of twelve episodes never
start a climb because they spend the budget chasing a live false positive on the
start floor, and a stage that makes false positives harder to disbelieve could
deepen that. If `region_proposal` is ever switched on in a cross-floor arm, read
`region_admits` and the directed-switch counters before reading SR.

### The proposal refusion (d8afa65)

Nine further commits replaced the stage's output: a region no longer enters as a
same-label `Detection` (which had cost 18 trials on the full 107) but as a
*proposal observation*, matched on the track's running-mean feature. Proposals
form a second population -- their own generator, ids from 1,000,000 up, hidden
from `tracks()` by default, never linked, invisible to the presence filter -- and
the commit bar moved onto tracks at `min_obs 4, tau 0.28`.

None of the floor, stair, portal or climb files are touched by it, and all 89
fusion-specific tests pass. The one shared file that changes substantially is
`object_layer.py`. It is safe here for three reasons worth recording, because
each is the kind of thing that would silently break a trajectory lock:

* `_rng_prop` is `default_rng(rng_seed + 1_000_003)` -- an independent generator,
  not a draw from `_rng`, so the detector's random stream is unchanged.
* `_note_best_detection` is a faithful extraction of the old inline block.
* One hunk IS a genuine behaviour change: `track.clip_ft` now accumulates even
  when `feature_memory` is None. It cannot fire in these arms, because
  `det.clip_ft` is only ever set by feature memory or the proposer and both are
  off in every fused preset.

v16 on 00808 reproduces all three episodes bit-for-bit against the pre-merge
baseline after both merges.

### Borrow their identity floor before blaming a merge

Their round-4 identity check found one trial diverging where the detector's own
scores differed on identical frames (0.596 -> 0.527) under two co-resident GPU
workers, and reproducing exactly when rerun alone. Run-to-run numerics, not a
leak. A trajectory diff on a busy GPU is not evidence until the episode has been
rerun on an idle one.
