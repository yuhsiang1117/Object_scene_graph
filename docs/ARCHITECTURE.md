# Architecture

How the dynamic-scene ObjectNav pipeline is put together, module by module.

The problem this system is built for is not an *incomplete* map — it is a map
that is **actively wrong**. Pass 1 explores a static layout and saves a snapshot.
The objects are then moved. Pass 2 starts from that snapshot, so the agent begins
each episode confidently believing the target is at A when it is at B. The
staleness *is* the experiment: an agent that rebuilds from scratch every episode
is never wrong about anything and therefore measures nothing.

Everything below follows from that. The mechanisms exist to let a map be wrong,
notice, and recover.

---

## One step

`NavAgent.act(frame)` is the whole control loop, and it reads as its own summary:

```python
def _act_inner(self, frame):
    floor_y = self.floors.observe(frame, self.step_count)   # which storey
    self.costmap.update(frame, floor_y=floor_y, ...)        # depth -> occupancy
    if self.kf_selector.is_keyframe(frame.T_wc):
        self._on_keyframe(frame)                            # detect, map, believe
        if self.cfg.exploration.search_posterior:
            self.exploration.glance(self._world(frame))     # surfaces in plain view
    if self.state in (State.INIT, State.EXPLORE, State.GOTO_FRONTIER):
        self.candidates.check()                             # does a track become a goal
    ...                                                     # dispatch to the state
```

Each line is one module. `nav_agent.py` owns the FSM state and nothing else.

## The module map

| module | owns |
|---|---|
| `agent/nav_agent.py` | the control loop and the FSM state (`state`, `_goal_xy`, `_current_path`, `_candidate_id`, `_target_obj_xy`, the deadline) |
| `agent/state.py` | the states and the two literal actions, so a handler can name a state without importing the agent |
| `agent/candidate.py` | what becomes a goal — presence, identity, evidence, reachability, and the optional VLM gate |
| `agent/approach.py` | the terminal walk and the stop decision, plus the arrival sweep |
| `agent/floor_policy.py` | storeys, portals, stairs, the climb. Inert unless `floor.enabled` |
| `agent/climb.py` | the climb itself — a flight carrot, or a stair route followed step by step (`agent.climb_route`) |
| `agent/close_look.py` | the close look at a surface — the frame an absence decision is allowed to rest on |
| `exploration/strategy.py` | where to go next — frontiers and mapped surfaces under one index |
| `exploration/search_belief.py` | `b(x)`, `d(x)`, and the inspection log that retires a searched surface |
| `exploration/selector.py` | the frontier half of that index |
| `exploration/storey_belief.py` | which storey the object is on — track evidence against coverage (`exploration.storey_belief`) |
| `objects/presence.py` | is the object still at its mapped pose |
| `objects/object_layer.py` | tracks, association, and the candidate gates |
| `perception/region_proposer.py` | region-proposal fusion: finds the target when the detector will not name it |
| `verification/absence.py` | "I got there and it was not there", as evidence |
| `verification/verifier.py` | the VLM, in both its roles |
| `graph/priors.py` | affordance and affinity — which surfaces could hold this class |
| `graph/map_store.py` | the snapshot pass 1 writes and pass 2 loads |
| `mapping/floors.py`, `floor_stack.py` | which storey the agent stands on; one occupancy grid per storey |
| `mapping/stairs.py`, `portals.py` | where the agent can change storey, and where it can see another one |
| `mapping/stair_route.py` | a walkable route up or down a staircase, one riser at a time |
| `planning/creep.py` | the approach's last metre, turned toward the goal (`agent.approach_creep_steer_deg`) |
| `pipeline/components.py` | which detector, scorer, verifier, benchmark |
| `pipeline/beliefs.py` | which belief models, with which constants |
| `perception/vocabulary.py` | which query strings the detector is asked about |
| `eval/runner.py` | build the stack, drive every episode, summarise |
| `eval/episode.py` | one episode start to STOP |
| `eval/attempts.py` | several navigation attempts per query |
| `eval/prior_map.py` | the two-pass protocol |
| `eval/instruments.py` | ground truth, on the runner's side of the wall |
| `eval/record.py` | the fields of `episodes.jsonl` (90, pinned by `test_episode_record_schema`) |
| `ros2/` | the real robot: sensor data in, a Nav2 goal out (`docs/ROS2.md`) |

---

## The mechanisms

### 1. Presence — is it still there?

`objects/presence.py`. A textbook binary Bayes filter in log-odds, with three
channels kept deliberately separate:

```
Z=1, E=1 :  dl = log( r / q )            positive
Z=0, E=1 :  dl = log( (1-r) / (1-q) )    NEGATIVE -- the channel a static map lacks
E=0      :  dl = 0                       unobserved is NOT observed-absent
```

`E` — *would we have seen it if it were there* — is the whole mechanism, and the
depth map decides it almost for free. An object that was **removed** leaves the
surface behind it visible, so the measured depth lands **beyond** the expected
band; an object hidden by a door reads **nearer** than the band. One signed
comparison separates the frame that must update the belief from the frame that
must not — which is why only the near side is gated:

```python
near = z_c - extent - self.depth_tol_m
occluded = float(np.mean(depths < near))
if occluded > self.occ_ratio_max:
    return None     # something is in front: this frame cannot see the pose
```

**No absorbing states.** The belief is clamped, so an object at p=0.002 is still
mapped, still projected, and one detection resurrects it. The clamp is
asymmetric on purpose (`l_clamp` 6.0 down, `l_clamp_pos` 3.0 up): a sighting is
worth +2.5 and a miss −0.9, so a symmetric clamp saturates after three sightings
and then needs seven clean misses to undo — which is exactly how a ghost
survives an agent standing in front of its empty shelf.

`min_presence = 0.45` is set by arithmetic, not taste. A belief reloaded from a
prior map at 0.82 lands at 0.485 after one detector-strength absence reading and
0.36 after a VLM one. So one detector miss does not retire a track — its silence
at close range is measurably unreliable — a second one does, and a single VLM
answer is decisive on its own. That asymmetry is the point of having two sensors
with different error rates.

### 2. Identity — is it *mine*?

Presence cannot answer this, and that is not a gap to be patched but a genuinely
different question. **A false positive is an object that really is there.** Every
look that disproves it as the target also re-detects it as an object and pins its
belief at the positive clamp.

Measured with the identity channel off: one episode committed to the same wrong
track **251 times in 500 steps**, its belief held at the clamp through 250
absence readings. `max_identity_rejections = 2` — one arrival can end on a bad
heading or a consumed path; two is a decision.

Ranking is the query-side payoff of the whole filter:

```python
out.sort(key=lambda t: -(t.best_score * t.presence.p))
```

A track the agent has looked for and not found sinks below one it has not
disproved, instead of being re-proposed on every replan.

### 3. The search posterior — then look *there*

`exploration/strategy.py` and `exploration/search_belief.py`.

DualMap's reaction to a failed candidate is to take the next-highest similarity
and add the failed one to an ignore list discarded when the query ends: the
discrete search problem with the belief thrown away. No model of where the object
went, no cost of getting there, no memory of what was searched, and no way for
the retry loop to decide to go *explore* instead.

The classical result is that the optimal order is by

```
    b(x) · d(x) / c(x)
```

and after an unsuccessful look, `b(x) ← b(x) · (1 − d(x))`. The pleasing part is
that `select_frontier` already computes exactly this index over frontiers
(`score / path_cost`). So this adds no objective and no planner — it widens the
candidate set to include the surfaces an object could have been moved to, and
lets the two compete. DualMap's ignore list is the degenerate case `b ← 0` after
one look.

```python
def searched(self, ref_id, detect_prob):
    f = self.factor(ref_id) * (1.0 - float(np.clip(detect_prob, 0.0, 0.99)))
    self.survived[int(ref_id)] = f
    return f
```

A visit multiplies belief rather than zeroing it, so a surface glanced at from
four metres stays plausible and one inspected closely mostly stops being — the
distinction an ignore list cannot make.

**Proximity, and a clip that was doing damage.** `b(x)` decays as `exp(-d/L)`
from where the object was last believed to be. `L` is 1.0 m, not the 4.0 m it
started at, because the benchmark's own displacements say so: in-anchor
relocations move a median 0.72 m and cross-anchor ones 6.06 m — a short mode plus
a long tail, not one exponential with a 4 m scale. Scored offline against the
true destination over 114 relocations, share where the true surface lands in the
top 5 (what one episode can afford to inspect):

| model | overall | in_anchor | cross_anchor |
|---|---|---|---|
| proximity dropped after absence | 12/114 | 5/57 | 7/57 |
| `L=4.0` with a 0.2 floor | 20/114 | 19/57 | 1/57 |
| **`L=1.0`, no floor** | **36/114** | **29/57** | **7/57** |

The floor was the bug, and an instructive one: `max(exp(-d/L), floor)` does not
*mix* two hypotheses, it **clips**. Every candidate past `L·ln(1/floor)` gets
exactly the same prior, so the whole far field ties and its order collapses to
track id — precisely the regime a cross-anchor move lives in.

**Ordering and scale are different decisions.** `affinity × proximity` is a
relative score, never a calibrated probability, and its magnitude depends on how
peaked the proximity model happens to be. But `_select_surface` also compares
that magnitude against a frontier's utility, where it decides whether the agent
searches or explores. Sharpening proximity improved the ordering (12/114 → 36/114)
and *silently switched the search line off*: a pilot went from 27 surface
inspections over six episodes to one. Anchoring the best candidate at a fixed
mass (`search_surface_mass`) fixes the scale without touching the order.

### 4. Absence on arrival — C5

`verification/absence.py`. Walking to where the map said an object was, finding
nothing, and stopping there is how a stale map converts a success into a
*confident* failure — and teaches the map nothing, so the next episode makes the
same trip.

Two sensors, with their own measured error rates, and the filter does not care
which produced the reading — the payoff of writing presence as a filter rather
than as detector bookkeeping:

```
detector silence over a whole approach   r = 0.8       (6790 logged expectations
                                                        give 0.812 in the regime
                                                        the visibility gate admits)
one VLM answer on a zoomed crop          r = 0.9, q = 0.2   (17/20 on real cases)
```

so one trusted "no" is worth about two detector misses — `log(0.15/0.98)` against
`log(0.5/0.95)` — with no fusion code.

Three things it deliberately does **not** do, each a mistake that was made and
measured:

- **A failed call is no information, never absence.** Treating a network error,
  or a "blocked" answer about an obstructed view, as evidence would quietly
  delete objects behind doors.
- **The detector's silence only counts where a detection was expected.** An
  unexpected miss says something about the view, not about the world.
- **Abandoning is not blacklisting.** On a *correct* map the agent once abandoned
  the bowl, wandered, and finished the episode standing 0.088 m from the goal —
  inside the success radius — unable to stop, because the only track that could
  have been the answer had been struck off for good.

And one thing the arrival sweep does not do: apply a negative reading per frame.
Twelve looks at the same object from the same pose are not twelve independent
observations — same range, lighting and viewing angle on the same geometry — so
multiplying their likelihoods turns one correlated detector failure into
overwhelming evidence of absence. Doing it dropped SR from 0.429 to 0.286. The
sweep's job is to give the detector a chance, not to vote.

### 5. The storey — which one, and how to get there

`exploration/storey_belief.py`, `mapping/stair_route.py`, `agent/climb.py`.
Benchmark v2 (`docs/BENCHMARK_V2.md`) starts half its episodes on another
storey, and on seed 42 the paper's arm reached the object's storey in 11 of 29
of them. Three mechanisms decided the storey before this, and none was a belief:
the "nothing near is left here" gate took the nearest flight either way (9 of 28
episodes that had to go down pursued only flights going up), two false-positive
STOPs "disproved" the storey the agent stood on (7 of the 11 that left the
object's own storey), and the container posterior's storey vote abstains by
design.

**The storey belief** scores each known storey by the target-labelled tracks on
it and discounts what this episode has already covered:

```
    P_k ∝ (w·a_k + (1 − w)/K) · (1 − d·c_k)        a_k = 1 − Π_t (1 − p_t)
```

`p_t` is a track's presence weighted by its identity evidence (`0.3^rejections`,
zero after a failed attempt) — presence alone let a false positive the gate had
already rejected keep its storey at `a = 1.0`. `c_k` is the share of the storey's
free 0.5 m blocks the agent has passed within 2 m of. The agent asks for another
storey once it is 1.8× this one, and while it knows two storeys it never switches
undirected. Storeys are the estimator's levels merged greedy-by-area at
`new_level_m` — a single-linkage merge chained 0.64, 1.74 and 3.16 m into one —
and the prior map's stair landings are dropped at load (`ycb.drop_prior_landings`):
00800's pass-1 graph held a landing at 1.74 m that a descent "arrived" on.

**The stair route** replaces the flight blob as the thing the climb walks.
`find_flights` finds a staircase as a blob of intermediate-height cells, and a
blob has no notion of walking: on 00821 its foot sat 1.5 m beside the real one
and two 300-step climbs rose 0.01 m. `plan_stair_route` asks which surfaces can
be walked to one riser at a time — a Dijkstra over 0.1 m cells with two surface
levels each (the lowest seen, and the highest seen below the camera, because an
open flight seen from below shows the floor beneath its treads), riser ≤ 0.25 m,
walls at this floor's height excluded — to flat floor near the storey asked for.
A follower turns toward a point 0.5 m along it and sweeps headings round a
refused step (`eval.allow_sliding: false` makes a step that touches an edge go
nowhere).

**What it bought, seed 42, the same four scenes (57 episodes):** started on
another storey and succeeded 5/29 → 8/29, reached the object's storey 11/29 →
14/29, flights pursued the wrong way 6/29 → 0/22, steps spent climbing per
episode 227 → 49. The storey belief alone, with the old flight climb, did as well
on the episodes it has finished (`mfv2_osg_storey`) — so the belief is the gain,
and the route is cheaper rather than better.

**A descent was pursued to the wrong end.** `find_flights` stores a descending
flight's *highest* tread in `foot_xy`; `mouth_xy` read `foot_xy` as every
flight's lowest tread and returned `top_xy` for a descent — the lowest tread
seen, under the storey the agent stood on. Every arm through 2026-10-17 ran that
way: no episode on 00862 or 00873 that had to go down ever did. From a distance
even the highest tread seen is a metre down the flight — the camera sees a tread
only along a line steeper than the flight — so `floor.descent_goal: extrapolated`
walks up the flight's own slope to where it meets the floor (`descent_mouth_xy`).

---

## The benchmarks

Every benchmark here runs the same two-pass protocol (`docs/DYNAMIC_EXPERIMENT_PROTOCOL.md`).
Two choices in it are worth stating, because both make the numbers *harder*, not easier:

- **Two passes.** `ycb.map_out` / `ycb.map_in`. Pass 2 navigates from pass 1's
  snapshot. `eval/prior_map.py`.
- **Several attempts per query.** DualMap allows a query several navigation
  attempts; scoring one attempt would be a stricter protocol than the system
  being compared against, so `eval/attempts.py` matches theirs. A failed attempt
  keeps everything the map learned and lowers the candidate's belief — it does
  not blacklist, for the same reason as above.

One agent runs on all of them: `configs/experiment/osg_unified_pipeline.yaml` carries every
mechanism flag, and a dataset preset adds only what to load (`docs/UNIFIED_PIPELINE.md`).

| benchmark | preset | success |
|---|---|---|
| DualMap's release, 00829 / 00848 / 00880 — 79 static, 54 in-anchor, 53 cross-anchor | `dualmap_osg_unified` | within 1 m horizontally of any instance of the query, ≤ 3 attempts (`eval/dualmap_release.py`) |
| multi-floor cross-anchor, 5 scenes, 25 episodes | `mf5_osg_on_ascent_map` (the paper's arm), `mf5_osg_unified` | the same 1 m rule (`ycb.score_by_object_distance`) |
| authored, 15 scenes | `ycb15_osg_unified` | the same 1 m rule |

The paper's tables and the run behind each number are `docs/PAPER.md` Appendix B; whether
they re-derive on the current tree is `docs/CROSS_ANCHOR_REPRODUCTION.md`.

The rule matters more than any mechanism. The earlier authored benchmark scored geodesic
distance to a goal *viewpoint*, sampled on rings out to 2.0 m — an effective tolerance of
about 2 m to the object — and the September ladder of conditions A–U that set many of the
constants in this document was measured under it. That ladder, with its funnels and its
null results, is `docs/history/AUTHORED_BENCHMARK_LADDER.md`; its headline 0.812 / 0.458
is 0.396 / 0.271 at 1 m.

---

## The instrument that mattered most

`eval/instruments.py`. Twice, a conclusion was drawn about *why* episodes failed,
acted on, and turned out to have been a guess — because `episodes.jsonl` could
not separate "the agent never pointed a camera at the new location" from "it did,
and the detection fell under the gate". Those have entirely different fixes.

`GroundTruthVisibility` projects the authored target position into every frame,
rejects it if outside the image or behind the camera, and rejects it again if the
depth buffer says something solid is in front. What survives is *the object was
in view, unoccluded, at this range* — and on keyframes, whether the detector then
named it.

Two details keep it honest:

```python
PROBE_OFFSETS_M = 0.06        # seven points, not one
MIN_VISIBLE_FRACTION = 0.5
```

Testing the centre pixel alone is far too permissive — an object nine tenths
hidden behind a chair back, with only its middle showing, passes — and an
instrument that counts those as "the agent looked at it" understates in-situ
recall by exactly the frames where the detector had no chance.

It is **ground truth**, and the contract is absolute: the agent is never given
it, never sees these fields, and nothing in that file writes to the agent. That
is why it lives on the runner's side rather than beside the perception it
measures.

---

## The vocabulary, which is not a detail

An open-vocabulary head does not detect objects; it scores **names**, and runs
class-competitive NMS over its own vocabulary. Which names are in the list
changes what gets found. `perception/vocabulary.py`.

**The name must be the one the detector answers to.** "pitcher" scores 0.00 on
the YCB asset at every resolution and against eight synonyms; "blue plastic
pitcher" reaches 0.71. Over 96 episodes the agent had the tomato soup can at
least half visible in 431 keyframes and the detector named it **twelve times** —
in-situ recall 0.03, against 0.90 for the bowl.

**But it must be the *specific* name, not the highest-scoring one.** Re-probed at
the dynamic poses, swapping one candidate into the vocabulary at a time:

```
005_tomato_soup_can            029_plate
  cylindrical can  0.75          red dish     0.79
  red can          0.56          red plate    0.76
  tin can          0.55          red disc     0.70
  tomato soup can  0.23          plate        0.47
```

Condition J of the ladder (`docs/history/AUTHORED_BENCHMARK_LADDER.md`) took the top
scorers and they cost more than they were worth. On their own targets they were a large
win — soup can recall 0.03 → 0.21, SR 0.083
→ 0.500 — but the rest of the vocabulary gave back eight episodes for their
seven, because "cylindrical can" describes a shape the pitcher and the bleach
bottle also have. The cracker box went recall 0.32 → 0.58 and SR 0.667 → 0.333.
K kept "tin can" and "red plate" and took in_anchor to its best value of the
campaign.

**`box` and `book` are deliberately absent** from the vocabulary. A generic class
that fits a target loosely takes the detection away from the specific class that
fits it exactly. With the target "cracker box", YOLOE labelled every sighting
"box" — 263 mapped tracks, three of them the target, none proposable. The 00848
cracker box is authored side-on, its nutrition panel reads as a menu, and "book"
scored 0.76 against "cracker box" 0.00 at every one of twenty viewpoints.

`target_vocabulary()` enforces the same rule per episode: drop a generic entry
that is a whole-word part of the target, leave everything else alone.

---

## Configuration

One module per Hydra group under `src/osg/core/config/`, because that is the unit
people override: `exploration.search_posterior=true` on a command line and
`exploration.py` in the package are the same object.

Nearly every default was chosen by an experiment and carries the measurement in a
comment beside it. `tests/unit/test_config_snapshot.py` pins every one of them, so
a constant can only change when someone changes the snapshot too — which puts it
in the diff as what it is.

The knobs that turn the dynamic mechanisms on, all off by default so each is an
explicit A/B:

```
scene_graph.presence.enabled=true          the belief filter
exploration.search_posterior=true          surfaces compete with frontiers
exploration.affinity_llm=true              LLM priors for classes the table lacks
verification.absence_only=true             VLM as absence sensor, not candidate gate
eval.attempts=3                            DualMap's protocol
ycb.map_in=<dir>                           start from a stale map
```

---

## Testing

```bash
pytest tests/unit -q          # 1535 passed, 2 skipped, 1 xfailed, ~50 s, no GPU or data
pytest -m sim                 # tests/integration: the trajectory and climb locks; Habitat + HM3D
```

Four locks exist specifically so the code can be moved without moving its
behaviour:

- **`test_config_snapshot`** — every flattened default (532), pinned; its sibling
  `experiment_fingerprints.json` pins what each preset composes to.
- **`test_episode_record_schema`** — the 90 `episodes.jsonl` keys, derived the
  way the runner builds them. Nine analysis scripts read this record by name and
  none would fail loudly on a drop; they would report zero, and the conclusion
  drawn from that zero would be wrong.
- **`test_trajectory_lock`** — a real authored YCB episode, hashed. The stub
  fixture (120 steps, no GPU) covers mapping, keyframing, frontier selection and
  the give-up nets; the dynamic fixture (real YOLOE + presence + search
  posterior, 300 steps) reaches DONE at step 236 having inspected three surfaces
  and made 1501 belief updates. It asserts a counter fingerprint *before* the
  action hash, so a failure names the module that moved.
- **`test_climb_lock`** — the climb, which neither trajectory fixture reaches:
  the real `_start_climb` + `_do_climb` from the foot of a ground-truth flight on
  00800, in three stair-height variants, with the action sequence and the climb
  counters pinned.

---

## Known limitations

**The benchmark's relocations are not semantic.** Of 53 moves whose destination
surface is mapped, objects land on: bed 26, desk 19, table 6, cabinet 3,
nightstand 3, bench 2, stool 2 — and 37 of the 53 land on a category the affinity
table does not list for that class. A tomato soup can is put on a bed seven
times. Ranking the 114 relocations by the true surface's position:

| model | top-1 | top-5 | median |
|---|---|---|---|
| affinity × proximity | 19 | 36 | 5 |
| proximity alone | 26 | 38 | 2 |
| affinity alone | 0 | 12 | 27 |
| arbitrary order | 2 | 14 | 25 |

Affinity alone is no better than arbitrary order, and multiplying it in makes
proximity *worse*. Absence from a six-entry list is therefore not evidence
(`UNLISTED_AFFINITY = 0.5`), and the ranking that remains is softened to a
tie-breaker (`AFFINITY_POWER = 0.5`). This is a limitation of the benchmark, not
a result about semantic priors: on a benchmark that moved objects the way people
do, a stronger affinity term would be worth more, and `AFFINITY_POWER` is the
knob.

**`min_presence = 0.45` is already at the best point on its curve — do not
lower it.** Two or three episodes per condition end with a correct track blocked
only by this gate, which looks like a cheap win. Measured over the 254 correct
and 709 wrong same-label tracks pooled across conditions K, L and M
(`docs/history/AUTHORED_BENCHMARK_LADDER.md`):

| threshold | correct admitted | wrong admitted | ratio |
|---|---|---|---|
| **0.45** | 201/254 | 289/709 | **0.696** |
| 0.35 | 205/254 | 300/709 | 0.683 |
| 0.30 | 207/254 | 320/709 | 0.647 |
| 0.25 | 217/254 | 393/709 | 0.552 |

0.45 is the maximum of that ratio. Dropping to 0.30 recovers **at most one** of
the blocked episodes while admitting 31 more wrong tracks, and 0.25 recovers two
to four for 104 more. Since candidate ranking has no distance term, every extra
wrong track is a chance to outrank the right one — so the gate can only be
loosened *after* the ranking is fixed, not before.

**No proximity model can rank a cross-anchor destination.** The prior decays as
`exp(-d/L)` from the last known pose, and `L=1.0` is right for in_anchor moves
(median 0.72 m) and wrong for cross_anchor ones (6.06 m). The obvious fix is a
mixture, which `container_prior`'s own comment anticipates. It was swept
offline over all 114 relocations, both as `w·exp(-d/L) + (1-w)` and as a proper
two-scale `w·exp(-d/L_near) + (1-w)·exp(-d/L_far)`:

| model | in_anchor | cross_anchor | total |
|---|---|---|---|
| shipped, `w=1.0` | 20/57 | 2/57 | 22/114 |
| flat mix `w=0.5` | 18/57 | 5/57 | 23/114 |
| flat mix `w=0.0` | 4/57 | 8/57 | 12/114 |
| two-scale `w=0.5, L_far=6` | 21/57 | 2/57 | 23/114 |
| two-scale `w=0.4, L_far=6` | 21/57 | 2/57 | 23/114 |

Every setting lands on the same frontier: the best total is 23/114 against the
shipped 22, and cross_anchor never exceeds 5/57 except by destroying in_anchor.
Combined with the earlier result that affinity alone ranks no better than
arbitrary order, this says the search prior **has no signal for a cross-room
move**. That is a limit of the feature set — distance from the old pose, plus a
category affordance table — not a tuning problem, and no amount of sweeping will
move it. Ranking work should go into candidate selection, where the information
does exist, or a genuinely new signal should be found.

**Offline ranking metrics are scale-blind.** A change that improved every ranking
number switched the search line off entirely, because the offline scorer measured
order and the live system also reads magnitude. Pilot before trusting a proxy.

**The trajectory locks are single-floor.** Both `test_trajectory_lock` fixtures run
`ycb_authored_nav`, which composes `floor.enabled=false`, so the storey machinery in
`agent/floor_policy.py` is guarded by `test_climb_lock` (the climb itself) and by unit
tests, not by a whole-episode hash. The multi-floor arm is on in every `*_osg_unified`
preset; why it is shaped as it is lives in `docs/history/MULTI_FLOOR.md` and
`docs/history/CROSS_FLOOR_CLIMB.md`.

---

## Related documents

- `docs/PAPER.md` — the method as submitted; Appendix A maps each equation to its module
- `docs/UNIFIED_PIPELINE.md` — the one agent config the three benchmarks share
- `docs/history/AUTHORED_BENCHMARK_LADDER.md` — conditions A–U, the measurements behind many constants here
- `docs/history/DYNAMIC_SCENES.md` — the full design and experiment log, in order
- `docs/history/REPORT_zh.md`, `docs/history/PROGRESS_zh.md` — report-facing summaries (中文)
- `docs/history/MULTI_FLOOR.md` — the multi-floor line, including two negative results
- `docs/history/INVESTIGATION.md` — the earlier SR-gap A/Bs
