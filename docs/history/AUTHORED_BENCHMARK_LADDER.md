# The authored-benchmark ladder, conditions A–U

> **History.** Moved out of `ARCHITECTURE.md` on 2026-10-15, unedited; the last section is two
> limitations that moved with it. These are the September 2026 conditions on the *authored* YCB
> layouts of DualMap's three scenes (96 episodes), scored by that benchmark's rule: geodesic
> distance to a goal **viewpoint** sampled on rings out to 2.0 m, so an effective tolerance of
> about 2 m to the object. They are not comparable to DualMap's 1 m rule or to the paper's
> tables. Condition N's 0.812 / 0.458 rescores to 0.396 / 0.271 at 1 m
> (`DUALMAP_PROTOCOL_EARLY_ARMS.md`, "The historical 0.812 / 0.458"). The paper's arms are
> `dualmap_osg_unified` and `mf5_osg_on_ascent_map` (`../PAPER.md` Appendix B);
> `ycb_dynamic_best` and `ycb_dynamic_cross` still compose to N and Q.

## The benchmark

Three scenes × 96 episodes, half `in_anchor` (median displacement 0.72 m) and
half `cross_anchor` (6.06 m). Success is Habitat's own criterion: geodesic
distance from the final pose to the nearest authored view point ≤ 0.18 m.

Two protocol choices are worth stating because both make the numbers *harder*,
not easier:

- **Two passes.** `ycb.map_out` / `ycb.map_in`. Pass 2 navigates from pass 1's
  snapshot. `eval/prior_map.py`.
- **Several attempts per query.** DualMap allows a query several navigation
  attempts; scoring one attempt would be a stricter protocol than the system
  being compared against, so `eval/attempts.py` matches theirs. A failed attempt
  keeps everything the map learned and lowers the candidate's belief — it does
  not blacklist, for the same reason as above.

### Results

| condition | change | in_anchor | cross_anchor | overall | SPL |
|---|---|---|---|---|---|
| A | baseline | 0.438 | 0.292 | 0.365 | 0.201 |
| B0 | + identity channel | 0.458 | 0.312 | 0.385 | 0.208 |
| B | + VLM as candidate gate | 0.333 | 0.188 | 0.260 | 0.140 |
| **C0** | + higher resolution (1280) | 0.542 | **0.375** | **0.458** | 0.220 |
| D | + search-prior fix | 0.479 | 0.292 | 0.385 | 0.196 |
| E | + unconditional frontier retirement | 0.500 | 0.354 | 0.427 | 0.221 |
| F | + two surface-search changes | 0.479 | **0.396** | 0.438 | 0.216 |
| G | + affinity fix | 0.479 | 0.375 | 0.427 | 0.211 |
| H | + per-class detection gates | 0.521 | 0.375 | 0.448 | 0.228 |
| I | (instrumentation only, no behaviour change) | 0.500 | 0.375 | 0.438 | 0.218 |
| J | + highest-scoring names | 0.521 | 0.354 | 0.438 | 0.201 |
| **K** | + specific names | 0.625 | 0.271 | 0.448 | 0.218 |
| L | + reachability at the viewpoint, non-absorbing; glance floor | 0.667 | 0.396 | 0.531 | 0.256 |
| **M** | + the target admitted on the detector's terms | **0.771** | 0.458 | **0.615** | **0.302** |
| M2 | (M repeated, identical config) | 0.729 | **0.479** | 0.604 | 0.290 |
| **N** | + candidates ranked by belief, not by belief x confidence | **0.812** | 0.458 | **0.635** | **0.319** |
| Q | + the search anchor dropped once the agent has been there | 0.708 | **0.500** | 0.604 | 0.303 |
| R | + the frontier weight back to its 1.0 default | — | — | *stopped* | — |
| Q2 | (Q repeated, identical config) | 0.708 | **0.542** | 0.625 | 0.304 |
| S | + absence readings refused beyond 3 m | 0.792 | 0.458 | 0.625 | 0.311 |
| T | + S, plus a false arrival is not an arrival (1.0 m) | 0.771 | 0.417 | 0.594 | 0.290 |
| U | + the same guard at 2.5 m | 0.750 | 0.458 | 0.604 | 0.299 |

**Q is replicated.** Q2 is the same configuration run again: in_anchor is
*identical* at 0.708 and cross_anchor is 0.500 → 0.542, the best of the whole
campaign. Mean of the two runs: in 0.708, cross 0.521, overall 0.615. It is also
the only condition that has ever moved 00848, which reached 14/30 under Q2
against 11/30 in K, L, M, M2, N, S and U.

So there are two configurations worth keeping, and they differ by exactly one
flag — a test pins that, because the one flag IS the trade:

| preset | in_anchor | cross_anchor | overall |
|---|---|---|---|
| `+experiment=ycb_dynamic_best` (N) | **0.812** | 0.458 | **0.635** |
| `+experiment=ycb_dynamic_cross` (Q) | 0.708 | **0.521** | 0.615 |

**N is still the best overall and Q the best cross_anchor.** R, S and T are all
negative, and each is a documented result rather than a dead end:

  **R** tested whether cross_anchor is short of coverage. On 00848 it tripled
  the exploration share — surface inspections 173 → 64, frontier goals 69 → 95 —
  and `P(look)` moved 0.40 → 0.47, the same value Q reached by another route.
  Stopped after two scenes. Coverage is not the lever.

  **S** range-gated absence readings, refusing 18 of them, and was a null. The
  reason is worth more than the result: `_absence_at_arrival` was doing two jobs,
  applying the evidence *and* returning an action that prevented the stop.
  Blocking the reading removed the second job, so three-attempt burnouts went
  4 → 12. The belief got more correct and the behaviour got worse.

  **T and U** added the missing half, at 1.0 m and 2.5 m. T takes the worst
  target on the benchmark — the 00848 pitcher — from 0/6 to 3/6; U puts it back
  to 0/6. Neither nets positive overall (0.594, 0.604).

  The threshold sweep is where the mechanism gave itself away. The guard
  measures distance to the agent's own GOAL, and `min_dist_to_goal` on the
  failing pitcher episodes is **0.03–0.48 m**: the agent reaches its goal every
  time. The goal is simply in the wrong place — it commits at step 1 to a track
  0.4 m from the object, and five steps later the navmesh reports
  arrived-or-unreachable for that approach goal. What T actually bought was
  budget: one extra firing per episode kept the episode alive for 251–500 steps
  instead of 288, and the pitcher was then found *incidentally*. That is not the
  same as steering the agent to the object, and it does not survive a change of
  threshold. The guard is aimed at the wrong quantity.

Q is the best `cross_anchor` of the campaign (0.500 against a previous best of
0.479) and the first condition ever to move scene 00848, which had sat at
in_anchor 0.600 for five consecutive runs. It buys that by trading, at worse
than one for one: −5 in_anchor for +2 cross_anchor.

### Two defects found by tracing one target

Chasing 00848's worst target found two real defects, both fixed and both
default-off:

  **an absence reading taken from 6.4 m.** In navmesh mode `_follow_to` returns
  None for arrived *and* unreachable, so an unreachable goal ends the approach
  exactly as an arrival does and a reading is taken from wherever the agent
  stands. The expectation gate would have caught it but is skipped whenever the
  VLM answers — "a VLM that answered about the region has already looked" —
  which holds at arm's length and not at six metres. The correct track went
  0.82 → 0.36 and the agent never went near the object again.
  `verification.absence_max_range_m`.

  **an approach that stops without arriving.** Same root cause, worse
  consequence: the agent commits at step 1 to a track 0.81 m from the true
  object, is told None on step 5 while still 6.4 m away, and STOPS — burning an
  attempt. Three attempts, episode over at step 78 with 420 steps unspent. The
  frontier side has had this distinction since `frontier_reach_m`; the approach
  never got it. `agent.approach_false_arrival_m`.

### Cross-anchor decomposes, and the half that moves is not the half that binds

```
    SR  =  P(look at it)  x  P(success | looked)
```

| | P(look) | P(convert) | SR |
|---|---|---|---|
| cross_anchor K | 0.71 | 0.38 | 0.271 |
| cross_anchor N | 0.69 | 0.67 | 0.458 |
| cross_anchor Q | 0.73 | 0.69 | 0.500 |
| in_anchor K | 0.94 | 0.67 | 0.625 |
| in_anchor N | **0.94** | 0.87 | 0.812 |
| in_anchor Q | **0.94** | 0.76 | 0.708 |

**Every gain from K to Q is conversion. `P(look)` has never moved** — 0.94 on
in_anchor and 0.69–0.73 on cross_anchor, in every condition. It is now the
binding constraint: cross_anchor cannot exceed 0.73 however good conversion
gets, and conversion is already 0.69. It also shows what Q actually cost — its
in_anchor loss is conversion (0.87 → 0.76), not coverage.

And the surface search is not what moves `P(look)`. It arrives at the true
destination surface **3 times in 96 in every condition measured** — K, M, M2, N
and Q alike — while consuming a growing share of the budget:

| | K | N | Q |
|---|---|---|---|
| surface inspections | 206 | 343 | 367 |
| frontier goals | **412** | 231 | **139** |

`search_frontier_weight = 0.3` devalues unexplored space by 70% to fund it.
Inspections nearly doubled, frontier goals fell by two thirds, and `P(look)`
did not move on either half.

N is the best measured configuration and is what
`+experiment=ycb_dynamic_best` composes. It is **+2.5 episodes over the mean of
M's two runs**, which is what the offline analysis predicted for it, and it is a
single run — M is the replicated result. The change it makes is one sort key:

| ranking key | P(correct track ranked first) |
|---|---|
| `best_score` alone | 0.635 |
| `best_score * presence.p` (K–M) | 0.729 |
| **`presence.p`, evidence as tie-break** (N) | **0.800** |

Multiplying by detector confidence *hurts*, because a confident false positive is
precisely a distant object that really does look like the target — `best_score`
is highest exactly where it misleads. Live, N commits to the correct track in
19/35 competing episodes against M's 19/40 and M2's 18/38, and all of the gain
lands in in_anchor and on 00880. Two other ranking fixes were killed offline
first: a distance/cost term (net −7, since for a cross-anchor move the ghost sits
near the start and the true object is far) and loosening `min_presence`.

M is +16 episodes on K and the best of the ladder on every column. L and M both
lift *both* halves at once rather than trading between them, which no condition
from A to K managed. Three defects, all found by tracing the largest failure
bucket rather than by tuning:

  an unreachable candidate was struck off **permanently**, and the question was
  asked about the object's own position rather than the pose the agent would
  drive to. Six of 00829's 36 target poses are off-navmesh with a reachable
  viewpoint — proven with no detector in the loop.

  a glance retired a surface **once per keyframe**, unbounded, so 64% of the
  candidate set was written off per episode, most of it never visited. The
  floor is `1 - search_detect_prob`, making true the claim `glance`'s own
  docstring already made: a passing look is weaker evidence than standing there.

  and the map discarded **33% of the times the detector named the target** —
  in 11 episodes it discarded *every* naming, so no track formed, so no
  candidate, so no approach, so the detection never got closer or bigger. All
  11 failed. Five of them were a contradiction rather than a threshold:
  `detector.class_conf` lowers the detector to 0.20 for the four weakest
  classes and `scene_graph.min_det_score` 0.35 then throws away everything they
  gained — boxes of 5146, 5077, 3102, 2808 and 1258 px, discarded on score
  alone. Two thresholds for one decision, in two config groups, the tighter one
  downstream. It is the likeliest reason condition H moved the population by a
  single episode.

All five ship OFF (`agent.reachable_via_viewpoint`,
`verification.unreachable_is_absorbing`, `exploration.search_glance_floor`,
`scene_graph.target_bypasses_gates`, `verification.target_bypasses_bbox_gate`),
so every condition of the ladder stays reproducible from its own overrides and
the winning combination is asserted in exactly one place:

```bash
python scripts/run_eval.py +experiment=ycb_dynamic_best \
  'ycb.scenes=[00829-QaLdnwvtxbs]' ycb.map_in=outputs/maps_v5/00829-QaLdnwvtxbs
```

`configs/experiment/ycb_dynamic_best.yaml` is condition M. It is pinned by
`test_the_best_known_configuration_still_composes`, because if it drifts the best
result on this benchmark stops being reproducible and nothing else would notice.

What L moved, down the funnel:

| | K | L | M |
|---|---|---|---|
| mapped it at the new pose | 48/96 | 62/96 | **71/96** |
| committed to a track on the real object | 54/96 | 60/96 | **66/96** |
| conversion once committed | 80% | 85% | **89%** |
| hit the 500-step cap | 37 | 35 | **23** |
| looked within 3 m and missed | 29 | 20 | **12** |
| in-situ recall | 0.471 | 0.523 | 0.525 |
| surfaces retired unvisited (median) | 52/82 | **1/54** | 1/54 |
| surface inspections per episode | 2.15 | **4.01** | 3.9 |

**M was repeated to find out how much of that is signal.** M2 is the same
configuration run a second time, and it reproduces to within one episode on the
headline and within one at every stage of the funnel:

| | M | M2 |
|---|---|---|
| SR | 0.615 (59/96) | 0.604 (58/96) |
| in_anchor / cross_anchor | 0.771 / 0.458 | 0.729 / 0.479 |
| mapped it at the new pose | 71/96 | 70/96 |
| committed to a track on the real object | 66/96 | 65/96 |
| conversion once committed | 89% | 89% |
| search arrived at the true surface | 3/96 | 3/96 |

Mean of the two runs is 0.609 against K's 0.448 — **+15.5 episodes at a noise
floor of about one**, so the gain is real. Per scene the repeat lands at −2, +1
and +0.

**The gain is not evenly spread, and one scene refuses to move at all:**

| scene | K | L | M | M2 | N |
|---|---|---|---|---|---|
| 00829 | 21/36 | 27/36 | 32/36 | 30/36 | 32/36 |
| 00848 | 11/30 | 11/30 | **11/30** | **12/30** | **11/30** |
| 00880 | 11/30 | 13/30 | 16/30 | 16/30 | 18/30 |

00848's in_anchor half is **identical at 0.600 across all five runs** — K, L, M,
M2 and N. That scene is not noisy, it is stuck, and everything changed so far has
been irrelevant to it.

00848 is 0.367 under all three conditions and *identical in both halves*
(in_anchor 0.600, cross_anchor 0.133) every time. Its failures say why: 13 of 19
never named the target at all and 11 of 30 episodes never looked at its new
pose. 115 bypass admissions fired there and changed nothing, because the fix
recovers detections that were discarded and on that scene the detections do not
exist. That is a coverage-and-perception scene, and none of these three changes
touches it. 00829, where the gain is largest, is also the scene most of the
tuning was done on.

The false-positive risk M takes did not materialise: 491 detections entered
through the exemption across 96 episodes and same-label tracks per episode are
unchanged (median 3, p90 6, max 8).

Two things the overall column hides:

**The two difficulties trade against each other.** `cross_anchor` trails
`in_anchor` by 0.1–0.35 throughout, which is intuitive: a cross-room move forces
a real *search* rather than a longer look nearby. No condition is best at both —
K is best at in_anchor (0.625), F at cross_anchor (0.396), C0 the most balanced.
J → K changed two query strings and moved in_anchor +0.104 and cross_anchor
−0.083. Recent work has been trading between them, not lifting both.

**Mechanism results are clearer than SR:**

| result | number |
|---|---|
| identity channel removes the livelock | repeat commits to one wrong track 251 → 6; livelocked episodes 7 → 0 |
| false-positive flood fixed (cracker box) | 15 tracks for one box → 1; SR 0.143 → 0.429 |
| query string fixed (soup can) | in-situ recall 0.03 → 0.19; that target's SR 0.083 → 0.583 |
| perception overall (I → K) | in-situ recall 0.357 → 0.471; false-positive-only episodes 20 → 6 |
| localization cliff | a track within 0.25 m of the object ⇒ SR **0.89**; beyond 2 m ⇒ 0.03 |

**Run-to-run noise is about one episode.** An operational slip launched the same
condition twice; both completed all 96 episodes and gave SR 0.438 vs 0.427, with
six of seven per-target SRs identical. H and I (I added only instrumentation)
gave 0.448 vs 0.438. Two independent repeats agree — so a two-or-three episode
difference is *not* automatically noise.

---

## Limitations as recorded at the time

*Moved from `ARCHITECTURE.md` "Known limitations" with the ladder. The admission-gate
deadlock was condition M's fix (`scene_graph.target_bypasses_gates`,
`verification.target_bypasses_bbox_gate`), which every arm since composes through
`ycb_dynamic_best`.*

**Two failure modes are now measured and unfixed.** 36% of the times the
detector names the target, the map discards the detection at the admission gate;
in four episodes of one run *every* naming was discarded, twice for a bowl named
7 and 10 times at 3.3 m with a best box of ~550 px against the 1200 gate. Since
no track forms, no candidate forms, so the agent never goes closer and the box
never gets bigger — a bootstrap deadlock. And 00848 did not move at all under L,
with cross_anchor there at 0.133.

**Perception improved across the board and SR did not move (as of K).** In-situ recall
0.357 → 0.471 and false-positive-only episodes 20 → 6 between I and K, for an
overall SR of 0.438 → 0.448. The gains are landing where episodes were already
being won.
