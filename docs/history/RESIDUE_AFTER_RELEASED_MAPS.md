# What is left after feature memory, the CLIP prior and the map rebuild

Paired reading of `outputs/osg_released_maps/flat_anchor_v2_island_close`
(107 dynamic trials, 29/54 in-anchor, 27/53 cross-anchor) against DualMap's
three measured seeds (`outputs/dualmap_official_bench/seed{12,13,14}`), plus
the navmesh measurement in `docs/history/REACHABILITY_CEILING.md`. Written 2026-09-11
to answer one question: where could more guidance still pay?

## 1. Trial-for-trial against DualMap

DualMap "wins" a trial when at least two of its three seeds score it.

| | both score | ours only | DualMap only | neither |
|---|---:|---:|---:|---:|
| in-anchor (54) | 22 | 7 | 13 | 12 |
| cross-anchor (53) | 10 | **17** | 6 | 20 |

Cross-anchor is ours: 17 discordant wins to 6. Every one of DualMap's 23
discordant losses ends `attempts_exhausted` at its text-matched anchor (bed /
desk / dining table) or `false_match` on a wrong local object; it does not
explore beyond the anchor. In-anchor is
theirs, 13 to 7, and the 19 trials only DualMap scores are:

| our bucket | n | what DualMap did |
|---|---:|---|
| perception_wall | 7 | mug x4 (00848), scissors x3: its detector emits a tight region, ours never names them beyond ~1 m |
| perception_close | 3 | 00829 cracker box in a cabinet x2, 00829 soup can: we saw them at 0.75-1.0 m and never named them; DualMap re-detects (LOCAL, 0.31-0.50 m) |
| never_in_view | 3 | scissors x2, cracker box: DualMap walks to `bed` and is credited at 0.79-0.84 m without a local path |
| committed_elsewhere | 6 | 00848 0117 banana (navmesh optimum 1.07 m, DualMap stands at 0.84 m by teleport); 00880 0116 pitcher (we stop at 1.03 m, optimum 0.97 m); 00880 0128-2 plate (200-step oscillation at 1.6-1.7 m after being at 0.84 m on step 24); 00880 0117 soup can (stale prior track 0.69 m off, can seen at 0.99 m, named once); 00848 0129-1 pitcher (committed to a track 1.58 m off, on a stool); 00829 0116 cracker box (the cabinet again, never nearer than 1.14 m) |

Thirteen of the nineteen are perception or the rule's geometry; none is the
search choosing a wrong surface with the object in reach.

## 2. The 51 failures, partitioned once

Each failure is placed in exactly one bin, decided from the record and the
navmesh:

| bin | in-anchor | cross-anchor | total | movable by |
|---|---:|---:|---:|---|
| no navmesh point within 1 m of the object | 10 | 5 | **15** | nothing in the agent; the rule and the floor plan (DualMap fails 13 of them too) |
| seen, never named, beyond 1 m (the wall) | 5 | 6 | 11 | a detector or region proposer that sees small objects at range |
| seen, never named, inside 1 m | 3 | 4 | 7 | close-range perception (item 3); the 00829 cabinet cracker box is 3 of these |
| never in view (all at the 500-step budget) | 3 | 7 | **10** | search coverage: the only bucket guidance can still move |
| other | 4 | 4 | 8 | two 2-3 cm misses, one oscillation, one budget-end commit, three wrong-instance commits, one cabinet box |

The nine in-anchor "committed elsewhere" trials that looked like search order
were the first bin: each had committed to the true track at 0.02-0.10 m and
its best stop was 0.00-0.06 m from the nearest point the agent can occupy.

## 3. Where the steps go

Successes take a median 133 steps; failures a median 500, and 42 of 51 hit
the budget. Inside the failures the budget is spent 69% in `goto_frontier`,
16% in approaches, 2% in close looks. A never-in-view trial is one where the
right surface was not proposed in time, not one where the agent looked and
missed.

## 4. What this says about "more guidance"

The three guidance mechanisms tried (feature memory, the CLIP container
prior, the argmax local pick) target surface choice and candidate admission.
On this benchmark neither is the bottleneck any more: among the reachable
in-anchor failures the search stood at the right surface in all but two (the
cabinet cracker box and a soup can, both perception inside a metre); on
cross-anchor our exploration already beats DualMap's anchor matching 17-6. The recent literature on the same problem -- IGV-RRT's
prior-map information-gain plus VLM score map with explored-region masking,
LLM-elicited per-container probabilities inside a model-based search
planner, commonsense scene graphs -- all aims at the never-in-view bucket,
which is 10 trials here, on a benchmark whose relocations are not semantic
(objects land on beds and desks regardless of class), so a class-conditioned
prior cannot be rewarded.

What can, in order of expected yield per hour:

1. **Report the reachable subset beside the full split** (no code). In-anchor
   29/44 = 65.9% is the number the agent can actually influence; state that
   DualMap's follower is not navmesh-bound.
2. **Approach hygiene**, three small fixes, each with its own counter:
   creep to the navmesh optimum instead of stopping inside the 0.1 m goal
   radius (two trials stopped 2-3 cm over the bar); the 00880 0128-2 plate
   oscillation (at 0.84 m on step 24, then 190 steps at 1.6-1.7 m); spend
   unspent attempts on the best-belief track before the budget ends (two
   trials committed at steps 471 and 497). Worth 3-5 trials, inside the
   +-3 noise floor individually, so gate on the counters.
3. **Never-in-view, the one search bucket**: an area-weighted flat prior
   (mass proportional to support area rather than one unit per container),
   so the 68 small kitchen containers of 00848 stop absorbing the queue and
   beds and tables are proposed early. Gate on "true surface selected within
   budget" before SR.
4. **Close-range perception** (18 trials): the class-agnostic proposal stage
   is the user's call; a cheaper first look is the 00829 cabinet cracker box,
   three in-anchor trials in one place that DualMap scores 3/3 at 0.31-0.45 m.
5. **Agent radius 0.10** (Habitat's default, which is what DualMap's collector
   constructs: a bare `AgentConfiguration()`) frees 6 of the 15 unreachable
   trials, of which 2 were near-miss
   stops. Legitimate only if declared; DualMap's teleporting follower is the
   larger deviation and should be declared with it.

Things not worth another arm: a semantic container prior, a VLM score map, a
threshold on feature similarity, and any further search-order tuning on
in-anchor -- the reachable in-anchor residue is perception, not order.

## Sources

- IGV-RRT, prior/real-time fusion for object search in changing environments: https://arxiv.org/html/2603.21887
- LLM-informed model-based planning for object search in partially known homes: https://arxiv.org/html/2603.23800v1
- Commonsense scene graph target localisation for object search: https://arxiv.org/html/2404.00343v1
- DualMap: https://arxiv.org/abs/2506.01950
