# Exploration & frontier selection: ours vs. ASCENT

Companion to [ASCENT_COMPARISON.md](ASCENT_COMPARISON.md), which covers the
multi-floor machinery. This one is about the layer underneath: **given a set of
frontiers, which do you go to?**

Source: `relative_works/ascent/` (`ascent/llm_planner.py`,
`ascent/mapping/value_map.py`, `ascent/map_controller.py`) against `osg` at
`ff8b661` (`exploration/selector.py`, `mapping/frontier.py`,
`agent/nav_agent.py`).

---

## 1. Side by side

| | ASCENT | ours |
|---|---|---|
| Frontier source | VLFM frontier extraction on the per-floor obstacle map | WFD on the per-floor costmap |
| **Semantic value** | **BLIP-2 image-text cosine, painted into a persistent 2D value map** | **none** |
| Distance | implicit — the LLM sees images, not costs | explicit: utility = `score / path_cost`, cost from the real planner |
| Direction / momentum | none | continuity bonus toward the current heading |
| Information gain | none | count of UNKNOWN cells within a radius |
| Final choice | top-k by value → **LLM picks one from images** | argmax of utility |
| Anti-thrash | disable on no distance change (20 steps); disable after 20 non-consecutive selections | location+floor blacklist (0.6 m); give-up after 15 steps without 0.2 m of movement |
| Per-step cost | BLIP-2 forward pass every step | nothing |
| LLM calls | 2.0–2.7 per episode | **0** (`exploration=sweep`) |

---

## 2. The one big difference: a semantic value map

ASCENT inherits VLFM's `ValueMap`. Every step:

```python
cosines = [self._itm.cosine(rgb, prompt.replace("target_object", target))
           for prompt in self._text_prompt.split("|")]
self._value_map[env].update_map(cosines, depth, tf_camera_to_episodic, ...)
```

The scalar is projected into a 2D grid through the camera's FOV cone, weighted
by a **confidence mask** (`_min_confidence = 0.25`, `_decision_threshold = 0.35`)
that falls off toward the edges of the field of view and with depth. Values
accumulate across views, so the map remembers "that direction looked like a
bedroom" long after the agent has turned away.

Frontiers are then ranked by reading that map:

```python
sorted_pts, sorted_values = value_map[env].sort_waypoints(frontiers, 0.5)
```

— the **max** value within 0.5 m of each frontier.

**We have no equivalent.** Our `select_frontier` computes

```
utility = score / path_cost
score   = prior × (1 + info_gain_weight · gain/gain_max)
                × (1 + continuity_weight · align)
```

and in the best config `prior` is a constant (`NullScorer`), so the ranking is
**purely geometric**: nearest, biased toward large unexplored areas and toward
the direction of travel. Nothing in it knows what a toilet looks like.

### Why our "LLM is redundant" finding does not cover this

`INVESTIGATION.md` records that LLM frontier scoring produced *byte-identical*
trajectories to geometric-nearest across all 35 episodes. That test scored
frontiers from a **serialized scene-graph text description**. ASCENT's value map
is a different signal in three ways:

1. **Visual, not symbolic** — image-text similarity on raw RGB, not a list of
   detected object labels. It fires on wall colour, furniture style, room shape:
   everything the detector vocabulary throws away.
2. **Dense and persistent** — a value per map cell accumulated over every frame,
   not a score per frontier computed on demand.
3. **Cheap enough to run every step** — one BLIP-2 forward pass, no network
   round-trip, so it never goes stale. Our async LLM scorer dropped in-flight
   requests and its scores arrived too late to change an argmax.

**Update: visual frontier scoring has now been tested here too, and it also
fails — but for a different and more interesting reason.** A CLIP value map was
implemented, measured over ~1500 episodes and reverted. Every candidate
explanation was eliminated in turn: the encoder discriminates well
(target-in-view vs absent, AUC 0.865; BLIP-2 is no better at 0.874), the spatial
attribution was fixed with a per-bearing depth clip, and the term is not swamped
(12.6% of selections change at weight 2). The changed decisions simply do not
help: 47 gained / 39 lost, McNemar p = 0.45.

A counterfactual sweep over 1969 selections shows the semantic term's influence
**saturating at 24% even at a 64× weight** — in the other 76%, one frontier
dominates on geometry. That looked like the cause, so ASCENT's selection shape
was implemented (`selection_mode=cascade`: nearest within 3 m, else pure value)
to lift the ceiling. It did not help: **44.6% vs a 44.6% baseline, 48 gained /
48 lost, p = 1.00** — 96 episodes changed outcome for zero net effect.

So the objective was not the obstacle. The signal answers the wrong question: a
value map says which *room type* to head toward, while this pipeline loses on
picking the wrong *instance* after arriving (24.6% of episodes) and on
reachability behind closed doors (18.6%). Details in
[ASCENT_GAP.md](ASCENT_GAP.md) §5.

**And the symbolic path has now been retested at ASCENT's own grain.** The three
distinctions above were the reasons to expect a different outcome from the old
"LLM is redundant" result. Two of them have since been removed: §4.3 gives an LLM
the same *coarse-to-fine* decision structure ASCENT uses, on object-label context
that is dense (9.8 objects per area) and genuinely distinguishable — and it loses
7 SR points. Only the first distinction, **visual vs symbolic**, is still
untested, and the CLIP value map result above already shows the visual signal
failing on its own terms. The honest summary is that frontier-level semantic
guidance has now failed here in symbolic form, in visual form, and in ASCENT's
own two-level form.

---

## 3. What we have that they do not

**Explicit path cost in the objective.** Our utility divides by the true
planner path cost, so a marginally better frontier across the building loses to
a decent one nearby. ASCENT ranks by value alone and hands the top-k to the LLM;
distance enters only through whatever the LLM infers from the images. Our
`min_path_cost_m` floor also stops adjacent frontiers from producing infinite
utility.

**A momentum term.** `continuity_weight` boosts frontiers ahead of the current
heading, so consecutive goals form a sweep rather than ping-ponging across the
map. This was the single largest exploration win in the project's history
(+8.5 SR, 51.4% from 42.9% at the time), and ASCENT has no analogue — its value
map is direction-agnostic and the LLM sees no heading.

**Information gain.** We boost frontiers by the UNKNOWN area they would reveal.
ASCENT's value map says nothing about how much is *unseen*, only how promising
what has been seen looks.

**Zero marginal cost.** Our best config makes no network calls and runs no
second model. ASCENT needs BLIP-2 resident for the value map plus Qwen2.5-7B for
the decisions.

---

## 4. What they have that we should take

Ranked by expected value against our measured failures
(2000-episode run: SR 49.8%, 40% frontier re-selection rate, 275 stub-blocks).

### 4.1 Frontier stickiness — implemented, and the defect was misdiagnosed

```python
STICKY_FRONTIER_DISTANCE_THRESHOLD = 0.3   # metres
STICKY_FRONTIER_STEP_THRESHOLD     = 20    # steps
REPEATED_SELECTION_THRESHOLD       = 20    # selections
```

`_handle_frontier_stick_and_disable` disables a frontier when the agent has
pursued it for 20 steps **without the distance to it changing by more than
0.3 m**, and disables any frontier selected 20 times non-consecutively.

Ours blacklists by *location* (within 0.6 m of an abandoned point) and gives up
after 15 steps without 0.2 m of *movement*. The distinction is real — ASCENT
measures progress **toward the goal**, we measure movement **at all**, and an
agent circling a room satisfies ours while satisfying nothing useful.

**Implemented and reverted.** Both mechanisms (disable after 20 steps without
closing 0.3 m; disable a location selected 20 times) on 500 paired episodes:

| | baseline | + stickiness |
|---|---|---|
| SR | 44.6% | 45.4% (26 gained / 22 lost, p = 0.67) |
| **revisit rate** | **47%** | **46%** |
| steps to success | 160 | 167 |

It fires hard — **266 stick-disables + 54 repeat-disables against 108
give-ups**, three times the reach of the existing net — and moves nothing,
including its own target metric.

**The defect was misdiagnosed.** The "40% of selections repeat a frontier we
already chose" figure was read as thrashing. But as the agent advances into
unexplored space the frontier boundary *recedes*, so re-selecting near a previous
choice is usually correct: you walked partway, the boundary moved, you continued.
That is frontier-following working, not pathology — which is why disabling those
pursuits made steps-to-success slightly *worse*. (A second, smaller flaw: blocks
are keyed within 0.6 m while the revisit metric counts 1.5 m, so many "revisits"
were never blockable anyway.)

Not fully refuted — a variant that also required no information gain before
disabling might behave differently — but nothing here justifies the two
parameters and the extra code path.

### 4.2 A visual value map — tried both ways, refuted

**Tested and reverted.** The encoder is fine (AUC 0.865), the attribution was
fixed, and the 24% influence ceiling was then *lifted* by implementing ASCENT's
cascade (nearest within 3 m, else pure value). Result: **44.6% against a 44.6%
baseline, 48 gained / 48 lost, p = 1.00**. Full authority, zero effect.

The signal answers the wrong question: a value map says which *room type* to head
for, while this pipeline's losses are picking the wrong *instance* on arrival
(24.6% of episodes) and reachability behind closed doors. See
[ASCENT_GAP.md](ASCENT_GAP.md) §5.

### 4.3 Coarse-to-fine LLM reasoning — implemented, and it HURT

The one borrowing on this list that did not come back null.
`exploration/coarse_to_fine.py` ports `ascent/llm_planner.py`: the LLM picks the
**storey** first (per-floor room/object summaries, HM3D-train floor priors, may
answer "stay"), then the **area** among the top-3 frontiers — and only when
nothing is within `ctf_nearby_m=3.0`, which is the gate that keeps ASCENT at
2-3 calls per episode instead of 35-149.

| | baseline | + coarse-to-fine |
|---|---|---|
| SR | **51.0%** | **44.0%** (2 gained / 9 lost, p = 0.065) |
| SPL | 0.240 | 0.227 |
| cross-floor | 20.8% | 8.3% |
| explore-fail | 18 | **22** |
| wrong-object (>3 m) | 25 | 24 |

It was not an outage: 157 calls, **zero errors**, **2.49 calls/episode** (ASCENT
reports 2.0-2.7), geometric best kept 55.9% of the time against 33% chance. On
the 58 episodes where it changed a decision SR went 41.4% → 32.8%; on the 42
where it was inert, 27 → 25 (the verifier noise floor).

**It breaks the sweep.** Section 3 of this document lists our momentum term as
the largest exploration win in the project's history (+8.5 SR), and ASCENT's fine
step overrides the geometric argmax with a pick that ignores momentum and
distance both — so every override interrupts a pursuit mid-flight. ASCENT has no
momentum term to break. Five of the nine lost episodes ended as explore-failures,
having never committed to any target.

The Places365 excuse was measured and does not hold: over 91 logged area
descriptions, 0% carried a room label but 100% carried objects (mean 9.8 each),
no decision had identical options, and mean pairwise Jaccard between option
object-sets was 0.48.

### 4.4 Frontier images for the decision

`extract_frontiers_with_image` crops the RGB region each frontier was observed
from and hands those to the LLM. Even without a value map, this changes what a
semantic scorer can see — our `to_prompt_text` gives it object labels only.
**Untested**, and the weakest remaining candidate on this list: §4.3 shows the
LLM already fails with good *symbolic* context, so the case for it rests
entirely on images carrying something object labels do not.

---

## 5. Where the losses actually are (2000 episodes)

Exploration improvements are bounded by how much of the loss is exploration:

| | share |
|---|---|
| never committed to a target (explore failure) | ~19% of episodes |
| committed but stopped in the wrong place | ~36% |
| of those, > 3 m from any goal (wrong object) | ~22% |

So exploration caps out at roughly **+19 points** even if made perfect, and the
larger pot is object commitment — where our context-prior attempt was
**refuted** (the agent commits before the room is mapped) and the VLM verifier
plateaued.

The relevant asymmetry: a value map would help *both*. It biases exploration
toward the right room **and** gives a semantic prior at the moment of
commitment, which is exactly what our context gate lacked.

---

## 6. Summary

Our frontier selection is a well-tuned **geometric** policy — path cost,
momentum, information gain — with anti-thrash rules that measure the wrong
quantity. ASCENT's is a **semantic** policy — a persistent visual value map —
with anti-thrash rules that measure the right one, and no geometric terms at
all.

They looked close to complementary. They are not: **all four transplants have now
been run here and none helped.**

| borrowing | result |
|---|---|
| value map (CLIP) | null — 47 gained / 39 lost, p = 0.45 |
| cascade selection (distance as gate) | null — 48/48, p = 1.00 |
| frontier stickiness | null — 26/22, p = 0.67, revisit rate unmoved |
| coarse-to-fine LLM reasoning | **−7 SR** — 2/9, p = 0.065 |

The reading that survives all four: our selector is not missing semantics, it is
**already tuned around geometry that works**, and each transplant degrades it by
overriding path cost, momentum, or both. Everything ASCENT gains at this layer,
it gains relative to a selector that had no momentum term to lose.

That closes the exploration side. The remaining gap to ASCENT lives in perception
— 492 of 2000 episodes ending >3 m from any goal — not in where the agent decides
to go. See [ASCENT_GAP.md](ASCENT_GAP.md).
