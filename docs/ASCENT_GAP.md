# Why we score 49.8% and ASCENT scores 65.4%

Both numbers are on the same benchmark: HM3D ObjectNav v1 val, 2000 episodes,
20 scenes. Ours is measured (`outputs/v1_full2000`, SR 49.8% ±2.2, SPL 0.218);
theirs is reported in arXiv 2505.23019 (65.4% / 0.335).

This is the evidence-backed account of the 15.6-point gap. Short version: it is
**not** mostly multi-floor, and it is **not** mostly exploration. It is
overwhelmingly the agent walking confidently to the **wrong object**.

---

## 1. Where our 2000 episodes actually go

```
2000 episodes: 995 success (49.8%)
  explore-fail   371 (18.6%)   never committed to a target
  approach-fail  634 (31.7%)   committed, then stopped in the wrong place
     dtg <1 m     59  (3.0%)   near miss
     dtg 1-3 m    83  (4.2%)
     dtg >3 m    492 (24.6%)   walked to something else entirely
```

**One quarter of all episodes end more than three metres from any goal.** That
single bucket is larger than the entire gap to ASCENT. Nothing else is close.

## 2. Two different failure modes, not one

Splitting by category separates them cleanly:

| category | SR | explore-fail | approach-fail | of which >3 m |
|---|---|---|---|---|
| tv_monitor | 28.1% | 10.7% | 61.2% | **51.2%** |
| plant | 34.5% | 16.7% | 48.8% | **40.5%** |
| bed | 45.7% | 19.2% | 35.1% | 32.3% |
| sofa | 50.0% | 14.4% | 35.6% | 21.3% |
| toilet | 56.8% | **32.2%** | 11.1% | 9.0% |
| chair | 64.3% | 14.5% | 21.3% | 13.6% |

**Profile A — precision (tv_monitor, plant, bed).** Finding *something* that
looks right is easy; it is the wrong instance. `tv_monitor` fails by walking to
a wrong object in **half of all its episodes**, while failing to find anything
at all only 10.7% of the time. Better exploration cannot help these.

**Profile B — coverage (toilet).** The mirror image: 32.2% never find one, but
when they do it is almost always the real one (9.0% wrong-object). Toilets sit
behind closed bathroom doors; they are hard to reach and unambiguous once seen.

Our weakest categories are Profile A. **That is a perception-precision problem.**

## 3. What ASCENT brings to exactly that problem

| | ASCENT | ours |
|---|---|---|
| detection | D-FINE (COCO) **+** Grounding-DINO (open-set) **+** Mobile-SAM | YOLOE-11l alone |
| per-pixel semantics | RedNet (MPCAT40) | none |
| scene/room tagging | Places365 + RAM | LLM room labels only |
| frontier value | BLIP-2 ITM value map | geometric only |
| decision LLM | Qwen2.5-7B | none (LLM-free) |
| hardware | 2× RTX 3090 | 6 GB laptop profile |

Two of these bear directly on the wrong-object failure:

- **Two detectors plus segmentation.** A single open-vocab model has to be both
  the recall and the precision stage. YOLOE fires "tv_monitor" on pictures,
  mirrors, windows and monitors; with no second opinion, the object layer
  commits. ASCENT can require agreement.
- **A semantic value map.** It biases exploration toward the room the target
  usually lives in, so the *annotated* instance is more often the first one
  encountered. Ours has no such bias, so the agent commits to whichever
  same-category object it happens to meet first — which in a large house is
  frequently not the annotated one.

The verifier is not the fix: it fired **1547 rejections** across the 2000
episodes and 492 wrong-object commits still got through, because they are
**category-correct** — a real TV that simply is not the goal. Appearance cannot
separate those; only context or spatial reasoning can, and our context-prior
attempt was refuted (see §5).

## 4. Multi-floor is a small part of the gap

We took cross-floor from **0.0% → 18.2%** (411 episodes). ASCENT reports 33.3%.
Closing that entire remaining gap is worth:

```
411/2000 × (33.3% − 18.2%) ≈ +3.1 points overall
```

So even matching them exactly on the thing this project spent most of its effort
on recovers a fifth of the deficit. The multi-floor work was necessary — those
episodes were structurally unwinnable before — but it was never going to be
sufficient.

The remaining cross-floor gap has a known, specific cause: **11 of 24 cross-floor
episodes still never see a portal**, because we need to observe either another
storey's *surface* or an open-vocab stair hit, while ASCENT detects staircases
with RedNet, a trained per-pixel segmenter. A staircase across a room is far
easier to see than the floor above it.

## 5. Why our cheap substitutes did not close it

Each was implemented, measured and reverted. The pattern is informative: every
one failed for a *structural* reason, not for want of tuning.

| substitute for | what we tried | outcome |
|---|---|---|
| RedNet stair segmentation | geometric height-gradient stair detection | **refuted** — flat tread interiors fragment a staircase into disconnected riser strips; only ramps are found |
| RedNet stair segmentation | YOLOE stair masks + portals | partial — blind cross-floor episodes 14 → 11, but over-triggering broke two working episodes |
| BLIP-2 value map | CLIP ViT-B/32 value map | **refuted** — see below |
| context reasoning for instance choice | co-occurrence commitment gate | **refuted** — the agent commits before the room is mapped, so there is no context to consult |

### The value map is the most thoroughly eliminated

Every candidate explanation was tested and ruled out in turn:

| hypothesis | verdict |
|---|---|
| encoder too weak | **no** — CLIP separates target-in-view from absent at AUC 0.865; BLIP-2 is no better (0.874 contrastive, 0.783 ITM) |
| spatial attribution smeared | **fixed** — per-bearing depth clip + range falloff |
| swamped by `path_cost` | **no** — 12.6% of selections change at weight 2 |
| the changed decisions help | **no** — 47 gained / 39 lost, McNemar p = 0.45 |

And the ceiling that explains it: a counterfactual sweep over 1969 selections
shows the semantic term's influence **saturates at 24%** even at a 64× weight.
In the other **76% of selections one frontier dominates on geometry** — nearer,
or the only reachable option — and no semantic weight overturns it.

That ceiling is a property of our *selector*, not of the encoder. `utility =
score / path_cost` with a true planner cost makes distance nearly decisive.
ASCENT has **no geometric term at all**: it ranks purely by value and hands the
top-k to an LLM. Their value map can therefore steer every decision; ours can
steer at most a quarter of them.

This also retro-explains an older finding. `INVESTIGATION.md` records that LLM
frontier scoring produced byte-identical trajectories to geometric-nearest. That
was read as "the LLM adds nothing". The truer reading is that **our objective
leaves almost no room for any semantic prior**, symbolic or visual. Same
ceiling, two different signals.

## 6. Honest accounting of what is and is not comparable

**Comparable:** the benchmark, split, episode count and success criterion are
identical. The 15.6-point gap is real and not a sampling artefact (our 95% CI is
±2.2).

**Not comparable:** compute and model budget. ASCENT runs six models on dual
3090s; we run one detector plus a hosted VLM verifier, targeting a 6 GB laptop.
A fair reading is not "our method is worse by 15.6 points" but "this much
perception is worth about that much SR on this benchmark".

**Also worth stating:** we never reproduced ASCENT's number ourselves. Their
65.4% is taken from the paper.

## 7. What would actually close it, in order of measured leverage

1. **Detector precision (~+12 points available).** The 492 wrong-object commits
   are the whole ballgame. A second detector or a segmentation cross-check on
   the committed instance attacks it directly; `tv_monitor` alone is 281
   episodes at 28.1%.
2. **Rework the objective so a semantic prior can act (unlocks the value map).**
   Not a bigger weight — the 24% ceiling is structural. Either drop `path_cost`
   from the ranking and re-introduce distance as a separate gate, or adopt
   ASCENT's value-sort-then-choose shape.
3. **Semantic stair segmentation (~+3 points).** Closes most of the remaining
   cross-floor gap. RedNet is small and the failure mode is understood.
4. **Toilet-style coverage.** 32.2% explore-fail on the second-best category
   suggests a specific, tractable "closed door / small room" problem.

Items 1 and 2 are where the gap lives. Item 3 is the one this project already
invested in, and is now the smallest of the three.
