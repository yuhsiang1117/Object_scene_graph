# Feature memory: measured, and why it is not the fix for scissors and mug

DualMap ranks mapped objects and live detections by the cosine between the
query's CLIP **text** feature and each object's CLIP **image** feature; the
detector's label never enters the ranking. We match by label string in four
places, so the plan was to give every stored crop a feature and ask the same
question we ask of a name.

The probe is `scripts/legacy/probe_feature_memory.py`; reports in
`outputs/feature_memory_probe/`. Two parts: the prior map (would the agent GO to
the right surface) and the dumped keyframes (would it RECOGNISE the object once
there). Both backbones: OpenAI CLIP ViT-B/32, the checkpoint already staged for
the value map, and DualMap's own MobileCLIP-S2.

**Result: every gate fails, and the reason for the headline case is not the
matcher.** Details below, in the order they were found.

## 1. The premise was wrong: there is no mis-named scissors in the map

The plan rested on a table showing a `bleach bottle` track 0.09-0.10 m from the
scissors in all three scenes, read as "the mapping pass saw the scissors and
called it something else". It did not. That track is a real bleach cleanser.

`outputs/maps_v5` -- the prior map every DualMap-benchmark arm loads, including
the shipped `island_close` run -- is built by `scripts/campaign/rebuild_maps.sh`,
which passes `ycb.layout_root=outputs/substituted_layouts`. Those layouts are the
released ones with `037_scissors` **replaced by `021_bleach_cleanser` at the same
pose**, authored by `scripts/author_substitute_layout.py` because neither
detector can name the scissors asset (`docs/history/ASSET_SUBSTITUTION_EVIDENCE.md`).
Diffed against the release, that substitution is the only change, plus a dropped
second mug in 00848:

| scene | released | substituted, same position |
|---|---|---|
| 00829 | `037_scissors` (4.70, 1.12, -4.84) | `021_bleach_cleanser` (4.70, 1.22, -4.84) |
| 00848 | `037_scissors` (-1.99, 0.71, -1.11) | `021_bleach_cleanser` (-1.99, 0.83, -1.11) |
| 00880 | `037_scissors` (3.56, 0.90, -0.92) | `021_bleach_cleanser` (3.56, 1.01, -0.92) |

The stored crops confirm it by eye: the track at 0.01 m from each scene's
scissors is a photograph of a cream bottle with a printed label, the same object
in all three houses, at detector confidence 0.62 / 0.78 / 0.82. The released
layouts place no bleach cleanser anywhere, and no authored layout puts one at
those coordinates.

So the DualMap-benchmark episodes inject the released layout and are scored
against the real scissors, while the prior map they start from was built in a
world where that object is a different one. The map is right about the *place*
and wrong about the *object*. No matcher can recover a scissors that was never
in the room when the map was made, and CLIP is correct to score that crop 0.11
for "scissors".

**This is a map-provenance bug, not a feature-memory finding, and it is worth
fixing on its own**: rebuild the released-benchmark maps from the released
layouts and rerun. It is unlikely to be a large SR change -- the phantom track
carries the wrong label, so the label-keyed candidate path already ignores it --
but every scissors number on this benchmark currently describes a map of a
different scene.

## 2. Part A: the map. Controls pass, the hard cases fail

Rank of the true object's nearest track among every track in the map, by cosine
against "a photo of a &lt;query&gt;". MobileCLIP-S2, the better of the two:

| scene | query | nearest track's label | distance | rank |
|---|---|---|---:|---:|
| 00829 | soup can | tin can | 0.00 m | **1** / 689 |
| 00829 | bowl | bowl | 0.01 m | **1** / 689 |
| 00829 | pitcher | blue plastic pitcher | 0.00 m | **1** / 689 |
| 00829 | cracker box | cracker box | 0.03 m | **1** / 689 |
| 00848 | banana | banana | 0.06 m | **1** / 343 |
| 00880 | pitcher | **mug** | 0.04 m | **1** / 520 |
| 00880 | soup can | tin can | 0.04 m | **1** / 520 |
| 00880 | cracker box | **picture** | 0.04 m | **3** / 520 |
| 00848 | plate | red plate | 0.04 m | 2 / 343 |
| 00848 | scissors | bleach cleanser | 0.01 m | 143 / 343 |
| 00829 | scissors | bleach cleanser | 0.01 m | 359 / 689 |
| 00880 | scissors | bleach cleanser | 0.00 m | 440 / 520 |
| 00848 | mug | curtain | 0.04 m | 105 / 343 |

Named targets reach the top five in 11 of 14 (gate A3 PASS on MobileCLIP-S2, and
FAIL on ViT-B/32 at 6 of 14 -- DualMap's backbone is clearly the right one, and
the 128 px stored crops are good enough for it). Two rows are the mechanism doing
exactly what it is for: 00880's pitcher and cracker box are ranked 1st and 3rd of
520 through tracks our detector labelled `mug` and `picture`, which the label
path discards outright.

The two hard queries rank 105-440. Gates A1 (true track top-5 on at least 3 of 4
pairs) and A2 (true container top-3, fused never worse than the baseline order)
both FAIL. Note also that the 00848 mug does have a track 0.04 m away, labelled
`curtain` -- correcting the earlier note that nothing was mapped within 0.46 m.
It is a 24x128 sliver, and it does not read as a mug.

A second measurement, on the 28 static queries our 47-class vocabulary cannot
name at all (`ottoman`, `washbasin counter`, `printer`, ...), where retrieval
should be worth the most: 2 of 28 in the top-5. A few are excellent (`printer`
1 / 520 with the only positive margin, `ironing board` 3 / 520,
`washbasin counter` 11 / 689) and most rank in the hundreds. That number is a
lower bound -- for small classes the nearest track to a ground-truth box is often
some other object inside it, so a bad rank there can mean "never mapped" rather
than "features failed" -- but it does not support the mechanism either.

### Along the way: the support relation binds almost nothing

DualMap's anchor inherits its neighbours' features. Our equivalent is
`container_id`, and over the three maps only **25 / 689, 8 / 343 and 20 / 520**
objects are bound to a surface: the support test wants the object's base within
0.15 m of the surface top and inside its footprint, and an ellipsoid fitted from
a few distant views misses that band more often than the object misses the table.
Two of the four target surfaces have no bound children at all. A strict
DualMap-parity child set would score them at the floor, so
`container_feature_scores` takes an optional radius that widens the child set by
proximity; both variants are reported.

## 3. Part B: the live frames. Ranking helps, admission does not

Every region the live detector proposes on the dumped keyframes, scored against
the query text, MobileCLIP-S2. "named" is the label baseline on the same frames.

| query | band | frames | true region found | top-1 | top-5 | named |
|---|---|---:|---:|---:|---:|---:|
| mug | 0-1 m | 3 | 66.7% | 33.3% | 66.7% | 0.0% |
| mug | 2-3 m | 70 | 85.7% | 38.6% | 42.9% | 0.0% |
| mug | all | 160 | 52.5% | 18.1% | 30.0% | **0.0%** |
| scissors | 0-1 m | 19 | 68.4% | 42.1% | 63.2% | **47.4%** |
| scissors | 1-2 m | 61 | 63.9% | 9.8% | 32.8% | 1.6% |
| scissors | all | 352 | 83.0% | 11.9% | 39.5% | 2.8% |

The mug is the encouraging half: the detector names it on 0 of 160 frames and
the feature ranks it first on 18%, 39% in the 2-3 m band. This is the same
effect DualMap's mug result comes from.

It does not survive contact with a threshold, which is what admission needs.
Sweeping tau against frames of other queries in the same scene, the first tau
whose false-admission rate is under 10%:

| query | tau | recall within 2 m | false admit |
|---|---:|---:|---:|
| mug | 0.21 | **0.0%** | 6.5% |
| scissors | 0.20 | 11.5% | 5.6% |

Gate B1 (top-1 at least 25% within 2 m) fails for both (22.2% and 17.5%), and
gate B3 finds no usable threshold. Ranking within a frame is informative;
the absolute cosine of the true region is not separable from the best distractor
across frames, which is exactly what an admission rule has to do. For the
scissors the feature channel is also *worse* than the label under a metre, where
the detector already names it on 47% of frames.

## 3b. Re-measured on the corrected map, and the answer gets cleaner

The probe above ran against `maps_v5`, so its scissors verdict was measured on a
map of the substituted world. Rebuilding from the release (`outputs/maps_released`,
built from `outputs/collector_layouts`) and re-running Part A on MobileCLIP-S2:

| scene | nearest track to the scissors | distance | feature rank | text-only rank |
|---|---|---:|---:|---:|
| 00829 | **`scissors`** | 0.01 m | 15 / 682 | **1 / 38** |
| 00848 | `bed` | 0.25 m | 779 / 800 | 19 / 42 |
| 00880 | `desk` | 0.23 m | 302 / 618 | 10 / 43 |

00829 improves enormously -- 359 / 689 to 15 / 682 -- because the map now holds a
real scissors track instead of a bottle. It still does not pass, and the last
column is why it does not matter: the track's label is `scissors`, so the
existing label match already ranks it first. Feature memory adds nothing where
the object is mapped under the right name.

In the other two scenes the detector never named the scissors during mapping, so
there is no scissors track at all and the nearest thing is the furniture it rests
on. There is nothing to retrieve, by feature or by label.

So on this benchmark the scissors has no regime where appearance matching is the
lever: mapped under the right name, the label path wins; not mapped, neither
works. The 00848 mug is the same story -- its nearest track is now a `dresser`
0.08 m away, ranked 605 / 800. Controls hold up at 10 / 15 top-5, so the
mechanism is intact; it simply has no work to do here.

## 3c. How DualMap does it, and why this probe was not a fair test of it

Measured from `outputs/dualmap_official_bench/seed{12,13,14}` (its own trial
records and per-keyframe traces) and from its source in
`outputs/dualmap_comparison/vendor/DualMap`.

**It works, on the mug, completely.** Over three seeds the mug is 20/21: 9/9
in-anchor, 8/9 cross-anchor, 3/3 static. Twenty of the twenty-one planned a
local path -- it genuinely re-detected the object and navigated to it -- and it
stopped 0.33-0.94 m away. Its detector never names the mug (0.06 recall; it
calls the blob "speaker"), so none of that comes from naming. The scissors is
much weaker: 19/27 in-anchor, 5/27 cross-anchor, and the anchor it commits to is
`bed` every time.

**The mechanism is two stages, and neither uses a threshold.** Globally it takes
the cosine of the query text against every prior-map anchor's own feature and
the features of what rested on it (`related_objs`), picks the best, and walks
there. Locally, on arrival, `filter_objects_in_global_bbox` narrows its live map
to the objects inside that anchor's bounding box and
`find_best_candidate_with_inquiry` returns `sorted_candidates[0]` -- the plain
argmax. There is no admission bar to clear, so the matcher cannot fail to
admit; it always names a best guess and goes.

**What it matches are persistent objects, not frames.** A candidate needs
`observed_num > 2`, and its feature is the renormalised running mean over every
detection of it (`utils/object.py:367-375`), optionally 0.7 image + 0.3 the text
of the detected label. Averaging over views denoises a feature that is very
noisy on one 2 m crop.

Against that, gate B3 above asked the wrong question. It looked for an absolute
cosine that would admit the mug in a *single frame* without admitting any of
~50 co-present regions, and there is none. DualMap never asks for one. It ranks
a handful of accumulated 3D objects near a surface it has already reached and
takes the top one. Three things differ at once -- multi-view averaging instead
of one crop, a spatially scoped competitor set instead of the whole frame, and
argmax instead of a bar -- and every one of them makes its problem easier than
the one measured here.

So the honest scope of this report's negative result is narrower than section 3
implies: **threshold-based admission on single-frame crops does not work**, and
that is measured. **Argmax ranking of accumulated tracks near a committed
surface is untested**, and it is what DualMap actually does. Part B's own numbers
are consistent with it being worth testing: some region covers the mug's pixel on
85.7% of frames in the 2-3 m band and the feature already ranks the true one
first on 38.6% of them, against ~50 competitors rather than the few that would
survive the spatial scoping.

The prior-map half of section 3b still stands on its own: for 00848 and 00880
our map contains no scissors track under any name, so nothing global can retrieve
it, and DualMap's own scissors numbers (5/27 cross-anchor) say its global stage
does not solve that either.

## 4. What was built, and what was not

Kept, because the measurement should be repeatable:

* `src/osg/perception/feature_encoder.py` -- one encoder for both backbones,
  shared by the probe and any future runtime so an offline threshold means the
  same thing online. Deliberately not part of `image_text.ClipScorer`, whose
  non-None value silently switches the value map on.
* `src/osg/objects/feature_memory.py` -- the fusion arithmetic
  (`feature_term`, `container_feature_scores`, `merge_running_mean`, `admits`)
  and a `FeatureMemory` runtime holder with its counters.
* `build_container_candidates(..., feature_scores=, feature_beta=, feature_floor=)`
  -- inert by default and tested to be byte-identical when unused.
* `scripts/legacy/probe_feature_memory.py`, `tests/unit/test_feature_memory.py`.
* `scripts/legacy/rank_search_surfaces.py:load_scene_graph` repaired: its shim predated
  the floor-stack rewrite and raised `AttributeError` on every map, so the
  offline surface ranker could not run at all. It now restores through a real
  `FloorStack` and prefers the snapshot's own room labels over a fresh Voronoi
  pass.

Not built, because the gates decide it: the container prior by feature, feature
admission of candidates, and the same-class fallback are **not** wired into the
agent, and no `_feat` preset or benchmark arm was run. The config group,
`ObjectLayer` hook, `_best_target_detection` clause and the presets in the plan
are unwritten; nothing in the shipped path changed, and `pytest tests/unit`
is green at 915 tests.

## 5. What would be worth doing

1. **Rebuild the released-benchmark prior maps from the released layouts.** The
   current maps describe a substituted world. This is a correctness fix
   independent of everything above.
2. **If feature memory is revisited, aim it where it measured well**: objects
   that are mapped but mislabelled (00880's pitcher and cracker box), and the
   static split, where 39 of 79 queries name classes our vocabulary does not
   contain and we score 30.8% against DualMap's 67.1%. Use MobileCLIP-S2. Use it
   to RANK candidates the map already holds, not to admit new ones on an absolute
   threshold -- that is the half that measured.
3. **Do not expect it to move scissors or mug.** The scissors is absent from the
   map by construction and unreadable to CLIP when present; the mug is findable
   by rank and not by threshold.

Commands:

```bash
HF_HUB_OFFLINE=1 python scripts/legacy/probe_feature_memory.py --part A --backbone both \
    --prompts "a photo of a {q}" "{q}" --save-crops --out outputs/feature_memory_probe
HF_HUB_OFFLINE=1 python scripts/legacy/probe_feature_memory.py --part B --backbone mobileclip_s2 \
    --regions yoloe --out outputs/feature_memory_probe/partB_yoloe
```
