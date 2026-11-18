# What can tell the target from what the detector mistakes for it

*2026-10-15. One labelled set, six recognisers. The answer to the review's "evaluate what the
VLM actually does", and the choice behind `verification.stop_gate`.*

## Why it was asked again

Most goal commits on the YCB benchmarks are wrong: 31 of 40 on the multi-floor set
(`outputs/RERUN/mf5`), 49 of 73 live commits on DualMap's static condition
(`outputs/RERUN_UPLOADMAP/osg_unified`). A false positive is a real object, so the detector
re-detects it on arrival, the absence sensor has nothing to say, and the STOP spends an
attempt. `docs/history/INVESTIGATION.md` closed verification as a dead end -- five variants, no
SR effect -- but on HM3D ObjectNav, where the false positives were real objects *of the right
category*. On the YCB benchmarks they are a ceiling fixture called "toy airplane", a book
called "cracker box", the soup can called "coffee can".

## The set

`scripts/build_verifier_set.py +experiment=mfv2_osg +probe.random=150 +probe.near=6`: the five
multi-floor scenes' static layouts; frames from the layout's own goal view points around every
object (0.8-2 m) and from 150 random navigable poses per scene; for every frame and every YCB
label, YOLOE exactly as pass 2 runs it for that query (11L, imgsz 1280, the arm's thresholds,
`target_vocabulary`). Each detection of the label is a clean crop, labelled TRUE when its mask
overlaps the object's semantic mask (IoU > 0.1), with its range from depth.

539 detections: 309 on the target, 230 false positives -- 71 of them on another benchmark
object (mostly the soup can and the coffee can for each other), 159 on the scene.

## Result

`scripts/eval_verifiers.py outputs/verifier_set --vlm-n 80 --workers 4` with three VLMs. At the
range a STOP is taken from, <= 1.5 m (250 targets, 131 false positives; VLM rows on a balanced
subsample of ~72 + ~63). r = P(accept | the target), q = P(accept | a false positive):

| recogniser | r | q | q, FP on scene | q, FP on another YCB object | cost |
|---|---:|---:|---:|---:|---|
| detector score (what the agent uses now) | 1.00 | 1.00 (AUC 0.73) | | | -- |
| MobileCLIP-S2 margin > 0, **generic** confusers | **0.95** | **0.46** | | | ms, local |
| MobileCLIP-S2 margin > 0, the arm's detector vocabulary | 0.90 (AUC 0.86) | 0.31 | 0.28 | 0.33 | ms, local |
| MobileCLIP-S2 margin > 0, + every benchmark query | 0.90 (AUC 0.92) | 0.24 | 0.26 | 0.21 | ms, local |
| llama-3.2-11b-vision, yes/no (the pipeline's verifier) | 0.75 | 0.56 | 0.68 | 0.44 | 5.3 s median, 27.6 s p90 |
| llama-3.2-11b-vision, forced choice | 0.56 | 0.23 | 0.37 | 0.09 | 3.9 s median |
| llama-3.2-90b-vision, yes/no | 0.75 | 0.27 | 0.29 | 0.25 | 47 s median, 211 s p90 |
| llama-3.2-90b-vision, forced choice | 0.76 | 0.32 | 0.39 | 0.26 | 44 s median |
| nemotron-3.5-super-vl | -- | -- | | | 60% of calls errored |

The margin over four confuser lists (the same script's sweep, `--no-vlm` reproduces it in a
minute), with the target class it keeps least often:

| confusers | margin -0.01 | margin 0 | margin +0.01 |
|---|---|---|---|
| `clip_indoor`: indoor vocabulary only (41) | r 1.00 q 0.69 | r 0.99 q 0.63 (coffee can 18/19) | r 0.97 q 0.59 |
| **`clip_generic`: indoor + COCO-80** | r 0.97 q 0.50 | **r 0.95 q 0.46** (coffee can 16/19) | r 0.93 q 0.43 |
| `clip_bg`: the arm's vocabulary (47) | r 0.96 q 0.38 | r 0.90 q 0.31 (coffee can 9/19) | r 0.85 q 0.28 |
| `clip_all`: the arm's + every benchmark query | r 0.96 q 0.34 | r 0.90 q 0.24 (coffee can 9/19) | r 0.82 q 0.15 |

## What it says

* **The pipeline's VLM barely separates them.** llama-3.2-11b accepts 56% of false positives
  and rejects a quarter of true targets, at seconds per call. As an absence sensor it is
  invoked 5 times in 53 episodes (PAPER.md Table III), which is why removing it changed
  nothing; as an identity check it would not have helped much either.
* **A local image-text margin does better than every VLM tried**, in milliseconds.
* **Most of the separation comes from naming other small objects.** Over the indoor
  vocabulary alone the margin rejects 37% of false positives; adding COCO's 80 names, 54%;
  adding the benchmark's own object names, 76%. The gate ships the middle one: fixed,
  published, and naming no benchmark object.
* The arm's detector vocabulary already names eight of the benchmark's objects (DualMap's
  seven targets and the bleach bottle) but not the three multi-floor-only ones. Used as
  confusers it rejects true coffee cans 10 times of 19, because "tin can" is in it and
  "coffee can" is not. That asymmetry is a zero-shot finding in its own right
  (`docs/ZERO_SHOT_AUDIT.md`).

## Caveats

* The set is from **static** layouts and view points the manifest chose to see each object;
  arrivals in a real episode come from wherever the approach ends. Pass-2 crops are
  `stop_gate.crop_with_margin` too, so the recogniser sees the same thing, but the pose
  distribution differs. The online A/B (`mfv2_osg` vs `mfv2_osg_gate`) is the number.
* One scene's objects cluster on one cabinet (00878), so near views of one object often
  contain others; that is where most of the "FP on another YCB object" rows come from.
* VLM rows are subsampled and their errors (failed calls) are excluded, not counted as "no".
