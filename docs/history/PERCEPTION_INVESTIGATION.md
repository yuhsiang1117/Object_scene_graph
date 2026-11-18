# Why our detector misses the benchmark's objects

Measured overnight 2026-09-08 against the released DualMap benchmark, on the
ground-truth keyframe dump written by `eval.gt_dump_dir` during the tight-ring
run -- the frames on which our agent genuinely had the object in view and did not
name it. Every number here is offline: the run had already paid for the
inference, so each hypothesis costs a minute rather than a 500-step episode.
The per-frame diagnosis of those misses -- what covered the object's pixel, which
class won them -- is `scripts/legacy/analyze_gt_dump.py`, written to
`outputs/osg_dualmap_tightring/GT_DUMP.md`.

Of 77 dynamic failures on the released benchmark, 44 never produced a detection.
But those 44 are two different problems and only one of them is perception:

| | in-anchor (54) | cross-anchor (53) |
|---|---:|---:|
| succeeded | 18 | 12 |
| **seen, never named** -- the detector | 10 | 15 |
| **never in view** -- search coverage | 6 | **13** |
| admit / commit / approach | 20 | 13 |
| ceiling if only the detector is a wall | **81.5%** | **71.7%** |

"Never in view" is an episode that never pointed the camera at the object. That
is a coverage failure belonging to the search, and it is movable -- the
camera-height arm below moves it 5 -> 2 while leaving recall untouched. Counting
it as a perception limit understates the ceiling badly: doing so gives 70.4% and
47.2%, and the honest numbers are 81.5% and 71.7%.

So the detector is a real wall, but not the only thing between here and a 70/50
target. Cross-anchor's largest single addressable bucket is coverage, not
naming.

## Three hypotheses, two of them dead

`scripts/legacy/probe_nearmiss.py` re-runs the detector over the dumped frames under
four conditions. Over four targets and roughly 200 frames:

| Target | frames | base | all 13 competing classes removed | zoomed to the object |
|---|---:|---:|---:|---:|
| tin can | 82 | 0.18 | **0.18** | **0.38** |
| cracker box | 35 | 0.20 | **0.20** | 0.11 |
| red plate | 35 | 0.49 | **0.49** | 0.20 |
| blue plastic pitcher | 41 | 0.63 | **0.63** | 0.24 |

**Class competition is not the problem.** Removing every competing class --
`sofa`, `bed`, `lamp`, `picture`, `desk`, `table` and eight more -- changes recall
by exactly zero on all four targets. This refutes the reading of `top_labels`
that suggested it: those labels record what else fired on the object's pixel, not
that it suppressed the target. `lamp` scoring 0.91 over a plate is a
co-occurrence, not a cause. (The earlier `box`/`book` finding was measured
differently and is not contradicted here; what is refuted is extending it to the
furniture classes.)

**The name is not the problem either.** Nine alternative names for the soup can
moved it from 15/82 to 18/82 (`red can`), and `tin can` was already the best name
on 65 of the 82 frames. `red plate` was the best of nine on 23 of 35. The
vocabulary work recorded in `ARCHITECTURE.md` has already taken what there was.

**Apparent size is a problem, but only for the smallest object.** Cropping to the
object and upscaling doubles the tin can (0.18 -> 0.38) and *hurts* every other
target, badly. So tiled or sliced inference would buy one object and cost three.

**Nor is the admission gate.** Sweeping it on the same frames, recall flatlines
below 0.20 -- the detector scores exactly 0.0 on the frames it misses, not
0.18-just-under-threshold:

| Target | gate 0.30 | 0.25 | 0.20 | 0.15 | 0.10 | 0.05 |
|---|---:|---:|---:|---:|---:|---:|
| tin can | 0.15 | 0.16 | 0.18 | 0.18 | 0.18 | 0.18 |
| cracker box | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| red plate | 0.46 | 0.49 | 0.49 | 0.49 | 0.49 | 0.49 |
| blue plastic pitcher | 0.56 | 0.59 | 0.63 | 0.63 | 0.63 | 0.63 |

A miss is not a near miss. Nothing is there to admit, which is the same thing the
"no detection mask at all" count says and rules out any threshold change.

## It is not the detector model either

DualMap's own detector -- `yolov8l-world.pt` against its 305-class list, both
from the vendored checkout the benchmark ran from -- over the same frames
(`scripts/legacy/detector_bakeoff.py`):

| Target | frames | ours (YOLOE) | DualMap (yolov8l-world) |
|---|---:|---:|---:|
| bowl | 90 | 0.88 | 0.92 |
| blue plastic pitcher | 41 | **0.63** | 0.00 |
| red plate | 35 | **0.49** | 0.26 |
| tin can | 82 | 0.18 | **0.28** |
| scissors | 69 | 0.11 | **0.06** |

We are better on three of five, decisively on the pitcher. **Their detector fails
scissors on these frames as thoroughly as ours does** -- 0.06 against our 0.11 --
so DualMap's 78% in-anchor scissors score on the benchmark is not superior
detection. It comes from the anchor-only successes documented in
`DUALMAP_OFFICIAL_RERUN.md` (all six 00848 scissors trials stop at `bed` having
planned no local path) and from in-anchor's degenerate 0.70 m displacement.

Not a like-for-like architecture comparison, and not meant as one: each system is
given the class name its own pipeline uses. It is the question that decides where
the work goes, and the answer is that swapping detectors would lose more than it
gains.

## What is left: we are looking at these objects edge-on

| | |
|---|---:|
| median height of the benchmark's target objects | **0.84 m** (p10 0.60, p90 1.01) |
| our camera height (`agent.camera_height`) | **0.88 m** |
| DualMap's camera height | **1.50 m** |

Our camera sits four centimetres above the objects it is looking for. A cup on a
table, viewed from a camera at the height of the table top, presents almost no
projected area and is frequently occluded by the near edge of the table itself.
DualMap views the same object from 0.66 m above it and sees the whole surface.

This is consistent with everything above that the other hypotheses could not
explain: that 45% of missed frames have *no detection mask at all* over the
object rather than a competing one; that in-situ recall (0.36) is so far below
recall at authored viewpoints (0.58), which were sampled to see the object; and
that zooming helps only the object that is genuinely small rather than the ones
that are merely edge-on.

### What the dump says before the arm reports

Over 426 in-view YCB keyframes, recall against range is not monotonic:

| Range | Frames | Recall |
|---|---:|---:|
| 0-1.5 m | 88 | 0.50 |
| 1.5-2.5 m | 100 | **0.58** |
| 2.5-4 m | 134 | 0.28 |
| 4 m+ | 104 | 0.20 |

Range explains the bulk of it -- past 2.5 m recall halves -- but the dip *inside*
1.5 m is the interesting part. Apparent size increases monotonically as the agent
approaches, so a pure size explanation predicts recall does too. A grazing
viewing angle does not: the closer a camera at 0.88 m gets to an object at
0.84 m, the more it is looking at that object's silhouette edge-on against
whatever stands behind it, and the more of it falls outside a 63-degree vertical
field of view.

Occlusion is a second, smaller effect: below a visible fraction of 0.95 recall
collapses to 0.05-0.07, but only 92 of 426 frames are in that state, and the
median missed frame is fully visible.

It is also the one asymmetry the comparison never resolved --
`DUALMAP_OFFICIAL_RERUN.md` lists the two camera heights under "what a paired
comparison cannot share" and leaves it there.

## The camera height does not explain it either

Measured: 18 paired 00829 in-anchor trials at `agent.camera_height=1.5`, against
the same trials at 0.88 m.

| | 0.88 m | 1.50 m |
|---|---:|---:|
| in-view keyframes | 314 | 329 |
| detected | 171 | 183 |
| **in-situ recall** | **0.545** | **0.556** |
| admitted to the object layer | 128 | 111 |
| SR | 7/18 | 9/18 |
| episodes that never saw the object | 5 | **2** |

Recall does not move. The hypothesis predicted a large gain and delivered one
percentage point, so raising the camera is not the fix for perception.

One real effect survives, and it is a different one: the higher camera puts the
object in frame more often -- three fewer episodes never saw their target at all.
That is a *coverage* gain belonging to the search, not a recall gain belonging to
the detector, and on 18 episodes the SR difference it produces (7 -> 9) is inside
the noise.

**Caveat on the strength of this test.** The two arms do not see the same frames:
changing the camera changes what the agent detects, hence where it goes, hence
which keyframes exist. Per-trial in-view counts diverge wildly as a result -- 31
frames against 1 on `0116 plate`. Only the aggregate over ~320 frames per arm is
worth reading, and a sharper test would replay a fixed trajectory at both
heights. The aggregate is flat, and the effect predicted was not subtle, so this
is enough to stop pursuing it.

## Where this leaves perception

Every cheap lever is now measured and closed: class competition, the target's
name, the admission gate, zoom, the detector model, and the camera height. The
in-situ recall of roughly 0.36-0.55 is close to what this detector does on this
data, and the released benchmark's perception ceiling -- 70.4% in-anchor, 47.2%
cross-anchor -- should be treated as real rather than as something a
configuration change will lift.

What this does *not* mean is that the SR targets are out of reach. With the
detector treated as fixed, the ceilings are 81.5% and 71.7%, and hitting 70/50
needs 19 of the 26 non-detector failures in-anchor and 14 of 26 cross-anchor.
The work is in the other three buckets:

* **coverage** (6 / 13 trials) -- the largest cross-anchor bucket. 29 of 53
  cross-anchor episodes exhaust the 500-step budget at a median 46.2 m travelled
  against DualMap's 18.0 m, so the agent is not short of capability here, it is
  short of time.
* **wasted trips** -- `goal_commit_log` shows a median of 3 commits per episode,
  79% of cross-anchor commits more than 1 m from the object, and 24 of 44
  episodes where *every* commit was wrong. Attributing each episode's APPROACH
  steps pro-rata across its commits:

  | | steps | in APPROACH | toward a WRONG goal |
  |---|---:|---:|---:|
  | in-anchor | 16831 | 6794 (40%) | 3847 = **23% of the budget** |
  | cross-anchor | 20710 | 6360 (31%) | 4976 = **24% of the budget** |

  A quarter of every episode is spent walking to something that is not the
  target, and it is the same budget the coverage bucket is short of. Note this
  is not simply a bug to remove: checking a remembered location is the correct
  opening move, and in-anchor it is right 47% of the time. The asymmetry is that
  cross-anchor pays the same cost for a prior that is almost always stale, so
  the lever is *when to stop* working down the list, not whether to start.

  **The obvious existing knob is the wrong one.**
  `scene_graph.presence.max_identity_rejections` (2) caps repeated visits to the
  SAME track, and that is not the failure here: the episodes visit three
  DIFFERENT wrong tracks, each once or twice. Lowering it to 1 would also walk
  into a trap the config already documents -- "one arrival can end on a bad
  heading or a consumed path, two is a decision", with a measured episode that
  hit the cap and then spent 350 steps not going to an object it had mapped at
  0.04 m from truth. `path_consumed` is the dominant stop reason in our
  failures, so a one-strike rule would retire correct tracks for navigation
  reasons. Whatever caps the list has to count *distinct* failed tracks, and
  that is new code rather than a setting.
* **the terminal approach** (15 trials), which the tight-ring arm addresses.

