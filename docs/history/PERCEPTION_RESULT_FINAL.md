# The proposal stage, with the absence sensor fixed

21 trials of `data/splits/dualmap_perception.json` -- every `_sensor_v2` failure
whose target was in view and never named. The baseline scores 0 on all of them
by construction.

| query | n | baseline | +region | +absence fix |
|---|---:|---:|---:|---:|
| soup can | 3 | 0 | 3 | **3** |
| scissors | 9 | 0 | 1 | **1** |
| pitcher | 2 | 0 | 1 | **1** |
| banana | 3 | 0 | 0 | **1** |
| cracker box | 2 | 0 | 0 | 0 |
| mug | 2 | 0 | 0 | 0 |
| **all** | **21** | **0** | **5** | **6** |

Paired: +6/-0 against the baseline, +1/-0 against the region arm.

The target is now named on **8 of 21** where the baseline named it on **0**.

## The absence fix is worth more than its one trial

`_best_target_detection` re-runs the raw detector and filters by label, and the
close look asks it before declaring a committed track absent. For these objects
that is circular -- the detector that could not name the thing is asked whether
the thing is there. Letting it fall through to the proposal stage:

| | baseline | +region | +absence fix |
|---|---:|---:|---:|
| silent close looks (a look that saw nothing) | 13 | 25 | **8** |
| goal commits | 19 | 45 | **37** |
| ended at the step budget | 20 | 14 | **10** |
| median steps | 500 | 500 | **464** |
| attempts spent | 18 | 28 | 36 |

The region arm made the false-absence problem WORSE before this fix -- silent
looks 13 -> 25 -- because it created tracks the label path then could not
confirm, so the agent walked to more objects and disbelieved more of them. With
the fix they fall to 8, below the baseline, and commits drop 45 -> 37: fewer
wasted trips.

`cross_anchor__0128-2__banana` is the shape of it. Baseline: 500 steps, 3.10 m,
never named. Region arm: 500 steps, 2.35 m, named 10 times, still 0. With the
fix: **201 steps, 0.59 m, scored** -- no close look ran at all, the target was
detected on arrival, and the approach ended in an ordinary `path_consumed` stop.
The false absence had been costing ~200 steps AND pushing the good commit past
the point the clock could absorb, which is why the deadline looked like a
separate bug.

## What is left in this bucket

* **scissors 1/9.** The wall, and it is the encoder rather than the proposer:
  FastSAM puts a tight region on the scissors on 72% of frames and MobileCLIP
  scores it 0.149, below its own distractors.
* **mug 0/2, cracker box 0/2.** Admitted regions but no conversion; worth a
  per-trial read before assuming they are the same failure.
* attempts spent rose 18 -> 36. The stage finds more, commits to more and
  therefore spends more. On a 3-attempt protocol that is a real risk elsewhere
  in the 107 and is the reason the full run still has to be the reported number.

## Caveat

This subset is selected for the failure the stage addresses, so it flatters it
by construction. An earlier arm read 26/44 on a selected subset and 19/44 on the
full 107. Nothing here is a benchmark number until the full run says so.
