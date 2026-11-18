# The class-agnostic proposal stage, measured

`+experiment=archive/dualmap_protocol_osg_sensor_v3_region` on the 21 trials of
`data/splits/dualmap_perception.json` -- every `_sensor_v2` failure whose target
was in view and never named, which is the only set the stage can fire usefully
on. The baseline scores 0 on all of them by construction.

## Result

| query | n | baseline | +region | admits | target now named |
|---|---:|---:|---:|---:|---:|
| soup can | 2 | 0 | **2** | 32 | 2/2 |
| scissors | 9 | 0 | **1** | 18 | 1/9 |
| pitcher | 2 | 0 | **1** | 6 | 1/2 |
| banana | 3 | 0 | 0 | 22 | **3/3** |
| cracker box | 2 | 0 | 0 | 23 | 0/2 |
| mug | 2 | 0 | 0 | 72 | 0/2 |
| **all** | **20** | **0** | **4** | 173 | **7/20** |

Paired +4/-0.

The mechanism reads clean. These trials are in the bucket precisely because the
target was never named: the baseline names it on **0 of 20**. With the stage it
is named on **7 of 20**. Objects the detector called `mirror` and `lamp` are
being recognised.

The threshold holds. 173 admissions against 7403 keyframes whose best region
was below tau -- a **2.3%** admission rate, and the per-episode cap never binds.
Flooding the map with wrong tracks was the main risk of this design and it did
not happen.

## What it cost

| | baseline | +region |
|---|---:|---:|
| goal commits | 18 | 42 |
| attempts spent | 17 | 25 |
| median best stop | 5.44 m | 4.76 m |
| ended at the step budget | 19 | 14 |

More commits and more attempts, which is the price of admitting anything at 85%
precision. Against it, the agent ends nearer the object and runs out of budget
less often.

## Two findings worth keeping

**Episode-level conversion beats the per-frame rate.** The probe measured
scissors at 3% rank-1 per frame and predicted ~0 of 9; it converted 1. An
episode offers hundreds of frames and one good commit is enough. The offline
per-frame number is a lower bound on what the live stage does, not an estimate
of it.

**Banana moved bucket rather than converting.** Named on 3 of 3, scored on 0 of
3. The stage found it every time and the trials still failed, so those three are
now navigation- or reachability-bound rather than perception-bound. That is a
transfer, not a loss, and it is only visible by running it.

## Against the estimate

`docs/history/PERCEPTION_GAP.md` predicted "about five trials, not twenty-one", weighted
from the measured rank-1 rates: cracker box ~2, soup can ~2, banana ~1,
scissors ~0. The outturn is 4, with the composition different from the
prediction -- soup can delivered both, scissors delivered one it was not
supposed to, cracker box delivered neither. The headline number was right for
roughly the wrong reasons, which is worth saying plainly.

## Next

* The 9 scissors trials remain the wall: 1 of 9, and the encoder is the limit,
  not the proposer. FastSAM puts a tight region on the scissors on 72% of
  frames and MobileCLIP scores it 0.149, below its own distractors.
* The 3 banana trials are now a navigation question and belong with that work.
* Worth testing on the full 107 before adopting: this subset is selected for the
  failure the stage addresses, and the cost side (42 commits against 18, 25
  attempts against 17) is exactly the kind of thing that can lose trials
  elsewhere. The stranded-44 lesson applies -- an arm read 26/44 on a selected
  subset and 19/44 on the full run.
