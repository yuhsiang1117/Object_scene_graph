# Why the detector emits nothing, and what a proposal stage would be worth

21 of the 50 remaining failures on `_sensor_v2` are perception: the target was
in view -- often close, centred and fully visible -- and never named. This is
the class-agnostic proposal stage that has been on the board since the feature
memory work. Two probes decide whether it is worth building.

The episode instrument reports a zero box and zero score on all 21, but it only
counts detections whose label ALREADY matches the target
(`eval/instruments.py:154`), so that zero means "no correctly-labelled box",
not "no box". The two are different defects with different fixes.

## 1. The region is there; YOLOE does not propose it

`scripts/legacy/probe_perception_gap.py`, 60 dumped keyframes per query, the object's
pixel known. "Tight" = a proposal covers that pixel and is at most 2% of the
frame and 25% of the largest covering proposal, i.e. it could BE the object
rather than the furniture it rests on.

| query | YOLOE covers | YOLOE tight | FastSAM covers | FastSAM tight |
|---|---:|---:|---:|---:|
| scissors | 45% | **0%** | 100% | **72%** |
| soup can | 75% | 25% | 100% | **100%** |
| cracker box | 65% | 20% | 97% | **92%** |
| banana | 88% | 48% | 100% | **87%** |

What YOLOE calls the boxes that do cover the object settles it:

- scissors: `mirror` x26, `scissors` x1
- soup can: `lamp` x32, `sofa` x19, `pillow` x15
- banana: `chair` x87, `bed` x10, `banana` x6
- cracker box: `cracker box` x18, `lamp` x13, `picture` x12

It proposes the furniture the object rests on. This is exactly DualMap's
structural advantage, now measured on our own frames: its detector emits a
tight crop and every stage downstream inherits it. FastSAM holds up at range
too -- 82% tight beyond 3 m.

## 2. But appearance can only pick it out for compact objects

`scripts/legacy/probe_region_ranking.py`, MobileCLIP-S2, ranking the object's region
against every other region in the frame (median 59 regions).

| query | tight region found | rank 1 | rank <=5 | true-region cosine | margin over best distractor |
|---|---:|---:|---:|---:|---:|
| cracker box | 46/50 | **91%** | 100% | 0.242 | **+0.028** |
| soup can | 50/50 | 60% | 80% | 0.223 | +0.011 |
| banana | 36/50 | 39% | 81% | 0.208 | -0.011 |
| **scissors** | 34/50 | **3%** | 9% | **0.149** | **-0.057** |
| all | 166/200 | 52% | 71% | | |

**Scissors is the finding that matters.** FastSAM finds it on 72% of frames and
MobileCLIP scores that region 0.149, below what it gives a typical distractor;
the margin is positive on 1 of 34 frames. A few dozen pixels of thin metal
carry too little signal for the encoder, and no amount of proposing fixes that.
Scissors is 9 of the 21 perception failures and 13 of all 50.

## 3. A threshold exists, at 85% precision and 24% recall

Admitting the argmax region, over all 166 frames with a tight region:

| tau | admits correct | admits wrong | precision |
|---:|---:|---:|---:|
| 0.20 | 82 | 69 | 54% |
| 0.22 | 69 | 38 | 64% |
| **0.24** | **40** | **7** | **85%** |
| 0.26 | 20 | 6 | 77% |
| 0.28 | 9 | 5 | 64% |

0.24 is the knee. Below it precision collapses; above it recall does.

## 4. Cost is not an obstacle

FastSAM-s is 13 ms a frame at imgsz 1024 on this GPU. Encoding ~60 crops is
the larger cost and would run only at close range.

## What it is worth, honestly

Perception failures by query: scissors 9, banana 3, soup can 3, cracker box 2,
mug 2, pitcher 2. Weighting by the measured rank-1 rates gives roughly **five
trials**, not twenty-one -- cracker box ~2, soup can ~2, banana ~1, scissors
~0. Episode-level conversion may beat the per-frame rate, since an episode
offers many frames and one good commit is enough; against that, a false
admission costs an attempt, and at 85% precision those will happen.

Five trials on 107 is worth having -- it is the same order as each navigation
fix delivered -- but it should be built knowing it does not touch the objects
that dominate the bucket. The honest framing for scissors stays what it was:
a shared ceiling that neither detector clears, now with the added detail that
it is not the detector at all but the encoder, and that a region proposer puts
a perfectly good crop in front of it 72% of the time.
