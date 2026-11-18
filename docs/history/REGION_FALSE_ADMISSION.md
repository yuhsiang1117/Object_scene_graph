# The proposal stage on frames without the object

positives: the query's own episodes;  negatives: other-target episodes, same scene
live tau = 0.24

## Overall

| tau | admits on NEGATIVE frames | admits on POSITIVE frames | ...and covers the object |
|---:|---:|---:|---:|
| 0.20 | 302/640 (47%) | 308/400 (77%) | 256/400 (64%) |
| 0.22 | 126/640 (20%) | 258/400 (64%) | 234/400 (58%) |
| 0.24 | 83/640 (13%) | 222/400 (56%) | 209/400 (52%) |
| 0.26 | 57/640 (9%) | 183/400 (46%) | 178/400 (44%) |
| 0.28 | 42/640 (7%) | 153/400 (38%) | 148/400 (37%) |
| 0.30 | 21/640 (3%) | 112/400 (28%) | 109/400 (27%) |
| 0.32 | 8/640 (1%) | 67/400 (17%) | 65/400 (16%) |
| 0.34 | 2/640 (0%) | 11/400 (3%) | 11/400 (3%) |
| 0.36 | 1/640 (0%) | 0/400 (0%) | 0/400 (0%) |

## Per query, at the live tau

| query | negative admits | positive admits | positive & covers |
|---|---:|---:|---:|
| scissors | 0/80 (0%) | 2/50 (4%) | 2/50 (4%) |
| mug | 13/80 (16%) | 7/50 (14%) | 7/50 (14%) |
| banana | 0/80 (0%) | 48/50 (96%) | 46/50 (92%) |
| soup_can | 2/80 (2%) | 25/50 (50%) | 25/50 (50%) |
| cracker_box | 6/80 (8%) | 29/50 (58%) | 25/50 (50%) |
| bowl | 34/80 (42%) | 49/50 (98%) | 45/50 (90%) |
| pitcher | 7/80 (9%) | 30/50 (60%) | 28/50 (56%) |
| plate | 21/80 (26%) | 32/50 (64%) | 31/50 (62%) |

## Score distributions

| population | n | p10 | median | p90 | max |
|---|---:|---:|---:|---:|---:|
| positive | 400 | 0.182 | 0.251 | 0.330 | 0.355 |
| negative | 640 | 0.167 | 0.199 | 0.251 | 0.361 |

## What this predicts, and what happened

The stage was run on the full 107 (`outputs/osg_v3_region_full`) against
`_sensor_v2`. Per query, the false-admission rate measured above against the
trials the stage actually won or lost:

| query | admits with no object in view | separable at a 2% budget | trials, v2 -> arm | delta |
|---|---:|---|---:|---:|
| bowl | 42% | no | 9 -> 6 | -3 |
| plate | 26% | no | 12 -> 12 | +0 |
| mug | 16% | no | 4 -> 0 | -4 |
| pitcher | 9% | no | 10 -> 9 | -1 |
| cracker_box | 8% | no | 8 -> 4 | -4 |
| soup_can | 2% | yes | 8 -> 10 | +2 |
| scissors | 0% | no | 5 -> 4 | -1 |
| banana | 0% | yes | 1 -> 2 | +1 |

The two classes that separate are the only two the stage wins on. Every class
that does not separate is flat or negative. Eight of eight.

## Verdict

The 0.24 threshold was chosen on frames that contained the object, where it
reads as 85% precision. On the population the agent actually runs it on -- every
keyframe, the target almost never in view -- it admits on 13% of frames, and an
admitted region is handed downstream as an ordinary detection under the target's
own label. There is nothing after it that can tell a proposed `bowl` from a
detected one, so it wins the candidate gate and spends an attempt.

That is the whole of the -18 on the full 107 and the -10 on the 57 trials of the
episode-gated rerun. The episode gate cut admissions 767 -> 421 and changed no
outcome, because the gate limits how OFTEN the stage fires and the defect is
what it emits when it does.

The distributions say no global threshold exists: positives median 0.251,
negatives median 0.199, and the negative p90 IS the positive median. Per class
at a 2% false-admission budget only banana (92% cover) and soup can (48%)
survive; scissors -- the largest perception bucket, the reason the stage was
built -- is 4% at any threshold, and mug is 0%.

So the stage is not a general perception fallback and must not ship as one. It
is a per-class one, for two classes, worth about +3 trials against a benchmark
noise floor of +-3.

