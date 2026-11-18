# The fused proposal arm on the full 107

run `outputs/osg_v4_fuse_cls_full` vs reference `outputs/osg_sensor_v2_full`, paired on 107 trials

| condition | v2 | fuse | delta | +/- |
|---|---:|---:|---:|---:|
| in_anchor | 31/54 | 34/54 | +3 | +3/-0 |
| cross_anchor | 26/53 | 28/53 | +2 | +2/-0 |
| **total** | 57/107 | 62/107 | +5 | +5/-0 |

Perception subset (21 trials the detector never names): v2 0 -> fuse 5
Everything else (86): v2 57 -> fuse 57

## Proposal-only commits

18 commits in 16 episodes; 13 attempts spent on them, 9 scored.

## Trials that changed

| trial | v2 | fuse | perception | proposal-only commit | fuse dist | attempts |
|---|---:|---:|---|---|---:|---:|
| 00829-QaLdnwvtxbs__in_anchor__0116__soup_can | 0 | 1 | yes | yes | 0.72 | 2 |
| 00829-QaLdnwvtxbs__in_anchor__0117__cracker_box | 0 | 1 | yes | yes | 0.62 | 2 |
| 00829-QaLdnwvtxbs__in_anchor__0118__cracker_box | 0 | 1 | yes | yes | 0.50 | 2 |
| 00880-Nfvxx8J5NCo__cross_anchor__0128-2__scissors | 0 | 1 | yes | yes | 0.81 | 1 |
| 00880-Nfvxx8J5NCo__cross_anchor__0129-1__soup_can | 0 | 1 | yes | yes | 0.84 | 2 |

## By query

| query | condition | v2 | fuse |
|---|---|---:|---:|
| banana | cross_anchor | 1/3 | 1/3 |
| banana | in_anchor | 0/3 | 0/3 |
| bowl | cross_anchor | 3/6 | 3/6 |
| bowl | in_anchor | 6/6 | 6/6 |
| cracker_box | cross_anchor | 5/8 | 5/8 |
| cracker_box | in_anchor | 3/9 | 5/9 **+** |
| mug | cross_anchor | 2/3 | 2/3 |
| mug | in_anchor | 2/3 | 2/3 |
| pitcher | cross_anchor | 5/9 | 5/9 |
| pitcher | in_anchor | 5/9 | 5/9 |
| plate | cross_anchor | 7/9 | 7/9 |
| plate | in_anchor | 5/9 | 5/9 |
| scissors | cross_anchor | 0/9 | 1/9 **+** |
| scissors | in_anchor | 5/9 | 5/9 |
| soup_can | cross_anchor | 3/6 | 4/6 **+** |
| soup_can | in_anchor | 5/6 | 6/6 **+** |

## Reading

**Headline: 62/107 (57.9%)** -- `_sensor_v2` 57, the 0.28 fused arm 59, the
privileged navmesh arm 56, **DualMap 51**. In-anchor 34/54, cross-anchor 28/53.

Paired against `_sensor_v2`: **+5 / -0.** The 21 trials the detector never
names go 0 -> 5 (soup can x2, cracker box x2, scissors), every one through a
proposal-only commit; the other 86 are 57 -> 57, unchanged trial for trial.
Paired against the 0.28 arm: +3 / -0 -- its three phantom losses (00829
cross-anchor 0129-1 bowl and plate, 00848 in-anchor 0117 mug) come back and
play out as v2, and none of its five wins is lost.

18 proposal-only commits in 16 episodes (29 in 24 at 0.28), 13 attempts, 9
scored. The per-class bar removed the phantoms and kept the true tracks, as the
offline probe said it would: it was set on the probe's per-class
false-admission rates, not on this benchmark.

+5 clears the +-3 noise floor, and every one of the five is attributed to the
mechanism by the commit log. This is the arm to report:
`dualmap_protocol_osg_sensor_v4_fuse_cls`.
