# The fused proposal arm on the full 107

> **Superseded by [`PROPOSAL_FUSION_CLS_RESULT.md`](PROPOSAL_FUSION_CLS_RESULT.md).** This is
> the `fuse` arm at 59/107; `fuse_cls` reached 62/107 and is the arm that shipped
> (`configs/experiment/dualmap_osg_unified.yaml`). Kept for the paired comparison.

run `outputs/osg_v4_fuse_full` vs reference `outputs/osg_sensor_v2_full`, paired on 107 trials

| condition | v2 | fuse | delta | +/- |
|---|---:|---:|---:|---:|
| in_anchor | 31/54 | 33/54 | +2 | +3/-1 |
| cross_anchor | 26/53 | 26/53 | +0 | +2/-2 |
| **total** | 57/107 | 59/107 | +2 | +5/-3 |

Perception subset (21 trials the detector never names): v2 0 -> fuse 5
Everything else (86): v2 57 -> fuse 54

## Proposal-only commits

29 commits in 24 episodes; 21 attempts spent on them, 9 scored.

## Trials that changed

| trial | v2 | fuse | perception | proposal-only commit | fuse dist | attempts |
|---|---:|---:|---|---|---:|---:|
| 00829-QaLdnwvtxbs__in_anchor__0116__soup_can | 0 | 1 | yes | yes | 0.72 | 2 |
| 00829-QaLdnwvtxbs__in_anchor__0117__cracker_box | 0 | 1 | yes | yes | 0.62 | 2 |
| 00829-QaLdnwvtxbs__in_anchor__0118__cracker_box | 0 | 1 | yes | yes | 0.50 | 2 |
| 00880-Nfvxx8J5NCo__cross_anchor__0128-2__scissors | 0 | 1 | yes | yes | 0.81 | 1 |
| 00880-Nfvxx8J5NCo__cross_anchor__0129-1__soup_can | 0 | 1 | yes | yes | 0.84 | 2 |
| 00829-QaLdnwvtxbs__cross_anchor__0129-1__bowl | 1 | 0 |  | yes | 6.12 | 3 |
| 00829-QaLdnwvtxbs__cross_anchor__0129-1__plate | 1 | 0 |  | yes | 1.57 | 3 |
| 00848-ziup5kvtCCR__in_anchor__0117__mug | 1 | 0 |  | yes | 1.15 | 2 |

## By query

| query | condition | v2 | fuse |
|---|---|---:|---:|
| banana | cross_anchor | 1/3 | 1/3 |
| banana | in_anchor | 0/3 | 0/3 |
| bowl | cross_anchor | 3/6 | 2/6 **-** |
| bowl | in_anchor | 6/6 | 6/6 |
| cracker_box | cross_anchor | 5/8 | 5/8 |
| cracker_box | in_anchor | 3/9 | 5/9 **+** |
| mug | cross_anchor | 2/3 | 2/3 |
| mug | in_anchor | 2/3 | 1/3 **-** |
| pitcher | cross_anchor | 5/9 | 5/9 |
| pitcher | in_anchor | 5/9 | 5/9 |
| plate | cross_anchor | 7/9 | 6/9 **-** |
| plate | in_anchor | 5/9 | 5/9 |
| scissors | cross_anchor | 0/9 | 1/9 **+** |
| scissors | in_anchor | 5/9 | 5/9 |
| soup_can | cross_anchor | 3/6 | 4/6 **+** |
| soup_can | in_anchor | 5/6 | 6/6 **+** |

## Reading

**Headline: 59/107 (55.1%)** -- `_sensor_v2` 57, the privileged navmesh arm 56,
DualMap 51. In-anchor 33/54, cross-anchor 26/53.

The +2 total is inside the benchmark's +-3 noise floor and is not, on its own,
a finding. What is a finding is where the trials moved, because every one of
the eight is attributed by the commit log:

* **The perception subset went 0 -> 5 of 21.** Those 21 are the trials the
  detector never names, where every arm before this one scored exactly 0 --
  the bucket was structurally closed. All five conversions scored through a
  proposal-only commit: two cracker boxes, two soup cans, and one **scissors**
  (cross_anchor 0128-2 in 00880: the 22-observation, 0.330 track from the
  observation run), which every earlier probe had written off as unreadable
  by the encoder.
* **The other 86 went 57 -> 54**, and all three losses are phantom commits
  that exhausted the attempts: bowl and plate in 00829's cross-anchor layout
  0129-1 -- the layout the observation run flagged as phantom-heavy (bowl
  produced 36 of 70 phantoms) -- and the 00848 in-anchor mug, where v2 named it
  late and the phantom spent the attempts first.
* 29 proposal-only commits in 24 episodes cost 21 attempts and scored 9.

So the fusion does what it was built to do -- convert the trials the label
path cannot reach -- and its price is three trials in known-bad layouts. The
per-frame stage that lost 18 became a fused stage that wins 5 and loses 3,
with the named path bit-identical to v2 everywhere a proposal did not commit.

**The lever left is the bar.** The observation tables put (4, 0.30) at 1
phantom episode in 37 against 5 at (4, 0.28), keeping 3 of the 8 true tracks;
the three losses here all sit below 0.30 while the scissors, mug and banana
true tracks sit above it. That is a defensible next arm, run on the full 107,
not read off this one. Lowering the per-frame observation bar (0.24) so weaker
objects accumulate a mean is the other, separate arm.

Provenance: `_sensor_v4_fuse`, commit a0ff0d0 and before; identity of the
named path verified on 43 trials with commits off (`docs/history/PROPOSAL_FUSION.md`).
