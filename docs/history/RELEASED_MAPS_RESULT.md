# Ceiling after the map rebuild

run       `outputs/osg_released_maps/flat_anchor_v2_island_close`  (107 trials)
reference `outputs/osg_close_full/flat_anchor_v2_island_close`  (107 trials)
paired on 107 shared trial ids

| condition | reference | run | delta | +/- | sign test |
|---|---:|---:|---:|---:|---:|
| in_anchor | 32/54 | 29/54 | -3 | +3/-6 | 0.508 |
| cross_anchor | 25/53 | 27/53 | +2 | +9/-7 | 0.804 |
| **total** | 57/107 | 56/107 | -1 | +12/-13 | 1.000 |

## By query

| query | condition | reference | run |
|---|---|---:|---:|
| banana | in_anchor | 2/3 | 2/3 |
| banana | cross_anchor | 2/3 | 3/3  **+** |
| bowl | in_anchor | 6/6 | 6/6 |
| bowl | cross_anchor | 4/6 | 5/6  **+** |
| cracker box | in_anchor | 3/9 | 3/9 |
| cracker box | cross_anchor | 6/8 | 4/8  **-** |
| mug | in_anchor | 3/3 | 0/3  **-** |
| mug | cross_anchor | 2/3 | 2/3 |
| pitcher | in_anchor | 5/9 | 4/9  **-** |
| pitcher | cross_anchor | 3/9 | 4/9  **+** |
| plate | in_anchor | 6/9 | 5/9  **-** |
| plate | cross_anchor | 4/9 | 6/9  **+** |
| scissors | in_anchor | 2/9 | 5/9  **+** |
| scissors | cross_anchor | 1/9 | 0/9  **-** |
| soup can | in_anchor | 5/6 | 4/6  **-** |
| soup can | cross_anchor | 3/6 | 3/6 |

## What the remaining failures are

**in_anchor**: 29/54 scored, 25 failed

| bucket | n | movable? |
|---|---:|---|
| committed_elsewhere | 13 | yes -- search order |
| perception_wall | 5 | no -- both detectors score ~0 beyond a metre |
| perception_close | 3 | yes -- seen inside a metre and still unnamed |
| never_in_view | 3 | yes -- coverage |
| near_miss | 1 | yes -- the 1.0-1.6 m ring |

Ceiling with those held fixed: 49/54 = 90.7%  (goal 38/54)

**cross_anchor**: 27/53 scored, 26 failed

| bucket | n | movable? |
|---|---:|---|
| perception_wall | 9 | no -- both detectors score ~0 beyond a metre |
| never_in_view | 7 | yes -- coverage |
| committed_elsewhere | 6 | yes -- search order |
| perception_close | 4 | yes -- seen inside a metre and still unnamed |

## Which objects the residue belongs to

The question the ceiling turns on: how much of what is left is the two
assets neither detector can see, and how much is everything else.

| query | perception_wall | perception_close | never_in_view | near_miss | committed_elsewhere | failed |
|---|---:|---:|---:|---:|---:|---:|
| banana | 0 | 0 | 0 | 0 | 1 | 1 |
| bowl | 0 | 0 | 0 | 0 | 1 | 1 |
| cracker box | 1 | 2 | 3 | 1 | 3 | 10 |
| mug | 4 | 0 | 0 | 0 | 0 | 4 |
| pitcher | 0 | 2 | 1 | 0 | 7 | 10 |
| plate | 0 | 0 | 2 | 0 | 5 | 7 |
| scissors | 7 | 2 | 4 | 0 | 0 | 13 |
| soup can | 2 | 1 | 0 | 0 | 2 | 5 |

Scissors and mug account for 17 of 51 remaining failures (33%).

## Scissors, specifically

The rebuild's whole point: 00829 now holds a real `scissors` track 0.01 m
from the object, where maps_v5 held a bleach cleanser.

| trial | ref | run | final distance | target named | closest approach |
|---|---:|---:|---:|---:|---:|
| cross_anchor__0128-1__scissors | 0 | 0 | 10.75 m | 0 | 2.20 m |
| cross_anchor__0128-2__scissors | 0 | 0 | 5.60 m | 0 | - |
| cross_anchor__0129-1__scissors | 0 | 0 | 4.72 m | 0 | 1.40 m |
| in_anchor__0116__scissors | 0 | 1 | 0.99 m | 0 | 0.77 m |
| in_anchor__0117__scissors | 0 | 1 | 0.55 m | 1 | 0.52 m |
| in_anchor__0118__scissors | 1 | 1 | 0.48 m | 6 | 0.44 m |
| cross_anchor__0128-1__scissors | 0 | 0 | 10.74 m | 0 | 4.49 m |
| cross_anchor__0128-2__scissors | 0 | 0 | 12.69 m | 0 | 1.42 m |
| cross_anchor__0129-1__scissors | 0 | 0 | 14.87 m | 0 | - |
| in_anchor__0116__scissors | 1 | 1 | 0.48 m | 5 | 0.42 m |
| in_anchor__0117__scissors | 0 | 0 | 2.38 m | 0 | - |
| in_anchor__0118__scissors | 0 | 0 | 4.35 m | 0 | - |
| cross_anchor__0128-1__scissors | 0 | 0 | 3.77 m | 0 | 0.66 m |
| cross_anchor__0128-2__scissors | 1 | 0 | 2.02 m | 0 | 2.06 m |
| cross_anchor__0129-1__scissors | 0 | 0 | 6.20 m | 0 | 0.75 m |
| in_anchor__0116__scissors | 0 | 1 | 0.48 m | 3 | 0.46 m |
| in_anchor__0117__scissors | 0 | 0 | 4.41 m | 0 | 2.46 m |
| in_anchor__0118__scissors | 0 | 0 | 14.88 m | 0 | 2.10 m |

## Verdict

Success rate is a wash: 57 -> 56, in-anchor -3, cross-anchor +2, sign test
p = 1.00. Read alone that says the rebuild did nothing. Two other numbers say it
did exactly what it was built to do.

**Scissors in-anchor, where the fix applies, final distance to the object:**

    old map   0.4  0.4  2.6  8.6  9.0 10.1 10.2 14.7 14.9     median 9.02 m
    new map   0.5  0.5  0.5  0.5  1.0  2.4  4.3  4.4 14.9     median 0.99 m

On the old map the agent finished ten metres from the scissors in five of nine
trials, because with no scissors track in the map the search had no proximity
anchor and the stale-anchor stop had nothing to stop at -- machinery every other
object had and this one did not. It now goes to the right place. Three trials
convert, 2/9 -> 5/9.

**Successes that actually localised the object** (the agent named it on some
keyframe, rather than stopping within a metre without ever recognising it):

    old map   53 of 57   (93%)
    new map   54 of 56   (96%)

So the corrected map produces MORE genuine successes than the contaminated one
while scoring one fewer trial. The lost trials were largely credit the 1 m rule
hands an agent for standing in the right place: two of the three mug in-anchor
successes it gave up had `named = 0`.

The rebuilt maps should be the standard root for this benchmark. They describe
the world the episodes are scored in, they raise localised success, and they
make every scissors number mean what it says. The headline becomes **29/54
in-anchor and 27/53 cross-anchor**.

Caveat kept in the open: this is not a single-variable change. The released
worlds hold more objects to hunt, so 00848 went from 343 tracks to 800 and
00880 from 520 to 618, and part of the churn on other queries is that extra
coverage re-routing exploration rather than the phantom's removal.

## Where the ceiling actually is

In-anchor is 29/54 against the 38/54 goal. The bucket table above reads the
residue as 13 "committed elsewhere" on pitcher and plate, and an earlier
version of this section called that search order. It is not. Measured after
the fact (`scripts/legacy/analyze_reachability_ceiling.py`,
`docs/history/REACHABILITY_CEILING.md`): 10 of the 54 in-anchor objects and 5 of the
53 cross-anchor objects have **no navmesh point within 1.0 m** for our agent
(radius 0.18, height 0.88, the mesh habitat-lab builds). Nine of the thirteen
"committed elsewhere" in-anchor failures are those trials; each had committed
to the true track at 0.02-0.10 m and stopped 0.00-0.06 m from the nearest
point the agent can occupy. The closing walk works; the object is mid-bed.

DualMap fails 13 of those 15 too. Its released runner moves the agent with
`set_agent_state` along its own occupancy grid, so it is not held to the
navmesh at all; the two it scores there are that.

On the reachable subset: in-anchor ours 29/44 (65.9%) vs DualMap seed 12
34/44 (77.3%); cross-anchor ours 27/48 (56.2%) vs 15/48 (31.2%).

What is genuinely left, over all 51 failures: 15 unreachable under the rule,
18 perception (11 beyond a metre, 7 inside it), 10 never in view (all at the
500-step budget), 8 other (two stops 2-3 cm over the bar, one 200-step
approach oscillation, one budget-end commit, three wrong-instance commits,
one cabinet cracker box). The full paired reading against DualMap's three
seeds is in `docs/history/RESIDUE_AFTER_RELEASED_MAPS.md`.
