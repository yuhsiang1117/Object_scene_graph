# Overnight iteration toward 70 / 50 on DualMap's released benchmark

Branch `experiment/sr-70-50`, started 2026-09-09 19:15. Goal: in-anchor 70%,
cross-anchor 50% at DualMap's own rule (1 m to the object, three attempts),
so that the paper can put our number beside DualMap's measured 64.8 / 30.2 on
the same 107 trials. Everything below is appended as it happens; the decision
rule is stated first so the reader can check that it was followed.

## Protocol

1. **Small batch first.** Every arm runs on the *current best run's failures*
   (`scripts/make_hard_subset.py --reference <best>`; scissors and mug dropped,
   they are a perception wall), six shards, ~25 minutes. A subset of failures
   can only show conversions, never regressions, so it answers one question:
   does the mechanism fire and convert anything at all.
2. **Full 107 for anything that converts.** Sharded (`data/splits/dualmap_all.json`,
   `SHARDS_PER_SCENE=2`), ~50 minutes. Paired against the best run with the sign
   test; the funnel and the mechanism counters are read before SR.
3. **One change per arm.** An arm's preset composes on the current best preset
   and sets exactly one thing.
4. **What "best" means.** Highest in-anchor + cross-anchor on the full 107 with
   no split worse by more than one trial than the previous best.
5. **GPU budget.** 10 GB card exposing 9.6 GB; a single-floor process is 1.3 GB.
   At most seven processes; the 15-scene authored run is paused for the night.

## Where we start

| run | in-anchor | cross-anchor | note |
|---|---:|---:|---:|
| DualMap, measured, seed 12 | 35/54 (64.8%) | 16/53 (30.2%) | its own harness on its own data |
| tight ring (baseline of the day) | 24/54 | 15/53 | |
| + look before absence (`look`) | 25/54 | 16/53 | |
| + flat proximity (`look_flat`) | 24/54 | 18/53 | own-surface arrivals 7 -> 13 |

Ceilings from the static-pass split (docs/history/SR_PROPOSAL_CLOSE_LOOK.md): the
target is in the prior map for 39/54 in-anchor and 39/53 cross-anchor trials;
the rest are scissors, mug and five cracker boxes at 20% / 7%. So 70/50 means
about 35/39 in-anchor and 26/39 cross-anchor on the trials the detector can see.

## Levers, in the order they are tried

1. **Stale-anchor policy** (`look_flat_anchor`): stop once at the silent stale
   anchor; retire its unseen prior-map twins after a refutation. Running.
2. **Proximity until refuted** (`search_drop_proximity_after_absence` on
   `search_proximity_len_m = 1.0`), so in-anchor's second attempt searches the
   stale surface's neighbourhood before sweeping.
3. **Stop on ambiguous absence** (`abandon_below_p` 0.3): a second stop where the
   belief has not collapsed, paid in attempts.
4. **Perception, scissors and mug**: names on the dumped frames first (offline),
   then a foveated second look at close container surfaces
   (`scene_graph.foveate_containers`) if the names do nothing. If neither moves
   in-situ recall off zero, the honest option is to substitute detectable
   objects in the released layouts, which costs a DualMap rerun on the
   substituted layouts (their harness, ~10 hours for three seeds) and is
   recorded here as a cost, not taken tonight.

## Log

### 19:30 — Perception: scissors and mug cannot be named in situ, by anything cheap

`scripts/legacy/probe_nearmiss.py` over every dumped in-view keyframe of the tight-ring
run (`outputs/nearmiss_tightring/{scissors,mug}.log`): 352 scissors frames and
160 mug frames, ranges 1.4-5.2 m, the frames on which the object was in front
of the camera and unoccluded.

| condition | scissors | mug |
|---|---:|---:|
| the run's own name and settings | 0/352 | 0/160 |
| every furniture competitor removed | 0/352 | 0/160 |
| window around the object, upscaled (160 px, 96 px) | 0/352, 0/352 | 0/160, 0/160 |
| ten alternative names | 0/352 for all ten | 0/160 for nine; `cup` 9/160 (0.06) |

The detector produces nothing over these two assets at any name, with no
competitor to lose to, at any magnification: not a threshold, not a name, not
scale. The bake-off (PERCEPTION_INVESTIGATION.md) already showed DualMap's own
detector at 0.06 on scissors. These 24 trials (18 scissors, 6 mug) are a
ceiling for both systems: without them the best attainable is 45/54 in-anchor
(83%) and 44/53 cross-anchor (83%), so 70/50 does not need them. The only
ways to move them are a different recogniser (a VLM on crops, at a per-frame
cost this pipeline cannot pay) or substituting detectable objects in the
released layouts, which requires rerunning DualMap on the substituted layouts
(~10 h of their harness for three seeds). Neither is taken tonight; the night
goes to the 83 trials the detector can see.

### 19:40 — Stale-anchor arm, first run: twin retirement alone (the stop was defective)

`outputs/osg_anchor/flat_anchor`, full 107, paired against `look_flat`
(`outputs/osg_anchor/TWINS_AB.md`). In this run the stale-anchor stop was
granted 29 times and taken zero: the close look's silent path re-approached
the ring and the once-only grant was spent before the arrival (fixed in
`ddacbca`). So this run measures the twin retirement, which fired in 58 of
107 trials.

| | look_flat | + twin retirement |
|---|---:|---:|
| in-anchor SR | 24/54 | 24/54 (+1 / -1) |
| cross-anchor SR | 18/53 | **20/53** (+4 / -2, p = 0.69) |
| cross-anchor: target ever named | 27 | 31 |
| cross-anchor: seen, never named | 16 | 13 |
| cross-anchor: episodes that selected a far surface | 40 | 44 |
| median steps | 371 | 500 |
| trials at budget | 48 | 54 |

Retiring the ghosts converts walking into searching: more surfaces selected,
more targets named, two more cross-anchor successes -- and every freed step is
spent, so the median episode now runs to the budget. The budget is the
binding constraint again, one level down. In-anchor is untouched, as expected
without the stop.

### 20:45 — Stale-anchor policy, fixed stop: cross-anchor 20, in-anchor still 24

`outputs/osg_anchor2` (`ANCHOR_AB.md`), full 107 against `look_flat`:
in-anchor 24/54 (+1 / -1), cross-anchor **20/53** (+4 / -2). `absence_abandon`
fell 52 to 5: the stops replaced the abandons. Then why no in-anchor gain?
Tracing all 54 in-anchor trials:

* The stale stop was taken in **9** trials, at a median **1.47 m** from the
  object: one inside the metre, four at 1.0-1.5 m, four beyond. The stop is
  taken on the approach's ring, 0.65-0.8 m from the track centre, and the
  object moved a median 0.7 m the other way. Right idea, wrong spot.
* In 00848 the pitcher (3 trials) and plate commit first to a **ghost** 8 m
  from the object -- a prior-map false positive that the detector keeps
  re-detecting live, in three fragment tracks 0.2 m apart -- and spend all
  three attempts stopping on it: the twin rule cannot touch a track that is
  seen live, and a failed attempt strikes one fragment while the next commit
  takes the next.
* Three 00848 cracker-box trials end the budget 0.4-1.8 m from the object
  without a stop: the true track's belief was driven to the negative clamp by
  missed expectations on the way and one live sighting cannot lift it over
  the bar.

Two fixes, both behind flags (`stale_stop_at_nearest_free`,
`failed_attempt_disables_place`): the stop walks to the nearest navigable
point to the track centre first, and a failed attempt disables every track
of the label within `fp_disable_radius_m` of the stop's centre. The third
item is the presence clamp and is left for later.

### 20:45 — Proximity until refuted, on the subset: rejected

`outputs/osg_next/flat_anchor_drop` on look_flat's 32 failures, against the
fixed stale-anchor arm on the same ids: in-anchor 2/8 against 1/8,
cross-anchor **2/24 against 4/24** (1 gained, 3 lost). Anchoring the search
on the stale spot until it is refuted costs cross-anchor what it was meant
to give in-anchor. Not carried forward.

### 21:15 — Second cut on the 21 in-anchor failures: 2 converted, the rest never commit

`outputs/osg_next/flat_anchor_v2` on the fixed arm's in-anchor failures with
scissors and mug dropped (`V2_INANCHOR.md`). The nearest-free stop fired 6
times and now lands at 0.66-1.8 m (was 1.15-1.9 on the ring); two soup-can
trials converted at 0.94 and 0.66 m. Place retirement fired 33 times. The
other 19 stay failures, and the funnel says where: **admitted, never
committed 9 of 21** -- the detector did name the true object, and the agent
never made it a goal. The true track's belief sits at the negative clamp,
driven there by missed expectations on the way, and a single sighting is not
enough to lift it over the bar. Two arms next, both one knob:
`presence.l_clamp` 6 -> 2 (a sighting can bring a track back) and
`rank_candidates_by_presence` off (confidence counts again).

### 21:25 — Second stop on ambiguous absence: null

`outputs/osg_next/flat_anchor_stop2` on look_flat's 32 failures, against the
fixed stale-anchor arm on the same ids: every funnel row identical, 5/32
both. `absence_abandon` was already 4 on these trials after the stale stop
took over the silent arrivals, so a knob that only changes what happens
after an abandon has almost nothing left to act on. Not carried forward.

### 21:50 — Belief clamp 6 -> 2: null

`outputs/osg_next/flat_anchor_v2_clamp` on the same 21 in-anchor failures:
funnel identical to the second cut, 2/21. The true track's belief is not
what keeps it from being committed; the block is elsewhere in the candidate
gates, and the next step is to read those gates on the ten trials directly.

### 22:00 — Found it: "unreachable", twice from the same pose, retires the true track

Reading the candidate gates on the in-anchor failures where the target was
admitted and never committed: the true object HAS a live track in most of
them -- 00848 pitcher: 0.00 m, belief 0.95, 19 observations; 00880 plate:
0.05 m, 0.95, 10 observations; 00880 cracker box: 0.03 m, 0.95 -- and it
was never a goal. `candidate_reject_log` says why, 37 times on the in-anchor
failures and 16 on the cross-anchor ones, every entry the same word:
**unreachable**. The viewpoint planner finds no standable ring pose at that
moment (a plate on a desk against a wall, a pitcher on a bed), the
reachability check strikes the track, the next step asks the same question
from the same pose and strikes it again, and `max_identity_rejections = 2`
retires it for the episode. The restrike guard (`unreachable_restrike_m`)
was written for exactly this and has been shipping at 0.0, off. Arm: 1.0 m,
so retiring a track takes two strikes from places more than a metre apart.

### 22:06 — Ranking by confidence x presence: null

`outputs/osg_next/flat_anchor_v2_rankp` on the 21 in-anchor failures: 2/21,
admitted-never-committed still 9. The ranking never sees the true track,
because the reachability check strikes it before ranking matters (previous
entry). Not carried forward.

### 22:27 — Restrike guard alone: the track survives and is still never a goal

`outputs/osg_next/flat_anchor_v2_restrike` on the 21 in-anchor failures:
2/21, unchanged. The guard did what it says -- 325 unreachable verdicts, 292
of them discounted as restrikes, so the true track is no longer retired --
but a candidate the check calls unreachable is not committed either, and
from the same pose the answer never changes. The block is the answer, not
the counting. The nearest-free arm (`v2_nf`) asks the check about the cell
the approach would actually drive to; running.

### 22:30 — Second cut on the 22 cross-anchor failures: 3 converted

`outputs/osg_next_cross/flat_anchor_v2` on the fixed arm's cross-anchor
failures (scissors and mug dropped), against the fixed arm on the same ids:
**3/22 against 0/22**; never-in-view 9 -> 5; on the 11 no-detection
failures the object's own surface was selected in 3 (was 1) and the target
named in 5 (was 0). Retiring a refuted place frees the budget the search
needed. The second cut is a gain on both halves (in-anchor +2 of 21,
cross-anchor +3 of 22 on the failure subsets) and becomes the base for the
full-107 confirmation once the reachability arm reports.

### 23:00 — Nearest-free reachability: null, and why -- the navmesh has islands on the furniture

`outputs/osg_next/flat_anchor_v2_nf` on the 21 in-anchor failures: 2/21,
the nearest-free answer fired 3 times against 333 "unreachable" verdicts.
So the point was not the problem; the function was. Checked offline on the
00880 plate with habitat-sim's own pathfinder: the scene's navmesh has four
islands -- the floor (63 m2) and three furniture tops of about 2 m2 -- and
`snap_point` of a point beside the desk, queried at the agent's height,
returns a point ON the desk top (island 1), because it is nearer in 3D than
the floor 0.6 m to the side. From the desk top no path exists to the agent.
The same query constrained to the agent's island (`snap_point(...,
island_index)`) returns the floor beside the desk, 11.6 m of geodesic away,
reachable.

This is not only the reachability check: the navmesh follower's goal is
snapped the same way, which is where "the follower reports None while the
agent is still 6.4 m away" came from. `agent.navmesh_snap_on_agent_island`
constrains both. Arm `v2_island`, running on both failure subsets.

### 23:06 — Restrike guard + nearest-free reachability on the 22 cross-anchor failures: 5 converted

`outputs/osg_next_cross/flat_anchor_v2_nf` against the fixed arm on the same
ids: **5/22 against 0/22** (the second cut alone gave 3/22); never-in-view
9 -> 5; on the 11 no-detection failures the target was named in 5 (was 0).
On cross-anchor the nearest-free answer does fire where the in-anchor one
did not -- the objects sit on beds and benches whose free cell is reachable
-- so the mitigation buys two trials the island fix should buy outright.

### 23:24 — Island snapping on the 21 in-anchor failures: 6 converted, none lost

`outputs/osg_next/flat_anchor_v2_island` against the second cut on the same
ids: **6/21 against 2/21** (+4 / -0, p = 0.125 on 21). Unreachable strikes
333 -> **0**; never-in-view 2 -> 0; admitted-never-committed 9 -> 5; trials at
budget 18 -> 11. The remaining failures now mostly *stop*, at 1.0-1.9 m: the
stale stop lands on the nearest navigable cell to the track, 0.5-1 m from
the centre on a bed or a desk, and the object moved 0.7 m the other way. That
is the next in-anchor lever, and it is geometry, not search. `v2_island` goes
to the full 107 now.

### 23:53 — Island snapping on the 22 cross-anchor failures: 5 converted, none lost

`outputs/osg_next_cross/flat_anchor_v2_island` against the fixed arm on the
same ids: **5/22 against 0/22**; never-in-view 9 -> 2; on the 11
no-detection failures the object's own surface was selected in 2 and the
target named in **6** (was 0). The island fix does on cross-anchor what the
reachability mitigations did, and does it by fixing the cause. The
confirmation run of `v2_island` on the full 107 is in flight (six shards).

### 00:09 — Relook after a failed stale stop: null

`outputs/osg_next/flat_anchor_v2_island_relook` on the 21 in-anchor
failures: 6/21, identical funnel. The relook was scheduled in only 3 trials
and started in 2, because a failed stale stop is now usually followed by a
live sighting of the target on the walk (the track is then `seen_live` and
the rule stands down), and the failures that remain are "committed
elsewhere" (6) and "admitted, never committed" (5), neither of which a
second look at the stale surface addresses. Not carried forward.

### 05:59 — `v2_island` on the full 107: in-anchor 30/54, cross-anchor 23/53

`outputs/osg_island_full` (report `ISLAND_FULL.md`, four arms paired on 107
ids). The two 00880 shards of this run were killed at 00:07 by a stray
`pkill` while a waiter was being stopped; the 28 unfinished 00880 trials were
rerun from 05:32 as four shards (the seven finished ones kept under
`done0/`, `done1/`), and the run is the same preset and code throughout
(commit e16f11c).

| split | tight ring | previous best (`anchor2`) | **island** | DualMap measured | paper |
|---|---:|---:|---:|---:|---:|
| in-anchor | 24/54 = 44.4% | 24/54 | **30/54 = 55.6%** | 35/54 = 64.8% | 70 |
| cross-anchor | 15/53 = 28.3% | 20/53 | **23/53 = 43.4%** | 16/53 = 30.2% | 50 |

Against the tight ring: in-anchor +10 / -4 (p = 0.18), cross-anchor +12 / -4
(p = 0.077). Against the previous best on the same ids: in-anchor +8 / -2,
cross-anchor +7 / -4; every scene is ahead or level (in-anchor 11/10/9 vs
8/8/8; cross 11/7/5 vs 11/5/4). The mechanism moved as on the subsets:
never-in-view 6 -> 0 in-anchor and 12 -> 2 cross-anchor, target named in 35
of 53 cross-anchor trials (was 20), own surface reached in 12 of the
baseline's 32 no-detection cross failures (was 6), trials at budget 47 ->
43, median steps 421 -> 371 with 9 m more travelled. The six lost trials are
two 00880 bowls in-anchor (stopped at 258 and 231 steps, elsewhere) and four
cross-anchor budget runs (plate x3, banana): the search now goes further and
sometimes arrives later.

Without scissors and mug, which no arm can name, the run is 25/42 in-anchor
and 20/41 cross-anchor. **`v2_island` is the working configuration**
(`configs/experiment/dualmap_protocol_osg_look_flat_anchor_v2_island.yaml`).
Cross-anchor is 13 points above DualMap's measured number and 7 short of the
paper; in-anchor is 9 below DualMap and 14 below the goal. What is left
in-anchor is no longer reachability: the funnel is "seen, never named" 9
(scissors/mug 7 of them), "committed elsewhere" 7, "admitted, never
committed" 5, "committed, never arrived" 3.

### 06:55 — "Committed elsewhere" is a near-miss stop followed by place retirement

The island run's seven in-anchor "committed elsewhere" trials read, commit
by commit: in six the agent committed to a *live* track 0.03-0.6 m from the
object, stopped, and the stop was scored a failure at 1.00-1.56 m; the
failed attempt then disabled the place (`failed_attempt_disables_place`),
the true track was bypassed, and the next commit was whatever remained,
metres away. The wrong final commit is the symptom; the near-miss stop is
the cause. Across the full run the band is wider than those seven:

| split | failures | with a stop in (1.0, 1.6] m | best stop <= 2.0 m | no stop at all |
|---|---:|---:|---:|---:|
| in-anchor | 24 | **12** | 15 | 5 |
| cross-anchor | 30 | **8** | 9 | 7 |

The geometry of the 20 (`nearmiss_geom.py`, habitat's own pathfinder on the
scene navmesh): the agent had reached its viewpoint goal to within 0.1 m in
most of them, so the goal itself was the problem. The ring ladder stops at
the innermost ring with a FREE costmap cell, and beside a bed or a desk the
inflated costmap has none nearer than about a metre from the track centre;
the navmesh floor, which already accounts for the agent's radius, comes
within 0.9 m of the object in 13 of the 20 (0.30-0.93 m) and 1.04-1.16 m in
the other 7, which no stopping rule can score.

The lever is the last metre, again ([[osg-standoff-costs-the-last-metre]]
was the same shape at 1.27 m): `agent.approach_close_last_metre_m` (commit
8f6a65f). When an approach path is consumed and the agent is still farther
than 0.8 m from the track centre, it asks the navmesh for the nearest
navigable point to the centre on its own island
(`HabitatObjectNavEnv.nearest_navigable_xy`, 5 ms) and walks there before
the stop; the costmap re-aim of the stale stop stands down once that walk
has been made. Arm `flat_anchor_v2_island_close`, first on the 20 near-miss
trials (`data/splits/dualmap_nearmiss.json`, `outputs/osg_next_close`),
then the full 107 if it converts.

### 07:25 — Closing the last metre on the 20 near-miss trials: 4 converted, none lost

`outputs/osg_next_close/flat_anchor_v2_island_close` against the island
run on the same ids: in-anchor 1/12, cross-anchor **3/8**, no trial lost.
Every conversion is a live track walked from 1.00-1.39 m to 0.55-0.86 m
(00880 bowl in-anchor; 00848 plate, 00880 bowl and plate cross-anchor), at
73-315 steps where the island run had spent 93-500. The walk started 11
times and was declined 18 times as "no gain". Two things in the per-trial
lines: a stop at 1.00004 m was left alone because the walk required a 0.15 m
gain, and twice the follower gave up short and the agent ended *farther*
from the track than the viewpoint it left (1.41 -> 1.53 m, 1.49 -> 2.14 m).
Both fixed in 8cb9763 (margin 0.05 m; a worse close walks back to the
viewpoint before the stop) and rerun on the same 20 before the full 107.

What the walk cannot reach: the in-anchor stops at 1.15-1.56 m on a *prior*
track that was never seen live (soup can, two pitchers, banana). The track
centre is the stale position and the object moved 0.7 m; the nearest floor
to the stale centre is not nearer to the object. Those are the stale-anchor
stops that the close look did not convert, and they are a perception
problem at the look, not a geometry one.

### 07:41 — Second cut on the same 20: 6 converted, none lost; to the full 107

`outputs/osg_next_close2` (commit 8cb9763) against the island run: in-anchor
**2/12**, cross-anchor **4/8**, no trial lost. The walk started 20 times,
walked back once, and was declined 7 times. Converted stops: 0.70, 0.81 m
in-anchor; 0.84, 0.86, 0.60, 0.63 m cross-anchor, at 73-401 steps where the
island run spent 93-500. The stops that stay outside the metre are the
prior-track stale stops (1.15-1.56 m, object moved away from the floor
side) and one geometry-bound plate (1.05 m, best navigable 0.58 m but the
walk was declined as no gain from that pose). The full 107 of
`flat_anchor_v2_island_close` is in flight (`outputs/osg_close_full`, six
shards).

### 08:50 — Closing walk on the full 107: in-anchor 32/54, cross-anchor 25/53

`outputs/osg_close_full` (`CLOSE_FULL.md`, `CLOSE_PAIRED.txt`), preset
`dualmap_protocol_osg_look_flat_anchor_v2_island_close`, commit 8cb9763.

| split | tight ring | island | **island + close** | DualMap measured | paper |
|---|---:|---:|---:|---:|---:|
| in-anchor | 24/54 = 44.4% | 30/54 | **32/54 = 59.3%** | 35/54 = 64.8% | 70 |
| cross-anchor | 15/53 = 28.3% | 23/53 | **25/53 = 47.2%** | 16/53 = 30.2% | 50 |

Against the tight ring: in-anchor +10 / -2 (sign test p = 0.039), cross-anchor
+12 / -2 (p = 0.013) -- the first arms of the night to clear p < 0.05 on
107 trials. Against the island run on the same ids: in-anchor **+2 / -0**,
cross-anchor **+5 / -3**. The walk started 52 times, walked back once, was
declined 16 times; stops within a metre went 30 -> 32 in-anchor and 23 -> 25
cross-anchor, median steps 371 -> 364, trials at budget 43 -> 41, 4 m less
travelled. Every scene is level or ahead (in-anchor 11/10/11 vs 11/10/9;
cross 11/9/5 vs 11/7/5). Without scissors and mug: 27/42 and 22/41.

The three cross-anchor losses are not the walk failing: a 00880 pitcher
whose track sat 0.44 m off the object, so walking to 0.61 m from the track
put the agent 1.01 m from the object where the island run's short stop had
been 0.74 m (a track-error case; the walk is neutral in expectation there);
a 00848 soup can the island run found at step 282 after the same first
attempt, which this run's later search did not (path divergence); and a
00829 bowl that used its three attempts on the same two wrong commits in
both runs and got a third lucky one only in the island run. The gains are
all the near-miss shape: 1.16 -> 0.84, 1.77 -> 0.91, 1.29 -> 0.86,
1.04 -> 0.60, 1.01 -> 0.63 m.

**`island_close` is the working configuration.** Cross-anchor is 17 points
above DualMap's measured number and 3 short of the paper; in-anchor is 5
below DualMap and 11 below the goal. In-anchor needs +6 of the 15 attainable
failures left (scissors and mug hold 12 of the 22): the stale-prior stops at
1.15-1.56 m (5), "admitted, never committed" (5), and what remains of
"committed elsewhere" (6, overlapping the first). The next question is
whether the failed near-miss stop's own place retirement
(`failed_attempt_disables_place`, 0.5 m) is hiding the object's new position
from a later live detection -- in-anchor the object moved only 0.7 m.

### 09:00 — What is left in-anchor: two mechanisms and a shared ceiling

The 15 attainable in-anchor failures of `island_close` (scissors and mug
excluded), read against DualMap's own per-trial results
(`outputs/dualmap_official_bench/seed12/trials/*/result.json`):

- **8 are DualMap failures too**, and in every one the object's nearest
  navigable floor point is more than a metre away on habitat's own navmesh
  (three 00848 pitchers, two 00880 plates, two 00880 cracker boxes, the
  00829 pitcher; 1.04-1.16 m on the shipped mesh). A shared ceiling under
  the 1 m rule, like scissors and mug.
- **The three 00829 cracker boxes are a bug, not a search problem.** The
  agent commits to a live track 0.03-0.14 m from the box, drives to within
  0.11-0.13 m of its viewpoint goal, and never stops: the follower reports
  arrival only inside its 0.1 m goal radius, and 0.25 m steps circle it
  forever. Across the run, 14 final approaches spent 60-201 steps within
  0.3 m of their goal (9 failures). `agent.approach_arrival_m` (commit
  65776cf) consumes the path within 0.3 m, so the closing walk and the stop
  follow. Arm `island_close_arrive`, first on the 9 spin failures
  (`data/splits/dualmap_spin.json`).
- **The last 8 cm are DualMap's navmesh.** HM3D ships navmeshes cut for a
  0.10 m agent and DualMap drives on them; ours is recut for 0.18 m, so
  every floor edge sits 8 cm farther out. The 00848 banana: DualMap stood
  at 0.91 m, our closing walk reached the edge at 1.09 m; the shipped mesh
  puts the floor at 0.93 m, ours at 1.11 m. Arm `island_close_r10`
  (`agent.agent_radius: 0.1`, height and camera unchanged) on the 16
  near-miss trials of the close run (`data/splits/dualmap_nearmiss2.json`)
  is in flight. The radius changes every path, so the full 107 decides it.

The soup can (00829) and cracker box (00848) that DualMap scores are
detector walls here (0 detections in 19 and 0 close, centred keyframes).

### 09:11 — DualMap's navmesh radius on the 16 near-miss trials: 4 converted, none lost

`outputs/osg_next_r10/flat_anchor_v2_island_close_r10` against the close run
on the same ids: in-anchor **2/11**, cross-anchor **2/5**. The conversions
are the shape predicted: the 00848 pitcher that had stopped at 1.00004 m
stops at 0.96 m; a 00880 pitcher closes 0.98 -> 0.48 m; a 00829 cracker
box and a 00880 plate reach 0.49 and 0.64 m on a third attempt. The banana
moves 1.09 -> 1.11 m (the 8 cm went elsewhere on this mesh), and the stops
of the two 00880 cracker boxes moved by metres in both directions: the
radius changes every path, not only the last one, and the sign on the full
107 is what matters. Goes into the combined preset
`island_close_arrive_r10` with the arrival tolerance once the spin subset
reports.

### 09:26 — Arrival by distance on the 9 spin failures: 3 converted, none lost

`outputs/osg_next_arrive/flat_anchor_v2_island_close_arrive` against the
close run on the same ids: in-anchor **2/5**, cross-anchor **1/4**. The two
00829 cracker boxes that had spun at the goal now stop at 0.98 and 0.99 m
on their third attempt (293 and 295 steps where the close run spent 500 and
never stopped); a 00848 pitcher stops at 0.73 m. Arrivals declared by
distance: 5 in 9 trials. The third cracker box (0116) was not in this split
because its spin ran into the budget rather than the deadline, and is
covered by the full run. Combined with the radius in
`island_close_arrive_r10`, in flight on the full 107
(`outputs/osg_combined_full`).

### 10:33 — Combined arrival + radius on the full 107: a regression, and it is the radius

`outputs/osg_combined_full` (`COMBINED_FULL.md`, `COMBINED_PAIRED.txt`):
in-anchor **28/54**, cross-anchor **23/53**, against 32/25 for the close
run on the same ids (+3 / -7 and +6 / -8). The two mechanisms did what the
subsets said -- the third 00829 cracker box stops at 0.45 m, the 00848
pitcher that had stopped at 1.00004 m stops at 0.96 m, three more
cross-anchor trials close from 0.83-0.91 m to 0.53-0.80 m -- and the run
still lost more than it gained, because the 0.10 m navmesh changes every
frontier path, not only the last metre. Never-in-view went 0 -> 6 in-anchor
and 4 -> 8 cross-anchor. Four of the seven in-anchor losses are mug and
scissors trials the close run had scored on a *live* commit at steps
386-450 -- the search reached a close view of the "undetectable" object
at the very end of the budget -- and on the new mesh the same 500 steps
took a different route and never got there. The banana lost its second,
live commit the same way.

Decision: the radius is dropped (its +7 on the subsets is real and its
path lottery on the other 90 trials costs more), and the arrival tolerance
alone, `island_close_arrive`, goes to the full 107
(`outputs/osg_arrive_full`). The four late mug and scissors successes are
also a note on the budget: they scored at 420-460 of 500 steps.

### 11:42 — Arrival by distance on the full 107: even in-anchor, -2 cross-anchor; not adopted

`outputs/osg_arrive_full` (`ARRIVE_FULL.md`, `ARRIVE_PAIRED.txt`): in-anchor
**32/54** (+3 / -3 against the close run), cross-anchor **23/53** (+3 / -5).
The mechanism did exactly what the subset said: all three 00829 cracker
boxes now stop at 0.46, 0.98 and 0.99 m instead of spinning at the goal,
and "committed, never arrived" is 0. The losses are the same shape as the
radius arm's: ending the first, wrong approach a few centimetres earlier
changes the pose the search resumes from, and three cross-anchor cracker
boxes the close run found on a later attempt (0.55-0.92 m at 210-224 steps)
are now wrong commits and a budget. Median steps 244 -> 294 in-anchor.

**Decision: `island_close` stays the working configuration** (32/54,
25/53). `approach_arrival_m` stays in the code, off, as the fix for a real
defect (the spin at the goal) that this benchmark does not reward on net.

## Where the morning ends

| split | tight ring (start) | **island + close** | DualMap measured | paper | goal |
|---|---:|---:|---:|---:|---:|
| in-anchor | 24/54 = 44.4% | **32/54 = 59.3%** | 35/54 = 64.8% | 70 | 70 |
| cross-anchor | 15/53 = 28.3% | **25/53 = 47.2%** | 16/53 = 30.2% | 50 | 50 |

Both against the tight ring at p < 0.05 on the sign test (0.039 / 0.013).
Preset `dualmap_protocol_osg_look_flat_anchor_v2_island_close`; four
mechanisms carried it: the flat prior, the look before absence, the
stale-anchor policy with twin retirement, the island-constrained navmesh
snap, and the closing walk on the navmesh.

What the last three arms taught, and what it means for the next step:

1. **The benchmark has a noise floor of about +-3 trials per split for any
   change that touches where an approach ends.** Three arms whose
   mechanisms converted every trial they were aimed at (+7, +3, +4 on
   their subsets, none lost) came back even or negative on the full 107,
   each time by re-routing the exploration of trials they were not aimed
   at. Runs are deterministic ([[osg-repeat-is-exact-on-hard-subset]]),
   so this is sensitivity, not randomness: a different pose at step 100
   is a different episode. Below about four trials, the mechanism counters
   are the result and SR is not.
2. **The ceiling under the 1 m rule on this dataset is near.** Of the 22
   in-anchor failures: 12 are scissors and mug beyond a metre (detectable
   only inside ~1 m, three genuine successes prove it), 8 are objects more
   than a metre from any navigable floor point that DualMap fails too, and
   the last 2 are the stale-prior stops where the object moved away from
   the floor side. 70% in-anchor (38/54) is not attainable without either
   the substitution (DualMap rerun, ~10 h) or a different success rule.
3. **Cross-anchor is 2 trials from the paper's 50%**, and the remaining
   failures are search exposure (never in view 4, seen never named 13),
   where the noise floor is the same size as the gap.

## 20:12 — The prior map was built in the wrong world; rebuilding from the release

Probing DualMap's feature matching (`docs/history/FEATURE_MEMORY_PROBE.md`) turned up a
provenance bug that has nothing to do with features. `outputs/maps_v5` -- the
prior map every arm above loads -- is built by `scripts/campaign/rebuild_maps.sh`
with `ycb.layout_root=outputs/substituted_layouts`. Those layouts are the
released ones with `037_scissors` replaced by `021_bleach_cleanser` at the same
pose, authored deliberately for the *authored* YCB benchmark because neither
detector can name the scissors asset. The DualMap-protocol episodes inject the
RELEASED layout, so every scored run started from a map holding a bleach
cleanser where the scissors is, and no scissors at all. The stored crop 0.01 m
from each scene's scissors is visibly a bottle, at detector confidence
0.62 / 0.78 / 0.82, and the released layouts place no bleach cleanser anywhere.

The fix is the map, not the code. `rebuild_maps.sh` now takes `LAYOUT_ROOT`
(default unchanged) and `SCENES` (one process per scene, since only the episodes
*within* a scene must run in order). `outputs/collector_layouts` is the faithful
import of `data/dualmap/HM3D_collect` -- verified object-for-object and
pose-for-pose against the source, the only difference being 00848's duplicate
mug, which shares semantic id 98 with the first and which six of its seven
layouts omit. Building `outputs/maps_released` from it:

```bash
for scene in 00829-QaLdnwvtxbs 00848-ziup5kvtCCR 00880-Nfvxx8J5NCo; do
  SCENES="$scene" LAYOUT_ROOT=outputs/collector_layouts \
    scripts/campaign/rebuild_maps.sh outputs/maps_released &
done
MAP_ROOT=outputs/maps_released ARMS=flat_anchor_v2_island_close \
  TRIAL_SET=data/splits/dualmap_all.json SHARDS_PER_SCENE=2 MAX_PARALLEL=6 \
  OUT_ROOT=outputs/osg_released_maps scripts/run_close_look_ab.sh
```

The scissors mapping episode is expected to find nothing -- recall is 0.00
beyond a metre -- and that is the point: the map then holds what the detector
can really see in the world the benchmark scores, instead of a phantom.

Expect a small SR effect rather than a large one. The phantom track carries the
label `bleach bottle`, and the candidate path is keyed on the target label, so
it was never proposed as a scissors; what changes is the surface prior, the
exploration it drove, and the honesty of every scissors number on this
benchmark.

`outputs/maps_released` is built and verified: the phantom is gone from all
three scissors positions, 00829 now holds a real `scissors` track 0.01 m from
the object, and 00848 and 00880 hold the furniture it rests on (0.25 m, 0.23 m)
because the detector never named it. Track counts 682 / 800 / 618 against
maps_v5's 689 / 343 / 520 -- 00848 and 00880 are richer because the released
worlds have more objects to hunt (seven in 00848 against five), so an SR
comparison against `outputs/osg_close_full` would mix the phantom's removal with
extra mapping coverage rather than isolating one variable.

**The 107 on these maps was started and stopped by choice after ~10 minutes**,
so the SR effect of the corrected map is UNMEASURED and the maps are the
deliverable. To measure it later, and note the confound above when reading it:

```bash
MAP_ROOT=outputs/maps_released ARMS=flat_anchor_v2_island_close \
  TRIAL_SET=data/splits/dualmap_all.json SHARDS_PER_SCENE=2 MAX_PARALLEL=6 \
  OUT_ROOT=outputs/osg_released_maps scripts/run_close_look_ab.sh
python scripts/compare_close_look.py \
  --arm close=outputs/osg_close_full/flat_anchor_v2_island_close \
  --arm released=outputs/osg_released_maps/flat_anchor_v2_island_close \
  --out outputs/osg_released_maps/RELEASED_MAPS.md
```

## Appendix: presets of the arms that were not carried forward

Removed from `configs/experiment/` in the clean-up of 2026-09-10; each is its parent preset plus the overrides below, so any logged arm can be recomposed by hand (`+experiment=<parent>` and the overrides on the command line).

- `dualmap_protocol_osg_look_belief`: parent `dualmap_protocol_osg_look_flat`; `agent.close_look_opportunistic=True`; `agent.close_look_by_belief=True`; `exploration.search_glance_range_m=2.5`
- `dualmap_protocol_osg_look_both`: parent `dualmap_protocol_osg_tightring`; `agent.close_look_opportunistic=True`; `agent.close_look_before_absence=True`; `exploration.search_glance_range_m=2.5`; `eval.gt_dump_dir=`
- `dualmap_protocol_osg_look_drop`: parent `dualmap_protocol_osg_look_inanchor`; `exploration.search_drop_proximity_after_absence=True`
- `dualmap_protocol_osg_look_flat_anchor_drop`: parent `dualmap_protocol_osg_look_flat_anchor`; `exploration.search_proximity_len_m=1.0`; `exploration.search_drop_proximity_after_absence=True`
- `dualmap_protocol_osg_look_flat_anchor_stop2`: parent `dualmap_protocol_osg_look_flat_anchor`; `verification.abandon_below_p=0.3`
- `dualmap_protocol_osg_look_flat_anchor_v2_clamp`: parent `dualmap_protocol_osg_look_flat_anchor_v2`; `scene_graph.presence={'l_clamp': 2.0}`
- `dualmap_protocol_osg_look_flat_anchor_v2_nf`: parent `dualmap_protocol_osg_look_flat_anchor_v2_restrike`; `agent.reachable_via_nearest_free=True`
- `dualmap_protocol_osg_look_flat_anchor_v2_rankp`: parent `dualmap_protocol_osg_look_flat_anchor_v2`; `verification.rank_candidates_by_presence=False`
- `dualmap_protocol_osg_look_flat_anchor_v2_restrike`: parent `dualmap_protocol_osg_look_flat_anchor_v2`; `agent.unreachable_restrike_m=1.0`
- `dualmap_protocol_osg_look_flat_anchor_v2_island_relook`: parent `dualmap_protocol_osg_look_flat_anchor_v2_island`; `verification.relook_after_stale_stop=True`
- `dualmap_protocol_osg_look_flat_stop`: parent `dualmap_protocol_osg_look_flat`; `verification.stop_at_stale_anchor_once=True`
- `dualmap_protocol_osg_look_flat_twins`: parent `dualmap_protocol_osg_look_flat`; `verification.retire_stale_twins_after_absence=True`
- `dualmap_protocol_osg_look_len4`: parent `dualmap_protocol_osg_look_inanchor`; `exploration.search_proximity_len_m=4.0`
- `dualmap_protocol_osg_look_opportunistic`: parent `dualmap_protocol_osg_tightring`; `agent.close_look_opportunistic=True`; `exploration.search_glance_range_m=2.5`; `eval.gt_dump_dir=`
- `dualmap_protocol_osg_look_flat_anchor_v2_island_close_arrive`: parent `dualmap_protocol_osg_look_flat_anchor_v2_island_close`; `agent.approach_arrival_m=0.3`
- `dualmap_protocol_osg_look_flat_anchor_v2_island_close_arrive_r10`: parent `dualmap_protocol_osg_look_flat_anchor_v2_island_close`; `agent.approach_arrival_m=0.3`; `agent.agent_radius=0.1`
- `dualmap_protocol_osg_look_flat_anchor_v2_island_close_r10`: parent `dualmap_protocol_osg_look_flat_anchor_v2_island_close`; `agent.agent_radius=0.1`
