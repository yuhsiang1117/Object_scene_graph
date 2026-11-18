# Why the agent does not climb

Three questions were open after the fused pipeline came back a null: why 55
switch attempts produce 0.17 m of ascent, whether a remembered staircase can be
reached when the gate refuses, and whether the cross-floor sample can be made
bigger. All three are answered.

## The sample, corrected

Counted from the authored layouts, over all 15 scenes and all three layout
indices:

| vertical move | count |
|---|---:|
| more than 0.5 m | 19 |
| more than 1.8 m, a real storey | 13 |

The first count was too loose: a 0.84 m move is floor to shelf, not floor to
floor. 1.8 m is `floor.new_level_m`, the threshold the estimator itself uses.

Every one is in `cross_anchor` at layout index 1. An in-anchor move is never
cross-floor, by construction. Five prior maps were built for this
(`scripts/legacy/build_crossfloor_maps.sh`), taking the runnable set from 7 episodes on
4 scenes to **12 on 6**. Of the remainder, 00824 and 00871 have no storey change
at all, and 00869's single episode cannot be constructed.

## The answer: the climb was delegated to something that was removed

`portals.py` and `docs/history/MULTI_FLOOR.md` both define a portal as "not the staircase
itself, but a place to walk toward, **after which the navmesh handles the
climb**". Commit 866ab0e took the YCB line sensor-only on 2026-09-08, replacing
habitat's follower with a PointNav policy, and nothing downstream was changed.
The step that was going to do the climbing no longer exists.

A `Portal.centroid_xy` is the mean grid position of cells whose only observed
surface is a storey away. That is a point under or across from the other floor,
not a stair mouth. The per-floor costmap is 2D and cannot represent a ramp, so
the PointNav policy has nothing to follow upward.

Restoring the navmesh tests this directly. On 00821's cracker box, the cleanest
episode in the set:

| arm | switch attempts | final height | climbed | reached the floor |
|---|---:|---:|---:|---|
| base | 55 | -3.47 | 0.17 m | no |
| v8 | 53 | -3.47 | 0.17 m | no |
| navmesh | **1** | **+0.13** | **3.60 m** | **yes** |

One attempt instead of fifty-five, and the whole storey.

## The campaign

Six scenes, 13 paired episodes, 12 of them genuine storey changes.

| | base | v8 | navmesh |
|---|---:|---:|---:|
| success, cross-floor | 0/12 | 0/12 | **2/12** |
| reached the object's floor | 2 | 2 | 2 |
| climbed more than 1 m | 1 | 2 | 3 |
| floor switch attempts | 76 | 60 | 9 |

Four episodes climbed under the navmesh and did not move at all without it:
00878's tin can rose 3.0 m against 0.0, its coffee can 2.8 m against 0.0,
00821's cracker box 3.6 m against 0.17, 00800's cracker box 0.82 m against 0.0.
The navmesh also lost one: 00808's yellow bottle climbed 3.2 m under base and
0.23 m under the navmesh.

**The navmesh arm is a diagnostic and never a headline.** It hands the agent
habitat's ShortestPathFollower planning on the ground-truth mesh and
`is_reachable` as an oracle, neither of which a sensor-only system has. Its two
successes measure the size of the gap, not progress against it.

Climbing is necessary and not sufficient: the navmesh arm reached 00821's upper
floor and still ended 7.98 m from the goal with the target never in view.

## What the two code changes were worth

Both fire and neither converts.

- `floor.prior_stairs_override_gate` (v7) lets a remembered staircase stand in
  for the "nothing near is left here" clause, which a large storey never
  satisfies. It fired twice. Those episodes reached `find_portals` instead of
  returning early, which is what it was for.
- `floor.portal_failure_memory` (v8) stops a portal that led nowhere being
  proposed again. It skipped 61 re-proposals and remembered failures in two
  episodes, taking climbs over a metre from 1 to 2. On 00821 it correctly
  dropped the failed patch and had nothing better to offer, because that scene's
  prior map is single-storey and holds no recorded traversal.

`floor.prefer_prior_stairs` (v6) stays off: it fired 106 times and took
reached-the-floor from 2 to 1.

## What would actually fix it

A sensor-only agent needs a traversable representation of stairs, not a better
portal or a better prior. Three candidates, in the order their evidence supports:

1. **A climb behaviour.** When a pursuit reaches a portal and stops making
   vertical progress, the failure is that forward is blocked by a step the
   policy will not take. A scripted ascent at the foot of a detected stair
   region is the smallest thing that could work.
2. **A multi-level costmap.** Per-floor 2D grids cannot express the connection
   between them. The stair cells are already detected; they need to be
   navigable, not masked out.
3. **Stair-aware goals.** Aim at the stair region's lower mouth rather than at
   the patch of upper floor that is visible from below.

Until one of those exists, cross-floor numbers on the sensor-only line measure
an agent that cannot ascend on purpose, and no work in the floor decision layer
can register.

## The climb, built and measured

Everything above was diagnosis. This is what happened when the sensor-only
climb was actually built.

**What was there.** `State.CLIMB` was in the enum; `_carrot_action`,
`_on_a_staircase` and `_left_the_stairs` were on `NavAgent`; the `ascent` policy
dispatched to a `_do_climb` that was never defined; nothing assigned the state
or any attribute it read. `StairDetector.accumulate` ran only on the one frame
of a periodic look-down, `down_look_every` defaulted to 0 so it never ran on any
YCB preset, and `StairDetector.extract` had no caller at all. In short, no
sensor-only climb existed, and no stair evidence was ever collected.

**What was built**, one arm at a time, each default-off:

| arm | adds |
|---|---|
| v9 | `agent.climb_enabled`: a pursuit that reaches the stairs becomes a climb (ASCENT's depth-ray carrot, tilt-down on descent, exit on committed storey / budget / stall, failure → portal memory). `floor.climb_targets: stairs_first`. |
| v10 | stair evidence on every keyframe; look down every 30 steps; extracted up/down regions offered as direction-aware targets |
| v11 | entry reach outside the mover's own 0.9 m stop radius; `floor.hold_pursuit` so a pursuit in flight is not re-issued every round (which is also what base's 55 attempts were) |
| v12 | carrot aimed at the detector's stamped stair cells; a turn after a run of blocked forwards; budget sized for a switchback |

**What it did**, six scenes, twelve genuine storey changes, paired:

| | base | navmesh | v10 | v11 | v12 |
|---|---:|---:|---:|---:|---:|
| success, cross-floor | 0 | 2 | 0 | 0 | 0 |
| reached the object's floor | 1 | 1 | 0 | 0 | 0 |
| climbs started | 0 | 0 | 4 | 11 | 11 |
| climbs that committed a storey | 0 | 0 | 0 | 0 | 0 |
| most height gained inside a climb | 0 | 0 | 0.34 m | 0.34 m | 0.34 m |
| floor switch attempts | 76 | 9 | 96 | 18 | 15 |
| look-downs | 0 | 0 | 87 | 83 | 69 |
| stair regions offered | 0 | 0 | 17 | 18 | 21 |

Every mechanism is live and measurable. Eleven climbs engaged, each ended
properly and was remembered, switch attempts fell from 76 to 15, the agent
looked down 69 times and was offered 21 stair regions. And the best any climb
managed was 0.34 m.

**Why it does not ascend.** On 00821's cracker box, where the navmesh climbs
the 3.6 m storey in one attempt, the stair target the detector produced sat at
(2.98, 2.62) against a true foot near (0.37, 2.0): about 2.7 m to the side of
the flight. v12's cell carrot then spent 386 steps steering at stamped cells
around that wrong position, with no forced-forwards logged, so the mover was
moving and simply never on a tread. The detector's `stairs` mask, seen from
below, stamps cells under the upper flight rather than at its foot. The depth
ray (v9-v11) has the same problem from a different direction: it works from ON
the flight, and the agent is never on it.

**The other half never opens.** On 00878's tin can, where the navmesh climbs
3.0 m and the prior map records two traversals, no sensor-only arm made a single
switch attempt. The posterior asked once, the model answered "stay", and that
settled it; the geometric gate stayed closed; and the gate override never ran
because the agent spent almost all 500 steps in approach and surface states,
where the storey question is never asked. The commit table says why: on all
twelve episodes the first commit is to a live detection of the target on the
start floor, a false positive on the wrong storey, and the prior anchor is
chosen in three.

**What is worth keeping.** v11's plumbing: reach outside the mover's radius,
`hold_pursuit`, portal failure memory, evidence every keyframe, look-down. Those
are correct regardless of the carrot and they cut the thrash from 76 attempts
to 15. Leave `climb_targets: stairs_first` and `climb_cell_carrot` off until the
stair evidence can be shown to land at a foot: on the two scenes checked it
does not.

**What would actually climb.** The navmesh arm shows the rest of the pipeline
can reach the floor once a mover can take stairs. The sensor-only mover cannot:
a frozen PointNav policy trained on flat point-goals treats a flight as a wall,
and the only thing that has moved an agent up one here is the depth carrot for
0.34 m. Three options, in order of evidence: a local planner on a costmap that
represents the flight as traversable (the stair cells are already detected;
they are masked out, not used); a stair-following controller that keeps the
agent square to the risers and pushes; or stair evidence taken from the height
layer's rising-tread signature rather than from the detector's mask, so the
target is the foot and not the balustrade.

## The structural change: flights, not portals

The target was the problem all along. A portal is a place another storey is
*visible from*; the detector's `stairs` mask, seen from below, stamps cells
*under* the upper flight. Neither is a place you can walk up from. Put the goal
on the treads and the PointNav mover climbs, slowly.

`mapping.stairs.find_flights` reads a staircase off the height layer:
connected cells whose height sits strictly between this floor and the next,
kept only when their heights span at least a metre. A table is a plateau and
spans nothing; a sloped floor is capped by size; a flight that joins two
storeys spans a metre by construction. The lowest tread is the foot.

This is the question Stage 4's detector should have asked. It looked for
steppable cells by their local height step, and a tread's interior is flat, so
every flight fragmented into riser edges. Asking which cells are *between two
storeys* keeps the treads whole.

`floor.climb_targets: flights_first` pursues the nearest flight going the right
way, marks its treads traversable, and hands the flight to the agent.
`agent.climb_flight_carrot` then aims each climb step at the tread 0.35-1.0 m
above the agent's standing height, so the goal stays on the flight the whole
way up.

### It climbs

| | base | navmesh | v11 | v13 | v15 |
|---|---:|---:|---:|---:|---:|
| success, cross-floor | 0/12 | 2/12 | 0/12 | 0/12 | 0/12 |
| climbs that committed a storey | 0 | 0 | 0 | **2** | **1** |
| most height gained inside a climb | 0 | 0 | 0.34 m | **2.34 m** | 0.69 m |
| flights found / pursued | 0 | 0 | 0 | 24 / 10 | 24 / 9 |
| floor switch attempts | 76 | 9 | 18 | 15 | 14 |

v13 is the first sensor-only arm whose climb reaches another storey. On 00808's
yellow bottle it descended the flight the height layer found and the estimator
committed the object's floor at step 314.

### Two things found on the way down

**It left again immediately.** At step 347 the original geometric gate --
"nothing near is left to explore" -- was true, because a floor reached 33 steps
ago has almost no map, and the agent climbed back up (`floor_log [1, 0, 1]`).
`floor.dwell_on_arrival` gives a newly reached storey the dwell the episode
start gets. That needed a second fix: "steps on this floor" was measured from
`first_step`, which is 0 for a storey restored from the prior map, so a 33-step
arrival read as 347 steps. `FloorStack.arrived_step` now records each arrival
and the guards count from it. With both, v15 descends and stays
(`floor_log [1, 0]`).

**It stops on a landing.** v15 ends that episode at y = 0.946 with the object at
y = 0.06: it descended 2.3 m of a 3.2 m storey, reached a half-landing, and the
floor estimator committed the landing as a storey -- which ends the climb, and
`goal_floor` then cannot be matched at all. This is exactly the risk
`docs/history/MULTI_FLOOR.md` lists under "Biggest risk": a multi-flight landing looks
like a new floor to any height-based estimator.

So the ascent primitive exists and is the right shape, and the remaining loss is
that one flight is not one storey. The next change is in the estimator, not the
climb: do not commit a level while the agent is standing on cells that belong to
a flight, and continue the climb through the landing to the next flight's foot.

## Not stopping in the middle of the stair

v13-v15 climbed and stopped on a half-landing: 2.3 m of a 3.2 m storey, the
agent standing at y = 0.946 with the object at y = 0.06. Two independent
mechanisms put it there, and both had to go.

**A landing became a storey.** It is off every known level, and roomy enough to
walk 2.5 m across, so `FloorEstimator`'s horizontal-run route created a level
there. `floor.no_level_on_flight` refuses to create a new level while the agent
is on a staircase. Standing on treads turned out not to be enough, because a
landing is flat ground *between* two flights: the flag is now set for the
duration of the climb. Arriving on a storey the stack already knows is
deliberately untouched, so a real arrival still commits.

**A landing redefined the floor below.** With new levels suppressed the climb
still ended, because `_refine` pulls each level toward the heights actually
stood on and a landing 0.89 m above floor 0 sits inside its capture radius.
Floor 0's estimate drifted *up to meet the agent*, which then "arrived" on a
storey it was not standing on. `_refine` is now skipped while the same flag is
set. Treads and landings are not floor and must not define where floor is.

With both, and `agent.climb_relink_flights` picking up the next flight when the
treads run out, the descent completes:

| 00808, yellow bottle | v13 | v15 | v16 |
|---|---:|---:|---:|
| descended | 2.37 m | 2.32 m | **3.20 m** |
| ended at | y 2.861 | y 0.946 | **y 0.061** |
| reached the object's floor | no | no | **yes** |
| distance to goal | 7.28 m | 6.56 m | **0.25 m** |
| target in view | 0 frames | 0 frames | **22 frames** |

Across the twelve genuine storey changes, v16 relinks flights 4 times and
suppresses 27 landing levels. Its remaining loss on that episode is not the
climb: it ends 0.25 m from the object with the target in view for 22 frames and
does not stop, which is the approach-termination problem, not a floor one.

Cross-floor success is still 0/12. Nine of the twelve never start a climb at
all, because the storey question is only asked in exploration rounds and those
episodes spend their budget in approach and surface states chasing a live false
positive on the start floor. That is the next thing to fix, and it is in the
decision layer rather than in the stairs.
