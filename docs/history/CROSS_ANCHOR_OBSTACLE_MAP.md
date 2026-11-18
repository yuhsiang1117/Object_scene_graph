*For the summary -- findings, fixes, what is left, roadmap -- read `docs/history/CROSS_ANCHOR_STATUS.md`. This file is the chronological log.*

# Navigating the presence pipeline on ASCENT's obstacle map

## THE PASTE WAS TRANSPOSED. Everything above about where things are was measured on a map 8-9 m from the truth.

This is the finding that outranks the rest of this document, found last, and
it invalidates the geometric readings above -- kept, with this on top, because
each was stated with numbers at the time.

**What was wrong.** vlfm's `BaseMap._xy_to_px` returns GEOMETRIC pixels,
`(row, col) = (size - y*ppm - origin, x*ppm + origin)`, and every point it
records -- `robot_px`, the stair endpoints, the frontiers -- is in that frame.
Its GRIDS are written `_map[px[:, 1], px[:, 0]]`: indexed `[col, row]`. On disk
the arrays are the transpose of the points. `apply_to_costmap` read both as
geometric, so the occupancy, the explored area and the stair masks all landed
transposed about the episode origin.

**How it was measured**, against the navmesh, which is the one frame that
cannot be mis-read -- navigable points bucketed by storey, and the treads
between (0.5-2.8 m), looked up in the pasted costmap:

| navmesh points land on the paste | as stored | transposed |
|---|---|---|
| lower storey  FREE / OCCUPIED / UNKNOWN | 0.04 / 0.04 / 0.92 | **0.49 / 0.00 / 0.51** |
| upper storey  FREE / OCCUPIED / UNKNOWN | 0.20 / 0.05 / 0.75 | **0.80 / 0.00 / 0.20** |
| treads on the pasted stair mask | 0.00 | **0.36** |

Zero navigable points on obstacles is the signature of the right orientation;
the UNKNOWN that remains is coverage. ASCENT's own trajectory agrees: 118 poses
within 1.5 m of the navmesh staircase and none within 5.9 m of the stair mask
as stored; in pixels, the walked path touches the stored mask 0 times and the
transposed mask 146 times.

**Why the earlier validation did not catch it.** "81.6% of the trajectory on
free cells" looked the map up through vlfm's own indexing and so confirmed
the map was consistent with itself, not with the world. The unit tests that
"check the inverse against the forward at four headings" planted their marks
in geometric `[row, col]`, which vlfm never writes, and so tested the POINT
transform, which was right, not the ARRAY layout, which was not.

**Consequences for what is written above.**

* The staircase at (-7, -12) does not exist. The navmesh has ONE traversable
  flight on 00800, at x -12..-9, z -4..-5, and that is where ASCENT climbed
  (its recorded endpoints, read as the geometric (row, col) they always were,
  land at (-10.12, -2.46) and (-11.41, -4.23)). Every climb from v4 to v9 was
  attempted at a location with no treads, and `flights_prefer_stair_mask` --
  the ranking built to steer the agent onto ASCENT's stairs -- is what steered
  it there.
* v3's "furniture" flights at (-9.42, -3.32) and (-11.02, -3.82) were the real
  staircase. They failed for the climb-direction bug, which was real and is
  fixed.
* The ramp-margin finding stands: it was measured on the navmesh's own flight
  with `probe_climb_osg.py`, not on the paste.
* The stair-endpoint "convention test" that concluded (col, row) was wrong: it
  compared the endpoints to a mask that was itself transposed. They are
  (row, col). Reverted.
* The map figures earlier in this document, and the coverage readings drawn
  from them, are of the transposed map.

**The fix** is `_to_geometric_layout` in `navigation/mapping/map_store.py`:
every square grid is transposed ONCE at load, so the paste, the figure and the
summaries all see geometric arrays, and `apply_obstacle_maps` hands vlfm's
layout back when restoring onto an `ObstacleMap`. Through the real paste path
now: lower storey FREE 0.49 / OCCUPIED 0.00, upper FREE 0.80 / OCCUPIED 0.00,
treads on the stair mask 0.36, and the flights `find_flights` extracts have
their feet 0.23 m and 0.14 m from real navmesh treads.

### The ramp climbs the REAL pasted flight, both ways -- once oriented by the walk

Three things stood between the paste and a climb, found in this order with
`scripts/probe_climb_osg.py`, each on the navmesh's own flight until the last:

1. **Margin.** The ramp must start AT the storey (`RAMP_MARGIN_M = 0.0`);
   any lift at the foot puts the agent's own cell in the carrot's 0.35-1.0 m
   band. Ascending: 0.00 m -> +1.83 m.
2. **Landings.** A ramp cannot know where the flat parts are. Descending from
   a landing, the ramp says the floor has dropped, the agent walks 0.5 m
   without descending, and the nearest "cell below" is under its feet
   (goal distance 0.68 -> 0.10 m, stalled at -0.12 m).
   `agent.climb_carrot_min_ahead_m` (0.4 in the arm) refuses any cell closer
   than that horizontally; with true heights it changes nothing (goals were
   0.66-1.01 m ahead throughout). Descending: -0.12 m -> -1.89 m.
3. **Orientation.** The pasted ramp was REVERSED: correlation with the true
   tread heights -0.91 (up) and -0.86 (down). The stored endpoints run from
   the top of the real staircase to the bottom on this map, and free-floor
   adjacency cannot tell the ends apart because ASCENT's lower-storey map
   bleeds up the stairs onto the upper landing (0.61 vs 0.47). What cannot be
   wrong is the ORDER the agent walked the treads in: floor 2's trajectory has
   143 poses on the flight, entry t=0.33, exit t=2.12. `_stair_ends_from_trajectory`
   orients this storey's ramp by that and mirrors it for the other storey,
   because it is one staircase. Correlation after: +0.99 and +1.00.

With all three, the production climb on the map exactly as `load_obstacle_map`
pastes it -- `variants pasted` -- from a start 1 m from the flight:

| | result |
|---|---|
| ascending, lower -> upper | **+1.86 m in 46 steps**, `storey_of_height` |
| descending, upper -> lower | **-1.85 m in 50 steps**, `storey_of_height` |

The first climb of an ASCENT-carried staircase by the OSG agent, in either
direction.

### First storey changes in a run (outputs/mf5_pass2_v11), and the two defects they exposed

| | toy airplane (upper -> lower) | cracker box (lower -> upper) |
|---|---|---|
| flight chosen | (-8.38, -4.88), the real staircase | (-11.17, -4.32), the real foot |
| climb | 200 steps, -1.51 m, ended on budget | **64 steps, +1.83 m, `climb_ok`** |
| storey change | yes, at step 368 (y 0.56) | crossed twice: up 2.15 m, back down |
| `goal_floor_reached` | **True** | False |
| distance to goal | **4.3 m** (was 23-26) | 22.5 m |

Both still 0: each spent its last attempt on a same-storey fake right after the
climb ended.

v12 ep1 then ARRIVED -- `climb_end_new_floor` at step 387, the lower storey
committed at y=0.17, 113 steps in hand, 54 same-storey fakes refused on the
way -- and at step 390 the search posterior selected the UPPER storey, the one
just disproved and left. The agent climbed back up (`climb_ok` twice in one
episode) and ended where it started. `forced_floor` had done its job and been
cleared on arrival; the mass rule was then free to choose the disproved storey.
The verdict now covers the posterior: `ExplorationStrategy.disproved_floors`
(the agent's own set, shared) is excluded from the storey argmax
(`floor_posterior_disproved_excluded`).

**Why the posterior sent it back: the storey it arrived on was EMPTY.** The
scene-graph snapshot stores tracks for both storeys (269 / 258 on 00800) but
no containers and no rooms -- those are rebuilt from tracks at keyframes, on
whichever storey the agent is on. Three steps after climbing down there had
been no keyframe there, so `floor_mass={'0': 11.8}`: the storey just left was
the only one with any mass. `floor_mass_margin` cannot see this (0 x margin
is 0), and the disproved-storey exclusion alone would have fallen back to the
same set. By the end of that episode the lower storey had 29 containers.

An earlier, deliberate decision says the opposite for a storey the agent has
BEEN on -- "with nothing at all here, anywhere else is better"
(tests/unit/test_floor_anchor_order.py) -- and it is kept. The two are told
apart by time since arrival: `exploration.empty_storey_settle_steps` (60 in
the arm, 0 shipped) makes an empty posterior mean "unmapped" for that long,
and "not here" after. `floor_posterior_empty_here` counts the former.

Still open from these runs: the first descent took 200 steps for 1.51 m where
the isolated probe takes 50 for 1.85 m. Something in the run's climb loop is
four times slower than the same code driven from the probe; not yet
understood.

* ep1: the climb ended on `climb_max_steps` at -1.51 m; three steps later,
  with `pursuing` False and `current_id` still the upper storey, the agent
  committed to an upper-storey track at y=4.0 from the treads. The storey was
  disproved; the verdict now covers the candidate channel outright
  (`candidates_skipped_disproved`), pursuit or not.
* ep2: the climb succeeded -- and was declared a storey at 1.83 m, because
  `floor.new_level_m` is 1.8 and 00800's storeys are 3.0 m apart. The
  estimator never committed the upper storey, the posterior picked a cabinet
  on the LOWER one, and the agent walked back down the stairs it had just
  climbed. `agent.climb_to_target_storey_tol_m` (0.3 in the arm) makes the
  climb close the gap the prior map knows, `|_goal_floor_y_cache - here|`,
  instead of a constant.

### v13 (outputs/mf5_pass2_v13): stays on the storey; the climb budget decides ep2

* ep1 (toy airplane): arrived on the lower storey at step 387, stayed
  (`floor_posterior_disproved_excluded` 3, `candidates_skipped_disproved` 10),
  searched a bed and a cabinet, and spent its last attempt on "toy airplane"
  id 578 -- 2.2 m above the floor, one observation, score 0.77. Ended 13.8 m
  west of a target it never saw.
* ep2 (cracker box): the 200-step climb budget ended the ascent at +1.97 m of
  the 2.7 m the gap rule asks for; the upper storey was never committed; a
  lower-storey fake pulled it back down; and `portal_failure_memory` then
  excluded the real staircase (`portals_skipped_after_failure` 3).
* `probe_climb_osg.py --full-fsm` (the whole `agent.act()` around the climb)
  is IDENTICAL to the direct `_do_climb` probe: 50 steps, +2.74 m, the same
  22/12/16 actions. The run's 4x slower climb is not the FSM wrapper.

### The climb stalled where the flight ended: `find_flights` cut the 3.0 m ramp at 1.6 m

Read off v13's video frame by frame: in ep1 the view is the same banister,
camera pitched down, from step 197 to 317; in ep2 the same railing from 244 to
304. The agent was not moving. The suspicion was a committed fake on the
origin storey pulling it back; the code rules that out -- in the CLIMB state
`candidates.check` is not run and `_do_climb` never reads the commit. The
only goal in that state is the carrot.

Replaying `_flight_carrot`'s rule offline on the stored map as pasted
(scratch `flight_geom.py`; `apply_to_costmap` + `find_flights` with the run's
settings):

| storey | flight | heights kept | ramp spans |
|---|---|---|---|
| upper, "down" | 357 cells | 2.92 -> 1.56 | 3.16 -> 0.16 |
| lower, "up" | 544 cells | 0.37 -> 1.76 | 0.16 -> 3.16 |

`find_flights` keeps `rel` strictly inside `(0.2, new_level_m - 0.2)` and
`new_level_m` is 1.8. So the flight had waypoints for the first 1.6 m; past
it the band is empty, `_relink_flight` (same cut) finds nothing, and the
cascade falls to `_update_carrot` -- ASCENT's farthest-depth-ray carrot. From
the middle of this staircase, beside a balustrade, the farthest point in
view is through the balusters, and the point-goal policy pushed into the
railing until the 200-step budget ran out. The stall heights are the cut:
ep1 `climb_max_dy_x100` 151 (flight bottom at 1.60 below the storey), ep2
197 (flight top at 1.60 above, plus momentum). The probe cleared the same
flight because it started 1 m off the treads facing along them: the depth
ray happened to point up the stairs.

Fix: `floor.flight_span_from_levels` (`FloorPolicy.flight_span_m`): the span
asked of `find_flights` is the gap to the nearest other known level, never
below `new_level_m`; used by the flight ranking, the mid-climb relink and the
probe's `pasted` variant. The heights ARE known here -- the prior map seeds
both storeys at episode start (v13 `floor_log`: 3.163 and 0.17 at step 1) and
the ramp was written from those two numbers. Without a prior there is no
ramp either, and the constant stands. Replayed with it: both flights span
0.36..2.96 m, a band waypoint at every tenth of the climb, `top/highest` at
the last, and the 2.7 m end rule is reachable. Five unit tests
(`test_flight_span.py`); goldens regenerated and verified by strip-and-rehash
(85/85 unchanged without the key; only `mf5_osg_on_ascent_map` sets it).
v14 = v13 + this flag.

### v14/v15: the widening had to be restricted, and a measurement of my own was wrong

v14 (span = the known storey gap, every cell) SPLIT: ep2 climbed and the
estimator committed the upper storey for the first time (`climb_end_new_floor`,
+2.62 m, ended 5.0 m from the target against 14.3 m in v13). ep1 did not climb
at all -- 392 steps walking to a goal it never reached, a regression from v13.

Cause: from the upper storey the agent sees the LOWER one over the banister.
Furniture tops there sit 2.0-2.8 m below the upper floor -- outside a 1.8 m
band, inside a 3.0 m one -- and they touch the stairwell in projection, so
`ndimage.label` merged them with the staircase into a 1460-cell component
whose mouth was 3 m from the real one. Fix: `find_flights(wide_span_m,
wide_mask)`; only cells the stair map vouches for may cross the old constant.
v15: ep1 descends 2.44 m of 3.0 m (was 1.51) and dies on the 200-step climb
budget while still moving, so the budget is now 300.

### THE COVERAGE NUMBERS I FIRST REPORTED WERE MIRRORED IN z

`PLANE = (0, 2)`: an OSG world xy is `(x, z)`. ASCENT's frame is `(x, -z)` and
`map_store` flips it once, at the paste. My analysis scripts flipped it a
second time, so every authored target was reflected about z=0 and I reported
the toy airplane as 8.0 m outside the map when it is 5.1 m, with a picture to
match. The pipeline was never affected -- only the measurement. Corrected, the
navmesh check agrees with the one in this log: lower FREE 0.51, upper 0.80.

### Coverage is set by WHICH mapping episode is kept, not by its budget

Measured on 00800, all 12 authored target positions against the stored map:
episode 1's toy airplane 5.1 m outside the mapped area, four others 2-5 m
outside. A 2000-step mapping pass covered LESS (28% / 27% of navigable area
against 51% / 80%) -- a different starting episode, so the two confound, but
more budget is clearly not the lever.

The lever: the protocol runs one mapping episode per authored object, six
here, each starting somewhere different, and the writer keeps only whichever
saw the most. Five explorations are discarded. On this scene they are:

| episode | storeys mapped | cells |
|---|---|---|
| 50001 | 1 | 19427 |
| 50002 | 1 | 13104 |
| 50003 | 1 | 10669 |
| 50004 | 2 | 19787 + 9324 |
| 50007 (the one kept) | 2 | 10652 + 19298 |
| 50008 | 1 | 11051 |

`ycb.obstacle_map_union` keeps them all and pastes them into the same
costmaps. Four of the six mapped a SINGLE storey, and order matching cannot
place those -- `_floors.append` on an up-stair and `insert(0, ...)` on a
down-stair means the index is meaningful only WITHIN one episode. So the
snapshot schema now records `floor_y` per floor: the mean height the mapping
agent stood at while on it, off the stairs, accumulated in the floor dict so
`insert(0, ...)` carries it along. It is written and never read by the control
flow, and all 172 ascentnav fidelity tests still pass. Pass 2 matches floors
to storeys by that height within 1 m, falls back to order for older
snapshots, and leaves an unmatchable floor OUT (`union_skipped`) rather than
forcing it onto a storey -- which is the transposition class of bug and the
one this path must not have twice.

Measured, all six snapshots matched by height and none skipped:

| storey | navigable area known, single best | UNION of six |
|---|---|---|
| lower | 0.51 | **0.81** |
| upper | 0.80 | **0.86** |

and every one of the 12 authored target positions is now within 1.2 m of
mapped free space; episode 1's toy airplane goes from 5.1 m outside the map to
0.4 m. v16 = v15 + the union.

### Two episodes cannot measure a climb change (v17-v21)

The climb eats the step budget -- 300 of 500 steps in v16 -- so four changes
were tried against it. What the runs actually established is a METHOD problem.

| run | change | descent | ascent |
|---|---|---|---|
| v17 | hold the carrot until reached | 623 climb steps, 107 forced-forwards | arrives step 161 (was 202), 1 policy reset in 350 steps |
| v18 | + release a tread the mover refuses | 329 steps, 6 forced-forwards | unchanged, byte-identical to v17 |
| v19 | + suppress turns inside a 15 deg deadband | 328 of 332 turns suppressed; 340 forwards produced 5.4 m of path -- pressing into the banister | never reached the storey |
| v20 | + yield the deadband when forward stalls | reversals 55 -> 39, distance 10.7 -> 10.6 m | arrives step 880, distance 8.3 m |
| v21 | deadband off, align to the flight first | byte-identical to v18 (alignment never fired) | arrives step 513, distance 12.7 m |

v21 differs from v18 by ONE ACTION: a single alignment turn at the first climb
step of episode 2. Episode 1, where the alignment never fired, is identical
step for step. Episode 2 diverges completely from that one turn -- 161 vs 513
for the arrival, 350 vs 606 climb steps. With two episodes, a chaotic
simulator and no repeats, a run-level number cannot tell a real climb effect
from that.

So: **measure climb changes with `scripts/probe_climb_osg.py`**, which starts
from a fixed pose on the pasted flight and reports steps and metres, and only
then bring the winner into the arm.

What is kept, on evidence: the carrot hold (the ascent's policy resets went to
1 in 350 steps) and the blocked-tread release (forced-forwards 107 -> 6). What
is off, in the code with its measurement beside it: the turn deadband (v19,
v20) and the align-first phase (v21, and its first version aligned to a tread
0.4 m away whose bearing swings tens of degrees for a few centimetres -- it
spun a full revolution without converging; it now aligns to the flight axis).

### The cracker box gave up on the storey the target was on

v18 episode 2: climbed to the CORRECT storey at step 161, searched three
containers, and from step 333 to 841 made six descent attempts, every one
ending in `portal_end_no_vertical_progress`. 508 steps, half the episode,
trying to leave. Three frontier selections in 1000 steps. Directed switches
(the posterior's or the storey LLM's request) bypass `may_switch` entirely, so
none of the dwell or interval guards applied. Fix:
`agent.max_failed_switches_per_storey` (2) with `agent.switch_ban_steps` (250)
-- the ban clears the standing request too, or the posterior re-raises it next
round, and it expires so a genuinely needed switch is only delayed. Not yet
exercised: it did not fire in v20 or v21.

The summary, open items and roadmap are in `docs/history/CROSS_ANCHOR_STATUS.md`.

## What this is for

The cross-anchor benchmark asks an agent to re-find an object that moved while
it was not looking. Until now one pass built everything: an agent walked the
static scene and left a snapshot holding both the scene graph (object tracks,
presence beliefs, rooms) and the per-storey costmap, and pass 2 navigated from
that single file.

This splits the two. The occupancy now comes from **ASCENT's navigation**
(`src/navigation/`), which explores better than OSG's on this repo's own
measurements, while the scene graph still comes from the OSG pass. Pass 2 then
plans over a map it did not build.

| | builder | artifact | carries |
|---|---|---|---|
| 1a | `ascentnav` | `ycb.obstacle_map_out` | per-storey `ObstacleMap`: obstacle, explored and stair masks, the episode anchor |
| 1b | `nav_agent` | `ycb.map_out` | object tracks, presence beliefs, storeys, connectivity |
| 2 | `nav_agent` | reads `ycb.obstacle_map_in` + `ycb.map_in` | navigates the moved layout |

`ycb.map_in_occupancy=false` is what makes 2 meaningful: the scene-graph
snapshot supplies the objects and the storeys and **none** of its occupancy, so
every cell the planner reads comes from ASCENT's map.

```bash
bash scripts/legacy/run_mf5_mapping.sh 00800-TEEsavR23oF     # 1a then 1b, one scene
python scripts/check_obstacle_map_reuse.py \
    --maps outputs/maps_mf5_ascent --run outputs/mf5_stage1/00800-TEEsavR23oF_ascent \
    --png /tmp/map.png                                # look at the picture
python scripts/run_eval.py +experiment=mf5_osg_on_ascent_map \
    'ycb.scenes=[00800-TEEsavR23oF]' \
    ycb.obstacle_map_in=outputs/maps_mf5_ascent \
    ycb.map_in=outputs/maps_mf5_osg ycb.map_in_occupancy=false
```

## The frame, which is the whole risk

`navigation/mapping/map_store.py` stores the stack; `apply_to_costmap` puts it
into an OSG `Costmap2D`. That is a **resample, not a copy**, because three
things separate the lattices: `BaseMap._xy_to_px` swaps the axes and flips the
row, ASCENT's frame is `(x, -z)` where OSG's `PLANE` is `(x, z)`, and the map is
rotated to the episode's start facing. Copying with a shifted origin -- all
`_CostmapView` does, and it is only ever drawn -- puts the walls **10 m out at
heading 0 and 22 m at pi/2** (measured).

What pins it: `tests/unit/test_navigation_map_store.py` checks the loader's
inverse against the obstacle map's own forward projection at four start
headings, and the `--png` is how a human confirms it on a real pass. On 00800,
bowl's map contains 81.6% of bowl's own path as free space and the trajectory
runs down its corridors.

## Three findings that cost time, recorded so they do not again

**`traj_on_map` is a coverage diagnostic, not a validity test.** It measures
whether the agent stood on its own *explored* area, and `explored_area` is what
the agent SAW: `obstacle_map.py:374` erases every cell within an agent-radius
dilation of an obstacle on every step, and `reveal_fog_of_war` propagates only
through navigable cells, so a stairwell interior is never marked at all. One
episode each on 00800, all three maps correctly framed: bowl 0.54/0.77, banana
0.00/0.18, pitcher **0.11 with zero climb steps**. A guard at 0.80 rejects every
map there is. It is recorded and never gated on.

**`start_on_prior_floor` is INERT on a static layout, and blaming it wasted a
run.** `ycb_env.py:617-620` sets `source_layout = None` when the layout IS the
static one, so `prior_floor_y` is never set and the flag has nothing to require.
Composing it True and re-running produced byte-identical starts (3.16 x4, 0.16,
3.16), the same 213 tracks and the same zero target tracks. The observation that
prompted it is real -- five of six 00800 episodes begin at y=3.16 while their
targets sit at y=0.6-1.2, none descend (`traj_y_range` <= 1.05 m), the pass
scores 0/6 having never had a target in view -- but the cause is that a static
pass samples starts anywhere at least `start_min_geodesic_m` from the target,
and on a multi-storey house that routinely lands on another storey. The flag is
for DYNAMIC layouts, where the static layout genuinely is the source.

**Retention ranks cells first, storeys only as a tie-break.** Ranking on storeys
first repeats, one layer up, the mistake of ranking on `len(_floors)` -- a
storey is allocated the moment a staircase is *detected*. Measured: on 00821 an
episode that ran 215 steps over a 3.4 x 1.2 m box while touching four storeys
beat episodes covering 10 x 28 m, and the stored map came out at 2975 cells
against 30-36k for its neighbours; 00878 showed the same signature at 5 storeys
x ~4500 cells.

## What the gate should be

Not SR, and not storey count. **Does the snapshot contain target tracks.**

A single-storey scene graph is fine: the agent starts on the object's prior
floor, the map insists the object is there, it is not, and the presence filter
and search posterior have to re-find it -- possibly upstairs. What the
experiment cannot survive is a map with no target in it, because then there is
nothing to be stale about.

Pass 1a's SR is not a number worth reading either: `ascentnav` drives on D-FINE,
which is closed-set COCO and cannot see a cracker box or a pitcher. It explores
its full budget regardless, and the map is what it is there to produce.

## Resolved: the scene-graph pass was out of budget

Measured on 00800, static layout, same arm otherwise:

| run | storeys | tracks | YCB target tracks | SR |
|---|---|---|---|---|
| baseline, `max_steps=500` | 1 | 213 | **none** | 0/6 |
| `max_steps=1500` | **2** | 527 | **bowl 3, banana 1, cracker box 1, red plate 1** | 1/6 |
| `obstacle_map_in`, 500 steps | 1 | 217 | none | 0/6 |

At 1500 steps the agent reaches the second storey and `bowl` succeeds on step
1104 -- past the old cap. So the pass was never failing to *detect* anything;
it was failing to *arrive*. Two earlier conclusions drawn from the 500-step
runs were wrong and are recorded here because they were stated confidently:
that `climb_attempts=0` meant "no stairs, therefore no portals, therefore no
climb regardless of budget", and that reusing the obstacle map would be the
lever that moved. Budget was the lever. `n_stair_tracks` is still 0-1 even in
the runs that change floors, so whatever carries the agent upstairs is not the
stair-track path, and that is worth understanding before anything is built on
it.

**Reuse is capped at the starting storey on a fresh agent**, which is why the
third row does nothing. `load_obstacle_map` pastes only into floor layers that
already exist, and an agent at episode start has exactly one, so 17002 of
29950 cells arrived and the upper storey -- the one the pass needed -- stayed
unknown (`unmatched_floors: 1`, visible in `prior_obstacle_map`). The reuse
machinery is working; its reach is bounded. Pass 2 loads a multi-storey
`map_in` first and therefore has the layers, but a mapping pass does not.
Lifting the cap means creating layers from the stored storeys, and an
`ObstacleMap` records no world height to place them at -- which is the same
gap the floor matching already documents.

## Measured: pass 2 runs, and the climb machinery is unreachable

`outputs/mf5_pass2_00800` (v1) and `outputs/mf5_pass2_v2` (v2, with the stair
height bridge) on 00800, `cross_anchor_01`, 500 steps, occupancy from ASCENT
only (`ycb.map_in_occupancy=false`):

| | toy airplane | cracker box |
|---|---|---|
| steps used / budget | 180 / 500 | 362 / 500 |
| distance to goal | 23.2 m | 16.2 m |
| start storey -> goal storey | 0 -> 1 | 1 -> 0 |
| `goal_floor_reached` | False | False |
| `climb_attempts` | 0 | 0 |
| attempts used | 3 of 3 | 3 of 3 |
| states entered | approach, done | approach, close_look, done |

The occupancy reuse itself works: 43139 cells written across BOTH storeys,
`unmatched_floors: 0`, and the presence filter runs (120 and 140 events).

**v2 is trajectory-identical to v1.** Same step counts, same distances, same
flags. The height ramp did write heights -- `stair_max_rise_m` appears where v1
had no stair stats at all -- but it changed no decision, because the code that
would read it is never reached.

Why it is never reached, in order:

* `flights_seen` and `portals_seen` are ABSENT from `agent_stats`. Both are
  written unconditionally (`max(get(...), len(...))`) once `try_switch` gets
  past its gate, so an absent key means the function returned early or was
  never called.
* `_try_floor_switch` is called from exactly one place: inside `_explore`
  (`nav_agent.py:1013`), as the `floor_switch` callback handed to the
  exploration strategy.
* The exploration ROUND never runs. `frontier_select_log` and
  `search_log_events` are both EMPTY across 180 and 362 steps.

The state log looked at first like the agent never enters `explore` at all --
it reads `approach -> done` three times. That reading is WRONG and the
correction is the finding. `rearm` does set `state = State.EXPLORE`, and
`state_log` misses it for a mundane reason: `act` captures `prev_state` at the
top of the step, `rearm` runs between steps, so the APPROACH -> EXPLORE
transition happens outside `act` and is never appended.

The agent does go back to EXPLORE. What it does not get is a ROUND. Two things
stand between the two, and they compound:

* `_act_inner` runs `candidates.check` BEFORE the EXPLORE branch, in EXPLORE
  and in GOTO_FRONTIER both. A commit there returns to APPROACH in the same
  step.
* `ExplorationStrategy.select` is rate-limited to one run per
  `exploration.select_every` steps (5 here).

Attempt 1 fails at step 70; the next commit lands at step 74. Those four
EXPLORE steps are all inside the rate limit, so `select` never fires, and
`floor_switch` -- the only call site of `_try_floor_switch` -- is never
invoked. Commits at 8, 74, 156 (and 3, 68, 123), episodes over at 180 and 362
of 500 steps.

So the whole multi-floor path -- `find_flights`, `pursuit_flight`,
`climb_flight_carrot`, and the stair bridge that feeds them -- is unreachable
in this configuration. **The bridge is still unproven, not disproven.**

What consumes the attempts is the candidate channel, not the map. `p=0.8176`
on every commit and `seen_live: true` on five of six: these are live tracks the
open-vocab detector created on the starting storey under the target's own
label, with the appearance channel declining to admit them
(`feature_admitted: false`, `feature_sim: -1.0`) and `target_bypassed` at 11
and 4. Each attempt is spent walking to one, failing, and committing to the
next one on the same floor. `eval.attempts: 3` then ends the episode.

Both episodes require a storey change to score. The agent does not spend its
budget failing to climb; it never asks to.

### The fix: `agent.explore_after_failed_attempt_steps`

Default 0, which is the shipped behaviour and leaves every measured arm
untouched; `mf5_osg_on_ascent_map` sets 25. After a failed attempt `rearm`
holds `candidates.check` open for that many steps AND calls
`ExplorationStrategy.force_select_next`, so the round runs immediately instead
of waiting on the rate limit. Both halves are needed: holding the commit
without dropping the rate limit buys silence, and dropping the rate limit
without holding the commit buys a round that a commit pre-empts in the same
step.

It is a hold, not a ban. Nothing is blacklisted, and candidates resume at the
end of the window -- `explore_hold_steps` counts what it cost.

The alternative considered and NOT taken was tightening
`scene_graph.target_bypasses_gates`, which is what lets a target-labelled
detection become a track without clearing the admission gates
(`target_bypassed` 11 and 4 here, `feature_admitted: false` on every commit).
That would change what the map contains, which is a bigger change and one that
every other arm is measured with.

## What to open

The pipeline runs across two packages and three passes, and nothing about it is
discoverable from one file. In dependency order:

### 1a -- ASCENT builds the occupancy

| file | what to look for |
|---|---|
| `src/navigation/mapping/obstacle_map.py` | ASCENT's own map, vendored. `_map`, `explored_area`, the four stair masks. Line 374 erases an agent-radius band each step, which is why `explored_area` is not a coverage measure |
| `src/navigation/mapping/map_store.py:138` | `save_obstacle_maps` -- the stack plus the episode anchor `{xy, heading}`, JSON with a sibling `.npz` |
| `src/osg/eval/prior_map.py:28` | `save_obstacle_map_for_scene` -- the RETENTION rule: rank on explored cells, storeys only as a tie-break |
| `configs/experiment/mf5_ascentnav_map.yaml` | the arm that produces it |

### 2 -- pasting it into an OSG costmap

This is the part with the real risk in it, and the part to read first if
anything looks geometrically wrong.

| file | what to look for |
|---|---|
| `src/navigation/mapping/map_store.py:293` | `apply_to_costmap` -- a RESAMPLE, not a copy. Three transforms separate the lattices; a naive copy puts the walls 10 m out at heading 0 and 22 m at pi/2 |
| `src/navigation/mapping/map_store.py:438` | `_ramp_stair_heights` -- the ASCENT->OSG multi-floor bridge. An `ObstacleMap` stores no heights, and `find_flights` reads `costmap.height`, so without this the climb machinery sees nothing. STILL UNPROVEN -- see above |
| `src/osg/eval/prior_map.py:76` | `load_obstacle_map` -- floors matched bottom-up by height; capped at layers that already exist |
| `tests/unit/test_navigation_map_store.py` | the inverse checked against the forward projection at four start headings |
| `scripts/check_obstacle_map_reuse.py --png` | how a human confirms it on a real pass |

### 1b -- the scene graph, and building both in one pass

| file | what to look for |
|---|---|
| `src/osg/graph/map_store.py:182` | `save_map` -- the schema-v2 snapshot: tracks, presence beliefs, storeys |
| `src/osg/graph/map_store.py:360` | `restore_grid` -- returns immediately when `restore_occupancy=False`, which is what lets ASCENT's map be the SOLE occupancy |
| `src/osg/agent/world_model.py` | OSG's world model detached from OSG's FSM, so `ascentnav` can drive while the scene graph is built alongside. `observe` is the mapping half of `NavAgent._act_inner` |
| `configs/experiment/mf5_mapping_both.yaml` | one pass, both artifacts. BUILT AND UNIT-TESTED, NEVER YET RUN ON A SCENE |

### 3 -- pass 2, and where it currently stops

| file | what to look for |
|---|---|
| `configs/experiment/mf5_osg_on_ascent_map.yaml` | the arm; `ycb.map_in_occupancy=false` is what makes the experiment mean anything |
| `src/osg/eval/episode.py:44` | `run_episode` -- `load_prior_map` then `load_obstacle_map`, in that order and for a reason: the snapshot creates the storey layers the paste needs |
| `src/osg/agent/nav_agent.py:516` | `rearm` -- and `explore_after_failed_attempt_steps`, the hold |
| `src/osg/agent/nav_agent.py:1144` | `_try_floor_switch` -- reachable ONLY from `_explore` (line 1032) |
| `src/osg/agent/floor_policy.py:382` | `try_switch` -- the gate that decides whether to leave a storey at all |
| `src/osg/mapping/stairs.py:172` | `find_flights` -- returns `[]` when `costmap.height is None`, which is what `_ramp_stair_heights` exists to prevent |
| `src/osg/agent/nav_agent.py:1878` | `_flight_carrot` -- the waypoint-at-a-time climb, as opposed to the navmesh pointing at the top |

### What to read in a finished run

`episodes.jsonl`, per episode:

* `prior_obstacle_map` -- `cells_written`, `matched_floors`, `unmatched_floors`.
  A non-zero `unmatched_floors` means storeys were stored that this run had no
  layer for. `traj_on_map` is a DIAGNOSTIC and must never be gated on.
* `goal_floor_reached`, `climb_attempts`, `floor_switches` -- whether the storey
  change the episode needs ever happened.
* `agent_stats`: `flights_seen` and `portals_seen` ABSENT means `try_switch`
  returned before looking; `explore_hold_steps` is what the hold cost.
* `frontier_select_log` and `search_log_events` EMPTY means no exploration
  round ran at all -- the signature of the failure above.
* `goal_commit_log` -- `prior: true` is a snapshot object, `seen_live: true` a
  track made this episode. All-live commits mean the run is chasing its own
  detector, not the map.

Videos land in `viz/debug/<scene>_ep<ID>.mp4` and are re-encoded to H.264 on
close (`eval.debug_video_crf`, default 30; 0 opts out).

### What the map picture shows (scripts/legacy/plot_cross_anchor_map.py)

```bash
python scripts/legacy/plot_cross_anchor_map.py --maps outputs/maps_mf5_ascent \
    --scene 00800-TEEsavR23oF --run outputs/mf5_pass2_v4 --out /tmp/map.png
```

Rendered with the targets on it, three things settle at once.

**The paste and the flight selection are right.** The stair mask is one
physical staircase, in the same world place on both storeys, and the flights
chosen sit on its correct ends -- the bottom (-5.52, -10.92) for an ascent from
the lower storey, the top (-8.38, -12.52) for a descent from the upper. The
ramp-orientation worry that cost an hour is answered by the picture: the ends
are right.

**CORRECTION.** The first version of this figure drew `target_obj_xy` as the
target. That field is the agent's LAST COMMITTED CANDIDATE
(`record.target_track_fields`), not the truth; the truth is
`authored_layout.target_position`. Every conclusion drawn from that star was
wrong and is retracted here: the toy airplane is NOT in the unmapped north
room -- it is at (3.65, 0.96, -7.23), in mapped free space on the lower storey,
8.9 m from where the agent was chasing it; and ep1 did not end "0.55 m above
its target", it ended 0.55 m from its own false positive. The figure now draws
both, with different marks, so they cannot be confused again.

**One coverage gap is real, on the other episode.** The cracker box's truth is
(-6.42, 3.80, 0.04) on the upper storey, and the upper storey's map stops at
z = -1.5: the target sits in grey. A successful ascent would arrive with no
prior occupancy there.

**The two episodes fail for different reasons.**

| | ep1 toy airplane | ep2 cracker box |
|---|---|---|
| storey | upper -> lower | lower -> upper |
| flight chosen | step 172, correct | step 68, correct |
| steps on the flight | 0 | 200 |
| height gained | -- | 0.00 m |
| ended | step 198 of 500 | step 500 |
| why | attempts exhausted | climbed the wrong way |

ep1 ends 0.55 m from its own last false positive, one storey above and 8.9 m
west of the true target, which is why `distance_to_goal` is 23.6 m.

### The climb direction came from the wrong place

`_start_climb` derived the direction from `target_y > here_y`. Instrumented on
00800 (`outputs/mf5_pass2_v5`):

```
climb_flight_kind   = up
climb_here_y_x100   = 16     -> 0.163
climb_target_y_x100 = 16     -> 0.163
climb_start_down    = 1
```

The two heights are EQUAL, and `>` is False at equality, so the tie fell
through to -1. The estimator had a level a few centimetres above the current
storey and `min(above)` picked it as "the floor above". `_flight_carrot` then
hunted for treads 0.35-1.0 m BELOW an agent standing on the ground floor,
found none on all 200 steps, and the agent milled at the foot of the stairs.

`agent.climb_direction_from_flight` takes the direction from the pursued
flight's own `kind` instead. Default off; on in `mf5_osg_on_ascent_map`.

Worth recording as method: reading the code could NOT find this. Three
premises all looked sound (`floor_transitions` 0, `floor_y_drift` 0.0, the
undirected branch returning `min(above)` strictly greater than `floor_y`), and
the false one only showed up once the three numbers were printed at the moment
of the decision.

### A commit can pre-empt a floor switch in flight

ep1, every run: the agent selects the down-flight at step 172 and commits to a
new same-floor track at step 181, nine steps later, abandoning it. All three
commits are live upper-storey detections (`prior: False`, `seen_live: True`),
and the third sits at (-4.87, -4.70) -- the target's own (x, z), but on the
wrong storey. `candidates.check` runs in GOTO_FRONTIER as well as EXPLORE, so
nothing protects a switch that has been decided but not yet walked.

### Measured in isolation: can the agent climb at all? (scripts/probe_climb.py)

Every failure above ended `climb_max_dy_x100 = 0`, found inside a 500-step
episode that also has to explore, detect, commit and re-plan. That is a poor
instrument. `probe_climb.py` removes everything else: it asks the PATHFINDER
for a route between the two largest navigable storeys -- whatever it returns
crosses the stairs, because there is no other way between storeys -- teleports
the agent to where that route starts to rise, and drives.

00800, ground-truth flight +3.00 m over 7 waypoints, foot (-11.10, 0.16,
-2.97) -> top (-7.05, 3.16, -4.12):

| mover | result | dy_max | steps |
|---|---|---|---|
| navmesh (PRIVILEGED) | **climbed** | +3.00 m | 37 |
| pointnav -> top | never left the floor | +0.00 m | 300 |
| carrot (path waypoints) | got 44% up, slid back | +1.32 m | 300 |

The height traces are the finding:

```
navmesh   0.16 0.21 0.25 0.31 0.45 0.56 0.79 1.15 1.55 1.98 2.40 2.85 3.08 3.16
pointnav  0.16 0.16 0.16 ... 0.16                       (flat for 300 steps)
carrot    0.16 0.31 0.44 0.56 0.69 0.56 0.54 0.16 0.29 0.44 0.82 1.48 1.36
          1.17 1.05 0.56 0.19 0.16 ... 0.16             (flat for 190 more)
```

So: **the simulator and the agent's step height are fine** -- the privileged
follower walks up in 37 steps. **The point-goal policy aimed at the top never
climbs.** **The waypoint carrot does climb, twice, and slides back both times**,
peaking at 1.48 m of a 3.00 m rise before returning to the floor and stalling
for the remaining 190 steps.

Each mover gets its OWN episode: three 300-step drives in one exhausts
Habitat's step limit and the third dies with "Episode over".

### Why ASCENT climbs and OSG does not

`ascentnav` reaches BOTH storeys of 00800 inside 500 steps -- that is where the
17002 + 26137 cell map came from -- using the SAME `pointnav` mover, with
`agent.climb_enabled: false`, so none of OSG's climb code runs. Its climb is
`navigation/agent.py:_climb_stair`, and the difference is the carrot:

| | OSG `_flight_carrot` | ASCENT `carrot_waypoint` |
|---|---|---|
| source | `costmap.height` of stored flight cells | the RAW depth image |
| rule | nearest cell 0.35-1.0 m above the agent | steer at the farthest thing in view, 0.8 m ahead |
| memory | none, re-picked every step | `ratchet_carrot` keeps whichever carrot is closer to the recorded stair end |

OSG's carrot reads heights that, for a pasted ASCENT map, are a SYNTHETIC ramp
(`_ramp_stair_heights`) over a 3.2 x 2.0 m blob. ASCENT's reads what the camera
can see this step and ratchets, so it cannot re-target backwards -- which is
exactly the failure mode in the trace above.

The pieces are already in this repo and already tested: `carrot_waypoint` and
`ratchet_carrot` in `src/navigation/stairs.py`.

### Who chooses the point, and why it kept choosing the same room

Three choosers, in strict priority every step: `candidates.check` (runs BEFORE
the EXPLORE branch, in EXPLORE and GOTO_FRONTIER); the search posterior
`_select_surface`; and `FloorPolicy.try_switch`, reached only on a storey
request or when the surfaces run out. On 00800 the first decided ~180 of
ep1's 198 steps.

The root is one mechanism with four symptoms: **the map fills with
target-labelled false positives on the wrong storey, and everything downstream
treats them as real.** ep1 ended with eight "toy airplane" tracks on the upper
storey, 6-16 m from the truth; the two it kept committing to are a ceiling
fixture at 2.6 m, detector scores 0.82 and 0.66, eight and five observations;
`target_bypassed` 14 (admitted only for carrying the label).

* The presence filter cannot drop them: 127 presence events, ZERO for the
  target label. A false positive IS present, so each look that disproves it
  as the target re-detects it as an object (`object_layer.py:507` says so).
  The identity channel needs two strikes per track; each fake is a different
  object and gets one.
* They eat the attempts: three commits, ~60 steps each, episode over at 198.
* They veto the switch: `floor_target_evidence` adds _TARGET_PRESENT for a
  believed same-label track and `may_switch` then refuses to leave.
* They pre-empt the switch that gets through: chosen at 172, abandoned at 181.

And nothing drives the storey the other way: `floor_mass_rule: mean` abstains
by design; `floor_llm` is consulted only DOWNSTREAM of a posterior request, so
it was never asked (`floor_llm_asks` absent); the anchor hold waits; and
`search_same_room_bonus: 4.0` keeps the surfaces in the current room -- stool,
bed, dresser, eight steps apart. That is the spinning.

Two of the four proposals are built (default off, on in the cross-anchor arm):

* **`agent.floor_disproved_after_failed_attempts`** -- failed attempts become
  evidence about the STOREY. At N on one storey, its same-label tracks stop
  carrying the veto (`_presence_for_floor_evidence`), and the nearest other
  storey is requested outright (`ExplorationStrategy.forced_floor`, which
  outranks the mass rule and the anchor hold and is cleared on arrival). Every
  failed attempt on the storey counts, the stale anchor's included: an attempt
  at the prior's own pose that finds nothing is the strongest "it moved"
  reading there is.
* **`agent.protect_floor_switch`** -- while `floors.pursuing`, `candidates.check`
  takes an `admit` predicate: only a track seen LIVE within
  `protect_floor_switch_range_m` with `best_score >= protect_floor_switch_min_score`
  may pre-empt. A refusal strikes nothing off.

**The verdict has to be honoured by everything that can overrule a switch.**
The first run with both on (`outputs/mf5_pass2_v7` ep1) disproved storey 0 at
step 153 and requested storey 1 -- and was overruled twice before the third
attempt ended it:

* step 158, the floor LLM, asked on the request, said STAY: "Floor 2
  explicitly lists 'toy airplane' among its contained objects" -- the very
  false positives the attempts had just disproved. The throttle let the next
  request through at 166.
* step 185, `pursuit_preempt_allowed`: the ceiling fixture at score 0.82,
  walked under at <1.5 m on the way to the stairs, passed the live/close/
  confident test. Four weaker ones had been blocked. Last attempt gone.

So on a disproved storey the model is not consulted
(`floor_llm_skipped_disproved`) and NO same-storey candidate may pre-empt
(`pursuit_preempt_blocked_disproved`): candidates are filtered to the current
storey, and two walks that found nothing outrank one more confident look.

Not built: closing the score half of `target_bypasses_gates`, and asking the
floor LLM when the posterior abstains rather than after it requests.

### Measured: OSG's own climb loop, three carrots (scripts/probe_climb_osg.py)

`probe_climb_osg.py` runs the production climb -- `NavAgent._start_climb` +
`_do_climb`, the arm's config, the carrot cascade, the stall test, the budget
-- from the foot of the same navmesh flight, and swaps only the flight the
carrot reads:

| carrot | what `_flight_carrot` sees | result |
|---|---|---|
| depth | no cells -> falls through to `_update_carrot`, ASCENT's depth ray | +0.15 m, stalled at 51 |
| **exact** | the true treads with their TRUE heights | **+1.87 m in 40 steps, `storey_of_height`** |
| ramp | the same treads, heights from `_ramp_stair_heights` | 0.00 m in 200, 161 turns : 39 forwards |

So the loop and the flight carrot are fine; the pasted HEIGHTS were the
defect, and the depth carrot -- which OSG already had, transcribed and
shadowed -- is not the answer here either (its own docstring at
`_stair_cell_carrot` records why: it works from ON the flight, not beside it).

**The ramp's fault is its margin, and the margin must be ZERO.** This took
three rounds to pin, and the wrong turns are kept because each was stated
confidently. `lo + (hi - lo) * (0.12 + 0.76 * frac)` lifted the foot end by a
fraction of the span, 0.36 m on a 3.0 m storey. `_flight_carrot` aims at the
nearest cell 0.35-1.0 m above where the agent STANDS, so any lift at the foot
pulls that cell toward the agent's own feet. The per-step trace of the carrot
(`probe_climb_osg.py` prints it) is what settled it -- same cells each time,
only the heights swapped:

| heights | carrot goal distance | result |
|---|---|---|
| true | 0.66-1.01 m ahead | +1.87 m in 40 steps |
| ramp, margin 0.36 m (fraction) | -- | 0.00 m in 200 |
| ramp, margin 0.25 m (first "fix") | **0.02-0.18 m** -- underfoot | 0.47 m in 200 |
| ramp over walked distance, margin 0 | 0.2-0.8 m | +1.87 m in 48 |
| **ramp on the chord, margin 0** (the paste's own axis) | 0.75-1.24 m | **+1.83 m in 51** |

The chord runs BACKWARDS on 7 of this flight's 49 steps (5.82 m walked over a
4.21 m chord -- it bends), and the carrot climbs anyway: it only needs the
nearest in-band cell to be genuinely ahead, and a monotone ramp on the right
cells with no lift at the foot gives it that. `find_flights`' own 0.2 m floor
margin then trims the bottom two treads off the Flight, which is harmless.

`RAMP_MARGIN_M = 0.0` in `navigation/mapping/map_store.py`.

## Open

**Superseded by the table above; kept for the reasoning that was eliminated.**
The OSG scene-graph pass reaches no objects at 500 steps.
On 00800 it ends with 213 furniture tracks and zero target tracks, in three
independent runs: as configured, with `start_on_prior_floor` flipped, and with
`agent.rednet_stairs=true`. All three are byte-identical, which is itself the
finding -- none of those levers touches the cause. `climb_attempts` is 0 in
every episode and `n_stair_tracks` is 0 or 1, so the agent never approaches a
staircase, and a start sampled on the wrong storey is therefore terminal.

Three levers remain, and they are not equivalent:

* **`agent.max_steps` above 500.** The agent ends 9-23 m from its target in a
  large two-storey house; it may simply run out of budget. Cheapest to test,
  changes the arm the least.
* **Let the mapping pass read `ycb.obstacle_map_in`.** ASCENT already mapped
  2-4 storeys of every scene, so the OSG pass would start with the building
  known and spend its whole budget on objects rather than geometry. This uses
  the machinery this document describes and is the most interesting option, but
  it makes the scene-graph pass depend on the obstacle-map pass.
* **Make the OSG agent climb.** The honest fix, and the largest.

Until one is chosen the cross-anchor experiment cannot run, because a map with
no target track in it has nothing to be stale about.

Each is one command, and the gate is the same for all three -- does the
snapshot come back with YCB target tracks in it:

```bash
# lever 1: more budget
python scripts/run_eval.py +experiment=mf5_osg_unified \
    'ycb.scenes=[00800-TEEsavR23oF]' 'ycb.layout_types=[static]' \
    'ycb.layout_indices=[1]' ycb.cross_floor_relocations_only=false \
    agent.max_steps=1500 ycb.map_out=outputs/maps_try_steps \
    eval.save_viz=false output_dir=outputs/try_steps

# lever 2: map the objects over ASCENT's geometry
python scripts/run_eval.py +experiment=mf5_osg_unified \
    'ycb.scenes=[00800-TEEsavR23oF]' 'ycb.layout_types=[static]' \
    'ycb.layout_indices=[1]' ycb.cross_floor_relocations_only=false \
    ycb.obstacle_map_in=outputs/maps_mf5_ascent \
    ycb.map_out=outputs/maps_try_obstacle \
    eval.save_viz=false output_dir=outputs/try_obstacle

# the gate, for either
python - <<'EOF'
import json
from collections import Counter
from pathlib import Path
YCB = {"bowl","tin can","cracker box","banana","red plate","coffee can",
       "blue plastic pitcher","toy airplane","scissors","mug","yellow bottle"}
for d in ("outputs/maps_try_steps", "outputs/maps_try_obstacle"):
    for p in Path(d).glob("*.json"):
        b = json.loads(p.read_text())
        labs = Counter(t["label"] for t in b["tracks"])
        print(d, {k: v for k, v in labs.items() if k in YCB} or "NO TARGET TRACKS")
EOF
```

Lever 2 is the one worth running first: it costs nothing extra (the maps are
already built), and if the OSG pass reaches the objects once the geometry is
handed to it, that is also evidence the obstacle-map reuse does what it was
built to do.
