# Fixing the sensor-only approach: results on the 44 stranded trials

The subset is every trial whose approach ended with `min_dist_to_goal_m` in
(0.30, 1.00] on `sensor_r10` -- stranded in the PointNav creep band
(`data/splits/dualmap_creep_stranded.json`). The navmesh arm's numbers on the
same 44 are the reference, not a target: it is a different mover with the
ground-truth mesh.

| | base | +stall | +stall+fit | +stall+close | navmesh |
|---|---:|---:|---:|---:|---:|
| scored /44 | 11 | 14 | **18** | **22** | 26 |
| paired vs base | - | +5/-2 | +11/-4 | +13/-2 | - |
| median steps | 500 | 406 | 451 | 355 | 248 |
| ended at the budget | 33 | 20 | 21 | 19 | 15 |
| median distance from its own goal | 0.56 | 0.56 | 0.42 | 0.38 | 0.08 |
| median best stop | 1.32 | 1.21 | 1.34 | 1.05 | 0.88 |
| `path_consumed` terminations | 0 | 24 | 22 | 25 | 29 |

## What each arm changed

**`+stall`** — `pointnav_creep_stall_steps: 8`. The creep band returns a blind
`move_forward`; if the goal cannot be stood on, habitat refuses it and the press
runs to the budget. Reporting arrival once the pursuit stops closing restores
the approach's ability to terminate at all: `path_consumed` 0 -> 24, budget
enders 33 -> 20. It does NOT get the agent closer -- distance-from-goal is
unchanged at 0.56 m, which is the point. It stops the waste.
Counters: `pointnav_creep_stalled` 201, `pointnav_arrived` 84,
`pointnav_policy_stop` 30.

**`+stall+fit`** — `verification.viewpoint_min_clearance_m: 0.25`. The viewpoint
planner computed a clearance transform and used it only to RANK candidates; a
cell one pixel from an obstacle counted as a valid pose. Making clearance a
constraint moves distance-from-goal 0.56 -> 0.42 and scores 18/44. Sensor-only:
the clearance is measured on the agent's own depth-built costmap.
Side effect worth watching: the strict viewpoint search now fails far more often
(`approach_viewpoint_none` 4 -> 45) and the relaxed fallback carries it. The
fallback enforces the same clearance, so the goals are still standable, but the
ladder is being relaxed on most approaches and that deserves tuning.

**`+stall+close`** — the closing walk restored (`approach_close_last_metre_m:
0.8`). It fired 43 times and retreated 3 (`approach_close_worse`). Best single
lever at 22/44, and the only one that also cuts steps hard (355 median).
It is navmesh-derived: its goal is `pathfinder.nearest_navigable_xy(object)`,
guaranteed standable by construction -- exactly the property the stranded goals
lacked. That is the whole of its advantage, and it is reproducible without the
mesh (see below).

## The remaining gap, and where it is

Against the navmesh arm on the full 81 completed trials, the stall test moved
the deficit from -18 to -15. Of the 21 trials it still loses:

    best stop, median                 1.27 m
    stopped within 1.5 m              13/21
    closest approach already <= 1.0 m 10/16 (of those with ground truth)
    ... and ALL TEN of those stopped outside 1.0 m

So the agent is reaching the right place and then stopping too far out. Over
all 81 trials it gives up a median 0.18 m between its closest approach and where
it stops, against the navmesh arm's 0.12 m; on the losses it gives up 0.04-1.72 m.

## Next, in order

1. **Stop at the best pose, not the last one.** Track the agent pose closest to
   the committed track during an approach; on concluding, if the current pose is
   materially worse, walk back before stopping. The closing walk already
   implements exactly this for its own walk (`approach_close_worse`); this
   generalises it to the whole approach. Sensor-only -- the distance is to the
   agent's own estimate of the track centre, not to ground truth. Sized above at
   ten of sixteen losses.
2. **A sensor-only closing walk.** The closing walk's entire advantage is a goal
   that is standable by construction. The costmap can supply that: the nearest
   cell to the object whose clearance exceeds the agent radius. That is the same
   query the mesh answers, asked of the agent's own map. If it recovers most of
   the 22/44, the privilege was never load-bearing.
3. **Tune the clearance ladder.** `approach_viewpoint_none` 4 -> 45 says 0.25 m
   is rejecting most strict viewpoints. Sweep the threshold and consider letting
   the ring ladder search outward before relaxing line-of-sight.
4. **`fit` + `close` together**, to price the privilege once both sensor-side
   fixes are in. The gap between that and (2) is what the navmesh is actually
   worth here.
