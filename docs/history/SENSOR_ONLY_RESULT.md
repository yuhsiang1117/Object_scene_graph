# The sensor-only agent matches the privileged one

`+experiment=dualmap_protocol_osg_sensor_v2`, 107 dynamic trials on
`outputs/maps_released`, same starts and the same 1 m / 3-attempt rule as
everything else.

| condition | DualMap (seed 12) | ours, navmesh | ours, sensor-only |
|---|---:|---:|---:|
| in_anchor | 35/54 | 29/54 | 31/54 |
| cross_anchor | 16/53 | 27/53 | 26/53 |
| **total** | **51/107** | **56/107** | **57/107** |

Paired: +14/-13 against the navmesh arm (sign test 1.00 -- level, not ahead)
and +23/-17 against DualMap, whose cross-anchor split is +17/-7 (p = 0.064).
On the 92 trials reachable at radius 0.18: sensor 57, navmesh 56, DualMap 49.

## What "sensor-only" means here

Nothing in the agent queries habitat's navmesh. Not the mover (a frozen
PointNav policy on depth and pose), not the goal (viewpoint rings on the
agent's own costmap), not the closing walk (`nearest_clear_xy` on that same
costmap), and not reachability (the `is_reachable` island oracle is inert
without the mesh). The navmesh arm in the middle column had all four.

## The four fixes, and what each was worth

Diagnosis first: the switch to the PointNav mover cost 18 trials, and the
cause was not the RL policy. Inside 1 m the policy is never consulted -- the
driver's creep band returns a blind `move_forward` -- and 27 of 44 stranded
approaches had a goal habitat's navmesh calls NON-navigable, i.e. inside the
furniture. The agent walked to the nearest standable point and pressed against
the boundary until the step budget.

On the 44 trials stranded in that band:

| | scored /44 |
|---|---:|
| sensor baseline | 11 |
| + creep stall test | 14 |
| + viewpoint clearance | 18 |
| + stop-at-best-pose | 20 |
| + costmap closing walk | 26 |
| **the corrected arm, from the full run** | **27** |
| navmesh arm, same trials | 26 |

1. **`pointnav_creep_stall_steps`** -- report arrival when the creep stops
   closing. Restores the approach's ability to terminate at all
   (`path_consumed` 0 -> 24, budget-enders 33 -> 20). Fired 393 times over the
   107.
2. **`verification.viewpoint_min_clearance_m`** -- the viewpoint planner
   computed a clearance transform and used it only to RANK candidates, so a
   cell one pixel from a wall counted as a valid pose. The navmesh follower had
   hidden this for every previous arm by SNAPPING the goal onto the mesh.
3. **`agent.approach_stop_at_best_m`** -- stop at the closest pose the approach
   reached, not the one its terminal rule fires in. Fired 33 times over the 107
   for 12.5 m recovered, about 38 cm a time.
4. **`agent.approach_close_source: costmap`** -- the closing walk, asked of the
   agent's own depth grid. The largest single lever, and the one that settles
   the privilege question below.

## The privilege was worth less than nothing

The same arm holding a NAVMESH closing walk scores 24/44 against the costmap
walk's 26, paired +3/-1 for the sensor-only version. A goal chosen from
ground-truth geometry the agent cannot see is one its own map may believe is
blocked, and it approaches it badly; a goal its own map calls clear is one it
can both believe and plan to. Consistent information beat better information
the planner could not use.

## The radius

0.10 against the standard 0.18, both with every fix: 46/107 against 41/107,
paired +13/-18 for the smaller body, almost all of it cross-anchor (19 v 15).
0.10 also raises the reachability ceiling from 92/107 to 98/107, since the
agent is physically confined to the navmesh habitat rebuilds for its radius
(docs/history/UNREACHABLE_SET_EXPERIMENT.md). Declared, not tuned.

## Caveats, kept in the open

* **Level, not ahead.** +14/-13 against the navmesh arm is a wash. The claim is
  that dropping the privilege costs nothing, not that it gains.
* **Slightly more do-nothing credit.** Localised successes -- the target named
  on some keyframe -- are 51 of 57 against the navmesh arm's 54 of 56.
* **It costs steps.** Median 408 against 322, and 43 of 107 still end at the
  budget.
* **DualMap still wins in-anchor**, 35 to 31. Our margin is entirely
  cross-anchor, which is the half that tests dynamic-scene handling: in-anchor
  moves objects a median 0.70 m against a 1 m rule, where a do-nothing agent
  scores 74.1% (docs, `osg-dualmap-in-anchor-is-degenerate`).
* **A subset flattered the fixes once already.** The 44 stranded trials reached
  parity before the full 107 did, and an intermediate arm read 26/44 on the
  subset and 19/44 on the full run. Only the 107 counts.

## One wrong turn, recorded

The clearance sweep ran WITHOUT the closing walk, where 0.15 and 0.25 tied at
20/44, and I broke the tie on "ladder health": 0.15 left more approaches on the
strict ring search and fewer on the relaxed fallback. That was backwards. The
relaxed fallback accepts UNKNOWN cells and drops the line-of-sight test, so it
can place a pose NEARER the object than any strict ring pose sits; forcing it
lands closer. Median stop 0.95 m at 0.25 against 1.10 m at 0.15, and 26/44
against 19/44 with the walk in place. `approach_viewpoint_none` is named like a
failure and was functioning as the mechanism.
