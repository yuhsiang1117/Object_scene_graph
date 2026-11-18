# The sensor-only arm loses in the last half metre, and not to the RL policy

Paired on the 81 trials the stopped `sensor_r10` run completed, against the
same trials in `outputs/osg_released_maps/flat_anchor_v2_island_close`.

## The decomposition

| stage | navmesh | sensor-only |
|---|---:|---:|
| committed to a track within 1 m of the true object | 54/81 | 49/81 |
| of those, scored | 40/54 = **74%** | 21/49 = **43%** |
| reached its own approach goal (within the 0.30 m arrival radius) | 57/72 | **17/71** |
| median distance from its own goal at closest | **0.08 m** | **0.49 m** |
| where that goal was placed, from the object | 0.46 m | 0.46 m |

Target selection is intact. The agent picks the same objects and places the
same goals -- goal-to-object is 0.46 m in both arms, because it is the same
viewpoint planner on the same costmap. What changed is that the mover no
longer gets to the goal it was given: 0.08 m becomes 0.49 m, and the
conversion of a correct commit falls from 74% to 43%.

Add it up and the failure is arithmetic: 0.46 m of goal offset plus 0.49 m of
mover error is 0.95 m, sitting exactly on the 1 m rule.

## Where they die: the creep band

`PointNavDriver.step` has three regimes, and the middle one has no policy in it:

```
rho < pointnav_arrival_m (0.30)      -> None, "arrived"
rho < pointnav_approach_creep_m (1.0) -> "move_forward", unconditional
otherwise                             -> ask the network
```

Between 0.30 m and 1.00 m the driver stops consulting the network and returns a
blind `move_forward` -- no obstacle test, no stall test, no give-up. Our
approach goals are viewpoints about half a metre from an object resting on a
bed, a desk or a cabinet, so the last stretch is very often blocked. Habitat
refuses the forward, the agent does not move, `rho` never falls below 0.30, and
the driver emits `move_forward` again. For ever.

| where the approach ended, relative to its own goal | navmesh | sensor |
|---|---:|---:|
| reached it (<= 0.30 m) | 57 | 17 |
| **stranded in the creep band (0.30-1.00 m)** | **10** | **44** |
| never got near it (> 1.00 m) | 5 | 10 |

Those 44 are the arm's whole deficit. They run a median of **262 approach
steps** against 86 for the navmesh arm, and **33 of the 44 end the episode at
the 500-step budget**.

The terminations agree:

| | navmesh | sensor |
|---|---:|---:|
| `path_consumed` (a clean arrival) | 48 | 14 |
| no conclusion at all | 32 | 54 |
| episodes ending at the budget | 32 | 54 |
| median steps | 330 | 500 |

## So: is the RL policy broken?

Not where it is being blamed. **Inside 1 m the policy is never consulted** --
the creep bypasses it entirely, and that is the band where the arm loses. The
frozen PointNav network is answering only beyond 1 m, and it gets the agent
there: `plan_fail` is 0 across all 81 trials and the agent reaches the
1.0 m band on essentially every correct commit.

The defect is the creep's contract. It was ported from ASCENT
(`ascent_policy.py:920-927`), where the terminal decision is made by an object
point cloud, so a blind press that never converges costs nothing. Our approach
has no such backstop with the closing walk switched off, and neither
`escape_window` nor `approach_abandon_steps` is enabled on this arm -- both are
0, so nothing interrupts the press.

## Candidate fixes, cheapest first

1. **A stall test inside the creep.** If `rho` has not decreased over N steps,
   return `arrived` -- the agent is as close as the geometry allows. Converts a
   262-step press into a stop around 0.5 m from the goal (about 0.95 m from the
   object, often scoring) and returns hundreds of steps to the episode. This is
   the narrowest fix and the one that matches what the navmesh follower did:
   report arrival when it could get no closer.
2. **Narrow the band.** `pointnav_approach_creep_m` 1.0 -> ~0.4 keeps the
   network steering, so it can sidestep rather than press.
3. **Widen the arrival.** `pointnav_arrival_m` 0.3 -> ~0.6 declares arrival
   where the agent actually ends up. Blunt, and it gives up distance that is
   sometimes reachable.
4. **Enable the unstick guard** (`escape_window`, `src/osg/planning/escape.py`),
   which upstream added for exactly this class of problem.

(1) and (2) are separable and both testable on the stranded subset before any
full run. None of them is a change to the RL policy.

## Caveat on the arm

`sensor_r10` changed three things at once against the navmesh record -- the
mover, the closing walk (off), and the body (0.18 -> 0.10). This analysis
isolates the mover, because goal placement is identical across the two arms and
the deficit localises to the creep band. The body change cannot explain it: a
smaller radius only adds free space.
