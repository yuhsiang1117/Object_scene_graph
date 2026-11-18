# The unreachable set, on the sensor-only mover

**Question.** `docs/history/REACHABILITY_CEILING.md` claimed, from geometry alone, that
15 of the 107 dynamic trials have no navmesh point within the benchmark's 1 m
rule for our agent. That was a measurement of the floor plan, not of the robot.
After the switch to `navigation: pointnav`, does a mover that plans from depth
rather than from the mesh reach any of them?

**Answer: no. 0/15 on both movers.** And the reason is not the mover.

## The agent is physically confined to that mesh

Every one of the 107 final poses in the navmesh run
(`outputs/osg_released_maps/flat_anchor_v2_island_close`) snaps to the navmesh
habitat-lab recomputes for our agent at **0.0000 m**, and Habitat reports
107/107 of them navigable. The constraint is the simulator's own collision
check in `move_forward`, so it binds every mover equally. Changing the
navigator cannot move this ceiling; only changing the body can.

## The run

15 trials, `+experiment=dualmap_protocol_osg_look_flat_anchor_v2_island_close`
on `outputs/maps_released`, one process per scene.

| | navmesh arm | pointnav arm |
|---|---:|---:|
| scored | 0/15 | 0/15 |
| best scored stop, median | 1.56 m | 1.87 m |
| closest the agent ever came, median | 1.24 m | 1.30 m |
| final distance, median | 8.04 m | **2.42 m** |
| final distance within 2 m | 2/15 | **6/15** |
| attempts spent | 33 | 22 |
| target named at least once | 12/15 | 11/15 |
| ended at the 500-step budget | 15/15 | 15/15 |

Per-trial detail: `outputs/osg_unreachable_pointnav/RESULT.txt`.

Both movers get to within about 1.2-1.3 m of the object and stop there,
because that is where the floor ends. The best either can do is bounded below
by the per-trial optimum, and the median optimum over this set is 1.16 m.

## The one real difference

The pointnav arm ends the episode **much closer** -- median 8.04 m -> 2.42 m,
and six trials inside 2 m against two. That is not better navigation; it is
the absence of the navmesh arm's behaviour of failing the near attempt and
then committing to something far away. Pointnav spends fewer attempts (22
against 33) and stays in the neighbourhood. Under a metric that scored
proximity rather than a 1 m threshold this would read as a large gain, and it
is worth remembering when quoting SPL or mean distance.

Neither arm produced a single clean approach termination here: `stop_reason`
is None on all 15, all 15 run to the budget.

## What would move it

Only the body. Recomputing the navmesh at smaller radii, over this same set:

| agent radius | of the 15, now within 1 m |
|---|---:|
| 0.18 (ours) | 0 |
| 0.15 | 4 |
| 0.12 | 4 |
| 0.10 (Habitat's default) | 6 |

Nine are out of reach at any body size; `in_anchor__0117__plate` is 1.75 m from
the nearest floor even at 0.10. 0.15 and 0.12 tie because the navmesh
voxelises the radius.

So the honest ceiling on this benchmark for a 0.18 m agent is 92/107, and
98/107 for a 0.10 m one. DualMap is not held to this at all -- its released
runner places the agent with `set_agent_state` along its own occupancy grid
(`scripts/run_dualmap_released_native.py`, `ReleasedFollower.advance`), which
is how it scores 2 of these 15.

## Reproduce

```bash
python scripts/legacy/analyze_reachability_ceiling.py \
  --run outputs/osg_released_maps/flat_anchor_v2_island_close \
  --dualmap outputs/dualmap_official_bench/seed12 \
  --out outputs/osg_released_maps/REACHABILITY.md \
  --json outputs/osg_released_maps/reachability.json
# the split is written from that json: ours_m > 1.0
# data/splits/dualmap_unreachable.json
```
