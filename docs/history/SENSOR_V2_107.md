# DualMap, our navmesh record, and our sensor-only record

107 trials shared by all three. Same starts, same 1 m / 3-attempt rule.

- DualMap  `outputs/dualmap_official_bench/seed12` (+2 more seeds for the spread)
- navmesh  `outputs/osg_released_maps/flat_anchor_v2_island_close`
- sensor   `outputs/osg_sensor_v2_full`

## Success rate

| condition | DualMap | ours, navmesh | ours, sensor-only r=0.10 |
|---|---:|---:|---:|
| in_anchor | 35/54 (seeds 35/32/34) | 29/54 | 31/54 |
| cross_anchor | 16/53 (seeds 16/16/16) | 27/53 | 26/53 |
| **total** | 51/107 | 56/107 | 57/107 |

## Paired, sensor-only against each

| condition | vs navmesh | sign test | vs DualMap seed 12 | sign test |
|---|---:|---:|---:|---:|
| in_anchor | +8/-6 | 0.791 | +6/-10 | 0.454 |
| cross_anchor | +6/-7 | 1.000 | +17/-7 | 0.064 |
| all | +14/-13 | 1.000 | +23/-17 | 0.430 |

## Against the reachability ceiling

The agent cannot stand within 1 m of some objects at all. That bound moves
with the body, and DualMap is not subject to it.

| set | n | DualMap | ours, navmesh (r=0.18) | ours, sensor (r=0.10) |
|---|---:|---:|---:|---:|
| reachable at r=0.18 | 92 | 49 | 56 | 57 |
| NOT reachable at r=0.18 | 15 | 2 | 0 | 0 |

## Did the agent actually find the object?

A success within 1 m that never named the target is credit for standing in
the right place. Localised = the target was named on some keyframe.

| | DualMap | ours, navmesh | ours, sensor |
|---|---:|---:|---:|
| localised successes | (not comparable) | 54/56 | 51/57 |

## Distance and cost

| | DualMap | ours, navmesh | ours, sensor |
|---|---:|---:|---:|
| median final distance | 1.64 m | 0.99 m | 0.97 m |
| final within 2 m | 55 | 60 | 65 |
| median SPL | 0.000 | 0.047 | 0.046 |
| median steps | - | 322 | 408 |

## By query

| query | condition | DualMap | navmesh | sensor |
|---|---|---:|---:|---:|
| banana | in_anchor | 2/3 | 2/3 | 0/3  **-** |
| banana | cross_anchor | 0/3 | 3/3 | 1/3  **-** |
| bowl | in_anchor | 6/6 | 6/6 | 6/6 |
| bowl | cross_anchor | 3/6 | 5/6 | 3/6  **-** |
| cracker box | in_anchor | 5/9 | 3/9 | 3/9 |
| cracker box | cross_anchor | 1/8 | 4/8 | 5/8  **+** |
| mug | in_anchor | 3/3 | 0/3 | 2/3  **+** |
| mug | cross_anchor | 2/3 | 2/3 | 2/3 |
| pitcher | in_anchor | 2/9 | 4/9 | 5/9  **+** |
| pitcher | cross_anchor | 2/9 | 4/9 | 5/9  **+** |
| plate | in_anchor | 5/9 | 5/9 | 5/9 |
| plate | cross_anchor | 5/9 | 6/9 | 7/9  **+** |
| scissors | in_anchor | 7/9 | 5/9 | 5/9 |
| scissors | cross_anchor | 2/9 | 0/9 | 0/9 |
| soup can | in_anchor | 5/6 | 4/6 | 5/6  **+** |
| soup can | cross_anchor | 1/6 | 3/6 | 3/6 |
