# Dynamic-scene experimental protocol

This authored relocation test is separate from the fixed-layout ObjectNav
baseline in `METHODOLOGY.md`.

The dynamic-scene test uses a prior map as a controlled input. In a static
layout, a mapping pass writes a schema-v2 scene-graph snapshot containing
object tracks, presence beliefs, storeys and completed stair connectivity.
After the objects are relocated, the search pass loads that snapshot and
must find the target in the changed layout. On reload, presence log-odds are
capped at `reload_max_log_odds = 1.5` so old sightings do not start the new
episode at full confidence.

The prior occupancy grid is independent of the scene-graph snapshot. The
same-system arm can restore the grid built during mapping. In the cross-anchor
arm, an `ascentnav` static pass separately writes its per-storey obstacle
maps; the dynamic `nav_agent` pass loads them through `ycb.obstacle_map_in`
while retaining OSG's scene graph (`ycb.map_in_occupancy=false`). This tests
belief revision and multi-floor recovery over another system's occupancy
prior.

In the cross-anchor pipeline, before pass 2, the coverage audit checks that
the prior includes the authored target locations and reachable stairs.
Scenes failing that audit are held back: otherwise a failure could reflect
an incomplete map rather than a failure of dynamic recovery. See
`CROSS_ANCHOR_STATUS.md` §1 and
`CROSS_ANCHOR_OBSTACLE_MAP.md` for the mapping arms, audit and limitations.
