# Multi-Floor and Dynamic Scene Integration Plan

## Intent and source history

Create `integration/multifloor-dynamic` from local `nearmiss-approach` at
`cc22704`. Treat upstream `multi-floor` at `8fb27d2` as a behavior snapshot,
not as a file-level merge: the histories are unrelated and overlap the core
modules. Preserve the dynamic-scene behavior from the local source, port the
complete multi-floor and ASCENT feature set into the refactored architecture,
and add an explicit combined mode in which stale objects may relocate between
floors. Finish with an ancestry-only `ours` merge of the upstream snapshot.

## Implementation workstreams

1. Establish the integration branch, preserve all existing untracked user files,
   and maintain a complete source inventory classifying upstream additions as
   ported, superseded, or merged into shared implementations.
2. Unify typed configuration and factories. Add canonical
   `agent.navigation` (`costmap`, `navmesh`, `pointnav`) and `agent.policy`
   (`nav_agent`, `ascent`, `ascentnav`), while validating the legacy
   `use_habitat_navmesh` alias and retaining all presets/tools.
3. Maintain one canonical floor-aware scene model: stable floor keys separate
   from height order; per-floor maps, planners, room labels, stair evidence,
   exploration state and optional value maps; floor hysteresis/transit
   rejection, stair freezing/fusion, portals, and retirement.
4. Make dynamic search floor-aware. Aggregate container posterior mass by floor,
   direct remote-floor mass to `FloorPolicy`, never plan remote coordinates in a
   local 2D map, scope object association/presence/rooms/containers by floor,
   and provide the `ycb_dynamic_multifloor` authored preset.
5. Upgrade persistence to schema v2 with stable floor metadata, one grid/room/
   stair state per floor, connectivity, value-map-compatible state and track
   floor keys. Read schema v1 as floor 0, select the restored floor using the
   first observed height, preserve beliefs/observations/crops, and reject
   corrupt or incompatible snapshots.
6. Extend episode records and summaries with start/goal/prior floor, relocation
   direction, selected search floor, floor switches, climb attempts, goal-floor
   reach, and SR/SPL groups for same-floor, cross-floor, upward and downward
   relocations. Keep the dynamic hierarchy `floor → room → container → object`.
7. Merge dynamic and ASCENT documentation, preserve upstream attribution and
   licenses, merge colliding comparison/weight tools, and pin configuration
   fingerprints and source inventory.

## Verification gates

1. Stage prerequisites using the mounted HM3D v0.2 scenes; download public HM3D
   ObjectNav v1 episodes; clone ASCENT inputs; generate priors; stage and load
   PointNav, RedNet, CLIP/MobileCLIP and YOLOE assets; record SHAs, checksums,
   GPU and model endpoints.
2. Capture source baselines: local unit suite and 96-episode dynamic run;
   isolated upstream unit suite and ASCENT 100-episode/cross-floor runs.
3. Run deterministic unit/contract coverage for floor allocation/order,
   transit/stair behavior, floor-scoped association and presence, room/container
   isolation, posterior aggregation, policy/alias selection, schema-v2/v1
   persistence, all constructors, and every preset fingerprint.
4. Replay synthetic traces with verification/hosted calls disabled and run the
   static authored smoke; require unchanged legacy action sequences, candidate
   order, presence, graph serialization and episode fields when new modes are
   disabled. Smoke costmap, navmesh, PointNav, `ascent` and `ascentnav`.
5. Run paired dynamic regressions with the same episode IDs, stale maps, seeds,
   detector profile and endpoints; enforce the stated success-loss and telemetry
   thresholds against the contemporaneous source run.
6. Run paired multi-floor regressions on `scenes20_ep0to4` and its cross-floor
   subset; enforce overall/same-floor/cross-floor success and stage-funnel
   thresholds, and check that single-floor episodes incur no regression.
7. Run combined multi-floor dynamic validation. Build schema-v2 static maps,
   retain the richest map per scene, select only relocations with different
   floors and starts on the prior floor, cover requested upward/downward scenes,
   prove every manifest has two floors, verify stale-target evidence and floor
   reassignment, and establish the first directional SR/SPL baseline.
8. Commit focused workstreams, verify the final unit/config gates, then record
   upstream ancestry with:

   ```bash
   git merge -s ours --allow-unrelated-histories --no-ff upstream/multi-floor
   ```

Generated datasets, external checkouts, weights, maps and benchmark outputs stay
ignored under `data/` and `outputs/`. Existing untracked user files are not part
of integration commits.
