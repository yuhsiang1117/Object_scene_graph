# Cross-anchor on ASCENT's obstacle map -- status, findings, fixes, roadmap

*Written 2026-09-14 after run v13. `docs/history/CROSS_ANCHOR_OBSTACLE_MAP.md` is the
chronological log with every measurement and every wrong turn; this is the
summary of where things stand and what to do next.*

## 1. The experiment

Pass 1a: `ascentnav` explores a static scene and stores its per-storey
`ObstacleMap` stack (`ycb.obstacle_map_out`). Pass 1b: `nav_agent` explores the
same scene and stores the scene graph (`ycb.map_out`). Pass 2: the objects are
moved to another storey (`cross_anchor` layouts), and `nav_agent` must re-find
them navigating over ASCENT's occupancy (`ycb.obstacle_map_in`,
`ycb.map_in_occupancy=false`) with OSG's presence filter and multi-floor
pipeline. Scene under test throughout: `00800-TEEsavR23oF`, two storeys 3.0 m
apart, one staircase, 500 steps, 3 attempts.

## 2. Where it stands

**Success rate is still 0/2 on 00800.** What changed is *why*. Twelve runs ago
the agent never left the room it started in; now it disproves the wrong
storey from its own failed attempts, requests the right one, walks to the
real staircase, climbs it, arrives, stays, and searches there. It fails at
the last step: finding the object on the correct storey with the attempts it
has left, against a map full of same-label false positives.

| run | what it added | toy airplane (upper -> lower) | cracker box (lower -> upper) |
|---|---|---|---|
| v2 | baseline | 180 steps, no exploration round, 23.2 m | 362 steps, 16.2 m |
| v3 | exploration hold after failed attempts | flights found; climbed furniture | climbed furniture x3, then the real stairs (wrong direction) |
| v4 | stair-map flight ranking | chose the "stairs" -- a location with no treads (see 4A) | same |
| v6 | climb direction from the flight | -- | carrot now fires, 0.00 m |
| v8 | storey disproval + protected switch | 488 steps, 200 on a flight, 0.00 m | -- |
| v10 | **paste orientation fixed** | first vertical progress, -0.23 m | -- |
| v11 | ramp margin, min-ahead carrot, ramp orientation | **arrived**, `goal_floor_reached`, 4.3 m | first `climb_ok`, +1.83 m, then back down |
| v12 | verdict covers candidates | arrived, then posterior sent it back up | same |
| v13 | gap-closing climb, verdict covers posterior | **arrived and stayed**, 13.8 m, last attempt on a fake | climbed to +1.97 m (past the old 1.8 m stop), ended on budget/stall short of the 2.7 m gap, back down; 14.3 m |
| v14 | flight spans the known storey gap (finding 15) | NO climb at all: the wide band merged the storey below into the staircase (finding 16); 25.5 m | **reached the upper storey**, `climb_end_new_floor`, +2.62 m; 5.0 m |
| v17-v21 | see "the climb interventions that did not work" below | |
| v15 | the widening restricted to stair-map cells | descends 2.44 m of 3.0 (was 1.51), ends on the climb budget; 16.5 m | unchanged: reached the storey, 5.0 m, 2 of 3 attempts |

Full per-episode numbers: `python - <<'EOF' ... ` in section 7, or the table in
the log.

## 3. What was found, by layer

### A. The map bridge (`src/navigation/mapping/map_store.py`)

1. **The paste was transposed.** vlfm's `_xy_to_px` returns geometric
   `(row, col)`; its grids are written `_map[px[:,1], px[:,0]]`, i.e.
   `[col, row]`. Points are geometric, arrays are their transpose, and the
   loader read both as geometric. Measured against the navmesh: lower-storey
   navigable points landed FREE 0.04 / OCC 0.04 / UNK 0.92 as stored and
   FREE 0.49 / OCC **0.00** / UNK 0.51 transposed. Every geometric conclusion
   before v10 was drawn on a map 8-9 m from the truth, and the "stairs" the
   agent climbed in v4-v9 do not exist. *Fix:* `_to_geometric_layout` at load,
   undone on restore. The earlier "81.6% of the path on free cells" validation
   looked the map up through vlfm's own indexing and could not see this.
2. **The ramp was lifted off the floor.** `_flight_carrot` aims at the nearest
   cell 0.35-1.0 m above where the agent stands; a foot tread reading 0.36 m
   "above" the agent standing on it is that cell. *Fix:* `RAMP_MARGIN_M = 0.0`.
3. **The ramp was reversed** (correlation with true tread heights -0.91/-0.86).
   ASCENT's stored stair endpoints run top-to-bottom on this map, and
   free-floor adjacency cannot tell the ends apart because the lower-storey
   map bleeds up onto the upper landing. *Fix:* orient by the order ASCENT's
   own trajectory walked the treads (`_stair_ends_from_trajectory`; 143 poses
   on the flight, entry t=0.33, exit t=2.12), mirrored for the other storey.
   Correlation after: +0.99/+1.00.
4. `traj_on_map` is a coverage diagnostic, not a validity test (`obstacle_map.py:374`
   erases an agent-radius band every step). Recorded, never gated on.
5. Retention ranks cells first, storeys as a tie-break (00821: 2975 cells vs
   30-36k when ranked on storeys).

### B. Search and commit (`src/osg/agent/nav_agent.py`, `exploration/strategy.py`)

6. **No exploration round ever ran** after a failed attempt: `candidates.check`
   runs before the EXPLORE branch every step, and `select` is rate-limited to
   one per 5 steps; the next same-label track committed within 4 steps.
   *Fix:* `agent.explore_after_failed_attempt_steps` (25): hold the commit and
   force the round.
7. **False positives veto the floor switch and eat the attempts.** Eight
   "toy airplane" tracks on the wrong storey; 0 presence events for the label
   (a false positive IS present); `floor_target_evidence` adds
   `_TARGET_PRESENT` for believed same-label tracks and `may_switch` refuses
   to leave. *Fix:* `agent.floor_disproved_after_failed_attempts` (2): failed
   attempts disprove the STOREY -- the veto is lifted, the nearest other
   storey is requested (`ExplorationStrategy.forced_floor`), and the verdict
   is honoured by the floor LLM (`floor_llm_skipped_disproved`), the
   candidate channel (`candidates_skipped_disproved`) and the storey posterior
   (`floor_posterior_disproved_excluded`).
8. **A decided switch was abandoned nine steps later** by a new same-floor
   commit. *Fix:* `agent.protect_floor_switch`: while `floors.pursuing`, only a
   live, close (1.5 m), confident (>= 0.6) candidate may pre-empt.
9. **An empty storey right after arrival looked like "nothing here, leave".**
   Containers are rebuilt from tracks at keyframes; 3 steps after climbing
   down, `floor_mass={'0': 11.8}` named only the storey just left, and the
   agent climbed back up. *Fix:* `exploration.empty_storey_settle_steps` (60)
   -- within it an empty posterior means "unmapped"; after it the shipped rule
   (`test_floor_anchor_order`: anywhere else is better) stands.
10. `target_obj_xy` in `episodes.jsonl` is the agent's last commit, NOT the
    truth; the truth is `authored_layout.target_position`. A figure and a
    coverage reading were wrong because of this and are retracted in the log.

### C. The climb (`nav_agent._do_climb`, `floor_policy.try_switch`, `mapping/stairs.py`)

11. **Flights were picked by nearest foot**, so furniture won. *Fix:*
    `floor.flights_prefer_stair_mask`: stair-map-corroborated flights first,
    and no `flight_up` from the topmost known storey.
12. **Climb direction from a height tie fell through to "down"**
    (`target_y > here_y` False at equality). *Fix:*
    `agent.climb_direction_from_flight`.
13. **A ramp cannot know where landings are**; descending from one, the
    carrot collapsed onto the agent's feet. *Fix:*
    `agent.climb_carrot_min_ahead_m` (0.4) -- a no-op with true heights.
14. **`new_level_m` (1.8) declared a 3.0 m storey climbed at 1.83 m** with the
    agent on the treads. *Fix:* `agent.climb_to_target_storey_tol_m` (0.3):
    close the gap the prior map knows.

15. **The flight the carrot follows stopped half-way up the staircase.**
    `find_flights` keeps cells within `new_level_m` (1.8 m, minus a 0.2 m
    margin) of the floor; the ramp pasted from the ASCENT map spans the real
    3.0 m gap. Replayed offline on the stored 00800 map: the "down" flight
    from the upper storey held heights 2.92..1.56 only, the "up" flight from
    the lower 0.37..1.76 only. Past that point `_flight_carrot` has no cell to
    offer, relinking finds nothing (same cut), and the climb falls back to
    ASCENT's depth-ray carrot -- the farthest visible point, which from the
    middle of this staircase is through the balusters into the room. v13's
    video shows exactly that: the same view of the banister for 120 steps in
    ep1 (197-317) and 60 in ep2 (244-304), the point-goal policy pushing into
    the railing. The stall heights match the cut to the centimetre: ep1
    -1.51 m, ep2 +1.97 m. Not a candidate pulling the agent back -- in the
    CLIMB state the candidate check does not run and nothing reads the
    commit. *Fix:* `floor.flight_span_from_levels`: the span asked of
    `find_flights` is the gap to the nearest other known level (the prior map
    seeds both storeys' heights at episode start; the ramp was written from
    the same two numbers), never less than `new_level_m`. Replayed with it,
    both flights span 0.36..2.96 and the carrot has a waypoint at every point
    of the climb. This is the "4x slower than the probe" of item 3 below: the
    probe started aligned with the flight, so the depth ray happened to point
    up the stairs.

16. **Widening that band for every cell merged the storey below into the
    staircase.** From the upper storey the agent sees the lower one over the
    banister; furniture tops down there sit 2.0-2.8 m under the upper floor,
    which the 1.8 m band excluded and a 3.0 m one admits. They touch the
    stairwell in projection, so `ndimage.label` joined them into one
    1460-cell "flight_down" whose mouth was 3 m from the real one. v14 ep1
    pursued it for 392 steps and never climbed -- a REGRESSION from v13.
    *Fix:* the widening applies only where the stair map (ASCENT's, pasted,
    plus this run's confirmed regions) says stairs; `find_flights` takes
    `wide_span_m` + `wide_mask`. v15: the descent reached 2.44 m of 3.0 m.
17. **The periodic look-down hunts for stairs wherever it stands.** It is
    time-gated only (`down_look_every` 30): pitch down, feed whatever is in
    front to the stair detector, pitch back -- two steps, 6-16 times an
    episode, none of them at the staircase. *Fix:*
    `agent.down_look_near_stairs_m` (3.0): only within that of a stair-map
    cell, and the gate lifts itself when there is no stair map, because
    discovering an unknown staircase is what the look-down is for.
18. **The climb budget, once the flight was whole.** v15 ep1 was still
    descending at 2.44 m of 3.0 m when `climb_max_steps` (200) ended it.
    `_carrot_stalled` already ends a climb that is NOT moving, so the budget
    is now 300.

Measured in isolation (`scripts/probe_climb_osg.py`, the production climb on
the map exactly as pasted): ascending +1.86 m in 46 steps, descending -1.85 m
in 50, both `storey_of_height`.

### D. Tooling

- `scripts/probe_climb.py` -- three movers up the navmesh's own flight
  (privileged follower climbs in 37 steps; pointnav-to-top never does).
- `scripts/probe_climb_osg.py` -- OSG's real climb loop; variants `exact`,
  `ramp`, `arclen`, `depth`, `measured`, `pasted`; `--descend`, `--full-fsm`;
  prints the carrot's per-step choice, which is what found 2, 13 and 14.
- `scripts/legacy/plot_cross_anchor_map.py` and, by default, `viz/obstacle_map_<scene>.png`
  in every run that reads an obstacle map (`eval.obstacle_map_png`): the map
  with the TRUE target, the agent's last commit, its end pose, the flights it
  chose and ASCENT's stair endpoints.
- Debug videos are H.264 on close (`eval.debug_video_crf`, 30; 36 MB -> 5-10 MB).
- Run monitors poll `episodes.jsonl` and exit on `summary.json` or process
  death; the earlier `tail -f` monitors watched block-buffered stdout and
  never ended.

## 3b. The second scene: 00808 (first run, 0/3)

Everything before this section was measured on 00800. 00808 was mapped in ONE
pass (`mf5_mapping_both`, run for the first time: 8 obstacle snapshots and a
263-track scene graph from a single walk) and three cross-anchor episodes ran
at 1000 steps. None succeeded, none reached the target storey, and the reason
is upstream of anything the navigation does.

1. **No mapping episode visited both storeys.** Six of eight stayed on the
   lower floor, two on the upper. So the scene-graph prior -- kept by whichever
   episode saw the most floors -- came from one that started mid-staircase and
   recorded its upper "storey" at 1.03 m, when the storeys are 0.06 and ~3.0.
   *Fix:* `ycb.seed_storeys_from_obstacle_map`. Storeys are CLUSTERED out of
   the snapshots (each floor's recorded height weighted by explored area,
   floors within 0.9 m merged -- 00808's upper storey arrives as 2.86, 3.07,
   3.11 and 3.26, plus a landing at 0.86), missing ones are added, and a
   scene-graph storey no cluster backs is left out of the paste. Measured:
   00808 -> 0.11 and 3.04, 00800 -> 0.16 and 3.09 (unchanged behaviour there).
2. **The upper storey's stair map is a different, partial staircase.** From
   the pasted union: the LOWER storey offers two up-flights spanning
   0.32-2.84 m at x~1.8, the whole gap. The UPPER storey offers one down-flight
   spanning 0.37-1.63 m at x~-3.9 -- a different place, and a third of the way
   down. Both episodes that start upstairs therefore stall around 0.4-1.2 m,
   and `climb_flight_carrot` fires 4 and 2 times (hundreds on 00800), so the
   climbs run on the depth-ray fallback.
3. **Excluding a phantom storey from the paste does not remove it from the
   agent.** The bowl episode climbed to 1.24 m, the estimator committed a
   level there (`floor_log` 0.061 -> 1.243 -> 0.061 -> 1.146 -> 0.061), and the
   climb ended. The excluded layer is still in the stack and the estimator
   reuses it.
4. **The prior scene graph holds 1 of 8 target objects** (00800: 5 of 6) and
   no objects at all on the upper storey, so most episodes have no stale
   belief to correct.

What this says about the pipeline: per-storey MAPPING quality decides whether
pass 2 can climb at all. The union fixed horizontal coverage; 00808 needs the
vertical equivalent -- mapping episodes that actually traverse the staircase.

## 4. What is left, with the evidence

1. **Same-label false positives decide the episode.** v13 ep1, on the correct
   storey with 113 steps and one attempt left, committed to "toy airplane"
   id 578: 2.2 m above the floor, one observation, score 0.77, 1262 px. The
   map held 13 target-label tracks (`target_bypassed: 20`), scores 0.30-0.77;
   the true object (y 0.96 on a 0.16 floor) was never in view. Eight of the
   thirteen sit >0.8 m above their storey.
2. **The target is outside the map, and the cause is the writer, not the
   budget.** Measured against the scene's navmesh, the kept snapshot covers
   51% of the lower storey's navigable area and 80% of the upper. Of the 12
   authored target positions on this scene, episode 1's toy airplane is 5.1 m
   outside the mapped area and four others are 2-5 m outside.

   A 2000-step mapping pass did NOT fix it: 28% / 27% coverage, worse than
   the 500-step one (different starting episode, so the two confound, but
   more budget is plainly not the lever). The lever is that the protocol runs
   ONE MAPPING EPISODE PER AUTHORED OBJECT -- six here, each starting
   somewhere different -- and `save_obstacle_map_for_scene` keeps only
   whichever saw the most, discarding five explorations.

   *Measured, union of all six on 00800:* navigable area known goes 51% ->
   **81%** on the lower storey and 80% -> **86%** on the upper; all six
   snapshots matched by height, none skipped; and every one of the 12
   authored target positions is now within 1.2 m of mapped free space --
   episode 1's toy airplane goes from 5.1 m OUTSIDE to 0.4 m.

   *Fix:* `ycb.obstacle_map_union`. Pass 1 writes one snapshot per episode;
   pass 2 pastes all of a scene's snapshots into the same costmaps. Floors
   are matched BY HEIGHT (`floor_y`, new in the snapshot schema: the mean
   height the mapping agent stood at on that floor, recorded passively in
   `navigation/agent.py`), falling back to order matching for older
   snapshots -- four of the six episodes here mapped a SINGLE storey and
   order matching cannot place those at all. A floor that matches no storey
   within 1 m is left out and counted in `prior_obstacle_map.union_skipped`,
   never forced onto one.
3. **The climb is 4x slower in the run than in the probe** -- explained by
   finding 15 (the flight ended at 1.6 m and the depth-ray fallback drove
   into the balustrade); v14 tests the fix. The rest of this item is kept as
   it was read before that finding. v13 ep2: 200 steps of climb (the budget)
   reached +1.97 m of the 2.7 m the gap rule asks for; the estimator never
   committed the upper storey; a lower-storey fake pulled the agent back
   down; and once that attempt failed, `portal_failure_memory` had marked the
   real staircase failed (`portals_skipped_after_failure: 3`), so the next
   two flights were a 306-cell blob and the 0.76 m platform. The probe
   climbs the same pasted flight in 50 steps -- and `--full-fsm` (driving
   `agent.act()` with the state set to CLIMB) gives the identical 50 steps,
   +2.74 m, the same 22/12/16 action counts. So the FSM wrapper is NOT the
   difference. What differs is how the climb begins: the probe starts 1 m off
   the treads facing along them with a fresh point-goal policy; the run
   arrives at the flight's mouth from a 56-step walk, from whatever side, with
   the policy's recurrent state carried in. Settling this needs the run's own
   per-step climb trace, which `nav_agent` does not record (item 6). A climb
   that gained two thirds of the gap should also not be remembered as a
   failure.
4. **The scene graph on a newly reached storey is empty until its first
   keyframe** (tracks are stored; containers and rooms are not). The settle
   gate works around it; rebuilding on arrival from stored tracks + pasted
   occupancy is the complete fix.
5. **Three attempts, each spent by one false positive.** `eval.attempts` is
   the DualMap protocol and is not touched; item 1 is the lever.
6. `eval.behaviour_log` is inert for `nav_agent` (`step_trace` exists only on
   the ascentnav agent). The run has no per-step trace; the probes do.
7. Only 00800 has been exercised end to end. `mf5_mapping_both` (one pass,
   both maps) is built and unit-tested and has never run on a scene; stage
   1b on the other four scenes has not run.
8. The stair-map ranking (11) trusts ASCENT's mask; it was steering the agent
   to a phantom for six runs because of (1). Verified against the navmesh now,
   on this scene only.

## 4b. The 1500-step rebuild: what was found before any map was rebuilt

*Written 2026-09-14, while stage 1 runs. Everything here was measured, and four
items correct something stated earlier in this file or in the log.*

### A. Two golden files were pinning this container's bind mounts

`tests/unit/golden/{config_snapshot,experiment_fingerprints}.json` both failed
on a clean tree, all 85 presets, with no default having changed. Four defaults
-- `ycb.data_root`, `ycb.hm3d_root`, `ycb.layout_root`, `eval.scenes_dir` --
resolve from environment variables at import time, and the mount moved from
`/datasets/habitat-data-collector` to `/habitat-data-collector`
(`docker/compose.yaml`). A snapshot whose job is "a number cannot move
unnoticed" was reporting a *mount* instead. Fixed by substituting placeholders
(`<DATA_ROOT>`, `<HM3D_SCENE_ROOT>`, ...) before hashing, longest root first.

And the documented regeneration command, `python tests/unit/test_config_snapshot.py`,
only ever wrote **one** of the two files -- which is how the second went stale
and handed the next change a red test it had not caused. It writes both now.

### B. Two shipped tools were blind to `ycb.obstacle_map_union`

`obstacle_map_viz.plot_obstacle_map` and `scripts/check_obstacle_map_reuse.py`
each opened `maps_dir/<scene>.json` directly. Union mode never writes that file,
so with the union on -- which is the configuration this whole section argues for
-- the per-run `viz/obstacle_map_<scene>.png` was a **silent no-op** and the
frame check printed `NO SNAPSHOT` for every scene. Both now go through
`prior_map.paste_scene`.

That helper is the second half of the fix: `load_obstacle_map`'s paste loop was
extracted into `paste_snapshots(paths, costmaps, heights)` and
`load_obstacle_map` re-expressed on top of it, so an offline tool pastes the
cells a run pastes *by construction* rather than by resemblance.
`tests/unit/test_paste_snapshots.py` asserts the two produce identical grids.
A second implementation is exactly how the transposition of section 3A happened.

### C. `navmesh_floor_heights` finds the storey COUNT, not the storey HEIGHT

`get_topdown_view` is asked for the slice at `y` with a 0.5 m tolerance, so
every peak lands ~0.35 m above the floor it names: 0.513/3.513 on 00800, whose
authored floors are 0.163/3.163. A third of a metre is more than
`ycb.storey_seed_tol_m` or `_match_floors` match with. `osg.eval.floors.navmesh_storeys`
now uses the peaks only to bucket an area-uniform sample of navigable points and
takes each storey's height as the **median** of its bucket. Checked against the
collector's independently-derived `authoring.floors[].y_min`:

| scene | navmesh_storeys | collector y_min |
|---|---|---|
| 00800 | 0.163, 3.163 | 0.163, 3.163 |
| 00808 | -2.739, 0.061, 3.261 | -2.739, 0.061, **2.861** |
| 00821 | -3.470, 0.130 | -3.470, 0.130 |
| 00873 | -2.975, 0.025 | -2.975, 0.025 |
| 00878 | -2.804, 0.057, 2.796 | -2.804, **1.730** (two floors only) |

Three scenes agree to three decimals. The two that disagree are the collector's
own defect, not this one -- see E. Gap-clustering the sample directly does not
work and must not be retried: a navigable staircase makes the height histogram
continuous and one cluster swallows the scene.

### D. RETRACTED: "navigable points on obstacles" is not a frame check on its own

Section 3A reports FREE 0.49 / OCC **0.00** for the correct paste and 0.04/0.04
for the transposed one, and it is tempting to read 0.04 occupied as the
signature. It is not. Measured on the correct 00800 union:

| storey | free | occupied |
|---|---|---|
| lower | 0.802 | **0.000** |
| upper | 0.876 | **0.040** |

and the upper storey's 0.040 does not come from the union piling up: it is
0.038 at two snapshots and 0.040 at six, while free climbs 0.742 -> 0.876.
Those points are a median 2.11 m from the nearest stair cell, so it is not
stairwell bleed either. It is furniture: ASCENT stamps 0.61-0.88 m above the
agent, and a table top over navmesh-navigable floor lands in that band.

**What separates a transposed paste from a furnished one is FREE collapsing**
(0.49 -> 0.04), not occupied rising.

*Second correction, and the one that settles it.* Re-gated at 0.08, the audit
then failed a CORRECT 1500-step map at 0.137 on 00800's lower storey -- while
the frame check passed at 100% of 1499 walked poses, all 12 authored targets
stayed inside, and KNOWN coverage went UP, 0.802 -> 0.887. Longer episodes see
more of the storey, and some of what they see is furniture standing in ASCENT's
0.61-0.88 m band over floor the navmesh calls navigable. The occupied cells form
217 blobs with 86 of 3+ cells: furniture-shaped, not scattered.

No threshold can work, and the log's own numbers say why: the transposed map
read FREE 0.04 / **OCC 0.04**. Its occupied fraction was LOWER than a healthy
furnished storey's, because a map rotated away from the world stamps almost
nothing where the floor actually is. A cutoff loose enough to pass 0.137 cannot
fire at 0.04. The occupied gate could never have caught the one case it was
added for.

So `occupied` is REPORTED and never gated. `--min-free-frac` is what catches a
transposition (0.80 -> 0.04 walks straight through it), and the real frame test
is `scripts/check_obstacle_map_reuse.py` against the poses the agent walked.

### E. 00878 could not be put on its two lowest storeys, and why

The benchmark rule adopted for this rebuild is: **authored targets live on at
most two storeys, and every one of them on the main navmesh island.** Measured
across the five scenes, only 00878 broke it -- its 8 targets were spread over
all three storeys (-2.80, 0.05, 2.80). Every target in every scene was already
on the main island.

Vacating the **top** storey turned out to be impossible, and the collector's own
numbers say why: of 17 usable anchors, 9 are on the 2.80 m storey and 8 on the
two lower ones -- but the basement holds exactly **one** (`table_385`), and
`cabinet_316` is affordance-rejected for all 8 targets. Seven usable anchors,
eight targets, one target per anchor. So the **basement** was vacated instead:
one target moved in each of the three layouts (`005_tomato_soup_can` static and
in_anchor -> `cabinet_235`; `072-a_toy_airplane` cross_anchor -> `table_252`),
and 00878 now uses storeys 1 and 2, the same shape 00808 already had.

It kept all **6** cross-floor relocations, scored by OSG's own rule
(`|snap(cross).y - snap(static).y| > ycb.relocation_floor_tolerance_m`,
`ycb_env.py:465-483`) -- which reads the NAVMESH, not `anchor.floor_index`. That
distinction is load-bearing here, because the collector's `cluster_floor_levels`
**under-segments 00878**: its `floors[0]` spans y_min -2.804 to y_max 0.396,
merging the basement with the ground storey, since the staircase between them
carries enough navigable area per 0.1 m bin that the 1.3 m gap rule never fires.
Recorded, not fixed -- nothing on this path reads it.

State after the edit, all five scenes: targets on exactly two storeys, all on
the main island, `validate_dualmap_authoring` OK 5/5, `verify_dualmap_layouts`
0 problems 5/5, `audit_reachability` 5/5 reachable on one connected navmesh,
worst approach 1.01 m.

### F. The released DualMap scenes have a reachability ceiling we do not set

00848 has 3 navmesh islands and 00880 has 4, and their released layouts put
targets on the small ones. Measured to the nearest point on the **main** island:
most are 0.43-0.67 m away and so are answerable under DualMap's 1.0 m rule, but
two are not -- 00880 static `003_cracker_box` at **1.54 m** and in_anchor
`029_plate` at **1.16 m**. Those layouts are the published benchmark and are not
ours to edit; the numbers are reported beside the SR, never inside it.

### G. Does moving a YCB object need a remap? (`scripts/probe_remap_need.py`)

Asked and answered from geometry. On 00800's cross_anchor layout, against the
union map: **obstacle map REUSE** -- 0 of 6 objects could change it, every
destination cell being already OCCUPIED by its support furniture (ring 0.67-1.00)
or UNKNOWN, never FREE-and-not-already-inflated. **Scene graph REBUILD** -- 0 of
6 targets carry a prior track, which is the 213-tracks-and-no-YCB-targets fact of
section 3b item 4 restated by an instrument instead of by hand.

The probe's object extents come from the render asset's glTF POSITION accessors,
rotated by the config's `(up, front)`. That rotation is load-bearing and was
checked against published YCB dimensions rather than assumed -- standing height,
rotated vs raw vs published: cracker box 0.213/0.164/0.213, soup can
0.102/0.068/0.101, bowl 0.055/0.161/0.055, banana 0.037/0.178/0.036. Rotated is
within 1-7 mm; raw is wrong by 2-5x.

### H. The mapping pass filed EVERY track on one storey

`mf5_mapping_both` inherits `ascentnav`, whose floor group is off
(`floor.enabled=false`, `estimate_only=true`). `FloorPolicy.observe` advances
`stack.current_id` only when `enabled and not estimate_only`, so in the mapping
pass it stayed 0 for the whole episode -- and `world_model.py:219` files every
track it creates under `floor_key=self.floors.current_id`.

Measured on 00800 at 1500 steps, before the fix: the snapshot recorded BOTH
storey heights (3.16 and 0.16) and put all **402 tracks on floor_key 0**, with
centre heights spanning -0.01 to 5.55 m and **184 of them nearer the lower
storey** than the one they were filed under.

It is not cosmetic, because the reader disagrees with the writer. Pass 2 runs
with `floor.enabled=true` and `scene_graph._rebuild_containers` selects tracks
BY `floor_key` (`scene_graph.py:214`), so a storey whose tracks are all filed
elsewhere rebuilds with nothing on it. That is section 3b item 4 ("the prior
scene graph holds 1 of 8 target objects and no objects at all on the upper
storey") and section 4 item 4 ("the scene graph on a newly reached storey is
empty until its first keyframe") seen from the writing end -- both were read as
a coverage or a keyframe problem, and at least part of each is this.

*Confirmed, same scene and same 1500-step budget:*

| | before | after |
|---|---|---|
| tracks | 402 | 411 |
| on `floor_key` 0 (y=3.16) | **402** | 156, centre y median 3.72 |
| on `floor_key` 1 (y=0.16) | **0** | 255, centre y median 0.83 |
| filed further from their centre than another storey | **184 (46%)** | **25 (6%)** |

The 6% residue is tall furniture reaching between storeys, which is what the
0.5 m slack in the test is for; the 46% was the defect.

Not fixed by this, and still open: the prior holds **1 of 6** YCB target labels
on 00800. That is the best-of rule, not the floor id -- `save_map_for_scene`
keeps ONE episode's graph however many episodes run, so more episodes cannot
help it. The lever is chaining `ycb.map_in == ycb.map_out` across a scene's
mapping episodes, which needs `load_prior_map` to unwrap `agent.world` the way
`save_map_for_scene` already does (section 5, item C).

*Fix:* `mf5_mapping_both` now composes `mf5_osg_unified`'s floor block, so the
prior is built under the same floor semantics it is read under. Safe against
the transcription: `src/navigation/agent.py` never reads `cfg.floor` -- ASCENT
carries its own storey stack in `_floors`, which is what writes the obstacle
map -- so this changes what the OBSERVER records and nothing about the control
flow being transcribed. Nothing inherits `mf5_mapping_both`, so exactly one
experiment fingerprint moves.

### I. How many mapping episodes are enough

The union of every episode was the right answer AT 500 STEPS, and the reason is
that a 500-step episode cannot cross a staircase: on 00800 the first snapshot
covered 0.775 of the lower storey's navigable area and **0.000** of the upper,
and two authored targets were outside the map entirely. Coverage and the target
gate, against the navmesh, as snapshots are added:

| episodes | lower | upper | mean | gain | targets outside 1.5 m |
|---|---|---|---|---|---|
| 1 | 0.775 | 0.000 | 0.388 | -- | **2** |
| 2 | 0.775 | 0.742 | 0.759 | +0.371 | 0 |
| 3 | 0.775 | 0.835 | 0.805 | +0.047 | 0 |
| 4 | 0.785 | 0.846 | 0.815 | +0.010 | 0 |
| 5 | 0.802 | 0.852 | 0.827 | +0.012 | 0 |
| 6 | 0.802 | 0.876 | 0.839 | +0.012 | 0 |

At **1500 steps** one episode crosses, and the same scene's first snapshot
measured 0.787 / 0.777 -- mean **0.782**, past the three-episode 500-step union,
with **0** targets outside. Paired against the same episode (50004, identical
start pose) at 500 steps, the upper storey had already saturated
(19787 -> 19790 cells) and the whole extra budget went into the lower one
(9324 -> **20110**).

This is also the answer to the log's "a 2000-step pass covered LESS (28%/27%)":
that comparison used a different starting episode and confounded budget with
start pose. Paired, the budget is worth 1.4x the cells.

So the tail is real but cheap to lose: episodes 4-6 bought about +0.01 of mean
coverage each for ~13 minutes apiece and moved no target. `MAX_EPISODES` in
`scripts/run_prior_maps.sh` cuts it. Note this bounds the OBSTACLE map only in
a useful way -- the scene graph is kept best-of, not unioned, so one episode's
graph wins whatever the count, and the lever for its YCB target coverage is
chaining `ycb.map_in == ycb.map_out`, not more episodes.

## 4c. The climb: one real bug, three measured negatives, and the constraint

*Written 2026-09-15, from the 15 pass-2 episodes of the 1500-step priors.*

**The discriminator is the climb, and nothing else.** Over all 15 episodes,
`climb_ok >= 1` if and only if the agent reached the target storey -- 15 of 15,
no exceptions. Four climbs completed and those four arrived; the other eleven
gained <= 0.40 m of the 2-3.8 m they needed. Storey logic, commit hygiene and
coverage are all downstream of that: fixing them cannot rescue an episode whose
climb gains 0.00 m, and two of them were measured doing exactly nothing.

### A. `foot_xy` is the mouth of an ASCENT only

`Flight.foot_xy` is the LOWEST tread by definition and `top_xy` the highest, so
the foot is where an ascent joins the run and the top is where a DESCENT does.
Every place the agent asked "where do I step on" used the foot: `_best_flight`'s
ranking, the portal failure memory, `nav_agent`'s relink, and the goal handed to
the mover. On a descent that is a point at the bottom of the staircase -- on the
storey the agent has not reached -- so it walked toward it, circled at the top
and stalled.

*Fix:* `stairs.mouth_xy`, used in all five places. Measured on 00821's bowl
episode, the same episode and the same prior:

| | before | after |
|---|---|---|
| `climb_max_dy` | **0.00 m** | **0.86 m** |
| `steps_on_a_flight` | 47 | 578 |
| `climb_flight_carrot` | 1 | 66 |
| `distance_to_goal` | 15.91 | 12.85 |

and 00800's cracker box, an ASCENT, is byte-identical -- which is the control:
for an ascent the mouth IS the foot, so the change is a no-op there by
construction. It is still not an arrival (0.86 m of a 3.6 m gap, ending
`stalled`), because of C.

### B. Three things that did not work, with their numbers

- **A forward-only carrot.** The idea was that a ramped height field lets the
  nearest in-band cell sit behind the agent. Filtering to the half-plane toward
  the far end changed 00821's descent by NOTHING, byte-identical. The carrot
  trace says why: the in-band cells were already ahead, and what spins is the
  MOVER -- 249 turns against 50 forwards on a goal 0.46-1.14 m away.
- **Clipping the ramp to ASCENT's recorded stair ends** instead of the whole
  mask: 1027 cells become 1026, correlation with the true surface +0.701 ->
  +0.702. The recorded ends already span the mask.
- **First-writer-wins on the union's ramps.** Correlation +0.70 -> **+0.10**,
  worse. Nine invented ramps averaged over a cell are CLOSER to the truth than
  any one of them; taking the first takes one episode's guess over their
  consensus.

All three stay in the code, off, as `climb_turn_deadband_deg` does.

### C. The constraint, measured

Against the navmesh under the flight's own cells, on 00821's pasted descent:

- correlation between the pasted ramp and the true surface: **+0.70**
- the ramp spans **3.17 m** where that surface spans **1.34 m** -- stretched 2.4x
- the flight's lowest and highest cells land **0.27 m apart** for a 3.18 m flight
- the union writes **nine** ramps over the same costmap, from nine endpoint
  pairs, several running in opposite directions

And the isolation probe, same flight and same climb loop, only the height field
differing: true tread heights **ARRIVE** (-3.32 m in 60 steps, 0 forced
forwards); a linear ramp manages -0.93 m in 300 steps with 249 turns; the stored
map gains 0.00 m.

ASCENT's map is 2D and the heights are invented. No rearrangement of an invented
field makes it a staircase. The two ways forward are therefore to give the climb
REAL heights -- the agent's own depth at the mouth, rather than the pasted ramp
-- or to stop asking a point-goal policy to chase waypoints 0.4-1.1 m apart,
which is where it spins. Both are larger than a flag.

## 4d. The climb, fixed for descents: two changes and what each was worth

*2026-09-15. Both were measured on one episode and one probe before the arm
moved, and the ascent is deliberately reported as still broken.*

### A. The mover will not converge on a goal a step away

The waypoint climber aims at the nearest tread 0.35-1.0 m further along in
HEIGHT, which on a real staircase is often under a metre away in the plane --
and a point-goal policy given a goal that close circles instead of arriving.
Measured with `probe_climb_osg.py` on 00821's descent, `ramp` heights (the
realistic case), one flight and one climb loop, only `climb_carrot_min_ahead_m`
changing:

| spacing | result |
|---|---|
| 0.4 (shipped) | -0.93 m, 300 steps, fell back 0.29, 249 turns against 50 forwards |
| 0.8 | -3.43 m, 263 steps, arrives |
| **1.2** | -3.34 m, **63 steps**, arrives |
| 1.6 | -3.31 m, 62 steps, arrives |

63 steps against the **60** the same climb takes on TRUE tread heights. At this
spacing the synthetic ramp costs almost nothing, which reverses the reading of
section 4c: the ramp was never the primary defect. The mover was.

### B. A real depth reading could never displace an invented height

`Costmap2D._record_heights` keeps a running MINIMUM per cell (`np.fmin.at`),
and the pasted ramp invents heights that run LOW -- on 00821 it reaches -3.26 m
where the true surface under the same cell is -1.21. So the agent's own depth,
which is higher and correct, lost every comparison and the climber followed the
fiction for the whole episode.

*Fix:* `Costmap2D.height_synthetic` marks what the ramp invents, and a first
real observation REPLACES it rather than being min'd against it; the cell then
rejoins the running minimum like any other.

### C. What the two are worth, on the episode rather than the probe

00821's bowl, a DESCENT, same episode and same prior throughout:

| | original | + `mouth_xy` | + both |
|---|---|---|---|
| `goal_floor_reached` | False | False | **True** |
| `floor_changes` | 0 | 0 | **2** |
| `traj_y_range` | 1.06 | 1.44 | **3.60** (the whole gap) |
| `climb_ok` | -- | -- | **2** |
| `climb_end_stalled` | 1 | 1 | **none** |
| `distance_to_goal` | 15.91 | 12.85 | **10.11** |

`floor_log` ends `0.13 -> -1.67 (step 279) -> -3.429 (step 317)`: both flights
descended. Reaching the storey is the binary that separated all four arriving
episodes from the eleven that failed, so this is the gate opening, not a
tuning gain.

### D. Still broken: the ASCENT on 00800

00800's cracker box gains 0.00 m with either spacing, so `climb_carrot_min_ahead_m`
is not its problem. The probe finds NO up-flight at all on that scene's pasted
map (`end={'why': 'no flight'}`), which puts the defect in the flight geometry
rather than the climb. Its `distance_to_goal` moved 16.73 -> 22.36, which is a
different exploration path and not a capability change.

So: descents climb, ascents on 00800 do not, and the next question is why
`find_flights` returns nothing upward on a map whose stair mask is plainly
there.

### E. The full-arm result: a wash on the paired set, not the gain it first read as

`outputs/mf5_pass2_fixed` (26 episodes, all five scenes) against the pre-fix
`outputs/mf5_pass2_p1500`. The two runs do NOT cover the same episodes -- 00821
ran 4 then 8, and 00878 has no pre-fix run at all -- so the aggregate 4 -> 9
"reached the goal storey" is mostly new episodes, not improved ones. Paired on
the **15 episodes both runs ran**:

| | before | after |
|---|---|---|
| success | 1/15 | 2/15 |
| reached the goal storey | 4/15 | **4/15** |
| `climb_ok >= 1` | 4/15 | 4/15 |

One gained the storey (00821 `50004`, the episode section C tuned on: dy
1.06 -> 3.60) and one lost it (00873 `50006`: dy 3.07 -> 1.80). So the honest
reading is that the change is **demonstrated on the episode it was measured on
and neutral across the paired set** -- not the capability gain the unpaired
tally suggested.

The 00873 regression has a specific and consistent cause: the wider carrot
doubled `climb_carrot_held` (562 -> 1174) and both climbs ended
`climb_end_budget` against `climb_max_steps: 300`, with `climb_max_dy_x100`
falling 271 -> 176. **`climb_carrot_min_ahead_m` trades steps for reach, and the
sign of that trade is scene-dependent**: on 00821's flight the wide carrot cut
the climb from 300 steps to 63, on 00873's it spent the budget without arriving.
A single global value is therefore the wrong shape for this parameter; spacing
relative to the measured flight span is the thing to try next.

The 11 unpaired episodes (00878's 6, 00821's 4 extra, 00808's 1) ran 3/11
success with 5 reaching the goal storey, which is the first number we have for
00878's re-authored layout and is not comparable to anything.

Two episodes also break the `climb_ok >= 1` <-> `goal_floor_reached`
equivalence that held 15/15 before (00808 `50001` climbs without arriving;
00873 `50001` arrives with no logged climb), so that equivalence is now a
regularity of the old run, not an invariant.

## 4e. Reversing the climb direction, and the anchor ceiling that limits it

### A. Every success in the arm is a descent

Split the 26-episode arm by the direction the episode demands -- the agent
starts on the storey the PRIOR puts the object on (`ycb.start_on_prior_floor:
True` is composed into this arm, with `start_min_geodesic_m: 1.0`), so the
demand is exactly `sign(cross_anchor_floor - static_floor)`:

| direction | n | success | reached the goal storey |
|---|---|---|---|
| UP | 11 | **0** | 1 |
| DOWN | 15 | **5** | 8 |

All five successes are descents. This is the same asymmetry §4d-D found on
00800 and it is not a 00800 quirk.

Distance to the stair mouth is a second, weaker lever: every success sat
1.4-9.2 m from a mouth and the worst failures sat 9-15 m out. But it does not
rescue an ascent -- 00873's two UP episodes sit **1.7 and 1.9 m** from the
mouth and still reach nothing.

### B. Direction can only be changed by re-authoring `static`

No config knob inverts it. `ycb.relocation_directions` only DROPS episodes, and
the start is pinned to the static storey. So an UP episode becomes a DOWN
episode only by exchanging the target's static and cross_anchor poses
(`scripts/flip_cross_anchor.py`). Both poses were generated and validated by
the same generator, so the exchange cannot invent a floating or unreachable
pose -- but it DOES mix two independently validated layouts, and the dry run
found what that costs: objects 0.02-0.33 m apart against the generator's 0.35 m
keep-out, and up to three targets on one anchor. Both ends therefore have to be
re-placed through `replace_placement.py` afterwards, not just written out.

### C. The ceiling: upper-storey anchor supply

A descent needs the target's STATIC pose on the upper storey, and
`allowed_anchors` filters by room as well as by kind -- food items want
kitchen / dining_room / living_room. Anchors affordable to a food target, per
scene, on the storey a descent must start from:

| scene | upper storey | affordable anchors | descents it can support |
|---|---|---|---|
| 00800 | +3.16 | 3 | 2 |
| 00808 | +3.26 | 4 (+ airplane's 8) | ~5 |
| 00821 | +0.13 | 16 | 8 (all) |
| 00873 | +0.02 | **1** (`table_133`) | **1** |
| 00878 | +2.80 | **2** (`cabinet_81`, `cabinet_164`) | **3** |

The generator allows one target per anchor, so this is a hard cap, and 00873
and 00878 were ALREADY at it before the flip. Their ascents cannot be reversed
without either weakening the affordance rules (the collector's design, not
ours to bend) or re-opening 00878's basement (vacated by decision in §4b-E).

### E. What the re-authoring actually achieved

`scripts/flip_cross_anchor.py` + `scripts/restage_near_stairs.py`, both driving
the collector's own `replace_placement.py` so every new pose is generated under
the full rule set, not written as coordinates.

| scene | before D/U/S | after D/U/S |
|---|---|---|
| 00800 | 1/1/4 | 2/0/4 |
| 00808 | 4/3/1 | 5/2/1 |
| 00821 | 6/2/0 | 7/1/0 |
| 00873 | 1/2/4 | 1/2/4 |
| 00878 | 3/3/2 | 3/3/2 |
| **total** | **15/11/11** | **18/8/11** |

26 cross-floor episodes preserved. Of 11 attempted flips, **3 survived** --
00873 and 00878 were already at their anchor ceiling (§4e-C), and one 00821 flip
was reverted because it forced two targets 0.28 m apart on the only free anchor,
under the dataset's own 0.35 m rule.

The "move the goal to the stairs" half delivered less than the direction change:

| | before | after |
|---|---|---|
| median goal -> stair mouth | 5.31 m | 5.29 m |
| goals within 4 m | 7/26 | 10/26 |
| goals beyond 9 m | 8/26 | 6/26 |

The median barely moves because the destination storeys are **saturated**: 00821's
lower storey has 7 living-room anchors for 8 targets and every free anchor there
is a bathroom or bedroom fixture the affordance rules reject. Where there was
supply the effect is large -- 00800's median goal distance went **9.6 m -> 1.8 m**.

All five scenes pass `verify_dualmap_layouts`, `validate_dualmap_authoring` and
`audit_reachability` (1 island each, worst approach 0.51-1.01 m), with the single
pre-existing exception that also fails on the untouched backup: the multifloor
set moves 6-8 targets across storeys where the validator's default range is 1-3.

Pass 1 must be re-run for the **4 scenes whose STATIC layout moved** (00800,
00808, 00821, 00878 -- 00873's statics did not move), because the scene-graph
prior records where pass 1 saw each object and that is now stale. ~132 min
serial at 3 episodes x 1500 steps, then ~97 min for pass 2.

### D. Stated plainly: this lowers the benchmark rather than fixing the agent

Turning ascents into descents removes the failing case from measurement. The
ascent defect -- §4d-D, where `find_flights` returns nothing upward on 00800's
pasted map -- is untouched and becomes invisible in any arm with no ascents
left. The remaining UP episodes are therefore worth keeping and reporting
separately rather than treated as noise.

## 5. Roadmap

Each step names the number that gates it. Nothing here is a tuning knob.

**A. Commit hygiene** (item 1; the head of the queue)
- Close the SCORE half of `scene_graph.target_bypasses_gates`; keep the size
  bypass (a distant true target is small). Gate: target-label tracks per
  episode (13 -> ?), `target_bypassed`, commits per episode.
- Height prior at commit: a track more than ~1 m above its storey needs a
  second observation before it may take an attempt. Gate: wrong commits above
  0.8 m (every one in v11-v13) -> 0.
- Then re-run 00800: the first attempt on the correct storey should go to a
  plausible place, and `gt_in_view_frames` should leave 0.

**B. Climb speed** (item 3)
- Done: `--full-fsm` is identical to the direct call, so the wrapper is
  cleared. Done: the flight was cut at 1.6 m of a 3.0 m gap (finding 15),
  `floor.flight_span_from_levels` in v14. Gate: `climb_max_dy_x100` >= 270
  and `climb_end_storey_of_height`/`new_floor` instead of `budget`.
- Still worth having: a per-step climb trace for `nav_agent` (agent xy,
  standing height, carrot goal, action -- what the probe prints), so the next
  climb defect is read off the run rather than off video frames.
- Raise `climb_max_steps` only once the cause is known.
- Do not let `portal_failure_memory` retire a flight the agent gained most of
  the gap on (v13 ep2: +1.97 of 3.0 m, then excluded). Gate: the real
  staircase is re-chosen after a budget-ended climb.

**C. Scene graph on arrival** (item 4)
- On a storey change, `rebuild_floor` for the new storey from its stored
  tracks with room labels from the pasted occupancy, before the first
  keyframe. Gate: `floor_posterior_empty_here` -> 0 with the settle gate off.

**D. Coverage** (item 2)
- Done: the union of every mapping episode's snapshot, matched by height
  (`ycb.obstacle_map_union`). Gate: navmesh coverage of each storey, and how
  many of the 12 authored target positions fall inside the map.
- If the union still leaves episode 1's target outside, the remaining options
  are a longer mapping budget per episode, `mf5_mapping_both`, or accepting
  that this particular target is unreachable-by-prior and saying so in the
  result rather than reporting it as a search failure.

**E. Scale** (item 7)
- Stage 1b on the other four scenes, one at a time; then pass 2 on all five.
  The bar set for this: about 3-4 successes per scene.
- Re-verify the paste against each scene's navmesh (the check in the log,
  section "THE PASTE WAS TRANSPOSED") before trusting any climb on it.

**F. Consolidation**
- An integration test (`pytest -m sim`) that pastes a stored map and checks
  navigable points land FREE / never OCCUPIED, so the transposition class of
  bug cannot return silently.
- `step_trace` for `nav_agent`, so a run's climb can be read the way the
  probe's can.

## 6. Flags added (all default to the shipped behaviour; the arm sets them)

| flag | default | `mf5_osg_on_ascent_map` |
|---|---|---|
| `ycb.obstacle_map_out` / `obstacle_map_in` / `obstacle_map_overwrite` / `map_in_occupancy` | "" / "" / false / true | set per run |
| `agent.osg_world_model` (+`_weights`) | false | `mf5_mapping_both` only |
| `agent.explore_after_failed_attempt_steps` | 0 | 25 |
| `agent.floor_disproved_after_failed_attempts` | 0 | 2 |
| `agent.protect_floor_switch` (+`_range_m`, `_min_score`) | false (1.5, 0.6) | true |
| `agent.climb_direction_from_flight` | false | true |
| `agent.climb_carrot_min_ahead_m` | 0.0 | 0.4 |
| `agent.climb_to_target_storey_tol_m` | 0.0 | 0.3 |
| `floor.flights_prefer_stair_mask` | false | true |
| `floor.flight_span_from_levels` | false | true |
| `agent.down_look_near_stairs_m` | 0.0 | 3.0 |
| `agent.climb_max_steps` | 200 | 300 |
| `ycb.obstacle_map_union` | false | set per run |
| `exploration.empty_storey_settle_steps` | 0 | 60 |
| `eval.debug_video_crf` | 30 | -- |
| `eval.obstacle_map_png` | true | -- |
| `navigation/mapping/map_store.RAMP_MARGIN_M` | 0.0 (constant) | -- |

Every addition was checked against all 85 presets: the new key is the only
difference in each composed config, and `mf5_osg_on_ascent_map` is the only
arm off the defaults.

## 7. How to run and read

```bash
# pass 2 on one scene
python scripts/run_eval.py +experiment=mf5_osg_on_ascent_map \
    'ycb.scenes=[00800-TEEsavR23oF]' ycb.obstacle_map_in=outputs/maps_mf5_ascent \
    ycb.map_in=outputs/maps_try_steps ycb.map_in_occupancy=false \
    eval.debug_frames=true output_dir=outputs/mf5_pass2_vN

# the climb in isolation, on the map as pasted
python scripts/probe_climb_osg.py --scene 00800-TEEsavR23oF --variants pasted [--descend] [--full-fsm]
```

In `episodes.jsonl`: `goal_floor_reached`, `floor_changes`, `traj_y_range`
say whether the storey change happened; `agent_stats.climb_max_dy_x100`,
`climb_ok`, `steps_on_a_flight` describe the climb; `floors_disproved`,
`floor_disproved_to`, `candidates_skipped_disproved`,
`floor_posterior_disproved_excluded`, `floor_posterior_empty_here` describe the
storey verdict; `goal_commit_log` (with `seen_live`, `center`) and
`target_tracks` (with `best_score`, `n_obs`) describe what took the attempts.
`authored_layout.target_position` is the truth; `target_obj_xy` is not.

In the container, run under `sg hm3ddata` with `umask 002`: the dataset is
group-readable only, and something on the host chowns the repo to uid 1007
mid-run.
