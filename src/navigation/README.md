# `ascentnav` — ASCENT's control flow, on ASCENT's models, in OSG's harness

## What it is

`AscentNavAgent` (`agent.py`) is a transcription of ASCENT's `Ascent_Policy.act`
and the `Map_Controller` it drives (`relative_work/ascent/ascent/`), for one
environment, on the vendored ASCENT maps. Every rule carries the reference
line it was transcribed from. It is the default arm (`+experiment=ascentnav`)
and the one `final_sensor` runs on the full split.

It replaced a port that scored 54.0% on `scenes20_ep0to4` against the
reference's 65.0%. Seventeen configuration and perception arms never beat it;
a paired trace diagnosis against the goal geometry showed why. Both agents
reach the object about equally often, but the port ended 14 episodes without
ever having the target in frame (the reference: 8) and timed out 6 more the
same way (reference: 1). Two mechanisms carried the loss:

* **H2 — the stair sink.** 17.5% of the port's steps were spent in a climb
  mode against the reference's 9.2%, most of it on same-floor episodes. Its
  frontiers ran out early (the sticky rule could retire a floor's last
  frontier, the disabled set was per-episode, there was no stairwell
  re-initialisation) and its stair mask was RedNet's union where the
  reference ANDs RedNet with GroundingDINO.
* **H1 — the premature STOP.** The port could stop on step 13 with the opening
  scan unfinished; the reference's earliest wrong stop is step 24, because
  13 turns precede the goal check, the gate needs `try_to_navigate` set on a
  PRIOR step, and the stall test needs two steps inside the metre.

Neither is a model or a threshold. Both are control flow, and both are what
this package now reproduces rather than approximates.

## Layout

```
ascentnav/
  agent.py         AscentNavAgent: act() dispatch, _navigate, _explore, _initialize,
                   stairwell re-initialisation, the give-up path, trace/stats
  stairs.py        StairController: passive entry, get_close/climb, the pause branch,
                   the disable path (with the reference's dead burn branch kept, F3)
  planner.py       AscentLLMPlanner: value ranking, force/nearby/sticky rules,
                   frontier images + SSIM dedup, the single-/multi-floor prompts,
                   the knowledge-graph and floor priors
  perception.py    per-step RAM++ tags + Places365 room, keyed by _floor_num_steps
  depth_filter.py  filter_depth / fill_in_multiscale (the maps see hole-filled depth)
  geometry.py      episode-frame anchor; OSG world frame -> ASCENT's episodic frame
  constants.py     verbatim from ascent/constants.py
  mapping/         obstacle_map, value_map, object_point_cloud_map (from ascent/)
                   map_store: the obstacle-map stack, saved and reused (below)
  vendor/          the frontier_exploration and vlfm helpers those maps import
```

## The models

The reference's own, served as ASCENT itself serves them
(`bash scripts/serve_perception.sh`, five Flask servers in the `ascent` env):

| role | model | port | rule |
|---|---|---|---|
| detector | D-FINE + one MobileSAM mask per box | 13186 / 13183 | COCO names, target detections at conf ≥ 0.8, **every** detection ingested |
| gate | BLIP-2 ITM cosine on the whole frame | 13182 | `_double_check_goal` latches at ≥ 0.15, read from the PREVIOUS step's value-map call, never cleared |
| stairs | RedNet ∧ GroundingDINO `"stair ."` ≥ 0.60 | 13184 | the strict fusion; `stair_up_mode: rednet` is the union A/B |
| prompt | RAM++ tags + Places365 room, per step | 13185 / in-process | ASCENT's prompts verbatim |
| LLM | Qwen2.5-7B, local ollama | — | ASCENT's system message; any failure keeps the value ranking |

`detector.strict` / `exploration.value_strict` make a server that does not
answer raise `PerceptionUnavailable` instead of returning a neutral value;
`probe_served_models` checks all five before Habitat loads. Under this gate a
silent 0 from BLIP-2 is an agent that never STOPs.

## Storing the obstacle map, and navigating on it later

`mapping/map_store.py` writes the `ObstacleMap` stack this agent builds -- one
storey per discovered staircase -- and gives it back to a later run. The point
is the cross-anchor experiment: **pass 1** lets ASCENT's navigation explore the
static layout and keeps its map; **pass 2** runs OSG's presence-filter pipeline
over the moved layout, planning on occupancy it did not build.

```bash
# pass 1 -- ASCENT's navigation and models, static layout, keep the map
python scripts/run_eval.py +experiment=mf5_ascentnav_map \
    'ycb.scenes=[00800-TEEsavR23oF]' ycb.obstacle_map_out=outputs/maps_mf5_ascent

# is the stored map in the frame the OSG side thinks it is?
python scripts/check_obstacle_map_reuse.py \
    --maps outputs/maps_mf5_ascent --run outputs/mf5_pass1 --png /tmp/map.png

# pass 2 -- the presence filter, on that map, on the MOVED layout.
# map_in_occupancy=false is the point: the snapshot supplies the object tracks
# and the storeys, and NONE of its occupancy, so the ASCENT map is the only
# thing the planner reads.
python scripts/run_eval.py +experiment=mf5_osg_on_ascent_map \
    'ycb.scenes=[00800-TEEsavR23oF]' \
    ycb.obstacle_map_in=outputs/maps_mf5_ascent \
    ycb.map_in=outputs/maps_mf5_osg ycb.map_in_occupancy=false
```

Three things are worth knowing before using it.

* **Loading is a resample, not a copy.** These maps are anchored at the pose
  the episode started from and rotated to its facing (A12), OSG's are
  world-axis aligned, `BaseMap._xy_to_px` swaps the axes and flips the row, and
  ASCENT's frame is `(x, -z)` where OSG's `PLANE` is `(x, z)`. Copying with a
  shifted origin -- which is all `_CostmapView` does, and it is only ever drawn
  -- puts the walls **10 m out at heading 0 and 22 m at pi/2** (measured).
  `apply_to_costmap` maps every destination cell back through all three.
* **The navigable maps are recomputed, never restored.** They are a dilation of
  the obstacle mask by the agent radius, so restoring them would freeze the
  radius a previous run happened to use into this one.
* **`traj_on_map` is a coverage diagnostic, not a validity test**, and it reads
  exactly like one, which is why it is flagged here. `explored_area` is what
  the agent SAW: `obstacle_map.py:374` erases every cell within an
  agent-radius dilation of an obstacle on every step, and `reveal_fog_of_war`
  propagates only through navigable cells, so a stairwell interior is never
  marked at all. One episode each on 00800, all three maps correctly framed:
  bowl 0.54/0.77, banana 0.00/0.18, pitcher **0.11 with zero climb steps**.
  Gating reuse on this number rejects every map there is. What validates the
  FRAME is the unit tests, which pin the transform against the obstacle map's
  own projection at four start headings, plus the picture the check script
  draws.

`apply_obstacle_maps` is the symmetric operation: it restores a stored stack
onto a fresh `AscentNavAgent` rather than into an OSG costmap, for the paired
A/B of whether ASCENT's own navigation is helped by its own prior map. Nothing
in the eval wiring calls it yet; `ycb.obstacle_map_in` is the OSG-side path.

Floors are matched **by order** -- the stored list is bottom storey first,
because `_new_floor` appends on an up-stair and inserts at 0 on a down-stair --
against the OSG stack sorted by height. An `ObstacleMap` records no world
height, so there is nothing else to match on; with a multi-storey `ycb.map_in`
already loaded the two orders agree, and `prior_obstacle_map` in
`episodes.jsonl` records which storey went where and how many cells each
contributed, so a snapshot that loaded onto the wrong floor is visible rather
than silent.

## Fidelity notes worth knowing before reading the code

* The gate reads the previous step's cosine (F1): object map before value map,
  as in the reference.
* `min_distance_xy` is overwritten every in-band step (a previous-step stall
  test, not a running minimum) and reset only at episode reset (F11).
* The failure path burns the cloud's cells, clears the cloud, resets
  `_try_to_navigate`, and returns `_explore()` on the same step; the gate stays
  latched. The abandon counter is cumulative per episode and tested after the
  mover ran, on the far branch only.
* A policy STOP on the far branch of `_navigate` and on the way to an
  unexplored floor's staircase is returned raw (it ends the episode); on a
  frontier it becomes FORWARD; on a flight it marks the centroid reached.
* `_disable_stair_and_reset_state` zeroes the climb flag before testing it, so
  the stair burn never runs and a failed staircase is retried at once (F3).
  `+agent.stair_disable_burns_map=true` runs the branch the code was written for (the
  flag is read with `getattr` and has no `AgentConfig` field, hence the `+`).
* The pause ≥ 30 branch does not switch floors: it copies the flight to the
  neighbour map and re-runs the 13-turn scan on the same floor (F5).
* The multi-floor LLM prompt is dead in the reference run (F2); it is ported
  behind `exploration.llm_multi_floor: false`.
* `_pointnav` is called with `stop_radius=0.0` everywhere (F8).
* Maps are anchored at the episode start, not the habitat world origin
  (`geometry.EpisodeAnchor`); the trace logs both frames.
* `agent.downstair_detector`: `ascent` is the reference's mirrored-depth
  trigger; `lip` is OSG's "missing floor" rewrite, kept as the A/B.

`open3d` is absent; its two `cluster_dbscan` calls are `sklearn.cluster.DBSCAN`.

## Beyond the reference (`agent.search.*`)

Everything above is the transcription. `src/navigation/search/` and the
`agent.search.*` flags (`AgentSearchConfig`, where each value carries its
measurement) are what this arm does that ASCENT does not. Every flag defaults
to off; with all of them off the policy is the transcription, action for
action, and `tests/unit/test_ascentnav_*.py` pin that. `tests/unit/
test_objsearch_*.py` pin what each flag adds. Why each exists, and what was
measured and NOT built, is `docs/history/AB_RESULTS.md` S83 onward.

| flag | what it adds | where |
|---|---|---|
| `trace` | logging only: storey identity, storey and live-frontier counts, stair flags, the object map's goal, D-FINE's target boxes below the 0.8 ingest bar | `search/trace.py`, `agent.py:_search_trace` |
| `storey_belief` (+ `storey_switch_ratio`, `storey_detect_prob`, `storey_prior_weight`, `storey_frontier_area_m2`, `storey_min_steps_left`) | each exploring step, a belief over the storeys ASCENT has mapped — prior × (1 − detect × searched share) — may ask for the staircase before the storey runs out of frontiers; the climb itself is ASCENT's (`_navigate_stair_if_unexplored_floor`) | `search/storey.py`, `agent.py:_storey_request` |
| `verify_climb` (+ `climb_same_level_m`) | an "arrival" (off the stair map after the centroid) on the level the attempt began is a failed attempt: no storey switch, no new map | `stairs.py:attempt_over`, `process_climb_state` |
| `climb_fail_cap`, `climb_ban_steps` | that many attempts on one storey in one direction that went nowhere stop the belief, the LLM's GO_UP/GO_DOWN and passive entry from sending the agent that way for a while. The ban expires and a storey with no frontier left ignores it; the staircase is NOT burnt, because a burn is permanent and a storey with no stairs and no frontiers STOPs | `stairs.py:banned` |
| `stair_stop_guard` | a point-goal STOP on the walk to a staircase becomes FORWARD instead of ending the episode (F8), once per storey and direction | `agent.py:_navigate_stair_if_unexplored_floor` |
| `climb_relay` (+ `climb_storey_m`) | a storey is climbed through its landings as one intent: a climb that stalls `climb_same_level_m` from where it began has arrived (the reference re-initialises the storey it left, F5), and from a landing the next flight the same way comes before any frontier and the way back is refused until `climb_storey_m` is covered. Measured and **not adopted** (S85: three rounds, no difference from `objsearch_storey` beyond noise) | `stairs.py:_relay_arrival`, `stall_arrival`; `agent.py:_explore` |

Two things the additions read that the reference does not: the pose's height
(`_base_y`, for `verify_climb`), and the floor prior under the target's HM3D
name (`floor_table`; the transcription, like the reference, asks under the
COCO name and reads zeros for sofa, tv_monitor and plant).

**The default since S86.** `configs/agent/s86.yaml`, which `configs/config.yaml`
selects, turns on `storey_belief`, `verify_climb` (cap 2) and
`stair_stop_guard`; `+experiment=ascentnav` (`agent: s71`) is the arm without
them, and every preset measured before S86 keeps `s71`. The dataclass defaults
stay off, so the transcription tests are untouched.

`configs/experiment/objsearch_storey.yaml` turns on the storey belief, the
climb verification with a cap of 2, and the guard; `objsearch_relay.yaml` adds
`climb_relay`. Compare a run against the
baseline with `python scripts/analyze_objectnav_search.py <baseline> <run>`:
it classes every episode by where it failed and pairs the two runs.

## Measuring it

```
python scripts/run_eval.py +experiment=ascentnav eval=scenes20_ep0to4 \
    eval.behaviour_log=true eval.save_viz=false output_dir=outputs/<run>
python scripts/compare_ascent_osg.py data/reference/ascent_behaviour_100 outputs/<run>
```

The ASCENT trace that command reads is not in this checkout (`docs/SETUP.md` §6): it was
never committed and cannot be regenerated from the unpatched submodule.

The comparison's last block scores both traces against the dataset's object
positions: SAW episodes, never-saw → STOP / timeout, climb share on same-floor
episodes, the earliest STOP, commits and P(success | committed).
