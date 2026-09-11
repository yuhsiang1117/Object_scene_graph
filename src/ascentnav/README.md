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
  `stair_disable_burns_map: true` runs the branch the code was written for.
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

## Measuring it

```
python scripts/run_eval.py +experiment=ascentnav eval=scenes20_ep0to4 \
    eval.behaviour_log=true eval.save_viz=false output_dir=outputs/<run>
python scripts/compare_ascent_osg.py relative_work/ascent/debug/behaviour_100 outputs/<run>
```

The comparison's last block scores both traces against the dataset's object
positions: SAW episodes, never-saw → STOP / timeout, climb share on same-floor
episodes, the earliest STOP, commits and P(success | committed).
