# Multi-floor/dynamic source inventory

This inventory compares dynamic source `cc2270424cdcf205ffeccb8e6e8aec63e628ff63`
with the unrelated multi-floor snapshot
`8fb27d2f861a3602608ca26d3ec56cca6a3cc958`. The latter is a behavior source,
not a file-level merge. The final `ours` merge records ancestry only after this
port is verified.

## Upstream additions

Every path reported as `A` by
`git diff --name-status cc22704 upstream/multi-floor` is classified below.

### Ported as dedicated components or provenance assets

- Evaluation configs: `configs/eval/{dev50,dev50_mf,hm3d_val_v1_full,scenes20_crossfloor,scenes20_ep0to4,viz10}.yaml`.
- Experiment configs: `configs/experiment/{ascent_aligned,ascent_matched,ascent_policy,ascent_sensor,ascent_sensor_all3,ascent_sensor_arrival,ascent_sensor_band,ascent_sensor_carrot,ascent_sensor_fix1,ascent_sensor_full,ascent_sensor_rednet,ascentnav,final_sensor}.yaml`.
- Exploration configs: `configs/exploration/{ascent_commit,ascent_frontier,ascent_full,ascent_llm,ascent_select,floor_llm,sg_rooms,value}.yaml`.
- Hardware profiles: `configs/profile/{laptop,report}.yaml`.
- Split provenance: `data/splits/{dev50,dev50_mf}.json` (formatting normalized; episode IDs and metadata retained).
- Results/protocol documentation: `docs/AB_RESULTS.md`.
- Tools: `scripts/{analyze_climb,calibrate_approach_recheck,make_dev_split,make_priors,measure_room_seg,measure_stair_accumulation,measure_stair_recall,smoke_clip}.py`.
- Complete alternative policy: every file under `src/ascentnav/**`, including
  its `mapping`, `vendor/frontier_exploration`, and `vendor/vlfm` trees.
  `src/ascentnav/vendor/LICENSES.md` additionally retains upstream attribution
  and MIT notices.
- Imported OSG components:
  `src/osg/agent/ascent_agent.py`,
  `src/osg/exploration/{ascent_ranker,ascent_selector,floor_planner,frontier_semantics,knowledge_prior,score_cache}.py`,
  `src/osg/mapping/{contour_frontier,value_map}.py`,
  `src/osg/perception/{image_text,room_classifier,stair_seg}.py`,
  every file under `src/osg/perception/rednet/**`,
  `src/osg/planning/{escape,pointnav_driver}.py`, and every file under
  `src/osg/planning/pointnav/**`.
- Upstream unit contracts:
  `tests/unit/test_{approach_recheck,ascent_agent,ascent_ranker,ascent_selector,ascentnav_geometry,ascentnav_stairs,climb_carrot,contour_frontier,down_look,escape,eval_uid,floor_planner,floor_reject,floor_scoping,fp_retraction,frontier_semantics,knowledge_prior,nav_agent_climb,navigation_mode,pointnav_driver,room_classifier,terminal_stop,value_map,verify_cooldown}.py`.

### Superseded by the typed architecture

- `src/osg/core/config.py`: upstream's monolithic dataclasses are represented by
  the typed modules under `src/osg/core/config/`. Every imported field is in
  its domain dataclass; Hydra still registers one `OSGConfig` tree.

### Merged into shared implementations

The imported OSG additions above retain their behavior but use shared factories,
typed config, the canonical `FloorStack`, and the benchmark-neutral runner.
They are therefore not byte copies. The `ascentnav` package intentionally
remains an alternative policy and does not replace the dynamic OSG model.

## Paths modified in both snapshots

The following upstream `M` paths were merged semantically:

- Configuration/docs: `.gitignore`, `README.md`,
  `configs/experiment/{full_v1_navmesh,matched_navmesh}.yaml`,
  `configs/exploration/{llm_text,nearest,sweep}.yaml`, `configs/llm/nim.yaml`,
  and `configs/verification/nim.yaml`.
- Tools: `scripts/{compare_runs,download_weights}.py` are merged:
  `compare_runs.py` accepts both the dynamic `NAME=path` funnel mode and paired
  source/treatment floor analysis; `download_weights.py` stages YOLOE/
  MobileCLIP, CLIP, PointNav and RedNet. Upstream edits to
  `scripts/{analyze_localization,analyze_refine_accuracy,analyze_stages,analyze_trackloc,dump_keyframe_seg,export_temporal_3d,run_episode_report,viz_frontier_runs}.py`
  are classified **superseded by the dynamic-source versions plus the imported
  `analyze_climb.py` and extended episode/summary schema**; their existing
  dynamic inputs and outputs are retained.
- Runtime: `src/osg/agent/nav_agent.py`, `src/osg/core/types.py`,
  `src/osg/eval/{metrics,runner,visualize}.py`,
  `src/osg/exploration/selector.py`,
  `src/osg/graph/{scene_graph,serialize}.py`,
  `src/osg/llm/{client,prompts}.py`,
  `src/osg/mapping/{costmap,floor_stack,frontier,room_seg,stairs}.py`,
  `src/osg/objects/{association,ellipsoid,linking,object_layer}.py`,
  `src/osg/perception/detector.py`, `src/osg/planning/planner.py`,
  `src/osg/sim/habitat_env.py`, and
  `src/osg/verification/{verifier,viewpoint}.py`.
- Shared tests: `tests/unit/test_{costmap,floor_stack,nav_agent,room_seg,selector,stairs,viewpoint}.py`.

Docker files (`docker/.env.example`, `docker/compose.yaml`, and
`docker/entrypoint.sh`) are classified as **superseded**: the dynamic source's
collector mounts, host UID/GID handling and secret boundaries are retained;
the teammate components need no additional container mutation.

## Files absent from the upstream snapshot

Every `D` entry in the same comparison is classified **retained from dynamic
source**. This includes all authored-YCB presets and tools, dynamic/presence
modules, the split typed-config package, `pipeline/components.py`, persistence,
evaluation records, and their tests/docs. None was removed to imitate the
unrelated upstream tree. Multi-floor behavior was folded into those modules:

- `agent/{approach,candidate,floor_policy,state}.py`
- `core/config/**` and `core/labels.py`
- `eval/{attempts,debug_video,episode,floors,gt_dump,instruments,prior_map,record}.py`
- `exploration/{search_belief,strategy}.py`
- `graph/{containers,map_store,priors}.py`
- `mapping/{floors,portals}.py`
- `objects/presence.py`, `perception/{foveate,vocabulary}.py`
- `pipeline/{__init__,beliefs,components}.py`
- `sim/{ycb_env,ycb_layouts}.py`, `verification/absence.py`
- all dynamic authored scripts/configs/docs/tests present at `cc22704`.

The only deliberately unported upstream deletion is
`.claude/settings.local.json`, a machine-local settings file. Generated data,
external checkouts, checkpoints, maps and benchmark outputs remain ignored.
