# Static-map baselines on DualMap's released benchmark

Other open-vocabulary scene-graph methods, run on the same 186 trials as the DualMap rows of
Table I. Each method maps the release's **static** sequence (`<release>/<scene>/{rgb,depth,pose.txt}`,
the frames DualMap's released global map was built from) offline, answers each query with a
ranking of its objects, and is driven by one shared navigator. Only the map and the ranking differ
between rows.

| step | what | where |
|---|---|---|
| map | the method's own pipeline on the static sequence, ground-truth poses | per method, below |
| rank | every object scored against each query by the method's own text encoder | `<method>/…` → `rankings.json` |
| navigate | DualMap's `calculate_global_path` on its released global map + a last leg to the free cell beside the candidate; the released collector follower | `static_map_nav.py` |
| score | `osg.eval.dualmap_release`: 1 m horizontal to any instance, 3 attempts, same start poses (collector seed) | `static_map_nav.py` → `trials/<id>/result.json` |
| table | SR by scene and condition, mean over seeds | `report_table.py` |

`rankings.json` is the one interface between a method and the navigator:
`{"method", "scene", "map", "encoder", "objects": [{"uid", "center", "dimensions"}], "rankings": {query: [[uid, score], …]}}`,
centres in the release's world frame (`pose.txt`'s, z up).

## What the comparison is, and is not

- **The map is never updated.** In the dynamic splits the agent goes where the static map says
  the object was. These rows are labelled "(static map)"; they measure how far a correct static
  map gets, not a method's own change handling (Khronos and Clio have some; it is not exercised).
- **One navigator for all.** A static map has no local planner, so the last leg is a straight
  segment to the free cell DualMap's own `snap_to_free_space_directional` picks beside the
  candidate, where DualMap's local planner would take over. The follower integrates velocity
  without collisions, exactly as in the DualMap rows.
- **Attempts.** Up to three, each on the best-ranked candidate more than 1 m (horizontal) from
  every candidate already tried: several of a map's objects are often fragments of one thing.
  A candidate with no global path uses up its attempt.
- **In-anchor does discriminate.** Standing at the static position would win most in-anchor
  trials, but no agent stands there: every successful stop, DualMap's and the baselines', is
  ~0.8 m from the object (median 0.76-0.84 m on 00848), the free-space standoff of the shared
  planner. A 0.2-1.6 m in-anchor move then takes a static map's stop past 1 m; DualMap wins
  those by re-detecting the object on arrival.

## Khronos (`khronos/`)

ROS 2 Jazzy (RoboStack env `/workspace/.conda-envs/khronos`), Khronos `63faadd` with its
dependencies pinned in `relative_work/baselines/khronos_ws` (gitignored). Open-set branch:
semantic_inference FastSAM + CLIP ViT-L/14.

```bash
bash scripts/baselines/khronos/map_sequence.sh 00829-QaLdnwvtxbs outputs/baselines/khronos/map/00829-QaLdnwvtxbs --min-period 0.6
# then, in the Khronos env with the repo's src on PYTHONPATH:
python scripts/baselines/khronos/query_map.py --map outputs/baselines/khronos/map/<scene>/khronos \
    --scene <scene> --out outputs/baselines/khronos/map/<scene>/rankings.json
```

Deviations from upstream, each forced by this setup and stated in the config headers:
- mapper config derived from the maintained `uHumans2.yaml` (the open-set configs are stale);
- `min_object_volume` 0.005 → 0.0002 m³ (every queried YCB object is under 5 L);
- the background prompts' embeddings arrive on a topic (`background_embeddings.py`), not the
  segmenter's service, and the map is saved on SIGINT, not by `finish_mapping_and_save`:
  services hosted by these nodes never become reachable from another process here;
- frames are paced on the segmenter's answer, so none is dropped; `map_sequence.sh` checks that
  the active window integrated every published frame.

## Clio (`clio/`)

Clio `f912b7c`, **batch** variant (`clio_batch.object_cluster.cluster_3d`, unchanged) on fine
primitives from the Khronos front end above (`clio/config/mapper_dualmap_clio_primitives.yaml`:
the settings of Clio's `realsense_fine.yaml`). Tasks are the scene's queries. Clio's own code is
ROS 1; its authors ask that paper comparisons use an archived snapshot, whose link
(README, "Evaluation") was dead on 2026-10-31, so this is the released code, and the README's
note that the current primitives score below the paper applies.

```bash
KHRONOS_MAPPER_CONFIG=scripts/baselines/clio/config/mapper_dualmap_clio_primitives.yaml \
  bash scripts/baselines/khronos/map_sequence.sh <scene> outputs/baselines/clio/map/<scene> --scale 0.53333 --min-period 0.6
# in the Khronos env, from relative_work/baselines/clio:
python scripts/baselines/clio/cluster_map.py --map outputs/baselines/clio/map/<scene>/khronos --scene <scene> \
    --out outputs/baselines/clio/map/<scene>/clio
```

Beyond the fine-primitive settings:
- frames at 640x363 (`frame_publisher.py --scale 0.53333`), the resolution Clio's pixel
  thresholds are for; at 1200x680 the primitives' memory grew 35 MB/frame;
- 12 object-extraction workers (default 2): with 2 the extraction fell behind and the waiting
  objects held their frames (memory, and fewer, larger primitives); throughput only;
- Khronos's backend change detection off: Clio runs its own backend. On 00829 and 00848, mapped
  both ways, every cell of the table is identical (runs with it on: `nav_cd/`; their maps, 122 GB,
  were deleted on 2026-11-04 once the comparison was settled).

## HOV-SG (`hovsg/`)

HOV-SG `d6e65a5`, `create_graph.py` on the release laid out as its HM3DSem walk
(`export_sequence.py`: rgb/depth linked, poses to Habitat's y-up world and OpenGL camera; the
release's 1200x680, fx 600 is the 90-degree HFOV HOV-SG assumes), SAM ViT-H + CLIP ViT-H-14
(laion2b), queried by CLIP over every object (`query_graph.py`; the hierarchical LLM parse has
nothing to do on object-only queries). Run through `run_headless.py`, which leaves HOV-SG's source
as it is and:
- swaps its hard-coded TkAgg backend for Agg (no display);
- replaces the sequential mask merge with `fast_merge.py`'s: the same IoU screen vectorised, and
  FAISS overlaps / DBSCAN denoising of unchanged inputs taken from a content cache; checked
  point for point against HOV-SG's own `seq_merge` on 40 and 80 real frames of 00829;
- gives the floor-pose assignment 10 cm of slack on its 1.5 m sensor height (the release's
  reconstructed floor sits 6 cm above the navmesh in 00829, so no pose fell on any floor).

`pipeline.skip_frames=25`, not 10: HOV-SG's own walk of 00829 has 1804 frames (181 at its skip of
10); the release's has 4530, and 25 gives 182. At 10, the sequential merge alone was on course for
about a day per scene.

## ConceptGraphs (`conceptgraphs/`)

ConceptGraphs `93277a0`, the class-agnostic variant (SAM "segment all", `--class_set none`),
`generate_gsa_results.py` then `cfslam_pipeline_batch.py` with its README's parameters, stride 5,
on the release laid out as a Replica scene (`export_sequence.py`; images linked, the release's
poses as `traj.txt`; `dualmap.yaml` carries the release's intrinsics and millimetre depth).
Queried as its own `visualize_cfslam_results.py` does (`query_map.py`). Environment: Python 3.10,
torch 2.0.1 + CUDA 11.8, Grounded-SAM at a4d76a2, as its README.

### On a recorded pass 1 (map quality, AB_RESULTS S80)

The same ConceptGraphs scored against the object layer's instance ground truth on the frames a
pass 1 recorded (`eval.record_object_frames_dir`; labelled by `scripts/label_object_frames.py`):

```bash
# nav env: re-render RGB + semantic at every 5th keyframe, write a Replica scene per scene
python scripts/baselines/conceptgraphs/export_recording.py +experiment=mf5_mapping_both \
    +export.rec=outputs/objrec_v2 +export.out=outputs/baselines/cg_rec +export.stride=5
# front end A: the recording's own detections (then the clip pass below with --detections gsa_detections_osg)
python scripts/baselines/conceptgraphs/osg_detections.py --rec outputs/objrec_v2 \
    --cg outputs/baselines/cg_rec --scene <scene>
# front end B, ConceptGraphs env, from its conceptgraph/ dir, GSA_PATH set, PYTORCH_CUDA_ALLOC_CONF unset:
python /workspace/scripts/baselines/conceptgraphs/run_detections.py --pass masks \
    --dataset_root <root> --dataset_config <root>/dataconfig.yaml --scene_id <scene> --class_set none --stride 1
python /workspace/scripts/baselines/conceptgraphs/run_detections.py --pass clip --dataset_root <root> --scene_id <scene>
# map (one scene at a time; RES keeps memory in bounds on long scenes), then score
bash scripts/baselines/conceptgraphs/map_recording.sh outputs/baselines/cg_rec osg <scenes...>
RES=640x480 bash scripts/baselines/conceptgraphs/map_recording.sh outputs/baselines/cg_rec none <scenes...>
python scripts/baselines/conceptgraphs/score_recording.py --rec outputs/objrec_v2 --cg outputs/baselines/cg_rec \
    --ours I6r1_s5=outputs/objreplay/I6r1_s5 --cg-map 'CGosg=pcd_saves/full_pcd_osg_*_post.pkl.gz' \
    --cg-map 'CG480=pcd_saves/full_pcd_none_*_480p_post.pkl.gz'
```

`run_detections.py` splits `generate_gsa_results.py` into a SAM pass and a CLIP pass so that one
large model is on a shared GPU at a time. Its outputs are the script's own. The object-layer arm
is `scripts/replay_object_layer.py` with `+replay.stride=5`, so both see the same frames.

## DualMap rerun

`scripts/run_dualmap_released_native.py` unchanged, seeds 12/13/14, in `dualmap-venv` (below).
The runner grows across trials (57 GB in one process), so it runs one process per (scene,
condition) group. DualMap itself crashes on a few trials (`ValueError: zero-size array to
reduction operation minimum` in its detector's `overlap_check`), as it did on 00848 kettle in the
October rerun; these are retried once and otherwise reported as missing.

## Navigation and table

```bash
PYTHONPATH=src:/habitat-data-collector /workspace/.conda-envs/dualmap-venv/bin/python \
    scripts/baselines/static_map_nav.py --rankings 'outputs/baselines/khronos/map/{scene}/rankings.json' \
    --output outputs/baselines/khronos/nav/seed12 --seed 12
python scripts/baselines/report_table.py --run 'Khronos=outputs/baselines/khronos/nav/seed*' --static-map Khronos
```

Retrieval quality, before any navigation (top-1 / any of top-3 within 1 m of an instance, over
the 79 static queries); MD-SG's prior map is exported to the same interface first:

```bash
python scripts/baselines/osg_rankings.py --maps outputs/maps_released_upload/maps_released \
    --scene <scene> --out outputs/baselines/mdsg/original/<scene>.json
python scripts/baselines/rank_quality.py --rankings 'Khronos=outputs/baselines/khronos/map/{scene}/rankings.json' \
    --rankings 'MD-SG=outputs/baselines/mdsg/original/{scene}.json'
```

`dualmap-venv` is the habitat env plus DualMap's own dependencies (open3d, faiss, its bundled
mobileclip, …), layered with `--system-site-packages` so neither shared env is modified.
`hovsg-venv` is built the same way; Khronos/Clio and ConceptGraphs have their own conda envs
(`/workspace/.conda-envs/{khronos,conceptgraph}`).
