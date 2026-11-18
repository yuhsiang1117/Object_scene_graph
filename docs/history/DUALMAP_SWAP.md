# The swapped benchmark: DualMap's released trials with scissors and mug replaced

> **Never run.** The file says so itself further down: paused 2026-09-10 before any trial.
> Kept as a design note, not as a result.

> **Status (2026-09-10): paused before any trial ran.** The copy, its
> re-rendered sequences and our prior maps exist as described below; DualMap's
> global map was being rebuilt when the work was stopped, and no benchmark
> run on the copy has been completed for either system.

A sibling copy of DualMap's released HM3D benchmark in which the two assets
neither system's detector can see (`docs/history/ASSET_SUBSTITUTION_EVIDENCE.md`) are
replaced by two objects from our 15-scene authored set. The released data
under `data/dualmap/HM3D_collect` is never edited; the copy lives beside it.

| released | replacement | why this one |
|---|---|---|
| `037_scissors` (query "scissors") | `072-a_toy_airplane` (query "toy airplane") | render test in the scissors' own places: ours 0.54-0.83 and DualMap's detector 0.53-0.96 at 0.7-3 m; 5/5 close-centred keyframes named on the 15-scene runs |
| `025_mug` (query "mug") | `006_mustard_bottle` (query "mustard bottle") | the only other 15-scene object; ours 0.50-0.93 at 0.6-3 m under the name "yellow bottle" (0.31-0.39 as "mustard bottle"); DualMap's detector calls it "soap bottle" / "bottle of detergent" at 0.78-0.87 and its CLIP matcher answers the query by feature, not label |

The third 15-scene candidate, `002_master_chef_can`, failed the same test
(ours 0.00 beyond 0.6 m, DualMap 0.70 at 0.6 m and 0.00 beyond) and is not
used. Renders and scores: `outputs/asset_evidence_cand{A,B}`,
`outputs/asset_evidence_swap` (`views.json`).

## What is in the copy

`/datasets/habitat-data-collector/data/dualmap_swap/HM3D_collect/` (made by
`scripts/make_dualmap_swap.py --swap 037_scissors=072-a_toy_airplane 025_mug=006_mustard_bottle`):

- Every `static_scene_config.json` and `dynamic_scene_config/*/*.json` with
  the two handles replaced in `id_handle_mapping`. Positions are the released
  ones; a swapped object stands upright (identity rotation, the YCB configs
  declare `up: [0,0,1]`) with its mesh bottom on the old object's supporting
  surface, so the bottle rests on the bed where the scissors lay. 34
  placements over 21 layout files; `swap.json` lists each with old and new
  heights.
- `rgb/`, `depth/`, `pose.txt`: the released mapping trajectory re-rendered in
  the swapped static scene by `scripts/render_dualmap_sequence.py`
  (habitat-sim, the shipped intrinsics, the collector's ROS world frame and
  an OpenCV camera; on the original scene the renders match the shipped
  frames to 0.0/255 in RGB and 1 mm in depth).
- `global_map/hm3d`: DualMap's static global map rebuilt from that sequence
  by its own dataset runner (`scripts/legacy/build_dualmap_swap_map.py`, its models
  and keyframe selection, its class list plus the two new names in
  `configs/dualmap_swap/hm3d300_classes_ycb_swap.txt`).
- Everything else (`class_bbox.json`, `class_num.json`, `data.zip`, the
  odometry) is a symlink to the release.

## Running the two systems on it

Ours: `OSG_DUALMAP_RELEASE_ROOT` selects the copy; `osg.eval.dualmap_release`
reads its `swap.json`, renames the queries in the protocol (trial ids change
with them, e.g. `..._0116__toy_airplane`), and aliases the released start
poses onto the new ids. Prior maps: `outputs/maps_swap/<scene>` = `maps_v5`
plus one static mapping episode per new object on the swapped scene.

    export OSG_DUALMAP_RELEASE_ROOT=/datasets/habitat-data-collector/data/dualmap_swap/HM3D_collect
    MAP_ROOT=outputs/maps_swap TRIAL_SET=data/splits/dualmap_swap_all.json SHARDS_PER_SCENE=2 \
      ARMS=flat_anchor_v2_island_close OUT_ROOT=outputs/osg_swap_full scripts/run_close_look_ab.sh

DualMap: the native harness (`scripts/run_dualmap_released_native.py`) reads
the same protocol, so the same environment variable points it at the copy,
and `DUALMAP_CLASS_LIST=configs/dualmap_swap/hm3d300_classes_ycb_swap.txt`
gives its detector the two names. Its map is preloaded from the copy's
`global_map/hm3d`.

## Labels

Our detector is asked for "yellow bottle" when the query is "mustard
bottle" (`YCB_TARGET_LABELS`), the same kind of system-side choice as
"blue plastic pitcher" for "pitcher". This also changes the name used on the
15-scene authored benchmark, whose earlier runs used "mustard bottle".
