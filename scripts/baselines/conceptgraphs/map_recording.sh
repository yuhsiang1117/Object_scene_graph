#!/bin/bash
# ConceptGraphs' mapper (cfslam_pipeline_batch.py, unchanged) on export_recording.py's scenes.
#
#   [RES=640x480] bash scripts/baselines/conceptgraphs/map_recording.sh <export root> <gsa_variant> <scene>...
#
# gsa_variant `osg`  (osg_detections.py: the recording's own detections) runs with the README's
#                    ConceptGraphs-Detect parameters; on these detections its two 2D filters
#                    (confidence 0.25, box area 0.5) remove nothing, so the mapper sees exactly
#                    what the object layer saw;
# gsa_variant `none` (run_detections.py: SAM "segment all") runs with the README's class-agnostic
#                    ConceptGraphs parameters.
# The export already kept every 5th keyframe, so stride=1 here.
#
# Scenes run ONE AT A TIME. The mapper keeps every merged detection's full-resolution mask on its
# object, so its memory grows with the frame count: at 1280x960 with SAM's masks it grew ~3.5 GB
# per 100 frames on 00808 (1527 frames) and then jumped past the machine's 124 GB near the end,
# and running scenes in parallel got them OOM-killed. RES=WxH sets the mapper's image_width /
# image_height (its own option; detections are resized by its resize_gobs), which cuts that
# memory 4x at 640x480; such a run's save_suffix and log carry the resolution.
set -uo pipefail
ROOT=$(realpath "$1"); VARIANT=$2; shift 2
CG=${CG_ROOT:-/workspace/relative_work/baselines/concept-graphs}/conceptgraph
PY=${CG_PY:-/workspace/.conda-envs/conceptgraph/bin/python}
unset PYTORCH_CUDA_ALLOC_CONF   # torch 2.0.1 does not know expandable_segments
cd "$CG"
case $VARIANT in
  osg)  SUFFIX=overlap_maskconf0.25_simsum1.2_dbscan.1
        ARGS="spatial_sim_type=overlap mask_conf_threshold=0.25 match_method=sim_sum sim_threshold=1.2
              dbscan_eps=0.1 gsa_variant=osg skip_bg=False max_bbox_area_ratio=0.5" ;;
  none) SUFFIX=overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub
        ARGS="spatial_sim_type=overlap mask_conf_threshold=0.95 match_method=sim_sum sim_threshold=1.2
              dbscan_eps=0.1 gsa_variant=none class_agnostic=True skip_bg=True max_bbox_area_ratio=0.5
              merge_interval=20 merge_visual_sim_thresh=0.8 merge_text_sim_thresh=0.8" ;;
  *) echo "variant osg|none"; exit 1 ;;
esac
TAG=""
if [ -n "${RES:-}" ]; then
  W=${RES%x*}; H=${RES#*x}
  ARGS="$ARGS image_width=$W image_height=$H"; TAG="_${H}p"; SUFFIX="${SUFFIX}${TAG}"
fi
fail=0
for s in "$@"; do
  LOG=$ROOT/$s/map_${VARIANT}${TAG}.log
  $PY slam/cfslam_pipeline_batch.py dataset_root=$ROOT dataset_config=$ROOT/dataconfig.yaml stride=1 \
      scene_id=$s $ARGS save_suffix=$SUFFIX hydra.run.dir=$ROOT/$s/.hydra_map_${VARIANT}${TAG} > "$LOG" 2>&1
  rc=$?
  echo "$s: exit $rc, $(grep -c "Saved full point cloud" "$LOG") saves -> pcd_saves/full_pcd_${VARIANT}_${SUFFIX}[_post].pkl.gz"
  [ $rc -ne 0 ] && fail=1
done
exit $fail
