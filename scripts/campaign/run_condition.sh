#!/usr/bin/env bash
# One condition of the dynamic YCB ladder: three scenes, 96 episodes.
#   ./run_condition.sh <TAG> [extra hydra overrides...]
set -u
TAG="$1"; shift
MAPS="${MAPS:-outputs/maps_hires}"
EXTRA=("$@")
set -a; . /workspace/.env; set +a
cd /workspace

COMMON=(
  +experiment=ycb_authored_nav
  ycb.layout_root=outputs/substituted_layouts
  'ycb.layout_types=[in_anchor,cross_anchor]'
  'ycb.layout_indices=[1,2,3]'
  scene_graph.presence.enabled=true
  scene_graph.presence.recall_model_path=outputs/recall/recall_model.json
  eval.attempts=3
  verification=nim llm=nim verification.absence_only=true
  verification.vlm_model=meta/llama-3.2-11b-vision-instruct
  verification.min_obs=1 verification.min_evidence=0.2
  exploration.search_posterior=true exploration.affinity_llm=true
  llm.text_model=nvidia/nemotron-3.5-lightning-30b-a3b
  exploration.search_frontier_weight=0.3
  eval.rgb_width=1280 eval.rgb_height=960 detector.imgsz=1280
  scene_graph.min_det_bbox_px=1200 verification.min_bbox_px=800
)

run_scene () {
  local scene="$1"; local targets="$2"
  echo "### $TAG / $scene  $(date +%H:%M:%S)"
  python3 scripts/run_eval.py "${COMMON[@]}" "${EXTRA[@]}" \
    "ycb.scenes=[$scene]" "ycb.targets=[$targets]" \
    "ycb.map_in=$MAPS/$scene" \
    "+run_tag=$TAG"
  echo "### $TAG / $scene rc=$?  $(date +%H:%M:%S)"
}

run_scene 00829-QaLdnwvtxbs "bowl,cylindrical can,cracker box,red dish,blue plastic pitcher,bleach bottle"
run_scene 00848-ziup5kvtCCR "bleach bottle,red dish,cracker box,banana,blue plastic pitcher"
run_scene 00880-Nfvxx8J5NCo "bowl,bleach bottle,red dish,blue plastic pitcher,cylindrical can"
echo "########## $TAG DONE $(date +%H:%M:%S)"
