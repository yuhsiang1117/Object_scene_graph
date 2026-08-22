#!/usr/bin/env bash
# Pass 1: rebuild the prior maps with the CURRENT vocabulary.
#
# The prior map's tracks carry the query string they were detected under, so a
# vocabulary change invalidates it: a map full of "tomato soup can" tracks holds
# nothing an agent asking for "cylindrical can" can use. save_map overwrites
# rather than merges, so targets accumulate by feeding each episode the map the
# previous one wrote -- the first with map_out alone, the rest with both.
set -u
OUT="${1:-outputs/maps_v4}"
set -a; . /workspace/.env; set +a
cd /workspace

COMMON=(
  +experiment=ycb_authored_nav
  ycb.layout_root=outputs/substituted_layouts
  'ycb.layout_types=[static]'
  scene_graph.presence.enabled=true
  scene_graph.presence.recall_model_path=outputs/recall/recall_model.json
  exploration.search_posterior=true exploration.affinity_llm=false
  eval.rgb_width=1280 eval.rgb_height=960 detector.imgsz=1280
  scene_graph.min_det_bbox_px=1200 verification.min_bbox_px=800
)

map_scene () {
  local scene="$1"; shift
  local first=1
  for target in "$@"; do
    echo "=== map $scene / $target  $(date +%H:%M:%S) ==="
    local extra=()
    [ $first -eq 0 ] && extra+=("ycb.map_in=$OUT/$scene")
    first=0
    python3 scripts/run_eval.py "${COMMON[@]}" "${extra[@]}" \
      "ycb.scenes=[$scene]" "ycb.targets=[$target]" \
      "ycb.map_out=$OUT/$scene" +run_tag=MAP
  done
}

map_scene 00829-QaLdnwvtxbs bowl "cylindrical can" "cracker box" "red dish" "blue plastic pitcher" "bleach bottle"
map_scene 00848-ziup5kvtCCR "bleach bottle" "red dish" "cracker box" banana "blue plastic pitcher"
map_scene 00880-Nfvxx8J5NCo bowl "bleach bottle" "red dish" "blue plastic pitcher" "cylindrical can"
echo "########## MAPS DONE $(date +%H:%M:%S)"
