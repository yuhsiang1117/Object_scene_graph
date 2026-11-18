#!/usr/bin/env bash
# Pass 1: rebuild the prior maps with the CURRENT vocabulary.
#
# The prior map's tracks carry the query string they were detected under, so a
# vocabulary change invalidates it: a map full of "tomato soup can" tracks holds
# nothing an agent asking for "cylindrical can" can use. save_map overwrites
# rather than merges, so targets accumulate by feeding each episode the map the
# previous one wrote -- the first with map_out alone, the rest with both.
#
# LAYOUT_ROOT picks the world the map is built in, and it is load-bearing.
# `outputs/substituted_layouts` (the default, and what built maps_v5) is the
# released static layout with 037_scissors swapped for 021_bleach_cleanser --
# right for the authored YCB benchmark the swap was made for, and WRONG for
# DualMap's released benchmark, whose episodes inject the real scissors. A map
# built there holds a bleach cleanser at the scissors position and no scissors
# at all. Use `outputs/collector_layouts` for the released benchmark.
set -u
OUT="${1:-outputs/maps_v4}"
: "${LAYOUT_ROOT:=outputs/substituted_layouts}"
set -a; . "$(git rev-parse --show-toplevel)/.env"; set +a
cd /workspace

COMMON=(
  +experiment=ycb_authored_nav
  "ycb.layout_root=$LAYOUT_ROOT"
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

# One mapping episode per object the layout actually places, named as OUR
# detector is asked for it. The substituted worlds hunt a bleach bottle where
# the released ones hunt the scissors; the scissors episode will not find it
# (recall 0.00 beyond a metre) and that is the point -- the map then holds what
# the detector can really see in the world the benchmark scores.
# A scene's episodes MUST run in order -- each one starts from the map the
# previous wrote -- but the scenes are independent, so SCENES exists to run one
# per process and cut the wall clock by three. Each needs about 1.3 GB of GPU.
: "${SCENES:=00829-QaLdnwvtxbs 00848-ziup5kvtCCR 00880-Nfvxx8J5NCo}"

targets_for () {
  if [ "$LAYOUT_ROOT" = "outputs/collector_layouts" ]; then
    case "$1" in
      00829-*) echo 'bowl|tin can|cracker box|red plate|blue plastic pitcher|scissors' ;;
      00848-*) echo 'red plate|cracker box|banana|blue plastic pitcher|tin can|mug|scissors' ;;
      00880-*) echo 'bowl|tin can|cracker box|red plate|blue plastic pitcher|scissors' ;;
    esac
  else
    case "$1" in
      00829-*) echo 'bowl|tin can|cracker box|red plate|blue plastic pitcher|bleach bottle' ;;
      00848-*) echo 'bleach bottle|red plate|cracker box|banana|blue plastic pitcher' ;;
      00880-*) echo 'bowl|bleach bottle|red plate|blue plastic pitcher|tin can' ;;
    esac
  fi
}

for scene in $SCENES; do
  IFS='|' read -r -a targets <<< "$(targets_for "$scene")"
  map_scene "$scene" "${targets[@]}"
done
echo "########## MAPS DONE ($SCENES) $(date +%H:%M:%S)"
