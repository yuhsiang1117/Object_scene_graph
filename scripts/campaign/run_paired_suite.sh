#!/usr/bin/env bash
# Two conditions over the full 96-episode benchmark, INTERLEAVED BY SCENE.
#
#   ./run_paired_suite.sh <TAG_A> <PRESET_A> <TAG_B> <PRESET_B>
#
# Interleaved because a ten-hour run is one power cut away from being half an
# arm and nothing to compare it against. Scene by scene, both conditions, so an
# interruption still leaves a complete PAIRED comparison on whatever scenes
# finished -- which is the only comparison worth having. The stored ladder runs
# are the cautionary tale: 33 different run triples match condition N's
# published aggregates and nothing on disk says which one it is.
set -u
TAG_A="$1"; PRESET_A="$2"; TAG_B="$3"; PRESET_B="$4"
MAPS="${MAPS:-outputs/maps_v5}"
set -a; . /workspace/.env; set +a
cd /workspace

run_one () {
  local tag="$1" preset="$2" scene="$3" targets="$4"
  echo "### $tag / $scene  $(date +%H:%M:%S)"
  python3 scripts/run_eval.py "+experiment=$preset" \
    ycb.layout_root=outputs/substituted_layouts \
    'ycb.layout_types=[in_anchor,cross_anchor]' 'ycb.layout_indices=[1,2,3]' \
    "ycb.scenes=[$scene]" "ycb.targets=[$targets]" \
    "ycb.map_in=$MAPS/$scene" "+run_tag=$tag" 2>&1 | tail -12
  echo "### $tag / $scene rc=$?  $(date +%H:%M:%S)"
}

# The campaign's own target lists, unchanged. Note 00848 scores `banana` and NOT
# `tin can` -- the near-miss work that motivated this arm was done on a target
# this scene does not score, and tin can appears only on 00829 and 00880.
scene_targets () {
  case "$1" in
    00829-QaLdnwvtxbs) echo "bowl,tin can,cracker box,red plate,blue plastic pitcher,bleach bottle" ;;
    00848-ziup5kvtCCR) echo "bleach bottle,red plate,cracker box,banana,blue plastic pitcher" ;;
    00880-Nfvxx8J5NCo) echo "bowl,bleach bottle,red plate,blue plastic pitcher,tin can" ;;
  esac
}

for scene in 00829-QaLdnwvtxbs 00848-ziup5kvtCCR 00880-Nfvxx8J5NCo; do
  t="$(scene_targets "$scene")"
  run_one "$TAG_A" "$PRESET_A" "$scene" "$t"
  run_one "$TAG_B" "$PRESET_B" "$scene" "$t"
done
echo "########## PAIRED SUITE DONE $(date +%H:%M:%S)"
