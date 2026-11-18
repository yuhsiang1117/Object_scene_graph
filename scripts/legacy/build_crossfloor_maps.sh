#!/usr/bin/env bash
# Build prior maps for the cross-floor scenes that do not have one.
#
# The cross-floor sample is limited by prior maps, not by the benchmark: of the
# 19 relocations that change storey, only 7 are in scenes with a map under
# outputs/maps_15. These five scenes hold the other 9 -- 00878 has 3, 00869 2,
# 00873 2, 00824 1, 00871 1.
#
# Pass 1 of the two-pass protocol: explore the STATIC layout with the map
# directory as both map_out and map_in, so every static episode starts from what
# the previous ones mapped and the snapshot accumulates. The baseline preset is
# used deliberately -- a prior map must not be an experimental variable, and
# save_map stores tracks and grids while containers are derived on load, so the
# container fix does not change what is written.
#
#   scripts/legacy/build_crossfloor_maps.sh
#   SCENES="00878-XB4GS9ShBRE" MAPS=outputs/maps_xf scripts/legacy/build_crossfloor_maps.sh
#
# Idempotent: a scene whose map json exists is skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PYTHONPATH="$(pwd)/src"

: "${SCENES:=00878-XB4GS9ShBRE 00869-MHPLjHsuG27 00873-bxsVRursffK 00824-Dd4bFSTQ8gi 00871-VBzV5z6i1WS}"
: "${PRESET:=ycb_authored_15}"
: "${MAPS:=outputs/maps_15}"
: "${MIN_FREE_MIB:=2600}"
: "${MAX_PARALLEL:=2}"

wait_for_vram() {
  for _ in $(seq 1 600); do
    local used total
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    [ $(( total - used )) -ge "$MIN_FREE_MIB" ] && return 0
    sleep 60
  done
}

build_one() {
  local scene=$1
  local dir="$MAPS/$scene"
  if [ -f "$dir/$scene.json" ]; then echo "[skip] $scene already mapped"; return 0; fi
  mkdir -p "$dir"
  echo "[map start] $scene $(date +%H:%M:%S)"
  python3 scripts/run_eval.py "+experiment=$PRESET" \
    "ycb.scenes=[$scene]" 'ycb.layout_types=[static]' \
    "ycb.map_out=$dir" "ycb.map_in=$dir" \
    "output_dir=$dir/run" "+run_tag=MAPXF" \
    > "$dir/map.log" 2>&1 \
    && echo "[map done] $scene $(date +%H:%M:%S)" \
    || echo "[map FAILED] $scene $(date +%H:%M:%S) -- see $dir/map.log"
}

for scene in $SCENES; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 30; done
  wait_for_vram
  build_one "$scene" &
  sleep 45
done
wait
echo "[all maps done] $(date +%H:%M:%S)"
