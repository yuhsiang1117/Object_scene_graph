#!/usr/bin/env bash
# Paired A/B for the fused pipeline on the multi-storey authored scenes.
#
# Both arms run the SAME code and therefore the same manifests, so the pairing
# is exact: the only difference is the two ordering flags the fused preset sets.
# The static prior maps under $MAPS are reused as-is -- the static manifest's
# cache key does not depend on the relocation source, so the instrument fix does
# not invalidate them and no map pass is needed.
#
#   scripts/legacy/run_fusion_ab.sh                       # default scenes, both arms
#   SCENES="00808-y9hTuugGdiq" scripts/legacy/run_fusion_ab.sh
#   ARMS="fused" OUT=outputs/ab2 scripts/legacy/run_fusion_ab.sh
#
# MAX_PARALLEL is sized to the free GPU, not the CPU: another campaign owns most
# of this card. A multi-storey process measures 1.6-1.9 GB.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PYTHONPATH="$(pwd)/src"

: "${SCENES:=00808-y9hTuugGdiq 00814-p53SfW6mjZe 00800-TEEsavR23oF}"
: "${ARMS:=base fused}"
: "${MAPS:=outputs/maps_15}"
: "${OUT:=outputs/fusion_ab}"
: "${MAX_PARALLEL:=3}"
: "${LAYOUTS:=[in_anchor,cross_anchor]}"

preset_for() { [ "$1" = "base" ] && echo ycb_authored_15 || echo ycb_authored_15_fused; }

run_one() {
  local scene=$1 arm=$2
  local preset; preset=$(preset_for "$arm")
  local dir="$OUT/$arm/$scene"
  if [ -f "$dir/episodes.jsonl" ] && [ -s "$dir/episodes.jsonl" ] \
     && grep -q '"success"' "$dir/episodes.jsonl" 2>/dev/null; then
    local done_n; done_n=$(wc -l < "$dir/episodes.jsonl")
    if [ -f "$dir/.complete" ]; then echo "[skip] $arm/$scene ($done_n episodes)"; return 0; fi
  fi
  mkdir -p "$dir"
  echo "[start] $arm/$scene $(date +%H:%M:%S)"
  python scripts/run_eval.py "+experiment=$preset" \
    "ycb.scenes=[$scene]" "ycb.layout_types=$LAYOUTS" \
    "ycb.map_in=$MAPS/$scene" \
    "output_dir=$dir" "+run_tag=FUSE_${arm}" \
    > "$dir/run.log" 2>&1 \
    && { touch "$dir/.complete"; echo "[done] $arm/$scene $(date +%H:%M:%S)"; } \
    || echo "[FAILED] $arm/$scene $(date +%H:%M:%S) -- see $dir/run.log"
}

jobs_running() { jobs -rp | wc -l; }
for scene in $SCENES; do
  for arm in $ARMS; do
    while [ "$(jobs_running)" -ge "$MAX_PARALLEL" ]; do sleep 20; done
    run_one "$scene" "$arm" &
    sleep 30   # stagger the sim/detector load so two never allocate at once
  done
done
wait
echo "[all done] $(date +%H:%M:%S)"
