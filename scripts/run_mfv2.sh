#!/usr/bin/env bash
# Benchmark v2 (configs/experiment/mfv2_dataset.yaml), one arm at one seed,
# scene by scene, into outputs/mfv2/<arm>_s<seed>/<scene>.
#
#   bash scripts/run_mfv2.sh osg 42                  # all five scenes
#   bash scripts/run_mfv2.sh ascent 42 00800-TEEsavR23oF
#   SMOKE=2 bash scripts/run_mfv2.sh osg 42 00878-XB4GS9ShBRE   # 2 episodes, own dir
#
# Arms:
#   osg      `mfv2_osg` over pass 1's maps (OBS / OSG below), as the paper's
#            multi-floor arm loads them (scripts/run_mf5_pass2.sh)
#   osg_gate `mfv2_osg_gate`: the same, with the STOP gate -- its A/B partner
#   ascent   `mfv2_ascent`, no prior map; needs scripts/serve_perception.sh
#
# Pass 1 must have run (scripts/run_prior_maps.sh) and its audit passed
# (scripts/audit_map_coverage.py --graphs): a prior that does not hold the
# targets measures nothing about revising one.
#
# ASCENT runs ONE scene at a time: the model servers are not reentrant
# (docs/SETUP.md). OSG uses none of them and runs MAX_PARALLEL scenes at once.
# Idempotent: a scene with a summary.json is skipped.
set -u
cd "$(dirname "$0")/.."
set -a; [ -f docker/.env ] && . <(grep -vE '^(UID|GID)=' docker/.env); set +a
umask 002

ARM=${1:?arm: osg | ascent}
SEED=${2:?seed, e.g. 42}
shift 2
OBS=${OBS:-outputs/maps_v2_ascent}
OSG=${OSG:-outputs/maps_v2_osg}
ROOT=${ROOT:-outputs/mfv2}
SMOKE=${SMOKE:-0}
SCENES=("$@")
[ ${#SCENES[@]} -eq 0 ] && SCENES=(00800-TEEsavR23oF 00821-eF36g7L6Z9M \
  00862-LT9Jq6dN3Ea 00873-bxsVRursffK 00878-XB4GS9ShBRE)

case "$ARM" in
  osg|osg_gate)
    PRESET=mfv2_$ARM
    MAX_PARALLEL=${MAX_PARALLEL:-2}
    INPUTS=("ycb.obstacle_map_in=$OBS" ycb.obstacle_map_union=true
            ycb.seed_storeys_from_obstacle_map=true ycb.storey_seed_tol_m=0.75
            "ycb.map_in=$OSG" ycb.map_in_occupancy=false)
    for need in "$OBS" "$OSG"; do
      [ -d "$need" ] || { echo "no pass-1 map at $need -- run scripts/run_prior_maps.sh" >&2; exit 2; }
    done ;;
  ascent)
    PRESET=mfv2_ascent
    MAX_PARALLEL=1
    INPUTS=() ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac

OUT="$ROOT/${ARM}_s${SEED}"
EXTRA=()
if [ "$SMOKE" -gt 0 ]; then
  OUT="$ROOT/_smoke_${ARM}_s${SEED}"
  EXTRA=("eval.num_episodes=$SMOKE")
fi
mkdir -p "$OUT"
rm -f "$OUT/.failed"
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "arm=$ARM preset=$PRESET seed=$SEED -> $OUT (${#SCENES[@]} scene(s), parallel $MAX_PARALLEL)"

for scene in "${SCENES[@]}"; do
  if [ -f "$OUT/$scene/summary.json" ]; then
    log "=== $scene: done, skipping"
    continue
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 20; done
  log "=== $scene"
  (
    python scripts/run_eval.py "+experiment=$PRESET" "ycb.scenes=[$scene]" \
      "ycb.seed=$SEED" "${INPUTS[@]}" "${EXTRA[@]}" \
      "output_dir=$OUT/$scene" > "$OUT/$scene.log" 2>&1
    status=$?
    if [ "$status" -ne 0 ]; then
      echo "$scene exit=$status" >> "$OUT/.failed"
      echo "[$(date +%H:%M:%S)]     $scene FAILED exit=$status -- $(grep -E 'Error|Exception' "$OUT/$scene.log" | tail -1)"
    else
      echo "[$(date +%H:%M:%S)]     $scene exit=0 $(grep -o '"success_rate": [0-9.]*' "$OUT/$scene.log" | tail -1)"
    fi
  ) &
  [ "$MAX_PARALLEL" -gt 1 ] && sleep 30   # stagger the scene loads
done
wait
if [ -s "$OUT/.failed" ]; then
  log "=== FAILURES:"; sed 's/^/      /' "$OUT/.failed"; exit 1
fi
log "=== $ARM s$SEED complete"
python scripts/analyze_multifloor.py "$ARM=$OUT"
