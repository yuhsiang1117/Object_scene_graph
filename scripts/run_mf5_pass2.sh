#!/usr/bin/env bash
# Pass 2 of the cross-anchor protocol: OSG's pipeline searches the MOVED world,
# planning over the occupancy ASCENT's navigation built in pass 1.
#
#   bash scripts/serve_perception.sh --stop      # FIRST -- see below
#   bash scripts/run_mf5_pass2.sh                # all five scenes
#   MAX_PARALLEL=1 bash scripts/run_mf5_pass2.sh 00800-TEEsavR23oF
#
# RUN THIS ONLY WITH THE ASCENT SERVERS STOPPED. They hold ~10 GB idle and ~16
# once exercised, and with ollama's qwen2.5:7b on top a single Habitat process
# already took the card to 19.6 of 24.5 GB during pass 1. `nav_agent` does not
# use ASCENT's served models at all, so pass 2 gets the whole card and can run
# scenes side by side -- which is the only reason this script has a parallel
# gate and `run_prior_maps.sh` does not.
#
# `ycb.map_in_occupancy=false` is the experiment, not a detail: the scene graph
# supplies the OBJECT tracks and their presence beliefs, and ASCENT's map
# supplies ALL of the occupancy, so what the planner reads is a map it did not
# build. The two arms do not mean the same thing by "obstacle" -- ASCENT stamps
# 0.61-0.88 m, OSG 0.15-1.50 m -- which is why `obstacle_map_overwrite` stays
# false and this run's own sensor corrects the loaded cells under its own band.
#
# `obstacle_map_union=true` is NOT optional here: pass 1 wrote one snapshot per
# episode and no `<scene>.json`, so without it `load_obstacle_map` raises.
set -u
cd "$(dirname "$0")/.."
# UID/GID in docker/.env are for compose; bash's UID is readonly, so skip them.
set -a; [ -f docker/.env ] && . <(grep -vE '^(UID|GID)=' docker/.env); set +a
umask 002

OBS=${OBS:-outputs/maps_p1500_ascent}
OSG=${OSG:-outputs/maps_p1500_osg}
OUT=${OUT:-outputs/mf5_pass2_p1500}
MAX_PARALLEL=${MAX_PARALLEL:-3}
MIN_FREE_MIB=${MIN_FREE_MIB:-4000}
SCENES=("$@")
[ ${#SCENES[@]} -eq 0 ] && SCENES=(00800-TEEsavR23oF 00808-y9hTuugGdiq \
  00821-eF36g7L6Z9M 00873-bxsVRursffK 00878-XB4GS9ShBRE)
mkdir -p "$OUT"
rm -f "$OUT/.failed"
log() { echo "[$(date +%H:%M:%S)] $*"; }

wait_for_vram() {
  for _ in $(seq 1 240); do
    local used total
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    [ $(( total - used )) -ge "$MIN_FREE_MIB" ] && return 0
    sleep 30
  done
  log "    WARNING: never saw ${MIN_FREE_MIB} MiB free; starting anyway"
}

for scene in "${SCENES[@]}"; do
  if [ -f "$OUT/$scene/summary.json" ]; then
    log "=== $scene: already run, skipping"
    continue
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 20; done
  wait_for_vram
  log "=== $scene: pass 2"
  (
    python scripts/run_eval.py +experiment=mf5_osg_on_ascent_map \
      "ycb.scenes=[$scene]" \
      "ycb.obstacle_map_in=$OBS" ycb.obstacle_map_union=true \
      ycb.seed_storeys_from_obstacle_map=true ycb.storey_seed_tol_m=0.75 \
      "ycb.map_in=$OSG" ycb.map_in_occupancy=false \
      eval.debug_frames=true "output_dir=$OUT/$scene" > "$OUT/$scene.log" 2>&1
    # Capture the status FIRST. `echo "exit=$? $(cmd)"` reports the exit code of
    # the command substitution, not of python -- and that is not hypothetical:
    # it reported exit=0 for every scene of a pass-2 run that died in 8 seconds
    # on a missing module, so the pipeline logged success and moved on.
    status=$?
    sr=$(grep -o '"success_rate": [0-9.]*' "$OUT/$scene.log" | tail -1)
    if [ "$status" -ne 0 ]; then
      # Scenes run in background subshells, so a shell variable cannot carry a
      # status back to the parent; a marker file can.
      echo "$scene exit=$status" >> "$OUT/.failed"
      echo "[$(date +%H:%M:%S)]     $scene FAILED exit=$status -- $(grep -E 'Error|Exception|error:' "$OUT/$scene.log" | tail -1)"
    else
      echo "[$(date +%H:%M:%S)]     $scene exit=0 $sr"
    fi
  ) &
  sleep 30   # stagger the scene loads
done
wait
if [ -s "$OUT/.failed" ]; then
  log "=== pass 2 had FAILURES:"
  sed 's/^/      /' "$OUT/.failed"
  log "    logs: $OUT/<scene>.log"
  exit 1
fi
log "=== pass 2 complete"
# `matched_by` must read `height`, not `order`: order matching needs the
# snapshot to have mapped the same NUMBER of storeys, and most do not.
python - "$OUT" <<'PY'
import json, sys, glob
from collections import Counter
total = Counter()
for path in sorted(glob.glob(f"{sys.argv[1]}/*/episodes.jsonl")):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not rows:
        continue
    scene = rows[0].get("scene", path)
    ok = sum(1 for r in rows if r.get("success"))
    reached = sum(1 for r in rows if r.get("goal_floor_reached"))
    how = Counter(m for r in rows
                  for m in ((r.get("prior_obstacle_map") or {}).get("matched_by") or []))
    skipped = sum(len((r.get("prior_obstacle_map") or {}).get("union_skipped") or [])
                  for r in rows)
    print(f"  {scene:<24} {ok}/{len(rows)} success, {reached} reached the goal storey, "
          f"matched_by={dict(how)}, union_skipped={skipped}")
    total["ok"] += ok; total["n"] += len(rows); total["reached"] += reached
print(f"  TOTAL {total['ok']}/{total['n']} success, {total['reached']} reached the goal storey")
PY
