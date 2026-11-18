#!/usr/bin/env bash
# Stage 1 of the cross-anchor experiment, one scene at a time.
#
# Two mapping passes per scene over the STATIC layout, because the occupancy
# and the scene graph now come from different agents:
#
#   1a  ascentnav  -> ycb.obstacle_map_out   ASCENT's per-storey ObstacleMap
#   1b  nav_agent  -> ycb.map_out            object tracks, presence, storeys
#
# Pass 2 then loads the scene graph WITHOUT its occupancy
# (`ycb.map_in_occupancy=false`) so the ASCENT map is the sole thing the
# planner reads. That arm is deliberately NOT run here.
#
#   bash scripts/legacy/run_mf5_mapping.sh [scene ...]
#
# Sequential on purpose: the five model servers already hold ~16 GB, so two
# Habitat processes at once is how a night's work ends in a CUDA OOM. A scene
# that fails is recorded and the next one still runs.
set -u
cd "$(dirname "$0")/.."
set -a; [ -f docker/.env ] && . ./docker/.env; set +a

SCENES=("$@")
if [ ${#SCENES[@]} -eq 0 ]; then
  SCENES=(00800-TEEsavR23oF 00808-y9hTuugGdiq 00821-eF36g7L6Z9M
          00873-bxsVRursffK 00878-XB4GS9ShBRE)
fi

OBS_ROOT=${OBS_ROOT:-outputs/maps_mf5_ascent}
OSG_ROOT=${OSG_ROOT:-outputs/maps_mf5_osg}
OUT=${OUT:-outputs/mf5_stage1}
mkdir -p "$OBS_ROOT" "$OSG_ROOT" "$OUT"

log() { echo "[$(date +%H:%M:%S)] $*"; }

for scene in "${SCENES[@]}"; do
  log "=== $scene : 1a  ASCENT obstacle map ==========================="
  if [ -f "$OBS_ROOT/$scene.json" ]; then
    log "    already built, skipping"
  else
    python scripts/run_eval.py +experiment=mf5_ascentnav_map \
      "ycb.scenes=[$scene]" "ycb.obstacle_map_out=$OBS_ROOT" \
      eval.behaviour_log=true eval.save_viz=false \
      "output_dir=$OUT/${scene}_ascent" > "$OUT/${scene}_ascent.log" 2>&1
    log "    exit=$?  $(grep -o '\"success_rate\": [0-9.]*' "$OUT/${scene}_ascent.log" | tail -1)"
  fi

  log "=== $scene : 1b  OSG scene graph =============================="
  if [ -f "$OSG_ROOT/$scene.json" ]; then
    log "    already built, skipping"
  else
    # cross_floor_relocations_only must be OFF on a STATIC layout: every target
    # resolves to direction 'same_floor' and is skipped, so the pass would
    # generate zero episodes (src/osg/sim/ycb_env.py:477).
    #
    # `start_on_prior_floor` STAYS ON, and turning it off was a real mistake:
    # for a static layout the relocation source IS that layout, so the "prior
    # floor" is the object's own floor and the flag samples the start there.
    # Without it, starts land anywhere -- measured on 00800, five of six
    # episodes began at y=3.16 while their targets sat at y=0.6-1.2, the agent
    # never descended (traj_y_range <= 1.05 m), and the pass scored 0/6 having
    # never once had a target in view. ycb_env.py:577-585 documents the same
    # trap from the other direction.
    python scripts/run_eval.py +experiment=mf5_osg_unified \
      "ycb.scenes=[$scene]" 'ycb.layout_types=[static]' 'ycb.layout_indices=[1]' \
      ycb.cross_floor_relocations_only=false \
      "ycb.map_out=$OSG_ROOT" eval.save_viz=false \
      "output_dir=$OUT/${scene}_osg" > "$OUT/${scene}_osg.log" 2>&1
    log "    exit=$?  $(grep -o '\"success_rate\": [0-9.]*' "$OUT/${scene}_osg.log" | tail -1)"
  fi

  obs=$([ -f "$OBS_ROOT/$scene.json" ] && echo yes || echo NO)
  osg=$([ -f "$OSG_ROOT/$scene.json" ] && echo yes || echo NO)
  log "    $scene -> obstacle map: $obs | scene graph: $osg"
done

log "=== stage 1 complete ==========================================="
for scene in "${SCENES[@]}"; do
  python - "$scene" "$OBS_ROOT" "$OSG_ROOT" <<'PY'
import json, sys
from pathlib import Path
scene, obs_root, osg_root = sys.argv[1], sys.argv[2], sys.argv[3]
def brief(p, kind):
    p = Path(p) / f"{scene}.json"
    if not p.exists():
        return f"{kind}: MISSING"
    b = json.loads(p.read_text())
    if kind == "obstacle":
        fl = [f for f in b["floors"] if f["explored_cells"] > 0]
        return (f"obstacle: {len(fl)} storeys, "
                f"{sum(f['explored_cells'] for f in fl)} explored cells")
    return f"graph: {len(b.get('tracks') or [])} tracks, {len(b.get('floors') or [])} storeys"
print(f"  {scene}  {brief(obs_root,'obstacle')} | {brief(osg_root,'graph')}")
PY
done
