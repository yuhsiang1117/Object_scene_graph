#!/usr/bin/env bash
# Our pipeline on our own authored benchmark: the 15-scene root under
# $OSG_YCB_AUTHORING_ROOT (111 objects, one
# layout index, in_anchor + cross_anchor), scored by the same rule as the
# released DualMap benchmark -- a STOP within 1 m horizontal of the object,
# within three attempts (`ycb.score_by_object_distance`).
#
# Two passes per scene, as docs/ARCHITECTURE.md describes: the prior map is
# built over the STATIC layout with the map directory as both map_out and
# map_in so every static episode starts from what the previous ones mapped,
# then the moved layouts are navigated from that stale map. Each scene is one
# process per pass; MAX_PARALLEL bounds how many share the GPU. A multi-storey
# process takes 1.6-1.9 GB (measured), and six of them ran the 10 GB card out
# of memory mid-batch; four is the safe number.
#
#   scripts/run_authored_15.sh                          # all 15, 4 at a time
#   python scripts/report_authored_15.py --run outputs/osg_authored_15   # the report
#   SCENES="00800-TEEsavR23oF" scripts/run_authored_15.sh
#   PRESET=ycb_authored_15 MAPS=outputs/maps_15 OUT=outputs/osg_authored_15
#
# Idempotent: a scene whose map exists skips the map pass, and a scene whose
# nav episodes.jsonl is complete skips the nav pass.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
: "${PRESET:=ycb_authored_15}"
: "${MAPS:=outputs/maps_15}"
: "${OUT:=outputs/osg_authored_15}"
: "${MAX_PARALLEL:=4}"
: "${ROOT:=${OSG_YCB_AUTHORING_ROOT:-/habitat-data-collector/outputs/dualmap_authoring}}"
: "${SCENES:=$(ls "$ROOT" | grep -E '^[0-9]{5}-' | tr '\n' ' ')}"
: "${DRY_RUN:=}"

run_scene() {
  local scene=$1
  local map_dir="$MAPS/$scene" nav_dir="$OUT/$scene"
  mkdir -p "$map_dir" "$nav_dir"
  local n_objects
  n_objects=$(python - "$ROOT/$scene/static_scene_config.json" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))["objects"]))
PY
)
  if [ -n "$DRY_RUN" ]; then echo "[dry] $scene: $n_objects objects -> map $map_dir, nav $nav_dir"; return 0; fi
  if [ ! -f "$map_dir/$scene.json" ]; then
    echo "[map start] $scene $(date +%H:%M:%S)"
    python scripts/run_eval.py "+experiment=$PRESET" \
      "ycb.scenes=[$scene]" 'ycb.layout_types=[static]' \
      "ycb.map_out=$map_dir" "ycb.map_in=$map_dir" \
      "output_dir=$map_dir/run" "+run_tag=MAP15" \
      > "$map_dir/map.log" 2>&1 \
      || { echo "[map FAILED] $scene $(date +%H:%M:%S) -- see $map_dir/map.log"; return 0; }
    echo "[map done] $scene $(date +%H:%M:%S)"
  else
    echo "[map skip] $scene"
  fi
  local want=$(( n_objects * 2 ))
  if [ -f "$nav_dir/episodes.jsonl" ] && [ "$(wc -l < "$nav_dir/episodes.jsonl")" -ge "$want" ]; then
    echo "[nav skip] $scene already complete"; return 0
  fi
  echo "[nav start] $scene $(date +%H:%M:%S)"
  python scripts/run_eval.py "+experiment=$PRESET" \
    "ycb.scenes=[$scene]" 'ycb.layout_types=[in_anchor,cross_anchor]' \
    "ycb.map_in=$map_dir" \
    "output_dir=$nav_dir" "+run_tag=NAV15" \
    > "$nav_dir.log" 2>&1 \
    && echo "[nav done] $scene $(date +%H:%M:%S)" \
    || echo "[nav FAILED] $scene $(date +%H:%M:%S) -- see $nav_dir.log"
}

jobs_running() { jobs -rp | wc -l; }
for scene in $SCENES; do
  while [ "$(jobs_running)" -ge "$MAX_PARALLEL" ]; do sleep 20; done
  run_scene "$scene" &
  [ -n "$DRY_RUN" ] || sleep 45
done
wait
echo "[all done] $(date +%H:%M:%S)"
