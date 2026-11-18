#!/usr/bin/env bash
# Run one dualmap arm over a trial split, bounded by GPU budget and resumable.
#
#   ARM=sensor_final SPLIT=data/splits/dualmap_all.json OUT=outputs/foo \
#     MAX_PARALLEL=2 scripts/run_dualmap_arm.sh
#
# CONDITIONS selects the protocol slice. The default is all three, which is the
# published benchmark (186 trials: 79 static, 54 in_anchor, 53 cross_anchor);
# `in_anchor,cross_anchor` is the 107-trial dynamic split alone. SPLIT only
# supplies the trial ids to resume against, so it must cover what CONDITIONS
# asks for -- `data/splits/dualmap_all.json` holds the 107 dynamic ones.
#
# MAP_ROOT is a FLAT directory of `<scene>.json`, which is what
# `prior_map._map_path` writes and `scripts/run_prior_maps.sh` produces. The
# older `maps_v5` layout put each scene in its own subdirectory; if that is what
# you have, set MAP_SUBDIR=1.
#
# OBSTACLE_MAPS is optional and is the cross-anchor experiment's point: pass 2
# plans over the occupancy ASCENT's navigation built rather than its own.
# MAX_PARALLEL bounds concurrent episodes; each process holds ~1.35 GB of GPU,
# so 2 is about 2.7 GB and 3 is about 4.0 GB. Default 2, to leave the card
# usable by anything else.
#
# Resumable: every trial already present in an episodes.jsonl anywhere under
# $OUT is skipped, and each attempt writes into its own `<scene>/<tag>`
# directory so nothing already measured is overwritten. Re-running after a kill
# picks up exactly the trials that are missing. Reports glob `**/episodes.jsonl`
# so the pieces merge on read.
set -u
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
: "${ARM:?set ARM, e.g. sensor_final}"
: "${SPLIT:=data/splits/dualmap_all.json}"
: "${OUT:?set OUT}"
: "${MAX_PARALLEL:=2}"
# `=` not `:=`: an explicitly EMPTY MAP_ROOT must stay empty. `${V:=d}`
# substitutes the default for an empty value too, so `MAP_ROOT=` still sent
# ycb.map_in=outputs/maps_released and every chunk died with MapStoreError.
# An empty map_in is how a policy that builds its own map -- ASCENT -- is
# told it has no prior to load (`prior_map.load_prior_map` returns None).
: "${MAP_ROOT=outputs/maps_released}"
: "${MAP_SUBDIR:=0}"
: "${OBSTACLE_MAPS:=}"
: "${CONDITIONS:=static,in_anchor,cross_anchor}"
: "${PRESET_PREFIX:=dualmap_protocol_osg_}"
: "${CHUNK:=4}"
mkdir -p "$OUT"
TAG="r$(date +%H%M%S)"

pending() {  # trial ids in $SPLIT for $1 that have no episode under $OUT yet
  python3 - "$SPLIT" "$1" "$OUT" <<'PY'
import json, sys, glob, os
split, scene, out = sys.argv[1], sys.argv[2], sys.argv[3]
want = [t["trial_id"] for t in json.load(open(split))["trials"] if t["scene"] == scene]
done = set()
for p in glob.glob(os.path.join(out, "**", "episodes.jsonl"), recursive=True):
    for line in open(p, encoding="utf-8"):
        if line.strip():
            b = (json.loads(line).get("authored_layout") or {}).get("dualmap") or {}
            if b.get("trial_id"):
                done.add(str(b["trial_id"]))
print(",".join(t for t in want if t not in done))
PY
}

jobs_running() { jobs -rp | wc -l; }

for scene in 00829-QaLdnwvtxbs 00848-ziup5kvtCCR 00880-Nfvxx8J5NCo; do
  ids="$(pending "$scene")"
  [ -z "$ids" ] && { echo "[skip] $scene: nothing pending"; continue; }
  # Split into small chunks so a kill loses at most CHUNK episodes of work.
  IFS=',' read -ra arr <<< "$ids"
  n=${#arr[@]}; k=0
  while [ $k -lt $n ]; do
    chunk=$(IFS=,; echo "${arr[*]:k:CHUNK}")
    while [ "$(jobs_running)" -ge "$MAX_PARALLEL" ]; do sleep 20; done
    d="$OUT/$scene/${TAG}_$k"
    mkdir -p "$d"
    echo "[start] $ARM $scene chunk $k/$n $(date +%H:%M:%S)"
    map_in="$MAP_ROOT"
    [ "$MAP_SUBDIR" = "1" ] && map_in="$MAP_ROOT/$scene"
    obstacle=()
    [ -n "$OBSTACLE_MAPS" ] && obstacle=("ycb.obstacle_map_in=$OBSTACLE_MAPS"
                                        ycb.obstacle_map_union=true
                                        ycb.seed_storeys_from_obstacle_map=true)
    python scripts/run_eval.py "+experiment=${PRESET_PREFIX}${ARM}" \
      "dualmap.scenes=[$scene]" "dualmap.conditions=[$CONDITIONS]" \
      "dualmap.trial_ids=[$chunk]" "ycb.map_in=$map_in" \
      "${obstacle[@]}" "output_dir=$d" > "$d.log" 2>&1 &
    k=$((k + CHUNK))
    sleep 20   # stagger the scene loads
  done
done
wait
echo "[arm complete] $ARM $(date +%H:%M:%S)"
