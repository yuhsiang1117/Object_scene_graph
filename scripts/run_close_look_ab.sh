#!/usr/bin/env bash
# The close-look A/B (docs/history/SR_PROPOSAL_CLOSE_LOOK.md): three arms x three scenes,
# dynamic trials only, paired against the tight-ring baseline already on disk
# at outputs/osg_dualmap_tightring. Each (arm, scene) is one process because the
# prior map is per scene; MAX_PARALLEL bounds how many share the GPU.
#
#   scripts/run_close_look_ab.sh              # the working configuration, 3 scenes
#   MAX_PARALLEL=4 ARMS="tightring flat_anchor_v2_island_close" scripts/run_close_look_ab.sh
#   TRIAL_SET=data/splits/dualmap_hard.json MAX_STEPS=300 ARMS=flat_anchor_v2_island \
#     OUT_ROOT=outputs/osg_hard scripts/run_close_look_ab.sh
#
# TRIAL_SET restricts each scene to the trial ids in a subset file written by
# scripts/make_hard_subset.py -- the failures a mechanism is aimed at, which
# are also the trials that run to the budget and cost 68% of a full batch --
# so an arm pairs against any full run on those ids and finishes in about an
# hour instead of five. MAX_STEPS caps the episode budget (agent.max_steps,
# 500 shipped) for mechanism-only iteration: the search-order gate is read in
# the first ~150 steps, and a 300-step cap cuts the failures' cost by ~40%
# while forfeiting the 6-7 of 107 successes that land after step 300. Results
# under either are for iterating, not for reporting. DRY_RUN=1 prints the
# commands and runs nothing.
#
# An arm named X runs the preset `${PRESET_PREFIX}X` (default prefix
# `dualmap_protocol_osg_look_`), so a new arm is a new preset and nothing else.
# Reads the hosted-VLM credential from .env; outputs land under
# $OUT_ROOT/<arm>/<scene>/, logs beside them. Idempotent: an (arm, scene)
# whose episodes.jsonl already holds the scene's trials is skipped, so a
# crashed batch is resumed by running the script again.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
: "${MAX_PARALLEL:=3}"
: "${ARMS:=flat_anchor_v2_island_close}"
: "${SCENES:=00829-QaLdnwvtxbs 00848-ziup5kvtCCR 00880-Nfvxx8J5NCo}"
: "${TRIALS_PER_SCENE:=35}"
: "${OUT_ROOT:=outputs/osg_closelook}"
: "${PRESET_PREFIX:=dualmap_protocol_osg_look_}"
: "${TRIAL_SET:=}"
: "${MAX_STEPS:=}"
: "${DRY_RUN:=}"
# With a subset, a scene's trials can be split across this many processes
# (shard directories under the scene, merged by every report's **/episodes.jsonl
# glob). The subset is lopsided -- 16 trials on 00848 against 6 on 00829 -- so
# one process per scene still takes an hour; six shards of five or six trials
# take twenty minutes. Each process re-loads the scene and prior map (~1 min).
: "${SHARDS_PER_SCENE:=1}"
# Prior maps per scene (outputs/maps_swap for the swapped release).
: "${MAP_ROOT:=outputs/maps_v5}"

# Trial ids for one (scene, shard) from the subset file, as a Hydra list
# literal, and how many that is (the completion check counts against it).
subset_ids() {
  python - "$TRIAL_SET" "$1" "$2" "$SHARDS_PER_SCENE" <<'PY'
import json, sys
ids = [t["trial_id"] for t in json.load(open(sys.argv[1]))["trials"] if t["scene"] == sys.argv[2]]
k, n = int(sys.argv[3]), int(sys.argv[4])
ids = ids[k::n]
print("[" + ",".join(ids) + "]" if ids else "")
PY
}
subset_count() {
  python - "$TRIAL_SET" "$1" "$2" "$SHARDS_PER_SCENE" <<'PY'
import json, sys
ids = [t["trial_id"] for t in json.load(open(sys.argv[1]))["trials"] if t["scene"] == sys.argv[2]]
print(len(ids[int(sys.argv[3])::int(sys.argv[4])]))
PY
}

run_one() {
  local arm=$1 scene=$2 shard=${3:-0}
  local out="$OUT_ROOT/$arm/$scene"
  local want="$TRIALS_PER_SCENE"
  local extra=()
  if [ -n "$TRIAL_SET" ]; then
    local ids; ids="$(subset_ids "$scene" "$shard")"
    if [ -z "$ids" ]; then echo "[skip] $arm $scene shard $shard: no subset trials"; return 0; fi
    extra+=("dualmap.trial_ids=$ids")
    want="$(subset_count "$scene" "$shard")"
    if [ "$SHARDS_PER_SCENE" -gt 1 ]; then out="$out/shard$shard"; fi
  fi
  if [ -n "$MAX_STEPS" ]; then extra+=("agent.max_steps=$MAX_STEPS"); fi
  mkdir -p "$out"
  if [ -f "$out/episodes.jsonl" ] && [ "$(wc -l < "$out/episodes.jsonl")" -ge "$want" ]; then
    echo "[skip] $arm $scene already complete"
    return 0
  fi
  if [ -n "$DRY_RUN" ]; then
    echo "[dry] $arm $scene: +experiment=${PRESET_PREFIX}${arm} ${extra[*]:-} -> $out ($want trials)"
    return 0
  fi
  echo "[start] $arm $scene${TRIAL_SET:+ shard $shard} $(date +%H:%M:%S)"
  python scripts/run_eval.py "+experiment=${PRESET_PREFIX}${arm}" \
    "dualmap.scenes=[$scene]" \
    'dualmap.conditions=[in_anchor,cross_anchor]' \
    "ycb.map_in=$MAP_ROOT/$scene" \
    "output_dir=$out" \
    "+run_tag=CLOSELOOK_${arm^^}" \
    "${extra[@]}" \
    > "$out.log" 2>&1 \
    && echo "[done] $arm $scene${TRIAL_SET:+ shard $shard} $(date +%H:%M:%S)" \
    || echo "[FAILED] $arm $scene${TRIAL_SET:+ shard $shard} $(date +%H:%M:%S) -- see $out.log"
}

jobs_running() { jobs -rp | wc -l; }

shards=1; [ -n "$TRIAL_SET" ] && shards="$SHARDS_PER_SCENE"
for arm in $ARMS; do
  for scene in $SCENES; do
    for ((shard = 0; shard < shards; shard++)); do
      # SHARD_ONLY=k runs just that shard index, for hand-scheduling a shard
      # onto a GPU slot that another batch will free later.
      [ -n "${SHARD_ONLY:-}" ] && [ "$shard" != "$SHARD_ONLY" ] && continue
      while [ "$(jobs_running)" -ge "$MAX_PARALLEL" ]; do sleep 20; done
      run_one "$arm" "$scene" "$shard" &
      [ -n "$DRY_RUN" ] || sleep 45  # stagger the scene loads
    done
  done
done
wait
echo "[all done] $(date +%H:%M:%S)"
