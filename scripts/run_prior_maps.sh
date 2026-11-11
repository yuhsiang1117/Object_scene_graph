#!/usr/bin/env bash
# Stage 1 of the cross-anchor protocol: build both priors for every benchmark
# scene, in ONE ASCENT walk per episode, at a 1500-step budget.
#
#   bash scripts/run_prior_maps.sh                    # all eight scenes
#   bash scripts/run_prior_maps.sh 00800-TEEsavR23oF  # just these
#   STEPS=500 OUT=outputs/smoke bash scripts/run_prior_maps.sh 00800-TEEsavR23oF
#
# `mf5_mapping_both` is `mf5_ascentnav_map` + `agent.osg_world_model=true`:
# ASCENT's control flow drives, and OSG's world model is fed the same frames,
# so one walk leaves BOTH the per-storey ObstacleMap stack and the schema-v2
# scene graph. The world model runs its own YOLOE head because ASCENT drives on
# closed-set D-FINE, which cannot see a cracker box.
#
# `ycb.obstacle_map_union=true` keeps EVERY episode's snapshot rather than the
# best one. That is the measured lever, not a preference: the protocol runs one
# mapping episode per authored target and the best-of rule discards the rest --
# on 00800 the union took lower-storey navmesh coverage from 0.51 to 0.81 and
# moved all 12 authored target positions to within 1.2 m of mapped free space,
# one of them from 5.1 m OUTSIDE. Pass 2 must then ALSO set
# `obstacle_map_union=true`, because union mode never writes `<scene>.json`.
#
# SEQUENTIAL ON PURPOSE. The five ASCENT model servers hold ~16 GB of the
# card and ollama's qwen2.5:7b another ~4.7 GB; two Habitat processes on top of
# that is how a night's work ends in a CUDA OOM.
#
# Never wait on a PID in this container: PID 1 is `conda run ... bash`, which is
# not an init and never reaps, so `kill -0` succeeds on a zombie for ever.
# Sequential statements in one script cannot have that bug.
set -u
cd "$(dirname "$0")/.."
# UID/GID in docker/.env are for compose; bash's UID is readonly, so skip them.
set -a; [ -f docker/.env ] && . <(grep -vE '^(UID|GID)=' docker/.env); set +a
umask 002

STEPS=${STEPS:-1500}
# Episodes kept per scene. The protocol generates one per authored target (6-8),
# and the union of all of them was the right answer AT 500 STEPS, where a single
# episode could not cross a staircase and so mapped one storey: on 00800 the
# first snapshot covered 0.775/0.000 of the two storeys' navigable area and it
# took the second to reach 0.759 mean. At 1500 steps one episode crosses, and
# the FIRST snapshot measured 0.787/0.777 -- mean 0.782, already past the
# 3-episode 500-step union. The marginal episode there was worth +0.010 to
# +0.012, so the tail is paid for at roughly 13 minutes per point of coverage.
# MAX_EPISODES is that tail cut off; set it to -1 to keep every episode.
#
# It was 3, not 1 and not 6, while only occupancy counted. One 1500-step episode already passes the gate that
# matters -- every authored target inside the mapped region -- but its worst
# target sat 1.46 m from mapped free space against a 1.5 m gate, which is no
# margin at all. Three is past where the 500-step curve flattened (+0.047 at
# the third episode, then +0.010, +0.012, +0.012) and costs ~38 min a scene
# against ~90. Raise it for a scene the audit fails; the resume check is
# count-aware, so re-running tops a scene up rather than skipping it.
#
# -1 (one episode per authored target) since the scene graph ACCUMULATES. The
# 3-episode cap above was set by OCCUPANCY coverage, and occupancy is not the
# whole prior: each episode's world model is prompted with its own target, and
# the scene graph used to be best-of rather than accumulated, so the snapshot
# named at most one target and in practice none -- 0 of the 25 multi-floor
# search episodes had their target at its static pose in
# outputs/maps_p1500_osg, and pass 2 never had a stale belief to revise. Every
# target now gets the episode whose vocabulary names it, chained through
# `ycb.map_in` exactly as the single-floor maps are (campaign/rebuild_maps.sh),
# and `audit_map_coverage.py --graphs` gates on the result.
MAX_EPISODES=${MAX_EPISODES:--1}
OBS=${OBS:-outputs/maps_v2_ascent}
OSG=${OSG:-outputs/maps_v2_osg}
OUT=${OUT:-outputs/prior_v2}
# The scene graph's own detector. The 11s head was the world model's default;
# pass 2 searches with 11L (configs/detector/yoloe.yaml), and a prior mapped by
# a weaker detector than the one that later disputes it is a confound.
WM_WEIGHTS=${WM_WEIGHTS:-data/weights/yoloe-11l-seg.pt}
# ...and at the resolution pass 2 searches at. `mf5_ascentnav_map` renders
# 640x480 for ASCENT, and the world model then ran YOLOE at imgsz 640 on those
# frames -- a quarter of the pixels pass 2 has (1280x960, imgsz 1280; the
# single-floor priors were built that way too, campaign/rebuild_maps.sh). On
# objects this small scale is the recall (perception/foveate.py): after six
# 1500-step episodes on 00800, each prompting its own target, the graph held a
# bowl and toy-airplane tracks and no cracker box, tin can, pitcher or banana.
# D-FINE and the other served models take any size, RedNet resizes, and the
# intrinsics come from the width, so ASCENT drives the same scene; only its
# perception calls get larger images. `detector.imgsz` reaches the world
# model's YOLOE alone -- D-FINE ignores it.
# The walk is not guaranteed to look at its target (two of six on 00800 never
# had it in view), and the benchmark's premise is that the robot saw it. Each
# episode ends with this many goal view points of its target shown to the
# world model (`ycb.map_tour_views`, eval/prior_map.py).
TOUR_VIEWS=${TOUR_VIEWS:-3}
RGB_W=${RGB_W:-1280}
RGB_H=${RGB_H:-960}
WM_IMGSZ=${WM_IMGSZ:-1280}
# The paper's five multi-floor scenes, 00808 (the development scene), then
# DualMap's released three. The released layouts are read in authored form
# from `outputs/collector_layouts` -- NOT `outputs/substituted_layouts`, which
# swaps the scissors for a bleach cleanser and would put a bleach bottle at the
# scissors' position. 00862 was missing here after it replaced 00808 in the
# paper's set (configs/experiment/mf5_osg_unified.yaml).
MF5="00800-TEEsavR23oF 00821-eF36g7L6Z9M 00862-LT9Jq6dN3Ea 00873-bxsVRursffK 00878-XB4GS9ShBRE 00808-y9hTuugGdiq"
DUALMAP="00829-QaLdnwvtxbs 00848-ziup5kvtCCR 00880-Nfvxx8J5NCo"
COLLECTOR_LAYOUTS=${COLLECTOR_LAYOUTS:-outputs/collector_layouts}
# Further run_eval.py overrides, space-separated, appended to every scene's
# command -- e.g. EXTRA="eval.record_object_frames_dir=/abs/objrec" to record
# what the world model's object layer is fed (eval/object_recorder.py).

SCENES=("$@")
[ ${#SCENES[@]} -eq 0 ] && SCENES=($MF5 $DUALMAP)
mkdir -p "$OBS" "$OSG" "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "budget ${STEPS} steps | episodes/scene ${MAX_EPISODES} | obstacle -> $OBS | graph -> $OSG | runs -> $OUT"

for scene in "${SCENES[@]}"; do
  # Idempotent, and COUNT-AWARE: a scene is done when it has as many snapshots
  # as MAX_EPISODES asks for, not merely when it has one. "Any snapshot exists"
  # was right while every episode was kept; with a cap it would leave a scene
  # that was killed after its first episode permanently one-episode-deep.
  have=$(compgen -G "$OBS/${scene}__*.json" > /dev/null && ls "$OBS/${scene}__"*.json | wc -l || echo 0)
  want=$MAX_EPISODES
  if [ "$want" -lt 0 ]; then
    # -1 means "every episode the protocol generates", which is one per authored
    # target -- so THAT is when the scene is complete, not one snapshot.
    # Treating -1 as want=1 made a top-up a no-op: three scenes held back by the
    # audit for missing storeys were each skipped as "already has 3 (want 1)",
    # the re-audit was byte-identical, and twenty minutes bought nothing.
    layout_root=$([ -n "${extra_root:-}" ] && echo "$extra_root" || echo "")
    want=$(python - "$scene" "$COLLECTOR_LAYOUTS" <<'PYCOUNT'
import json, os, sys
scene, collector = sys.argv[1], sys.argv[2]
for root in (os.environ.get("OSG_YCB_MULTI_FLOOR_ROOT",
                            "/habitat-data-collector/outputs/dualmap_multifloor"),
             collector):
    path = os.path.join(root, scene, "static_scene_config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            print(len(json.load(handle).get("objects") or []) or 1)
        break
else:
    print(1)
PYCOUNT
)
    [ -z "$want" ] && want=1
  fi
  if [ "$have" -ge "$want" ]; then
    log "=== $scene: already has $have snapshot(s) (want $want), skipping"
    continue
  fi
  [ "$have" -gt 0 ] && log "    $scene has $have snapshot(s); re-running the scene to reach $want"
  extra=()
  case " $DUALMAP " in *" $scene "*) extra+=("ycb.layout_root=$COLLECTOR_LAYOUTS");; esac

  log "=== $scene: mapping at $STEPS steps ${extra[*]:-}"
  python scripts/run_eval.py +experiment=mf5_mapping_both \
    "ycb.scenes=[$scene]" "agent.max_steps=$STEPS" \
    "eval.num_episodes=$MAX_EPISODES" \
    "ycb.obstacle_map_out=$OBS" "ycb.map_out=$OSG" "ycb.map_in=$OSG" \
    "agent.osg_world_model_weights=$WM_WEIGHTS" \
    "eval.rgb_width=$RGB_W" "eval.rgb_height=$RGB_H" "detector.imgsz=$WM_IMGSZ" \
    "ycb.map_tour_views=$TOUR_VIEWS" \
    ycb.obstacle_map_union=true \
    eval.behaviour_log=true eval.save_viz=false \
    "${extra[@]}" ${EXTRA:-} "output_dir=$OUT/$scene" > "$OUT/$scene.log" 2>&1
  status=$?
  snaps=$(compgen -G "$OBS/${scene}__*.json" > /dev/null && ls "$OBS/${scene}__"*.json | wc -l || echo 0)
  graph=$([ -f "$OSG/$scene.json" ] && echo yes || echo NO)
  log "    exit=$status  obstacle snapshots=$snaps  scene graph=$graph"
  [ "$status" -ne 0 ] && log "    FAILED -- see $OUT/$scene.log"
done

log "=== stage 1 complete; audit with:"
cat <<EOF
    python scripts/audit_map_coverage.py \\
        --maps $OBS --graphs $OSG \\
        --layout-root /habitat-data-collector/outputs/dualmap_multifloor \\
        --layout-root $COLLECTOR_LAYOUTS --layout-indices 1 \\
        --max-target-track-m 1.0 \\
        --exempt-storey 00808-y9hTuugGdiq:0 \\
        --exempt-storey 00878-XB4GS9ShBRE:0 \\
        --out outputs/audit/map_coverage_${STEPS}.json --label ${STEPS}-union

  The two exemptions are the basements. Neither holds an authored target in any
  layout -- 00808's never did, and 00878's was vacated to put that scene on two
  storeys -- so a storey no episode is scored on is not gated. Declared on the
  command line and echoed in the report header, never assumed inside the code.
EOF
