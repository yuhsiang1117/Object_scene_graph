#!/bin/bash
# Map one DualMap release sequence with Khronos (open-set) and save its scene graph.
#
#   bash scripts/baselines/khronos/map_sequence.sh <scene> <output_dir> [frame_publisher args...]
#   e.g. bash scripts/baselines/khronos/map_sequence.sh 00829-QaLdnwvtxbs outputs/baselines/khronos/map/00829 --count 300
#
# Writes <output_dir>/khronos/ (Khronos's own experiment directory: dsg.json, and the complete
# final map with the last active-window objects, backend/dsg_with_mesh.sparkdsg) and the logs of
# each process next to it.
# (no `set -u`: the conda and ROS activation scripts read unset variables)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE="$1"; OUT="$(realpath -m "$2")"; shift 2
RELEASE="${OSG_DUALMAP_RELEASE_ROOT:-/habitat-data-collector/data/dualmap/HM3D_collect}"
mkdir -p "$OUT"
source "$HERE/env.sh"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-$((40 + RANDOM % 50))}
echo "scene $SCENE -> $OUT (ROS_DOMAIN_ID $ROS_DOMAIN_ID)"

# Its own process group, so that every node it starts goes down with it: a segmenter left
# behind keeps its GPU memory and, on the same domain, answers the next run's frames twice.
# KHRONOS_MAPPER_CONFIG selects another mapper config (Clio's primitives: scripts/baselines/clio/).
setsid ros2 launch "$HERE/dualmap_khronos.launch.yaml" output_dir:="$OUT/khronos" \
  ${KHRONOS_MAPPER_CONFIG:+mapper_config:="$(realpath "$KHRONOS_MAPPER_CONFIG")"} > "$OUT/launch.log" 2>&1 &
LAUNCH=$!
cleanup() {
  kill -INT -- "-$LAUNCH" 2>/dev/null; sleep 10
  kill -9 -- "-$LAUNCH" 2>/dev/null
}
trap cleanup EXIT

# Khronos initialises before it runs -- the background embeddings, then the camera info -- and a
# frame published before it runs is lost, so the publisher sends camera info until the launch log
# says it is running (frame_publisher.py --wait-log), then the frames.
python "$HERE/frame_publisher.py" --sequence "$RELEASE/$SCENE" --wait-log "$OUT/launch.log" "$@" \
  2>&1 | tee "$OUT/publisher.log"
grep -q "\[khronos_node\]: Running" "$OUT/launch.log" || { echo "Khronos never started"; exit 1; }
# Khronos writes its final map when it shuts down (ExperimentManager::run -> logData), so it is
# interrupted alone and given time to write before the rest of the launch goes down. (Its
# finish_mapping_and_save service would do the same, but services hosted by these nodes never
# become reachable from another process here.)
KPID=$(pgrep -g "$LAUNCH" -x khronos_node)
[ -n "$KPID" ] || { echo "no khronos_node in process group $LAUNCH"; exit 1; }
echo "stopping khronos_node ($KPID) to save the map"
kill -INT "$KPID"
alive() { [ -d "/proc/$1" ] && ! grep -q '^State:[[:space:]]*Z' "/proc/$1/status" 2>/dev/null; }
for i in $(seq 1 360); do
  alive "$KPID" || break
  sleep 5
done
alive "$KPID" && echo "khronos_node still running after 30 min"
grep -q "Experiment Finished Cleanly" "$OUT/khronos/experiment_log.txt" 2>/dev/null \
  && echo "saved: $OUT/khronos" || echo "Khronos did not finish cleanly; see $OUT/launch.log"
# The publisher paces on the segmenter, not on Khronos, so a Khronos that fell behind would be
# interrupted with frames still queued. Its timing log has one row per frame the active window
# integrated; anything short of the published count means the map is missing the tail.
PUBLISHED=$(grep -o "done: [0-9]* frames" "$OUT/publisher.log" | grep -o "[0-9]*")
TIMING="$OUT/khronos/timing/active_window_all_timing_raw.csv"
INTEGRATED=$(( $( [ -f "$TIMING" ] && wc -l < "$TIMING" || echo 1 ) - 1 ))
echo "frames published ${PUBLISHED:-?}, integrated $INTEGRATED"
ls -la "$OUT/khronos" "$OUT/khronos/backend" 2>&1 | head -30
