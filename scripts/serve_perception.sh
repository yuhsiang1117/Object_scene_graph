#!/usr/bin/env bash
# ASCENT's perception models, as HTTP services, for the OSG side to call.
#
#   bash scripts/serve_perception.sh          # start
#   bash scripts/serve_perception.sh --stop
#
# Process-per-model over HTTP is ASCENT's own architecture (`model_api/`), not a
# workaround: lavis (BLIP-2) and habitat cannot share one environment -- lavis
# needs numpy 1.x builds and transformers pins habitat's env does not have -- so
# each model runs in the `ascent` env and answers on a port.
#
#   blip2itm :13182  value map + commit gate cosine (`value_model: blip2itm`)
#   sam      :13183  MobileSAM, one mask per detection box
#   gdino    :13184  GroundingDINO, the detector half of the stair fusion
#   ram      :13185  RAM++, per-step scene tags for the LLM prompt
#   dfine    :13186  D-FINE, ASCENT's closed-set COCO target detector
set -eo pipefail
cd "$(dirname "$0")/.."

ASCENT_DIR="${ASCENT_DIR:-relative_work/ascent}"
ASCENT_PYTHON="${ASCENT_PYTHON:-/workspace/.conda-envs/ascent/bin/python}"
SESSION="${OSG_MODEL_SESSION:-osg_models}"

if [ "${1:-}" = "--stop" ]; then
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "no session $SESSION"
    exit 0
fi

[ -x "$ASCENT_PYTHON" ] || { echo "no ascent env at $ASCENT_PYTHON" >&2; exit 1; }
mkdir -p "$ASCENT_DIR/debug/vlm_logs"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION"

launch() {  # name module port
    tmux new-window -t "$SESSION" -n "$1" \
        "cd $PWD/$ASCENT_DIR && PYTHONPATH=$PWD/$ASCENT_DIR \
         HF_HOME=/workspace/data/weights/hf CUDA_VISIBLE_DEVICES=0 \
         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
         $ASCENT_PYTHON -m $2 --port $3 2>&1 | tee debug/vlm_logs/$1.log"
}
launch blip2itm model_api.blip2itm_out       13182
launch sam      model_api.sam_out            13183
launch gdino    model_api.grounding_dino_out 13184
launch ram      model_api.ram_out            13185
launch dfine    model_api.dfine_out          13186

cat <<EOF
Started '$SESSION'. Weights load for up to ~90s. Check with:
    curl -s -o /dev/null -w '%{http_code}\\n' http://localhost:13182/blip2itm
Stop with:
    bash scripts/serve_perception.sh --stop
EOF
