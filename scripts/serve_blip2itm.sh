#!/usr/bin/env bash
# Serve ASCENT's BLIP-2 ITM value model for OSG to score its value map against.
#
#   bash scripts/serve_blip2itm.sh &
#   python scripts/run_eval.py +experiment=ascentnav_blip2 eval=scenes20_ep0to4
#
# It runs in the `ascent` conda env, not habitat's: lavis needs numpy 1.x builds
# and transformers pins that habitat's env does not have, and the two cannot
# co-exist in one interpreter. Process-per-model over HTTP is ASCENT's own
# architecture (`scripts/launch_vlm_servers_ascent.sh`), so this is their
# deployment, not a workaround around it.
set -euo pipefail
ENV_PY="${ASCENT_PYTHON:-/tmp/.conda/envs/ascent/bin/python}"
PORT="${BLIP2ITM_PORT:-13182}"
cd "$(dirname "$0")/../relative_work/ascent"
[ -x "$ENV_PY" ] || { echo "no ascent env at $ENV_PY (see docker/Dockerfile.ascent)"; exit 1; }
echo "serving BLIP-2 ITM on :$PORT  (first call loads ~2.4 GB to GPU)"
exec env BLIP2ITM_PORT="$PORT" PYTHONPATH=. "$ENV_PY" -m model_api.blip2itm_out --port "$PORT"
