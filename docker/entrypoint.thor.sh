#!/bin/bash
# No `conda run` wrapper: on Jetson the pipeline env IS the system python, and
# the two other interpreters in this deployment live in other places entirely
# (the models venv, selected by ASCENT_PYTHON; rclpy, in the bridge container).
set -euo pipefail
umask "${UMASK:-0002}"
for runtime_dir in /workspace/data/weights /workspace/outputs; do
    sudo mkdir -p "${runtime_dir}"
    sudo chown "$(id -u):$(id -g)" "${runtime_dir}"
done
exec "$@"
