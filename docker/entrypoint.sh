#!/bin/bash
set -euo pipefail

# Docker creates named-volume and bind-mount leaf directories as root. Limit
# the ownership repair to the two repository runtime locations. External
# collector mounts retain their host ownership and are never recursively
# changed by this entrypoint.
runtime_uid="$(id -u)"
runtime_gid="$(id -g)"
for runtime_dir in /workspace/data/weights /workspace/outputs; do
  sudo mkdir -p "${runtime_dir}"
  sudo chown "${runtime_uid}:${runtime_gid}" "${runtime_dir}"
done

# osg is importable via PYTHONPATH=/workspace/src (no pip install needed, so
# the container can run as a non-root user without writing to site-packages).
exec conda run --no-capture-output -n habitat "$@"
