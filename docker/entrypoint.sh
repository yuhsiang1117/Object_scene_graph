#!/bin/bash
set -e
# osg is importable via PYTHONPATH=/workspace/src (no pip install needed, so
# the container can run as a non-root user without writing to site-packages).
exec conda run --no-capture-output -n habitat "$@"
