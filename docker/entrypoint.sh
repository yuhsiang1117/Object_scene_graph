#!/bin/bash
set -e

# Install the bind-mounted osg package once per container if not yet importable.
if ! conda run -n habitat python -c "import osg" >/dev/null 2>&1; then
    if [ -f /workspace/pyproject.toml ]; then
        conda run -n habitat pip install --no-deps -e /workspace >/dev/null
    fi
fi

exec conda run --no-capture-output -n habitat "$@"
