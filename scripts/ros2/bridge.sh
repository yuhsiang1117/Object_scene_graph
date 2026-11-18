#!/usr/bin/env bash
# The ROS 2 half of the pipeline, on the SYSTEM python.
#
#   bash scripts/ros2/bridge.sh [--rgb-topic /a --depth-topic /b ...]
#
# Not the conda python, and not `conda run`: rclpy is built against Ubuntu
# 22.04's python3.10 and both conda envs in this image are 3.9 (habitat-sim
# 0.3.1 pins it). That is the whole reason there is a socket here instead of an
# import -- see src/osg/ros2/__init__.py.
set -eo pipefail
cd "$(dirname "$0")/../.."

# Overridable because the two images put ROS in different places: the x86
# development image installs it beside the pipeline, while on the Thor the
# bridge has a container to itself (docker/Dockerfile.bridge) and sets both of
# these in its environment.
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
ROS_PYTHON="${ROS_PYTHON:-/usr/bin/python3}"
[ -f "$ROS_SETUP" ] || { echo "no ROS 2 at $ROS_SETUP (rebuild the image)" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ROS_SETUP"

# Topics, frames and speeds come from the `ros2` config group, so changing one
# is a yaml edit rather than a code change -- but the bridge runs on the system
# python, which has no hydra. The habitat env composes the config and prints
# the flags; anything passed here is appended, so a command line still wins.
#
#   bash scripts/ros2/bridge.sh                             # the defaults
#   bash scripts/ros2/bridge.sh +experiment=stretch3_map    # a preset's values
#   bash scripts/ros2/bridge.sh -- --rgb-topic /other       # a one-off override
#   bash scripts/ros2/bridge.sh --rviz                      # plus RViz (scripts/ros2/rviz.sh)
#
# `--rviz` is this script's, not Hydra's: it opens RViz beside the node in the
# same shell, which is the x86 single-container case. On the Thor the bridge
# is headless and RViz is its own compose service; see rviz.sh.
HYDRA_ARGS=()
PASSTHROUGH=()
WITH_RVIZ=0
for arg in "$@"; do
    if [ "$arg" = "--" ]; then shift $((${#HYDRA_ARGS[@]} + WITH_RVIZ + 1)); PASSTHROUGH=("$@"); break; fi
    if [ "$arg" = "--rviz" ]; then WITH_RVIZ=1; continue; fi
    HYDRA_ARGS+=("$arg")
done

# Whichever interpreter has hydra. On x86 that is the habitat env; in the
# bridge container it is the image's own python3, which the image sets here.
CONFIG_PYTHON="${CONFIG_PYTHON:-/opt/conda/envs/habitat/bin/python}"
[ -x "$CONFIG_PYTHON" ] || CONFIG_PYTHON="$(command -v python || command -v python3)"
BRIDGE_FLAGS="$("$CONFIG_PYTHON" scripts/ros2/bridge_args.py "${HYDRA_ARGS[@]}")" || {
    echo "could not read the ros2 config group; is the habitat env on PATH?" >&2
    exit 1
}

# `src` on the path, not the habitat env's site-packages: osg.ros2 is written
# to import under both interpreters, and nothing else is imported here.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC2086 -- bridge_args.py shell-quotes each token
eval "set -- $BRIDGE_FLAGS"

if [ "$WITH_RVIZ" = "1" ]; then
    # Background, and not fatal: a missing rviz2 or DISPLAY prints why and the
    # bridge still comes up. The bridge is what the run needs.
    bash scripts/ros2/rviz.sh &
fi
exec "$ROS_PYTHON" -m osg.ros2.bridge_node "$@" "${PASSTHROUGH[@]}"
