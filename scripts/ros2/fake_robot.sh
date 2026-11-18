#!/usr/bin/env bash
# A robot that does not exist, for checking the pipeline with no robot.
#
#   bash scripts/ros2/fake_robot.sh [--abort-goals] [--floor-switch-after 5]
#
# Publishes the camera topics, the TF chain and a NavigateToPose server, and
# actually moves when it is driven. System python, same reason as bridge.sh.
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

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# The camera topic names come from the same config the bridge reads
# (scripts/ros2/bridge_args.py), so a `.../compressed` name in
# configs/ros2/stretch3.yaml makes this fake publish CompressedImage and the
# bridge decode it -- the check exercises the transport a run will use.
# Flags given on the command line still win (they come last).
CONFIG_PYTHON="${CONFIG_PYTHON:-/opt/conda/envs/habitat/bin/python}"
[ -x "$CONFIG_PYTHON" ] || CONFIG_PYTHON="$(command -v python || command -v python3)"
USER_ARGS=("$@")
TOPIC_FLAGS=()
if FLAGS="$("$CONFIG_PYTHON" scripts/ros2/bridge_args.py 2>/dev/null)"; then
    # shellcheck disable=SC2086 -- bridge_args.py shell-quotes each token
    eval "set -- $FLAGS"
    while [ $# -gt 0 ]; do
        case "$1" in
            --rgb-topic|--depth-topic|--camera-info-topic) TOPIC_FLAGS+=("$1" "$2"); shift 2 ;;
            *) shift ;;
        esac
    done
fi
exec "$ROS_PYTHON" -m osg.ros2.fake_robot "${TOPIC_FLAGS[@]}" "${USER_ARGS[@]}"
