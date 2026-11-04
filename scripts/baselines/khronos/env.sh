# Source me: the Khronos environment (ROS 2 Jazzy from RoboStack) and, once built, its workspace.
#
#   source scripts/baselines/khronos/env.sh
#
# The nav image is Ubuntu 22.04 / ROS 2 Humble, and Khronos does not build on a ROS 2 older than
# Iron, so it gets its own conda environment (RoboStack `robostack-jazzy`, at $KHRONOS_ENV)
# instead of a change to the image.
KHRONOS_ENV=${KHRONOS_ENV:-/workspace/.conda-envs/khronos}
KHRONOS_WS=${KHRONOS_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/relative_work/baselines/khronos_ws}
# The nav image's Python path (osg, the collector) must not leak into a Python 3.12 interpreter.
unset PYTHONPATH
source /opt/conda/etc/profile.d/conda.sh
conda activate "$KHRONOS_ENV"
export PKG_CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export ROS_LOCALHOST_ONLY=1
if [ -f "$KHRONOS_WS/install/setup.bash" ]; then
  source "$KHRONOS_WS/install/setup.bash"
fi
