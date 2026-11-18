#!/usr/bin/env bash
# The operator's two windows for a Stretch 3 run, each on its own:
#
#   camera   scripts/ros2/image_window.py on /osg/detections -- the rotated frame
#            the pipeline consumes, with the detector's boxes, masks and labels
#   map      rviz2 with scripts/ros2/osg_map.rviz -- the map the robot is
#            building, its scan and pose, the goal handed to Nav2, the plan,
#            and the scene graph as markers
#
#   bash scripts/ros2/rviz.sh                 # both windows (the default)
#   bash scripts/ros2/rviz.sh --view map      # one of them
#   bash scripts/ros2/rviz.sh --view camera
#   bash scripts/ros2/rviz.sh --view full     # the old everything-in-one rviz (osg.rviz)
#   bash scripts/ros2/bridge.sh --rviz        # alongside the bridge, same shell
#
# On the Thor, neither is typed by hand: the `rviz` compose service runs this
# in the bridge image's rviz target, on the Thor's own screen --
#   docker compose -f docker/compose.thor.yaml --env-file docker/.env run --rm rviz
#   docker compose -f docker/compose.thor.yaml --env-file docker/.env run --rm rviz --view map
#
# System python / system ROS, same reason as bridge.sh: rviz2 is Humble's.
set -eo pipefail
cd "$(dirname "$0")/../.."

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
[ -f "$ROS_SETUP" ] || { echo "no ROS 2 at $ROS_SETUP (rebuild the image)" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ROS_SETUP"

VIEW=both
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --view) VIEW="$2"; shift 2 ;;
        --view=*) VIEW="${1#--view=}"; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
case "$VIEW" in both|map|camera|camera-rqt|full) ;; *)
    echo "--view must be both, map, camera, camera-rqt or full (got '$VIEW')" >&2; exit 2 ;;
esac

command -v rviz2 >/dev/null 2>&1 || {
    echo "rviz2 is not in this image. On the Thor use the rviz compose service" >&2
    echo "(docker/Dockerfile.bridge, target rviz); on x86, apt install ros-humble-rviz2." >&2
    exit 1
}
[ -n "${DISPLAY:-}" ] || {
    echo "DISPLAY is unset: rviz needs a screen. The Thor's GNOME session is :1" >&2
    echo "(docs/THOR.md, 'Watching it in RViz')." >&2
    exit 1
}
# A local display that does not exist is the failure that leaves nothing
# behind: Qt exits at once, and `run --rm` then deletes the container and its
# log. Seen for real with DISPLAY=:0 exported in an ssh shell -- that was the
# LAPTOP's display; the Thor's is :1. Say so instead.
case "$DISPLAY" in
    :*) n="${DISPLAY#:}"; n="${n%%.*}"
        [ -S "/tmp/.X11-unix/X$n" ] || {
            echo "no X server at DISPLAY=$DISPLAY on this host. The ones that exist:" >&2
            ls /tmp/.X11-unix/ 2>/dev/null | sed 's/^X/  :/' >&2
            echo "An ssh shell's DISPLAY is the machine you came FROM; the Thor's desktop is" >&2
            echo ":1 -- run with DISPLAY=:1, or view that desktop over RDP (docs/THOR.md)." >&2
            exit 1
        } ;;
esac

# Extra arguments go straight to rviz2 (e.g. another -d config). The config
# paths are absolute on purpose: rviz2 keeps the config's directory for its own
# bookkeeping, and a relative one is a second variable in a load that already
# has one (see the note at the top of osg.rviz).
CAMERA_TOPIC="${OSG_CAMERA_TOPIC:-/osg/detections}"

camera_window() {
    # scripts/ros2/image_window.py: one topic, one window, sized for the
    # portrait frame. rqt_image_view (also in the image) shows the same
    # picture but opens as a small strip: `--view camera-rqt` if preferred.
    exec python3 scripts/ros2/image_window.py "$CAMERA_TOPIC"
}

case "$VIEW" in
    camera) camera_window ;;
    camera-rqt) exec ros2 run rqt_image_view rqt_image_view --force-discover "$CAMERA_TOPIC" ;;
    map)    exec rviz2 -d "$PWD/scripts/ros2/osg_map.rviz" "${ARGS[@]}" ;;
    full)   exec rviz2 -d "$PWD/scripts/ros2/osg.rviz" "${ARGS[@]}" ;;
    both)
        # The camera window in the background, the map window in the
        # foreground: closing the map window ends the container (`run --rm`),
        # and the camera window with it.
        ( camera_window ) &
        exec rviz2 -d "$PWD/scripts/ros2/osg_map.rviz" "${ARGS[@]}" ;;
esac
