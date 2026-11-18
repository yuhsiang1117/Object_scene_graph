#!/usr/bin/env bash
# The operator's floor switch, as one command.
#
#   bash scripts/ros2/switch_floor.sh 1 lab_upstairs    # load ament_ws/lab_upstairs.yaml, wait for
#                                                       #   localisation, declare floor 1
#   bash scripts/ros2/switch_floor.sh 0 lab_ground --initial-pose "0 0 90"
#   bash scripts/ros2/switch_floor.sh 1 --via none      # virtual storey: keep the current map
#   bash scripts/ros2/switch_floor.sh 1 --check-only    # is the robot inside the current map?
#   bash scripts/ros2/switch_floor.sh 1 lab_upstairs --skip-check   # the operator knows better
#
# Nothing climbs (docs/THOR.md, "Two storeys: the operator's protocol"), so a
# storey change is a carry followed by three things that used to be typed one
# at a time, in an order that matters:
#
#   1. the robot's navigation map becomes that storey's -- <map_name>.yaml
#      through Nav2's map_server (`/map_server/load_map`), or <map_name>.db
#      through RTAB-Map (`/rtabmap/load_database` + localization mode), whichever the
#      robot is running; `--via` forces one, `--via none` skips this step for
#      a virtual storey on the same physical floor, which keeps one map;
#   2. the robot is localised INSIDE that map -- a goal posted while
#      map->base_link lies outside the grid is accepted by Nav2 and never
#      driven (measured 2026-09-22: goal 8 held 68 s, robot never moved);
#   3. the pipeline is told (`/osg/floor N`), and only then.
#
# Runs inside the bridge container (it needs ROS); from the host it re-enters
# itself through `docker exec osg-bridge`. Maps live on the ROBOT's filesystem:
# the name is handed to a service running there, which looks for
# `<maps-dir>/<map_name>.yaml` -- `~/ament_ws` on the Stretch, not a path here.
set -eo pipefail

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
if [ ! -f "$ROS_SETUP" ]; then
    # No ROS here. On the host, re-enter through the bridge container; inside
    # a container without ROS (osg-thor) there is no docker client either, and
    # bash would find the repo's docker/ directory instead of a binary.
    if [ -f /.dockerenv ] || ! DOCKER_BIN="$(type -P docker)"; then
        echo "no ROS 2 in this shell and no docker client: run this from the HOST shell" >&2
        echo "  (elsalab@elsalab:~/Object_scene_graph\$ bash scripts/ros2/switch_floor.sh $*)" >&2
        echo "or from inside osg-bridge, not osg-thor." >&2
        exit 1
    fi
    BRIDGE_CONTAINER="${BRIDGE_CONTAINER:-osg-bridge}"
    TTY=""; [ -t 0 ] && TTY="-t"
    exec "$DOCKER_BIN" exec -i $TTY "$BRIDGE_CONTAINER" bash "scripts/ros2/$(basename "$0")" "$@"
fi
# shellcheck disable=SC1090
source "$ROS_SETUP"

usage() {
    sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
    echo
    echo "usage: $(basename "$0") N [MAP_NAME] [--maps-dir DIR] [--via auto|map_server|rtabmap|none]"
    echo "       MAP_NAME: the robot loads <maps-dir>/MAP_NAME.yaml (map_server) or .db (rtabmap);"
    echo "                 required unless --via none or --check-only"
    echo "                      [--initial-pose \"X Y YAW_DEG\"] [--timeout S] [--base-frame F]"
    echo "                      [--check-only] [--skip-check] [--dry-run]"
    exit 2
}

FLOOR=""
MAP_NAME=""
MAPS_DIR="${OSG_ROBOT_MAPS_DIR:-/home/hello-robot/ament_ws}"
VIA="auto"
INITIAL_POSE=""
TIMEOUT_S=60
BASE_FRAME="base_link"
MAP_FRAME="map"
CHECK_ONLY=0
SKIP_CHECK=0
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --maps-dir) MAPS_DIR="$2"; shift 2 ;;
        --via) VIA="$2"; shift 2 ;;
        --initial-pose) INITIAL_POSE="$2"; shift 2 ;;
        --timeout) TIMEOUT_S="$2"; shift 2 ;;
        --base-frame) BASE_FRAME="$2"; shift 2 ;;
        --map-frame) MAP_FRAME="$2"; shift 2 ;;
        --check-only) CHECK_ONLY=1; shift ;;
        --skip-check) SKIP_CHECK=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage ;;
        -*) echo "unknown option $1" >&2; usage ;;
        *) if [ -z "$FLOOR" ]; then FLOOR="$1"; elif [ -z "$MAP_NAME" ]; then MAP_NAME="$1"; else usage; fi; shift ;;
    esac
done
[[ "$FLOOR" =~ ^[0-9]+$ ]] || usage
if [ -z "$MAP_NAME" ] && [ "$VIA" != "none" ] && [ "$CHECK_ONLY" = "0" ]; then
    echo "a MAP_NAME is needed to switch the robot's map (or --via none to keep the current one)" >&2
    usage
fi

say() { echo "[switch_floor] $*"; }
run() {
    if [ "$DRY_RUN" = "1" ]; then echo "  (dry run) $*"; else "$@"; fi
}
have_service() { ros2 service list 2>/dev/null | grep -qx "$1"; }

# ------------------------------------------------------------ 1. the map
if [ "$VIA" = "auto" ]; then
    if have_service /map_server/load_map; then VIA=map_server
    elif have_service /rtabmap/load_database; then VIA=rtabmap
    else
        say "no map service on the domain (/map_server/load_map or /rtabmap/load_database)."
        say "Is the robot's nav stack up? For a virtual storey on one map, pass --via none."
        exit 1
    fi
fi

if [ "$CHECK_ONLY" = "1" ]; then
    say "check only: not loading a map, not publishing /osg/floor"
elif [ "$VIA" = "none" ]; then
    say "keeping the robot's current map (virtual storey $FLOOR on the same physical floor)"
elif [ "$VIA" = "map_server" ]; then
    MAP_URL="$MAPS_DIR/$MAP_NAME.yaml"
    say "map_server: loading $MAP_URL (a path on the robot)"
    if [ "$DRY_RUN" = "1" ]; then
        echo "  (dry run) ros2 service call /map_server/load_map nav2_msgs/srv/LoadMap \"{map_url: '$MAP_URL'}\""
    else
        OUT="$(ros2 service call /map_server/load_map nav2_msgs/srv/LoadMap "{map_url: '$MAP_URL'}" 2>&1)" || {
            echo "$OUT" | tail -3; say "the load_map call failed"; exit 1; }
        # nav2_msgs/LoadMap: 0 success, 1 map does not exist, 2 invalid data,
        # 3 invalid metadata, 255 undefined failure.
        CODE="$(echo "$OUT" | sed -n 's/.*result=\([0-9]*\).*/\1/p' | head -1)"
        case "$CODE" in
            0) say "$MAP_NAME.yaml loaded" ;;
            1) say "map_server says $MAP_URL DOES NOT EXIST on the robot (result=1)"; exit 1 ;;
            2|3) say "map_server rejected $MAP_URL (result=$CODE: invalid map data/metadata)"; exit 1 ;;
            *) echo "$OUT" | tail -3; say "load_map returned result=${CODE:-?}"; exit 1 ;;
        esac
    fi
elif [ "$VIA" = "rtabmap" ]; then
    DB="$MAPS_DIR/$MAP_NAME.db"
    say "rtabmap: loading $DB (a path on the robot), then localization mode"
    run ros2 service call /rtabmap/load_database rtabmap_msgs/srv/LoadDatabase \
        "{database_path: '$DB', clear: false}" >/dev/null
    run ros2 service call /rtabmap/set_mode_localization std_srvs/srv/Empty >/dev/null
    [ "$DRY_RUN" = "1" ] || say "$MAP_NAME.db loaded; RTAB-Map in localization mode"
else
    say "--via must be auto, map_server, rtabmap or none"; exit 2
fi

# ------------------------------------------------- 2. an initial pose (AMCL)
if [ -n "$INITIAL_POSE" ] && [ "$CHECK_ONLY" = "0" ]; then
    read -r PX PY PYAW <<<"$INITIAL_POSE"
    say "publishing /initialpose ($PX, $PY, yaw ${PYAW} deg) in $MAP_FRAME"
    run ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "$(python3 - "$PX" "$PY" "$PYAW" "$MAP_FRAME" <<'EOF'
import math, sys
x, y, yaw, frame = float(sys.argv[1]), float(sys.argv[2]), math.radians(float(sys.argv[3])), sys.argv[4]
cov = [0.0] * 36; cov[0] = cov[7] = 0.25; cov[35] = math.radians(10) ** 2
print("{header: {frame_id: '%s'}, pose: {pose: {position: {x: %f, y: %f, z: 0.0}, "
      "orientation: {z: %f, w: %f}}, covariance: %s}}" % (frame, x, y, math.sin(yaw / 2), math.cos(yaw / 2), cov))
EOF
)" >/dev/null
fi

# ---------------------------------- 3. is the robot inside the map it will drive on?
say "waiting up to ${TIMEOUT_S}s for $MAP_FRAME->$BASE_FRAME inside the /map grid"
if [ "$DRY_RUN" = "1" ]; then
    echo "  (dry run) skipping the localisation check"
elif [ "$SKIP_CHECK" = "1" ]; then
    say "--skip-check: NOT verifying the robot is inside the map; a goal outside it is accepted and never driven"
else
    python3 - "$MAP_FRAME" "$BASE_FRAME" "$TIMEOUT_S" <<'EOF'
import sys, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener

map_frame, base_frame, timeout_s = sys.argv[1], sys.argv[2], float(sys.argv[3])
rclpy.init()
node = Node("osg_switch_floor_check")
buf = Buffer(); TransformListener(buf, node)
grid = {}
qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                 reliability=ReliabilityPolicy.RELIABLE)
node.create_subscription(OccupancyGrid, "/map", lambda m: grid.__setitem__("m", m), qos)

t0 = time.time(); last = None; ok = False
while time.time() - t0 < timeout_s:
    rclpy.spin_once(node, timeout_sec=0.2)
    m = grid.get("m")
    try:
        tf = buf.lookup_transform(map_frame, base_frame, rclpy.time.Time())
    except Exception:
        tf = None
    if m is None or tf is None:
        continue
    x, y = tf.transform.translation.x, tf.transform.translation.y
    i = m.info
    c = int((x - i.origin.position.x) / i.resolution)
    r = int((y - i.origin.position.y) / i.resolution)
    inside = 0 <= c < i.width and 0 <= r < i.height
    val = m.data[r * i.width + c] if inside else None
    x1 = i.origin.position.x + i.width * i.resolution
    y1 = i.origin.position.y + i.height * i.resolution
    last = (x, y, i.origin.position.x, x1, i.origin.position.y, y1, i.width, i.height, inside, val)
    # Nav2 plans from an unknown cell (NavFn/Smac allow_unknown defaults to
    # true); what it cannot do is plan from outside the grid or out of a
    # lethal cell. Measured 2026-09-22: the goal that was never driven had the
    # robot OUTSIDE the map; a put-down spot the map left unknown is fine.
    if inside and val is not None and (val < 0 or val < 50):
        ok = True
        break
    time.sleep(0.5)
rclpy.shutdown()
if last is None:
    print("[switch_floor] no /map or no %s->%s TF in %.0fs: is the robot's nav stack up, "
          "and is this shell on its DDS domain?" % (map_frame, base_frame, timeout_s))
    sys.exit(1)
x, y, x0, x1, y0, y1, w, h, inside, val = last
where = ("inside, on a free cell" if ok and val is not None and val >= 0 else
         "inside, on an UNKNOWN cell (Nav2 may plan from it; if it does not move, re-localise)" if ok else
         "inside but on an OCCUPIED cell (value %s)" % val if inside else "OUTSIDE the grid")
print("[switch_floor] /map %dx%d cells, x[%.2f, %.2f] y[%.2f, %.2f]; robot at (%.2f, %.2f): %s"
      % (w, h, x0, x1, y0, y1, x, y, where))
if not ok:
    print("[switch_floor] not localised in the map it will drive on. Point the camera at a mapped, "
          "textured view, or pass --initial-pose \"X Y YAW_DEG\" for AMCL, then run again.")
    sys.exit(1)
EOF
fi

# ------------------------------------------------------- 4. tell the pipeline
if [ "$CHECK_ONLY" = "1" ]; then
    say "check passed; run again without --check-only to declare floor $FLOOR"
    exit 0
fi
say "declaring floor $FLOOR on /osg/floor"
if [ "$DRY_RUN" = "1" ]; then
    echo "  (dry run) ros2 topic pub --once /osg/floor std_msgs/msg/Int32 \"{data: $FLOOR}\""
    exit 0
fi
# A subscriber first, so the publish can be seen to arrive on the domain the
# bridge listens on -- `pub --once` prints nothing on success.
HEARD="$(mktemp)"
( timeout 10 ros2 topic echo --once /osg/floor std_msgs/msg/Int32 >"$HEARD" 2>/dev/null ) &
ECHO_PID=$!
sleep 1.5
ros2 topic pub --once /osg/floor std_msgs/msg/Int32 "{data: $FLOOR}" >/dev/null
wait "$ECHO_PID" || true
if grep -q "data: $FLOOR" "$HEARD"; then
    say "floor $FLOOR published and heard on the domain. The run applies it on its next step;"
    say "the bridge log shows 'floor switch -> $FLOOR' (docker logs osg-bridge)."
else
    say "published, but no echo heard within 10 s -- check ROS_DOMAIN_ID / CycloneDDS peers."
    rm -f "$HEARD"; exit 1
fi
rm -f "$HEARD"
