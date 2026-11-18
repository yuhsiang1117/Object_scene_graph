#!/usr/bin/env bash
# Does the ROS layer work, with no robot and no Habitat?
#
#   bash scripts/ros2/check_pipeline.sh          # start both, run the checks
#   bash scripts/ros2/check_pipeline.sh --stop
#
# Brings up the fake robot and the bridge in a tmux session (the pattern
# scripts/serve_perception.sh uses), then runs the checks from the habitat env
# -- so the two interpreters are exercised exactly as they are in a real run.
#
# What it proves: frames arrive and convert into the pipeline's world; a goal
# the pipeline picks reaches NavigateToPose in ROS coordinates; driving there
# lands where the pipeline asked; discrete actions move the base the right way;
# and the operator's floor switch arrives. See docs/ROS2.md.
set -eo pipefail
cd "$(dirname "$0")/../.."

SESSION="${OSG_ROS2_SESSION:-osg_ros2}"

# No ROS needed: the fake robot answers on the socket directly, so everything
# above the bridge is checked even in an image built before the ROS layer.
if [ "${1:-}" = "--loopback" ]; then
    exec python scripts/check_ros2_pipeline.py --loopback "${@:2}"
fi

if [ "${1:-}" = "--stop" ]; then
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "no session $SESSION"
    exit 0
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION"
mkdir -p outputs/ros2_check
tmux new-window -t "$SESSION" -n fake_robot \
    "bash scripts/ros2/fake_robot.sh --floor-switch-after 6 2>&1 | tee outputs/ros2_check/fake_robot.log"
sleep 2
tmux new-window -t "$SESSION" -n bridge \
    "bash scripts/ros2/bridge.sh 2>&1 | tee outputs/ros2_check/bridge.log"

echo "waiting for the bridge to accept connections..."
for _ in $(seq 40); do
    if python -c "
import socket, sys
s = socket.socket()
s.settimeout(0.5)
sys.exit(0 if s.connect_ex(('127.0.0.1', 18765)) == 0 else 1)
" 2>/dev/null; then break; fi
    sleep 0.5
done

set +e
python scripts/check_ros2_pipeline.py "$@"
status=$?
set -e

cat <<EOS

Logs:    outputs/ros2_check/{fake_robot,bridge}.log
Attach:  tmux attach -t $SESSION
Stop:    bash scripts/ros2/check_pipeline.sh --stop
EOS
exit $status
