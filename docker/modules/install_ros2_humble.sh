#!/bin/bash
# ROS 2 Humble, for the Stretch 3 layer (src/osg/ros2, docs/ROS2.md).
#
# Binaries, not a source build: the base image is Ubuntu 22.04, which is
# Humble's native apt target, so there is nothing to compile and nothing to
# pin. (habitat-data-collector's Dockerfile.humble builds from source because
# it needs a different Python; we do not.)
#
# Installed against the SYSTEM python3.10, deliberately left out of both conda
# envs. rclpy is built for 3.10 and habitat-sim 0.3.1 pins those envs to 3.9 --
# see the note beside the habitat env above. The two halves talk over a socket
# instead (src/osg/ros2/wire.py), so nothing here may leak onto the pipeline's
# PYTHONPATH: sourcing setup.bash in a login shell would put ROS's 3.10
# site-packages in front of the habitat env's imports.
set -euo pipefail
echo "Installing ROS 2 Humble (system python3.10)"

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
    locales curl gnupg2 lsb-release software-properties-common
# ROS's logging and message handling assume a UTF-8 locale.
locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
add-apt-repository -y universe

curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
    > /etc/apt/sources.list.d/ros2.list

apt-get update
# ros-base, not desktop: no rviz, no demos, no Gazebo. What is here is what
# src/osg/ros2/bridge_node.py imports --
#   nav2-msgs        NavigateToPose, the goal the pipeline publishes
#   control-msgs     FollowJointTrajectory, the head tilt for look_up/look_down
#   message-filters  the RGB/depth time synchroniser
#   tf2-ros-py       map -> camera lookups
#   rmw-cyclonedds   the Stretch's DDS implementation
# -- plus numpy, because images are decoded by hand rather than with cv_bridge.
apt-get install -y --no-install-recommends \
    ros-humble-ros-base \
    ros-humble-nav2-msgs \
    ros-humble-control-msgs \
    ros-humble-message-filters \
    ros-humble-tf2-ros-py \
    ros-humble-rmw-cyclonedds-cpp \
    python3-numpy
rm -rf /var/lib/apt/lists/*

# Fail the build here rather than at the first robot run.
bash -c 'source /opt/ros/humble/setup.bash && /usr/bin/python3 -c "
import rclpy, numpy, message_filters, tf2_ros
import nav2_msgs.action, control_msgs.action, sensor_msgs.msg, geometry_msgs.msg
print(\"ROS 2 Humble OK on\", __import__(\"sys\").version.split()[0])
"'

echo "ROS 2 Humble installation completed!"
