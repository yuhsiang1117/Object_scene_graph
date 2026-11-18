"""The ROS 2 layer: sensor data in, a Nav2 goal out.

Two interpreters, because they cannot be one. Humble's `rclpy` is built against
Ubuntu 22.04's Python 3.10; both conda envs in this image are 3.9 (habitat-sim
0.3.1 pins it, and `docker/Dockerfile` says so where the habitat env is built).
So `bridge_node.py` runs on `/usr/bin/python3` beside the ROS graph, the rest of
the pipeline runs on the habitat interpreter, and they speak over a local socket
-- the same shape the five perception models already use, minus HTTP.

Everything in this package except `bridge_node.py` and `fake_robot.py` must
therefore import under BOTH interpreters: numpy and the standard library only,
no torch, no habitat, no hydra. `osg/__init__.py` and `osg/core/__init__.py` are
both import-light, so `osg.core.types` is reachable from the ROS side too.
"""
