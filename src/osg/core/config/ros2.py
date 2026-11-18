"""`ros2` group: the robot, and everything about it that has to be checked.

Almost every field here is a question that can only be settled against the real
Stretch -- which topic the aligned depth comes out on, which way the head
camera is rotated, whether the base takes velocity on `/stretch/cmd_vel`. They
are config rather than constants precisely for that reason: the verify-on-robot
list in docs/ROS2.md is this dataclass, and settling one of those questions is
a yaml edit, not a code change.

See `osg/ros2/` for the bridge and `sim/ros2_env.py` for the env.
"""
from __future__ import annotations

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Ros2Config:
    # ------------------------------------------------------------ the bridge
    # rclpy is built for the system Python 3.10 and the pipeline runs on the
    # habitat env's 3.9, so the ROS node is a separate process and this is the
    # socket between them (osg/ros2/wire.py). Loopback: the two processes are
    # always in the same container.
    bridge_addr: str = "127.0.0.1:18765"
    authkey: str = "osg-ros2"
    # How long `run_robot.py` waits for `scripts/ros2/bridge.sh` to come up.
    connect_timeout_s: float = 30.0

    # ------------------------------------------------------------- the camera
    rgb_topic: str = "/camera/color/image_raw"
    # Aligned to colour, so one set of intrinsics describes both images. An
    # unaligned depth image silently mis-places every object by the baseline.
    depth_topic: str = "/camera/aligned_depth_to_color/image_raw"
    camera_info_topic: str = "/camera/color/camera_info"
    # RGB and depth are published separately and their stamps never match
    # exactly; this is the window the pair is matched in. Wider than a frame
    # interval at 30 Hz, far short of the distance the base covers in a step.
    sync_slop_s: float = 0.05
    # The frame the pose is looked up in. Nav2's `map`, so a goal published in
    # the same frame needs no further transform.
    map_frame: str = "map"
    # Empty = whatever the image header says (normally
    # `camera_color_optical_frame`), which is the frame the intrinsics belong
    # to. Override only if the driver publishes a header frame that TF does not
    # carry.
    camera_frame: str = ""
    # The head camera is mounted portrait on the Stretch, so the published
    # image is the room on its side -- and every consumer assumes upright: the
    # detector was trained on upright photographs, the costmap bands by height.
    # Positive is counter-clockwise. VERIFY ON ROBOT: which sign, and whether
    # the driver already rotates (then 0). check_ros2_pipeline.py saves the
    # rotated RGB so this can be settled by looking at it.
    rotate_deg: float = 90.0
    # 16UC1 millimetres, the RealSense convention. Ignored for 32FC1.
    depth_scale: float = 0.001

    # --------------------------------------------------------------- driving
    nav_action: str = "navigate_to_pose"
    goal_frame: str = "map"
    # How far a goal must move before it is re-posted. A frontier centroid
    # shifts by a cell as the map fills, and re-posting restarts Nav2's global
    # planner, so a small dead band keeps one pursuit continuous.
    goal_resend_m: float = 0.5
    # A goal Nav2 neither finishes nor refuses within this is treated as
    # refused (`planning/nav2_driver.py`, converted to control ticks by
    # `step_period_s`). Nav2's recovery behaviours can spin on an unreachable
    # goal indefinitely without ever reporting ABORTED, and the pipeline has a
    # frontier it could retire instead of spending the whole run there. 0
    # waits forever.
    nav_timeout_s: float = 20.0
    # How long a step may wait for a camera frame before the run is failed.
    # Separate from nav_timeout_s, which is about the navigator: a missing
    # frame means the camera or TF is gone, and continuing on a frozen view is
    # the failure that looks like a bad planner.
    frame_timeout_s: float = 10.0
    # How long a run keeps WAITING through missing frames before it gives up,
    # printing once per timeout. This is the carry: on the Stretch a second
    # storey is reached by hand, and slam_toolbox (2D) has to be relaunched
    # there, which takes `map -> camera` away for the minutes it takes to
    # carry the robot up and bring the stack back. Without patience the
    # mapping pass died on the stairs. 0 = fail at the first timeout.
    frame_patience_s: float = 900.0
    # The FSM's own discrete actions (the opening scan, the escape window, the
    # close look) are metered on velocity against odometry rather than sent
    # through the navigator -- a 0.25 m pose per step would invoke a global
    # planner twelve times for one spin.
    cmd_vel_topic: str = "/stretch/cmd_vel"
    odom_frame: str = "odom"
    base_frame: str = "base_link"
    linear_speed: float = 0.15
    angular_speed: float = 0.5
    move_timeout_s: float = 10.0
    # look_up / look_down, which stair_sense and the close look both use.
    head_traj_action: str = "/stretch_controller/follow_joint_trajectory"
    head_tilt_joint: str = "joint_head_tilt"
    look_step_deg: float = 30.0
    # Seconds between control decisions while Nav2 is driving. The pipeline's
    # step is a decision, not a fixed distance, once the base moves
    # continuously; at 0.15 m/s this is ~0.08 m, well inside one `forward_m`.
    step_period_s: float = 0.5

    # ----------------------------------------------------------- the operator
    # Which storey the robot is on. The real-world stand-in for habitat's z:
    # `ros2 topic pub --once /osg/floor std_msgs/Int32 '{data: 1}'`. Needs
    # `floor.source=external` to have any effect.
    floor_topic: str = "/osg/floor"

    # -------------------------------------------------------------- the runs
    # Run 1 explores and writes the map; run 2 loads it and searches. Two
    # invocations of scripts/run_robot.py differing in this one field.
    map_mode: str = "map"  # map | search
    map_dir: str = "outputs/robot_maps"
    # Which map. One tag per physical place, so a place can be re-mapped
    # without disturbing another.
    map_tag: str = "lab"
    # The mapping pass rewrites its map every this many steps, at the path the
    # search pass reads, so a run that dies keeps the map up to its last
    # checkpoint (two runs were lost whole to a closed shell). 0 = only at the
    # end. A save is a few MB of npz+json; at 20 steps it is well under a
    # second every few minutes of driving.
    map_checkpoint_steps: int = 20
    # What to look for. There is no episode dataset on a robot.
    target: str = "chair"
    # Where the staircase is, as (x, y) in the robot's `map` frame -- e.g.
    # `ros2.stairs_xy=[3.2,-1.5]` at launch. Nothing climbs: when the agent
    # asks for a storey nobody has mapped (agent.request_new_storey_when_
    # exhausted) it drives here and waits for the carry, which is where the
    # operator wants it, and what the demo shows. Unset, it waits where it is.
    stairs_xy: Optional[List[float]] = None
    # Hold after the map is restored, before the first step, until the
    # operator declares a storey on /osg/floor. To RESUME a two-floor demo
    # whose search pass died after the carry: the run starts on floor 0 with
    # the stale graph on screen, the switch is published, and the storey lift
    # is shown as it would have been. Frames flow during the hold; the base
    # gets no command and no step is spent.
    wait_for_switch: bool = False
    # RViz: draw the scene graph of every storey, stacked in z by
    # floor.virtual_storey_m (the default -- both floors of a map, never
    # overprinted), or only the storey the agent is on, so a switch visibly
    # clears the floor it left. The bridge prefixes every marker array with
    # DELETEALL, so what is not published is not shown.
    viz_other_storeys: bool = True
    # An operator waypoint: the first exploration goal on storey
    # `waypoint_floor` (-1: the run's first round) is this (x, y) in the
    # robot's `map` frame, facing `waypoint_yaw_deg` on arrival (a quaternion
    # (z, w) is yaw = 2 * atan2(z, w)); the search then resumes as usual from
    # there. A human choosing where the search starts -- say so when the run
    # is reported. Unset, nothing changes.
    waypoint_xy: Optional[List[float]] = None
    waypoint_yaw_deg: Optional[float] = None
    waypoint_floor: int = -1
