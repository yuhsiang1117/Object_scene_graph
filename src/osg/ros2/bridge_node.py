"""The ROS half: a node beside the graph, a socket toward the pipeline.

Runs on the SYSTEM python3.10 (`/opt/ros/humble` is built against it), never
inside a conda env -- `scripts/ros2/bridge.sh` is the only supported way to
start it. Everything it does is mechanical: cache the newest synchronised
frame, hand goals to `NavigateToPose`, meter a discrete move on cmd_vel, and
pass the operator's floor switch through. No decisions, no conversions; the
pipeline side owns both (osg/ros2/frames.py).

Images are decoded with numpy rather than cv_bridge, and quaternions turned by
hand rather than by tf_transformations, so the image needs only
`ros-humble-ros-base` plus the message packages.
"""
from __future__ import annotations

import argparse
import threading
import time
from typing import Optional

import numpy as np

from . import frames as F
from .wire import serve_once

# ROS encodings this bridge can decode. Anything else is refused loudly: a
# silently mis-decoded image is a map of a room that does not exist.
_RGB_ENCODINGS = {"rgb8": (3, False), "bgr8": (3, True), "rgba8": (4, False),
                  "bgra8": (4, True), "mono8": (1, False)}
_DEPTH_ENCODINGS = {"16UC1": np.uint16, "mono16": np.uint16, "32FC1": np.float32}


def is_compressed_topic(topic: str) -> bool:
    """image_transport's naming: `<base>/compressed` (jpeg/png colour) and
    `<base>/compressedDepth` (png-packed depth). The bridge subscribes with
    the matching message type, so the transport is chosen by topic name alone
    (configs/ros2/stretch3.yaml)."""
    return str(topic).endswith("/compressed") or str(topic).endswith("/compressedDepth")


def compressed_to_array(msg):
    """`sensor_msgs/CompressedImage` -> (ndarray, encoding).

    Colour: a jpeg/png of the bgr8 frame, decoded to rgb8. Depth:
    compressed_depth_image_transport's layout -- a 12-byte ConfigHeader
    (int32 format, float32 depthParam[2]) followed by a PNG. For a 16UC1
    source the PNG holds the millimetres unchanged, which is what the
    RealSense's aligned depth is; a 32FC1 source is inverse-depth quantised
    and undone with depthParam. RVL is refused: cv2 cannot read it, and a
    silently wrong depth is a map of a room that does not exist.
    """
    import cv2

    fmt = str(msg.format)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if "compressedDepth" in fmt:
        # The codec is not always named: the Stretch's plugin says just
        # '16UC1; compressedDepth' (png, its default), a newer one
        # '16UC1; compressedDepth png' or '... rvl'. Trust the bytes: a PNG
        # starts with its signature right after the 12-byte header.
        if "rvl" in fmt or bytes(raw[12:20]) != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"cannot decode depth transport {fmt!r}: not a png payload "
                             "(rvl? set the driver's compressedDepth format to png)")
        a = cv2.imdecode(raw[12:], cv2.IMREAD_UNCHANGED)
        if a is None:
            raise ValueError(f"undecodable compressedDepth frame ({fmt!r}, {raw.size} bytes)")
        src = fmt.split(";")[0].strip()
        if src == "32FC1":
            q_a, q_b = np.frombuffer(bytes(msg.data)[4:12], dtype=np.float32)
            a = a.astype(np.float32)
            return np.where(a > 0, q_a / np.maximum(a - q_b, 1e-6), 0.0).astype(np.float32), "32FC1"
        return a.astype(np.uint16), "16UC1"
    a = cv2.imdecode(raw, cv2.IMREAD_COLOR)   # BGR, whatever the source was
    if a is None:
        raise ValueError(f"undecodable compressed colour frame ({fmt!r}, {raw.size} bytes)")
    return np.ascontiguousarray(a[..., ::-1]), "rgb8"


def grid_to_occupancy(grid, origin_xy, resolution):
    """Costmap2D's array -> OccupancyGrid's (data, origin_xy, (width, height)).

    grid[i, j] sits at pipeline (x, z) = origin + (i + .5, j + .5) * res.
    OccupancyGrid is row-major with rows along ROS y and columns along ROS x,
    and ROS y = -z (frames.py): so rows are z flipped, columns are x, and the
    origin is the corner at the most negative y, i.e. the LARGEST z. Pure, so
    the CPU suite can pin it (tests/unit/test_ros2_deployment.py).
    """
    g = np.asarray(grid, dtype=np.int8)
    n_x, n_z = g.shape
    res = float(resolution)
    ox, oz = (float(v) for v in origin_xy)
    data = np.ascontiguousarray(g.T[::-1, :]).reshape(-1)
    return data, (ox, -(oz + n_z * res)), (n_x, n_z)


def image_to_array(msg) -> np.ndarray:
    """`sensor_msgs/Image` -> ndarray, without cv_bridge."""
    enc = str(msg.encoding)
    if enc in _DEPTH_ENCODINGS:
        dtype = _DEPTH_ENCODINGS[enc]
        a = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(msg.height, msg.width)
        return a.copy()
    if enc not in _RGB_ENCODINGS:
        raise ValueError(f"unsupported image encoding {enc!r}")
    channels, bgr = _RGB_ENCODINGS[enc]
    a = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    a = a.reshape(msg.height, msg.width, channels)
    if channels == 1:
        a = np.repeat(a, 3, axis=2)
    else:
        a = a[..., :3]
        if bgr:
            a = a[..., ::-1]
    return np.ascontiguousarray(a)


def stamp_to_float(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class BridgeNode:
    def __init__(self, args) -> None:
        import rclpy
        from control_msgs.action import FollowJointTrajectory
        from geometry_msgs.msg import Twist
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import Int32
        from tf2_ros import Buffer, TransformListener

        import message_filters

        self.args = args
        self.node = Node("osg_bridge")
        self._lock = threading.Condition()
        self._frame: Optional[dict] = None
        self._seq = 0
        self._info = None
        self._floor: Optional[int] = None
        self._goal_handle = None
        self._goal_id = 0
        self._goal_state = "idle"
        self._distance = float("nan")
        self.last_goal: Optional[dict] = None
        # Where the head is pointing. look_up/look_down are relative, so the
        # bridge has to remember; the Stretch reports it on /joint_states too,
        # but only once the controller is up.
        self._tilt = 0.0

        self.tf_buffer = Buffer()
        self._tf_listener = TransformListener(self.tf_buffer, self.node)

        qos = 10
        # Compressed transport is a topic-name choice (is_compressed_topic):
        # jpeg colour + png depth are ~10x fewer bytes over the wired link
        # than raw 1280x720 frames, which is latency the pipeline never sees.
        from sensor_msgs.msg import CompressedImage

        def _type(topic):
            return CompressedImage if is_compressed_topic(topic) else Image
        rgb = message_filters.Subscriber(self.node, _type(args.rgb_topic), args.rgb_topic)
        depth = message_filters.Subscriber(self.node, _type(args.depth_topic), args.depth_topic)
        self.node.get_logger().info(
            f"camera: {args.rgb_topic} ({'compressed' if is_compressed_topic(args.rgb_topic) else 'raw'}), "
            f"{args.depth_topic} ({'compressed' if is_compressed_topic(args.depth_topic) else 'raw'})")
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb, depth], queue_size=qos, slop=float(args.sync_slop_s))
        self._sync.registerCallback(self._on_images)
        self.node.create_subscription(
            CameraInfo, args.camera_info_topic, self._on_info, qos)
        self.node.create_subscription(Int32, args.floor_topic, self._on_floor, qos)

        self._cmd_vel = self.node.create_publisher(Twist, args.cmd_vel_topic, qos)
        # The goal as handed to Nav2, republished for RViz (scripts/ros2/osg.rviz).
        # An action goal is not a topic, so without this the one thing a person
        # watching the robot most wants to see -- where the pipeline is sending
        # it -- is invisible. Transient-local, so an RViz opened mid-run gets
        # the current goal at once. Nothing in the pipeline reads it.
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile

        self._goal_pub = self.node.create_publisher(
            PoseStamped, args.goal_topic,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._image_pubs: dict = {}  # topic -> publisher, see publish_image
        self._marker_pubs: dict = {}  # likewise, see publish_markers
        self._grid_pubs: dict = {}    # likewise, see publish_grid
        self._nav = ActionClient(self.node, NavigateToPose, args.nav_action)
        self._head = ActionClient(
            self.node, FollowJointTrajectory, args.head_traj_action)
        self._Twist = Twist
        self._NavigateToPose = NavigateToPose
        self._FollowJointTrajectory = FollowJointTrajectory
        self._rclpy = rclpy

    # ------------------------------------------------------------ subscribers

    def _on_info(self, msg) -> None:
        self._info = (np.asarray(msg.k, dtype=float).copy(), int(msg.width), int(msg.height))

    def _on_floor(self, msg) -> None:
        with self._lock:
            self._floor = int(msg.data)
            self._lock.notify_all()
        self.node.get_logger().info(f"floor switch -> {int(msg.data)}")

    def _on_images(self, rgb_msg, depth_msg) -> None:
        if self._info is None:
            return  # CameraInfo is latched-ish; a frame without it has no K
        frame_id = str(self.args.camera_frame or rgb_msg.header.frame_id)
        stamp = stamp_to_float(rgb_msg.header.stamp)
        try:
            T_map_cam = self._lookup(frame_id, rgb_msg.header.stamp)
        except Exception as exc:  # noqa: BLE001 -- a missing TF is normal at startup
            self.node.get_logger().warn(f"no {self.args.map_frame} -> {frame_id}: {exc}",
                                        throttle_duration_sec=5.0)
            return
        K, width, height = self._info
        try:
            rgb_arr = compressed_to_array(rgb_msg)[0] if hasattr(rgb_msg, "format") else image_to_array(rgb_msg)
            if hasattr(depth_msg, "format"):
                depth_arr, depth_enc = compressed_to_array(depth_msg)
            else:
                depth_arr, depth_enc = image_to_array(depth_msg), str(depth_msg.encoding)
        except Exception as exc:  # noqa: BLE001 -- say what, keep serving
            self.node.get_logger().error(f"dropped a frame: {exc}", throttle_duration_sec=5.0)
            return
        with self._lock:
            self._seq += 1
            self._frame = {
                "seq": self._seq, "stamp": stamp,
                "rgb": rgb_arr,
                "depth": depth_arr,
                "depth_encoding": depth_enc,
                "K": K, "width": width, "height": height,
                "T_map_cam": T_map_cam, "frame_id": frame_id,
            }
            self._lock.notify_all()

    def _lookup(self, frame_id: str, stamp) -> np.ndarray:
        from rclpy.duration import Duration
        from rclpy.time import Time

        try:
            tf = self.tf_buffer.lookup_transform(
                self.args.map_frame, frame_id, stamp,
                timeout=Duration(seconds=0.05))
        except Exception:
            # The exact stamp is not in the buffer yet. The latest is a better
            # answer than no frame at all -- at 30 Hz it is one frame stale.
            tf = self.tf_buffer.lookup_transform(
                self.args.map_frame, frame_id, Time())
        t, r = tf.transform.translation, tf.transform.rotation
        return F.transform_to_matrix((t.x, t.y, t.z), (r.x, r.y, r.z, r.w))

    # -------------------------------------------------------------------- ops

    _RMW_VENDORS = {(0x01, 0x10): "cyclonedds", (0x01, 0x0f): "fastdds", (0x01, 0x01): "connext"}

    def robot_rmw(self) -> dict:
        """Which DDS vendor the robot's nodes are on, from the GID of every
        publisher on /tf (the driver, robot_state_publisher, the localiser --
        the nodes a run cannot do without).

        Cross-vendor DDS is a trap this bridge fell into for a whole run:
        topics interoperate, so the map, TF, the camera and even cmd_vel all
        worked -- but services and actions do NOT, so every NavigateToPose
        goal got no reply, was timed out as refused, and the base never moved
        while the pipeline retired 25 frontiers. Discovery crosses vendors
        too, which is why `wait_for_server` kept saying yes. The Stretch runs
        CycloneDDS when launched as stretch_main intends; after a reboot it
        comes up on ROS's default Fast-DDS unless RMW_IMPLEMENTATION is
        exported in every shell that starts part of its stack.
        """
        by_vendor: dict = {}
        for info in self.node.get_publishers_info_by_topic("/tf"):
            gid = list(info.endpoint_gid)
            vendor = self._RMW_VENDORS.get((gid[0], gid[1]), f"{gid[0]:02x}.{gid[1]:02x}") if len(gid) >= 2 else "?"
            by_vendor.setdefault(vendor, []).append(str(info.node_name))
        return by_vendor

    def ping(self) -> dict:
        with self._lock:
            seq = self._seq
        return {"node": "osg_bridge", "frames": seq,
                "ros_time": self.node.get_clock().now().nanoseconds * 1e-9,
                "has_camera_info": self._info is not None,
                # Whether Nav2's action SERVER is up -- not merely the action
                # name, which `ros2 action list` also shows for our own client.
                # Read by run_robot.py before it loads a single model.
                "nav_server": bool(self._nav.server_is_ready()),
                # {vendor: [node, ...]} for the robot's /tf publishers; see robot_rmw.
                "robot_rmw": self.robot_rmw(),
                # Who publishes the map the goals are planned on: slam_toolbox
                # when the robot is mapping as it goes, map_server (AMCL) when
                # it was given one, rtabmap on the earlier setup. Printed by
                # run_robot.py, because a run against the wrong one looks the
                # same until the first goal.
                "map_publishers": self.map_publishers()}

    def map_publishers(self) -> list:
        return sorted({str(info.node_name)
                       for info in self.node.get_publishers_info_by_topic("/map")})

    def get_frame(self, after_seq: int = -1, timeout_s: float = 5.0):
        """The newest frame, optionally waiting for one newer than `after_seq`.

        Sequence numbers rather than timestamps: the caller's clock and the
        camera's are not the same clock. Image stamps are ROS time, which under
        `use_sim_time` starts near zero, so a wall-clock threshold from the
        pipeline could never be satisfied and every step would time out.
        A sequence is the bridge's own count and needs no clock at all.
        """
        deadline = time.time() + float(timeout_s)
        after_seq = int(after_seq)
        with self._lock:
            while True:
                frame = self._frame
                if frame is not None and int(frame["seq"]) > after_seq:
                    return frame
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    return None
                self._lock.wait(remaining)

    def send_goal(self, x: float, y: float, yaw: float, frame_id: str) -> dict:
        from geometry_msgs.msg import PoseStamped

        if not self._nav.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(
                f"no {self.args.nav_action} action server. Is Nav2 running?")
        pose = PoseStamped()
        pose.header.frame_id = str(frame_id)
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y = float(x), float(y)
        qx, qy, qz, qw = F.yaw_to_quat(float(yaw))
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        goal = self._NavigateToPose.Goal()
        goal.pose = pose
        self._goal_pub.publish(pose)

        # Every callback below is stamped with the id of the goal that caused
        # it and ignored once a newer goal exists. Without that, the ABORTED
        # Nav2 reports for a PREEMPTED goal lands milliseconds after its
        # replacement went active and overwrites the replacement's state -- so
        # the pipeline reads `policy_stop` and retires the frontier the robot
        # is at that moment driving to. The pipeline re-posts routinely (any
        # goal that moves past `goal_resend_m`, and the approach's closing
        # walk), so this is not a rare race.
        self._cancel_goal()
        with self._lock:
            self._goal_id += 1
            goal_id = self._goal_id
            self._goal_state = "active"
            self._distance = float("nan")
            self.last_goal = {"x": float(x), "y": float(y), "yaw": float(yaw),
                              "frame_id": str(frame_id), "goal_id": goal_id}
        future = self._nav.send_goal_async(
            goal, feedback_callback=lambda msg: self._on_feedback(goal_id, msg))
        future.add_done_callback(lambda fut: self._on_goal_response(goal_id, fut))
        return {"goal_id": goal_id}

    def _is_current(self, goal_id: int) -> bool:
        """Is this callback about the goal the pipeline is waiting on?"""
        return int(goal_id) == int(self._goal_id)

    def _on_feedback(self, goal_id: int, msg) -> None:
        with self._lock:
            if not self._is_current(goal_id):
                return
            self._distance = float(
                getattr(msg.feedback, "distance_remaining", float("nan")))

    def _on_goal_response(self, goal_id: int, future) -> None:
        handle = future.result()
        with self._lock:
            current = self._is_current(goal_id)
            if current:
                if not handle.accepted:
                    self._goal_state = "rejected"
                    # Said out loud: a goal Nav2 refuses at the door was, until
                    # now, indistinguishable in this log from one it drove.
                    self.node.get_logger().warn(
                        f"goal {goal_id} REJECTED by {self.args.nav_action}: "
                        f"{self.last_goal}")
                    return
                self._goal_handle = handle
                self.node.get_logger().info(
                    f"goal {goal_id} accepted by {self.args.nav_action}: "
                    f"({self.last_goal['x']:.2f}, {self.last_goal['y']:.2f}) in "
                    f"{self.last_goal['frame_id']}")
        if not current:
            # Superseded while still in flight. Cancel it rather than leave a
            # goal Nav2 is driving with nothing tracking it.
            if handle.accepted:
                handle.cancel_goal_async()
            return
        handle.get_result_async().add_done_callback(
            lambda fut: self._on_result(goal_id, fut))

    def _on_result(self, goal_id: int, future) -> None:
        from action_msgs.msg import GoalStatus

        status = future.result().status
        with self._lock:
            if not self._is_current(goal_id):
                return
            self._goal_state = {
                GoalStatus.STATUS_SUCCEEDED: "succeeded",
                GoalStatus.STATUS_ABORTED: "aborted",
                GoalStatus.STATUS_CANCELED: "canceled",
            }.get(status, "aborted")
            self._goal_handle = None
            state = self._goal_state
        # Nav2 gives no reason with the result; the reason is on the ROBOT in
        # bt_navigator's / planner_server's log. This line at least says WHEN
        # and WHICH, so those logs can be read at the right second.
        # Two call sites on purpose. rclpy caches the severity per call site
        # and raises ("Logger severity cannot be changed between calls") when
        # one line is used at two levels -- a CANCELED goal followed by a
        # SUCCEEDED one took the executor thread down this way, and the bridge
        # then served the pipeline with no ROS behind it.
        msg = f"goal {goal_id} {state.upper()} (Nav2 status {int(status)})"
        if state == "succeeded":
            self.node.get_logger().info(msg)
        else:
            self.node.get_logger().warn(msg)

    def nav_status(self) -> dict:
        with self._lock:
            state = self._goal_state
            # `canceling` is an internal waiting state; to the pipeline a goal
            # being taken away has already stopped being active.
            return {"state": "canceled" if state == "canceling" else state,
                    "goal_id": self._goal_id,
                    "distance_remaining": self._distance}

    def cancel(self) -> dict:
        cancelled = self._cancel_goal()
        return {"cancelled": cancelled}

    def _cancel_goal(self) -> bool:
        """Take the base back from Nav2, and wait until it has let go.

        The wait is the point. `cancel_goal_async` only ASKS; Nav2's controller
        keeps publishing to cmd_vel at ~20 Hz until it processes the request,
        so a single zero Twist here would be overwritten within 50 ms and the
        metered move `Ros2Env` starts on the next line would drive against a
        controller that has not yet stopped.
        """
        with self._lock:
            handle, self._goal_handle = self._goal_handle, None
            was_active = self._goal_state == "active"
            if was_active:
                self._goal_state = "canceling"
        if handle is not None:
            handle.cancel_goal_async()
        if was_active:
            deadline = time.time() + float(self.args.cancel_settle_s)
            while time.time() < deadline:
                with self._lock:
                    if self._goal_state != "canceling":
                        break
                time.sleep(0.02)
            with self._lock:
                if self._goal_state == "canceling":
                    self._goal_state = "canceled"
        # Zero the base AFTER Nav2 has stopped publishing, so this is the last
        # command on the topic rather than the first of two.
        self._stop_base()
        return handle is not None

    # ------------------------------------------------- the FSM's own actions

    def execute(self, action: str, forward_m: float, turn_deg: float) -> dict:
        """One discrete move, metered against odometry.

        Closed on TF rather than timed, because a timed open-loop move drifts
        with battery and carpet, and the pipeline's `forward_m` is a distance
        the costmap and the frontier reach radius both assume.
        """
        if action == "move_forward":
            return self._drive(distance_m=float(forward_m), yaw_deg=0.0)
        if action == "turn_left":
            return self._drive(distance_m=0.0, yaw_deg=float(turn_deg))
        if action == "turn_right":
            return self._drive(distance_m=0.0, yaw_deg=-float(turn_deg))
        raise ValueError(f"cannot execute {action!r} on the base")

    def _base_pose(self):
        from rclpy.time import Time

        tf = self.tf_buffer.lookup_transform(
            self.args.odom_frame, self.args.base_frame, Time())
        t, r = tf.transform.translation, tf.transform.rotation
        R = F.quat_to_matrix(r.x, r.y, r.z, r.w)
        return np.array([t.x, t.y]), float(np.arctan2(R[1, 0], R[0, 0]))

    def _drive(self, distance_m: float, yaw_deg: float) -> dict:
        start_xy, start_yaw = self._base_pose()
        target_yaw = np.radians(yaw_deg)
        twist = self._Twist()
        deadline = time.time() + float(self.args.move_timeout_s)
        achieved = 0.0
        rate = 0.05
        while time.time() < deadline:
            xy, yaw = self._base_pose()
            if distance_m > 0.0:
                achieved = float(np.linalg.norm(xy - start_xy))
                if achieved >= distance_m:
                    break
                twist.linear.x = float(self.args.linear_speed)
                twist.angular.z = 0.0
            else:
                achieved = float(np.arctan2(np.sin(yaw - start_yaw),
                                            np.cos(yaw - start_yaw)))
                if abs(achieved) >= abs(target_yaw) - 1e-3:
                    break
                twist.linear.x = 0.0
                twist.angular.z = float(np.sign(target_yaw) * self.args.angular_speed)
            self._cmd_vel.publish(twist)
            time.sleep(rate)
        self._stop_base()
        return {"achieved": achieved, "timed_out": time.time() >= deadline}

    def _stop_base(self) -> None:
        self._cmd_vel.publish(self._Twist())

    def look(self, tilt_delta_deg: float) -> dict:
        """Tilt the head. `stair_sense` and the close look both use this."""
        from builtin_interfaces.msg import Duration as MsgDuration
        from trajectory_msgs.msg import JointTrajectoryPoint

        if not self._head.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"no {self.args.head_traj_action} action server")
        target = self._tilt + np.radians(float(tilt_delta_deg))
        goal = self._FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [self.args.head_tilt_joint]
        point = JointTrajectoryPoint()
        point.positions = [float(target)]
        point.time_from_start = MsgDuration(sec=1)
        goal.trajectory.points = [point]
        self._head.send_goal_async(goal)
        self._tilt = float(target)
        return {"tilt_deg": float(np.degrees(target))}

    def pop_floor_switch(self) -> dict:
        with self._lock:
            floor, self._floor = self._floor, None
        return {"floor": floor}

    def publish_markers(self, topic: str, markers: list, frame_id: str = "map") -> dict:
        """Republish pipeline-side geometry as a MarkerArray for RViz.

        Each entry is a plain dict the pipeline built with no ROS types in
        reach: {ns, id, type, xyz, scale, rgba, text?, points?}. `type` is one
        of sphere | cube | cylinder | arrow | text | line_strip | points. The
        array is prefixed with DELETEALL so what RViz shows is exactly what
        the pipeline last said, with nothing stale left behind -- the scene
        graph forgets objects, and so must the picture. /osg/* only, as
        publish_image.
        """
        from visualization_msgs.msg import Marker, MarkerArray

        if not str(topic).startswith("/osg/"):
            raise ValueError(f"publish_markers is for /osg/* topics, not {topic!r}")
        kinds = {"sphere": Marker.SPHERE, "cube": Marker.CUBE, "cylinder": Marker.CYLINDER,
                 "arrow": Marker.ARROW, "text": Marker.TEXT_VIEW_FACING,
                 "line_strip": Marker.LINE_STRIP, "points": Marker.POINTS}
        pub = self._marker_pubs.get(topic)
        if pub is None:
            pub = self.node.create_publisher(MarkerArray, topic, 1)
            self._marker_pubs[topic] = pub
        stamp = self.node.get_clock().now().to_msg()
        arr = MarkerArray()
        wipe = Marker(); wipe.action = Marker.DELETEALL
        wipe.header.frame_id = str(frame_id); wipe.header.stamp = stamp
        arr.markers.append(wipe)
        for m in markers:
            mk = Marker()
            mk.header.frame_id = str(frame_id); mk.header.stamp = stamp
            mk.ns = str(m.get("ns", "osg")); mk.id = int(m.get("id", 0))
            mk.type = kinds[str(m.get("type", "sphere"))]; mk.action = Marker.ADD
            x, y, z = (float(v) for v in m.get("xyz", (0.0, 0.0, 0.0)))
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = x, y, z
            mk.pose.orientation.w = 1.0
            sx, sy, sz = (float(v) for v in m.get("scale", (0.1, 0.1, 0.1)))
            mk.scale.x, mk.scale.y, mk.scale.z = sx, sy, sz
            r, g, b, a = (float(v) for v in m.get("rgba", (1.0, 1.0, 1.0, 1.0)))
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = r, g, b, a
            if "text" in m:
                mk.text = str(m["text"])
            for pt in m.get("points", ()):
                from geometry_msgs.msg import Point
                p_ = Point(); p_.x, p_.y, p_.z = (float(v) for v in pt)
                mk.points.append(p_)
            arr.markers.append(mk)
        pub.publish(arr)
        return {"published": len(markers)}

    def publish_grid(self, topic: str, grid: np.ndarray, origin_xy, resolution: float,
                     frame_id: str = "map", z: float = 0.0) -> dict:
        """The pipeline's own costmap as an OccupancyGrid, for the map window.

        `grid` is Costmap2D's array: grid[i, j] at pipeline (x, z) =
        origin + (i, j) * resolution, values UNKNOWN -1 / FREE 0 / OCCUPIED 100
        -- OccupancyGrid's own vocabulary. ROS's y is -z, so rows and columns
        swap and the z axis flips; `z` lifts the grid for a stacked storey.
        Latched, so a window opened mid-run gets the current map at once.
        /osg/* only, as publish_image.
        """
        from nav_msgs.msg import OccupancyGrid
        from rclpy.qos import DurabilityPolicy, QoSProfile

        if not str(topic).startswith("/osg/"):
            raise ValueError(f"publish_grid is for /osg/* topics, not {topic!r}")
        pub = self._grid_pubs.get(topic)
        if pub is None:
            pub = self.node.create_publisher(
                OccupancyGrid, topic,
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self._grid_pubs[topic] = pub
        data, (ox, oy), (w, h) = grid_to_occupancy(grid, origin_xy, resolution)
        msg = OccupancyGrid()
        msg.header.frame_id = str(frame_id)
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.info.resolution = float(resolution)
        msg.info.width, msg.info.height = int(w), int(h)
        msg.info.origin.position.x = ox
        msg.info.origin.position.y = oy
        msg.info.origin.position.z = float(z)
        msg.info.origin.orientation.w = 1.0
        msg.data = data.tolist()
        pub.publish(msg)
        return {"published": True}

    def publish_image(self, topic: str, image: np.ndarray, encoding: str = "bgr8") -> dict:
        """Republish a pipeline-side image for RViz (scripts/ros2/osg.rviz).

        The pipeline has no rclpy, so the picture it draws for itself -- the
        rotated frame it consumes, the detector's boxes and masks, the
        simulator-style debug panel -- can only reach a screen through here.
        Confined to /osg/: the bridge relays a debug view, it does not become a
        general publisher. Publishers are created on first use, keyed by topic.
        """
        from sensor_msgs.msg import Image

        if not str(topic).startswith("/osg/"):
            raise ValueError(f"publish_image is for /osg/* topics, not {topic!r}")
        arr = np.ascontiguousarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
            raise ValueError(f"expected an HxWx3 uint8 image, got {arr.shape} {arr.dtype}")
        if encoding not in ("rgb8", "bgr8"):
            raise ValueError(f"encoding must be rgb8 or bgr8, not {encoding!r}")
        pub = self._image_pubs.get(topic)
        if pub is None:
            pub = self.node.create_publisher(Image, topic, 1)
            self._image_pubs[topic] = pub
        msg = Image()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.args.camera_frame or "camera_color_optical_frame"
        msg.height, msg.width = int(arr.shape[0]), int(arr.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = int(arr.shape[1] * 3)
        msg.data = arr.tobytes()
        pub.publish(msg)
        return {"published": True}

    def get_last_goal(self):
        return self.last_goal

    def handlers(self) -> dict:
        return {
            "ping": self.ping, "get_frame": self.get_frame,
            "send_goal": self.send_goal, "nav_status": self.nav_status,
            "cancel": self.cancel, "execute": self.execute, "look": self.look,
            "pop_floor_switch": self.pop_floor_switch, "last_goal": self.get_last_goal,
            "publish_image": self.publish_image, "publish_markers": self.publish_markers,
            "publish_grid": self.publish_grid,
        }


def serve(bridge, addr, authkey) -> None:
    """Accept pipeline connections until the process is killed.

    One client at a time: the pipeline is a single control loop, and a second
    connection would mean two things steering one robot.
    """
    from multiprocessing.connection import Listener

    from multiprocessing.connection import AuthenticationError

    with Listener(addr, authkey=authkey) as listener:
        bridge.node.get_logger().info(f"bridge listening on {addr}")
        while True:
            # A peer that goes away is the pipeline's business, never the
            # bridge's. Seen for real: the pipeline container exiting while a
            # reply was in flight (ConnectionResetError from send), and a bare
            # TCP probe that closed before the auth challenge (EOFError from
            # accept). Either used to take the whole node down -- and with it
            # the TF buffer and the outstanding Nav2 goal -- until compose
            # restarted it.
            try:
                conn = listener.accept()
            except (EOFError, ConnectionResetError, AuthenticationError) as exc:
                bridge.node.get_logger().warn(f"rejected a connection: {exc!r}")
                continue
            bridge.node.get_logger().info("pipeline connected")
            try:
                while serve_once(conn, bridge.handlers()):
                    pass
            except (EOFError, ConnectionResetError, BrokenPipeError) as exc:
                bridge.node.get_logger().warn(f"pipeline dropped mid-call: {exc!r}")
            finally:
                conn.close()
                bridge.node.get_logger().info("pipeline disconnected")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bridge-addr", default="127.0.0.1:18765")
    p.add_argument("--authkey", default="osg-ros2")
    p.add_argument("--rgb-topic", default="/camera/color/image_raw")
    p.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    p.add_argument("--sync-slop-s", type=float, default=0.05)
    p.add_argument("--map-frame", default="map")
    p.add_argument("--camera-frame", default="")
    p.add_argument("--nav-action", default="navigate_to_pose")
    p.add_argument("--cmd-vel-topic", default="/stretch/cmd_vel")
    p.add_argument("--odom-frame", default="odom")
    p.add_argument("--base-frame", default="base_link")
    p.add_argument("--linear-speed", type=float, default=0.15)
    p.add_argument("--angular-speed", type=float, default=0.5)
    p.add_argument("--move-timeout-s", type=float, default=10.0)
    p.add_argument("--cancel-settle-s", type=float, default=1.0)
    p.add_argument("--head-traj-action",
                   default="/stretch_controller/follow_joint_trajectory")
    p.add_argument("--head-tilt-joint", default="joint_head_tilt")
    p.add_argument("--floor-topic", default="/osg/floor")
    # Bridge-only, like --cancel-settle-s: a visualisation topic is not a
    # pipeline setting, so it has no field in the ros2 config group.
    p.add_argument("--goal-topic", default="/osg/goal")
    return p.parse_args(argv)


def _spin_or_die(executor, node) -> None:
    """Spin, and take the process down if spinning stops.

    A callback that raises ends `executor.spin()`; on a daemon thread that was
    silent, and the socket server kept accepting the pipeline while no frame
    and no TF could ever arrive again -- `run_robot.py` waited 900 s on "no
    camera frame". Exiting instead drops the pipeline's connection
    (BridgeUnavailable, loudly) and lets compose's restart bring the node back.
    """
    import os
    import traceback

    try:
        executor.spin()
    except BaseException:  # noqa: BLE001 -- anything here means the ROS side is gone
        traceback.print_exc()
        node.get_logger().fatal("executor stopped; exiting so the bridge is restarted")
        os._exit(3)


def main(argv=None) -> None:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    from .wire import parse_addr

    args = parse_args(argv)
    rclpy.init()
    bridge = BridgeNode(args)
    executor = MultiThreadedExecutor()
    executor.add_node(bridge.node)
    spin = threading.Thread(target=_spin_or_die, args=(executor, bridge.node),
                            daemon=True)
    spin.start()
    try:
        serve(bridge, parse_addr(args.bridge_addr), args.authkey.encode("utf-8"))
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        bridge.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
