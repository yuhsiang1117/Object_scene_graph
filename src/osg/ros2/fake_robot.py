"""A robot that does not exist, publishing everything the bridge subscribes to.

The point is to answer "is the plumbing right?" with no robot in the room and
no Habitat either: topics, TF, the action contract, the coordinate conversions,
and the floor switch, all exercised end to end by
`scripts/check_ros2_pipeline.py`.

The room is a real box rendered by ray casting, not a constant depth image,
because a constant image maps to no free space and produces no frontier -- and
"the pipeline chose a goal and Nav2 received it" is precisely the property
being checked. Driving is honest too: the fake `NavigateToPose` server moves
the fake base toward the goal, so arriving really does move the published TF.

Runs on the system python3.10 alongside the bridge; `scripts/ros2/fake_robot.sh`.
"""
from __future__ import annotations

import argparse
import math
import threading
import time

import numpy as np

# A 4 m square room, 2.5 m to the ceiling. Wide enough that a 5 m depth clip
# sees walls on every heading, so the costmap closes and a frontier appears at
# the doorway gap below.
ROOM_HALF_M = 4.0
CEILING_M = 2.5
# A gap in the +x wall: unexplored space beyond it is what makes a frontier.
DOOR_HALF_M = 0.6


def render_depth(width, height, K, T_map_cam, max_m=10.0) -> np.ndarray:
    """Z-depth of the room, per pixel, for a camera at `T_map_cam`.

    Rays are parameterised by their camera-frame z, so the intersection
    parameter IS the depth an aligned depth image reports.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    dirs_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    R, o = T_map_cam[:3, :3], T_map_cam[:3, 3]
    d = dirs_cam @ R.T  # world directions, still scaled so s == camera z

    best = np.full((height, width), np.inf)
    planes = ((0, ROOM_HALF_M), (0, -ROOM_HALF_M), (1, ROOM_HALF_M),
              (1, -ROOM_HALF_M), (2, 0.0), (2, CEILING_M))
    for axis, value in planes:
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (value - o[axis]) / d[..., axis]
        hit = o + s[..., None] * d
        ok = np.isfinite(s) & (s > 0.05)
        for other in range(3):
            if other == axis:
                continue
            limit = CEILING_M if other == 2 else ROOM_HALF_M
            low = 0.0 if other == 2 else -ROOM_HALF_M
            ok &= (hit[..., other] >= low - 1e-6) & (hit[..., other] <= limit + 1e-6)
        if axis == 0 and value > 0:  # the doorway: no wall to hit there
            ok &= np.abs(hit[..., 1]) > DOOR_HALF_M
        best = np.where(ok & (s < best), s, best)
    depth = np.where(np.isfinite(best) & (best <= max_m), best, 0.0)
    return depth.astype(np.float32)


# ROS body axes (x forward, y left, z up) -> optical (x right, y down, z fwd).
BODY_TO_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def camera_matrix(pose, camera_height: float) -> np.ndarray:
    """`map -> camera_optical` for a base at `pose` = (x, y, yaw).

    Shared with `loopback.py` rather than written twice: the two fakes have to
    agree on this exactly, or the ROS-free check passes on a convention the
    real bridge does not use, which is the one failure a second fake is
    supposed to rule out.
    """
    x, y, yaw = pose
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                   [math.sin(yaw), math.cos(yaw), 0.0],
                   [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = Rz @ BODY_TO_OPTICAL
    T[:3, 3] = [x, y, float(camera_height)]
    return T


def render_rgb(depth: np.ndarray) -> np.ndarray:
    """Something with structure, so a saved frame is worth looking at."""
    d = depth.copy()
    d[d == 0.0] = np.nan
    near, far = np.nanmin(d) if np.isfinite(d).any() else 0.0, 6.0
    shade = np.nan_to_num((far - d) / max(far - near, 1e-6), nan=0.0)
    shade = np.clip(shade, 0.0, 1.0)
    rgb = np.zeros(depth.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = (shade * 200).astype(np.uint8)
    rgb[..., 1] = (shade * 160).astype(np.uint8)
    rgb[..., 2] = (shade * 120).astype(np.uint8)
    # A horizon line, so a rotated image is obviously rotated.
    rgb[depth.shape[0] // 2, :, :] = 255
    return rgb


class FakeRobot:
    def __init__(self, args) -> None:
        import rclpy
        from geometry_msgs.msg import Twist
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import Int32
        from tf2_ros import TransformBroadcaster

        self.args = args
        self.node = Node("fake_robot")
        self.width, self.height = int(args.width), int(args.height)
        fx = self.width / (2.0 * math.tan(math.radians(args.hfov_deg) / 2.0))
        self.K = np.array([[fx, 0.0, self.width / 2.0 - 0.5],
                           [0.0, fx, self.height / 2.0 - 0.5],
                           [0.0, 0.0, 1.0]])
        # Base pose in `map`: x, y, yaw. The camera rides `camera_height` above.
        self.pose = np.array([float(args.start_x), float(args.start_y), 0.0])
        self.camera_height = float(args.camera_height)
        self._lock = threading.Lock()
        self._twist = np.zeros(2)  # (linear x, angular z)
        self._goal = None
        self._started = time.time()
        self._floor_sent = False

        # The same rule as the bridge (bridge_node.is_compressed_topic): a
        # topic named `.../compressed` or `.../compressedDepth` carries
        # CompressedImage, jpeg for colour and the 12-byte header + png for
        # depth, exactly as image_transport's plugins publish them.
        from sensor_msgs.msg import CompressedImage

        self._rgb_compressed = str(args.rgb_topic).endswith("/compressed")
        self._depth_compressed = str(args.depth_topic).endswith("/compressedDepth")
        self._CompressedImage = CompressedImage
        self._rgb_pub = self.node.create_publisher(
            CompressedImage if self._rgb_compressed else Image, args.rgb_topic, 10)
        self._depth_pub = self.node.create_publisher(
            CompressedImage if self._depth_compressed else Image, args.depth_topic, 10)
        self._info_pub = self.node.create_publisher(CameraInfo, args.camera_info_topic, 10)
        self._floor_pub = self.node.create_publisher(Int32, args.floor_topic, 10)
        # The tick and cmd_vel go in SEPARATE callback groups, and that is what
        # makes `execute()` testable at all. Left in the node's one default
        # (mutually exclusive) group, they cannot run at the same time however
        # many threads the executor has -- and the tick is not cheap: it
        # ray-casts a `width x height` depth image every period (53 ms for the
        # default 640x480 on a Jetson, against a 100 ms period at 10 Hz). The
        # timer is then ready again almost as soon as it yields, so `_on_twist`
        # is starved and the base never moves: the bridge drives it for the
        # full `move_timeout_s` and reports `achieved: 0.0`, which reads as a
        # broken `execute()` when nothing is wrong with the bridge at all.
        # Two groups let the MultiThreadedExecutor above overlap them; `pose`
        # and `_twist` are already `_lock`-guarded for exactly this.
        self._tick_group = MutuallyExclusiveCallbackGroup()
        self._cmd_group = MutuallyExclusiveCallbackGroup()
        self.node.create_subscription(Twist, args.cmd_vel_topic, self._on_twist, 10,
                                      callback_group=self._cmd_group)
        self._tf = TransformBroadcaster(self.node)
        self._Image, self._CameraInfo, self._Int32 = Image, CameraInfo, Int32
        self._rclpy = rclpy

        self._make_servers()
        self.node.create_timer(1.0 / float(args.rate_hz), self._tick,
                               callback_group=self._tick_group)
        self._last_tick = time.time()

    # ------------------------------------------------------------- the servers

    def _make_servers(self) -> None:
        from control_msgs.action import FollowJointTrajectory
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionServer

        self._NavigateToPose = NavigateToPose
        self._nav_server = ActionServer(
            self.node, NavigateToPose, self.args.nav_action, self._on_nav)
        self._head_server = ActionServer(
            self.node, FollowJointTrajectory, self.args.head_traj_action,
            self._on_head)
        self._FollowJointTrajectory = FollowJointTrajectory

    def _on_nav(self, goal_handle):
        """Drive the fake base to the goal, or refuse it if asked to."""
        target = goal_handle.request.pose.pose.position
        goal = np.array([target.x, target.y])
        self.node.get_logger().info(f"goal ({goal[0]:.2f}, {goal[1]:.2f})")
        with self._lock:
            self._goal = goal
        if self.args.abort_goals:
            with self._lock:
                self._goal = None
            goal_handle.abort()
            return self._NavigateToPose.Result()
        deadline = time.time() + float(self.args.nav_timeout_s)
        feedback = self._NavigateToPose.Feedback()
        while time.time() < deadline:
            with self._lock:
                if self._goal is None:  # cancelled, or superseded
                    goal_handle.abort()
                    return self._NavigateToPose.Result()
                remaining = float(np.linalg.norm(goal - self.pose[:2]))
            if goal_handle.is_cancel_requested:
                with self._lock:
                    self._goal = None
                goal_handle.canceled()
                return self._NavigateToPose.Result()
            if remaining <= float(self.args.goal_tolerance_m):
                with self._lock:
                    self._goal = None
                goal_handle.succeed()
                return self._NavigateToPose.Result()
            feedback.distance_remaining = remaining
            goal_handle.publish_feedback(feedback)
            time.sleep(0.05)
        with self._lock:
            self._goal = None
        goal_handle.abort()
        return self._NavigateToPose.Result()

    def _on_head(self, goal_handle):
        goal_handle.succeed()
        return self._FollowJointTrajectory.Result()

    def _on_twist(self, msg) -> None:
        with self._lock:
            self._twist = np.array([float(msg.linear.x), float(msg.angular.z)])

    # --------------------------------------------------------------- the tick

    def _integrate(self, dt: float) -> None:
        with self._lock:
            goal, twist = self._goal, self._twist.copy()
            if goal is not None:
                delta = goal - self.pose[:2]
                dist = float(np.linalg.norm(delta))
                if dist > 1e-6:
                    self.pose[2] = math.atan2(delta[1], delta[0])
                    stepped = min(float(self.args.nav_speed) * dt, dist)
                    self.pose[:2] += delta / dist * stepped
            else:
                self.pose[2] += twist[1] * dt
                self.pose[:2] += np.array([math.cos(self.pose[2]),
                                           math.sin(self.pose[2])]) * twist[0] * dt
            self.pose[:2] = np.clip(self.pose[:2],
                                    -ROOM_HALF_M + 0.3, ROOM_HALF_M - 0.3)

    def _camera_matrix(self) -> np.ndarray:
        with self._lock:
            pose = tuple(self.pose)
        return camera_matrix(pose, self.camera_height)

    def _tick(self) -> None:
        now = time.time()
        dt, self._last_tick = now - self._last_tick, now
        self._integrate(min(dt, 0.2))
        stamp = self.node.get_clock().now().to_msg()
        T = self._camera_matrix()
        depth = render_depth(self.width, self.height, self.K, T)
        self._publish_tf(stamp, T)
        self._publish_images(stamp, depth)
        if (self.args.floor_switch_after > 0 and not self._floor_sent
                and now - self._started >= self.args.floor_switch_after):
            self._floor_sent = True
            msg = self._Int32()
            msg.data = 1
            self._floor_pub.publish(msg)
            self.node.get_logger().info("published floor switch -> 1")

    def _publish_tf(self, stamp, T) -> None:
        from geometry_msgs.msg import TransformStamped

        with self._lock:
            x, y, yaw = self.pose
        # map -> base_link (the fake has no odom drift, so odom == map here,
        # published so the bridge's metered moves have a frame to close on).
        for parent, child, xy, angle in (
                (self.args.map_frame, self.args.odom_frame, (0.0, 0.0), 0.0),
                (self.args.odom_frame, self.args.base_frame, (x, y), yaw)):
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id, tf.child_frame_id = parent, child
            tf.transform.translation.x = float(xy[0])
            tf.transform.translation.y = float(xy[1])
            tf.transform.rotation.z = float(math.sin(angle / 2.0))
            tf.transform.rotation.w = float(math.cos(angle / 2.0))
            self._tf.sendTransform(tf)
        # base_link -> camera_optical, as a fixed mount.
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.args.base_frame
        tf.child_frame_id = self.args.camera_frame
        tf.transform.translation.z = self.camera_height
        qx, qy, qz, qw = _matrix_to_quat(BODY_TO_OPTICAL)
        tf.transform.rotation.x, tf.transform.rotation.y = qx, qy
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self._tf.sendTransform(tf)

    def _publish_images(self, stamp, depth) -> None:
        rgb = render_rgb(depth)
        mm = (depth * 1000.0).astype(np.uint16)

        if self._rgb_compressed:
            import cv2
            img = self._CompressedImage()
            img.header.stamp, img.header.frame_id = stamp, self.args.camera_frame
            img.format = "rgb8; jpeg compressed bgr8"
            ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[..., ::-1]))
            img.data = buf.tobytes()
        else:
            img = self._Image()
            img.header.stamp, img.header.frame_id = stamp, self.args.camera_frame
            img.height, img.width = self.height, self.width
            img.encoding, img.is_bigendian = "rgb8", 0
            img.step = self.width * 3
            img.data = rgb.tobytes()
        self._rgb_pub.publish(img)

        if self._depth_compressed:
            import struct
            import cv2
            dimg = self._CompressedImage()
            dimg.header.stamp, dimg.header.frame_id = stamp, self.args.camera_frame
            dimg.format = "16UC1; compressedDepth png"
            ok, buf = cv2.imencode(".png", mm)
            dimg.data = struct.pack("<iff", 0, 0.0, 0.0) + buf.tobytes()
        else:
            dimg = self._Image()
            dimg.header.stamp, dimg.header.frame_id = stamp, self.args.camera_frame
            dimg.height, dimg.width = self.height, self.width
            dimg.encoding, dimg.is_bigendian = "16UC1", 0
            dimg.step = self.width * 2
            dimg.data = mm.tobytes()
        self._depth_pub.publish(dimg)

        info = self._CameraInfo()
        info.header.stamp, info.header.frame_id = stamp, self.args.camera_frame
        info.height, info.width = self.height, self.width
        info.k = self.K.flatten().tolist()
        info.distortion_model = "plumb_bob"
        self._info_pub.publish(info)


def _matrix_to_quat(R):
    """3x3 -> (x, y, z, w). Only used by the fake's fixed camera mount."""
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
    q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
    q[i], q[j], q[k] = 0.25 * s, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    return tuple(q)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rgb-topic", default="/camera/color/image_raw")
    p.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    p.add_argument("--cmd-vel-topic", default="/stretch/cmd_vel")
    p.add_argument("--floor-topic", default="/osg/floor")
    p.add_argument("--nav-action", default="navigate_to_pose")
    p.add_argument("--head-traj-action",
                   default="/stretch_controller/follow_joint_trajectory")
    p.add_argument("--map-frame", default="map")
    p.add_argument("--odom-frame", default="odom")
    p.add_argument("--base-frame", default="base_link")
    p.add_argument("--camera-frame", default="camera_color_optical_frame")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--hfov-deg", type=float, default=69.0)
    p.add_argument("--camera-height", type=float, default=1.3)
    p.add_argument("--start-x", type=float, default=0.0)
    p.add_argument("--start-y", type=float, default=0.0)
    p.add_argument("--rate-hz", type=float, default=10.0)
    p.add_argument("--nav-speed", type=float, default=0.6)
    p.add_argument("--nav-timeout-s", type=float, default=30.0)
    p.add_argument("--goal-tolerance-m", type=float, default=0.2)
    p.add_argument("--abort-goals", action="store_true",
                   help="refuse every goal, to exercise the blocked path")
    p.add_argument("--floor-switch-after", type=float, default=0.0,
                   help="seconds after start to publish a switch to floor 1")
    return p.parse_args(argv)


def main(argv=None) -> None:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    args = parse_args(argv)
    rclpy.init()
    robot = FakeRobot(args)
    executor = MultiThreadedExecutor()
    executor.add_node(robot.node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        robot.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
