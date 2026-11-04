"""Feed a posed RGB-D sequence to Khronos as the ROS 2 topics its launch files expect.

Runs in the Khronos environment (ROS 2 Jazzy, `/workspace/.conda-envs/khronos`), not in the
habitat one. Source: a DualMap release sequence directory (`rgb/`, `depth/`, `pose.txt`,
`camera_intrinsics.json`) -- the same frames DualMap's released `global_map` was built from.

What it publishes, all stamped with the sequence's own timestamps on a simulated `/clock`:

  <ns>/rgb/image_raw      sensor_msgs/Image rgb8
  <ns>/rgb/camera_info    sensor_msgs/CameraInfo (the release's pinhole intrinsics)
  <ns>/depth/image_raw    sensor_msgs/Image 16UC1, millimetres (hydra converts by 1e-3)
  /tf                     odom -> base_link, the release pose
  /tf_static              base_link -> camera (optical), constant

`pose.txt` is T_world_camera in the OpenCV optical convention with the world z-up (checked:
the camera's y axis points down the world z, its z axis forward). base_link is that camera
with ROS body axes (x forward, y left, z up), so odom -> base_link is T_world_camera R_cam_body
and the static base_link -> camera is its inverse rotation.

Pacing: frame i+1 is published only once the open-set segmenter has answered frame i (its
FeatureImage carries frame i's stamp), and no sooner than `--min-period` seconds after frame i,
so neither the segmenter nor Khronos drops a frame for being busy.

`--scale` resizes colour (area average) and depth (nearest) and the intrinsics with them, to feed a
method at its own working resolution (Clio's fine-primitive pixel thresholds are for 640x360).

    python frame_publisher.py --sequence <dir> [--every 1] [--min-period 0.25] [--scale 1.0]
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage

# R_cam_body: the ROS body axes (x forward, y left, z up) as columns, in the camera's optical
# frame (x right, y down, z forward) -- forward is +z, left is -x, up is -y.
R_CAM_BODY = np.array([[0.0, -1.0, 0.0],
                       [0.0, 0.0, -1.0],
                       [1.0, 0.0, 0.0]])


def to_time(t: float) -> Time:
    sec = int(np.floor(t))
    return Time(sec=sec, nanosec=int(round((t - sec) * 1e9)) % 1_000_000_000)


def quat(R: np.ndarray):
    """(x, y, z, w) of a rotation matrix."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    return x, y, z, w


def transform(parent: str, child: str, stamp: Time, R: np.ndarray, t: np.ndarray) -> TransformStamped:
    m = TransformStamped()
    m.header.stamp = stamp
    m.header.frame_id = parent
    m.child_frame_id = child
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = (float(v) for v in t)
    q = quat(R)
    m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = q
    return m


class Sequence:
    def __init__(self, root: Path, every: int, scale: float = 1.0) -> None:
        self.scale = float(scale)
        self.rgb = sorted((root / "rgb").glob("*.png"))
        self.depth = sorted((root / "depth").glob("*.png"))
        poses = np.loadtxt(root / "pose.txt")
        if not (len(self.rgb) == len(self.depth) == len(poses)):
            raise ValueError(f"{root}: {len(self.rgb)} rgb, {len(self.depth)} depth, {len(poses)} poses")
        for a, b in zip(self.rgb, self.depth):
            if a.name != b.name:
                raise ValueError(f"rgb {a.name} and depth {b.name} out of step")
        self.stamps = poses[:, 0]
        self.T = poses[:, 1:].reshape(-1, 4, 4)
        self.index = list(range(0, len(self.rgb), max(1, int(every))))
        self.K = json.loads((root / "camera_intrinsics.json").read_text())


class Publisher(Node):
    def __init__(self, seq: Sequence, ns: str, frame: str, semantic_topic: str) -> None:
        super().__init__("dualmap_frame_publisher")
        self.seq, self.frame = seq, frame
        self.pub_rgb = self.create_publisher(Image, f"{ns}/rgb/image_raw", 10)
        self.pub_info = self.create_publisher(CameraInfo, f"{ns}/rgb/camera_info", 10)
        self.pub_depth = self.create_publisher(Image, f"{ns}/depth/image_raw", 10)
        self.pub_tf = self.create_publisher(TFMessage, "/tf", 100)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.pub_tf_static = self.create_publisher(TFMessage, "/tf_static", latched)
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        self.answered = threading.Event()
        self.waiting_for = None
        self.n_answers = 0
        if semantic_topic:
            from semantic_inference_msgs.msg import FeatureImage
            self.create_subscription(FeatureImage, semantic_topic, self._on_semantic, 10)

    def _on_semantic(self, msg) -> None:
        self.n_answers += 1
        s = msg.header.stamp
        if self.waiting_for is not None and (s.sec, s.nanosec) == self.waiting_for:
            self.answered.set()

    def clock(self, t: float) -> None:
        self.pub_clock.publish(Clock(clock=to_time(t)))

    def static_tf(self, t: float) -> None:
        self.pub_tf_static.publish(TFMessage(transforms=[
            transform("base_link", self.frame, to_time(t), R_CAM_BODY.T, np.zeros(3))]))

    def frame_msgs(self, i: int):
        seq = self.seq
        stamp = to_time(float(seq.stamps[i]))
        rgb_img = PILImage.open(seq.rgb[i]).convert("RGB")
        depth_img = PILImage.open(seq.depth[i])
        s = seq.scale
        if s != 1.0:
            # Colour by area averaging, depth by nearest neighbour (never a blend of two surfaces).
            size = (round(rgb_img.width * s), round(rgb_img.height * s))
            rgb_img = rgb_img.resize(size, PILImage.BOX)
            depth_img = depth_img.resize(size, PILImage.NEAREST)
        rgb = np.asarray(rgb_img)
        depth = np.asarray(depth_img).astype(np.uint16)
        h, w = rgb.shape[:2]
        img = Image(height=h, width=w, encoding="rgb8", is_bigendian=0, step=3 * w,
                    data=rgb.tobytes())
        img.header.stamp, img.header.frame_id = stamp, self.frame
        dep = Image(height=h, width=w, encoding="16UC1", is_bigendian=0, step=2 * w,
                    data=depth.tobytes())
        dep.header.stamp, dep.header.frame_id = stamp, self.frame
        # Pinhole intrinsics follow the resize: focal lengths by the axis' scale, and the principal
        # point as a pixel-centre coordinate, (c + 0.5) * s - 0.5.
        sx, sy = w / seq.K["width"], h / seq.K["height"]
        fx, fy = seq.K["fx"] * sx, seq.K["fy"] * sy
        cx, cy = (seq.K["cx"] + 0.5) * sx - 0.5, (seq.K["cy"] + 0.5) * sy - 0.5
        info = CameraInfo(height=h, width=w, distortion_model="plumb_bob")
        info.header.stamp, info.header.frame_id = stamp, self.frame
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.d = [0.0] * 5
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        T = seq.T[i]
        body = transform("odom", "base_link", stamp, T[:3, :3] @ R_CAM_BODY, T[:3, 3])
        return stamp, img, info, dep, body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True, type=Path)
    ap.add_argument("--every", type=int, default=1, help="publish every n-th frame")
    ap.add_argument("--min-period", type=float, default=0.25, help="real seconds between frames")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="resize the images (and intrinsics) by this factor, e.g. to a method's own resolution")
    ap.add_argument("--ns", default="/dualmap")
    ap.add_argument("--frame", default="camera")
    ap.add_argument("--semantic-topic", default="/semantic_inference/semantic/image_raw",
                    help="FeatureImage topic to pace on; '' to pace on --min-period alone")
    ap.add_argument("--answer-timeout", type=float, default=30.0)
    ap.add_argument("--start-delay", type=float, default=5.0)
    ap.add_argument("--wait-log", default="", help="publish camera info until this log ...")
    ap.add_argument("--wait-pattern", default=r"\[khronos_node\]: Running", help="... matches this")
    ap.add_argument("--wait-timeout", type=float, default=1200.0)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=0, help="0: to the end")
    args = ap.parse_args()

    seq = Sequence(args.sequence, args.every, args.scale)
    index = seq.index[args.first:]
    if args.count:
        index = index[:args.count]
    rclpy.init()
    node = Publisher(seq, args.ns, args.frame, args.semantic_topic)
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    t0 = float(seq.stamps[index[0]])
    node.clock(t0)
    node.static_tf(t0)
    if args.wait_log:
        # Khronos reads CameraInfo while it initialises and only then starts, so the constant
        # camera info (with the first frame's stamp and pose) goes out until it reports running.
        _, _, info0, _, body0 = node.frame_msgs(index[0])
        pattern = re.compile(args.wait_pattern)
        deadline = time.monotonic() + args.wait_timeout
        while not pattern.search(Path(args.wait_log).read_text(errors="replace")
                                 if Path(args.wait_log).exists() else ""):
            if time.monotonic() > deadline:
                raise SystemExit(f"'{args.wait_pattern}' never appeared in {args.wait_log}")
            node.clock(t0)
            node.pub_tf.publish(TFMessage(transforms=[body0]))
            node.pub_info.publish(info0)
            time.sleep(0.5)
        node.get_logger().info(f"'{args.wait_pattern}' seen in {args.wait_log}")
    time.sleep(args.start_delay)
    missed = 0
    log_every = max(1, len(index) // 50)
    for n, i in enumerate(index):
        started = time.monotonic()
        stamp, img, info, dep, body = node.frame_msgs(i)
        node.clock(float(seq.stamps[i]))
        node.pub_tf.publish(TFMessage(transforms=[body]))
        node.static_tf(float(seq.stamps[i]))
        node.answered.clear()
        node.waiting_for = (stamp.sec, stamp.nanosec)
        node.pub_info.publish(info)
        node.pub_depth.publish(dep)
        node.pub_rgb.publish(img)
        if args.semantic_topic and not node.answered.wait(args.answer_timeout):
            missed += 1
            node.get_logger().warning(f"no segmentation for frame {i} in {args.answer_timeout} s")
        rest = args.min_period - (time.monotonic() - started)
        if rest > 0:
            time.sleep(rest)
        if n % log_every == 0:
            node.get_logger().info(f"frame {n + 1}/{len(index)} (sequence index {i}), "
                                   f"segmentations {node.n_answers}, missed {missed}")
    # Let the last frame's data clear the active window before the caller saves.
    t_end = float(seq.stamps[index[-1]])
    for k in range(1, 41):
        node.clock(t_end + 0.25 * k)
        time.sleep(0.05)
    node.get_logger().info(f"done: {len(index)} frames, {node.n_answers} segmentations, {missed} missed")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
