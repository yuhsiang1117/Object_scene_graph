#!/usr/bin/env python3
"""One image topic in one window, sized for a portrait frame.

The operator's camera window: the rotated frame the pipeline consumes with the
detector's boxes, masks and labels on it (/osg/detections, from
src/osg/eval/debug_stream.py). rqt_image_view shows the same picture but opens
as a 400x200 strip whose saved geometry could not be seeded from outside; this
is the window at the size a 720x1280 frame wants, scaled to fit if resized.

    python3 scripts/ros2/image_window.py [/topic] [--title T] [--geometry WxH+X+Y]

System python / system ROS (rclpy + python_qt_binding, both in the rviz image).
"""
import argparse
import re
import sys

import numpy as np
import rclpy
from python_qt_binding.QtCore import Qt, QTimer
from python_qt_binding.QtGui import QImage, QPixmap
from python_qt_binding.QtWidgets import QApplication, QLabel
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class Window(QLabel):
    def __init__(self, node, topic: str, title: str) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #202020; color: #bbbbbb; font-size: 16px;")
        self.setText(f"waiting for {topic} ...")
        self.setMinimumSize(160, 120)
        self._pix = None
        self._n = 0
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        node.create_subscription(Image, topic, self._on_image, qos)
        self._timer = QTimer(self)
        self._timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.0))
        self._timer.start(20)

    def _on_image(self, msg: Image) -> None:
        h, w = int(msg.height), int(msg.width)
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        enc = msg.encoding.lower()
        if enc in ("bgr8", "rgb8"):
            arr = buf.reshape(h, msg.step)[:, : w * 3].reshape(h, w, 3)
            if enc == "bgr8":
                arr = arr[..., ::-1]
            arr = np.ascontiguousarray(arr)
            img = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
        elif enc in ("mono8", "8uc1"):
            arr = np.ascontiguousarray(buf.reshape(h, msg.step)[:, :w])
            img = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
        else:
            self.setText(f"unsupported encoding {msg.encoding}")
            return
        self._pix = QPixmap.fromImage(img.copy())
        self._n += 1
        self._show()

    def _show(self) -> None:
        if self._pix is None:
            return
        self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt)
        super().resizeEvent(event)
        self._show()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("topic", nargs="?", default="/osg/detections")
    p.add_argument("--title", default=None)
    p.add_argument("--geometry", default="560x1000+20+40", help="WxH+X+Y, default sized for a 720x1280 frame")
    args, qt_args = p.parse_known_args(argv)
    m = re.fullmatch(r"(\d+)x(\d+)\+(\d+)\+(\d+)", args.geometry)
    if not m:
        p.error("--geometry must look like 560x1000+20+40")
    w, h, x, y = (int(v) for v in m.groups())

    rclpy.init(args=None)
    node = rclpy.create_node("osg_image_window")
    app = QApplication([sys.argv[0]] + qt_args)
    win = Window(node, args.topic, args.title or f"OSG camera  {args.topic}")
    win.setGeometry(x, y, w, h)
    win.show()
    try:
        return app.exec_()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
