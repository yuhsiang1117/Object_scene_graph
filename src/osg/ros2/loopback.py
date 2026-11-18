"""The fake robot again, with the ROS taken out.

`fake_robot.py` proves the ROS half works: real topics, real TF, a real
`NavigateToPose` server. It needs ROS installed to say so. This serves the same
wire protocol directly, so everything ABOVE the bridge -- the socket, the
transport, `Ros2Env`, `Nav2Driver`, the agent, the two-run map protocol -- is
runnable and testable with no ROS in the image at all.

The two are deliberately not independent: the room, the camera and the driving
model are `fake_robot`'s, imported. What differs is only how the pose and the
images reach the pipeline. So a check that passes here and fails against the
real bridge has localised the fault to the ROS half, which is the entire reason
for having both.

    python -m osg.ros2.loopback --floor-switch-after 5
"""
from __future__ import annotations

import argparse
import math
import threading
import time
from typing import Optional

import numpy as np

from .fake_robot import ROOM_HALF_M, camera_matrix, render_depth, render_rgb
from .wire import parse_addr, serve_once


class LoopbackRobot:
    """A base that drives, a camera that renders, and a floor switch."""

    def __init__(self, args) -> None:
        self.args = args
        self.width, self.height = int(args.width), int(args.height)
        fx = self.width / (2.0 * math.tan(math.radians(args.hfov_deg) / 2.0))
        self.K = np.array([[fx, 0.0, self.width / 2.0 - 0.5],
                           [0.0, fx, self.height / 2.0 - 0.5],
                           [0.0, 0.0, 1.0]])
        self.pose = np.array([float(args.start_x), float(args.start_y), 0.0])
        self.camera_height = float(args.camera_height)
        self._lock = threading.Lock()
        self._goal: Optional[np.ndarray] = None
        self._goal_id = 0
        self._state = "idle"
        self._floor: Optional[int] = None
        self._seq = 0
        self.last_goal: Optional[dict] = None
        self._started = time.time()
        self._floor_sent = False
        self.tilt_deg = 0.0

    # -------------------------------------------------------------- the world

    def _camera_matrix(self) -> np.ndarray:
        return camera_matrix(tuple(self.pose), self.camera_height)

    def _advance(self) -> None:
        """Drive toward an outstanding goal, as the fake navigator does."""
        with self._lock:
            if self._goal is None or self._state != "active":
                return
            delta = self._goal - self.pose[:2]
            dist = float(np.linalg.norm(delta))
            if dist <= float(self.args.goal_tolerance_m):
                self._state = "succeeded"
                self._goal = None
                return
            self.pose[2] = math.atan2(delta[1], delta[0])
            stepped = min(float(self.args.nav_speed) * float(self.args.tick_s), dist)
            self.pose[:2] += delta / dist * stepped
            self._clamp()

    def _clamp(self) -> None:
        self.pose[:2] = np.clip(self.pose[:2], -ROOM_HALF_M + 0.3, ROOM_HALF_M - 0.3)

    # -------------------------------------------------------------------- ops

    def ping(self) -> dict:
        return {"node": "osg_bridge", "frames": self._seq, "ros_time": time.time(),
                "has_camera_info": True, "loopback": True}

    def get_frame(self, after_seq: int = -1, timeout_s: float = 5.0):
        """Render the room as it is now.

        No waiting: this camera samples when it is asked rather than streaming,
        so every frame already postdates every command that preceded the call
        and `after_seq` is satisfied by construction. The real bridge has to
        wait because its frames arrive on their own schedule.
        """
        self._advance()
        self._maybe_switch_floor()
        with self._lock:
            T = self._camera_matrix()
            self._seq += 1
            seq = self._seq
            stamp = time.time()
        depth = render_depth(self.width, self.height, self.K, T)
        return {
            "seq": seq, "stamp": stamp, "rgb": render_rgb(depth),
            "depth": (depth * 1000.0).astype(np.uint16), "depth_encoding": "16UC1",
            "K": self.K.flatten(), "width": self.width, "height": self.height,
            "T_map_cam": T, "frame_id": "camera_color_optical_frame",
        }

    def send_goal(self, x: float, y: float, yaw: float, frame_id: str) -> dict:
        with self._lock:
            self._goal_id += 1
            self.last_goal = {"x": float(x), "y": float(y), "yaw": float(yaw),
                              "frame_id": str(frame_id), "goal_id": self._goal_id}
            if self.args.abort_goals:
                self._state, self._goal = "aborted", None
            else:
                self._state = "active"
                self._goal = np.array([float(x), float(y)])
            return {"goal_id": self._goal_id}

    def nav_status(self) -> dict:
        self._advance()
        with self._lock:
            remaining = (float(np.linalg.norm(self._goal - self.pose[:2]))
                         if self._goal is not None else float("nan"))
            return {"state": self._state, "goal_id": self._goal_id,
                    "distance_remaining": remaining}

    def cancel(self) -> dict:
        with self._lock:
            had = self._goal is not None
            self._goal = None
            if self._state == "active":
                self._state = "canceled"
            return {"cancelled": had}

    def execute(self, action: str, forward_m: float, turn_deg: float) -> dict:
        # Driving the base by hand ends any goal, the way the real fake lets
        # the navigator win while one is outstanding (fake_robot._integrate).
        # Without this the two fakes disagree about who owns the wheels, which
        # is the one property a second fake exists to cross-check.
        self.cancel()
        with self._lock:
            if action == "move_forward":
                self.pose[:2] += np.array([math.cos(self.pose[2]),
                                           math.sin(self.pose[2])]) * float(forward_m)
                self._clamp()
                achieved = float(forward_m)
            elif action in ("turn_left", "turn_right"):
                sign = 1.0 if action == "turn_left" else -1.0
                achieved = sign * math.radians(float(turn_deg))
                self.pose[2] += achieved
            else:
                raise ValueError(f"cannot execute {action!r} on the base")
            return {"achieved": achieved, "timed_out": False}

    def look(self, tilt_delta_deg: float) -> dict:
        self.tilt_deg += float(tilt_delta_deg)
        return {"tilt_deg": self.tilt_deg}

    def _maybe_switch_floor(self) -> None:
        if (self.args.floor_switch_after > 0 and not self._floor_sent
                and time.time() - self._started >= self.args.floor_switch_after):
            self._floor_sent = True
            with self._lock:
                self._floor = 1

    def pop_floor_switch(self) -> dict:
        self._maybe_switch_floor()
        with self._lock:
            floor, self._floor = self._floor, None
        return {"floor": floor}

    def get_last_goal(self):
        return self.last_goal

    def handlers(self) -> dict:
        return {
            "ping": self.ping, "get_frame": self.get_frame,
            "send_goal": self.send_goal, "nav_status": self.nav_status,
            "cancel": self.cancel, "execute": self.execute, "look": self.look,
            "pop_floor_switch": self.pop_floor_switch, "last_goal": self.get_last_goal,
                    "publish_image": lambda topic, image, encoding="bgr8": {"published": True},
            "publish_markers": lambda topic, markers, frame_id="map": {"published": len(markers)},
            "publish_grid": lambda topic, grid, origin_xy, resolution, frame_id="map", z=0.0: {"published": True},
        }


def serve(robot: LoopbackRobot, addr, authkey: bytes, ready=None) -> None:
    from multiprocessing.connection import Listener

    with Listener(addr, authkey=authkey) as listener:
        if ready is not None:
            ready.set()
        while True:
            conn = listener.accept()
            try:
                while serve_once(conn, robot.handlers()):
                    pass
            finally:
                conn.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bridge-addr", default="127.0.0.1:18765")
    p.add_argument("--authkey", default="osg-ros2")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--hfov-deg", type=float, default=69.0)
    p.add_argument("--camera-height", type=float, default=1.3)
    p.add_argument("--start-x", type=float, default=0.0)
    p.add_argument("--start-y", type=float, default=0.0)
    p.add_argument("--tick-s", type=float, default=0.05)
    p.add_argument("--nav-speed", type=float, default=1.5)
    p.add_argument("--goal-tolerance-m", type=float, default=0.2)
    p.add_argument("--abort-goals", action="store_true")
    p.add_argument("--floor-switch-after", type=float, default=0.0)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    robot = LoopbackRobot(args)
    print(f"loopback robot on {args.bridge_addr} (no ROS)", flush=True)
    try:
        serve(robot, parse_addr(args.bridge_addr), args.authkey.encode("utf-8"))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
