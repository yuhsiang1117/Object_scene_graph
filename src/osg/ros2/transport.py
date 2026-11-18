"""The pipeline's end of the bridge socket.

Everything the robot can do or say, as ordinary blocking Python calls. This is
the seam the tests replace: `Ros2Env` and `RosNav2Backend` hold a `Transport`
and never import ROS, so a `FakeTransport` with these same methods exercises
the whole pipeline in the CPU suite (tests/unit/test_ros2_env.py).

Frames come back as the raw bridge payload -- ROS units, ROS frame -- because
`Ros2Env` is where a payload becomes a `FrameData`, and it needs the config to
do it (rotation, depth scale, the virtual storey offset).
"""
from __future__ import annotations

import time
import numpy as np
from typing import Optional

from .wire import RemoteError, call, parse_addr


class BridgeUnavailable(RuntimeError):
    """The bridge is not answering.

    Deliberately fatal, like `PerceptionUnavailable`: a run that cannot reach
    the robot is not a degraded run, and a neutral value here is an agent
    navigating a frozen frame.
    """


class Transport:
    def __init__(self, ros_cfg, *, connect_timeout_s: float = 30.0) -> None:
        self.cfg = ros_cfg
        self._addr = parse_addr(str(ros_cfg.bridge_addr))
        self._authkey = str(ros_cfg.authkey).encode("utf-8")
        self._conn = None
        self._connect_timeout_s = float(connect_timeout_s)

    # ------------------------------------------------------------ connection

    def connect(self) -> "Transport":
        from multiprocessing.connection import Client

        deadline = time.time() + self._connect_timeout_s
        last = None
        while time.time() < deadline:
            try:
                self._conn = Client(self._addr, authkey=self._authkey)
                return self
            except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
                last = exc
                time.sleep(0.5)
        raise BridgeUnavailable(
            f"no ROS bridge at {self.cfg.bridge_addr} after "
            f"{self._connect_timeout_s:.0f}s ({last}). Start it with "
            "`bash scripts/ros2/bridge.sh`."
        )

    @property
    def conn(self):
        if self._conn is None:
            self.connect()
        return self._conn

    def _call(self, op: str, **args):
        try:
            return call(self.conn, op, **args)
        except (EOFError, BrokenPipeError, ConnectionResetError) as exc:
            raise BridgeUnavailable(f"ROS bridge went away during {op!r}: {exc}") from exc

    # ------------------------------------------------------------------- ops

    def ping(self) -> dict:
        return self._call("ping")

    def get_frame(self, after_seq: int = -1, timeout_s: float = 5.0) -> dict:
        """The newest frame, waiting for one newer than `after_seq`.

        Latest-only, never a queue: the pipeline runs at whatever rate its
        perception allows and the camera at 30 Hz, so buffering would hand the
        agent an older and older view of the room. Raises rather than returning
        a stale frame on timeout -- a frozen frame is the failure that looks
        like a bad planner.
        """
        payload = self._call("get_frame", after_seq=int(after_seq),
                             timeout_s=float(timeout_s))
        if payload is None:
            raise BridgeUnavailable(
                f"no camera frame within {timeout_s:.1f}s. Check "
                f"{self.cfg.rgb_topic} / {self.cfg.depth_topic} and the "
                f"{self.cfg.map_frame} -> camera TF."
            )
        return payload

    def send_goal(self, x: float, y: float, yaw: float, frame_id: str) -> int:
        return int(self._call("send_goal", x=float(x), y=float(y), yaw=float(yaw),
                              frame_id=str(frame_id))["goal_id"])

    def nav_status(self) -> dict:
        return self._call("nav_status")

    def cancel(self) -> None:
        self._call("cancel")

    def execute(self, action: str, forward_m: float, turn_deg: float) -> dict:
        """Run one of the FSM's own discrete actions on the base.

        The pipeline's control loop still emits `move_forward` / `turn_left` in
        several states -- the opening 360-degree scan, the escape window, the
        close look -- and those are not navigation goals; handing them to Nav2
        as 0.25 m poses would invoke a global planner per step. The bridge
        meters them on cmd_vel against odometry instead.
        """
        return self._call("execute", action=str(action), forward_m=float(forward_m),
                          turn_deg=float(turn_deg))

    def look(self, tilt_delta_deg: float) -> dict:
        return self._call("look", tilt_delta_deg=float(tilt_delta_deg))

    def publish_markers(self, topic: str, markers: list, frame_id: str = "map") -> None:
        """Hand RViz geometry to the bridge. Plain dicts; see bridge publish_markers."""
        self._call("publish_markers", topic=str(topic), markers=list(markers),
                   frame_id=str(frame_id))

    def publish_grid(self, topic: str, grid, origin_xy, resolution: float,
                     frame_id: str = "map", z: float = 0.0) -> None:
        """Hand the pipeline's costmap to the bridge for the map window."""
        self._call("publish_grid", topic=str(topic),
                   grid=np.ascontiguousarray(grid, dtype=np.int8),
                   origin_xy=[float(v) for v in origin_xy], resolution=float(resolution),
                   frame_id=str(frame_id), z=float(z))

    def publish_image(self, topic: str, image, encoding: str = "bgr8") -> None:
        """Hand an image to the bridge for RViz. Debug only; see DebugStream."""
        self._call("publish_image", topic=str(topic), image=np.ascontiguousarray(image),
                   encoding=str(encoding))

    def pop_floor_switch(self) -> Optional[int]:
        """The operator's floor switch, if one arrived since the last call."""
        floor = self._call("pop_floor_switch").get("floor")
        return None if floor is None else int(floor)

    def last_goal(self) -> Optional[dict]:
        """What the bridge last handed to Nav2. For the pipeline check only."""
        return self._call("last_goal")

    def close(self) -> None:
        if self._conn is None:
            return
        try:
            call(self._conn, "shutdown")
        except (RemoteError, EOFError, BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            try:
                self._conn.close()
            finally:
                self._conn = None
