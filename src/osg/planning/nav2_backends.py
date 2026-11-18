"""What actually drives, behind `Nav2Driver`: the robot, or habitat.

Two backends, one contract (`observe`, `send_goal`, `poll`, `cancel`). Having
the simulated one at all is what makes the ROS path debuggable: the goal
selection, the ownership handshake and the FSM's response to an aborted goal
can all be exercised in habitat, where there is ground truth and a repeatable
episode, before any of it meets a robot.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import FrameData
from ..mapping.costmap import PLANE
from ..ros2 import frames
from .nav2_driver import BackendStatus


class SimNav2Backend:
    """Nav2's semantics, played by habitat's navmesh follower.

    PRIVILEGED NAVIGATION, exactly as `agent.navigation=navmesh` is: it is
    given simulator geometry (`action_to_goal`), so an SR/SPL measured through
    this backend must never be compared against a sensor-only mover. It exists
    to test the ROS control flow, not to produce a benchmark number -- which is
    fair, because the robot's Nav2 is privileged in the same way for the same
    reason: it is driving on a map the robot was given.
    """

    def __init__(self, env, *, arrival_m: float = 0.25) -> None:
        self.env = env
        self.arrival_m = float(arrival_m)
        self._goal: Optional[np.ndarray] = None
        self._agent_xy = np.zeros(2)

    def observe(self, frame: FrameData) -> None:
        self._agent_xy = frame.camera_position[list(PLANE)].copy()

    def send_goal(self, goal_xy) -> None:
        self._goal = np.asarray(goal_xy, dtype=float).copy()

    def cancel(self) -> None:
        self._goal = None

    def poll(self) -> BackendStatus:
        if self._goal is None:
            return BackendStatus("idle")
        # No floor height: this mover is same-storey (see Nav2Driver). None
        # means "the agent's own floor", which is what a same-storey goal wants.
        action = self.env.action_to_goal(self._goal, None)
        rho = float(np.linalg.norm(self._goal - self._agent_xy))
        if action is not None:
            return BackendStatus("active", action=action, distance_remaining=rho)
        # The follower returned None: arrived, or it gave up. Nav2 distinguishes
        # those with SUCCEEDED and ABORTED, and so must this -- collapsing them
        # is the mistake `NavStep` exists to prevent.
        state = "succeeded" if rho <= self.arrival_m else "aborted"
        return BackendStatus(state, distance_remaining=rho)


class RosNav2Backend:
    """The real thing: `NavigateToPose` on the robot, over the bridge.

    The only place in the pipeline where a goal changes coordinate systems.
    Everything above this speaks the pipeline's world frame; the bridge and the
    robot speak ROS `map`, and `frames.py` is the single conversion between
    them.
    """

    def __init__(self, transport, ros_cfg) -> None:
        self.transport = transport
        self.cfg = ros_cfg
        self._agent_xy = np.zeros(2)
        # What was last handed to Nav2, in ROS coordinates. Read by
        # scripts/check_ros2_pipeline.py to close the conversion loop.
        self.last_sent_ros: Optional[dict] = None

    def observe(self, frame: FrameData) -> None:
        self._agent_xy = frame.camera_position[list(PLANE)].copy()

    def set_goal_yaw(self, goal_xy, yaw_ros: float) -> None:
        """Face `yaw_ros` (map frame, radians) on arrival at THIS goal instead
        of the direction of travel -- an operator waypoint carries a heading.
        Applies to the next send of a goal within 5 cm of `goal_xy`, once."""
        self._posed = (np.asarray(goal_xy, dtype=float).copy(), float(yaw_ros))

    def send_goal(self, goal_xy) -> None:
        x, y = frames.pipeline_xy_to_ros(goal_xy)
        yaw = frames.goal_yaw_ros(self._agent_xy, goal_xy)
        posed = getattr(self, "_posed", None)
        if posed is not None and float(np.linalg.norm(posed[0] - np.asarray(goal_xy, dtype=float))) <= 0.05:
            yaw = posed[1]
            self._posed = None
        frame_id = str(self.cfg.goal_frame)
        goal_id = self.transport.send_goal(x, y, yaw, frame_id)
        self.last_sent_ros = {"x": x, "y": y, "yaw": yaw, "frame_id": frame_id,
                              "goal_id": goal_id}

    def cancel(self) -> None:
        self.transport.cancel()

    def poll(self) -> BackendStatus:
        status = self.transport.nav_status()
        return BackendStatus(
            state=str(status.get("state", "idle")),
            distance_remaining=float(status.get("distance_remaining", float("nan"))),
        )
