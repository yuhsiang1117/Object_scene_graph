"""The robot, wearing the env contract the episode loop expects.

`eval/episode.py` asks an env for exactly four things -- `reset()`, `step(action)`,
`episode_over`, and frames that are `FrameData` -- and every one of them has an
answer on a real robot. That is why the ROS layer needed no parallel control
loop: the same `run_episode`, the same `NavAgent`, the same scene graph, with
this class where habitat used to be.

Two things are genuinely different and both live here.

THE BASE HAS ONE OWNER. In habitat every tick is one discrete action, so the
question never comes up. Here the FSM sometimes issues its own action (the
opening 360-degree scan, the escape window, a look_down near stairs) while Nav2
may still be driving toward a goal. `step` asks the driver which of them acted
this tick, and cancels the outstanding goal before executing anything the FSM
decided -- otherwise the two fight over the wheels and neither arrives.

THE VERTICAL IS SYNTHETIC. Nav2's `map` is 2D: both storeys report the same z,
while `floor_of_height`, the costmap's obstacle band and the LLM prompt's floor
ordering all separate storeys by height. So the operator's floor switch IS the
height sensor -- the pose is lifted by `key * floor.virtual_storey_m` -- and
every height consumer downstream sees the geometry it was calibrated on.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np

from ..core.types import FrameData
from ..ros2 import frames as F
from ..ros2.transport import BridgeUnavailable, Transport

# The actions the FSM can emit. `stop` ends the episode; the look pair moves the
# head; the rest are metered on the base.
_LOOKS = {"look_up": 1.0, "look_down": -1.0}


class Ros2Env:
    def __init__(self, cfg, transport: Optional[Transport] = None) -> None:
        self.cfg = cfg
        self.ros = cfg.ros2
        # Injected by the tests; built here in a real run.
        self.transport = transport if transport is not None else Transport(
            cfg.ros2, connect_timeout_s=float(cfg.ros2.connect_timeout_s))
        self.driver = None
        # Called with the new storey key when the operator switches floors.
        # `scripts/run_robot.py` points this at `agent.floors.request_floor`.
        self.on_floor_switch: Optional[Callable[[int], None]] = None
        self.floor_key = 0
        self.floor_switches: list = []
        self._last_seq = -1
        self._frame_id = 0
        self._steps = 0
        self._stopped = False
        self._last_payload: Optional[dict] = None
        self._episode_id = f"{self.ros.map_tag}_{int(time.time())}"

    def wait_for_floor_switch(self, on_frame=None) -> FrameData:
        """Hold before the first step until the operator declares a storey.

        Fresh frames keep coming (so RViz shows the restored graph and the
        camera), the switch topic is polled on each, no command reaches the
        base and no step is counted. For resuming a demo whose search pass
        died after the carry: the run restores the map on floor 0, the person
        publishes /osg/floor 1, and the storey lift happens on the first step
        -- on screen, as it would have. Returns the frame the switch arrived on.
        """
        print(f"[robot] holding on floor {self.floor_key}: waiting for /osg/floor "
              f"(bash scripts/ros2/switch_floor.sh N ...)", flush=True)
        while not self.floor_switches:
            time.sleep(float(self.ros.step_period_s))
            frame = self._next_frame(fresh=True)
            if on_frame is not None:
                on_frame(frame)
        print(f"[robot] floor {self.floor_key} declared; starting", flush=True)
        return frame

    def attach_driver(self, driver) -> None:
        """The mover, so `step` can ask who owns the base this tick."""
        self.driver = driver

    # ---------------------------------------------------------------- episode

    def reset(self) -> FrameData:
        self.transport.ping()
        self._frame_id = 0
        self._steps = 0
        self._stopped = False
        self._last_seq = -1
        # A second run on the same env starts on the ground floor, not on the
        # storey the previous one happened to end on.
        self.floor_key = 0
        self.floor_switches = []
        # Drain a switch left over from before the run started, so run 2 does
        # not begin by switching to the floor run 1 ended on.
        self.transport.pop_floor_switch()
        return self._next_frame(fresh=True)

    def step(self, action: str) -> FrameData:
        """Execute one control decision on the robot.

        See the module docstring: the branch here is about who owns the base,
        not about what the action means.
        """
        self._steps += 1
        stepped, goal_active = (
            self.driver.consume_tick() if self.driver is not None else (False, False))
        # `stepped` says the mover ran this tick; it does NOT say the action
        # reaching here is the mover's. `NavAgent.act` post-processes after the
        # mover returns -- the escape window rewrites a long run of forwards
        # into a turn (agent/nav_agent.py:682) -- and an action the FSM
        # substituted has to take the base like any other FSM action. Asking
        # the driver what it actually returned is the only way to tell them
        # apart, and getting it wrong drops the override silently.
        driver_action = getattr(self.driver, "last_action", None)
        mover_owns_tick = stepped and goal_active and action == driver_action

        if action == "stop":
            self._cancel()
            self._stopped = True
            return self._next_frame(fresh=False)

        if mover_owns_tick:
            # Nav2 is driving. The action is `Nav2Driver`'s placeholder and
            # executing it would fight the navigator for the wheels; the tick
            # is a pause to let the base make progress and look again.
            time.sleep(float(self.ros.step_period_s))
            return self._next_frame(fresh=False)

        # The FSM is steering. Take the base back first.
        if goal_active:
            self._cancel()
        if action == "wait":
            # Standing at the stairs for the carry (agent/state.py WAIT_ACTION):
            # no wheel command, one tick's pause, and a fresh look.
            time.sleep(float(self.ros.step_period_s))
            return self._next_frame(fresh=True)
        if action in _LOOKS:
            self.transport.look(_LOOKS[action] * float(self.ros.look_step_deg))
        else:
            self.transport.execute(
                action, float(self.cfg.agent.forward_m), float(self.cfg.agent.turn_deg))
        # Strictly newer than the frame the decision was made on: acting on a
        # view from before the base moved is the failure that looks like a bad
        # planner.
        return self._next_frame(fresh=True)

    @property
    def episode_over(self) -> bool:
        return self._stopped or self._steps >= int(self.cfg.agent.max_steps)

    @property
    def current_episode(self):
        """There is no episode dataset on a robot, but the record wants one."""
        return SimpleNamespace(
            episode_id=self._episode_id, scene_id=str(self.ros.map_tag),
            object_category=str(self.ros.target), start_position=None,
            start_rotation=None, goals=[], info={},
        )

    def target_category(self) -> str:
        return str(self.ros.target)

    def metrics(self) -> dict:
        """Nothing here is scored. A real deployment has no ground truth, and
        reporting a 0.0 SR as though it were measured would put a fabricated
        number in `summary.json` beside the real ones.

        NaN rather than None, because `eval/record.py` calls `float()` on all
        three and shared eval code should not grow a robot-only branch.
        `scripts/run_robot.py` turns the non-finite values into JSON `null` on
        the way to disk, where a bare `NaN` token would be unreadable by any
        parser outside Python.
        """
        return {"success": float("nan"), "spl": float("nan"),
                "distance_to_goal": float("nan"), "steps": self._steps}

    def attempt_scored(self, frame, cfg) -> bool:
        """Did this STOP find the object? Unknowable without ground truth.

        Answering at all is what keeps `eval/attempts.py` off `env.env.sim`,
        which does not exist here. False means a multi-attempt protocol would
        keep searching; with `eval.attempts=1` (the robot presets) the STOP
        simply ends the run and the operator judges it.
        """
        return False

    def episode_metadata(self) -> dict:
        return {"map_tag": str(self.ros.map_tag), "map_mode": str(self.ros.map_mode),
                "floor_switches": list(self.floor_switches),
                "target": str(self.ros.target)}

    def close(self) -> None:
        try:
            self._cancel()
        finally:
            self.transport.close()

    # -------------------------------------------------------------- internals

    def _cancel(self) -> None:
        self.transport.cancel()
        if self.driver is not None:
            self.driver.mark_cancelled()

    def _next_frame(self, fresh: bool) -> FrameData:
        timeout = float(self.ros.step_period_s) + float(self.ros.frame_timeout_s)
        patience = float(getattr(self.ros, "frame_patience_s", 0.0) or 0.0)
        waited = 0.0
        while True:
            try:
                payload = self.transport.get_frame(
                    after_seq=self._last_seq if fresh else -1, timeout_s=timeout)
                break
            except BridgeUnavailable:
                # No frame is the camera or `map -> camera` gone. On the
                # Stretch that is also what a carry to the other storey
                # looks like -- slam_toolbox relaunched there -- so wait it
                # out, saying so, before calling the run dead.
                waited += timeout
                if waited >= patience:
                    raise
                print(f"[robot] no camera frame for {waited:.0f}s (camera or map->camera TF "
                      f"missing; carrying?) -- waiting up to {patience:.0f}s", flush=True)
        if waited:
            print(f"[robot] frames are back after {waited:.0f}s", flush=True)
        self._last_seq = int(payload.get("seq", self._last_seq + 1))
        self._poll_floor_switch()
        return self._to_frame(payload)

    def _poll_floor_switch(self) -> None:
        key = self.transport.pop_floor_switch()
        if key is None or int(key) == self.floor_key:
            return
        self.floor_key = int(key)
        self.floor_switches.append((self._steps, self.floor_key))
        # The map under the pursuit is about to change, so the goal chosen on
        # the old storey is no longer meaningful.
        self._cancel()
        if self.on_floor_switch is not None:
            self.on_floor_switch(self.floor_key)

    def _to_frame(self, payload: dict) -> FrameData:
        self._last_payload = payload
        depth = F.depth_to_metres(
            payload["depth"], str(payload.get("depth_encoding", "16UC1")),
            float(self.ros.depth_scale), max_m=float(self.cfg.eval.depth_max_m))
        rgb = np.ascontiguousarray(payload["rgb"][..., :3])
        intr = F.intrinsics_from_camera_info(
            payload["K"], int(payload["width"]), int(payload["height"]))
        T_wc = F.ros_pose_to_pipeline(payload["T_map_cam"])
        rgb, depth, intr, T_wc = F.rotate_frame(
            rgb, depth, intr, T_wc, float(self.ros.rotate_deg))
        # The switch is the height sensor: see the module docstring.
        T_wc = T_wc.copy()
        T_wc[1, 3] += self.floor_key * float(self.cfg.floor.virtual_storey_m)
        self._frame_id += 1
        return FrameData(
            frame_id=self._frame_id, rgb=rgb, depth=depth, T_wc=T_wc,
            intrinsics=intr, timestamp=float(payload.get("stamp", 0.0)),
        )
