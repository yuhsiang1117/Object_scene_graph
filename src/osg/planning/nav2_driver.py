"""The mover that hands the goal to somebody else.

Every other mover in this repo converts a metric goal into the next discrete
action: the costmap planner plans and steers, PointNav infers one, the navmesh
follower is told one by habitat. On the Stretch none of that is wanted -- the
robot already runs Nav2, which has the robot's own footprint, its local
costmap, its recovery behaviours and a controller tuned for the base. The
pipeline's job ends at "go here".

So `Nav2Driver` is `PointNavDriver`-shaped (`planning/pointnav_driver.py:80`)
but decides nothing: `step(goal_xy)` posts the goal and reports what the
navigator says about it. Being that shape is what lets it slot into
`NavAgent`'s existing `pointnav` branch (`nav_agent.py:1574`, `:2010`) with no
change to the FSM at all -- and it is the reason the `NavStep` reason vocabulary
is reused rather than extended: `aborted` is the pipeline's `policy_stop`, which
the FSM already knows how to treat as a blocked frontier.

The backend is what differs between the robot and habitat, so the state machine
above it is tested once, against a fake (tests/unit/test_nav2_driver.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.types import FrameData
from ..mapping.costmap import PLANE
from .pointnav_driver import NavStep

# Nav2 is still working on it. The pipeline's control loop has to return SOME
# action per tick; `Ros2Env` recognises that the driver owns this tick and does
# not execute the placeholder (sim/ros2_env.py).
DRIVING = "move_forward"


@dataclass
class BackendStatus:
    """What the navigator says about the goal it was last given.

    `action` is only ever set by the simulated backend, where "still driving"
    has a concrete discrete action attached. On the robot it is None: the base
    is already moving and nothing is left for the caller to execute.
    """

    state: str  # idle | active | succeeded | aborted | rejected | canceled
    action: Optional[str] = None
    distance_remaining: float = float("nan")


class Nav2Driver:
    """`goal_xy -> NavStep`, with the driving done by a Nav2-shaped backend.

    SAME STOREY ONLY. A goal carries no height: Nav2 navigates one 2D map, and
    on the robot the storey is the operator's switch rather than a height at
    all (`floor.source=external`). Reaching another floor is not something this
    mover can do, which is why the robot presets turn the climb machinery off.
    """

    def __init__(self, backend, *, stop_radius: float = 0.9,
                 goal_resend_m: float = 0.5, timeout_steps: int = 0) -> None:
        self.backend = backend
        # Read by `nav_agent._frontier_reach_m` (:373) and the climb (:137) to
        # decide how close counts as having reached a frontier, so it has to
        # mean the same thing it means for PointNav.
        self.stop_radius = float(stop_radius)
        self.goal_resend_m = float(goal_resend_m)
        # Control ticks a single goal may stay `active` before it is treated as
        # refused. Nav2's recovery behaviours can spin on an unreachable goal
        # for a very long time without ever aborting, and a pursuit that never
        # ends is a whole run spent on one frontier. 0 disables.
        self.timeout_steps = int(timeout_steps)
        self.reset()

    def reset(self) -> None:
        """Per EPISODE. The runner builds one mover for the whole run
        (pipeline/components.py), so counters left standing here would make
        every episode's record report the run total -- the same mistake
        `scorer_before`/`verifier_before` exist to prevent in eval/record.py."""
        self._last_goal: Optional[np.ndarray] = None
        self._agent_xy: Optional[np.ndarray] = None
        self.goal_active = False
        self.stepped_this_tick = False
        # What `step` last returned, so `sim/ros2_env.py` can tell the mover's
        # own action from one the FSM substituted for it afterwards.
        self.last_action: Optional[str] = None
        self._ticks_on_goal = 0
        self.n_goals_sent = 0
        self.n_aborts = 0
        self.n_resets = 0
        self.n_timeouts = 0
        self.last_rho = float("nan")
        self.backend.cancel()

    # ------------------------------------------------------------- per step

    def observe(self, frame: FrameData) -> None:
        """Cache this step's pose. Called once per step by `NavAgent.act`."""
        self._agent_xy = frame.camera_position[list(PLANE)].copy()
        self.stepped_this_tick = False
        self.backend.observe(frame)

    def step(self, goal_xy, *, stop_radius: Optional[float] = None,
             creep_below: float = 0.0) -> NavStep:
        """Post `goal_xy` to the navigator and report where the pursuit stands.

        `creep_below` is accepted and ignored. It exists because PointNav's own
        stop radius (0.9 m) is far outside the success distance, so the caller
        has to force the last metre itself; Nav2 has the robot's footprint and
        a goal tolerance of its own, and blind-forwarding into furniture is
        exactly what it is there to prevent. The terminal rule that ends an
        approach still belongs to the caller either way.
        """
        if self._agent_xy is None:
            raise RuntimeError("Nav2Driver.observe(frame) must be called each step")
        self.stepped_this_tick = True
        goal = np.asarray(goal_xy, dtype=float)
        radius = self.stop_radius if stop_radius is None else float(stop_radius)
        rho = float(np.linalg.norm(goal - self._agent_xy))
        self.last_rho = rho

        # Arrival first, as in PointNavDriver: a creep or a resend that runs
        # before the radius test swallows the arrival entirely.
        if radius > 0.0 and rho < radius:
            self._cancel_if_active()
            return self._report(NavStep(None, "arrived"))

        first = self._last_goal is None
        moved = first or float(np.linalg.norm(goal - self._last_goal)) > self.goal_resend_m
        if moved or not self.goal_active:
            # A moved goal preempts the old one inside Nav2; resending only on
            # a real move keeps a frontier goal that jitters by a cell from
            # restarting the global planner every step.
            self.backend.send_goal(goal)
            self._last_goal = goal.copy()
            self.goal_active = True
            self._ticks_on_goal = 0
            self.n_goals_sent += 1
            if moved and not first:
                # A pursuit that changed its mind, not one that started. Read
                # back as `pointnav_resets`, where it prices how often the goal
                # is moving out from under the navigator.
                self.n_resets += 1

        status = self.backend.poll()
        if status.state == "active":
            self._ticks_on_goal += 1
            if 0 < self.timeout_steps <= self._ticks_on_goal:
                # Still "active" long past any reasonable leg. Nav2 can spin in
                # its recovery behaviours indefinitely without ever reporting
                # ABORTED, and the pipeline has a frontier it could retire
                # instead of spending the rest of the run here.
                self._cancel_if_active()
                self.n_timeouts += 1
                self.n_aborts += 1
                return self._report(NavStep(None, "policy_stop"))
            return self._report(NavStep(status.action or DRIVING, "moving"))
        if status.state == "succeeded":
            self.goal_active = False
            return self._report(NavStep(None, "arrived"))
        if status.state in ("aborted", "rejected"):
            # Nav2 has given up: no plan, or the recoveries ran out. That is a
            # statement about the goal, which is what `policy_stop` means to
            # the FSM -- with `agent.pointnav_stop_means_blocked` it retires the
            # frontier instead of pressing into it.
            self.goal_active = False
            self.n_aborts += 1
            return self._report(NavStep(None, "policy_stop"))
        # idle or canceled: somebody else took the base (a discrete action the
        # FSM issued). Re-post on the next tick rather than reporting failure.
        self.goal_active = False
        return self._report(NavStep(DRIVING, "moving"))

    def _report(self, step: NavStep) -> NavStep:
        self.last_action = step.action
        return step

    def __call__(self, goal_xy, *, stop_radius: Optional[float] = None,
                 creep_below: float = 0.0) -> Optional[str]:
        return self.step(goal_xy, stop_radius=stop_radius, creep_below=creep_below).action

    # -------------------------------------------------------- goal ownership

    def consume_tick(self):
        """`(the driver acted this tick, a goal is outstanding)`, and clear.

        The base has one owner. `NavAgent` issues its own discrete actions in
        several states -- the opening scan, the escape window, a look_down --
        without consulting any mover, and those must not be executed while Nav2
        is still driving toward a goal. `Ros2Env.step` asks this, and cancels
        when the answer is "the FSM is steering now".
        """
        stepped, self.stepped_this_tick = self.stepped_this_tick, False
        return stepped, self.goal_active

    def mark_cancelled(self) -> None:
        """The env cancelled the goal out from under the driver."""
        self.goal_active = False
        self._last_goal = None

    def cancel(self) -> None:
        self._cancel_if_active()
        self._last_goal = None

    def _cancel_if_active(self) -> None:
        if self.goal_active:
            self.backend.cancel()
            self.goal_active = False
            self._ticks_on_goal = 0
