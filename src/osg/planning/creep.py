"""The approach's last metre, walked with a heading.

`PointNavDriver`'s creep (ASCENT's terminal behaviour, ascent_policy.py:920-927)
forces MOVE_FORWARD inside `pointnav_approach_creep_m`, whatever the goal's
bearing: the network's own stop radius (0.9 m) is outside the success distance,
so inside it nothing else moves the agent. Forward is only toward the goal when
the agent already faces it. The approach's closing walk
(`agent.approach_close_last_metre_m`) picks the clear cell nearest the track,
which beside a table is off to one side, and the arrival turn has just pointed
the camera at the object -- so the press goes into the furniture. Measured on
Benchmark v2 seed 42 (mfv2_osg_climb): 00862 cross_anchor_01 banana pressed for
ten steps at 1.07 m depth and stopped 1.15 m from the object, 0.37 m from a
viewpoint 0.8 m from it; 00873's pitcher stopped at 1.10 m in all four OSG arms.

`SteeredCreep` replaces that forward: turn toward the goal when it is more than
`steer_deg` off the heading, step forward otherwise, and when a forward is
refused (`eval.allow_sliding: false`: a step that touches an edge goes nowhere)
sweep headings either side of the goal's bearing, nearest first. Its stall test
counts FORWARDS that fail to close on the goal, so the turns of a sweep do not
end it; `stall_steps` of them and the agent is as close as the geometry allows.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

FORWARD = "move_forward"
TURN_LEFT = "turn_left"
TURN_RIGHT = "turn_right"


class SteeredCreep:
    """`step(agent_xy, rho, theta, last_action) -> (action | None, reason)`.

    `rho`, `theta` are the goal's polar offset in the agent's frame with theta
    positive to the LEFT (`pointnav_driver.rho_theta`). Reasons: `creep_turn`,
    `creep_sweep`, `creep`, `creep_stalled` (action None).
    """

    SWEEP = (1, -1, 2, -2, 3, -3)

    def __init__(self, turn_deg: float = 30.0, steer_deg: float = 20.0,
                 stall_steps: int = 6, stall_eps: float = 0.05,
                 goal_change_m: float = 0.1) -> None:
        self.turn_deg = float(turn_deg)
        self.steer_deg = float(steer_deg)
        self.stall_steps = int(stall_steps)
        self.stall_eps = float(stall_eps)
        self.goal_change_m = float(goal_change_m)
        self.n_stalls = 0
        self.reset()

    def reset(self) -> None:
        self._goal: Optional[np.ndarray] = None
        self._last_xy: Optional[np.ndarray] = None
        self._queue: List[str] = []
        self._blocks = 0
        self._side = 1.0
        self._best_rho: Optional[float] = None
        self._idle_forwards = 0

    def step(self, goal_xy, agent_xy, rho: float, theta: float,
             last_action: Optional[str]) -> Tuple[Optional[str], str]:
        goal = np.asarray(goal_xy, dtype=float)
        agent_xy = np.asarray(agent_xy, dtype=float)
        if self._goal is None or float(np.linalg.norm(goal - self._goal)) > self.goal_change_m:
            self.reset()
            self._goal = goal.copy()
        moved = (float(np.linalg.norm(agent_xy - self._last_xy))
                 if self._last_xy is not None else 1.0)
        self._last_xy = agent_xy.copy()
        if last_action == FORWARD and self._best_rho is not None:
            # Judge the forward just taken: did it close on the goal?
            if rho < self._best_rho - self.stall_eps:
                self._idle_forwards = 0
            else:
                self._idle_forwards += 1
                if self._idle_forwards >= self.stall_steps:
                    self.n_stalls += 1
                    self._queue = []
                    return None, "creep_stalled"
        if self._best_rho is None or rho < self._best_rho:
            self._best_rho = float(rho)
        blocked = last_action == FORWARD and moved < 0.03
        if moved >= 0.05:
            self._blocks = 0
        if self._queue:
            return self._queue.pop(0), "creep_sweep"
        theta_deg = float(np.degrees(theta))
        if blocked:
            if self._blocks == 0:
                self._side = 1.0 if theta_deg >= 0.0 else -1.0
            self._blocks += 1
            k = (self._blocks - 1) % len(self.SWEEP)
            want = theta_deg + self.SWEEP[k] * self._side * self.turn_deg
            want = float((want + 180.0) % 360.0 - 180.0)
            n = int(round(abs(want) / self.turn_deg))
            if n == 0:
                return FORWARD, "creep_sweep"
            turn = TURN_LEFT if want > 0.0 else TURN_RIGHT
            self._queue = [turn] * (n - 1) + [FORWARD]
            return turn, "creep_sweep"
        if abs(theta_deg) > self.steer_deg:
            return (TURN_LEFT if theta_deg > 0.0 else TURN_RIGHT), "creep_turn"
        return FORWARD, "creep"
