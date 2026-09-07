"""ASCENT's mover: a frozen PointNav policy driving to a ground-plane goal.

This is the sensor-only replacement for habitat's `ShortestPathFollower`. It
consumes only depth and the agent's own pose, so it has no knowledge of
geometry it has not seen -- which is the entire point (docs/AB_RESULTS.md, S8).

Ported from `ascent/ascent_policy.py:837 _pointnav` / `:876 _navigate`, whose
wrapper is `ascent/pointnav_policy.py:51 WrappedPointNavResNetPolicy`.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..core.types import FrameData
from ..mapping.costmap import PLANE
from .controller import agent_heading
from .pointnav import ACTION_NAMES, load_pointnav_policy

HIDDEN_SIZE = 512


class NavStep(NamedTuple):
    """One mover decision, and why.

    `action` is None whenever there is nothing to execute; `reason` says which
    kind of nothing:

      arrived      inside the goal radius -- the pursuit succeeded
      policy_stop  the network emitted STOP while still short of the goal
      creep        forced forward inside the terminal creep radius
      moving       an ordinary action

    Collapsing `arrived` and `policy_stop` into a bare None is what made the
    first port retire a frontier on every spurious network STOP.
    """

    action: Optional[str]
    reason: str


def to_ccw_frame(xy: np.ndarray) -> np.ndarray:
    """OSG's ground plane -> a counter-clockwise-positive 2D frame.

    OSG's plane is `PLANE = (0, 2)` = world (x, z), and in it `turn_left`
    DECREASES `agent_heading` (planning/controller.py:82; asserted by
    tests/unit/test_controller.py -- facing +x, a waypoint at +z needs
    TURN_RIGHT). `rho_theta` below, like every pointgoal sensor, defines theta
    as radians to turn LEFT. Flipping the second axis makes the frame
    CCW-positive so the two agree.

    ASCENT does the same thing to its GPS reading, one line earlier in the
    pipeline: `camera_position = np.array([x, -y, h])` (ascent_policy.py:234).

    Get this wrong and the agent mirrors every turn while still producing
    perfectly plausible-looking actions -- hence a unit test, not a comment.
    """
    xy = np.asarray(xy, dtype=float)
    return np.array([xy[0], -xy[1]])


def rho_theta(pos: np.ndarray, heading: float, goal: np.ndarray) -> Tuple[float, float]:
    """Polar offset of `goal` in the agent's frame, CCW-positive.

    Port of `vlfm/utils/geometry_utils.py:9`. Both `pos` and `goal` must
    already be in the CCW frame, and `heading` measured in it.
    """
    c, s = np.cos(-heading), np.sin(-heading)
    delta = np.asarray(goal, dtype=float) - np.asarray(pos, dtype=float)
    local = np.array([c * delta[0] - s * delta[1], s * delta[0] + c * delta[1]])
    return float(np.linalg.norm(local)), float(np.arctan2(local[1], local[0]))


class PointNavDriver:
    """`goal_xy -> next action`, a drop-in for `HabitatObjectNavEnv.action_to_goal`.

    `observe(frame)` once per step, then call as many times as the FSM needs;
    the network runs at most once per `__call__`.
    """

    def __init__(
        self,
        weights_path: str,
        *,
        stop_radius: float = 0.9,
        depth_shape: Tuple[int, int] = (224, 224),
        depth_min_m: float = 0.5,
        depth_max_m: float = 5.0,
        goal_change_m: float = 0.1,
        device: Optional[str] = None,
    ) -> None:
        self.stop_radius = float(stop_radius)
        self.depth_shape = (int(depth_shape[0]), int(depth_shape[1]))
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.goal_change_m = float(goal_change_m)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.policy = load_pointnav_policy(weights_path).to(self.device)
        self._n_layers = self.policy.num_recurrent_layers
        # Diagnostics, read by the debug video / episode log.
        self.last_rho: Optional[float] = None
        self.last_theta: Optional[float] = None
        self._depth: Optional[torch.Tensor] = None
        self._pos_ccw: Optional[np.ndarray] = None
        self._heading_ccw: float = 0.0
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """New episode: clear the recurrent state and forget the last goal."""
        self._hidden = torch.zeros(1, self._n_layers, HIDDEN_SIZE, device=self.device)
        self._prev_action = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self._last_goal: Optional[np.ndarray] = None
        # How often the recurrent state was wiped because the goal moved. A
        # policy reset every step is a policy with no memory of the corner it
        # is standing in.
        self.n_resets = 0
        self._started = False

    def _reset_recurrent(self) -> None:
        """Goal changed: the LSTM's memory is about a route to somewhere else.

        ASCENT resets on any goal move over 0.1 m and passes masks=0 for that
        one step (ascent_policy.py:849-853).
        """
        self._hidden = torch.zeros_like(self._hidden)
        self._prev_action = torch.zeros_like(self._prev_action)
        self._started = False

    # ------------------------------------------------------------- per step

    def observe(self, frame: FrameData) -> None:
        """Cache this step's depth and pose. Call once per step, before acting."""
        self._depth = self._prepare_depth(frame.depth)
        self._pos_ccw = to_ccw_frame(frame.camera_position[list(PLANE)])
        self._heading_ccw = -agent_heading(frame.T_wc)

    def _prepare_depth(self, depth: np.ndarray) -> torch.Tensor:
        """Metres -> the normalised 224x224 image the policy was trained on.

        `sim/habitat_env.py` sets `normalize_depth = False`, so OSG's depth is
        in metres (already clipped to [min, max] by habitat itself,
        habitat_simulator.py:180). PointNav consumes the normalised image, so
        undo the difference here rather than changing the sensor -- the costmap
        and the object layer both want metres.

        `mode="area"` on the torch side, not cv2.INTER_AREA: this feeds a
        frozen network, and VLFM resizes with torch's area interpolation
        (`image_resize(..., interpolation_mode="area")`).
        """
        d = np.clip(np.asarray(depth, dtype=np.float32), self.depth_min_m, self.depth_max_m)
        d = (d - self.depth_min_m) / (self.depth_max_m - self.depth_min_m)
        t = torch.from_numpy(d).to(self.device)[None, None]  # (1, 1, H, W)
        t = F.interpolate(t, size=self.depth_shape, mode="area")
        return t.permute(0, 2, 3, 1).contiguous()  # (1, H, W, 1), channels last

    def step(
        self,
        goal_xy: np.ndarray,
        *,
        stop_radius: Optional[float] = None,
        creep_below: float = 0.0,
    ) -> "NavStep":
        """Next action toward `goal_xy`, with the REASON attached.

        The reason matters because "I am there" and "the network gave up" call
        for opposite responses, and a bare None cannot tell them apart -- they
        met exactly at `rho == stop_radius`, since the caller disambiguated them
        by comparing against `_frontier_reach_m`, which is pinned to that same
        radius. See `NavStep`.

        `creep_below` ports ASCENT's terminal behaviour (ascent_policy.py:920-927):
        inside that radius it stops asking the network and forces MOVE_FORWARD,
        because the policy's own stop radius (0.9 m) is far outside the success
        distance and would strand the approach. The caller's terminal rule is
        what actually ends an approach.
        """
        if self._depth is None or self._pos_ccw is None:
            raise RuntimeError("PointNavDriver.observe(frame) must be called each step")
        goal = to_ccw_frame(np.asarray(goal_xy, dtype=float))
        if self._last_goal is None or np.linalg.norm(goal - self._last_goal) > self.goal_change_m:
            self._reset_recurrent()
            self.n_resets += 1
        self._last_goal = goal

        rho, theta = rho_theta(self._pos_ccw, self._heading_ccw, goal)
        self.last_rho, self.last_theta = rho, theta

        # Arrival is tested BEFORE the creep, or the creep swallows it: with
        # `creep_below` at 1.0 m and the radius at 0.9 m, every approach inside
        # a metre force-forwarded forever and the driver could never report
        # arriving. That is why the pointnav arm produced `path_consumed` zero
        # times against the navmesh arm's 46 of 100 -- the navigator had no way
        # to say "you are there".
        radius = self.stop_radius if stop_radius is None else float(stop_radius)
        if rho < radius:
            return NavStep(None, "arrived")
        if creep_below > 0.0 and rho < creep_below:
            return NavStep("move_forward", "creep")

        obs = {
            "depth": self._depth,
            "pointgoal_with_gps_compass": torch.tensor(
                [[rho, theta]], dtype=torch.float32, device=self.device
            ),
        }
        masks = torch.tensor([[self._started]], dtype=torch.bool, device=self.device)
        action, self._hidden = self.policy.act(
            obs, self._hidden, self._prev_action, masks, deterministic=True
        )
        self._prev_action = action.to(dtype=torch.long)
        self._started = True

        idx = int(action.item())
        if idx == 0:
            # The network called STOP while still outside the goal radius.
            #
            # ASCENT treats this as noise, not as a verdict: for an explore
            # frontier it overwrites the action with MOVE_FORWARD and keeps the
            # same target (ascent_policy.py:705-711). It disables the target on a
            # network STOP only in the two STAIR paths (:810-815, :1011-1014).
            # An earlier comment here cited :708-711 as authority for retiring
            # the frontier, which is the opposite of what those lines do.
            #
            # Reporting the reason rather than acting on it keeps that policy
            # decision with the caller, which is the only place that knows
            # whether this is a frontier or a staircase.
            return NavStep(None, "policy_stop")
        return NavStep(ACTION_NAMES[idx], "moving")

    def __call__(
        self,
        goal_xy: np.ndarray,
        *,
        stop_radius: Optional[float] = None,
        creep_below: float = 0.0,
    ) -> Optional[str]:
        """Action-only form, matching `HabitatObjectNavEnv.action_to_goal`.

        Kept so the two movers share one call signature at sites that do not
        care why the mover stopped.
        """
        return self.step(goal_xy, stop_radius=stop_radius, creep_below=creep_below).action


def build_pointnav(cfg) -> PointNavDriver:
    """Construct the mover from a resolved config.

    Called once per RUN by the runner and shared across episodes, in the same
    way the detector and scorer are: a NavAgent is built per episode, and the
    checkpoint is 34 MB. `NavAgent.reset` clears the recurrent state, so sharing
    the instance carries nothing between episodes.
    """
    return PointNavDriver(
        str(cfg.agent.pointnav_weights),
        stop_radius=float(cfg.agent.pointnav_stop_radius),
        depth_shape=tuple(cfg.agent.pointnav_depth_shape),
        depth_min_m=float(getattr(cfg.eval, "depth_min_m", 0.5)),
        depth_max_m=float(getattr(cfg.eval, "depth_max_m", 5.0)),
    )
