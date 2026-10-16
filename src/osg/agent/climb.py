"""The climb: getting up or down a staircase, once a floor switch is decided.

Lifted out of `nav_agent.py`, whose module docstring and CLAUDE.md both say it
owns the FSM state and nothing else. It had drifted to 2760 lines, and ~600 of
them were this.

Shaped like `ApproachPolicy` and `CloseLookPolicy` rather than like
`FloorPolicy`. `FloorPolicy` answers questions and owns no FSM, so it takes
`(cfg, stats, ...)`; the climb is a STATE HANDLER -- it is dispatched on
`State.CLIMB`, it writes `state`, `_goal_xy` and `_current_path`, and it reads
`_goal_floor_y_cache` and `_last_action`. It therefore takes the agent and
reaches those five through `self.nav`, which remains the single owner of FSM
state.

The nine pass-through properties below exist so the method bodies could move
BYTE-IDENTICAL: `self.cfg.agent.climb_carrot_hold_m`,
`self.stats["climb_flight_carrot"]` and `self.floors.pursuit_flight` all still
read the way they did. `stats` is a property and not a captured reference
because `NavAgent.reset()` clears that dict in place precisely so `FloorPolicy`
and `AbsenceSensor` keep pointing at the same object.

Behaviour is pinned by `tests/integration/test_climb_lock.py`, which runs three
real climbs on 00800-TEEsavR23oF in twelve seconds. It was green before this
move and after it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import FrameData
from ..mapping.costmap import PLANE
from ..planning.controller import agent_heading
from .state import FORWARD_ACTION, TURN_ACTION, State


class ClimbPolicy:
    """State handler for `State.CLIMB`."""

    def __init__(self, nav) -> None:
        self.nav = nav
        self.reset()

    def reset(self) -> None:
        self._climb_carrot = bool(getattr(self.cfg.agent, "climb_carrot", False))
        self._climb_goal_xy: Optional[np.ndarray] = None
        self._climb_centroid_xy: Optional[np.ndarray] = None
        self._climb_cells_xy: Optional[np.ndarray] = None
        self._climb_start_y = 0.0
        self._climb_from_floor: Optional[int] = None
        self._climb_direction = 0
        self._climb_steps = 0
        self._climb_max_dy = 0.0
        self._climb_pitched = False
        self._climb_blocked_run = 0
        self._climb_blocked_carrots: list = []
        self._climb_last_dist = None
        self._climb_paused_steps = 0
        self._climb_last_xy = None
        self._climb_turn_locked = False
        self._climb_suppressed_run = 0
        self._climb_aligned = False
        self._climb_align_turns = 0
        self._climb_trace: list = []
        self._flight_carrot_xy = None
        self._flight_carrot_h = 0.0
        self._carrot_xy = None
        self._carrot_disable_end = False
        # NOT in NavAgent.reset(), which was correct only because `_start_climb`
        # is the sole way into State.CLIMB and always writes it first. One
        # AttributeError away from being a bug; initialised here.
        self._climb_relink_step = 0
        # The stair route (`agent.climb_route`, mapping/stair_route.py).
        self._route = None
        self._route_plan_step = -100
        self._route_probe_step = -100
        self._route_blocks = 0
        self._route_queue: list = []
        self._route_short = False
        self._route_sweep_axis = 0.0
        self._route_sweep_side = 1.0
        # Where forward steps were refused this episode, for the planner.
        self._route_bumps: list = []
        self._route_scanned = False
        self._route_explores = 0
        self._route_best_dy = 0.0
        self._route_seek_step = -100
        self._route_last_xy = None
        self._route_failures = 0
        self._route_progress_step = 0

    # -------------------------------------------------- the agent's own state
    # Named exactly as on NavAgent so the moved bodies did not have to change.

    @property
    def cfg(self):
        return self.nav.cfg

    @property
    def stats(self):
        return self.nav.stats

    @property
    def floors(self):
        return self.nav.floors

    @property
    def floor_layer(self):
        return self.nav.floor_layer

    @property
    def costmap(self):
        return self.nav.costmap

    @property
    def stair_detector(self):
        return self.nav.stair_detector

    @property
    def pointnav(self):
        return self.nav.pointnav

    @property
    def profiler(self):
        return self.nav.profiler

    @property
    def step_count(self):
        return self.nav.step_count

    @property
    def trace(self) -> list:
        """`eval/record.py` writes this into every episode record."""
        return self._climb_trace

    def _at_the_stairs(self, frame: FrameData) -> bool:
        """Has the pursuit reached somewhere a climb can start?

        Either the stair detector's own evidence is under the agent, or the
        pursuit goal is within `stair_reach_m`. The second matters because the
        detector's `stairs` label is available on a minority of episodes; a
        stair-track goal was chosen BECAUSE it is a staircase, so arriving at
        it is enough.
        """
        agent_xy = frame.camera_position[list(PLANE)]
        if self._route_on and self._route_from_here(frame, agent_xy):
            return True
        if self.nav.stairs._on_a_staircase(agent_xy):
            return True
        reach = float(getattr(self.cfg.agent, "stair_reach_m", 0.6))
        if self.pointnav is not None:
            # The PointNav policy stops at its own radius (0.9 m) and will not
            # close further, so a reach inside that radius can never be met.
            # Measured on 00821: 82 pursuits of one stair target, the agent
            # standing 0.87 m from it at the end, zero climbs started.
            reach = max(reach, float(getattr(self.pointnav, "stop_radius", 0.9)) + 0.2)
        return float(np.linalg.norm(agent_xy - self.nav._goal_xy)) <= reach
    def _start_climb(self, frame: FrameData) -> None:
        agent_xy = frame.camera_position[list(PLANE)]
        self._climb_goal_xy = np.asarray(self.nav._goal_xy, dtype=float).copy()
        self._climb_centroid_xy = self._climb_goal_xy.copy()
        self._climb_start_y = float(frame.camera_position[1])
        self._climb_from_floor = int(self.floors.current_id)
        target_y = self.nav._goal_floor_y_cache
        here_y = float(self.floors.estimator.height_of(self._climb_from_floor))
        pursued = getattr(self.floors, "pursuit_flight", None)
        kind = str(getattr(pursued, "kind", "") or "")
        if bool(getattr(self.cfg.agent, "climb_direction_from_flight", False)) and kind:
            # THE FLIGHT KNOWS WHICH WAY IT GOES. Deriving the direction from
            # two storey heights instead is fragile in exactly the case that
            # matters, and it failed measurably: on 00800 cross_anchor_01 the
            # agent pursued a flight_up and started a DOWN climb with
            # `climb_target_y_x100` and `climb_here_y_x100` BOTH 16 -- the
            # estimator offered a level a few centimetres above the current
            # storey, `min(above)` picked it, and `target_y > here_y` is False
            # at equality, so the tie fell through to -1. `_flight_carrot` then
            # looked for treads 0.35-1.0 m BELOW an agent on the ground floor,
            # found none every step for 200 steps, and the agent milled at the
            # foot of the stairs having risen 0.00 m.
            self._climb_direction = 1 if kind == "up" else -1
        else:
            self._climb_direction = (
                0 if target_y is None else (1 if float(target_y) > here_y else -1)
            )
        self._climb_steps = 0
        self._climb_max_dy = 0.0
        self._climb_pitched = False
        self._climb_blocked_run = 0
        self._climb_relink_step = -100
        self._flight_carrot_xy = None
        self._flight_carrot_h = 0.0
        self._climb_blocked_carrots = []
        self._carrot_xy = None
        self._carrot_disable_end = False
        self._climb_last_dist = None
        self._climb_paused_steps = 0
        if self.step_count - self._route_plan_step > 5:
            self._route = None              # planned for another pose: plan again
        self._route_blocks = 0
        self._route_queue = []
        self._route_short = False
        self._route_scanned = False
        self._route_explores = 0
        self._route_best_dy = 0.0
        self._route_last_xy = None
        self._route_failures = 0
        self._route_progress_step = int(self.step_count)
        # The stair cells near the agent, for `_left_the_stairs`; None when the
        # detector has no evidence here, in which case the height test decides.
        layer = self.floor_layer
        cells = None
        if self.stair_detector is not None and layer.up_stair_hits is not None:
            mask = (
                (layer.up_stair_hits >= self.stair_detector.min_hits)
                | (layer.down_stair_hits >= self.stair_detector.min_hits)
            )
            rc = np.argwhere(mask)
            if len(rc):
                xy = np.stack([layer.costmap.grid_to_world(r) for r in rc])
                near = np.linalg.norm(xy - agent_xy, axis=1) < 3.0
                cells = xy[near] if near.any() else None
        self._climb_cells_xy = cells
        self.nav._current_path = None
        self.floors.climbing = True
        self.nav.state = State.CLIMB
        self.stats["climb_start"] = self.stats.get("climb_start", 0) + 1
        self.stats["climb_start_up"] = self.stats.get("climb_start_up", 0) + int(self._climb_direction > 0)
        self.stats["climb_start_down"] = self.stats.get("climb_start_down", 0) + int(self._climb_direction < 0)
        # The three values the direction is derived from, recorded because
        # reading the code could not explain a measured contradiction: on
        # 00800 cross_anchor_01 the agent pursued a flight_up and started a
        # DOWN climb, with `floor_transitions` 0 and `floor_y_drift` 0.0, so
        # `here_y` never moved and every traceable path says +1.
        self.stats["climb_target_y_x100"] = (
            -99999 if target_y is None else int(round(float(target_y) * 100)))
        self.stats["climb_here_y_x100"] = int(round(here_y * 100))
        self.stats["climb_from_floor"] = int(self._climb_from_floor)
        flight = getattr(self.floors, "pursuit_flight", None)
        self.stats["climb_flight_kind"] = (
            "none" if flight is None else str(getattr(flight, "kind", "?")))
    def _end_climb(self, ok: bool, why: str) -> None:
        self.floors.climbing = False
        if not ok:
            self.nav._note_failed_switch()
        self.stats["climb_ok" if ok else "climb_fail"] = (
            self.stats.get("climb_ok" if ok else "climb_fail", 0) + 1
        )
        self.stats[f"climb_end_{why}"] = self.stats.get(f"climb_end_{why}", 0) + 1
        self.stats["climb_max_dy_x100"] = max(
            int(self.stats.get("climb_max_dy_x100", 0)), int(round(self._climb_max_dy * 100))
        )
        if not ok and self.floors.pursuing:
            # Records the failure against the goal when portal_failure_memory is
            # on, so the next selection round does not send the agent straight
            # back to the same foot of the same wall.
            self.floors.end_pursuit("no_vertical_progress")
        self.nav._goal_xy = None
        self.nav._current_path = None
        self._carrot_disable_end = False
        self.nav.state = State.EXPLORE
    def _do_climb(self, frame: FrameData) -> str:
        agent_xy = frame.camera_position[list(PLANE)]
        cam_y = float(frame.camera_position[1])
        self._climb_steps += 1
        dy = (cam_y - self._climb_start_y) * (self._climb_direction or 1)
        self._climb_max_dy = max(self._climb_max_dy, dy)
        # Success is the estimator committing a new storey. `observe` runs
        # before the dispatch every step, so the id has already moved.
        if int(self.floors.current_id) != int(self._climb_from_floor):
            self._end_climb(True, "new_floor")
            return "look_up" if self._climb_pitched else TURN_ACTION
        # With levels suppressed for the duration of the climb, a storey is not
        # announced by the estimator while the agent is on it. Judge arrival by
        # height instead: a full `new_level_m` of gain is a storey however many
        # flights it took, and the estimator commits it on the next frame once
        # the climb releases the suppression.
        needed = float(self.cfg.floor.new_level_m)
        tol = float(getattr(self.cfg.agent, "climb_to_target_storey_tol_m", 0.0) or 0.0)
        if tol > 0.0 and self.nav._goal_floor_y_cache is not None:
            # The gap to the storey being climbed to is KNOWN; a constant
            # storey height ends the climb on the treads when the real one is
            # taller (config: climb_to_target_storey_tol_m).
            here = float(self.floors.estimator.height_of(self._climb_from_floor))
            needed = max(needed, abs(float(self.nav._goal_floor_y_cache) - here) - tol)
        routed = self._route_on and self._route is not None
        if routed and self._route_arrived(frame, agent_xy):
            self._end_climb(True, "route_end")
            return "look_up" if self._climb_pitched else TURN_ACTION
        if (
            bool(getattr(self.cfg.floor, "no_level_on_flight", False))
            and dy >= needed
            # A route knows where the floor is; a height threshold computed
            # from a storey estimate can end the climb on the treads (00800's
            # prior puts the lower floor at 0.64 for a true 0.16).
            and not routed
        ):
            self._end_climb(True, "storey_of_height")
            return "look_up" if self._climb_pitched else TURN_ACTION
        # A pursuit `observe` ended for another reason (deadline) while we were
        # climbing: judge by height gained, not by what ended it.
        if not self.floors.pursuing and dy >= 0.6 * float(self.cfg.floor.new_level_m):
            self._end_climb(True, "height")
            return "look_up" if self._climb_pitched else TURN_ACTION
        if self._climb_steps > int(getattr(self.cfg.agent, "climb_max_steps", 80)):
            self._end_climb(False, "budget")
            return "look_up" if self._climb_pitched else TURN_ACTION
        if routed:
            if self._route_stalled(agent_xy) or self._route_failures >= 3:
                self._end_climb(False, "stalled" if self._route_failures < 3 else "route_lost")
                return "look_up" if self._climb_pitched else TURN_ACTION
        elif self._carrot_stalled(agent_xy) and dy < 0.3:
            self._end_climb(False, "stalled")
            return "look_up" if self._climb_pitched else TURN_ACTION
        # One flight is not one storey. When the treads run out part-way -- on a
        # half-landing -- look for the next flight from here and carry on, rather
        # than handing back to exploration 2.3 m into a 3.2 m descent.
        if (
            bool(getattr(self.cfg.agent, "climb_relink_flights", False))
            and not routed                       # a route already runs through landings
            and self._flight_carrot(frame, agent_xy) is None
            and self.step_count - self._climb_relink_step >= 10
        ):
            self._climb_relink_step = self.step_count
            if self._relink_flight(frame, agent_xy):
                self.stats["climb_relinked"] = self.stats.get("climb_relinked", 0) + 1
                self._flight_carrot_xy = None       # a new flight, a new carrot
                self._climb_blocked_carrots = []
                self._carrot_xy = None
                self._climb_last_dist = None
                self._climb_paused_steps = 0
        oracle = self._oracle_action(frame, agent_xy)
        if oracle is not None:
            return self._traced_climb(frame, agent_xy, oracle,
                                      standing=cam_y - float(self.cfg.agent.camera_height), dy=dy)
        # Descending: tilt the camera down once so the carrot sees the treads
        # below rather than the far wall (ASCENT's phase 2, `:1120-1127`).
        align = self._climb_align_action(frame, agent_xy)
        if align is not None:
            return self._traced_climb(frame, agent_xy, align, standing=None, dy=dy)
        if self._climb_direction < 0 and not self._climb_pitched:
            self._climb_pitched = True
            return self._traced_climb(frame, agent_xy, "look_down", standing=None, dy=dy)
        if self._route_on:
            action = self._route_action(frame, agent_xy)
            if action is not None:
                return self._traced_climb(
                    frame, agent_xy, action,
                    standing=cam_y - float(self.cfg.agent.camera_height), dy=dy)
        return self._traced_climb(
            frame, agent_xy, self._carrot_action(frame, agent_xy),
            standing=cam_y - float(self.cfg.agent.camera_height), dy=dy)
    # ------------------------------------------------------------ stair route

    @property
    def _route_on(self) -> bool:
        return bool(getattr(self.cfg.agent, "climb_route", False))

    def _route_direction(self) -> int:
        """Which way the pursuit goes: the climb's own, else the flight's, else
        the storey asked for against this one; 0 when nothing says."""
        if self.nav.state is State.CLIMB and self._climb_direction:
            return int(self._climb_direction)
        kind = str(getattr(getattr(self.floors, "pursuit_flight", None), "kind", "") or "")
        if kind in ("up", "down"):
            return 1 if kind == "up" else -1
        target = self.nav._goal_floor_y_cache
        if target is None:
            return 0
        here = float(self.floors.estimator.height_of(int(self.floors.current_id)))
        return 1 if float(target) > here else -1

    def _plan_route(self, frame: FrameData, agent_xy: np.ndarray, direction: int,
                    explore: bool = False):
        from ..mapping.stair_route import plan_stair_route

        floor_id = (self._climb_from_floor
                    if self.nav.state is State.CLIMB and self._climb_from_floor is not None
                    else int(self.floors.current_id))
        floor_y = float(self.floors.estimator.height_of(int(floor_id)))
        standing = float(frame.camera_position[1]) - float(self.cfg.agent.camera_height)
        with self.profiler.timeit("stair_route"):
            route = plan_stair_route(
                self.costmap, floor_y, agent_xy, standing, int(direction),
                target_y=self.nav._goal_floor_y_cache,
                storey_m=float(self.cfg.floor.new_level_m),
                bumps_xy=self._route_bumps,
                min_gain_m=float(getattr(self.cfg.agent, "climb_route_min_gain_m", 0.3)),
                explore=explore,
            )
        key = "climb_route_plans" if route is not None else "climb_route_none"
        self.stats[key] = self.stats.get(key, 0) + 1
        return route

    def route_to_storey(self, frame: FrameData, target_key: int):
        """A pursuit along a stair route to the storey asked for, or None.

        For when `FloorPolicy.try_switch` has nothing to drive to: the old
        flight search needs a blob it accepts from where the agent happens to
        be, and on 00862 (seed 42, static 50007) the belief asked for the
        lower storey 37 times over 1000 steps and no switch was ever tried.
        A route that leaves the floor within the planning radius is a
        staircase walked to, not a sightline. Rate-limited to every
        `climb_route_probe_steps`."""
        if not self._route_on:
            return None
        every = int(getattr(self.cfg.agent, "climb_route_probe_steps", 10))
        if self.step_count - self._route_seek_step < every:
            return None
        self._route_seek_step = int(self.step_count)
        layer = self.floors.stack.by_key(int(target_key))
        if layer is None:
            return None
        target_y = float(layer.floor_y)
        here = float(self.floors.estimator.height_of(int(self.floors.current_id)))
        direction = 1 if target_y > here else -1
        agent_xy = frame.camera_position[list(PLANE)]
        saved = self.nav._goal_floor_y_cache
        self.nav._goal_floor_y_cache = target_y
        try:
            route = self._plan_route(frame, agent_xy, direction)
        finally:
            self.nav._goal_floor_y_cache = saved
        if route is None or (not route.reaches_target
                             and route.gain_m < float(getattr(
                                 self.cfg.agent, "climb_route_start_gain_m", 0.5))):
            return None
        off = np.flatnonzero(np.abs(route.heights - here) > 0.15)
        goal = route.xy[off[0]] if len(off) else route.end_xy
        goal = self.floors.pursue_route(frame, int(self.step_count), goal, target_y,
                                        "up" if direction > 0 else "down")
        if goal is not None:
            self.stats["climb_route_seek"] = self.stats.get("climb_route_seek", 0) + 1
        return goal

    def _route_from_here(self, frame: FrameData, agent_xy: np.ndarray) -> bool:
        """Is there a walk from where the agent stands onto the stairs, within
        `climb_route_start_m`? Then the climb starts here, wherever the pursuit
        goal is: on 00821 the flight's foot was 1.5 m beside the true one, and
        the agent passed within 1.1 m of the real foot on its way there."""
        if self.step_count - self._route_probe_step < 5:
            return False
        self._route_probe_step = int(self.step_count)
        direction = self._route_direction()
        if direction == 0:
            return False
        route = self._plan_route(frame, agent_xy, direction)
        if route is None:
            return False
        if not route.reaches_target and route.gain_m < float(getattr(
                self.cfg.agent, "climb_route_start_gain_m", 0.5)):
            # 00878 (seed 42, static 50003): a 0.3 m "route" up something
            # step-like 4 m from the stairs started three climbs, two of
            # which pressed into it for their whole 300-step budget.
            return False
        floor_y = float(self.floors.estimator.height_of(int(self.floors.current_id)))
        start_m = float(getattr(self.cfg.agent, "climb_route_start_m", 2.5))
        if route.leaves_floor_after_m(floor_y) > start_m:
            return False
        self._route = route
        self._route_plan_step = int(self.step_count)
        self.stats["climb_route_start"] = self.stats.get("climb_route_start", 0) + 1
        return True

    def _record_climb_surfaces(self, frame: FrameData) -> None:
        """Keep the route's height layer current while the occupancy update is
        suppressed on the stairs -- the upper flight is only seen from the
        lower one."""
        cm = self.costmap
        if getattr(cm, "height_top", None) is None or not bool(getattr(self.floors, "on_stairs", False)):
            return
        from ..core.geometry import backproject

        m = self.cfg.mapping
        pts = backproject(frame.depth, frame.intrinsics, frame.T_wc,
                          stride=int(m.depth_stride), max_depth=float(m.max_range_m))
        if pts.shape[0] == 0:
            return
        floor_y = float(self.floors.estimator.height_of(int(self._climb_from_floor)))
        cm.ensure_contains(frame.camera_position[list(PLANE)], margin_m=float(m.max_range_m) + 1.0)
        cm.record_top(pts[np.abs(pts[:, 1] - floor_y) < 3.5],
                      cap_y=float(frame.camera_position[1]))

    def _route_action(self, frame: FrameData, agent_xy: np.ndarray) -> Optional[str]:
        """Follow the route: turn toward a point `climb_route_lookahead_m` along
        it and walk forward within the deadband. None hands the step to the
        carrot cascade (no route at all).

        Blocked forwards are the hard part. Turns are 30 degrees, so the agent
        faces a flight's axis to within 15 at best, and the simulator does not
        slide a blocked step along the wall: a forward that touches an edge
        goes nowhere (`eval.allow_sliding: false`). Re-centring on the route
        after a turn walks straight back into the same edge -- measured at the
        foot of the 00800 probe flight, 40 steps on one spot, where every
        heading with a -z component is blocked and the way on is 90 degrees
        off the route. So a block starts a sweep (`_route_sweep`).
        """
        self._record_climb_surfaces(frame)
        moved = (float(np.linalg.norm(agent_xy - self._route_last_xy))
                 if self._route_last_xy is not None else 1.0)
        self._route_last_xy = np.asarray(agent_xy, dtype=float).copy()
        blocked = self.nav._last_action == FORWARD_ACTION and moved < 0.03
        if moved >= 0.05:
            self._route_blocks = 0
            self._route_short = False
        if self._route_queue:
            return self._route_queue.pop(0)
        step = int(self.step_count)
        route = self._route
        replan = route is None or step - self._route_plan_step >= int(
            getattr(self.cfg.agent, "climb_route_replan_steps", 10))
        if route is not None and not replan:
            off = float(np.min(np.linalg.norm(route.xy - agent_xy, axis=1)))
            at_end = float(np.linalg.norm(route.end_xy - agent_xy)) < 0.4
            replan = off > 0.6 or (at_end and not route.reaches_target)
        if replan:
            fresh = self._plan_route(frame, agent_xy, self._climb_direction or 1)
            self._route_plan_step = step
            if fresh is None:
                if not self._route_scanned:
                    # Nothing walkable in view -- typically a landing, facing its
                    # wall, with the next flight behind (00821's switchback:
                    # the first leg in 19 steps, then "lost" at the turn). Look
                    # round once, recording surfaces, before giving up.
                    # Pitched down: the camera never sees the floor within
                    # ~1.4 m of itself, and on a landing that ring is all
                    # there is (00821: the route from the landing was one cell
                    # long). A descent is already pitched.
                    self._route_scanned = True
                    n = max(1, int(round(360.0 / float(self.cfg.agent.turn_deg))))
                    self._route_plan_step = -100
                    self.stats["climb_route_scan"] = self.stats.get("climb_route_scan", 0) + 1
                    if self._climb_pitched:
                        self._route_queue = ["turn_left"] * (n - 1)
                        return "turn_left"
                    self._route_queue = ["turn_left"] * n + ["look_up"]
                    return "look_down"
                adopted = False
                if self._route_explores < 2:
                    # Looked round and still nothing: walk to the far end of
                    # this level and look again. On 00821 the upper leg starts
                    # behind the wall between the two legs, out of sight from
                    # where the lower leg arrives.
                    wander = self._plan_route(frame, agent_xy, self._climb_direction or 1,
                                              explore=True)
                    if wander is not None:
                        self._route_explores += 1
                        self._route_scanned = False      # look again when there
                        self._route = route = wander
                        adopted = True
                        self.stats["climb_route_explore"] = (
                            self.stats.get("climb_route_explore", 0) + 1)
                if not adopted:
                    self._route_failures += 1
                if route is None:
                    return None
            else:
                self._route_failures = 0
                self._route_scanned = False
                self._route = route = fresh
        i = int(np.argmin(np.linalg.norm(route.xy - agent_xy, axis=1)))
        look = float(getattr(self.cfg.agent, "climb_route_lookahead_m", 0.5))
        if self._route_short:
            look = 0.25
        j = min(int(np.searchsorted(route.cost_m, float(route.cost_m[i]) + look)), len(route.xy) - 1)
        goal = np.asarray(route.xy[j], dtype=float)
        self._flight_carrot_xy = goal          # what `_traced_climb` records
        delta = goal - agent_xy
        err = float(np.arctan2(delta[1], delta[0]) - agent_heading(frame.T_wc))
        err = float(np.degrees((err + np.pi) % (2 * np.pi) - np.pi))
        # In the (x, z) plane `agent_heading` measures, a POSITIVE error is a
        # turn to the RIGHT -- `WaypointController` (planning/controller.py)
        # has always turned that way. (`_climb_align_action` below turns left
        # for a positive error; it is off in every current preset.)
        toward = "turn_right" if err >= 0 else "turn_left"
        if blocked:
            # Remember where the step was refused, and plan round it.
            h = agent_heading(frame.T_wc)
            self._route_bumps.append(np.asarray(agent_xy, dtype=float)
                                     + 0.25 * np.array([np.cos(h), np.sin(h)]))
            self._route_plan_step = -100
            return self._route_sweep(frame, err)
        deadband = float(getattr(self.cfg.agent, "climb_route_deadband_deg", 20.0))
        if abs(err) > deadband and float(np.linalg.norm(delta)) > 0.15:
            return toward
        return FORWARD_ACTION

    _SWEEP = (1, -1, 2, -2, 3, -3, 4, -4)

    def _route_sweep(self, frame: FrameData, err_deg: float) -> str:
        """The forward just taken went nowhere: try headings further and further
        either side of the route direction -- 30 degrees toward the error's
        side, 30 away, 60 toward, 60 away, out to 120 -- each committed to with
        a forward step. Any real motion ends the sweep (`_route_action`); a
        sweep that runs out re-plans with a short lookahead and starts over.
        """
        turn = float(self.cfg.agent.turn_deg)
        heading = float(np.degrees(agent_heading(frame.T_wc)))
        if self._route_blocks == 0:
            # The reachable heading nearest the route direction, the sweep's axis.
            self._route_sweep_axis = heading + round(err_deg / turn) * turn
            self._route_sweep_side = 1.0 if err_deg >= 0 else -1.0
        self._route_blocks += 1
        self.stats["climb_route_blocked"] = self.stats.get("climb_route_blocked", 0) + 1
        if self._route_blocks > len(self._SWEEP):
            self._route_blocks = 0
            self._route_short = True
            self._route_plan_step = -100
            self.stats["climb_route_sweep_exhausted"] = (
                self.stats.get("climb_route_sweep_exhausted", 0) + 1)
            return FORWARD_ACTION
        offset = self._SWEEP[self._route_blocks - 1] * self._route_sweep_side * turn
        want = float((self._route_sweep_axis + offset - heading + 180.0) % 360.0 - 180.0)
        n = int(round(abs(want) / turn))
        if n == 0:
            return FORWARD_ACTION
        # A POSITIVE angle is a turn to the RIGHT (see `_route_action`).
        action = "turn_right" if want > 0 else "turn_left"
        self._route_queue = [action] * (n - 1) + [FORWARD_ACTION]
        return action

    def _route_arrived(self, frame: FrameData, agent_xy: np.ndarray) -> bool:
        """At the end of a route that reaches the storey, standing at its height."""
        route = self._route
        if route is None or not route.reaches_target:
            return False
        standing = float(frame.camera_position[1]) - float(self.cfg.agent.camera_height)
        # The route ends on FLAT floor, so standing within a tread of its height
        # is standing on the storey -- not two steps short of it, which 0.2 m
        # allowed on 00821 (ended at -0.17 for a floor at 0.13).
        return (float(np.linalg.norm(route.end_xy - agent_xy)) < 0.4
                and abs(standing - float(route.heights[-1])) < 0.1)

    def _route_stalled(self, agent_xy: np.ndarray) -> bool:
        """No HEIGHT gained in the climb's direction for
        `climb_route_stall_steps` -- not "no progress along the route", which
        every re-plan reset: 00878's two budget-length climbs re-planned 133
        times and never looked stalled. The look-round and the walk to a
        landing's far end take ~25 steps between gains."""
        if self._climb_max_dy > self._route_best_dy + 0.1:
            self._route_best_dy = float(self._climb_max_dy)
            self._route_progress_step = int(self.step_count)
        return int(self.step_count) - self._route_progress_step > int(
            getattr(self.cfg.agent, "climb_route_stall_steps", 40))

    def _oracle_action(self, frame: FrameData, agent_xy: np.ndarray) -> Optional[str]:
        """`agent.climb_oracle`: Habitat's follower walks the staircase.

        The goal is the pursued flight's far end -- its top on an ascent, its
        foot on a descent -- snapped at the target storey's height, or at that
        end's own tread height when no storey is known. None (arrived, or the
        follower refuses) hands the step back to the sensor-only controller,
        whose end tests then judge the climb as they always do.
        """
        fn = getattr(self.nav, "_climb_nav_fn", None)
        if fn is None:
            return None
        up = self._climb_direction >= 0
        flight = getattr(self.floors, "pursuit_flight", None)
        far_y = None
        if flight is not None and getattr(flight, "n_cells", 0):
            far = np.asarray(flight.top_xy if up else flight.foot_xy, dtype=float)
            heights = np.asarray(flight.heights, dtype=float)
            if heights.size:
                far_y = float(heights.max() if up else heights.min())
        else:
            far = np.asarray(self._climb_goal_xy, dtype=float)
        target_y = self.nav._goal_floor_y_cache
        floor_y = float(target_y) if target_y is not None else far_y
        action = fn(far, floor_y)
        key = "climb_oracle_steps" if action is not None else "climb_oracle_none"
        self.stats[key] = self.stats.get(key, 0) + 1
        return action

    def _traced_climb(self, frame: FrameData, agent_xy, action: str,
                      *, standing, dy: float) -> str:
        """Record what this climb step did, when the run asked for it.

        The climb is the single largest consumer of the step budget -- 300 of
        500 steps in outputs/mf5_pass2_v16 ep1, for 2.56 m -- and nothing in
        `episodes.jsonl` says where those steps went: `eval.behaviour_log`
        writes `step_trace` only for the ascentnav agent. Six numbers a climb
        step, and only while climbing.
        """
        if not (self.cfg.eval.debug_frames
                or bool(getattr(self.cfg.eval, "behaviour_log", False))):
            return action
        goal = self._flight_carrot_xy
        self._climb_trace.append({
            "step": int(self.step_count),
            "xy": [round(float(v), 2) for v in agent_xy],
            "standing": None if standing is None else round(float(standing), 3),
            "dy": round(float(dy), 3),
            "goal": None if goal is None else [round(float(v), 2) for v in goal],
            "goal_dist": (None if goal is None
                          else round(float(np.linalg.norm(goal - agent_xy)), 2)),
            "action": str(action),
            "resets": int(getattr(self.pointnav, "n_resets", 0) or 0),
        })
        return action
    def _carrot_goal(
        self, frame: FrameData, agent_xy: np.ndarray
    ) -> Optional[np.ndarray]:
        """Place ASCENT's short stair waypoint along the farthest depth ray."""
        depth = frame.depth
        if depth.size == 0:
            return None
        max_value = float(np.max(depth))
        if not np.isfinite(max_value):
            return None
        rows_cols = np.argwhere(depth == max_value)
        if rows_cols.size == 0:
            return None
        u = float(np.mean(rows_cols[:, 1]))
        intr = frame.intrinsics
        hfov = 2.0 * float(np.arctan(intr.width / (2.0 * intr.fx)))
        normalized_u = float(np.clip((u - float(intr.cx)) / float(intr.cx), -1.0, 1.0))
        heading = agent_heading(frame.T_wc) + normalized_u * hfov / 2.0
        distance = float(getattr(self.cfg.agent, "climb_carrot_m", 0.8))
        return agent_xy + distance * np.array([np.cos(heading), np.sin(heading)])
    def _update_carrot(
        self, frame: FrameData, agent_xy: np.ndarray
    ) -> Optional[np.ndarray]:
        fresh = self._carrot_goal(frame, agent_xy)
        if fresh is None:
            return self._carrot_xy
        end = getattr(self, "_climb_goal_xy", None)
        near_end = end is not None and float(np.linalg.norm(end - agent_xy)) <= 0.5
        if self._carrot_xy is None or end is None or near_end or self._carrot_disable_end:
            self._carrot_xy = fresh
        elif np.linalg.norm(fresh - end) < np.linalg.norm(self._carrot_xy - end):
            self._carrot_xy = fresh
        return self._carrot_xy
    def _stair_cell_carrot(self, agent_xy: np.ndarray) -> Optional[np.ndarray]:
        """Steer at the detector's own stair cells rather than the farthest thing
        in view.

        ASCENT's depth-ray carrot works because ASCENT starts it standing ON the
        flight, where the farthest visible point is up it. Ours starts within
        reach of a stair target that is often beside the flight, not at its
        foot: on 00821 seven climbs pushed forward 238 times into whatever the
        far wall was and rose 0.00 m. The stamped `stairs` cells say where the
        treads actually are. Ascending, aim at the farthest of them within a
        few metres -- the top of the visible flight; descending, the nearest --
        the lip.
        """
        if not bool(getattr(self.cfg.agent, "climb_cell_carrot", False)):
            return None
        layer = self.floor_layer
        if self.stair_detector is None or layer.up_stair_hits is None:
            return None
        hits = layer.up_stair_hits if self._climb_direction >= 0 else layer.down_stair_hits
        mask = hits >= self.stair_detector.min_hits
        if layer.disabled_stair is not None:
            mask &= ~layer.disabled_stair
        rc = np.argwhere(mask)
        if not len(rc):
            return None
        xy = np.stack([layer.costmap.grid_to_world(r) for r in rc])
        d = np.linalg.norm(xy - agent_xy, axis=1)
        near = d <= 3.0
        if not near.any():
            return None
        xy, d = xy[near], d[near]
        pick = int(np.argmax(d)) if self._climb_direction >= 0 else int(np.argmin(d))
        self.stats["climb_cell_carrot"] = self.stats.get("climb_cell_carrot", 0) + 1
        return xy[pick]
    def _relink_flight(self, frame: FrameData, agent_xy: np.ndarray) -> bool:
        """Pick up the next flight of a multi-flight staircase.

        Re-reads the height layer from where the agent is standing and takes the
        nearest flight going the same way whose foot is within a few metres --
        the next flight down from a half-landing. Returns whether one was found.
        """
        from ..mapping.stairs import find_flights, mouth_xy

        floor_y = float(self.floors.height_of(self.floors.current_id))
        flights = find_flights(
            self.costmap, floor_y,
            new_level_m=float(self.cfg.floor.new_level_m),
            min_span_m=float(getattr(self.cfg.floor, "flight_relink_span_m", 0.5)),
            min_cells=int(getattr(self.cfg.floor, "flight_min_cells", 150)),
            wide_span_m=self.floors.flight_span_m(floor_y),
            wide_mask=getattr(self.costmap, "stair_mask", None),
            order_path=bool(getattr(self.cfg.agent,
                "climb_carrot_follow_path", False)),
        )
        want = "up" if self._climb_direction >= 0 else "down"
        current = getattr(self.floors, "pursuit_flight", None)
        near = []
        for f in flights:
            if f.kind != want:
                continue
            if float(np.linalg.norm(mouth_xy(f) - agent_xy)) > 4.0:
                continue
            if current is not None and float(np.linalg.norm(mouth_xy(f) - mouth_xy(current))) < 0.5:
                continue  # the flight just finished
            near.append(f)
        if not near:
            return False
        self.floors.pursuit_flight = min(
            near, key=lambda f: float(np.linalg.norm(mouth_xy(f) - agent_xy)))
        return True
    def _flight_carrot(self, frame: FrameData, agent_xy: np.ndarray) -> Optional[np.ndarray]:
        """The next tread. Keeps the goal ON the flight, which is what the
        PointNav mover will climb.

        From the flight the pursuit chose (`FloorPolicy.pursuit_flight`), aim at
        the tread 0.35-1.0 m above the agent's standing height (below it on a
        descent), nearest in the plane; at the top, where none is left, at the
        highest tread; with no flight at all, None and the older carrots apply.
        """
        if not bool(getattr(self.cfg.agent, "climb_flight_carrot", False)):
            return None
        flight = getattr(self.floors, "pursuit_flight", None)
        if flight is None or not flight.n_cells:
            return None
        standing = float(frame.camera_position[1]) - float(self.cfg.agent.camera_height)
        xy = np.stack([self.costmap.grid_to_world(rc.astype(float)) for rc in flight.cells_rc])
        h = np.asarray(flight.heights, dtype=float)
        sign = 1.0 if self._climb_direction >= 0 else -1.0
        # A carrot the agent is still walking to is not re-picked
        # (config: climb_carrot_hold_m).
        hold_m = float(getattr(self.cfg.agent, "climb_carrot_hold_m", 0.0) or 0.0)
        held = self._flight_carrot_xy
        blocked = self._climb_blocked_carrots if hold_m > 0.0 else []
        if hold_m > 0.0 and held is not None:
            reached = float(np.linalg.norm(held - agent_xy)) <= hold_m
            passed = (float(self._flight_carrot_h) - standing) * sign <= 0.1
            if not reached and not passed:
                self.stats["climb_carrot_held"] = self.stats.get("climb_carrot_held", 0) + 1
                return held
            self.stats["climb_carrot_repick"] = self.stats.get("climb_carrot_repick", 0) + 1
        ahead = (h - standing) * sign
        # A staircase that doubles back has no usable straight axis, so before
        # any height-band rule, try following the run itself. `flight.path_m`
        # is walking distance from the mouth THROUGH the flight's own cells, so
        # it increases along the direction of travel even around a turn, where
        # both "nearest in the plane" and "highest tread" aim across the
        # banister. Measured on 00873 ep50005: 28 of the flight's 57
        # ground-truth steps run BACKWARDS along its foot->top chord.
        path = getattr(flight, "path_m", None)
        spacing = float(getattr(self.cfg.agent, "climb_carrot_min_ahead_m", 0.0) or 0.0)
        if (bool(getattr(self.cfg.agent, "climb_carrot_follow_path", False))
                and path is not None and len(path) == len(h)):
            finite = np.isfinite(path)
            # ...but only once the agent is ON the flight. Following the run
            # from an arbitrary entry point while still standing off it aims at
            # a tread 0.5 m further ALONG THE STAIRCASE, which from the floor
            # below is across the room: measured on 00808, the carrot opened at
            # 2.96 m and grew to 4.45 m while the agent walked 180 steps and
            # rose 0.00. Off the flight, the older carrots take over and head
            # for the mouth, which is what gets it on.
            on_flight = float(getattr(
                self.cfg.agent, "climb_carrot_path_max_offset_m", 1.0) or 1.0)
            near = (np.linalg.norm(xy - agent_xy, axis=1) <= on_flight) & finite
            if near.any():
                here = int(np.argmin(np.where(
                    near, np.linalg.norm(xy - agent_xy, axis=1), np.inf)))
                want_m = float(path[here]) + max(spacing, 0.5)
                onward = finite & (path > float(path[here]) + 1e-6)
                if onward.any():
                    pick = int(np.argmin(np.where(
                        onward, np.abs(path - want_m), np.inf)))
                    self.stats["climb_carrot_path"] = (
                        self.stats.get("climb_carrot_path", 0) + 1)
                    return self._hold_flight_carrot(xy, h, pick)
        band = (ahead >= 0.35) & (ahead <= 1.0)
        # Cells under or beside the agent are never the way up a staircase,
        # whatever height the map gives them (config: climb_carrot_min_ahead_m).
        min_ahead = float(getattr(self.cfg.agent, "climb_carrot_min_ahead_m", 0.0) or 0.0)
        if min_ahead > 0.0:
            far_enough = np.linalg.norm(xy - agent_xy, axis=1) >= min_ahead
            if (band & far_enough).any():
                band &= far_enough
            elif band.any():
                self.stats["climb_carrot_underfoot"] = self.stats.get("climb_carrot_underfoot", 0) + 1
                band = np.zeros_like(band)         # fall through to the far end
        # ...and AHEAD of the agent, not behind it. See the flag's comment:
        # a ramped height field makes an iso-height contour a line across the
        # flight, so the nearest in-band cell can be back the way it came.
        if bool(getattr(self.cfg.agent, "climb_carrot_forward_only", False)) and band.any():
            far = int(np.argmax(np.where(band, ahead, -np.inf)))
            axis = xy[far] - agent_xy
            norm = float(np.linalg.norm(axis))
            if norm > 1e-6:
                forward = ((xy - agent_xy) @ (axis / norm)) > 0.0
                if (band & forward).any():
                    band &= forward
                else:
                    self.stats["climb_carrot_none_ahead"] = (
                        self.stats.get("climb_carrot_none_ahead", 0) + 1)
        if len(blocked):
            # A tread the mover has already refused to drive to is not offered
            # again this climb: holding a goal means holding an UNREACHABLE one
            # too, and the point-goal policy answers that by pressing into
            # whatever is in the way. Measured (outputs/mf5_pass2_v17 ep1, the
            # descent): 107 forced-forwards and 24 blocked turns in a climb
            # where v16 had none of either, because the held tread was across
            # the banister. Only while the band still offers something else.
            reachable = np.min(np.linalg.norm(
                xy[:, None, :] - np.asarray(blocked)[None, :, :], axis=2), axis=1) > 0.25
            if (band & reachable).any():
                band &= reachable
        if band.any():
            d = np.linalg.norm(xy[band] - agent_xy, axis=1)
            self.stats["climb_flight_carrot"] = self.stats.get("climb_flight_carrot", 0) + 1
            return self._hold_flight_carrot(xy[band], h[band], int(np.argmin(d)))
        above = ahead > 0.1
        if min_ahead > 0.0:
            far_enough = np.linalg.norm(xy - agent_xy, axis=1) >= min_ahead
            if (above & far_enough).any() or not bool(getattr(
                    self.cfg.agent, "climb_carrot_relax_min_ahead", False)):
                above &= far_enough
            elif above.any():
                # LAST RESORT, and it aims at the NEXT tread, not the highest.
                #
                # Two things are wrong without this. `min_ahead` exists because
                # the point-goal mover circles a goal a step away, but applying
                # it to the final fallback means that when every remaining
                # tread is nearer than `min_ahead` the carrot is None, and
                # `_carrot_action` walks plain FORWARD -- into the wall, at a
                # turn. And the fallback below picks `argmax(ahead)`, the
                # HIGHEST tread, which on a switchback lies across the banister
                # rather than along the run, so the mover presses into the
                # corner instead of turning.
                #
                # Measured on 00873 ep50005, the ascent that knows it must
                # climb: it rises 1.80 m of 3.20 and then slides back to the
                # bottom three times, with ~105 cells still in band. Aiming at
                # the lowest tread still above the agent follows the staircase
                # one step at a time, which is the only ordering `Flight` can
                # give -- `cells_rc` is a set, not a polyline.
                self.stats["climb_carrot_relaxed"] = (
                    self.stats.get("climb_carrot_relaxed", 0) + 1)
                pick = int(np.argmin(np.where(above, ahead, np.inf)))
                return self._hold_flight_carrot(xy, h, pick)
        if above.any():
            self.stats["climb_flight_carrot_top"] = self.stats.get("climb_flight_carrot_top", 0) + 1
            return self._hold_flight_carrot(xy[above], h[above], int(np.argmax(ahead[above])))
        return None
    def _hold_flight_carrot(self, xy, heights, pick: int) -> np.ndarray:
        """Remember the tread just chosen, so the next step can keep it."""
        self._flight_carrot_xy = np.asarray(xy[pick], dtype=float).copy()
        self._flight_carrot_h = float(heights[pick])
        return self._flight_carrot_xy
    def _carrot_action(self, frame: FrameData, agent_xy: np.ndarray) -> str:
        goal = self._flight_carrot(frame, agent_xy)
        if goal is None:
            goal = self._stair_cell_carrot(agent_xy)
        if goal is None:
            goal = self._update_carrot(frame, agent_xy)
        if goal is None:
            return FORWARD_ACTION
        if self.pointnav is not None:
            with self.profiler.timeit("mover"):
                nav = self.pointnav.step(goal, stop_radius=0.0)
            if nav.action is None:
                self.stats["climb_forced_forward"] = (
                    self.stats.get("climb_forced_forward", 0) + 1
                )
                self._climb_blocked_run += 1
                limit = int(getattr(self.cfg.agent, "climb_blocked_turn_after", 0))
                if limit > 0 and self._climb_blocked_run >= limit:
                    # The mover has said STOP this many times running and the
                    # agent has not risen: it is pressing into something. A
                    # turn re-aims the carrot; pushing again does not.
                    self._climb_blocked_run = 0
                    self._carrot_xy = None
                    if (float(getattr(self.cfg.agent, "climb_carrot_hold_m", 0.0) or 0.0) > 0.0
                            and self._flight_carrot_xy is not None):
                        # The held tread is what the mover is refusing; let go
                        # of it and do not pick it again this climb.
                        self._climb_blocked_carrots.append(
                            np.asarray(self._flight_carrot_xy, dtype=float).copy())
                        self._flight_carrot_xy = None
                        self.stats["climb_carrot_blocked_release"] = (
                            self.stats.get("climb_carrot_blocked_release", 0) + 1)
                    self.stats["climb_blocked_turn"] = (
                        self.stats.get("climb_blocked_turn", 0) + 1
                    )
                    return TURN_ACTION
                return FORWARD_ACTION
            self._climb_blocked_run = 0
            return self._climb_turn_lock(frame, agent_xy, goal, nav.action)
        action = self.nav._follow_to(frame, goal)
        return action if action is not None else FORWARD_ACTION
    def _climb_align_action(self, frame: FrameData, agent_xy: np.ndarray):
        """Face the flight before stepping onto it, or None when already facing.

        The probe climbs this flight in a third of the steps a run takes, and
        the difference is its starting pose (config: climb_align_first_deg).
        """
        limit = float(getattr(self.cfg.agent, "climb_align_first_deg", 0.0) or 0.0)
        if limit <= 0.0 or self._climb_aligned:
            return None
        # The flight's own axis, NOT the carrot: a tread 0.4 m away has a
        # bearing that swings tens of degrees for a few centimetres of pose
        # change, so aligning to it can spin a full revolution without ever
        # landing inside the tolerance -- measured once in
        # outputs/mf5_pass2_v21 (`climb_align_gave_up` 1).
        flight = getattr(self.floors, "pursuit_flight", None)
        axis = None
        if flight is not None and getattr(flight, "n_cells", 0):
            axis = np.asarray(flight.top_xy, dtype=float) - np.asarray(flight.foot_xy, dtype=float)
            if self._climb_direction < 0:
                axis = -axis
            if float(np.linalg.norm(axis)) < 0.3:
                axis = None
        if axis is None:
            goal = self._flight_carrot(frame, agent_xy)
            if goal is None:
                self._climb_aligned = True
                return None
            axis = np.asarray(goal, dtype=float) - np.asarray(agent_xy, dtype=float)
        turns_allowed = int(round(360.0 / max(1.0, float(self.cfg.agent.turn_deg))))
        if self._climb_align_turns >= turns_allowed:
            self._climb_aligned = True          # a full revolution: get on with it
            self.stats["climb_align_gave_up"] = self.stats.get("climb_align_gave_up", 0) + 1
            return None
        err = float(np.arctan2(axis[1], axis[0]) - agent_heading(frame.T_wc))
        err = np.degrees((err + np.pi) % (2 * np.pi) - np.pi)
        if abs(err) <= limit:
            self._climb_aligned = True
            self.stats["climb_align_turns"] = (
                self.stats.get("climb_align_turns", 0) + self._climb_align_turns)
            return None
        self._climb_align_turns += 1
        return "turn_left" if err > 0 else "turn_right"
    def _climb_turn_lock(self, frame: FrameData, agent_xy: np.ndarray,
                         goal: np.ndarray, action: str) -> str:
        """Suppress a turn that would overshoot, and stay suppressed.

        A turn is `turn_deg` (30 degrees). Correcting a 10-degree error with a
        30-degree turn leaves a 20-degree error the other way, which the next
        step corrects back: that is the oscillation on the stairs, and the
        traces measure it as a quarter of turns on the descent and nearly half
        on the ascent immediately reversing the previous one. So: within the
        deadband, go forward; the lock then holds until the error exceeds the
        release angle, so the agent does not chatter on the boundary.
        """
        deadband = float(getattr(self.cfg.agent, "climb_turn_deadband_deg", 0.0) or 0.0)
        moved = (
            float(np.linalg.norm(np.asarray(agent_xy, dtype=float) - self._climb_last_xy))
            if self._climb_last_xy is not None else 1e9
        )
        self._climb_last_xy = np.asarray(agent_xy, dtype=float).copy()
        if deadband <= 0.0 or not str(action).startswith("turn"):
            if action == FORWARD_ACTION:
                self._climb_turn_locked = True
            return action
        # Forward is not working: the mover's turn is how it gets around
        # whatever is in the way, and suppressing it walks into the banister.
        eps = float(getattr(self.cfg.agent, "climb_turn_stuck_eps_m", 0.0) or 0.0)
        if eps > 0.0 and self.nav._last_action == FORWARD_ACTION and moved < eps:
            self._climb_turn_locked = False
            self._climb_suppressed_run = 0
            self.stats["climb_turn_yield_stuck"] = (
                self.stats.get("climb_turn_yield_stuck", 0) + 1)
            return action
        cap = int(getattr(self.cfg.agent, "climb_turn_suppress_max", 0) or 0)
        if cap > 0 and self._climb_suppressed_run >= cap:
            self._climb_turn_locked = False
            self._climb_suppressed_run = 0
            self.stats["climb_turn_yield_run"] = (
                self.stats.get("climb_turn_yield_run", 0) + 1)
            return action
        release = float(getattr(self.cfg.agent, "climb_turn_release_deg", 0.0) or 0.0)
        if release <= 0.0:
            release = float(self.cfg.agent.turn_deg)
        delta = np.asarray(goal, dtype=float) - np.asarray(agent_xy, dtype=float)
        err = float(np.arctan2(delta[1], delta[0]) - agent_heading(frame.T_wc))
        err = abs(np.degrees((err + np.pi) % (2 * np.pi) - np.pi))
        if err <= (release if self._climb_turn_locked else deadband):
            self._climb_turn_locked = True
            self._climb_suppressed_run += 1
            self.stats["climb_turn_suppressed"] = (
                self.stats.get("climb_turn_suppressed", 0) + 1)
            return FORWARD_ACTION
        self._climb_turn_locked = False
        self._climb_suppressed_run = 0
        return action
    def _carrot_stalled(self, agent_xy: np.ndarray) -> bool:
        ref = getattr(self, "_climb_centroid_xy", None)
        if ref is None:
            return False
        distance = float(np.linalg.norm(agent_xy - ref))
        if self._climb_last_dist is None or abs(self._climb_last_dist - distance) > 0.2:
            self._climb_last_dist = distance
            self._climb_paused_steps = 0
        else:
            self._climb_paused_steps += 1
        if self._climb_paused_steps > 15:
            self._carrot_disable_end = True
        return self._climb_paused_steps > 30
    def _left_the_stairs(self, agent_xy: np.ndarray) -> bool:
        cells = getattr(self, "_climb_cells_xy", None)
        if cells is None or not len(cells):
            return True
        distance = float(np.linalg.norm(cells - agent_xy, axis=1).min())
        return distance > float(getattr(self.cfg.agent, "stair_exit_m", 0.5))
