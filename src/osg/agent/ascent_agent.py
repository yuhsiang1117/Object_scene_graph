"""ASCENT's navigation control flow, on OSG's perception.

Why a whole policy rather than another ported mechanism
-------------------------------------------------------
On `scenes20_ep0to4` ASCENT scores 70% sensor-only. OSG scores 63% *with* a
ground-truth navmesh and 42% sensor-only. ASCENT's navigation beats OSG's even
when OSG is allowed to cheat, so the gap is the pipeline rather than its
settings.

Four mechanisms were ported into `NavAgent` one at a time and three measured
null or worse (S31 carrot, S33 RedNet, S34 cadence). S36 says why that keeps
happening: the differences are structural, not parametric. ASCENT has **no state
machine** -- `ascent_policy.py:385-637` rebuilds every input and re-decides from
scratch on every step -- while `NavAgent` commits to a frontier for a median 23
steps and to an approach goal for the whole approach. A mechanism lifted out of
the first structure and dropped into the second loses the context that made it
work. S34 is the clearest case: ASCENT's re-selection cadence without its
`_force_frontier` damping thrashed 4002 times and cost 7 episodes.

So this subclasses `NavAgent` and overrides exactly one method, `_dispatch`.
Perception, mapping, the floor stack, the value map, the object layer, the stair
detector, the PointNav driver and the escape guard are all inherited unchanged,
which keeps the comparison to a single variable: the control flow.

The dispatch, against `ascent_policy.py:439-580`
------------------------------------------------
    goal = nearest point of the target's cloud      every step, with hysteresis
    if climbing:            climb the staircase
    elif not initialised:   12 x turn_left
    elif goal is None:      explore
    else:                   navigate to goal

Deliberate deviations, all of them keeping a measured OSG win
-------------------------------------------------------------
* **Candidate gate.** ASCENT commits to any cloud of the target class
  (`has_object`). OSG's `candidates()` gate (min_obs / score / bbox / evidence)
  is worth +3 on this split (S13, confirmed S15), so it stays.
* **Verifier.** ASCENT's `_double_check_goal` is a BLIP-2 cosine >= 0.15 gate
  (`map_controller.py:770-776`); `lavis` is not installed here. OSG's VLM
  verifier is the largest measured win in the log, so it is kept in its place.
* **Terminal stop and the 100-step abandon** are already ASCENT's, ported in
  S30, and are reused rather than rewritten.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import FrameData
from ..exploration.selector import frontier_goal_xy
from ..mapping.costmap import PLANE
from .nav_agent import FORWARD_ACTION, STOP_ACTION, TURN_ACTION, NavAgent, State


class AscentAgent(NavAgent):
    """`NavAgent` with ASCENT's dispatch in place of the FSM."""

    def reset(self, target_category: str) -> None:
        super().reset(target_category)
        self._ascent_goal_xy: Optional[np.ndarray] = None
        self._ascent_init_left = (
            int(round(360.0 / self.cfg.agent.turn_deg))
            if self.cfg.agent.initial_scan else 0
        )
        self._navigate_steps = 0

    # ------------------------------------------------------------- the goal

    def _object_goal(self, agent_xy: np.ndarray) -> Optional[np.ndarray]:
        """Nearest point of the target's surface cloud, re-aimed every step.

        Port of `get_best_object` (object_point_cloud_map.py:127-150). The
        hysteresis is the load-bearing part, not the re-aiming: without it the
        goal jitters with every mask update, and `PointNavDriver` resets its
        recurrent state on any goal move over 0.1 m, so a jittering goal would
        cost more than the staleness it fixes.

            move < 0.1 m                      -> keep the old goal
            move < 0.5 m while further than 2 m -> keep the old goal
        """
        track = (
            self.object_layer.get(self._candidate_id)
            if self._candidate_id is not None else None
        )
        if track is None or track.blacklisted:
            self._ascent_goal_xy = None
            return None
        xy = self.object_layer.nearest_point_xy(track, agent_xy)
        if xy is None:
            return self._ascent_goal_xy
        if self._ascent_goal_xy is None:
            self._ascent_goal_xy = np.asarray(xy, dtype=float).copy()
            return self._ascent_goal_xy
        delta = float(np.linalg.norm(xy - self._ascent_goal_xy))
        far = float(np.linalg.norm(agent_xy - xy)) > 2.0
        if delta < 0.1 or (delta < 0.5 and far):
            return self._ascent_goal_xy
        self._ascent_goal_xy = np.asarray(xy, dtype=float).copy()
        return self._ascent_goal_xy

    # ---------------------------------------------------------- the dispatch

    def _dispatch(self, frame: FrameData, y_obs: float, layer) -> str:
        agent_xy = frame.camera_position[list(PLANE)]

        if self.state == State.CLIMB:
            return self._do_climb(frame, y_obs)

        # A target is looked for on every step, not only in selected states --
        # ASCENT runs its whole perception stack and re-reads the object map
        # unconditionally (`ascent_policy.py:411-441`).
        if self._candidate_id is None:
            self._check_candidates()
            if self.state is State.CLIMB:
                return self._do_climb(frame, y_obs)

        # 12 x turn_left, as ASCENT's `_initialize` (:638-646). ASCENT places
        # this BEFORE its goal check (`:566-576`: the `not done_initializing`
        # branch precedes `elif goal is None`), so a target spotted during the
        # opening spin waits until the spin finishes. That ordering is kept.
        # The candidate check above still runs, so the target is recorded when
        # first seen rather than only after the scan.
        if self._ascent_init_left > 0:
            self._ascent_init_left -= 1
            self.state = State.EXPLORE
            return TURN_ACTION

        goal = self._object_goal(agent_xy)
        if goal is None:
            self.state = State.EXPLORE
            self._navigate_steps = 0
            return self._ascent_explore(frame, agent_xy)
        self.state = State.APPROACH
        return self._ascent_navigate(frame, agent_xy, goal)

    # ------------------------------------------------------------- exploring

    def _ascent_explore(self, frame: FrameData, agent_xy: np.ndarray) -> str:
        """Re-select every step, drive at the winner, never conclude arrival.

        ASCENT's `_explore` (`ascent_policy.py:648-711`). Three properties that
        `NavAgent` does not have, and that S34 showed have to arrive together:

        * the frontier is re-chosen every step;
        * `FrontierCommitState` is fed on every choice, so the sticky counter,
          `force_xy` and the retirement set actually accumulate -- this is the
          damping S34 left behind, and without it re-selection thrashes;
        * a network STOP is overwritten with a forward step (`:708-710`), and
          being inside the stop radius means nothing, because every frontier
          call passes `stop=False` (`:869-872`). A pursuit ends when the
          frontier stops being extracted or is retired, never by arriving.
        """
        self._select_new_frontier(frame)
        f = self._current_frontier
        if f is None:
            # Nothing selectable: keep turning so the map grows, which is what
            # ASCENT's own fallback does before it gives up on the floor.
            return TURN_ACTION
        if self.commit_state is not None:
            self.commit_state.observe(f, agent_xy)
            if self.commit_state.is_disabled(f.centroid_xy):
                self._current_frontier = None
                self.stats["ascent_frontier_retired"] = (
                    self.stats.get("ascent_frontier_retired", 0) + 1
                )
                return TURN_ACTION

        goal = frontier_goal_xy(f, self.costmap)
        if self.pointnav is None:
            action = self._follow_path(frame)
            return action if action is not None else TURN_ACTION
        nav = self.pointnav.step(goal)
        if nav.action is None:
            self.stats["ascent_explore_forced_forward"] = (
                self.stats.get("ascent_explore_forced_forward", 0) + 1
            )
            return FORWARD_ACTION
        return nav.action

    # ------------------------------------------------------------ navigating

    def _ascent_navigate(
        self, frame: FrameData, agent_xy: np.ndarray, goal: np.ndarray
    ) -> str:
        """Drive at the re-aimed goal; stop, creep or abandon.

        `ascent_policy.py:876-938`. The stop rule and the 100-step abandon are
        already ASCENT's -- ported in S30 as `_nearest_point_stop` and
        `_abandon_approach` -- so they are called rather than rewritten, which
        keeps this arm and `ascent_sensor` deciding terminal questions
        identically and leaves the control flow as the only difference.
        """
        self._navigate_steps += 1
        if (
            self._navigate_steps
            >= int(getattr(self.cfg.agent, "approach_abandon_steps", 100) or 100)
        ):
            self._ascent_goal_xy = None
            self._navigate_steps = 0
            return self._abandon_approach()

        stop_reason = self._nearest_point_stop(agent_xy)
        if stop_reason is not None:
            action = self._commit_terminal_stop(stop_reason, frame, det=None)
            if action is not None:
                return action

        if self.pointnav is None:
            action = self._follow_to(frame, goal)
            return action if action is not None else STOP_ACTION
        nav = self.pointnav.step(
            goal, creep_below=float(getattr(self.cfg.agent, "pointnav_approach_creep_m", 1.0))
        )
        if nav.action is None:
            # ASCENT force-forwards rather than concluding anything from a
            # network STOP during an approach (`:927`); the abandon counter and
            # the terminal rule above are what end this.
            self.stats["ascent_navigate_forced_forward"] = (
                self.stats.get("ascent_navigate_forced_forward", 0) + 1
            )
            return FORWARD_ACTION
        return nav.action
