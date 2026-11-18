"""A close look at a surface: the frame a decision is allowed to rest on.

Two decisions in this agent conclude something from a single frame taken
wherever the follower happened to stop. The search retires a surface's belief
on arrival, and the absence sensor abandons a committed track when the
detector says nothing at the goal. Measured on the released benchmark
(`docs/history/DUALMAP_PROTOCOL_EARLY_ARMS.md`, "What the old headline was made of"), both
frames are usually the wrong ones:

* Across the 32 cross-anchor failures without a detection, the search reached
  a container within 1.5 m of the object in 6, five of them detector walls;
  among the 24 it could have converted, once. Twelve had the object in frame
  at 2-4 m during `goto_frontier`, where in-situ recall is 0.28.
* In-anchor, the first commit is already within 1 m of the moved object in
  23 of 44 episodes, and the ones that fail arrive at the tight ring, 0.35 to
  0.65 m out, where a 79-degree camera frames 0.3-0.5 m of table -- and the
  object moved a median 0.7 m.

`CloseLookPolicy` is one primitive for both: drive to a facing pose on a
1.5 m ring around the surface (where recall peaks at 0.58 and the frame is
2.4 m wide), turn until the surface is inside 15 degrees, hold a few
keyframes with the detector on, and only then let the conclusion happen. It
is a navigation policy, not a model: the price is steps, and every one of
them is counted (`close_look_*` stats, `close_look_log`).

Two entry points, each behind its own flag so an A/B can hold one still:

* `maybe_opportunistic` (`agent.close_look_opportunistic`) -- on a keyframe in
  EXPLORE / GOTO_FRONTIER, an affording container within
  `close_look_trigger_range_m`, in frame and not yet looked at, is worth a
  detour. The pursuit it interrupts is resumed afterwards.
* `before_absence` (`agent.close_look_before_absence`) -- the approach is
  about to conclude absence at a track it never saw live; look first, and if
  the target shows up, re-aim the approach instead.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import FrameData
from ..exploration.strategy import container_in_view
from ..graph.priors import affords
from ..mapping.costmap import PLANE, nearest_free_xy
from ..planning.controller import TURN_LEFT, TURN_RIGHT, _wrap, agent_heading
from .state import TURN_ACTION, State

# Four turns that end on the heading they started from, so a hold of four
# steps leaves the agent as it was; the surface stays in a 79-degree frame at
# 30 degrees off-axis, which is the point of holding rather than spinning.
HOLD_PATTERN = (TURN_LEFT, TURN_RIGHT, TURN_RIGHT, TURN_LEFT)
FACING_TOL_RAD = np.radians(15.0)


def _container_radius_m(nav, node) -> float:
    from .nav_agent import _horizontal_radius_m  # local: nav_agent imports this module

    radius = 0.0
    for tid in getattr(node, "track_ids", []) or []:
        track = nav.object_layer.get(int(tid))
        if track is not None:
            radius = max(radius, _horizontal_radius_m(track))
    return radius


class CloseLookPolicy:
    def __init__(self, nav) -> None:
        self.nav = nav
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.cid: Optional[int] = None
        self.centre_xy: Optional[np.ndarray] = None
        self.goal_xy: Optional[np.ndarray] = None
        self.phase: Optional[str] = None
        self.started_step = 0
        self.face_turns_left = 0
        self.hold_left = 0
        self.hold_index = 0
        self.resume: Optional[str] = None  # "explore" | "absence"
        self.absence_reason: Optional[str] = None
        self.saved_goal_xy: Optional[np.ndarray] = None
        self.saved_state: Optional[State] = None
        self.detected = False
        # What has had its look this episode. Container ids are positive; a
        # look aimed at a committed track is keyed by the negative track id so
        # the two namespaces cannot collide.
        self.looked: set = set()
        self.log: list = []
        self.count = 0

    @property
    def cfg(self):
        return self.nav.cfg.agent

    def can_start(self) -> bool:
        return self.count < int(self.cfg.close_look_max_per_episode)

    # ---------------------------------------------------------------- triggers

    def maybe_opportunistic(self, frame: FrameData) -> bool:
        """A surface worth a detour is in view: start looking at it."""
        if not bool(self.cfg.close_look_opportunistic):
            return False
        nav = self.nav
        if nav.state not in (State.EXPLORE, State.GOTO_FRONTIER):
            return False
        if getattr(nav.floors, "pursuing", False):
            return False  # a portal pursuit is not interrupted for a table
        if not self.can_start():
            return False
        sg = nav.scene_graph
        containers = getattr(sg, "containers", {}) or {}
        if not containers:
            return False
        floor_id = int(nav.floors.current_id)
        floor_node = (getattr(sg, "floors", {}) or {}).get(floor_id)
        floor_y = float(getattr(floor_node, "height_y", 0.0))
        rng = float(self.cfg.close_look_trigger_range_m)
        skip = getattr(nav.exploration, "search_container", None)
        inspected = getattr(nav.exploration, "inspected", set())
        by_belief = bool(self.cfg.close_look_by_belief)
        beliefs = nav.exploration.surface_beliefs(nav._world(frame)) if by_belief else {}
        min_belief = float(self.cfg.close_look_min_belief)
        best = None
        for cid, node in containers.items():
            cid = int(cid)
            if cid in self.looked or cid in inspected:
                continue
            if skip is not None and cid == int(skip):
                continue  # the search is already driving there to inspect it
            if int(getattr(node, "floor_id", getattr(node, "floor", 0))) != floor_id:
                continue
            if not affords(nav.target, float(node.top_h) - floor_y, float(node.area_m2)):
                continue
            z = container_in_view(frame, node, rng)
            if z is None:
                continue
            belief = beliefs.get(cid, 0.0)
            if by_belief and belief < min_belief:
                nav.stats["close_look_below_belief"] = (
                    nav.stats.get("close_look_below_belief", 0) + 1
                )
                continue
            # Nearest first, or the most believed first: the key decides.
            key = (-belief, z) if by_belief else (z,)
            if best is None or key < best[0]:
                best = (key, z, cid, node, belief)
        if best is None:
            return False
        _, z, cid, node, belief = best
        centre_xy = np.asarray(node.center, dtype=float)[list(PLANE)]
        self.start(cid, centre_xy, _container_radius_m(nav, node), "explore",
                   label=str(node.label), trigger_range_m=z,
                   belief=belief if by_belief else None)
        return True

    def before_absence(self, frame: FrameData, reason: str) -> Optional[str]:
        """The approach is about to conclude absence; look first.

        Returns the first action of the look, or None to let the absence
        decision proceed as it always has.
        """
        if not bool(self.cfg.close_look_before_absence):
            return None
        if not self.can_start():
            return None
        nav = self.nav
        if nav._candidate_id is None or nav._seen_this_approach():
            return None  # seen live during this approach: not an absence question
        track = nav.object_layer.get(nav._candidate_id)
        if track is None:
            return None
        key = -int(track.id)
        if key in self.looked:
            return None  # one look per track; the second silence is the answer
        from .nav_agent import _horizontal_radius_m

        centre_xy = np.asarray(nav.object_layer.center_of(track), dtype=float)[list(PLANE)]
        self.start(key, centre_xy, _horizontal_radius_m(track), "absence",
                   reason=reason, label=str(track.label))
        return self.step(frame)

    # ------------------------------------------------------------------ lifecycle

    def start(self, cid: int, centre_xy: np.ndarray, radius_m: float, resume: str,
              reason: Optional[str] = None, label: Optional[str] = None,
              trigger_range_m: Optional[float] = None,
              belief: Optional[float] = None,
              exclude_xy: Optional[np.ndarray] = None) -> None:
        nav = self.nav
        ring = [float(self.cfg.close_look_ring_m)]
        exclude = None if exclude_xy is None else [np.asarray(exclude_xy, dtype=float)]
        view = nav.viewpoint_planner.approach_viewpoint(
            centre_xy, nav.costmap, obj_radius_m=radius_m, radii=ring,
            exclude=exclude, exclude_radius_m=1.0,
        )
        planned = "strict"
        if view is None:
            view = nav.viewpoint_planner.approach_viewpoint(
                centre_xy, nav.costmap, obj_radius_m=radius_m, radii=ring,
                require_line_of_sight=False, allow_unknown=True,
                exclude=exclude, exclude_radius_m=1.0,
            )
            planned = "relaxed"
        if view is None:
            view = nearest_free_xy(nav.costmap, centre_xy)
            planned = "nearest_free"
        self.active = True
        self.cid = int(cid)
        self.centre_xy = np.asarray(centre_xy, dtype=float).copy()
        self.goal_xy = np.asarray(view, dtype=float).copy()
        self.phase = "goto"
        self.started_step = int(nav.step_count)
        self.face_turns_left = int(self.cfg.close_look_face_turns)
        self.hold_left = int(self.cfg.close_look_hold_steps)
        self.hold_index = 0
        self.resume = resume
        self.absence_reason = reason
        self.detected = False
        self.saved_goal_xy = None if nav._goal_xy is None else nav._goal_xy.copy()
        self.saved_state = nav.state
        self.count += 1
        nav.stats["close_look_started"] = nav.stats.get("close_look_started", 0) + 1
        nav.stats[f"close_look_{resume}"] = nav.stats.get(f"close_look_{resume}", 0) + 1
        nav._goal_xy = self.goal_xy.copy()
        nav._current_path = None
        nav.approach.path_goal = None
        if resume == "explore":
            nav._goal_floor_y_cache = None  # own floor; a stale portal height must not snap it
        nav.state = State.CLOSE_LOOK
        self.log.append({
            "step": self.started_step,
            "container_id": int(cid),
            "label": label,
            "resume": resume,
            "reason": reason,
            "goal": planned,
            "goal_xy": [float(v) for v in self.goal_xy],
            "centre_xy": [float(v) for v in self.centre_xy],
            "trigger_range_m": None if trigger_range_m is None else round(float(trigger_range_m), 3),
            "belief": None if belief is None else round(float(belief), 3),
        })

    def abort(self) -> None:
        """A new attempt or a reset: forget the look in progress, keep the ledger."""
        if self.active and self.log:
            self.log[-1].update({"end_step": int(self.nav.step_count), "aborted": True})
        self.active = False
        self.phase = None

    def interrupted(self) -> None:
        """A candidate commit pre-empted the look: that is the look succeeding."""
        self.detected = True
        self._close(pre_empted=True)

    def step(self, frame: FrameData) -> str:
        nav = self.nav
        agent_xy = frame.camera_position[list(PLANE)]
        if self.resume == "absence":
            # The approach has ended, so nothing else runs the detector on
            # this frame; the look does, and it tells the absence sensor that
            # the object was expected in view, as the arrival sweep would.
            self._note_expectation(frame)
            if nav._best_target_detection(frame) is not None:
                self.detected = True
                return self.finish(frame)
        if self.phase == "goto":
            if nav.step_count - self.started_step >= int(self.cfg.close_look_max_steps):
                nav.stats["close_look_goto_timeout"] = (
                    nav.stats.get("close_look_goto_timeout", 0) + 1
                )
                self.phase = "face"
            else:
                action = nav.approach.follow_to(frame, self.goal_xy)
                if action is not None:
                    return action
                self.phase = "face"
        if self.phase == "face":
            to_surface = self.centre_xy - agent_xy
            if float(np.linalg.norm(to_surface)) > 1e-3 and self.face_turns_left > 0:
                err = _wrap(float(np.arctan2(to_surface[1], to_surface[0]))
                            - agent_heading(frame.T_wc))
                if abs(err) > FACING_TOL_RAD:
                    self.face_turns_left -= 1
                    nav.stats["close_look_face_turns"] = (
                        nav.stats.get("close_look_face_turns", 0) + 1
                    )
                    return TURN_RIGHT if err > 0 else TURN_LEFT
            self.phase = "hold"
        if self.phase == "hold" and self.hold_left > 0:
            self.hold_left -= 1
            action = HOLD_PATTERN[self.hold_index % len(HOLD_PATTERN)]
            self.hold_index += 1
            return action
        return self.finish(frame)

    def finish(self, frame: FrameData) -> str:
        nav = self.nav
        agent_xy = frame.camera_position[list(PLANE)]
        self._close(pre_empted=False, final_range_m=float(np.linalg.norm(agent_xy - self.centre_xy)))
        if self.resume == "absence":
            return self._finish_absence(frame)
        return self._finish_explore(frame)

    # ------------------------------------------------------------------ helpers

    def _note_expectation(self, frame: FrameData) -> None:
        nav = self.nav
        pf = nav.object_layer.presence_filter
        track = nav.object_layer.get(nav._candidate_id) if nav._candidate_id is not None else None
        if pf is not None and track is not None:
            if pf.expectation(track, frame, center_only=True) is not None:
                nav.approach.scan_expected += 1

    def _close(self, pre_empted: bool, final_range_m: Optional[float] = None) -> None:
        nav = self.nav
        steps = int(nav.step_count) - self.started_step
        if self.log:
            self.log[-1].update({
                "end_step": int(nav.step_count),
                "steps": steps,
                "detected": bool(self.detected),
                "pre_empted": bool(pre_empted),
                "phase": self.phase,
                "final_range_m": None if final_range_m is None else round(final_range_m, 3),
            })
        self.looked.add(int(self.cid))
        nav.stats["close_look_steps"] = nav.stats.get("close_look_steps", 0) + steps
        key = "close_look_detected" if self.detected else "close_look_silent"
        nav.stats[key] = nav.stats.get(key, 0) + 1
        if self.cid is not None and self.cid >= 0:
            nav.exploration.close_looked(int(self.cid), int(nav.step_count), detected=self.detected)
        self.active = False
        self.phase = None

    def _finish_explore(self, frame: FrameData) -> str:
        nav = self.nav
        nav._current_path = None
        if self.saved_goal_xy is not None and self.saved_state is State.GOTO_FRONTIER:
            # Resume the pursuit the look interrupted, frontier or surface.
            nav._goal_xy = self.saved_goal_xy.copy()
            nav.state = State.GOTO_FRONTIER
            action = nav._follow_path(frame)
            if action is not None:
                return action
            nav.exploration.current_frontier = None
        nav._goal_xy = None
        nav.state = State.EXPLORE
        return nav._act_inner_post_transition(frame)

    def _finish_absence(self, frame: FrameData) -> str:
        nav = self.nav
        track = nav.object_layer.get(nav._candidate_id) if nav._candidate_id is not None else None
        if track is None:
            nav._candidate_id = None
            nav._target_obj_xy = None
            nav._goal_xy = None
            nav.state = State.EXPLORE
            return TURN_ACTION
        centre = np.asarray(nav.object_layer.center_of(track), dtype=float)
        if self.detected:
            # The object is there, and the track has been refined by the frames
            # the look produced: aim the approach at it again.
            nav.stats["close_look_reapproach"] = nav.stats.get("close_look_reapproach", 0) + 1
            nav._start_approach(centre[list(PLANE)], floor_y=nav.floors.goal_floor_y(centre))
            return nav._do_approach(frame)
        abandon = nav._absence_at_arrival(frame, self.absence_reason or "close_look",
                                          from_look=True)
        if abandon is not None:
            return abandon
        # The reading did not abandon (the VLM saw it, or nothing was expected).
        # The stop the approach was about to make still stands -- from the ring
        # it had reached, not from here. `looked` holds the track's key, so the
        # second silent arrival goes straight to the sensor as it always did.
        nav.stats["close_look_stop_kept"] = nav.stats.get("close_look_stop_kept", 0) + 1
        nav._start_approach(centre[list(PLANE)], floor_y=nav.floors.goal_floor_y(centre))
        return nav._do_approach(frame)
