"""The terminal approach: walk to the object and decide where to stop.

This is the state that ends an episode, and the whole of the success metric
lives in it. HM3D scores success as the geodesic distance from the final pose to
the nearest sampled GOAL VIEW POINT -- not to the object -- and those viewpoints
sit on rings at fixed radii (0.8 / 1.2 / 1.5 / 2.0 m). Three distance-based
stopping strategies all stalled at dtg 0.107-0.147 m before that was understood,
which is why the code below prefers driving ONTO a ring to closing on the object
until its depth crosses a threshold.

Two mechanisms here belong to the dynamic-scene line rather than to navigation:

  the arrival sweep. A viewpoint is a pose the object is visible FROM, but the
  follower arrives on whatever heading the path ended with, and one frame from
  one heading is a thin basis for deciding an object is gone. Measured both
  ways: concluding absence from the arrival frame abandoned a bowl that was
  exactly where the map said, while stopping without looking declared success on
  empty space.

  and what the sweep deliberately does NOT do: apply a negative reading per
  frame. Twelve looks at the same object from the same pose are not twelve
  independent observations -- same range, lighting and viewing angle on the same
  geometry -- so multiplying their likelihoods turns one correlated detector
  failure into overwhelming evidence of absence. Doing it dropped SR from 0.429
  to 0.286. The sweep's job is to give the detector a chance, not to vote.

Ownership: this policy owns everything only an approach uses -- the retreat
pose, the step budget, the sweep counters, the bbox calibration log and the
navigation diagnostics. `NavAgent` remains the single owner of FSM state
(`state`, `_goal_xy`, `_candidate_id`), which this handler reads and writes
through `self.nav` exactly as any state handler must.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import FrameData
from ..mapping.costmap import PLANE, cell_status, nearest_free_xy
from .state import STOP_ACTION, TURN_ACTION, State


class ApproachPolicy:
    def __init__(self, nav) -> None:
        self.nav = nav
        self.reset()

    def reset(self) -> None:
        # The last pose from which the target was confirmed visible, and the
        # retreat target when a step carries the agent behind an occluder the
        # 2D costmap line-of-sight check cannot see.
        self.last_good_xy: Optional[np.ndarray] = None
        self.steps_left = 0
        self.at_viewpoint = False
        self.scan_turns_left = 0
        self.scan_expected = 0
        # APPROACH's path-goal cache is separate from the agent's
        # `_goal_xy`/`_current_path` because APPROACH alternates between an
        # "advance toward the object" goal and a "retreat to the last visible
        # pose" goal within the same episode phase.
        self.path_goal: Optional[np.ndarray] = None
        self.last_follow_none_reason: Optional[str] = None
        # Calibration data for approach_stop_bbox_px (P1c): every bbox_px
        # observed during APPROACH, plus why the episode's approach ended.
        self.bbox_log: list = []
        self.stop_reason: Optional[str] = None
        # Why the agent stopped short of a correctly-mapped target: splits
        # path_consumed into planner-no-path vs controller-arrived and records
        # the approach geometry. See scripts/analyze_approach.py.
        self.diag: dict = {}
        # Retargeting: how many times this approach re-derived its goal from a
        # refined centre, and what each one moved. The log records the DECISION
        # (the goal actually moved) rather than the state, so a run where the
        # knob fired but changed nothing is legible as exactly that.
        self.retargets = 0
        self.retarget_log: list = []

    def step(self, frame: FrameData) -> str:
        """Walk toward the verified object while it stays visible.

        Every step re-runs the detector on the current pose:
        - visible and within the target metric range (median mask depth <=
          approach_stop_depth_m) -> close and in clear view, stop. Depth is the
          primary signal (bbox area is object-size-dependent); bbox is a
          fallback for when the mask carries no valid depth.
        - visible but still too far -> record this pose as good, advance one
          more step toward the object.
        - not visible -> if a previous pose was confirmed visible, retreat
          there (a step just carried us behind an occluder the 2D costmap
          LOS check cannot see, e.g. a desk edge) and stop; otherwise the
          object was never visible from this approach at all, so keep
          advancing toward it (there is nothing better to retreat to) until
          the deadline.
        """
        agent_xy = frame.camera_position[list(PLANE)]
        # Track how close the agent gets to its approach goal this episode.
        if self.diag and self.nav._goal_xy is not None:
            dg = float(np.linalg.norm(agent_xy - self.nav._goal_xy))
            cur = self.diag.get("min_dist_to_goal_m")
            if cur is None or dg < cur:
                self.diag["min_dist_to_goal_m"] = dg
        self.retarget(agent_xy)
        det = self.nav._best_target_detection(frame)

        if det is not None:
            self.last_good_xy = agent_xy.copy()
            x1, y1, x2, y2 = det.bbox_xyxy
            bbox_px = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            depth = self.nav._detection_depth(det, frame)
            # Log (step, bbox_px, depth) for terminal calibration.
            self.bbox_log.append(
                (self.nav.step_count, round(float(bbox_px), 1),
                 round(float(depth), 3) if depth is not None else None)
            )
            stop_reason: Optional[str] = None
            # When the goal IS a viewpoint, arriving at it is the stop
            # condition. A depth stop would fire en route -- the viewpoint sits
            # at 0.8-1.2 m and the depth threshold is 1.0 m -- and leave the
            # agent short of the pose success is actually measured at.
            if self.at_viewpoint:
                pass
            elif self.nav.cfg.agent.approach_depth_stop:
                if depth is not None:
                    if depth <= self.nav.cfg.agent.approach_stop_depth_m:
                        stop_reason = "depth"
                # fallback for when the mask carries no valid depth
                elif bbox_px >= self.nav.cfg.agent.approach_stop_bbox_px:
                    stop_reason = "bbox"
            if stop_reason is not None:
                # Terminal-view verification: the agent is close and the target
                # fills the view -- this live close-up is the decisive frame.
                # Ask the VLM before committing STOP; a rejection means the
                # detector locked onto a false positive, so blacklist it and
                # resume exploring rather than stopping on empty/wrong space.
                if self.nav._terminal_verify:
                    self.nav.stats["terminal_verify"] = (
                    self.nav.stats.get("terminal_verify", 0) + 1
                )
                    # Full live frame with the target boxed (scene context).
                    if not self.nav.verifier.verify_bbox(frame.rgb, det.bbox_xyxy, self.nav.target):
                        self.nav.stats["terminal_reject"] = (
                            self.nav.stats.get("terminal_reject", 0) + 1
                        )
                        if self.nav._candidate_id is not None:
                            self.nav.object_layer.blacklist(self.nav._candidate_id)
                        self.nav._candidate_id = None
                        self.nav._target_obj_xy = None
                        self.nav.state = State.EXPLORE
                        return TURN_ACTION
                self.nav.state = State.DONE
                self.stop_reason = stop_reason
                return STOP_ACTION
        elif (
            not self.nav._use_navmesh  # navmesh knows the path; a momentary FOV loss
            # while turning along it must NOT trigger a retreat, or the agent
            # oscillates (approach -> lose detection -> retreat -> re-detect ...)
            # until the deadline. Costmap mode keeps the LOS-occlusion retreat.
            and self.last_good_xy is not None
            and np.linalg.norm(agent_xy - self.last_good_xy) > 0.1
        ):
            action = self.follow_to(frame, self.last_good_xy)
            if action is not None:
                return action
            abandon = self.nav._absence_at_arrival(frame, "retreat")
            if abandon is not None:
                return abandon
            self.nav.state = State.DONE  # retreat path consumed/unreachable: stop here
            self.stop_reason = "retreat"
            return STOP_ACTION

        if self.nav.step_count > self.nav._goto_deadline or self.steps_left <= 0:
            abandon = self.nav._absence_at_arrival(frame, "deadline")
            if abandon is not None:
                return abandon
            self.nav.state = State.DONE
            self.stop_reason = "deadline"
            return STOP_ACTION
        self.steps_left -= 1
        self.last_follow_none_reason = None
        action = self.follow_to(frame, self.nav._goal_xy)
        if action is None:  # path consumed or unreachable: as close as it gets
            turn = self.scan_at_viewpoint(det, frame)
            if turn is not None:
                return turn
            # ...unless it is not as close as it gets. On the navmesh the
            # follower returns None for arrived AND for unreachable, and the
            # approach has been treating both as an arrival: it stops, which
            # ends the episode or burns an attempt.
            #
            # Measured on 00848: the agent commits at step 1 to a track 0.81 m
            # from the true object, the follower reports None on step 5 while
            # the agent is still 6.4 m away, and it STOPS. Three attempts go the
            # same way and the episode is over at step 78 with 420 unspent. Four
            # to six episodes per condition end like that, and none of them ever
            # scores.
            #
            # The frontier side has had this distinction since `frontier_reach_m`
            # -- "a pursuit that ended retires its frontier, whichever way it
            # ended", but only an arrival within reach counts as reaching. This
            # is the same test for the approach.
            tol = float(self.nav.cfg.agent.approach_false_arrival_m)
            if tol > 0.0 and self.nav._goal_xy is not None:
                error = float(np.linalg.norm(agent_xy - self.nav._goal_xy))
                if error > tol:
                    self.nav.stats["approach_false_arrival"] = (
                        self.nav.stats.get("approach_false_arrival", 0) + 1
                    )
                    track = (
                        self.nav.object_layer.get(self.nav._candidate_id)
                        if self.nav._candidate_id is not None else None
                    )
                    if track is not None:
                        # Could not get there. That is evidence about this
                        # candidate and belongs in the identity channel, which
                        # retires it after two -- not a stop, and not a
                        # blacklist.
                        track.identity_rejections += 1
                    self.nav._candidate_id = None
                    self.nav._target_obj_xy = None
                    self.nav.state = State.EXPLORE
                    return TURN_ACTION
            abandon = self.nav._absence_at_arrival(frame, "path_consumed")
            if abandon is not None:
                return abandon
            self.nav.state = State.DONE
            self.stop_reason = "path_consumed"
            if self.diag is not None:
                self.diag["path_consumed_cause"] = self.last_follow_none_reason
                if self.last_follow_none_reason == "planner_no_path":
                    self.diag["plan_fail"] = self.diag.get("plan_fail", 0) + 1
            return STOP_ACTION
        return action

    def scan_at_viewpoint(self, det, frame: FrameData) -> Optional[str]:
        """Sweep in place on arrival, until the target is seen or the budget ends.

        A viewpoint is a pose the object is visible FROM, but the navmesh
        follower arrives on whatever heading the path happened to end with, and
        one frame from one heading is a thin basis for deciding an object is
        gone. Measured both ways: concluding absence from the arrival frame
        abandoned a bowl that was exactly where the map said, while stopping
        without looking declared success on empty space. A full sweep costs a
        dozen steps and makes the detector's silence mean something.
        """
        if not self.at_viewpoint or det is not None:
            return None
        if self.scan_turns_left <= 0:
            return None
        # Turn TOWARD the object, not blindly. A full blind sweep ends on the
        # heading it started from -- the navmesh follower's arrival heading --
        # so the absence decision was being taken on whatever happened to be in
        # front. Captured at the moment of one such decision on a CORRECT map:
        # a wall and a painting, with the bowl's table off frame to the right.
        # The VLM answered "bare" and was right about the pixels it was shown.
        from ..planning.controller import TURN_LEFT, TURN_RIGHT, _wrap, agent_heading

        if self.nav._target_obj_xy is not None:
            agent_xy = frame.camera_position[list(PLANE)]
            to_obj = self.nav._target_obj_xy - agent_xy
            if float(np.linalg.norm(to_obj)) > 1e-3:
                err = _wrap(
                    float(np.arctan2(to_obj[1], to_obj[0])) - agent_heading(frame.T_wc)
                )
                if abs(err) > np.radians(15.0):
                    self.scan_turns_left -= 1
                    self.nav.stats["approach_face_turns"] = (
                        self.nav.stats.get("approach_face_turns", 0) + 1
                    )
                    return TURN_RIGHT if err > 0 else TURN_LEFT
                # Facing it and still nothing: that is the informative frame, so
                # decide here rather than sweeping on past it.
                self.scan_turns_left = 0
                return None
        # Record whether the object was EXPECTED at any heading of the sweep.
        # The decision below used to test only the frame the sweep ended on --
        # after a full circle, the arrival heading again, which need not face
        # the object -- so the agent arrived at a ghost, swept right past it and
        # concluded nothing. Measured: all nine cross-anchor episodes stopped at
        # 29-58 steps with 440+ unspent, and the re-search never ran once.
        pf = self.nav.object_layer.presence_filter
        track = (
            self.nav.object_layer.get(self.nav._candidate_id)
            if self.nav._candidate_id is not None else None
        )
        if pf is not None and track is not None:
            if pf.expectation(track, frame, center_only=True) is not None:
                self.scan_expected += 1

        # Note what is deliberately NOT done here: applying a negative reading
        # per sweep frame. Twelve looks at the same object from the same pose
        # are not twelve independent observations -- same range, same lighting,
        # same viewing angle on the same geometry -- so multiplying their
        # likelihoods turns one correlated detector failure into overwhelming
        # evidence of absence. Measured: doing it dropped SR from 0.429 to
        # 0.286 by abandoning a bowl that was exactly where the map said. The
        # sweep's job is to give the detector a chance, not to vote.
        self.scan_turns_left -= 1
        self.nav.stats["approach_scan_turns"] = self.nav.stats.get("approach_scan_turns", 0) + 1
        return TURN_ACTION

    def retarget(self, agent_xy: Optional[np.ndarray]) -> None:
        """Move the goal if the object did.

        The centre `start()` aimed at is the ellipsoid's estimate at commit
        time, which is the worst one this approach will ever hold: the track is
        refined from every keyframe (`min_obs_for_refine`/`refine_every`), and
        the keyframes that arrive during the walk are the closest and best
        framed of the episode. Leaving the goal where it was throws that away at
        the one moment it is worth the most -- success is scored at 0.18 m from
        an authored viewpoint, so a viewpoint computed on a ring 0.33 m off
        centre cannot score however well the object was found.

        Only the goal moves. The deadline, the step budget and `last_good_xy`
        are all left alone: this is the same approach, re-aimed, not a new one.
        """
        tol = float(getattr(self.nav.cfg.agent, "approach_retarget_m", 0.0) or 0.0)
        if tol <= 0.0 or self.nav._target_obj_xy is None:
            return
        if self.retargets >= int(getattr(self.nav.cfg.agent,
                                         "approach_retarget_max", 3)):
            return
        if self.nav._candidate_id is None:
            return
        track = self.nav.object_layer.get(self.nav._candidate_id)
        if track is None:
            return
        centre = np.asarray(
            self.nav.object_layer.center_of(track), dtype=float)[list(PLANE)]
        moved = float(np.linalg.norm(centre - self.nav._target_obj_xy))
        if moved < tol:
            return
        before = None if self.nav._goal_xy is None else self.nav._goal_xy.copy()
        self._aim(centre, agent_xy)
        if before is None or self.nav._goal_xy is None:
            return
        goal_moved = float(np.linalg.norm(self.nav._goal_xy - before))
        self.retargets += 1
        self.nav.stats["approach_retargeted"] = (
            self.nav.stats.get("approach_retargeted", 0) + 1
        )
        self.retarget_log.append(
            (self.nav.step_count, round(moved, 3), round(goal_moved, 3))
        )
        # The path was planned to the old goal; drop it so the next follow_to
        # replans, or the agent walks the rest of the way to a stale pose.
        self.nav._current_path = None
        self.path_goal = None
        if self.diag is not None:
            self.diag["retargets"] = self.retargets
            self.diag["obj_xy"] = [float(x) for x in centre]
            self.diag["goal_xy"] = [float(x) for x in self.nav._goal_xy]
            self.diag["goal_to_obj_m"] = float(
                np.linalg.norm(self.nav._goal_xy - centre))

    def _aim(self, obj_xy: np.ndarray,
             agent_xy: Optional[np.ndarray] = None) -> None:
        """Where to stand to see `obj_xy`, and the goal that gets there.

        Split out of `start()` so an approach already under way can be
        re-aimed at a refined centre through exactly the same viewpoint
        logic -- a second copy of this ring search would be a second thing
        to keep in step with `ViewpointPlanner`.
        """
        self.at_viewpoint = False
        if self.nav._use_navmesh and self.nav.cfg.agent.approach_to_viewpoint:
            # HM3D scores success as the distance from the final pose to the
            # nearest GOAL VIEW POINT, and those are sampled on rings at fixed
            # radii around the object. Stopping when the target's depth reaches
            # approach_stop_depth_m puts the agent at 1.0 m -- radially between
            # the 0.8 m and 1.2 m rings, about 0.2 m from the nearest viewpoint
            # either way. Measured: four of seven batch episodes ended at 0.18,
            # 0.19, 0.21 and 0.28 m against a 0.18 m radius, having found the
            # object. ViewpointPlanner samples the SAME radii, so driving to one
            # of its poses puts the agent ON a ring, where the only error left
            # is angular -- at worst half the sampling step, about 0.10 m.
            view_xy = self.nav.viewpoint_planner.approach_viewpoint(obj_xy, self.nav.costmap)
            if view_xy is not None:
                self.nav.stats["approach_viewpoint"] = (
                    self.nav.stats.get("approach_viewpoint", 0) + 1
                )
            else:
                # Not observable from mapped FREE space yet. The fallback used to
                # be the object's own centre, and that is unwinnable by
                # construction: a tabletop object's centre is an occupied cell
                # inside the furniture, so the follower stalls against it and the
                # agent ends up INSIDE the innermost 0.8 m viewpoint ring, where
                # HM3D cannot score a success however well the object was found.
                # Measured over 42 episodes: 23 approaches took this branch, and
                # the 10 episodes that ended on such a goal scored SR 0.100
                # against 0.516 for the rest, stalling at 0.54-1.51 m.
                #
                # So relax the viewpoint search instead of abandoning it --
                # unmapped is not unstandable, and a ray that clips the object's
                # own table is not a blocked view. Any pose ON a ring beats any
                # pose off it.
                view_xy = self.nav.viewpoint_planner.approach_viewpoint(
                    obj_xy, self.nav.costmap, require_line_of_sight=False, allow_unknown=True
                )
                self.nav.stats["approach_viewpoint_none"] = (
                    self.nav.stats.get("approach_viewpoint_none", 0) + 1
                )
                if view_xy is not None:
                    self.nav.stats["approach_viewpoint_relaxed"] = (
                        self.nav.stats.get("approach_viewpoint_relaxed", 0) + 1
                    )
            if view_xy is not None:
                self.nav._goal_xy = np.asarray(view_xy, dtype=float).copy()
                self.at_viewpoint = True
            else:
                # Every ring pose is out of bounds. The nearest free cell is
                # still a cell the agent can stand in, which the object's own
                # centre is not.
                self.nav._goal_xy = nearest_free_xy(self.nav.costmap, obj_xy)
                self.nav.stats["approach_goal_nearest_free"] = (
                    self.nav.stats.get("approach_goal_nearest_free", 0) + 1
                )
        elif self.nav._use_navmesh:
            # Navigate to the object itself; the navmesh snaps to the nearest
            # standable point (effectively a viewpoint), like old /goal_object.
            self.nav._goal_xy = obj_xy.copy()
        elif self.nav.cfg.agent.approach_navigable_goal and agent_xy is not None:
            self.nav._goal_xy = self._standoff_goal(obj_xy, agent_xy)
        else:
            self.nav._goal_xy = nearest_free_xy(self.nav.costmap, obj_xy)
        self.nav._target_obj_xy = obj_xy.copy()

    def start(
        self,
        obj_xy: np.ndarray,
        agent_xy: Optional[np.ndarray] = None,
        floor_y: Optional[float] = None,
    ) -> None:
        # Height to snap the navmesh goal at for the rest of this approach.
        # None keeps the legacy "use the agent's own height" behaviour.
        self.nav._goal_floor_y_cache = floor_y
        self._aim(obj_xy, agent_xy)
        self.retargets = 0
        self.retarget_log = []
        self.scan_turns_left = int(self.nav.cfg.agent.approach_scan_turns)
        self.scan_expected = 0
        self.nav.state = State.APPROACH
        self.nav._current_path = None
        self.path_goal = None
        if self.nav._use_navmesh:
            # Navmesh drives the FULL distance to the object (no viewpoint
            # pre-positioning), so the short-leg cap (approach_max_steps ~= 3 m)
            # cuts the approach off while the target is still in view. Let it
            # navigate to the object, bounded only by a generous deadline.
            self.nav._goto_deadline = (
                self.nav.step_count + self.nav.cfg.agent.navmesh_approach_steps
            )
            self.steps_left = 10 ** 9
        else:
            self.nav._goto_deadline = self.nav.step_count + 100
            self.steps_left = self.nav.cfg.agent.approach_max_steps
        # Snapshot the terminal approach for navigation diagnostics: the goal
        # cell (nearest-free to the mapped object), how far it sits from the
        # object, the agent's start distance to it, and the costmap status of
        # the goal cell -- an UNKNOWN/OCCUPIED goal cell explains an
        # unreachable-path stop. min_dist_to_goal is filled in per step.
        self.diag = {
            "goal_xy": [float(x) for x in self.nav._goal_xy],
            "obj_xy": [float(x) for x in obj_xy],
            "goal_to_obj_m": float(np.linalg.norm(self.nav._goal_xy - obj_xy)),
            "goal_cell": cell_status(self.nav.costmap, self.nav._goal_xy),
            "start_step": self.nav.step_count,
            "min_dist_to_goal_m": None,
            "plan_fail": 0,
            "path_consumed_cause": None,
        }
        self.last_good_xy = None

    def follow_to(self, frame: FrameData, goal_xy: np.ndarray) -> Optional[str]:
        """Follow a path to an explicit goal, replanning when the goal
        changes (APPROACH alternates between an advance goal and a retreat
        goal within the same state, unlike the other terminal states which
        have one fixed goal for their whole visit).

        Used only by APPROACH, so tightens both the planner's and the
        controller's stopping tolerance beyond the loose defaults used for
        frontier/verify-view travel (P1f): geometry analysis against
        HM3D's actual view_points showed several near-miss episodes
        stopped within 5-8 cm (straight-line) of a real view_point, yet
        habitat's geodesic distance_to_goal still read 0.15-0.17 m --
        the default 0.2-0.3 m tolerances left slack for a short geodesic
        detour around a nearby thin obstacle to blow the 0.13 m success
        radius even when we were geometrically almost there.
        """
        if self.nav._use_navmesh:
            # navmesh drives to the object; None = arrived
            return self.nav._nav_fn(goal_xy, self.nav._goal_floor_y_cache)
        need_replan = (
            self.nav._current_path is None
            or self.path_goal is None
            or np.linalg.norm(self.path_goal - goal_xy) > 0.05
        )
        if need_replan:
            self.nav._plan_to(
                frame, goal_xy,
                goal_tolerance_m=self.nav.cfg.agent.approach_goal_tolerance_m,
            )
            self.path_goal = goal_xy.copy() if self.nav._current_path is not None else None
            if self.nav._current_path is None:
                # planner could not reach goal_xy (costmap disconnected /
                # goal in unknown/inflated space) -- distinct from the
                # controller reporting arrival below.
                self.last_follow_none_reason = "planner_no_path"
                return None
        action = self.nav.controller.act(
            frame.T_wc, self.nav._current_path,
            arrival_tol_m=self.nav.cfg.agent.approach_arrival_tol_m,
        )
        if action is None:
            self.nav._current_path = None
            self.path_goal = None
            # controller consumed the path (thinks it arrived at goal_xy) --
            # a false "arrival" here means the planned path was a stub / ended
            # short of the true goal.
            self.last_follow_none_reason = "controller_arrived"
        return action

    def _standoff_goal(self, obj_xy: np.ndarray, agent_xy: np.ndarray) -> np.ndarray:
        """Navigable approach goal: a standoff point at approach_standoff_m from
        the object along the ray toward the agent (the side the object was
        observed from -> open, reachable space), snapped to the nearest free
        cell. Avoids the enclosed-pocket goals that _nearest_free_xy produces
        by placing the goal in front of the object rather than hard against it.
        """
        standoff = float(self.nav.cfg.agent.approach_standoff_m)
        to_agent = agent_xy - obj_xy
        dist = float(np.linalg.norm(to_agent))
        if dist < 1e-3:
            return nearest_free_xy(self.nav.costmap, obj_xy)
        # If the agent is already closer than the standoff, keep the goal at the
        # standoff (do not push it behind the agent past the object).
        cand = obj_xy + (to_agent / dist) * min(standoff, dist)
        return nearest_free_xy(self.nav.costmap, cand)
