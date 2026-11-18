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
from ..verification.verifier import note_vlm
from .state import STOP_ACTION, TURN_ACTION, State


class ApproachPolicy:
    def __init__(self, nav) -> None:
        self.nav = nav
        self.reset()

    def reset(self) -> None:
        # Horizontal extent of the track under approach; see _ring_offset_m.
        self.obj_radius_m = 0.0
        # The last pose from which the target was confirmed visible, and the
        # retreat target when a step carries the agent behind an occluder the
        # 2D costmap line-of-sight check cannot see.
        self.last_good_xy: Optional[np.ndarray] = None
        # How far away the target was at that sighting (median mask depth, m;
        # None when the depth was invalid). A sighting from across the room
        # is not the same evidence as one at arm's length -- see
        # NavAgent._seen_this_approach.
        self.last_good_range_m: Optional[float] = None
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
        # Closing the last metre (`agent.approach_close_last_metre_m`): walking
        # from the reached viewpoint to the navmesh point nearest the track,
        # and whether that walk has been made for this approach.
        self.closing = False
        self.closed = False
        self.close_start_xy: Optional[np.ndarray] = None
        self.close_retreating = False
        # The closest the agent has actually stood to the committed track during
        # this approach, and whether it has been walked back to. The pose an
        # approach ENDS in is not the closest one it reached: measured over 81
        # trials the agent gives up a median 0.18 m between the two, and on the
        # trials it loses, 0.04-1.72 m. Ten of sixteen losses had come inside
        # 1.0 m and every one of them stopped outside it.
        self.best_xy: Optional[np.ndarray] = None
        self.returning = False
        self.returned = False
        self._return_from_d = float("inf")
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
        self._note_best(agent_xy)
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
            self.last_good_range_m = None if depth is None else float(depth)
            gate = getattr(self.nav, "stop_gate", None)
            if gate is not None:
                gate.note_sighting(frame, det, self.last_good_range_m)
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
                # Arriving at the viewpoint IS the stop condition -- but only
                # the follower ever said so, and it never says so while the
                # agent is sitting ON the goal. Measured on 00848's
                # cross_anchor_01 tin can, an episode where everything upstream
                # worked: the can found at step 416 (15 of 22 keyframes, track
                # 0.039 m from truth, p=0.95), the viewpoint reached to 0.111 m,
                # and then 84 steps of the same 8640 px detection at the same
                # 0.769 m depth every 14 steps -- one full 12-turn revolution,
                # spinning on the goal until the episode ran out.
                #
                # So say it here: on the goal, with the target in view, this is
                # the pose success is measured at and there is nothing left to
                # improve by turning.
                tol = float(getattr(self.nav.cfg.agent, "viewpoint_stop_m", 0.0) or 0.0)
                if (tol > 0.0 and self.nav._goal_xy is not None
                        and float(np.linalg.norm(agent_xy - self.nav._goal_xy)) <= tol):
                    stop_reason = "viewpoint"
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
                    cand = (self.nav.object_layer.get(self.nav._candidate_id)
                            if self.nav._candidate_id is not None else None)
                    note_vlm(self.nav.verifier, "terminal", self.nav.step_count, cand)
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
                back = self._return_to_best(frame, agent_xy)
                if back is not None:
                    return back
                self.nav.state = State.DONE
                self.stop_reason = stop_reason
                return STOP_ACTION
        elif (
            not self.nav._direct_approach  # a driver that owns the path (navmesh
            # or pointnav) knows it; a momentary FOV loss while turning along it
            # must NOT trigger a retreat, or the agent oscillates (approach ->
            # lose detection -> retreat -> re-detect ...) until the deadline.
            # Costmap mode keeps the LOS-occlusion retreat.
            and self.last_good_xy is not None
            and np.linalg.norm(agent_xy - self.last_good_xy) > 0.1
        ):
            action = self.follow_to(frame, self.last_good_xy)
            if action is not None:
                return action
            abandon = self.nav._absence_at_arrival(frame, "retreat")
            if abandon is not None:
                return abandon
            back = self._return_to_best(frame, agent_xy)
            if back is not None:
                return back
            self.nav.state = State.DONE  # retreat path consumed/unreachable: stop here
            self.stop_reason = "retreat"
            return STOP_ACTION

        if self.nav.step_count > self.nav._goto_deadline or self.steps_left <= 0:
            look = self.nav.close_look.before_absence(frame, "deadline")
            if look is not None:
                return look
            abandon = self.nav._absence_at_arrival(frame, "deadline")
            if abandon is not None:
                return abandon
            back = self._return_to_best(frame, agent_xy)
            if back is not None:
                return back
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
            close = self._close_last_metre(frame, agent_xy)
            if close is not None:
                return close
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
            # Before concluding absence from a frame the tight ring cannot
            # fit the moved object into, look from the 1.5 m ring
            # (agent/close_look.py). Off unless `close_look_before_absence`.
            look = self.nav.close_look.before_absence(frame, "path_consumed")
            if look is not None:
                return look
            abandon = self.nav._absence_at_arrival(frame, "path_consumed")
            if abandon is not None:
                return abandon
            back = self._return_to_best(frame, agent_xy)
            if back is not None:
                return back
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
        if self.nav._direct_approach and self.nav.cfg.agent.approach_to_viewpoint:
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
            view_xy = self.nav.viewpoint_planner.approach_viewpoint(
                obj_xy, self.nav.costmap, obj_radius_m=self._ring_offset_m()
            )
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
                    obj_xy,
                    self.nav.costmap,
                    require_line_of_sight=False,
                    allow_unknown=True,
                    obj_radius_m=self._ring_offset_m(),
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
        elif self.nav._direct_approach:
            # Navigate to the object itself. The navmesh snaps to the nearest
            # standable point (effectively a viewpoint), like old /goal_object;
            # pointnav walks at it under `pointnav_approach_creep_m`. Either way
            # the mover owns the last stretch, so the goal is the object.
            self.nav._goal_xy = obj_xy.copy()
        elif self.nav.cfg.agent.approach_navigable_goal and agent_xy is not None:
            self.nav._goal_xy = self._standoff_goal(obj_xy, agent_xy)
        else:
            self.nav._goal_xy = nearest_free_xy(self.nav.costmap, obj_xy)
        self.nav._target_obj_xy = obj_xy.copy()

    def _best_distance(self) -> float:
        """How far the remembered best pose is from the track, measured NOW.

        Only the pose is stored, never the distance: `retarget` moves
        `_target_obj_xy` mid-approach, and a cached distance would then be
        measured against a centre that no longer exists.
        """
        obj_xy = self.nav._target_obj_xy
        if self.best_xy is None or obj_xy is None:
            return float("inf")
        return float(np.linalg.norm(self.best_xy - obj_xy))

    def _note_best(self, agent_xy: np.ndarray) -> None:
        """Remember the closest the agent has stood to the committed track.

        Distance is to `_target_obj_xy`, the agent's OWN estimate of the track
        centre, so this needs nothing from the simulator.
        """
        obj_xy = self.nav._target_obj_xy
        if obj_xy is None:
            return
        d = float(np.linalg.norm(np.asarray(agent_xy, dtype=float) - obj_xy))
        if d < self._best_distance():
            self.best_xy = np.asarray(agent_xy, dtype=float).copy()

    def _return_to_best(self, frame: FrameData, agent_xy: np.ndarray) -> Optional[str]:
        """About to stop: if a closer pose was reached earlier, go back to it.

        The approach stops wherever it happens to be when its terminal rule
        fires, and that is routinely farther from the object than somewhere it
        already stood -- the mover overshoots, or turns to face and drifts, or
        the creep presses past the tangent point. The closing walk already
        implements this for its own walk (`approach_close_worse`); this is the
        same idea for the approach as a whole.

        Returns an action while walking back, None when there is nothing to do
        (off, already done, or this pose is no worse than the best by the
        configured margin). `agent.approach_stop_at_best_m` is that margin; 0
        disables it and is what every arm measured before this ran on.
        """
        margin = float(getattr(self.nav.cfg.agent, "approach_stop_at_best_m", 0.0) or 0.0)
        if margin <= 0.0 or self.returned:
            return None
        stats = self.nav.stats
        if self.returning:
            # The walk back is consumed; stop here whether or not it arrived.
            self.returning = False
            self.returned = True
            obj_xy = self.nav._target_obj_xy
            if obj_xy is not None:
                here = float(np.linalg.norm(np.asarray(agent_xy, dtype=float) - obj_xy))
                stats["approach_best_pose_gain_cm"] = (
                    stats.get("approach_best_pose_gain_cm", 0)
                    + int(round(100.0 * max(0.0, self._return_from_d - here)))
                )
            return None
        obj_xy = self.nav._target_obj_xy
        if obj_xy is None or self.best_xy is None:
            return None
        here = float(np.linalg.norm(np.asarray(agent_xy, dtype=float) - obj_xy))
        if here <= self._best_distance() + margin:
            self.returned = True
            return None
        self.returning = True
        self._return_from_d = here
        self.nav._goal_xy = self.best_xy.copy()
        self.nav._current_path = None
        self.path_goal = None
        self.steps_left = max(int(self.steps_left), 40)
        self.nav._goto_deadline = max(int(self.nav._goto_deadline), self.nav.step_count + 40)
        stats["approach_best_pose_returns"] = stats.get("approach_best_pose_returns", 0) + 1
        action = self.follow_to(frame, self.nav._goal_xy)
        if action is not None:
            return action
        self.returning = False
        self.returned = True
        return None

    def _close_last_metre(self, frame: FrameData, agent_xy: np.ndarray) -> Optional[str]:
        """The viewpoint is reached; walk the rest of the way before stopping.

        The rings stop at the innermost FREE costmap cell, and beside a bed or
        a desk the inflation leaves none nearer than about a metre from the
        track. The navmesh knows where the floor really ends, so ask it for
        the point nearest the track centre and drive there; when that walk is
        consumed the normal stop follows. Returns an action while closing,
        None when there is nothing to close (off, already closed, no navmesh,
        already near, or the navmesh point is no nearer than this pose).
        """
        close_m = float(getattr(self.nav.cfg.agent, "approach_close_last_metre_m", 0.0) or 0.0)
        if close_m <= 0.0 or self.closed:
            return None
        # NOTE: this walk asks the pathfinder for the navigable point nearest the
        # track (`_nearest_navigable_fn`), which is navmesh-derived and therefore
        # privileged in a sensor-only arm -- it chooses the GOAL, the mover still
        # has to reach it from depth alone. It was gated on `_use_navmesh` and so
        # went silently dead when the line switched to pointnav; the honest gate
        # is whether the query exists, which the caller checks below. Set
        # `agent.approach_close_last_metre_m: 0` for a navmesh-free arm.
        obj_xy = self.nav._target_obj_xy
        stats = self.nav.stats
        if self.closing:
            # The closing walk was consumed: this is the pose to stop at --
            # unless the follower gave up short and left the agent farther
            # from the track than the viewpoint it came from (measured twice
            # on the near-miss subset, 1.41 -> 1.53 m and 1.49 -> 2.14 m), in
            # which case walk back to the viewpoint and stop there.
            self.closing = False
            self.closed = True
            here_d = float(np.linalg.norm(agent_xy - obj_xy)) if obj_xy is not None else 0.0
            if self.diag is not None:
                self.diag["close_to_m"] = here_d
            stats["approach_close_arrived"] = stats.get("approach_close_arrived", 0) + 1
            if (not self.close_retreating and self.close_start_xy is not None and obj_xy is not None
                    and here_d > float(np.linalg.norm(self.close_start_xy - obj_xy)) + 0.05):
                stats["approach_close_worse"] = stats.get("approach_close_worse", 0) + 1
                self.close_retreating = True
                self.closing = True
                self.closed = False
                self.nav._goal_xy = self.close_start_xy.copy()
                self.nav._current_path = None
                self.path_goal = None
                self.steps_left = max(int(self.steps_left), 40)
                self.nav._goto_deadline = max(int(self.nav._goto_deadline), self.nav.step_count + 40)
                action = self.follow_to(frame, self.nav._goal_xy)
                if action is not None:
                    return action
                self.closing = False
                self.closed = True
            return None
        # Where to get "the nearest point to the object I can stand on".
        #
        #   navmesh  ask the pathfinder -- ground-truth floor geometry, and so
        #            privileged in a sensor-only arm
        #   costmap  ask the agent's own depth-built grid for the nearest cell
        #            with agent-radius clearance. Same question, no oracle.
        #
        # The walk's entire advantage is that its goal is standable BY
        # CONSTRUCTION, which is exactly the property the stranded approach
        # goals lacked; that property does not require the mesh.
        source = str(getattr(self.nav.cfg.agent, "approach_close_source", "navmesh"))
        if obj_xy is None:
            return None
        here_d = float(np.linalg.norm(agent_xy - obj_xy))
        if here_d <= close_m or here_d > 3.0:
            return None  # near enough already, or this was no arrival at all
        if source == "costmap":
            from ..verification.viewpoint import nearest_clear_xy

            clear = float(getattr(self.nav.cfg.agent, "approach_close_clearance_m", 0.0) or 0.0)
            if clear <= 0.0:
                clear = float(self.nav.cfg.agent.agent_radius) + 0.05
            goal = nearest_clear_xy(self.nav.costmap, obj_xy, clear)
        else:
            fn = getattr(self.nav, "_nearest_navigable_fn", None)
            if fn is None:
                return None
            goal = fn(obj_xy, self.nav._goal_floor_y_cache)
        if goal is None:
            return None
        goal = np.asarray(goal, dtype=float).ravel()[:2]
        if float(np.linalg.norm(goal - obj_xy)) >= here_d - 0.05:
            stats["approach_close_no_gain"] = stats.get("approach_close_no_gain", 0) + 1
            self.closed = True
            return None
        self.closing = True
        self.close_start_xy = np.asarray(agent_xy, dtype=float).copy()
        self.nav._goal_xy = goal.copy()
        self.nav._current_path = None
        self.path_goal = None
        self.steps_left = max(int(self.steps_left), 40)
        self.nav._goto_deadline = max(int(self.nav._goto_deadline), self.nav.step_count + 40)
        stats["approach_close_started"] = stats.get("approach_close_started", 0) + 1
        if self.diag is not None:
            self.diag["close_from_m"] = here_d
            self.diag["close_goal_to_obj_m"] = float(np.linalg.norm(goal - obj_xy))
        action = self.follow_to(frame, self.nav._goal_xy)
        if action is not None:
            return action
        # Nothing to walk (the follower has us there, or refuses): stop here.
        self.closing = False
        self.closed = True
        stats["approach_close_arrived"] = stats.get("approach_close_arrived", 0) + 1
        if self.diag is not None:
            self.diag["close_to_m"] = here_d
        return None

    def _ring_offset_m(self) -> float:
        """How far to push the viewpoint rings out to clear the object itself.

        Zero unless `verification.ring_radius_extent_aware` is set, so the
        default behaviour -- and every result measured under it -- is unchanged.
        """
        if not self.nav.cfg.verification.ring_radius_extent_aware:
            return 0.0
        return float(self.obj_radius_m)

    def start(
        self,
        obj_xy: np.ndarray,
        agent_xy: Optional[np.ndarray] = None,
        floor_y: Optional[float] = None,
        obj_radius_m: float = 0.0,
    ) -> None:
        # Horizontal extent of the track being approached, held for the whole
        # approach so retargets re-aim on the same ladder.
        self.obj_radius_m = float(obj_radius_m)
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
        self.closing = False
        self.closed = False
        self.close_start_xy = None
        self.close_retreating = False
        # Per-APPROACH, exactly like the closing walk's state above. Left to the
        # per-episode reset it was a stale pose near the PREVIOUS target: the
        # next approach would find `here` greater than a `best_d` measured
        # against a different object, and walk back to a pose metres away. It
        # also latched `returned`, so only the first approach of an episode
        # could ever correct itself.
        self.best_xy = None
        self.returning = False
        self.returned = False
        self._return_from_d = float("inf")
        gate = getattr(self.nav, "stop_gate", None)
        if gate is not None:
            gate.new_approach()
        if self.nav._direct_approach:
            # A driver (navmesh or pointnav) covers the FULL distance to the
            # object, so the short-leg cap (approach_max_steps ~= 3 m) would cut
            # the approach off while the target is still in view. Let it
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
        self.last_good_range_m = None

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
        if self.nav.pointnav is not None:
            return self.nav._follow_to(frame, goal_xy)
        if self.nav._use_navmesh:
            # navmesh drives to the object; None = arrived. The follower only
            # says so inside its 0.1 m goal radius, which 0.25 m steps can
            # circle forever; `approach_arrival_m` calls it from farther out.
            tol = float(getattr(self.nav.cfg.agent, "approach_arrival_m", 0.0) or 0.0)
            if tol > 0.0:
                agent_xy = frame.camera_position[list(PLANE)]
                if float(np.linalg.norm(agent_xy - np.asarray(goal_xy, dtype=float))) <= tol:
                    self.last_follow_none_reason = "arrived_within_tol"
                    self.nav.stats["approach_arrived_by_distance"] = (
                        self.nav.stats.get("approach_arrived_by_distance", 0) + 1
                    )
                    return None
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
