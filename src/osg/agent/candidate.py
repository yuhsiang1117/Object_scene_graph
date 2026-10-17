"""What becomes a navigation goal, and what does not.

This is where the dynamic-scene machinery decides. A track is proposable only if
it clears four gates, and the two interesting ones are the ones a static-map
system does not have:

  presence    `min_presence` (0.45) replaces blacklisting when an approach finds
              nothing. The track stays in the map, drops below the bar, and
              comes back if it is seen again -- no absorbing states. The value is
              set by arithmetic rather than taste: a belief reloaded from a prior
              map at 0.82 lands at 0.485 after one detector-strength absence
              reading and 0.36 after a VLM one, so ONE detector miss is not
              enough to retire a track and one VLM answer is decisive.

  identity    presence cannot answer "is it MINE". A false positive is an object
              that really is there, so every look that disproves it as the target
              also re-detects it as an object and pins its belief at the positive
              clamp. Measured with the identity channel off: one episode
              committed to the same wrong track 251 times in 500 steps. Two
              rejections retires it -- one arrival can end on a bad heading, two
              is a decision.

Ranking is `best_score * presence.p`, which is the whole query-side payoff of
the filter: a track the agent has looked for and not found sinks below one it
has not disproved, instead of being re-proposed on every replan.

The VLM candidate gate is here too, and it is OFF in the shipped config for a
measured reason: verifying a track's stored best crop accepts ~97% of the time,
because a distant chair-like object really does look like a chair. In the
ablation ladder it was the one change that HURT -- SR 0.385 -> 0.260 -- and of
the first three rejections in a pilot all three were correct candidates. When it
does reject, that is one piece of identity evidence, not a verdict.
"""
from __future__ import annotations


import numpy as np

from ..core.types import FrameData
from ..mapping.costmap import HEIGHT_AXIS, PLANE
from ..planning.controller import TURN_LEFT, TURN_RIGHT, _wrap, agent_heading
from ..verification.verifier import note_vlm
from .state import TURN_ACTION, State


class CandidatePolicy:
    def __init__(self, nav) -> None:
        self.nav = nav
        self.reset()

    def reset(self) -> None:
        self.went_to_best_cam = False
        self.center_turns = 0  # centring turns spent on the current candidate
        # What the map believed at the moment it committed. A commit to a track
        # the agent has already looked for and failed to find is a stale goal --
        # the failure DualMap's ignore list exists to paper over.
        self.goal_commit_log: list = []
        # Which track was struck off, and why. `unreachable_skip` counts the
        # event and cannot say WHICH candidate it happened to, so a run where
        # the agent found the object and then retired it for unreachability is
        # indistinguishable from one where it retired a stale prior. Measured on
        # 00848's cross_anchor red plate: the episode ends holding the real
        # plate at 0.01 m with p=0.953 and never goes to it, `unreachable_skip`
        # is 2, `max_identity_rejections` is 2, and the aggregate counter cannot
        # tell you whether those are the same track. Name the decision.
        self.reject_log: list = []
        # Where the agent was standing when a track was last struck off as
        # unreachable, per track id. `check()` runs every step, re-picks the
        # same top candidate, and asks the pathfinder the same question from the
        # same pose -- so on 00848's cross_anchor_02 the REAL plate (0.04 m from
        # truth, p=0.818) took two strikes at steps 150 and 151 and hit
        # `max_identity_rejections`, which is meant to mean two separate
        # failures to find it. One verdict counted twice retires a correct track
        # in two steps and leaves 350 unspent.
        self._unreachable_from: dict = {}

    def _height_plausible(self, candidates: list) -> list:
        """Hold back a one-shot track sitting far above the storey.

        A target rests on the floor or on furniture; a detection whose centre
        is metres above the storey the agent is standing on is usually a
        mis-projected sliver -- a reflection, a ceiling fitting, or a box fitted
        to a wall texture. Measured on 00800: the toy-airplane episode spent
        both early attempts, by step 95, on single-observation tracks at y 5.64
        and 5.09 above a 3.16 m storey, while the real object was at 0.96 on the
        other one. Every wrong commit in v11-v13 was more than 0.8 m up.

        This asks for CORROBORATION, never a strike-off: a real object can sit
        high on a shelf, so at `commit_high_min_obs` observations the track
        competes normally. And it never empties the list -- if every candidate
        is high, the best of them is still offered, because refusing to commit
        at all is how an agent ends an episode having done nothing.
        """
        limit = float(getattr(self.nav.cfg.verification,
                              "commit_max_above_storey_m", 0.0) or 0.0)
        if limit <= 0.0:
            return candidates
        min_obs = int(getattr(self.nav.cfg.verification, "commit_high_min_obs", 2))
        try:
            floor_y = float(self.nav.floors.height_of(self.nav.floors.current_id))
        except Exception:
            return candidates
        kept = []
        held = 0
        for track in candidates:
            centre = self.nav.object_layer.center_of(track)
            above = float(centre[HEIGHT_AXIS]) - floor_y
            if above > limit and int(getattr(track, "n_obs", 0)) < min_obs:
                held += 1
                continue
            kept.append(track)
        if held:
            self.nav.stats["commit_held_high_track"] = (
                self.nav.stats.get("commit_held_high_track", 0) + held)
        return kept or candidates

    def check(self, agent_xy=None, admit=None) -> None:
        candidates = self.nav.object_layer.candidates(
            self.nav.target,
            min_obs=self.nav.cfg.verification.min_obs,
            min_score=self.nav.cfg.verification.min_score,
            min_bbox_px=self.nav.cfg.verification.min_bbox_px,
            min_evidence=self.nav.cfg.verification.min_evidence,
            min_presence=self.nav.cfg.scene_graph.presence.min_presence,
            max_identity_rejections=int(self.nav.cfg.scene_graph.presence.max_identity_rejections),
            target_bypasses_bbox=self.nav.cfg.verification.target_bypasses_bbox_gate,
            rank_by_presence=self.nav.cfg.verification.rank_candidates_by_presence,
            floor_key=self.nav.floors.current_id,
            step=self.nav.step_count,
            **self._proposal_gate(),
        )
        if candidates:
            candidates = self._height_plausible(candidates)
        if candidates and bool(self.nav.cfg.verification.retire_stale_twins_after_absence):
            candidates = self._without_stale_twins(candidates)
        if not candidates:
            picked = self._appearance_pick()
            if picked is None:
                return
            candidates = [picked]
        track = candidates[0]
        if admit is not None and not admit(track):
            # The caller has a reason not to be interrupted right now (a floor
            # switch in flight) and this candidate is not strong enough to
            # override it. Nothing is struck off: the track is still there
            # next step, when the reason may be gone.
            return
        self.nav._candidate_id = track.id
        self.center_turns = 0  # fresh centering budget for this candidate
        obj_center = self.nav.object_layer.center_of(track)
        obj_xy = obj_center[list(PLANE)]

        # Direct commit: hand the goal to the mover and let it drive there,
        # then STOP on arrival -- like publishing /goal_object. No viewpoint
        # pre-positioning, and no VERIFYING state.
        #
        # The predicate is "is there a driver that owns the path", not "is it
        # the navmesh". It was `_use_navmesh`, so when the line switched to
        # `navigation: pointnav` this whole branch -- the commit log and
        # `_start_approach` with it -- went dead and every episode fell through
        # to the GOTO_VERIFY_VIEW machine below. The island/reachability oracle
        # inside is the one genuinely navmesh-dependent part and keeps its own
        # `_reachable_fn is not None` guard, so it simply does not run without
        # the mesh.
        if self.nav._direct_approach:
            # Don't commit to a target on a disconnected navmesh island (a
            # visible-but-unreachable object, e.g. in a sealed bathroom): the
            # agent can never get there, so blacklist it and keep exploring for
            # a reachable goal instead of stopping and failing the episode.
            if self.nav._reachable_fn is not None and not self._reachable(track, obj_xy, obj_center):
                self.nav.stats["unreachable_skip"] = (
                    self.nav.stats.get("unreachable_skip", 0) + 1
                )
                if self._restrike_is_the_same_verdict(track, agent_xy):
                    self.nav.stats["unreachable_restrike"] = (
                        self.nav.stats.get("unreachable_restrike", 0) + 1
                    )
                    self.nav._candidate_id = None
                    return
                self._log_reject(track, "unreachable")
                if self.nav.cfg.verification.unreachable_is_absorbing:
                    self.nav.object_layer.blacklist(track.id)
                else:
                    # Not a verdict. "I could not get there" is one piece of
                    # evidence about this candidate, and the identity channel is
                    # where evidence that presence cannot carry already goes.
                    track.identity_rejections += 1
                self.nav._candidate_id = None
                return
            # VLM verify the candidate before committing (no VERIFYING state in
            # navmesh mode). Reject -> blacklist and keep exploring; this is the
            # only FP gate in the navmesh path.
            if self.nav.verifier is not None and not self.nav.cfg.verification.absence_only:
                note_vlm(self.nav.verifier, "candidate", self.nav.step_count, track)
                with self.nav.profiler.timeit("verification"):
                    ok = self.nav.verifier.verify(track, self.nav.target)
                if not ok:
                    # The VLM looked at a picture of this object and said it is
                    # not the target. That is identity evidence and belongs in
                    # the identity channel; blacklisting would make it permanent,
                    # which is the mistake this file has had to unlearn three
                    # times.
                    #
                    # It counts as ONE piece of evidence, not a verdict, and that
                    # is a measurement rather than caution. The picture is the
                    # stored crop of the best detection, and for these targets it
                    # is 46-101 px on its longest side -- there is no more image
                    # to be had, the objects are simply small in the frame. Given
                    # decisive weight it cost real successes: of the first three
                    # rejections in a pilot run all three were CORRECT
                    # candidates, and two of them had converted in the run
                    # without the gate. One doubt plus one failed approach
                    # retires a track; one doubt alone does not.
                    track.identity_rejections += 1
                    self.nav._candidate_id = None
                    self.nav.stats["verify_reject"] = self.nav.stats.get("verify_reject", 0) + 1
                    return
            self._log_goal_commit(track)
            self.nav._start_approach(
                obj_xy, floor_y=self.nav.floors.goal_floor_y(obj_center)
            )
            return

        # Always pre-position at a viewpoint from which the object is visible
        # before approaching -- HM3D success requires stopping at such a pose,
        # not merely near the object's 3D center. When verification is off the
        # VERIFYING state simply skips the VLM call (see _do_verification).
        view_xy = self.nav.viewpoint_planner.approach_viewpoint(obj_xy, self.nav.costmap)
        if view_xy is None:
            return  # not yet observable from mapped space; keep exploring
        self.nav._goal_xy = view_xy
        self.nav.state = State.GOTO_VERIFY_VIEW
        self.nav._current_path = None
        self.nav._goto_deadline = self.nav.step_count + 80

    def _proposal_gate(self) -> dict:
        """The bar a proposal-only track must clear to be a candidate."""
        rp = getattr(self.nav.cfg, "region_proposal", None)
        if rp is None:
            return {}
        return {
            "proposal_commits": bool(getattr(rp, "commits", True)),
            "proposal_min_obs": int(getattr(rp, "commit_min_obs", 1)),
            "proposal_tau": float(getattr(rp, "commit_tau", -1.0)),
            "proposal_tau_by_class": {
                str(k).lower().replace(" ", "_"): float(v)
                for k, v in dict(getattr(rp, "commit_tau_by_class", {}) or {}).items()
            },
        }

    def _appearance_pick(self):
        """The surface is reached, the label path has nothing: which nearby
        object LOOKS most like the target?

        This is DualMap's local inquiry and the one place its result is
        reproduced. It fires only on arrival at a searched surface, takes an
        argmax rather than clearing a threshold, and hands its winner back into
        the ordinary commit path below -- so reachability, the VLM, the identity
        channel, the absence sensor and the attempt protocol all still judge it.
        Nothing here can stop an approach or score a trial by itself.

        Measured before it was built: on the dumped mug frames a box small enough
        to BE the mug covers it on 39 of 160, labelled `towel` 27 times, and
        ranking every detection in the frame by cosine picks it on 27.8% of
        frames in the 2.5-4 m band. The label path scores 0 on all 160.
        """
        fm = getattr(self.nav, "feature_memory", None)
        if fm is None or not bool(fm.cfg.local_pick_on_arrival):
            return None
        exploration = getattr(self.nav, "exploration", None)
        cid = getattr(exploration, "arrived_container", None)
        if cid is None:
            return None
        exploration.arrived_container = None  # one inquiry per arrival
        if fm.counters["feature_local_picks"] >= int(fm.cfg.max_local_picks):
            return None
        node = self.nav.scene_graph.containers.get(int(cid))
        if node is None:
            return None
        centre_xy = np.asarray(node.center, dtype=float)[list(PLANE)]
        picked = fm.best_near(
            centre_xy,
            self.nav.object_layer.tracks(),
            floor_key=self.nav.floors.current_id,
        )
        if picked is None:
            return None
        picked.feature_admitted = True
        self.nav.stats["feature_local_pick_step"] = int(self.nav.step_count)
        self.nav.feature_pick_log.append({
            "step": int(self.nav.step_count),
            "container_id": int(cid),
            "container_label": str(node.label),
            **(fm.last_pick or {}),
        })
        return picked

    def _restrike_is_the_same_verdict(self, track, agent_xy) -> bool:
        """Has this track already been ruled unreachable from where we stand?

        Reachability is a fact about a (track, pose) pair. `check()` asks it
        every step, and while nothing has moved the answer cannot change -- so
        counting each answer as fresh evidence spends the identity channel's
        whole budget on one verdict. A genuine second failure, from somewhere
        else in the house, still counts; that is what the channel is for.
        """
        tol = float(getattr(self.nav.cfg.agent, "unreachable_restrike_m", 0.0) or 0.0)
        if tol <= 0.0 or agent_xy is None:
            return False
        was = self._unreachable_from.get(track.id)
        here = np.asarray(agent_xy, dtype=float)
        if was is not None and float(np.linalg.norm(here - was)) < tol:
            return True
        # Only a COUNTED strike moves the reference pose. Updating it on every
        # attempt compares against the previous STEP instead of the previous
        # verdict, and the agent moves 0.05-0.25 m per step -- so the guard
        # suppresses for ever and the identity channel stops working at all.
        # Measured: 350 suppressions in one episode, one strike counted, outcome
        # unchanged.
        self._unreachable_from[track.id] = here.copy()
        return False

    def _log_reject(self, track, reason: str) -> None:
        """One line per candidate struck off, with the belief it was carrying.

        `presence` and `rejections` are the two channels that decide whether the
        strike is recoverable, so both belong in the record: a track retired at
        p=0.95 is a different bug from one retired at p=0.10.
        """
        centre = self.nav.object_layer.center_of(track)
        self.reject_log.append({
            "step": self.nav.step_count,
            "track_id": int(track.id),
            "reason": reason,
            "p": round(float(track.presence.p), 4),
            "n_obs": int(track.n_obs),
            "rejections": int(track.identity_rejections) + 1,
            "center": [round(float(x), 3) for x in centre],
        })

    def _without_stale_twins(self, candidates):
        """Once a prior-map track of the label has been refuted in place, the
        other prior-map tracks of the label that this episode has never seen
        are ghosts of the same static pass, not places to go."""
        refuted = any(
            bool(getattr(t, "from_prior", False))
            and (int(getattr(t, "absence_arrivals", 0)) > 0
                 or int(getattr(t, "failed_attempts", 0)) > 0)
            for t in self.nav.object_layer.tracks(include_blacklisted=True)
            if t.label.lower().replace(" ", "_") == self.nav.target.lower().replace(" ", "_")
        )
        if not refuted:
            return candidates
        kept = [t for t in candidates if not (getattr(t, "from_prior", False) and not t.seen_live)]
        dropped = len(candidates) - len(kept)
        if dropped:
            self.nav.stats["stale_twins_retired"] = (
                self.nav.stats.get("stale_twins_retired", 0) + dropped
            )
        return kept

    def _log_goal_commit(self, track) -> None:
        """What the map believed at the moment it committed. A commit to a
        track the agent has already looked for and failed to find is a stale
        goal -- the failure DualMap's ignore list exists to paper over."""
        fm = getattr(self.nav, "feature_memory", None)
        if fm is not None and bool(getattr(track, "feature_admitted", False)):
            fm.counters["feature_local_committed"] += 1
        self.goal_commit_log.append(
            {
                "step": int(self.nav.step_count),
                "track_id": int(track.id),
                "label": str(track.label),
                "p": round(float(track.presence.p), 4),
                "n_missed": int(track.presence.n_missed),
                "center": [float(v) for v in self.nav.object_layer.center_of(track)],
                "prior": bool(getattr(track, "from_prior", False)),
                "seen_live": bool(track.seen_live),
                # Proposed by appearance rather than by its label.
                "feature_admitted": bool(getattr(track, "feature_admitted", False)),
                "feature_sim": round(float(getattr(track, "feature_sim", -1.0)), 4),
                # Never named by the detector: committed on accumulated
                # appearance from the proposal stage.
                "proposal_only": bool(getattr(track, "proposal_only", False)),
                "n_proposal_obs": int(getattr(track, "n_proposal_obs", 0)),
                "proposal_sim": round(float(getattr(track, "proposal_sim", -1.0)), 4),
            }
        )

    def _reachable(self, track, obj_xy, obj_center) -> bool:
        """Can the agent get to this candidate?

        The honest form of the question is about the pose it would DRIVE to.
        `_start_approach` already sends the agent to a viewpoint on the ring,
        never to the object's own position -- which for anything resting on
        furniture is inside the furniture and off the navmesh. Asking about the
        object's position and then striking the track off is how a solvable
        episode is abandoned: on 00829, six of thirty-six authored target poses
        are off-navmesh and all six have a reachable viewpoint.
        """
        floor_y = self.nav.floors.goal_floor_y(obj_center)
        if self.nav.cfg.agent.reachable_via_viewpoint:
            view_xy = self.nav.viewpoint_planner.approach_viewpoint(
                obj_xy, self.nav.costmap
            )
            if view_xy is None:
                view_xy = self.nav.viewpoint_planner.approach_viewpoint(
                    obj_xy, self.nav.costmap,
                    require_line_of_sight=False, allow_unknown=True,
                )
            if view_xy is not None and self.nav._reachable_fn(view_xy, floor_y):
                return True
        if bool(getattr(self.nav.cfg.agent, "reachable_via_nearest_free", False)):
            from ..mapping.costmap import nearest_free_xy

            near_xy = np.asarray(nearest_free_xy(self.nav.costmap, obj_xy), dtype=float)
            if self.nav._reachable_fn(near_xy, floor_y):
                self.nav.stats["reachable_via_nearest_free"] = (
                    self.nav.stats.get("reachable_via_nearest_free", 0) + 1
                )
                return True
        return bool(self.nav._reachable_fn(obj_xy, floor_y))

    def verify(self, frame: FrameData) -> str:
        track = (
            self.nav.object_layer.get(self.nav._candidate_id)
            if self.nav._candidate_id is not None else None
        )
        if track is None:
            self.nav.state = State.EXPLORE
            return TURN_ACTION
        # Center-then-verify: if a VLM verifier is active and the target is
        # actually visible in the live view, bring its detection to the middle
        # of the camera before the VLM call, then verify that well-framed frame.
        if (
            self.nav.verifier is not None
            and self.nav.cfg.verification.center_before_verify
        ):
            det = self.nav._best_target_detection(frame)
            if det is not None:
                bbox_cx = 0.5 * (float(det.bbox_xyxy[0]) + float(det.bbox_xyxy[2]))
                offset = float(np.arctan2(bbox_cx - frame.intrinsics.cx, frame.intrinsics.fx))
                if (
                    abs(offset) > np.radians(self.nav.cfg.verification.center_tol_deg)
                    and self.center_turns < self.nav.cfg.verification.center_max_turns
                ):
                    self.center_turns += 1
                    self.nav.stats["center_turn"] = self.nav.stats.get("center_turn", 0) + 1
                    # target right of centre (offset>0) -> turn right to centre it
                    return TURN_RIGHT if offset > 0 else TURN_LEFT
                # Centred (or out of centring budget): verify the live framed view.
                if not self.nav._terminal_verify:
                    note_vlm(self.nav.verifier, "candidate_live", self.nav.step_count, track)
                with self.nav.profiler.timeit("verification"):
                    accepted = (
                        True if self.nav._terminal_verify
                        else self.nav.verifier.verify_bbox(
                            frame.rgb, det.bbox_xyxy, self.nav.target
                        )
                    )
                if accepted:
                    obj_xy = self.nav.object_layer.center_of(track)[list(PLANE)]
                    self.nav._start_approach(
                        obj_xy, agent_xy=frame.camera_position[list(PLANE)]
                    )
                    return self.nav.approach.step(frame)
                self.nav.object_layer.blacklist(track.id)
                self.nav._candidate_id = None
                self.nav.state = State.EXPLORE
                return TURN_ACTION
            # target not visible in the live view -> fall through to the
            # 3D-facing / best-cam recovery below.

        # Face the object first so the live view actually shows it.

        obj_xy = self.nav.object_layer.center_of(track)[list(PLANE)]
        agent_xy = frame.camera_position[list(PLANE)]
        to_obj = obj_xy - agent_xy
        if np.linalg.norm(to_obj) > 0.05:
            err = _wrap(float(np.arctan2(to_obj[1], to_obj[0])) - agent_heading(frame.T_wc))
            if abs(err) > np.radians(20.0):
                return TURN_RIGHT if err > 0 else TURN_LEFT
        # HM3D success viewpoints require the object to actually be VISIBLE
        # from the stop pose; 2D line-of-sight misses desk-height occluders
        # (we stopped 0.12 m outside the viewpoint set). If the detector
        # cannot see the target from here, return to the pose the best
        # detection was made from — proven reachable AND proven visible
        # (ring alternatives proved unreachable and thrashed the deadline).
        if (
            not self.nav._target_visible(frame)
            and not self.went_to_best_cam
            and track.best_cam_xy is not None
            and np.linalg.norm(track.best_cam_xy - agent_xy) > 0.35
        ):
            self.went_to_best_cam = True
            self.nav._goal_xy = track.best_cam_xy.copy()
            self.nav.state = State.GOTO_VERIFY_VIEW
            self.nav._current_path = None
            self.nav._goto_deadline = self.nav.step_count + 60
            return self.nav._follow_path(frame) or TURN_ACTION
        if self.nav.verifier is not None and not self.nav._terminal_verify:
            note_vlm(self.nav.verifier, "candidate", self.nav.step_count, track)
        with self.nav.profiler.timeit("verification"):
            # Pre-approach verification. Skipped (accept) when the verifier is
            # off (old-fidelity mode) OR in terminal-view mode, where the
            # decisive VLM check is deferred to the STOP moment in _do_approach.
            accepted = (
                True if (self.nav.verifier is None or self.nav._terminal_verify)
                else self.nav.verifier.verify(track, self.nav.target, live_view=frame.rgb)
            )
        if accepted:
            obj_xy = self.nav.object_layer.center_of(track)[list(PLANE)]
            self.nav._start_approach(
                obj_xy, agent_xy=frame.camera_position[list(PLANE)]
            )
            return self.nav.approach.step(frame)
        self.nav.object_layer.blacklist(track.id)
        self.nav._candidate_id = None
        self.nav.state = State.EXPLORE
        return TURN_ACTION
