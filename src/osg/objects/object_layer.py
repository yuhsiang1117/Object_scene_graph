"""ObjectLayer orchestrates the per-keyframe object pipeline:
associate -> init new tracks -> refine due tracks -> relink.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..core.geometry import backproject, det_ellipse, mask_pixels
from ..core.labels import normalize_label
from ..core.types import Detection, FrameData
from ..mapping.costmap import HEIGHT_AXIS, PLANE
from .association import DataAssociator, Observation, ObjectTrack
from .ellipsoid import Ellipsoid
from .feature_memory import cosine, merge_running_mean
from .linking import object_center, relink
from .presence import PresenceFilter
from .optimization import WassersteinRefiner


PROPOSAL_ID_BASE = 1_000_000   # proposal-only track ids live above here


def _geometry_active(ecfg) -> bool:
    return bool(
        ecfg.init != "moments" or ecfg.reinit_truncated or ecfg.buffer
        or ecfg.refine_mode != "2d" or ecfg.normalize_labels
        or float(ecfg.assoc_3d_overlap) > 0.0 or float(ecfg.merge_split_overlap) > 0.0
        or ecfg.finalize
    )


class ObjectLayer:
    def __init__(
        self,
        assoc_score_thresh: float = 0.4,
        assoc_depth_gate_m: float = 0.5,
        assoc_category_gate: bool = True,
        min_obs_for_refine: int = 3,
        refine_every: int = 3,
        refine_max_center_move_m: float = 0.5,
        link_dist_m: float = 1.0,
        link_max_frame_gap: Optional[int] = None,
        min_det_score: float = 0.0,
        min_det_bbox_px: float = 0.0,
        target_bypasses_gates: bool = False,
        confirm_baseline_m: float = 0.0,
        repeat_view_discount: float = 0.2,
        presence_filter: Optional[PresenceFilter] = None,
        feature_memory=None,
        max_range_m: float = 5.0,
        fp_disable_radius_m: float = 0.5,
        cloud_stride: int = 4,
        cloud_cap: int = 2000,
        rng_seed: int = 0,
        ellipsoid=None,
    ) -> None:
        self._tracks: Dict[int, ObjectTrack] = {}
        self._next_id = 0
        # scene_graph.ellipsoid (core/config/scene_graph.EllipsoidConfig), or
        # None. `_geo` is False unless some part of it is switched on, and while
        # it is False no line below behaves differently from before it existed.
        self.ecfg = ellipsoid
        self._geo = ellipsoid is not None and _geometry_active(ellipsoid)
        self._manhattan = None
        if self._geo and str(getattr(ellipsoid, "yaw_mode", "fit")) == "manhattan":
            from .point_buffer import ManhattanFrame

            self._manhattan = ManhattanFrame()
        self._associator = DataAssociator(
            assoc_score_thresh, assoc_depth_gate_m, category_gate=assoc_category_gate,
            normalize_labels=bool(self._geo and ellipsoid.normalize_labels),
        )
        self._kf_index = -1
        self._merged_into: Dict[int, int] = {}
        self._refiner = WassersteinRefiner(max_center_move_m=refine_max_center_move_m)
        self.min_obs_for_refine = min_obs_for_refine
        self.refine_every = refine_every
        self.link_dist_m = link_dist_m
        self.link_max_frame_gap = link_max_frame_gap
        self.min_det_score = min_det_score
        self.min_det_bbox_px = min_det_bbox_px
        # The gates below exist to keep INCIDENTAL noise out of the map. The
        # episode's target is not incidental, and it is the one class whose
        # confidence threshold was chosen deliberately per class
        # (detector.class_conf). See `_admits`.
        self.target_bypasses_gates = bool(target_bypasses_gates)
        self.target_label: str = ""
        # confirm_baseline_m: camera-position distance from a track's first
        # sighting beyond which a re-match counts as genuine multi-view
        # corroboration (full evidence weight) rather than a repeated glance
        # from nearly the same spot (discounted -- see
        # _view_diversity_weight). 0 disables the discount entirely.
        self.confirm_baseline_m = confirm_baseline_m
        self.repeat_view_discount = repeat_view_discount
        self.presence_filter = presence_filter
        # None unless feature_memory.enabled; nothing is loaded and no code
        # path below changes while it is None.
        self.feature_memory = feature_memory
        # The query phrase's text feature, set per episode by the agent when
        # the proposal stage is on. A proposal observation carries its region's
        # image feature; the track keeps the running mean and its cosine
        # against this text is `proposal_sim`.
        self.proposal_text: Optional[np.ndarray] = None
        self.max_range_m = float(max_range_m)
        self.fp_disable_radius_m = float(fp_disable_radius_m)
        self.cloud_stride = int(cloud_stride)
        self.cloud_cap = int(cloud_cap)
        self.keep_cloud_labels: set = set()
        self._disabled_pts: List[tuple] = []
        self._rng = np.random.default_rng(rng_seed)
        # Proposal tracks draw from their own generator and take ids from
        # their own range: sharing either shifted every later named track's
        # depth sample and id, and a plate committed at the same step to the
        # same object landed 10 cm off and missed. The named map must be the
        # named map, draw for draw.
        self._rng_prop = np.random.default_rng(rng_seed + 1_000_003)
        self._next_prop_id = PROPOSAL_ID_BASE
        # eval/object_recorder.py, attached by the runner when
        # `eval.record_object_frames_dir` is set. One-way: it is handed what
        # `update` is handed and returns nothing.
        self.recorder = None
        # Track-creation funnel. Instrumentation only: nine failures of the last
        # campaign named the target 4-36 times at its new pose and ended with
        # the ONLY same-label tracks in the map being the ones loaded from the
        # prior -- distance 0.00 m to a prior track, i.e. no new track was
        # created at all, not a badly placed one. Nothing said which of the four
        # ways that can happen actually happened.
        self.funnel = {
            "det_seen": 0,        # detections the detector produced
            "det_admitted": 0,    # cleared the score and size gates
            "obs_rejected": 0,    # no ellipse, or no valid depth under the mask
            "ellipsoid_rejected": 0,  # depth too sparse to back-project a quadric
            "tracks_created": 0,
            "target_bypassed": 0,  # admitted only because it is the target
            "proposal_obs": 0,     # observations the proposal stage contributed
        }

    # ------------------------------------------------------------------ api

    def set_target(self, label: str) -> None:
        """Which class this episode is hunting. Only read by `_admits`."""
        self.target_label = normalize_label(label)

    def set_proposal_text(self, text_ft: Optional[np.ndarray]) -> None:
        self.proposal_text = None if text_ft is None else np.asarray(text_ft, dtype=np.float32)

    def _admits(self, det: Detection) -> bool:
        """Is this detection worth putting in the map?

        The two gates price "is this worth remembering" for a scene full of
        furniture the agent is not looking for. Applied to the TARGET they
        create a deadlock, measured over 96 episodes: 33% of the times the
        detector named the target the map discarded it, and in 11 episodes it
        discarded EVERY naming -- so no track formed, so no candidate formed,
        so the agent never approached, so the detection never got bigger or
        more confident. All 11 failed. A bowl was named 20 times at 3.26 m with
        a score of 0.91 and a 608 px box, and the map refused all twenty.

        The score half is worse than a bad threshold, it is a contradiction.
        `detector.class_conf` lowers the DETECTOR to 0.20 for the pitcher, the
        tin can, the banana and the red plate -- condition H, chosen from a
        900-pose false-positive census -- and this gate then discards everything
        those four classes gained between 0.20 and 0.35. Five of the eleven
        deadlocked episodes are exactly that: boxes of 5146, 5077, 3102, 2808
        and 1258 px, far above the size gate, thrown away on score alone.

        So when the flag is on, the target is admitted on the DETECTOR's terms:
        the per-class threshold already decided it. Admission is not candidacy --
        evidence, observation count, presence and the identity channel all still
        gate whether a track may become a goal.
        """
        if (
            self.target_bypasses_gates
            and self.target_label
            and normalize_label(det.label) == self.target_label
        ):
            return True
        return (
            det.score >= self.min_det_score
            and self._bbox_px(det) >= self.min_det_bbox_px
        )

    def update(
        self, frame: FrameData, dets: List[Detection], floor_key: int = 0
    ) -> None:
        if self.recorder is not None:
            self.recorder.write(self, frame, dets, floor_key)
        self._kf_index += 1
        if self._manhattan is not None:
            self._manhattan.add(frame)
        # Node-creation quality gate: a low-confidence or sliver detection
        # shouldn't seed a new track, or even lend support to an existing
        # one -- association still runs against every current track (a
        # would-be match still consumes that track's slot so a second, good
        # detection of the same object this frame doesn't spawn a duplicate),
        # but only detections clearing the bar reach track creation/update.
        admitted = [d for d in dets if self._admits(d)]
        self.funnel["det_seen"] += len(dets)
        self.funnel["det_admitted"] += len(admitted)
        if self.target_bypasses_gates and self.target_label:
            self.funnel["target_bypassed"] += sum(
                1 for d in admitted
                if normalize_label(d.label) == self.target_label
                and not (d.score >= self.min_det_score
                         and self._bbox_px(d) >= self.min_det_bbox_px)
            )
        # Presence runs on EVERY keyframe, before the early return and against
        # the UNFILTERED detections. A frame with nothing admitted is precisely
        # the frame where negative evidence is worth the most -- the agent is
        # looking at the surface and the detector produced nothing -- and a
        # detection too small to seed a track is still proof that something is
        # there, so it must not be counted as a miss.
        # Two populations that never touch. A proposal is an appearance
        # hypothesis under the target's label; let it associate with, lend
        # presence to, or link with a track the detector named and it moves
        # that track's centre, keeps its belief alive and changes when it
        # becomes a candidate -- measured as 5 of 15 trials diverging from
        # `_sensor_v2` with commits off. So proposals only ever form and
        # extend proposal-only tracks, the detector's tracks never see them,
        # and the named path is bit-identical to a map without the stage.
        is_prop = lambda d: str(getattr(d, "source", "detector")) == "proposal"
        same_floor = [
            t for t in self._tracks.values()
            if int(getattr(t, "floor_key", 0)) == int(floor_key) and not t.proposal_only
        ]
        prop_floor = [
            t for t in self._tracks.values()
            if int(getattr(t, "floor_key", 0)) == int(floor_key) and t.proposal_only
        ]
        if self.presence_filter is not None:
            self.presence_filter.update(same_floor, frame, [d for d in dets if not is_prop(d)])
        dets = admitted
        if not dets:
            return
        # One batched CLIP forward per keyframe over what the map is about to
        # remember. Detections, not frames: the cost is a function of how much
        # is in view, not of the control rate.
        if self.feature_memory is not None:
            self.feature_memory.embed_detections([d for d in dets if not is_prop(d)])
        pts_of: Dict[int, object] = {}
        if self._geo:
            dets, pts_of = self._prepare_points(dets, frame, is_prop)
        det_idx = [i for i, d in enumerate(dets) if not is_prop(d)]
        prop_idx = [i for i, d in enumerate(dets) if is_prop(d)]
        matches = [
            (det_idx[i], tid)
            for i, tid in self._associator.associate([dets[i] for i in det_idx], frame, same_floor)
        ] if det_idx else []
        if prop_idx:
            matches += [
                (prop_idx[i], tid)
                for i, tid in self._associator.associate([dets[i] for i in prop_idx], frame, prop_floor)
            ]
        if self._geo and float(getattr(self.ecfg, "assoc_2d_min_overlap", 0.0)) > 0.0:
            matches = self._veto_2d(matches, dets, pts_of, is_prop)
        if self._geo and float(getattr(self.ecfg, "assoc_2d_reassign", 0.0)) > 0.0:
            matches = self._reassign_2d(matches, dets, pts_of, same_floor, is_prop)
        if self._geo and float(self.ecfg.assoc_3d_overlap) > 0.0:
            matches = self._associate_3d(matches, dets, pts_of, same_floor, is_prop)
        K = frame.intrinsics.K()
        T_cw = frame.T_cw
        cam_xy = frame.camera_position[list(PLANE)]

        relink_needed = False
        for det_idx, track_id in matches:
            det = dets[det_idx]
            obs = self._make_observation(det, frame, K, T_cw)
            if obs is None:
                self.funnel["obs_rejected"] += 1
                continue
            proposal = str(getattr(det, "source", "detector")) == "proposal"
            if track_id is None:
                ell = None
                if self._geo and self.ecfg.init == "points":
                    ell = Ellipsoid.init_from_points(pts_of.get(det_idx), frame, self.ecfg, yaw=self._yaw_ref())
                if ell is None:
                    ell = Ellipsoid.init_from_detection(
                        det, frame, rng=self._rng_prop if proposal else self._rng
                    )
                if ell is None:
                    self.funnel["ellipsoid_rejected"] += 1
                    continue
                self.funnel["tracks_created"] += 1
                if proposal:
                    new_id, self._next_prop_id = self._next_prop_id, self._next_prop_id + 1
                else:
                    new_id, self._next_id = self._next_id, self._next_id + 1
                track = ObjectTrack(
                    id=new_id, label=det.label, ellipsoid=ell, first_cam_xy=cam_xy.copy(),
                    floor_key=int(floor_key), out_of_range=self._marginal(det, frame),
                )
                if not proposal:
                    track.evidence += det.score  # first sighting: full weight
                self._tracks[track.id] = track
                if self._in_disabled_region(track):
                    track.blacklisted = track.disabled = True
                if self._geo:
                    track.geom_provisional = self._truncated(det, frame)
                relink_needed = True  # visible immediately -- join the scene graph now
            else:
                track = self._tracks[track_id]
                if not proposal:
                    track.evidence += det.score * self._view_diversity_weight(track, cam_xy)
                if not self._marginal(det, frame):
                    track.out_of_range = False
                if self._geo and self.ecfg.reinit_truncated and track.geom_provisional \
                        and not self._truncated(det, frame):
                    ell = (Ellipsoid.init_from_points(pts_of.get(det_idx), frame, self.ecfg, yaw=self._yaw_ref())
                           if self.ecfg.init == "points" else None)
                    if ell is None:
                        ell = Ellipsoid.init_from_detection(det, frame, rng=self._rng)
                    if ell is not None:
                        track.ellipsoid = ell
                        track.assoc_ellipsoid = None
                        track.geom_provisional = False
                        relink_needed = True
            track.observations.append(obs)
            if self._geo:
                track.last_kf = self._kf_index
                self._buffer_view(track, det, frame, pts_of.get(det_idx))
            if det.clip_ft is not None:
                track.clip_ft, track.clip_n = merge_running_mean(
                    track.clip_ft, track.clip_n, det.clip_ft
                )
                if self.feature_memory is not None:
                    self.feature_memory.tag(track)
            if proposal:
                track.n_proposal_obs += 1
                self.funnel["proposal_obs"] += 1
                if self.proposal_text is not None:
                    track.proposal_sim = cosine(track.clip_ft, self.proposal_text)
                # A proposal is a view and a feature, nothing more: it does
                # not add detector evidence and its constant score must not
                # displace the best detection's crop or the pose it was made
                # from -- that pose is the approach's terminal stop target,
                # and `admit_score` beats every class the detector was
                # loosened to 0.20 for.
                # ...unless the proposal is all this track has: then the
                # region IS its best view -- the crop the VLM will be shown,
                # the size gate, and the pose an approach returns to.
                if track.proposal_only and det.score > track.best_score:
                    self._note_best_detection(track, det, frame, cam_xy)
                self._accumulate_cloud(track, det, frame)
                continue
            if det.score > track.best_score:
                self._note_best_detection(track, det, frame, cam_xy)

            self._accumulate_cloud(track, det, frame)

            due = (
                track.n_obs >= self.min_obs_for_refine
                and track.n_obs - track.refined_at_obs >= self.refine_every
            )
            if due:
                refined = (self._refine_geo(track) if self._geo and self.ecfg.refine_mode != "2d"
                           else self._refiner.refine(track))
                if refined is not None:
                    track.ellipsoid = refined
                    relink_needed = True
                track.refined_at_obs = track.n_obs

        if self._geo and self.ecfg.finalize:
            relink_needed |= self._finalize_far(frame, floor_key)

        if relink_needed:
            # Proposal-only tracks are never linked: a link would fold a
            # region's centre into a named object's `center_of`.
            relink([t for t in self._tracks.values() if not t.proposal_only],
                   self.link_dist_m, max_frame_gap=self.link_max_frame_gap)

    # --------------------------------------------------------- surface clouds

    def _accumulate_cloud(
        self, track: ObjectTrack, det: Detection, frame: FrameData
    ) -> None:
        if track.label not in self.keep_cloud_labels:
            return
        pts = backproject(
            frame.depth, frame.intrinsics, frame.T_wc, mask=det.mask,
            stride=self.cloud_stride, max_depth=self.max_range_m,
        )
        if not len(pts):
            return
        centre_h = float(track.ellipsoid.center[HEIGHT_AXIS])
        pts = pts[np.abs(pts[:, HEIGHT_AXIS] - centre_h) < 2.5]
        if not len(pts):
            return
        if len(pts) > self.cloud_cap:
            rng = self._rng_prop if track.proposal_only else self._rng
            pts = pts[rng.choice(len(pts), self.cloud_cap, replace=False)]
        track.points_w = (
            pts if track.points_w is None
            else np.vstack([track.points_w, pts])[-self.cloud_cap:]
        )

    def _track_cloud(self, track: ObjectTrack) -> Optional[np.ndarray]:
        clouds = [track.points_w]
        for linked_id in track.linked_ids:
            other = self._tracks.get(linked_id)
            if (
                other is not None and not other.blacklisted
                and other.floor_key == track.floor_key
                and other.points_w is not None
            ):
                clouds.append(other.points_w)
        clouds = [cloud for cloud in clouds if cloud is not None and len(cloud)]
        return np.vstack(clouds) if clouds else None

    def nearest_point_xy(
        self, track: ObjectTrack, agent_xy: np.ndarray
    ) -> Optional[np.ndarray]:
        cloud = self._track_cloud(track)
        if cloud is None:
            return None
        pts = cloud[:, list(PLANE)]
        return pts[int(np.argmin(np.linalg.norm(pts - np.asarray(agent_xy), axis=1)))]

    def nearest_point_dist_xy(
        self, track: ObjectTrack, agent_xy: np.ndarray, percentile: float = 0.0
    ) -> Optional[float]:
        cloud = self._track_cloud(track)
        if cloud is None:
            return None
        distances = np.linalg.norm(
            cloud[:, list(PLANE)] - np.asarray(agent_xy), axis=1
        )
        return float(
            distances.min()
            if percentile <= 0.0 else np.percentile(distances, percentile)
        )

    # ------------------------------------------------- false-positive handling

    def _marginal(self, det: Detection, frame: FrameData) -> bool:
        x1, _, x2, _ = det.bbox_xyxy
        width = float(frame.rgb.shape[1])
        if (x2 <= width / 3.0 and x1 <= 0.05 * width) or (
            x1 >= 2.0 * width / 3.0 and x2 >= 0.95 * width
        ):
            return True
        ys, xs = np.nonzero(det.mask)
        if not len(ys):
            return False
        depth = frame.depth[ys, xs]
        valid = depth > 1e-3
        return bool(
            valid.any() and np.median(depth[valid]) > 0.95 * self.max_range_m
        )

    def _in_disabled_region(self, track: ObjectTrack) -> bool:
        center = track.ellipsoid.center[list(PLANE)]
        return any(
            label == track.label
            and float(np.linalg.norm(center - xy)) < self.fp_disable_radius_m
            for xy, label in self._disabled_pts
        )

    def retract_unconfirmed(
        self, frame: FrameData, dets: List[Detection], half_range_m: float,
        fov_rad: float, floor_key: Optional[int] = None,
    ) -> int:
        cam_xy = frame.camera_position[list(PLANE)]
        forward = frame.T_wc[:3, :3] @ np.array([0.0, 0.0, 1.0])
        heading = forward[list(PLANE)]
        norm = float(np.linalg.norm(heading))
        if norm < 1e-6:
            return 0
        heading /= norm
        seen = {d.label for d in dets}
        retracted = 0
        for track in self._tracks.values():
            if (
                track.blacklisted or not track.out_of_range or track.proposal_only
                or (floor_key is not None and track.floor_key != floor_key)
                or track.label in seen
            ):
                continue
            vector = self.center_of(track)[list(PLANE)] - cam_xy
            distance = float(np.linalg.norm(vector))
            if distance > half_range_m or distance < 1e-3:
                continue
            if float(np.dot(vector / distance, heading)) < np.cos(fov_rad / 2.0):
                continue
            track.blacklisted = track.disabled = True
            self._disabled_pts.append(
                (self.center_of(track)[list(PLANE)].copy(), track.label)
            )
            retracted += 1
        return retracted

    def _view_diversity_weight(self, track: ObjectTrack, cam_xy: np.ndarray) -> float:
        """Full weight for a re-observation from a meaningfully different
        camera pose than the track's first sighting (genuine multi-view
        corroboration); discounted (not zeroed) for a repeated glance from
        nearly the same spot, so a burst of near-duplicate frames can't
        inflate evidence as fast as real parallax can (P1i follow-up:
        FUS3DMaps-style evidence accumulation, replacing the earlier hard
        confirmed/tentative visibility gate that starved scene_graph
        context during early exploration)."""
        if self.confirm_baseline_m <= 0.0 or track.first_cam_xy is None:
            return 1.0
        baseline = float(np.linalg.norm(cam_xy - track.first_cam_xy))
        return 1.0 if baseline >= self.confirm_baseline_m else self.repeat_view_discount

    def tracks(
        self, include_blacklisted: bool = False, include_proposals: bool = False
    ) -> List[ObjectTrack]:
        """The map. Proposal-only tracks are left out unless asked for: the
        scene graph, the search anchor, presence bookkeeping and the prior
        map must see exactly the map the detector built. The candidate gate
        reads `_tracks` directly and judges them on their own terms."""
        return [
            t for t in self._tracks.values()
            if (include_blacklisted or not t.blacklisted)
            and (include_proposals or not t.proposal_only)
        ]

    def get(self, track_id: int) -> Optional[ObjectTrack]:
        # A track folded into another by the finalise step answers as the
        # track it became, so an id held elsewhere (a commit) stays valid.
        seen = set()
        while track_id in self._merged_into and track_id not in seen:
            seen.add(track_id)
            track_id = self._merged_into[track_id]
        return self._tracks.get(track_id)

    def candidates(
        self,
        target_label: str,
        min_obs: int = 2,
        min_score: float = 0.0,
        min_bbox_px: float = 0.0,
        min_evidence: float = 0.0,
        min_presence: float = 0.0,
        max_identity_rejections: int = 0,
        target_bypasses_bbox: bool = False,
        rank_by_presence: bool = False,
        floor_key: Optional[int] = None,
        step: Optional[int] = None,
        proposal_commits: bool = True,
        proposal_min_obs: int = 1,
        proposal_tau: float = -1.0,
        proposal_tau_by_class: Optional[dict] = None,
    ) -> List[ObjectTrack]:
        """Non-blacklisted tracks matching the target with enough support,
        detection quality, accumulated evidence (fragment detections and
        single-glimpse noise make useless candidates), and enough remaining
        belief that the object is still there.

        Ranking by `best_score * presence.p` rather than `best_score` alone is
        the whole query-side payoff of the presence filter: a track the agent
        has since looked at and not found sinks below one it has not disproved,
        instead of being re-proposed on every replan.

        `rank_by_presence` goes one step further, and the step is measured. Over
        the 170 within-episode pairs of K, L and M where a correct and a wrong
        BELIEVED track compete, the probability that the key puts the correct one
        first:

            best_score alone                   0.635
            best_score * presence.p  (shipped) 0.729
            presence.p alone                   0.800

        Multiplying by detector confidence HURTS, which makes sense once stated:
        a confident false positive is exactly a distant object that really does
        look like the target, so `best_score` is high precisely where it misleads.
        Presence is the channel that asks "did I look there recently and see it",
        which is the question a stale map needs answered.

        Presence saturates at the clamp, so it needs a tie-break, and evidence
        beats score there too. Per episode with a real choice to make, correct
        track chosen: 64/93 shipped, 69/93 by presence alone, 73/93 by presence
        then evidence.

        `max_identity_rejections` (0 disables) retires a track the agent has
        walked to and found was not the target that many times. Presence cannot
        do this job: a false positive is an object that IS present, so every look
        that disproves it as the target also re-detects it as an object and
        restores its belief. Measured, with the identity channel off: one episode
        committed to the same wrong track 251 times in 500 steps, its belief
        pinned at the 0.95 positive clamp through 250 absence readings.

        A track the proposal stage alone has seen (`proposal_only`) carries the
        target's label without the detector ever having said so. Measured on
        the full 107 (docs/history/REGION_FALSE_ADMISSION.md), letting such a track
        through on the label test cost 18 trials: a per-frame region admitted
        on 13% of frames with no object in view became an ordinary same-label
        track, won this gate and spent an attempt. So it is judged on what it
        actually is -- an appearance hypothesis accumulated over views: at
        least `proposal_min_obs` observations, a running-mean feature whose
        cosine to the query clears `proposal_tau`, and never ahead of a track
        the detector has named. `proposal_commits=False` keeps them out
        entirely, which is how the bar was measured before it was set.
        """
        target = target_label.lower().replace(" ", "_")
        if proposal_tau_by_class:
            proposal_tau = float(proposal_tau_by_class.get(target, proposal_tau))
        out = []
        for t in self._tracks.values():
            if t.blacklisted or t.n_obs < min_obs:
                continue
            if floor_key is not None and t.floor_key != floor_key:
                continue
            if step is not None and t.suppressed_until > step:
                continue
            # The evidence, score and size gates are calibrated on DETECTOR
            # output, and a proposal-only track has none: no evidence, a
            # constant score, a region's box. Its bar is the proposal bar
            # below. Measured: the first fused run committed to zero proposal
            # tracks in 49 episodes while tracks past the bar sat in the map
            # with evidence 0.0 < min_evidence 1.0, rejected here in silence.
            if not t.proposal_only:
                if t.evidence < min_evidence or t.best_score < min_score:
                    continue
                # The size gate rejects slivers of furniture. For the target
                # it re-creates the admission deadlock one stage later: a
                # track seeded from a distant sighting can only grow its best
                # box by being approached, and it can only be approached by
                # being proposed.
                if not target_bypasses_bbox and t.best_bbox_px < min_bbox_px:
                    continue
            if t.presence.p < min_presence:
                continue
            if max_identity_rejections and t.identity_rejections >= max_identity_rejections:
                continue
            if t.label.lower().replace(" ", "_") != target:
                continue
            if t.proposal_only:
                if not proposal_commits:
                    continue
                if t.n_proposal_obs < int(proposal_min_obs):
                    continue
                if float(t.proposal_sim) < float(proposal_tau):
                    continue
            out.append(t)
        # Named tracks first; a proposal-only track competes only among its
        # own kind. The tier is 0 for every track the detector (or the prior
        # map) has named, so with no proposals in the map the order is exactly
        # what it was.
        # Within the proposal tier the order is the appearance itself: a
        # proposal track has no detector evidence and a constant score, so
        # the named keys would tie and fall back to insertion order. Measured
        # on the observation runs, a trial can hold a true track at 0.325 and
        # a phantom at 0.286 that both clear the bar; this puts the true one
        # first.
        tier = lambda t: 1 if t.proposal_only else 0
        prop_key = lambda t: (-float(t.proposal_sim), -int(t.n_proposal_obs)) if t.proposal_only else (0.0, 0)
        if rank_by_presence:
            out.sort(key=lambda t: (tier(t), *prop_key(t), -t.presence.p, -t.evidence))
        else:
            out.sort(key=lambda t: (tier(t), *prop_key(t), -(t.best_score * t.presence.p)))
        return out

    @staticmethod
    def _note_best_detection(track: ObjectTrack, det: Detection, frame: FrameData, cam_xy: np.ndarray) -> None:
        track.best_score = det.score
        track.best_crop = det.crop if det.crop is not None else det.crop_from(frame.rgb)
        track.best_frame_rgb = frame.rgb  # shared by ref across same-frame tracks
        track.best_bbox_xyxy = np.asarray(det.bbox_xyxy, dtype=float).copy()
        x1, y1, x2, y2 = det.bbox_xyxy
        track.best_bbox_px = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
        # The pose this detection was made from is a proven
        # "object visible from here" pose -- the terminal stop target.
        track.best_cam_xy = cam_xy.copy()

    @staticmethod
    def _bbox_px(det: Detection) -> float:
        x1, y1, x2, y2 = det.bbox_xyxy
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

    def center_of(self, track: ObjectTrack) -> np.ndarray:
        return object_center(track, self._tracks)

    def blacklist(self, track_id: int) -> None:
        tr = self._tracks.get(track_id)
        if tr is not None:
            tr.blacklisted = True

    def suppress(self, track_id: int, until_step: int) -> None:
        track = self._tracks.get(track_id)
        if track is not None:
            track.suppressed_until = max(track.suppressed_until, int(until_step))

    def disable_target(self, track_id: int) -> bool:
        track = self._tracks.get(track_id)
        if track is None:
            return False
        track.blacklisted = track.disabled = True
        self._disabled_pts.append(
            (self.center_of(track)[list(PLANE)].copy(), track.label)
        )
        return True

    def disable_place(self, xy: np.ndarray, label: str) -> None:
        self._disabled_pts.append((np.asarray(xy, dtype=float)[:2].copy(), label))

    # ------------------------------------------------- depth geometry (M1-M5)

    def _truncated(self, det: Detection, frame: FrameData) -> bool:
        from .point_buffer import truncated

        h, w = frame.depth.shape
        return truncated(det, w, h, int(self.ecfg.border_px))

    def _prepare_points(self, dets, frame, is_prop):
        """Cleaned depth points per detection index, and -- when asked -- the
        same-frame, same-label detections that are one object split in two
        masks, merged into one before they can seed two tracks."""
        from .point_buffer import VoxelBuffer, clean_points

        pts_of = {i: clean_points(d, frame, self.ecfg) for i, d in enumerate(dets)}
        thr = float(self.ecfg.merge_split_overlap)
        if thr <= 0.0:
            return dets, pts_of
        normalize = bool(self.ecfg.normalize_labels)
        from .merge import same_label

        alive = list(range(len(dets)))
        merged = True
        while merged:
            merged = False
            for a in alive:
                for b in alive:
                    if b <= a or is_prop(dets[a]) or is_prop(dets[b]):
                        continue
                    pa, pb = pts_of.get(a), pts_of.get(b)
                    if pa is None or pb is None or not same_label(dets[a].label, dets[b].label, normalize):
                        continue
                    va = VoxelBuffer(self.ecfg.voxel_m)
                    va.add(pa.world)
                    vb = VoxelBuffer(self.ecfg.voxel_m)
                    vb.add(pb.world)
                    ov = max(float(vb.contains(pa.world[::max(1, len(pa.world) // 300)]).mean()),
                             float(va.contains(pb.world[::max(1, len(pb.world) // 300)]).mean()))
                    if ov < thr:
                        continue
                    keep, drop = (a, b) if dets[a].score >= dets[b].score else (b, a)
                    d = dets[keep]
                    mask = np.asarray(d.mask, bool) | np.asarray(dets[drop].mask, bool)
                    ys, xs = np.nonzero(mask)
                    bbox = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                                    dtype=np.asarray(d.bbox_xyxy).dtype)
                    nd = Detection(label=d.label, score=max(d.score, dets[drop].score),
                                   bbox_xyxy=bbox, mask=mask, crop=d.crop, clip_ft=d.clip_ft,
                                   source=d.source)
                    for attr in ("ref", "gt_instance"):
                        if hasattr(d, attr):
                            setattr(nd, attr, getattr(d, attr))
                    dets[keep] = nd
                    pts_of[keep] = clean_points(nd, frame, self.ecfg)
                    alive.remove(drop)
                    merged = True
                    break
                if merged:
                    break
        kept = sorted(alive)
        return [dets[i] for i in kept], {n: pts_of[i] for n, i in enumerate(kept)}

    def _veto_2d(self, matches, dets, pts_of, is_prop):
        """Undo a projection match whose detection's depth points are not where
        the track is: inside its ellipsoid inflated by `assoc_3d_inflate`, or
        within a voxel of its buffer. The projection gate admits any bbox
        overlap above 1%, so a neighbour of the same label can be taken for
        the object; the vetoed detection goes on to 3D association or a new
        track. A detection with no clean points keeps its match."""
        from .merge import point_overlap

        thr = float(self.ecfg.assoc_2d_min_overlap)
        out = []
        for i, tid in matches:
            pts = pts_of.get(i)
            if tid is not None and pts is not None and not is_prop(dets[i]):
                track = self._tracks.get(tid)
                if track is not None and point_overlap(
                        track, pts.world, float(self.ecfg.assoc_3d_inflate)) < thr:
                    self.funnel["assoc_2d_vetoed"] = self.funnel.get("assoc_2d_vetoed", 0) + 1
                    tid = None
            out.append((i, tid))
        return out

    def _reassign_2d(self, matches, dets, pts_of, same_floor, is_prop):
        """Move a projection match to the same-label track that actually holds
        the detection's points: when under `assoc_2d_reassign` of them are at
        the matched track and at least `assoc_3d_overlap` are at another one
        not matched this frame. Never unmatches, so it cannot create tracks
        (the veto of `assoc_2d_min_overlap` created seven times as many, S78)."""
        from .merge import point_overlap, same_label

        low = float(self.ecfg.assoc_2d_reassign)
        need = float(self.ecfg.assoc_3d_overlap)
        inflate, mode = float(self.ecfg.assoc_3d_inflate), str(self.ecfg.overlap_mode)
        dilate = int(getattr(self.ecfg, "overlap_dilate", 1))
        normalize = bool(self.ecfg.normalize_labels)
        taken = {tid for _i, tid in matches if tid is not None}
        out = []
        for i, tid in matches:
            pts = pts_of.get(i)
            track = self._tracks.get(tid) if tid is not None else None
            if track is None or pts is None or is_prop(dets[i]) \
                    or point_overlap(track, pts.world, inflate, mode, dilate) >= low:
                out.append((i, tid))
                continue
            best, best_ov = None, need
            for other in same_floor:
                if other.id == tid or other.id in taken or other.proposal_only \
                        or not same_label(other.label, track.label, normalize):
                    continue
                ov = point_overlap(other, pts.world, inflate, mode, dilate)
                if ov >= best_ov:
                    best, best_ov = other, ov
            if best is None:
                out.append((i, tid))
                continue
            taken.discard(tid)
            taken.add(best.id)
            self.funnel["assoc_2d_reassigned"] = self.funnel.get("assoc_2d_reassigned", 0) + 1
            out.append((i, best.id))
        return out

    def _associate_3d(self, matches, dets, pts_of, same_floor, is_prop):
        from .merge import associate_3d

        taken = {tid for _i, tid in matches if tid is not None}
        unmatched = [(i, dets[i].label, None if pts_of.get(i) is None else pts_of[i].world)
                     for i, tid in matches if tid is None and not is_prop(dets[i])]
        found = associate_3d(unmatched, same_floor, taken,
                             float(self.ecfg.assoc_3d_overlap), float(self.ecfg.assoc_3d_inflate),
                             bool(self.ecfg.normalize_labels), str(self.ecfg.overlap_mode),
                             int(self.ecfg.overlap_dilate), bool(self.ecfg.assoc_3d_unique))
        self.funnel["assoc_3d"] = self.funnel.get("assoc_3d", 0) + len(found)
        return [(i, found.get(i, tid) if tid is None else tid) for i, tid in matches]

    def _buffer_view(self, track, det, frame, pts) -> None:
        from .point_buffer import VoxelBuffer, view_ok

        if not self.ecfg.buffer or not view_ok(det, frame, pts, self.ecfg):
            return
        if track.buffer is None:
            track.buffer = VoxelBuffer(self.ecfg.voxel_m, self.ecfg.buffer_cap)
            track.buffer.summary_voxels = int(self.ecfg.summary_voxels)
        track.buffer.add(pts.world)
        d = pts.world.mean(axis=0) - frame.camera_position
        d = np.array([d[0], d[2]])
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            dirs = getattr(track.buffer, "view_dirs", [])
            track.buffer.view_dirs = (dirs + [d / n])[-64:]

    def _yaw_ref(self) -> Optional[float]:
        """The yaw a points fit snaps its axes to, or None to keep its own."""
        mode = str(getattr(self.ecfg, "yaw_mode", "fit"))
        if mode == "fit":
            return None
        if mode == "world":
            return 0.0
        if mode == "manhattan":
            return None if self._manhattan is None else self._manhattan.yaw
        raise ValueError(f"scene_graph.ellipsoid.yaw_mode: unknown {mode!r}")

    def _refine_geo(self, track) -> Optional[Ellipsoid]:
        from .optimization import refine_with_points

        buf = track.buffer
        if buf is None or len(buf) < 8:
            return self._refiner.refine(track)
        move = float(self.ecfg.depth_max_center_move_m)
        if move < 0.0:
            move = float(self._refiner.max_center_move_m)
        if self.ecfg.refine_mode == "points":
            if buf.n_views < int(getattr(self.ecfg, "refit_min_views", 2)):
                return self._refiner.refine(track)
            pts = buf.points()
            ell = Ellipsoid.fit_points(pts, np.asarray(getattr(buf, "view_dirs", [])), self.ecfg,
                                       yaw=self._yaw_ref())
            if ell is not None and bool(getattr(self.ecfg, "decouple_geometry", False)):
                # The limit gates what association sees, not the geometry.
                gated = track.ellipsoid if track.assoc_ellipsoid is None else track.assoc_ellipsoid
                if move > 0.0 and float(np.linalg.norm(ell.center - gated.center)) > move:
                    track.assoc_ellipsoid = gated
                    self.funnel["geometry_decoupled"] = self.funnel.get("geometry_decoupled", 0) + 1
                else:
                    track.assoc_ellipsoid = None
                return ell
            if ell is not None and move > 0.0 and float(
                    np.linalg.norm(ell.center - track.ellipsoid.center)) > move:
                override = float(getattr(self.ecfg, "refit_override_overlap", 0.0))
                if override <= 0.0:
                    return None
                # The ellipsoid alone: `merge.point_overlap` also counts the
                # track's own buffer voxels, which these points always are.
                e = track.ellipsoid
                inside = np.linalg.norm(((pts - e.center) @ e.R) / e.axes, axis=1) <= 1.0
                if float(inside.mean()) >= override:
                    return None
                self.funnel["refit_override"] = self.funnel.get("refit_override", 0) + 1
            return ell
        if self.ecfg.refine_mode == "joint":
            pts = buf.points()
            k = int(self.ecfg.depth_max_pts)
            if len(pts) > k:
                pts = pts[np.linspace(0, len(pts) - 1, k).astype(int)]
            return refine_with_points(track, pts, float(self.ecfg.depth_weight), move)
        raise ValueError(f"scene_graph.ellipsoid.refine_mode: unknown {self.ecfg.refine_mode!r}")

    def _finalize_far(self, frame: FrameData, floor_key: int) -> bool:
        """Tracks the agent has walked away from: refit from the buffer, fold
        in any same-label track that fills the same space, compress."""
        from .merge import choose_keeper, mutual_overlap, same_label

        cam = frame.camera_position
        changed = False
        for track in list(self._tracks.values()):
            if track.id not in self._tracks or track.proposal_only or track.buffer is None:
                continue
            if int(track.floor_key) != int(floor_key):
                continue
            if track.buffer.n_views <= track.finalized_views:
                continue
            if self._kf_index - track.last_kf < int(self.ecfg.finalize_unseen_kf):
                continue
            if float(np.linalg.norm(track.ellipsoid.center - cam)) < float(self.ecfg.finalize_dist_m):
                continue
            ell = Ellipsoid.fit_points(track.buffer.points(),
                                       np.asarray(getattr(track.buffer, "view_dirs", [])), self.ecfg,
                                       yaw=self._yaw_ref())
            if ell is not None:
                track.ellipsoid = ell
            for other in list(self._tracks.values()):
                if other.id == track.id or other.id not in self._tracks or other.proposal_only:
                    continue
                if int(other.floor_key) != int(track.floor_key) or not same_label(
                        other.label, track.label, bool(self.ecfg.normalize_labels)):
                    continue
                if mutual_overlap(track, other, float(self.ecfg.assoc_3d_inflate),
                                  str(self.ecfg.overlap_mode), int(self.ecfg.overlap_dilate),
                                  bool(self.ecfg.merge_symmetric)) < float(self.ecfg.merge_overlap):
                    continue
                keep, drop = choose_keeper(track, other)
                from .merge import merge_into

                merge_into(keep, drop)
                del self._tracks[drop.id]
                self._merged_into[drop.id] = keep.id
                self.funnel["merged"] = self.funnel.get("merged", 0) + 1
                if keep.buffer is not None and keep.buffer.n_views >= 1:
                    refit = Ellipsoid.fit_points(keep.buffer.points(),
                                                 np.asarray(getattr(keep.buffer, "view_dirs", [])), self.ecfg,
                                                 yaw=self._yaw_ref())
                    if refit is not None:
                        keep.ellipsoid = refit
                track = keep
            track.buffer.compress(int(self.ecfg.summary_voxels))
            track.finalized_views = track.buffer.n_views
            self.funnel["finalized"] = self.funnel.get("finalized", 0) + 1
            changed = True
        return changed

    def consolidate(self) -> int:
        """Fold small same-label fragments into the track they belong to
        (merge.consolidate_minor), before a snapshot is written. Returns the
        number of tracks folded; 0 and untouched when
        `scene_graph.ellipsoid.consolidate_max_obs` is 0."""
        k = int(self.ecfg.consolidate_max_obs)
        if k <= 0:
            return 0
        from .merge import consolidate_minor, merge_into

        pairs = consolidate_minor(list(self._tracks.values()), k,
                                  float(self.ecfg.consolidate_gap_m),
                                  float(self.ecfg.consolidate_ratio),
                                  bool(self.ecfg.normalize_labels))
        for drop_id, keep_id in pairs:
            keep, drop = self._tracks[keep_id], self._tracks.pop(drop_id)
            merge_into(keep, drop)
            self._merged_into[drop_id] = keep_id
        if pairs:
            self.funnel["consolidated"] = self.funnel.get("consolidated", 0) + len(pairs)
            relink([t for t in self._tracks.values() if not t.proposal_only],
                   self.link_dist_m, max_frame_gap=self.link_max_frame_gap)
        return len(pairs)

    # ------------------------------------------------------------- internals

    @staticmethod
    def _make_observation(
        det: Detection, frame: FrameData, K: np.ndarray, T_cw: np.ndarray
    ) -> Optional[Observation]:
        ellipse = det_ellipse(det)
        if ellipse is None:
            return None
        ys, xs = mask_pixels(det)
        d = frame.depth[ys, xs]
        valid = d > 1e-3
        if not valid.any():
            return None
        return Observation(
            frame_id=frame.frame_id,
            mu=ellipse.mu,
            cov=ellipse.cov,
            K=K,
            T_cw=T_cw.copy(),
            mean_depth=float(np.median(d[valid])),
        )
