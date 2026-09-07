"""ObjectLayer orchestrates the per-keyframe object pipeline:
associate -> init new tracks -> refine due tracks -> relink.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..core.geometry import backproject, ellipse_from_mask
from ..core.labels import normalize_label
from ..core.types import Detection, FrameData
from ..mapping.costmap import HEIGHT_AXIS, PLANE
from .association import DataAssociator, Observation, ObjectTrack
from .ellipsoid import Ellipsoid
from .linking import object_center, relink
from .presence import PresenceFilter
from .optimization import WassersteinRefiner


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
        max_range_m: float = 5.0,
        fp_disable_radius_m: float = 0.5,
        cloud_stride: int = 4,
        cloud_cap: int = 2000,
        rng_seed: int = 0,
    ) -> None:
        self._tracks: Dict[int, ObjectTrack] = {}
        self._next_id = 0
        self._associator = DataAssociator(
            assoc_score_thresh, assoc_depth_gate_m, category_gate=assoc_category_gate
        )
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
        self.max_range_m = float(max_range_m)
        self.fp_disable_radius_m = float(fp_disable_radius_m)
        self.cloud_stride = int(cloud_stride)
        self.cloud_cap = int(cloud_cap)
        self.keep_cloud_labels: set = set()
        self._disabled_pts: List[tuple] = []
        self._rng = np.random.default_rng(rng_seed)
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
        }

    # ------------------------------------------------------------------ api

    def set_target(self, label: str) -> None:
        """Which class this episode is hunting. Only read by `_admits`."""
        self.target_label = normalize_label(label)

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
        same_floor = [
            t for t in self._tracks.values()
            if int(getattr(t, "floor_key", 0)) == int(floor_key)
        ]
        if self.presence_filter is not None:
            self.presence_filter.update(same_floor, frame, dets)
        dets = admitted
        if not dets:
            return
        matches = self._associator.associate(dets, frame, same_floor)
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
            if track_id is None:
                ell = Ellipsoid.init_from_detection(det, frame, rng=self._rng)
                if ell is None:
                    self.funnel["ellipsoid_rejected"] += 1
                    continue
                self.funnel["tracks_created"] += 1
                track = ObjectTrack(
                    id=self._next_id, label=det.label, ellipsoid=ell, first_cam_xy=cam_xy.copy(),
                    floor_key=int(floor_key), out_of_range=self._marginal(det, frame),
                )
                track.evidence += det.score  # first sighting: full weight
                self._next_id += 1
                self._tracks[track.id] = track
                if self._in_disabled_region(track):
                    track.blacklisted = track.disabled = True
                relink_needed = True  # visible immediately -- join the scene graph now
            else:
                track = self._tracks[track_id]
                track.evidence += det.score * self._view_diversity_weight(track, cam_xy)
                if not self._marginal(det, frame):
                    track.out_of_range = False
            track.observations.append(obs)
            if det.score > track.best_score:
                track.best_score = det.score
                track.best_crop = det.crop if det.crop is not None else det.crop_from(frame.rgb)
                track.best_frame_rgb = frame.rgb  # shared by ref across same-frame tracks
                track.best_bbox_xyxy = np.asarray(det.bbox_xyxy, dtype=float).copy()
                x1, y1, x2, y2 = det.bbox_xyxy
                track.best_bbox_px = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
                # The pose this detection was made from is a proven
                # "object visible from here" pose — the terminal stop target.
                track.best_cam_xy = cam_xy.copy()

            self._accumulate_cloud(track, det, frame)

            due = (
                track.n_obs >= self.min_obs_for_refine
                and track.n_obs - track.refined_at_obs >= self.refine_every
            )
            if due:
                refined = self._refiner.refine(track)
                if refined is not None:
                    track.ellipsoid = refined
                    relink_needed = True
                track.refined_at_obs = track.n_obs

        if relink_needed:
            relink(list(self._tracks.values()), self.link_dist_m,
                   max_frame_gap=self.link_max_frame_gap)

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
            pts = pts[self._rng.choice(len(pts), self.cloud_cap, replace=False)]
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
                track.blacklisted or not track.out_of_range
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

    def tracks(self, include_blacklisted: bool = False) -> List[ObjectTrack]:
        return [t for t in self._tracks.values() if include_blacklisted or not t.blacklisted]

    def get(self, track_id: int) -> Optional[ObjectTrack]:
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
        """
        target = target_label.lower().replace(" ", "_")
        out = []
        for t in self._tracks.values():
            if t.blacklisted or t.n_obs < min_obs or t.evidence < min_evidence:
                continue
            if floor_key is not None and t.floor_key != floor_key:
                continue
            if step is not None and t.suppressed_until > step:
                continue
            if t.best_score < min_score:
                continue
            # The size gate rejects slivers of furniture. For the target it
            # re-creates the admission deadlock one stage later: a track seeded
            # from a distant sighting can only grow its best box by being
            # approached, and it can only be approached by being proposed.
            if not target_bypasses_bbox and t.best_bbox_px < min_bbox_px:
                continue
            if t.presence.p < min_presence:
                continue
            if max_identity_rejections and t.identity_rejections >= max_identity_rejections:
                continue
            if t.label.lower().replace(" ", "_") == target:
                out.append(t)
        if rank_by_presence:
            out.sort(key=lambda t: (-t.presence.p, -t.evidence))
        else:
            out.sort(key=lambda t: -(t.best_score * t.presence.p))
        return out

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

    # ------------------------------------------------------------- internals

    @staticmethod
    def _make_observation(
        det: Detection, frame: FrameData, K: np.ndarray, T_cw: np.ndarray
    ) -> Optional[Observation]:
        ellipse = ellipse_from_mask(det.mask)
        if ellipse is None:
            return None
        ys, xs = np.nonzero(det.mask)
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
