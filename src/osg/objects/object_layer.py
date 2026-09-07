"""ObjectLayer orchestrates the per-keyframe object pipeline:
associate -> init new tracks -> refine due tracks -> relink.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..core.geometry import backproject, ellipse_from_mask
from ..core.types import Detection, FrameData
from ..mapping.costmap import HEIGHT_AXIS, PLANE
from .association import DataAssociator, Observation, ObjectTrack
from .ellipsoid import Ellipsoid
from .linking import object_center, relink
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
        min_det_score: float = 0.0,
        min_det_bbox_px: float = 0.0,
        confirm_baseline_m: float = 0.0,
        repeat_view_discount: float = 0.2,
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
        self.min_det_score = min_det_score
        self.min_det_bbox_px = min_det_bbox_px
        # confirm_baseline_m: camera-position distance from a track's first
        # sighting beyond which a re-match counts as genuine multi-view
        # corroboration (full evidence weight) rather than a repeated glance
        # from nearly the same spot (discounted -- see
        # _view_diversity_weight). 0 disables the discount entirely.
        self.confirm_baseline_m = confirm_baseline_m
        self.repeat_view_discount = repeat_view_discount
        # Depth sensor range, for spotting detections at the far edge of it.
        self.max_range_m = max_range_m
        self.fp_disable_radius_m = fp_disable_radius_m
        # (ground-plane position, label) of every retracted false positive, so
        # re-detecting the same thing does not resurrect it.
        self._disabled_pts: List[tuple] = []
        # Surface point clouds are kept only for these labels -- set to the
        # episode target by NavAgent.reset. Accumulating them for everything
        # would be unbounded in a cluttered scene for no benefit: only the
        # object the agent is walking to needs a precise surface distance.
        self.keep_cloud_labels: set = set()
        self.cloud_stride = cloud_stride
        self.cloud_cap = cloud_cap
        self._rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------ api

    def update(self, frame: FrameData, dets: List[Detection], floor_key: int = 0) -> None:
        # Node-creation quality gate: a low-confidence or sliver detection
        # shouldn't seed a new track, or even lend support to an existing
        # one -- association still runs against every current track (a
        # would-be match still consumes that track's slot so a second, good
        # detection of the same object this frame doesn't spawn a duplicate),
        # but only detections clearing the bar reach track creation/update.
        dets = [
            d for d in dets
            if d.score >= self.min_det_score and self._bbox_px(d) >= self.min_det_bbox_px
        ]
        if not dets:
            return
        # Associate only against tracks on the floor the agent is standing on:
        # a track a storey below projects into the lower image rows and can
        # otherwise capture a detection of a different object entirely.
        same_floor = [t for t in self._tracks.values() if t.floor_key == floor_key]
        matches = self._associator.associate(dets, frame, same_floor)
        K = frame.intrinsics.K()
        T_cw = frame.T_cw
        cam_xy = frame.camera_position[list(PLANE)]

        relink_needed = False
        for det_idx, track_id in matches:
            det = dets[det_idx]
            obs = self._make_observation(det, frame, K, T_cw)
            if obs is None:
                continue
            marginal = self._marginal(det, frame)
            if track_id is None:
                ell = Ellipsoid.init_from_detection(det, frame, rng=self._rng)
                if ell is None:
                    continue
                track = ObjectTrack(
                    id=self._next_id, label=det.label, ellipsoid=ell, first_cam_xy=cam_xy.copy(),
                    floor_key=floor_key, out_of_range=marginal,
                )
                track.evidence += det.score  # first sighting: full weight
                self._next_id += 1
                self._tracks[track.id] = track
                # A track born where an earlier one was retracted, with the same
                # label, is the same false positive being re-detected. Kill it
                # on arrival, or the agent re-commits to it every time it looks
                # that way again.
                if self._in_disabled_region(track):
                    track.blacklisted = track.disabled = True
                relink_needed = True  # visible immediately -- join the scene graph now
            else:
                track = self._tracks[track_id]
                track.evidence += det.score * self._view_diversity_weight(track, cam_xy)
                if not marginal:
                    track.out_of_range = False  # a clean look clears the doubt
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
            relink(list(self._tracks.values()), self.link_dist_m)

    # --------------------------------------------------------- surface clouds

    def _accumulate_cloud(self, track: ObjectTrack, det: Detection, frame: FrameData) -> None:
        """Fold this detection's masked points into the track's surface cloud."""
        if track.label not in self.keep_cloud_labels:
            return
        pts = backproject(
            frame.depth, frame.intrinsics, frame.T_wc,
            mask=det.mask, stride=self.cloud_stride, max_depth=self.max_range_m,
        )
        if pts.shape[0] == 0:
            return
        # A mask that bleeds onto a window or the ceiling behind the object
        # would otherwise drag the "nearest surface" metres away.
        centre_h = float(track.ellipsoid.center[HEIGHT_AXIS])
        pts = pts[np.abs(pts[:, HEIGHT_AXIS] - centre_h) < 2.5]
        if pts.shape[0] == 0:
            return
        if pts.shape[0] > self.cloud_cap:
            pts = pts[self._rng.choice(pts.shape[0], self.cloud_cap, replace=False)]
        track.points_w = (
            pts if track.points_w is None
            else np.vstack([track.points_w, pts])[-self.cloud_cap:]
        )

    def nearest_point_xy(
        self, track: ObjectTrack, agent_xy: np.ndarray
    ) -> Optional[np.ndarray]:
        """The observed surface point closest to the agent, on the ground plane.

        ASCENT navigates to THIS rather than to a fitted centre
        (object_point_cloud_map.py:127-130, _get_closest_point at :225). The
        distinction matters because a centre is inferred and can land somewhere
        never observed -- including inside or behind a wall -- while a cloud
        point is a real depth return and is therefore, by construction,
        somewhere the agent has actually seen.

        Read-only for now: recorded alongside the centre so the two can be
        scored against ground truth before anything navigates to it.
        """
        clouds = [track.points_w]
        for lid in track.linked_ids:
            other = self._tracks.get(lid)
            if other is not None and not other.blacklisted and other.points_w is not None:
                clouds.append(other.points_w)
        clouds = [c for c in clouds if c is not None and c.shape[0]]
        if not clouds:
            return None
        pts = np.vstack(clouds)[:, list(PLANE)]
        return pts[int(np.argmin(np.linalg.norm(pts - np.asarray(agent_xy), axis=1)))]

    def nearest_point_dist_xy(
        self, track: ObjectTrack, agent_xy: np.ndarray, percentile: float = 0.0
    ) -> Optional[float]:
        """Ground-plane distance to the nearest point of the object's SURFACE.

        Spans the linked component, matching object_center: an L-shaped sofa is
        two ellipsoids but one object, and the near end is what the agent
        should stop at.

        `percentile` guards against the one statistic this cloud cannot support.
        The cloud accumulates over hundreds of frames of mask noise and pose
        drift and spans linked tracks, so its *minimum* is set by its worst
        stray point rather than by the object. Measured on dev50: stopping on
        the raw min put 14 of 31 approaches within 0.1 m of the goal and
        scattered the rest out to 2.3 m. That bimodality is the signature of
        outliers, not of a threshold set too large -- a threshold error would
        shift the whole distribution, not split it. 0.0 keeps the exact min.
        """
        clouds = [track.points_w]
        for lid in track.linked_ids:
            other = self._tracks.get(lid)
            if other is not None and not other.blacklisted and other.points_w is not None:
                clouds.append(other.points_w)
        clouds = [c for c in clouds if c is not None and c.shape[0]]
        if not clouds:
            return None
        pts = np.vstack(clouds)[:, list(PLANE)]
        d = np.linalg.norm(pts - np.asarray(agent_xy), axis=1)
        if percentile <= 0.0:
            return float(d.min())
        return float(np.percentile(d, percentile))

    # ------------------------------------------------- false-positive retraction

    def _marginal(self, det: Detection, frame: FrameData) -> bool:
        """Is this detection too poorly observed to be trusted on its own?

        Two cases, both ported from ASCENT's object point-cloud map, where they
        exist for the same reason: the dominant ObjectNav failure is walking
        20 m across a house to a confidently-detected object that is not there.

        * **Hard against a left or right image edge.** The object is mostly out
          of frame, so both the label and the 3D extent are guesses.
        * **At the far end of the depth range.** Depth is least reliable there
          and the mask covers few pixels.
        """
        x1, _, x2, _ = det.bbox_xyxy
        w = float(frame.rgb.shape[1])
        third = w / 3.0
        if x2 <= third and x1 <= 0.05 * w:
            return True
        if x1 >= 2 * third and x2 >= 0.95 * w:
            return True
        d = self._median_masked_depth(det, frame)
        return d is not None and d > 0.95 * self.max_range_m

    @staticmethod
    def _median_masked_depth(det: Detection, frame: FrameData) -> Optional[float]:
        ys, xs = np.nonzero(det.mask)
        if ys.size == 0:
            return None
        d = frame.depth[ys, xs]
        valid = d > 1e-3
        return float(np.median(d[valid])) if valid.any() else None

    def _in_disabled_region(self, track: ObjectTrack) -> bool:
        c = track.ellipsoid.center[list(PLANE)]
        return any(
            lbl == track.label and float(np.linalg.norm(c - xy)) < self.fp_disable_radius_m
            for xy, lbl in self._disabled_pts
        )

    def retract_unconfirmed(
        self, frame: FrameData, dets: List[Detection], half_range_m: float, fov_rad: float
    ) -> int:
        """Disbelieve marginal tracks that a clean, close look does not confirm.

        A track created from an edge-of-frame or max-range detection is a
        hypothesis. When the agent later has that position well inside its view
        cone at close range and the detector reports nothing of that category
        there, the hypothesis is refuted -- and refuting it now is what stops
        the agent walking to it later. Costs nothing but the detections already
        computed for this frame.

        Returns the number retracted.
        """
        cam_xy = frame.camera_position[list(PLANE)]
        fwd = frame.T_wc[:3, :3] @ np.array([0.0, 0.0, 1.0])
        heading = fwd[list(PLANE)]
        n = float(np.linalg.norm(heading))
        if n < 1e-6:
            return 0
        heading = heading / n
        seen = {d.label for d in dets}

        retracted = 0
        for track in list(self._tracks.values()):
            if track.blacklisted or not track.out_of_range:
                continue
            if track.label in seen:
                continue  # something of this category is visible: not refuted
            v = self.center_of(track)[list(PLANE)] - cam_xy
            dist = float(np.linalg.norm(v))
            if dist > half_range_m or dist < 1e-3:
                continue
            # Well inside the cone, not just at its edge, so a missed detection
            # really means "not there" rather than "clipped".
            if float(np.dot(v / dist, heading)) < np.cos(fov_rad / 2.0):
                continue
            track.blacklisted = track.disabled = True
            self._disabled_pts.append((self.center_of(track)[list(PLANE)].copy(), track.label))
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
        floor_key: Optional[int] = None,
        step: Optional[int] = None,
    ) -> List[ObjectTrack]:
        """Non-blacklisted tracks matching the target with enough support,
        detection quality, and accumulated evidence (fragment detections
        and single-glimpse noise make useless candidates).

        `floor_key` restricts to one storey. It defaults to None (no filter)
        deliberately: a target mapped on ANOTHER floor is not a candidate to
        walk to, but it is exactly the signal that should send the agent up or
        down the stairs, so the caller decides.
        """
        target = target_label.lower().replace(" ", "_")
        out = []
        for t in self._tracks.values():
            if t.blacklisted or t.n_obs < min_obs or t.evidence < min_evidence:
                continue
            if step is not None and t.suppressed_until > step:
                continue
            if floor_key is not None and t.floor_key != floor_key:
                continue
            if t.best_score < min_score or t.best_bbox_px < min_bbox_px:
                continue
            if t.label.lower().replace(" ", "_") == target:
                out.append(t)
        out.sort(key=lambda t: -t.best_score)
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
        """Set a track aside until `until_step`, rather than for good.

        ASCENT's rejection is a retryable gate: `_double_check_goal`
        (map_controller.py:770-776) is re-evaluated EVERY step until the score
        passes, and the target is abandoned only at close range where the view
        is best (ascent_policy.py:910-922). OSG asked once, from wherever the
        agent happened to be, and treated a NO as final -- which left the agent
        standing on the goal unable to stop, measured in 3 of 22 episodes.
        """
        track = self._tracks.get(track_id)
        if track is not None:
            track.suppressed_until = max(track.suppressed_until, int(until_step))

    def disable_target(self, track_id: int) -> bool:
        """Retire the PLACE a track occupies, not just its id.

        `blacklist` retires an identity, so the same physical object comes
        straight back the moment association gives it a new track id -- one
        smoke run rejected the same thing six times that way
        (docs/AB_RESULTS.md, S29). ASCENT retires the CELLS instead
        (`_disabled_object_map`, object_point_cloud_map.py:102).

        This is the same retirement `retract_unconfirmed` already performs, and
        it feeds the same `_disabled_pts` list, so a re-detection is killed at
        track birth (see `update`) rather than filtered at every query.
        """
        track = self._tracks.get(track_id)
        if track is None:
            return False
        track.blacklisted = track.disabled = True
        self._disabled_pts.append((self.center_of(track)[list(PLANE)].copy(), track.label))
        return True

    def disable_place(self, xy: np.ndarray, label: str) -> None:
        """`disable_target` for a target whose track is already gone."""
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
