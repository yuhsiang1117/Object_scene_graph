"""ObjectLayer orchestrates the per-keyframe object pipeline:
associate -> init new tracks -> refine due tracks -> relink.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..core.geometry import ellipse_from_mask
from ..core.types import Detection, FrameData
from ..mapping.costmap import PLANE
from .association import DataAssociator, Observation, ObjectTrack
from .ellipsoid import Ellipsoid
from .linking import object_center, relink
from .optimization import WassersteinRefiner


class ObjectLayer:
    def __init__(
        self,
        assoc_score_thresh: float = 0.4,
        assoc_depth_gate_m: float = 0.5,
        min_obs_for_refine: int = 3,
        refine_every: int = 3,
        link_dist_m: float = 1.0,
        min_det_score: float = 0.0,
        min_det_bbox_px: float = 0.0,
        confirm_baseline_m: float = 0.0,
        rng_seed: int = 0,
    ) -> None:
        self._tracks: Dict[int, ObjectTrack] = {}
        self._next_id = 0
        self._associator = DataAssociator(assoc_score_thresh, assoc_depth_gate_m)
        self._refiner = WassersteinRefiner()
        self.min_obs_for_refine = min_obs_for_refine
        self.refine_every = refine_every
        self.link_dist_m = link_dist_m
        self.min_det_score = min_det_score
        self.min_det_bbox_px = min_det_bbox_px
        self.confirm_baseline_m = confirm_baseline_m
        self._rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------ api

    def update(self, frame: FrameData, dets: List[Detection]) -> None:
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
        matches = self._associator.associate(dets, frame, list(self._tracks.values()))
        K = frame.intrinsics.K()
        T_cw = frame.T_cw
        cam_xy = frame.camera_position[list(PLANE)]

        relink_needed = False
        for det_idx, track_id in matches:
            det = dets[det_idx]
            obs = self._make_observation(det, frame, K, T_cw)
            if obs is None:
                continue
            if track_id is None:
                ell = Ellipsoid.init_from_detection(det, frame, rng=self._rng)
                if ell is None:
                    continue
                track = ObjectTrack(
                    id=self._next_id, label=det.label, ellipsoid=ell, first_cam_xy=cam_xy.copy(),
                    # confirm_baseline_m <= 0 disables multi-view confirmation
                    # entirely: every quality-gated sighting is trusted immediately.
                    confirmed=self.confirm_baseline_m <= 0.0,
                )
                self._next_id += 1
                self._tracks[track.id] = track
                if track.confirmed:
                    relink_needed = True
            else:
                track = self._tracks[track_id]
                if not track.confirmed and track.first_cam_xy is not None:
                    baseline = float(np.linalg.norm(cam_xy - track.first_cam_xy))
                    if baseline >= self.confirm_baseline_m:
                        track.confirmed = True
                        relink_needed = True
            track.observations.append(obs)
            if det.score > track.best_score:
                track.best_score = det.score
                track.best_crop = det.crop if det.crop is not None else det.crop_from(frame.rgb)
                x1, y1, x2, y2 = det.bbox_xyxy
                track.best_bbox_px = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
                # The pose this detection was made from is a proven
                # "object visible from here" pose — the terminal stop target.
                track.best_cam_xy = cam_xy.copy()

            due = (
                track.confirmed
                and track.n_obs >= self.min_obs_for_refine
                and track.n_obs - track.refined_at_obs >= self.refine_every
            )
            if due:
                refined = self._refiner.refine(track)
                if refined is not None:
                    track.ellipsoid = refined
                    relink_needed = True
                track.refined_at_obs = track.n_obs

        if relink_needed:
            relink([t for t in self._tracks.values() if t.confirmed], self.link_dist_m)

    def tracks(self, include_blacklisted: bool = False, include_unconfirmed: bool = False) -> List[ObjectTrack]:
        return [
            t for t in self._tracks.values()
            if (include_blacklisted or not t.blacklisted) and (include_unconfirmed or t.confirmed)
        ]

    def get(self, track_id: int) -> Optional[ObjectTrack]:
        return self._tracks.get(track_id)

    def candidates(
        self,
        target_label: str,
        min_obs: int = 2,
        min_score: float = 0.0,
        min_bbox_px: float = 0.0,
    ) -> List[ObjectTrack]:
        """Confirmed, non-blacklisted tracks matching the target with enough
        support and detection quality (fragment detections make useless
        candidates)."""
        target = target_label.lower().replace(" ", "_")
        out = []
        for t in self._tracks.values():
            if t.blacklisted or not t.confirmed or t.n_obs < min_obs:
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
