"""Data association between new detections and existing object tracks.

Paper formula for large objects that exceed the frame:
    score = max(I / A1, I / A2)
with I approximated by the intersection of the ellipse-enclosing bboxes.
Same-class ambiguity is resolved by depth consistency (closest expected
depth), not camera-pose proximity as in VOOM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..core.geometry import bbox_area, bbox_intersection_area, ellipse_from_mask
from ..core.types import Detection, FrameData
from .ellipsoid import Ellipsoid


@dataclass
class Observation:
    frame_id: int
    mu: np.ndarray  # (2,) observed ellipse center
    cov: np.ndarray  # (2, 2) observed ellipse covariance
    K: np.ndarray  # (3, 3)
    T_cw: np.ndarray  # (4, 4)
    mean_depth: float


@dataclass
class ObjectTrack:
    id: int
    label: str
    ellipsoid: Ellipsoid
    observations: List[Observation] = field(default_factory=list)
    best_crop: Optional[np.ndarray] = None
    best_score: float = 0.0
    best_bbox_px: float = 0.0  # bbox area of the best detection, px^2
    best_cam_xy: Optional[np.ndarray] = None  # camera ground-plane pose of the best detection
    blacklisted: bool = False
    linked_ids: set = field(default_factory=set)
    refined_at_obs: int = 0
    first_cam_xy: Optional[np.ndarray] = None  # ground-plane pose of the first sighting
    # Evidence-score corroboration (P1i follow-up, replaces the earlier
    # hard confirmed/tentative gate that starved scene_graph.rebuild() of
    # objects early in exploration -- see ObjectLayer._view_diversity_weight).
    # A track is visible immediately on creation; evidence accumulates every
    # observation, weighted down for repeated glances from nearly the same
    # spot so a genuine multi-view corroboration still counts for more than
    # a burst of near-duplicate frames, without ever hiding the track.
    evidence: float = 0.0

    @property
    def n_obs(self) -> int:
        return len(self.observations)


class DataAssociator:
    def __init__(self, score_thresh: float = 0.4, depth_gate_m: float = 0.5) -> None:
        self.score_thresh = score_thresh
        self.depth_gate_m = depth_gate_m

    def associate(
        self, dets: List[Detection], frame: FrameData, tracks: List[ObjectTrack]
    ) -> List[Tuple[int, Optional[int]]]:
        """Returns [(det_idx, track_id or None)]; None means create new track."""
        K = frame.intrinsics.K()
        T_cw = frame.T_cw

        projections = {}
        for tr in tracks:
            ell = tr.ellipsoid.project(K, T_cw)
            if ell is not None:
                projections[tr.id] = (tr, ell)

        pairs = []  # (score, det_idx, track_id)
        det_info = []
        for i, det in enumerate(dets):
            obs_ellipse = ellipse_from_mask(det.mask)
            det_info.append(obs_ellipse)
            if obs_ellipse is None:
                continue
            obs_bbox = obs_ellipse.bbox()
            ys, xs = np.nonzero(det.mask)
            d = frame.depth[ys, xs]
            det_depth = float(np.median(d[d > 1e-3])) if (d > 1e-3).any() else -1.0
            for tid, (tr, proj) in projections.items():
                if tr.label != det.label:
                    continue
                proj_bbox = proj.bbox()
                inter = bbox_intersection_area(obs_bbox, proj_bbox)
                a1, a2 = bbox_area(obs_bbox), bbox_area(proj_bbox)
                if a1 <= 0 or a2 <= 0:
                    continue
                score = max(inter / a1, inter / a2)
                if score < self.score_thresh:
                    continue
                # Depth-consistency gate: observed median depth vs expected
                # depth of the track center in this camera.
                if det_depth > 0:
                    expected = tr.ellipsoid.mean_depth_at(T_cw)
                    depth_err = abs(det_depth - expected)
                    if depth_err > self.depth_gate_m:
                        continue
                    score = score - 0.1 * depth_err  # prefer depth-consistent match
                pairs.append((score, i, tid))

        pairs.sort(key=lambda p: -p[0])
        matched_dets: set = set()
        matched_tracks: set = set()
        result: List[Tuple[int, Optional[int]]] = []
        for score, i, tid in pairs:
            if i in matched_dets or tid in matched_tracks:
                continue
            matched_dets.add(i)
            matched_tracks.add(tid)
            result.append((i, tid))
        for i in range(len(dets)):
            if i not in matched_dets:
                result.append((i, None))
        return result
