"""Data association between new detections and existing object tracks.

Ported from ObjectSceneGraph_old (VOOM/OA-SLAM `MatchObjectsWasserDistance`,
ObjectMatcher.cc): a detection is matched to the map object whose 2D projected
ellipse is most similar under the normalized Gaussian-Wasserstein distance.

    ellipse -> Gaussian (mu, Sigma = R diag(a^2,b^2) R^T)   [Ellipse2D carries mu, cov]
    W2(e1,e2) = ||mu1-mu2||^2 + tr(S1 + S2 - 2 (S1^.5 S2 S1^.5)^.5)
    NWD       = exp(-sqrt(W2) / C)                           similarity in (0,1]

Greedy per detection, gated by bbox IoU > iou_gate, accept if NWD > nwd_accept.
The old matcher has NO category-label check (category_gate=False reproduces it);
set category_gate=True to require label equality (safer for SR-style eval).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..core.geometry import Ellipse2D, ellipse_from_mask, sqrtm_2x2_spd
from ..core.types import Detection, FrameData
from .ellipsoid import Ellipsoid


def _gaussian_w2(e1: Ellipse2D, e2: Ellipse2D) -> float:
    """Squared 2-Wasserstein between the two ellipses-as-Gaussians."""
    dmu = e1.mu - e2.mu
    s1, s2 = e1.cov, e2.cov
    s1h = sqrtm_2x2_spd(s1)
    inner = sqrtm_2x2_spd(s1h @ s2 @ s1h)
    return float(dmu @ dmu + np.trace(s1 + s2 - 2.0 * inner))


def _nwd(e1: Ellipse2D, e2: Ellipse2D, C: float) -> float:
    """Normalized Gaussian-Wasserstein similarity in (0, 1]; 1 = identical."""
    return float(np.exp(-np.sqrt(max(_gaussian_w2(e1, e2), 0.0)) / C))


def _bbox_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    ix0, iy0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix1, iy1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0


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
    def __init__(
        self,
        score_thresh: float = 0.4,      # kept for construction compat (unused in Wasserstein path)
        depth_gate_m: float = 0.5,      # kept for construction compat
        wasser_C: float = 100.0,        # NWD kernel bandwidth (old ObjectMatcher.cc:138)
        iou_gate: float = 0.01,         # bbox-IoU spatial gate (old)
        nwd_accept: float = 1e-5,       # min NWD similarity to accept a match (old)
        category_gate: bool = False,    # old has NO label check; True = require label equality
    ) -> None:
        self.score_thresh = score_thresh
        self.depth_gate_m = depth_gate_m
        self.wasser_C = wasser_C
        self.iou_gate = iou_gate
        self.nwd_accept = nwd_accept
        self.category_gate = category_gate

    def associate(
        self, dets: List[Detection], frame: FrameData, tracks: List[ObjectTrack]
    ) -> List[Tuple[int, Optional[int]]]:
        """Returns [(det_idx, track_id or None)]; None means create new track."""
        K = frame.intrinsics.K()
        T_cw = frame.T_cw

        # Project every track's ellipsoid into the current frame (2D ellipse).
        projections = {}
        for tr in tracks:
            ell = tr.ellipsoid.project(K, T_cw)
            if ell is not None:
                projections[tr.id] = (tr, ell)

        pairs = []  # (nwd, det_idx, track_id)
        for i, det in enumerate(dets):
            obs = ellipse_from_mask(det.mask)
            if obs is None:
                continue
            obs_bbox = obs.bbox()
            for tid, (tr, proj) in projections.items():
                if self.category_gate and tr.label != det.label:
                    continue
                if _bbox_iou(obs_bbox, proj.bbox()) <= self.iou_gate:
                    continue
                nwd = _nwd(proj, obs, self.wasser_C)
                if nwd <= self.nwd_accept:
                    continue
                pairs.append((nwd, i, tid))

        # Greedy: highest-similarity pairs first, each detection/track once.
        pairs.sort(key=lambda p: -p[0])
        matched_dets: set = set()
        matched_tracks: set = set()
        result: List[Tuple[int, Optional[int]]] = []
        for nwd, i, tid in pairs:
            if i in matched_dets or tid in matched_tracks:
                continue
            matched_dets.add(i)
            matched_tracks.add(tid)
            result.append((i, tid))
        for i in range(len(dets)):
            if i not in matched_dets:
                result.append((i, None))
        return result
