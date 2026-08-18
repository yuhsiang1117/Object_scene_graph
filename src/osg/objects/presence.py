"""Presence belief: is this object STILL where we mapped it?

The gap this closes: every object in the map used to be either present (we saw
it once) or gone (the track was never created). There was no way for the world
to tell the system an object had left -- and no way to tell "I turned away" apart
from "I am looking right at it and it is not there". DualMap has the same hole
and fills it at query time with a per-query ignore list that is discarded when
the query ends (docs/DYNAMIC_SCENES.md, Phase 1).

The filter is a textbook binary Bayes filter in log-odds with three channels
kept deliberately separate:

    Z=1, E=1 :  dl = log( r / q )            positive
    Z=0, E=1 :  dl = log( (1-r) / (1-q) )    NEGATIVE -- the new channel
    E=0      :  dl = 0                       unobserved is NOT observed-absent

`E` -- "would we have seen it if it were there" -- is the whole mechanism, and
the depth map decides it almost for free: an object that was REMOVED leaves the
surface behind it visible, so the measured depth lands BEYOND the expected band;
an object hidden by a door reads NEARER than the band. One signed comparison
separates the case that must update the belief from the case that must not.

No absorbing states: the belief is clamped, so an object at p=0.002 is still
mapped, still projected, and one detection resurrects it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set

import numpy as np

from ..core.geometry import Ellipse2D, ellipse_from_mask
from ..core.types import Detection, FrameData


@dataclass
class PresenceState:
    """Belief that the track's object is still at its mapped pose."""

    log_odds: float = 1.5  # p ~ 0.82: we just saw it, but one sighting is not proof
    last_seen_kf: int = -1
    last_absent_kf: Optional[int] = None
    n_expected: int = 0  # keyframes this belief actually rests on
    n_missed: int = 0

    @property
    def p(self) -> float:
        return 1.0 / (1.0 + math.exp(-self.log_odds))


@dataclass
class Expectation:
    """Why a track was, or was not, expected to be detected in this frame."""

    area_px: float = 0.0
    depth_m: float = 0.0
    cos_incidence: float = 1.0
    occ_ratio: float = 0.0
    n_samples: int = 0
    recall: float = 0.0


class RecallModel:
    """P(detected | present, this view).

    A constant is the honest default before anything is fitted: it makes every
    negative update the same size, which is wrong but not biased. The logistic
    form is what `scripts/fit_recall_model.py` produces from a logged run --
    features are chosen to be the ones that actually drive a detector miss
    (apparent size first, then range, then how obliquely we are looking).
    """

    FEATURES = ("bias", "log_area_px", "depth_m", "cos_incidence")

    def __init__(self, constant: float = 0.6, weights: Optional[Sequence[float]] = None,
                 floor: float = 0.02, ceil: float = 0.95) -> None:
        self.constant = float(constant)
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.floor, self.ceil = float(floor), float(ceil)

    @classmethod
    def load(cls, path: str, constant: float = 0.6) -> "RecallModel":
        """Fitted weights if the file exists, constant fallback if it does not --
        the filter must run before any fit exists."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return cls(constant=constant)
        return cls(constant=constant, weights=blob.get("weights"))

    def features(self, exp: Expectation) -> np.ndarray:
        return np.array(
            [1.0, math.log(max(exp.area_px, 1.0)), exp.depth_m, exp.cos_incidence],
            dtype=float,
        )

    def __call__(self, exp: Expectation) -> float:
        if self.weights is None:
            r = self.constant
        else:
            z = float(self.features(exp) @ self.weights)
            r = 1.0 / (1.0 + math.exp(-z))
        return float(np.clip(r, self.floor, self.ceil))


class PresenceFilter:
    """Applies the three channels to every track, once per keyframe."""

    def __init__(
        self,
        recall: Optional[RecallModel] = None,
        q_false_alarm: float = 0.05,
        l_clamp: float = 6.0,
        occ_ratio_max: float = 0.30,
        depth_tol_m: float = 0.15,
        min_area_px: float = 1500.0,
        range_m: tuple = (0.4, 6.0),
        img_inside_frac: float = 0.5,
        min_depth_samples: int = 12,
        max_samples: int = 256,
        max_tracks: int = 64,
        z_overlap_iou: float = 0.05,
        log_path: str = "",
    ) -> None:
        self.recall = recall or RecallModel()
        self.q = float(q_false_alarm)
        self.l_clamp = float(l_clamp)
        self.occ_ratio_max = float(occ_ratio_max)
        self.depth_tol_m = float(depth_tol_m)
        # Expectation and ADMISSION must share a scale threshold: expecting a
        # detection at a size ObjectLayer would have filtered out anyway
        # manufactures a false negative on every distant object in the room.
        self.min_area_px = float(min_area_px)
        self.range_m = (float(range_m[0]), float(range_m[1]))
        self.img_inside_frac = float(img_inside_frac)
        self.min_depth_samples = int(min_depth_samples)
        self.max_samples = int(max_samples)
        self.max_tracks = int(max_tracks)
        self.z_overlap_iou = float(z_overlap_iou)
        self.log_path = str(log_path)
        self.n_expected = 0
        self.n_negative = 0
        self.n_positive = 0

    # ------------------------------------------------------------- geometry

    @staticmethod
    def _inside_image_frac(ellipse: Ellipse2D, w: int, h: int) -> float:
        x1, y1, x2, y2 = ellipse.bbox()
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area <= 0.0:
            return 0.0
        ix = max(0.0, min(x2, w) - max(x1, 0.0))
        iy = max(0.0, min(y2, h) - max(y1, 0.0))
        return float(ix * iy / area)

    def _sample_pixels(self, ellipse: Ellipse2D, w: int, h: int) -> Optional[np.ndarray]:
        """Integer pixels inside the projected ellipse, capped in count."""
        x1, y1, x2, y2 = ellipse.bbox()
        x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        x2, y2 = min(w, int(np.ceil(x2)) + 1), min(h, int(np.ceil(y2)) + 1)
        if x2 <= x1 or y2 <= y1:
            return None
        n = (x2 - x1) * (y2 - y1)
        stride = max(1, int(np.sqrt(n / max(self.max_samples, 1))))
        us, vs = np.meshgrid(np.arange(x1, x2, stride), np.arange(y1, y2, stride))
        pts = np.stack([us.ravel(), vs.ravel()], axis=1).astype(float)
        d = pts - ellipse.mu[None, :]
        try:
            inv = np.linalg.inv(ellipse.cov)
        except np.linalg.LinAlgError:
            return None
        inside = np.einsum("ij,jk,ik->i", d, inv, d) <= 1.0
        return pts[inside].astype(int) if inside.any() else None

    def expectation(
        self, track, frame: FrameData, ellipse: Optional[Ellipse2D] = None
    ) -> Optional[Expectation]:
        """None means E=0: this frame says nothing about the track.

        Order matters only for speed -- the cheap gates (cheirality, range) run
        before the depth read. `ellipse` lets `update` hand in the projection it
        already computed for the sighting test; projecting every track twice per
        keyframe was measurably the filter's largest cost.
        """
        T_cw = frame.T_cw
        if ellipse is None:
            ellipse = track.ellipsoid.project(frame.intrinsics.K(), T_cw)
        if ellipse is None:
            return None

        h, w = frame.depth.shape
        if self._inside_image_frac(ellipse, w, h) < self.img_inside_frac:
            return None

        z_c = track.ellipsoid.mean_depth_at(T_cw)
        if not (self.range_m[0] <= z_c <= self.range_m[1]):
            return None
        if ellipse.area < self.min_area_px:
            return None

        pts = self._sample_pixels(ellipse, w, h)
        if pts is None:
            return None
        depths = frame.depth[pts[:, 1], pts[:, 0]]
        valid = depths > 1e-3
        if int(valid.sum()) < self.min_depth_samples:
            return None  # nothing readable here; refuse to conclude anything
        depths = depths[valid]

        # Expected depth band from the quadric, along the centre ray. Using one
        # ray for the whole silhouette is an approximation that costs nothing
        # and is well inside depth_tol_m for object-scale ellipsoids.
        cam_pos = frame.camera_position
        ray = track.ellipsoid.center - cam_pos
        norm = float(np.linalg.norm(ray))
        if norm < 1e-6:
            return None
        extent = track.ellipsoid.world_extent(ray / norm)
        near = z_c - extent - self.depth_tol_m
        far = z_c + extent + self.depth_tol_m

        occluded = float(np.mean(depths < near))
        if occluded > self.occ_ratio_max:
            return None  # something is in front: this frame cannot see the pose

        # How obliquely are we looking? Cheap proxy: the camera's optical axis
        # against the direction to the object.
        optical = frame.T_wc[:3, 2]
        cos_inc = float(abs(optical @ (ray / norm)))

        exp = Expectation(
            area_px=float(ellipse.area),
            depth_m=float(z_c),
            cos_incidence=cos_inc,
            occ_ratio=occluded,
            n_samples=int(valid.sum()),
        )
        exp.recall = self.recall(exp)
        return exp

    # -------------------------------------------------------------- updates

    def _detected_ids(self, projections, dets: Sequence[Detection]) -> Set[int]:
        """Z=1 for any track whose projection overlaps ANY detection, whatever
        its label.

        Association gates on category (ObjectLayer passes
        assoc_category_gate=True), so a mug relabelled 'bowl' would otherwise
        register as a miss and collapse the mug's belief on a RELABEL rather
        than a removal. Presence asks whether something is there; which thing it
        is stays the association's problem.
        """
        seen: Set[int] = set()
        if not dets or not projections:
            return seen
        det_boxes = []
        for det in dets:
            e = ellipse_from_mask(det.mask)
            if e is not None:
                det_boxes.append(e.bbox())
        if not det_boxes:
            return seen
        for track, proj in projections:
            pb = proj.bbox()
            if any(_bbox_iou(pb, db) > self.z_overlap_iou for db in det_boxes):
                seen.add(track.id)
        return seen

    def update(self, tracks: List, frame: FrameData, dets: Sequence[Detection]) -> None:
        """One keyframe of evidence for every track in view.

        Runs on frames with ZERO detections -- that is the frame where negative
        evidence is worth the most.
        """
        if not tracks:
            return
        cam = frame.camera_position
        near_first = sorted(
            tracks, key=lambda t: float(np.linalg.norm(t.ellipsoid.center - cam))
        )[: self.max_tracks]

        # Project once; both the sighting test and the expectation need it.
        K, T_cw = frame.intrinsics.K(), frame.T_cw
        projections = []
        for track in near_first:
            proj = track.ellipsoid.project(K, T_cw)
            if proj is not None:
                projections.append((track, proj))
        seen = self._detected_ids(projections, dets)

        rows = []
        for track, proj in projections:
            state = track.presence
            exp = self.expectation(track, frame, ellipse=proj)
            detected = track.id in seen
            if exp is not None:
                self.n_expected += 1
                state.n_expected += 1
            if detected:
                # A sighting is positive evidence whether or not the geometry
                # said we should have got one.
                r = exp.recall if exp is not None else self.recall.constant
                state.log_odds += math.log(r / self.q)
                state.last_seen_kf = frame.frame_id
                self.n_positive += 1
            elif exp is not None:
                state.log_odds += math.log((1.0 - exp.recall) / (1.0 - self.q))
                state.last_absent_kf = frame.frame_id
                state.n_missed += 1
                self.n_negative += 1
            state.log_odds = float(np.clip(state.log_odds, -self.l_clamp, self.l_clamp))

            if self.log_path and exp is not None:
                rows.append(
                    {
                        "kf": int(frame.frame_id),
                        "track_id": int(track.id),
                        "label": str(track.label),
                        "area_px": round(exp.area_px, 1),
                        "depth_m": round(exp.depth_m, 3),
                        "cos_incidence": round(exp.cos_incidence, 4),
                        "occ_ratio": round(exp.occ_ratio, 3),
                        "detected": int(detected),
                    }
                )
        if rows:
            self._log(rows)

    def _log(self, rows: List[dict]) -> None:
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
        except OSError:
            self.log_path = ""  # logging is diagnostic; never fail a run over it


def _bbox_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    ix0, iy0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix1, iy1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0
