"""Telling one object's tracks apart from two objects' -- by the space they fill.

Projection association (objects/association.py) asks whether a detection's
ellipse looks like a track's projected ellipse. It fails exactly when the
track's ellipsoid is wrong: the projection then misses the IoU gate and the
same object starts a second track. These tests ask a question the bad
ellipsoid cannot spoil as easily -- do this detection's depth points lie
where the track's own points (or its ellipsoid, inflated) already are?

The same measure keeps the ghost rule of linking.relink
(docs/history/DYNAMIC_SCENES.md): an object moved half a metre leaves its old
track's points where it WAS, and the new points do not overlap them, so the
two are never merged -- unlike a 1 m centre radius, which merges them.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.labels import normalize_label
from .association import ObjectTrack
from .feature_memory import merge_running_mean

_PROBE_POINTS = 300


def _probe(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float)
    if len(pts) <= _PROBE_POINTS:
        return pts
    return pts[np.linspace(0, len(pts) - 1, _PROBE_POINTS).astype(int)]


def surface_samples(track: ObjectTrack, n: int = 64) -> np.ndarray:
    """A track's buffered points, or (no buffer) points on its ellipsoid."""
    buf = getattr(track, "buffer", None)
    if buf is not None and len(buf):
        return buf.points()
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    unit = np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], axis=1)
    e = track.ellipsoid
    return (unit * e.axes) @ e.R.T + e.center


def point_overlap(track: ObjectTrack, pts: np.ndarray, inflate: float,
                  mode: str = "ellipsoid", dilate: int = 1) -> float:
    """Share of `pts` inside the track's ellipsoid scaled by `inflate`, or in
    (one voxel of) its buffer. `mode="voxel"`: the buffer only, so a track
    with no buffered surface overlaps nothing."""
    pts = _probe(pts)
    if not len(pts):
        return 0.0
    buf = getattr(track, "buffer", None)
    if mode == "voxel":
        if buf is None or not len(buf):
            return 0.0
        return float(buf.contains(pts, dilate=dilate).mean())
    e = track.ellipsoid
    q = ((pts - e.center) @ e.R) / (e.axes * float(inflate))
    inside = np.linalg.norm(q, axis=1) <= 1.0
    if buf is not None and len(buf):
        rest = ~inside
        if rest.any():
            inside[rest] = buf.contains(pts[rest], dilate=dilate)
    return float(inside.mean())


def same_label(a: str, b: str, normalize: bool) -> bool:
    return normalize_label(a) == normalize_label(b) if normalize else a == b


def associate_3d(
    unmatched: Sequence[Tuple[int, str, np.ndarray]],
    tracks: Sequence[ObjectTrack],
    taken: set,
    min_overlap: float,
    inflate: float,
    normalize: bool,
    mode: str = "ellipsoid",
    dilate: int = 1,
    unique: bool = False,
) -> Dict[int, int]:
    """{detection index: track id} for detections the projection test left
    unmatched, greedy by overlap. `taken` holds the tracks already matched
    this frame: one detection per track per frame, as in DataAssociator."""
    pairs = []
    for i, label, pts in unmatched:
        if pts is None or not len(pts):
            continue
        for t in tracks:
            if t.id in taken or t.proposal_only or not same_label(t.label, label, normalize):
                continue
            ov = point_overlap(t, pts, inflate, mode, dilate)
            if ov >= min_overlap:
                pairs.append((ov, i, t.id))
    if unique:
        hits = Counter(i for _ov, i, _t in pairs)
        pairs = [p for p in pairs if hits[p[1]] == 1]
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    out: Dict[int, int] = {}
    used = set(taken)
    for ov, i, tid in pairs:
        if i in out or tid in used:
            continue
        out[i] = tid
        used.add(tid)
    return out


def mutual_overlap(a: ObjectTrack, b: ObjectTrack, inflate: float,
                   mode: str = "ellipsoid", dilate: int = 1, symmetric: bool = False) -> float:
    """The larger of the two one-way overlaps: a fragment lies inside the
    whole even though the whole does not lie inside the fragment.
    `symmetric`: the smaller -- each must lie inside the other."""
    return (min if symmetric else max)(point_overlap(b, surface_samples(a), inflate, mode, dilate),
               point_overlap(a, surface_samples(b), inflate, mode, dilate))


def merge_into(keep: ObjectTrack, drop: ObjectTrack) -> None:
    """Fold `drop` into `keep`: one object's evidence, observations and views."""
    keep.observations = sorted(keep.observations + drop.observations,
                               key=lambda o: int(o.frame_id))
    keep.evidence += drop.evidence
    keep.n_proposal_obs += drop.n_proposal_obs
    if drop.best_score > keep.best_score:
        for name in ("best_score", "best_crop", "best_frame_rgb", "best_bbox_xyxy",
                     "best_bbox_px", "best_cam_xy"):
            setattr(keep, name, getattr(drop, name))
    if drop.clip_ft is not None:
        if keep.clip_ft is None:
            keep.clip_ft, keep.clip_n = drop.clip_ft, drop.clip_n
        else:
            for _ in range(int(drop.clip_n)):
                keep.clip_ft, keep.clip_n = merge_running_mean(keep.clip_ft, keep.clip_n, drop.clip_ft)
    keep.presence.log_odds = max(keep.presence.log_odds, drop.presence.log_odds)
    keep.presence.last_seen_kf = max(keep.presence.last_seen_kf, drop.presence.last_seen_kf)
    keep.presence.n_expected += drop.presence.n_expected
    keep.presence.n_missed += drop.presence.n_missed
    for name in ("identity_rejections", "absence_arrivals", "failed_attempts"):
        setattr(keep, name, max(getattr(keep, name), getattr(drop, name)))
    keep.from_prior = bool(keep.from_prior and drop.from_prior)
    keep.out_of_range = bool(keep.out_of_range and drop.out_of_range)
    keep.last_kf = max(getattr(keep, "last_kf", -1), getattr(drop, "last_kf", -1))
    if getattr(drop, "buffer", None) is not None:
        if keep.buffer is None:
            keep.buffer = drop.buffer
        else:
            keep.buffer.merge(drop.buffer)
    if drop.points_w is not None:
        keep.points_w = (drop.points_w if keep.points_w is None
                         else np.vstack([keep.points_w, drop.points_w])[-max(len(keep.points_w), len(drop.points_w)):])


def choose_keeper(a: ObjectTrack, b: ObjectTrack) -> Tuple[ObjectTrack, ObjectTrack]:
    """The better-supported track survives; ties go to the older id."""
    key = lambda t: (not t.blacklisted, t.n_obs, -t.id)
    return (a, b) if key(a) >= key(b) else (b, a)


def world_aabb(track: ObjectTrack) -> Tuple[np.ndarray, np.ndarray]:
    """(lo, hi) of the track's ellipsoid on the world axes."""
    e = track.ellipsoid
    half = np.sqrt(np.maximum(np.diag(e.R @ np.diag(np.asarray(e.axes, float) ** 2) @ e.R.T), 0.0))
    return e.center - half, e.center + half


def consolidate_minor(
    tracks: Sequence[ObjectTrack],
    max_obs: int,
    gap_m: float,
    ratio: float,
    normalize: bool,
) -> List[Tuple[int, int]]:
    """[(minor id, major id)]: each same-label, same-floor track with at most
    `max_obs` observations goes to the nearest track with more than `max_obs`
    and at least `ratio` times its observations, whose world AABB is no more
    than `gap_m` from its own (negative: overlapping by that much) and with
    which it never shared a keyframe. Smallest minors first; a major's
    keyframes grow with what it absorbs, its observation count does not.

    Measured on the I6r1 pass-1 snapshots (scratch simulation against
    instance GT): 44% of the duplicate tracks have <= 2 observations; a
    geometric pairwise rule between ANY two same-label tracks is right only
    about half the time, adjacent same-label objects being as close as one
    object's fragments, so only the minor-into-major direction is used."""
    def key(t):
        return (normalize_label(t.label) if normalize else t.label, int(t.floor_key))

    live = [t for t in tracks if not t.proposal_only and not t.blacklisted]
    n = {t.id: len(t.observations) for t in live}
    box = {t.id: world_aabb(t) for t in live}
    frames = {t.id: {int(o.frame_id) for o in t.observations} for t in live}
    groups: Dict[Tuple[str, int], List[ObjectTrack]] = {}
    for t in live:
        groups.setdefault(key(t), []).append(t)
    out: List[Tuple[int, int]] = []
    for group in groups.values():
        majors = [t.id for t in group if n[t.id] > max_obs]
        if not majors:
            continue
        for m in sorted((t.id for t in group if n[t.id] <= max_obs), key=lambda i: (n[i], i)):
            lo, hi = box[m]
            best = None
            for M in majors:
                if n[M] < ratio * n[m] or frames[m] & frames[M]:
                    continue
                gap = float(np.max(np.maximum(lo, box[M][0]) - np.minimum(hi, box[M][1])))
                if gap <= gap_m and (best is None or (gap, M) < best):
                    best = (gap, M)
            if best is not None:
                out.append((m, best[1]))
                frames[best[1]] |= frames[m]
    return out
