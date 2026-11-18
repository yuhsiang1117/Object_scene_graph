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
    if getattr(drop, "fuse_buffer", None) is not None:
        # `fuse_report`: the caller refits `extent` from the union.
        if keep.fuse_buffer is None:
            keep.fuse_buffer = drop.fuse_buffer
        else:
            keep.fuse_buffer.merge(drop.fuse_buffer)
        keep.fuse_fit_n = 0
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


def _extent(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.max(hi - lo))


def associate_xlabel(
    unmatched: Sequence[Tuple[int, str, np.ndarray]],
    tracks: Sequence[ObjectTrack],
    taken: set,
    min_overlap: float,
    size_ratio: float,
    normalize: bool,
    dilate: int = 1,
) -> Dict[int, int]:
    """`scene_graph.ellipsoid.xlabel_overlap`: {detection index: track id} for
    detections no same-label track took, joined to an OTHER-label track whose
    buffer voxels hold at least `min_overlap` of the detection's points and
    whose world extent is comparable (smaller / larger >= `size_ratio`).
    Greedy by overlap, one detection per track per frame."""
    pairs = []
    boxes = {t.id: world_aabb(t) for t in tracks
             if not t.proposal_only and getattr(t, "buffer", None) is not None and len(t.buffer)}
    for i, label, pts in unmatched:
        if pts is None or not len(pts):
            continue
        plo, phi = pts.min(axis=0), pts.max(axis=0)
        pe = _extent(plo, phi)
        for t in tracks:
            box = boxes.get(t.id)
            if box is None or t.id in taken or same_label(t.label, label, normalize):
                continue
            lo, hi = box
            if np.any(plo > hi + 0.05) or np.any(lo > phi + 0.05):
                continue
            te = _extent(lo, hi)
            if min(pe, te) < size_ratio * max(pe, te):
                continue
            ov = point_overlap(t, pts, 1.0, "voxel", dilate)
            if ov >= min_overlap:
                pairs.append((ov, i, t.id))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    out: Dict[int, int] = {}
    used = set(taken)
    for _ov, i, tid in pairs:
        if i in out or tid in used:
            continue
        out[i] = tid
        used.add(tid)
    return out


def xlabel_merge_pairs(
    tracks: Sequence[ObjectTrack],
    min_small: float,
    min_big: float,
    size_ratio: float,
    dilate: int = 1,
) -> List[Tuple[int, int]]:
    """`scene_graph.ellipsoid.xlabel_merge_every`: (drop id, keep id) pairs of
    same-floor tracks, any label, whose buffers overlap at least `min_small`
    of the smaller's points and `min_big` of the larger's, with comparable
    world extents, never observed in the same keyframe. Greedy by the
    smaller's overlap; a track takes part in at most one fold per call."""
    live = [t for t in tracks if not t.blacklisted and not t.proposal_only
            and getattr(t, "buffer", None) is not None and len(t.buffer) >= 8]
    if len(live) < 2:
        return []
    lo = np.array([world_aabb(t)[0] for t in live])
    hi = np.array([world_aabb(t)[1] for t in live])
    ext = (hi - lo).max(axis=1)
    frames = {}
    cands = []
    for a in range(len(live)):
        near = np.all(lo[a] <= hi + 0.05, axis=1) & np.all(lo <= hi[a] + 0.05, axis=1)
        near[: a + 1] = False
        for b in np.nonzero(near)[0]:
            ta, tb = live[a], live[int(b)]
            if int(ta.floor_key) != int(tb.floor_key):
                continue
            if min(ext[a], ext[b]) < size_ratio * max(ext[a], ext[b]):
                continue
            for t in (ta, tb):
                if t.id not in frames:
                    frames[t.id] = {int(o.frame_id) for o in t.observations}
            if frames[ta.id] & frames[tb.id]:
                continue
            small, big = (ta, tb) if len(ta.buffer) <= len(tb.buffer) else (tb, ta)
            f_small = float(big.buffer.contains(_probe(small.buffer.points()), dilate=dilate).mean())
            if f_small < min_small:
                continue
            f_big = float(small.buffer.contains(_probe(big.buffer.points()), dilate=dilate).mean())
            if f_big < min_big:
                continue
            cands.append((f_small, ta, tb))
    cands.sort(key=lambda c: (-c[0], c[1].id, c[2].id))
    used, out = set(), []
    for _f, ta, tb in cands:
        if ta.id in used or tb.id in used:
            continue
        keep, drop = choose_keeper(ta, tb)
        out.append((drop.id, keep.id))
        used.update((ta.id, tb.id))
    return out
