"""Same-class ellipsoid linking (paper post-processing): objects that a
single ellipsoid cannot represent (L-shaped sofas) are linked; the object
center used for navigation and the scene graph is the mean of the linked
component's centers.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .association import ObjectTrack


class _UnionFind:
    def __init__(self, ids: List[int]) -> None:
        self.parent = {i: i for i in ids}

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _last_seen(track: ObjectTrack) -> Optional[int]:
    return int(track.observations[-1].frame_id) if track.observations else None


def relink(
    tracks: List[ObjectTrack],
    link_dist_m: float = 1.0,
    max_frame_gap: Optional[int] = None,
) -> None:
    """Rebuild linked_ids for all tracks (idempotent).

    `max_frame_gap` requires two tracks to have been observed at about the same
    TIME before they may be merged. Linking exists to reunite fragments of one
    object that a single ellipsoid cannot cover -- and those fragments are seen
    together, in the same frames. An object and its own past are not: when a
    mug moves half a metre, the stale track and the fresh one sit within
    link_dist_m of each other, get merged, and object_center then reports the
    midpoint of where the mug WAS and where it IS -- a place with no mug, which
    neither observation can ever contradict. Requiring co-observation separates
    "two halves of a sofa" from "an object and its ghost".
    """
    by_label: Dict[str, List[ObjectTrack]] = {}
    for tr in tracks:
        if not tr.blacklisted:
            by_label.setdefault(tr.label, []).append(tr)

    for tr in tracks:
        tr.linked_ids = set()

    for label, group in by_label.items():
        if len(group) < 2:
            continue
        uf = _UnionFind([t.id for t in group])
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                d = np.linalg.norm(group[i].ellipsoid.center - group[j].ellipsoid.center)
                if d >= link_dist_m:
                    continue
                if max_frame_gap is not None:
                    fi, fj = _last_seen(group[i]), _last_seen(group[j])
                    if fi is None or fj is None or abs(fi - fj) > max_frame_gap:
                        continue
                uf.union(group[i].id, group[j].id)
        roots: Dict[int, List[ObjectTrack]] = {}
        for t in group:
            roots.setdefault(uf.find(t.id), []).append(t)
        for members in roots.values():
            if len(members) < 2:
                continue
            ids = {m.id for m in members}
            for m in members:
                m.linked_ids = ids - {m.id}


def object_center(track: ObjectTrack, all_tracks: Dict[int, ObjectTrack]) -> np.ndarray:
    """True object center = mean of linked-component centers."""
    centers = [track.ellipsoid.center]
    for lid in track.linked_ids:
        other = all_tracks.get(lid)
        if other is not None and not other.blacklisted:
            centers.append(other.ellipsoid.center)
    return np.mean(np.stack(centers), axis=0)
