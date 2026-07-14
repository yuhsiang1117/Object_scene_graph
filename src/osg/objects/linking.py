"""Same-class ellipsoid linking (paper post-processing): objects that a
single ellipsoid cannot represent (L-shaped sofas) are linked; the object
center used for navigation and the scene graph is the mean of the linked
component's centers.
"""
from __future__ import annotations

from typing import Dict, List

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


def relink(tracks: List[ObjectTrack], link_dist_m: float = 1.0) -> None:
    """Rebuild linked_ids for all tracks (idempotent)."""
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
                if d < link_dist_m:
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
