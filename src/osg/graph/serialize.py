"""Scene graph serialization: compact hierarchy text for LLM prompts and
JSON for logging / ScanNet-style graph evaluation (milestone M7).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from ..mapping.costmap import PLANE
from .scene_graph import ObjectNodeView, SceneGraph

NEAR_DIST_M = 1.5


def _near_pairs(objs: List[ObjectNodeView]) -> Dict[int, List[str]]:
    near = defaultdict(list)
    for i in range(len(objs)):
        for j in range(len(objs)):
            if i == j:
                continue
            d = np.linalg.norm(objs[i].center - objs[j].center)
            if d < NEAR_DIST_M:
                near[i].append(objs[j].label)
    return near


def to_prompt_text(sg: SceneGraph, max_near: int = 2) -> str:
    """e.g. ``Room 2 (bedroom): bed x1, lamp x2 (near bed), wardrobe x1``

    Rooms are grouped under a ``Floor n:`` heading only when more than one
    storey has been mapped -- on a single floor the heading is noise in the
    prompt, and its absence keeps the text identical to the single-floor agent.
    """
    lines = []
    floors = sorted({r.floor for r in sg.rooms.values()} | {o.floor for o in sg.objects})
    show_floors = len(floors) > 1
    room_ids = sorted(sg.rooms.keys(), key=lambda rid: (sg.rooms[rid].floor, rid))
    current_floor = None
    for rid in room_ids:
        room = sg.rooms[rid]
        objs = sg.objects_in_room(rid)
        if not objs:
            continue
        if show_floors and room.floor != current_floor:
            current_floor = room.floor
            lines.append(f"Floor {current_floor}:")
        near = _near_pairs(objs)
        by_label: Dict[str, List[int]] = defaultdict(list)
        for idx, o in enumerate(objs):
            by_label[o.label].append(idx)
        parts = []
        for label, idxs in sorted(by_label.items()):
            near_labels = Counter()
            for idx in idxs:
                near_labels.update(near.get(idx, []))
            near_labels.pop(label, None)
            suffix = ""
            if near_labels:
                top = ", ".join(l for l, _ in near_labels.most_common(max_near))
                suffix = f" (near {top})"
            parts.append(f"{label} x{len(idxs)}{suffix}")
        name = f" ({room.label})" if room.label else ""
        lines.append(f"Room {rid}{name}: " + ", ".join(parts))
    unassigned = sg.objects_in_room(0)
    if unassigned:
        counts = Counter(o.label for o in unassigned)
        lines.append("Hallway/other: " + ", ".join(f"{l} x{n}" for l, n in sorted(counts.items())))
    return "\n".join(lines) if lines else "(no objects mapped yet)"


def to_json(sg: SceneGraph) -> dict:
    return {
        "rooms": [
            {
                "id": r.id,
                "label": r.label,
                "centroid_xy": r.centroid_xy.tolist(),
                "n_cells": r.n_cells,
            }
            for r in sg.rooms.values()
        ],
        "objects": [
            {
                "track_id": o.track_id,
                "label": o.label,
                "center": o.center.tolist(),
                "room_id": o.room_id,
                "n_obs": o.n_obs,
            }
            for o in sg.objects
        ],
    }
