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
    """e.g. ``Room 2 (bedroom): bed x1, lamp x2 (near bed), wardrobe x1``"""
    lines = []
    room_ids = sorted(sg.rooms.keys())
    for rid in room_ids:
        room = sg.rooms[rid]
        objs = sg.objects_in_room(rid)
        if not objs:
            continue
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
