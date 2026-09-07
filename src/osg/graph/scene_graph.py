"""Hierarchical scene graph: building -> room -> object.

Kept as plain dataclasses over the live ObjectLayer (object nodes are thin
views over tracks; the graph never owns object state). Object-object "near"
edges are derived on demand during serialization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..mapping.costmap import PLANE, Costmap2D
from ..objects.object_layer import ObjectLayer
from ..perception.keyframe import KeyframeRef


# Room ids are namespaced by floor: each floor runs its own segmenter, and
# every one of them emits a room 1. Without this the graph would merge
# unrelated rooms from different storeys under one id.
ROOM_IDS_PER_FLOOR = 1000


@dataclass
class RoomNode:
    id: int
    label: Optional[str] = None  # e.g. "bedroom"; set by the LLM, cached
    centroid_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    n_cells: int = 0
    floor: int = 0  # FloorLayer.key


@dataclass
class ObjectNodeView:
    track_id: int
    label: str
    center: np.ndarray  # (3,) world
    room_id: int  # 0 = unassigned
    n_obs: int
    best_crop: Optional[np.ndarray] = None
    floor: int = 0  # FloorLayer.key


class SceneGraph:
    def __init__(self) -> None:
        self.rooms: Dict[int, RoomNode] = {}
        self.objects: List[ObjectNodeView] = []

    def rebuild(
        self,
        room_labels: np.ndarray,  # (H, W) int32 room-id map (0 = none)
        costmap: Costmap2D,
        object_layer: ObjectLayer,
    ) -> None:
        """Single-floor rebuild: replaces the whole graph."""
        self.rebuild_floor(room_labels, costmap, object_layer, floor_key=0)

    def rebuild_floor(
        self,
        room_labels: np.ndarray,
        costmap: Costmap2D,
        object_layer: ObjectLayer,
        floor_key: int = 0,
    ) -> None:
        """Rebuild only the nodes belonging to `floor_key`.

        Other floors' rooms and objects are left untouched: the segmentation
        and costmap passed in describe one storey, and re-running it must not
        drop everything mapped elsewhere -- the agent still needs that context
        for frontier scoring and for knowing where it has already been.
        """
        prev_room_names = {rid: r.label for rid, r in self.rooms.items()}
        base = floor_key * ROOM_IDS_PER_FLOOR
        self.rooms = {rid: r for rid, r in self.rooms.items() if r.floor != floor_key}
        for rid in np.unique(room_labels):
            if rid == 0:
                continue
            mask = room_labels == rid
            rc = np.argwhere(mask).mean(axis=0)
            gid = base + int(rid)
            self.rooms[gid] = RoomNode(
                id=gid,
                label=prev_room_names.get(gid),
                centroid_xy=costmap.grid_to_world(rc),
                n_cells=int(mask.sum()),
                floor=floor_key,
            )

        self.objects = [o for o in self.objects if o.floor != floor_key]
        for track in object_layer.tracks():
            if getattr(track, "floor_key", 0) != floor_key:
                continue
            center = object_layer.center_of(track)
            rc = costmap.world_to_grid(center[list(PLANE)])
            room_id = 0
            if costmap.in_bounds(rc) and room_labels[rc[0], rc[1]] > 0:
                room_id = base + int(room_labels[rc[0], rc[1]])
            else:
                room_id = self._nearest_room(center[list(PLANE)], floor=floor_key)
            self.objects.append(
                ObjectNodeView(
                    track_id=track.id,
                    label=track.label,
                    center=center,
                    room_id=room_id,
                    n_obs=track.n_obs,
                    best_crop=track.best_crop,
                    floor=floor_key,
                )
            )

    def _nearest_room(
        self, xy: np.ndarray, max_dist: float = 3.0, floor: Optional[int] = None
    ) -> int:
        """Nearest room by ground-plane distance. `floor` confines the search
        to one storey -- rooms directly above and below are metres apart in 3D
        but coincide in (x, z), so an unconstrained search happily assigns an
        object to the room above it."""
        best, best_d = 0, max_dist
        for rid, room in self.rooms.items():
            if floor is not None and room.floor != floor:
                continue
            d = float(np.linalg.norm(room.centroid_xy - xy))
            if d < best_d:
                best, best_d = rid, d
        return best

    def objects_near(self, xy: np.ndarray, radius_m: float) -> List[ObjectNodeView]:
        out = []
        for obj in self.objects:
            if np.linalg.norm(obj.center[list(PLANE)] - xy) <= radius_m:
                out.append(obj)
        return out

    def objects_in_room(self, room_id: int) -> List[ObjectNodeView]:
        return [o for o in self.objects if o.room_id == room_id]

    def room_of_point(self, xy: np.ndarray) -> Optional[RoomNode]:
        rid = self._nearest_room(xy)
        return self.rooms.get(rid)

    def unlabeled_rooms(self) -> List[RoomNode]:
        return [r for r in self.rooms.values() if r.label is None and self.objects_in_room(r.id)]
