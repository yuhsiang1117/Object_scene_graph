"""Hierarchical scene graph: building -> room -> object.

Kept as plain dataclasses over the live ObjectLayer (object nodes are thin
views over tracks; the graph never owns object state). Object-object "near"
edges are derived on demand during serialization.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..mapping.costmap import HEIGHT_AXIS, PLANE, Costmap2D
from ..objects.object_layer import ObjectLayer
from ..perception.keyframe import KeyframeRef


@dataclass
class FloorNode:
    """A storey. The missing middle of the advertised building -> room -> object
    hierarchy (docs/MULTI_FLOOR.md); ids come from mapping.floors.FloorEstimator
    and are creation-ordered, so never infer "upstairs" from the id."""

    id: int
    height_y: float
    label: Optional[str] = None
    room_ids: List[int] = field(default_factory=list)


@dataclass
class RoomNode:
    id: int
    label: Optional[str] = None  # e.g. "bedroom"; set by the LLM, cached
    centroid_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    n_cells: int = 0
    floor_id: int = 0


@dataclass
class ObjectNodeView:
    track_id: int
    label: str
    center: np.ndarray  # (3,) world
    room_id: int  # 0 = unassigned
    n_obs: int
    best_crop: Optional[np.ndarray] = None
    floor_id: int = 0


class SceneGraph:
    def __init__(self) -> None:
        self.rooms: Dict[int, RoomNode] = {}
        self.objects: List[ObjectNodeView] = []
        self.floors: Dict[int, FloorNode] = {}

    def rebuild(
        self,
        room_labels: np.ndarray,  # (H, W) int32 room-id map (0 = none)
        costmap: Costmap2D,
        object_layer: ObjectLayer,
        floors=None,  # mapping.floors.FloorEstimator, or None
    ) -> None:
        """Rebuild the graph over the live object layer.

        `floors` is optional: without it every node lands on floor 0, which is
        exactly the previous single-floor behaviour. With it, an object's storey
        comes from its 3D centre height -- the height that used to be discarded
        here, so a bed upstairs and a bed directly below it were one node's
        worth of ambiguity to every consumer.
        """
        prev_room_names = {rid: r.label for rid, r in self.rooms.items()}
        self.rooms = {}
        for rid in np.unique(room_labels):
            if rid == 0:
                continue
            mask = room_labels == rid
            rc = np.argwhere(mask).mean(axis=0)
            self.rooms[int(rid)] = RoomNode(
                id=int(rid),
                label=prev_room_names.get(int(rid)),
                centroid_xy=costmap.grid_to_world(rc),
                n_cells=int(mask.sum()),
            )

        self.objects = []
        for track in object_layer.tracks():
            center = object_layer.center_of(track)
            rc = costmap.world_to_grid(center[list(PLANE)])
            room_id = 0
            if costmap.in_bounds(rc) and room_labels[rc[0], rc[1]] > 0:
                room_id = int(room_labels[rc[0], rc[1]])
            else:
                room_id = self._nearest_room(center[list(PLANE)])
            self.objects.append(
                ObjectNodeView(
                    track_id=track.id,
                    label=track.label,
                    center=center,
                    room_id=room_id,
                    n_obs=track.n_obs,
                    best_crop=track.best_crop,
                    floor_id=floors.floor_of_height(float(center[HEIGHT_AXIS]))
                    if floors is not None and floors.levels else 0,
                )
            )

        self._rebuild_floors(floors)

    def _rebuild_floors(self, floors) -> None:
        """Attach floor nodes and push each room onto a storey.

        While there is one shared costmap, a room is a 2D region that cannot
        itself be split by height, so a room takes the storey most of its
        objects are on. Per-floor room segmentation supersedes this once each
        floor has its own costmap.
        """
        self.floors = {}
        if floors is None or not floors.levels:
            return
        for fid, height in floors.levels.items():
            self.floors[fid] = FloorNode(id=fid, height_y=height)
        for rid, room in self.rooms.items():
            objs = [o for o in self.objects if o.room_id == rid]
            if objs:
                room.floor_id = Counter(o.floor_id for o in objs).most_common(1)[0][0]
            else:
                room.floor_id = floors.current
            self.floors.setdefault(
                room.floor_id, FloorNode(id=room.floor_id, height_y=0.0)
            ).room_ids.append(rid)

    def _nearest_room(self, xy: np.ndarray, max_dist: float = 3.0) -> int:
        best, best_d = 0, max_dist
        for rid, room in self.rooms.items():
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
