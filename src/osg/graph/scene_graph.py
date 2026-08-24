"""Hierarchical scene graph: building -> room -> object.

Kept as plain dataclasses over the live ObjectLayer (object nodes are thin
views over tracks; the graph never owns object state). Object-object "near"
edges are derived on demand during serialization.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..mapping.costmap import HEIGHT_AXIS, PLANE, Costmap2D
from ..objects.object_layer import ObjectLayer
from . import containers as containers_mod


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
    container_ids: List[int] = field(default_factory=list)


@dataclass
class ContainerNode:
    """A support surface: the layer DualMap calls an anchor.

    A container is a VIEW over tracks that already exist, never a new entity --
    a table is one ObjectTrack that appears both here and in `objects`, keeping
    this module's contract that the graph never owns object state. `id` is the
    smallest track id in the linked component (linking.relink already unions an
    L-shaped sofa's two ellipsoids), so one physical surface is one anchor.
    """

    id: int
    label: str
    track_ids: List[int]
    center: np.ndarray  # (3,) world; the linked component's mean centre
    top_h: float  # world height of the support surface
    area_m2: float  # ground footprint area, summed over members
    room_id: int = 0
    floor_id: int = 0
    object_ids: List[int] = field(default_factory=list)


@dataclass
class ObjectNodeView:
    track_id: int
    label: str
    center: np.ndarray  # (3,) world
    room_id: int  # 0 = unassigned
    n_obs: int
    best_crop: Optional[np.ndarray] = None
    floor_id: int = 0
    # None = resting on no mapped surface. Deliberately NOT a reason to drop the
    # object: DualMap deletes a high-mobility detection with no supporting
    # anchor (utils/local_map_manager.py:316), which is why a mug on the floor
    # is unmappable there.
    container_id: Optional[int] = None
    p_rel: Optional[np.ndarray] = None  # container-relative centre, R_c^T (t_o - t_c)


class SceneGraph:
    def __init__(
        self,
        container_top_h_m: Tuple[float, float] = containers_mod.DEFAULT_TOP_H_M,
        container_min_area_m2: float = containers_mod.DEFAULT_MIN_AREA_M2,
        container_support_tol_m: float = containers_mod.DEFAULT_SUPPORT_TOL_M,
        container_min_obs: int = 1,
        container_min_score: float = 0.0,
        container_merge_m: float = 0.0,
    ) -> None:
        self.rooms: Dict[int, RoomNode] = {}
        self.objects: List[ObjectNodeView] = []
        self.floors: Dict[int, FloorNode] = {}
        self.containers: Dict[int, ContainerNode] = {}
        self._container_top_h_m = tuple(container_top_h_m)
        self._container_min_area_m2 = float(container_min_area_m2)
        self._container_support_tol_m = float(container_support_tol_m)
        self._container_min_obs = int(container_min_obs)
        self._container_min_score = float(container_min_score)
        self._container_merge_m = float(container_merge_m)

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
            room_id = self._room_at(center[list(PLANE)], room_labels, costmap)
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

        self._rebuild_containers(object_layer, room_labels, costmap)
        self._rebuild_floors(floors)

    def _room_at(self, xy: np.ndarray, room_labels: np.ndarray, costmap: Costmap2D) -> int:
        rc = costmap.world_to_grid(xy)
        if costmap.in_bounds(rc) and room_labels[rc[0], rc[1]] > 0:
            return int(room_labels[rc[0], rc[1]])
        return self._nearest_room(xy)

    def _rebuild_containers(
        self, object_layer: ObjectLayer, room_labels: np.ndarray, costmap: Costmap2D
    ) -> None:
        """Attach the container layer between rooms and objects.

        Recomputed from scratch every rebuild, like linking.relink -- there is no
        incremental state to drift. Tracks without an ellipsoid (the fakes in
        tests/unit/test_scene_graph_floors.py, and anything a future data source
        supplies pose-only) simply never qualify, so this is a no-op rather than
        an error on such a layer.
        """
        self.containers = {}
        for room in self.rooms.values():
            room.container_ids = []

        tracks = {t.id: t for t in object_layer.tracks()}
        views = {o.track_id: o for o in self.objects}

        # 1. Linked components: an L-shaped sofa split across two ellipsoids is
        #    ONE surface. relink() gives every member the full component, so the
        #    minimum id is a stable canonical name for it.
        # A search candidate has to be a surface that is really there. The map
        # held 112 of them for a six-object hotel suite -- 29 "bed"s, 34 seen
        # exactly once, 46 scoring under 0.5 -- and the search budget is seven
        # to nine inspections, so two thirds of every episode was spent on
        # detector noise. Filtering to twice-seen, half-confident surfaces and
        # merging duplicates takes the true destination of a cross-anchor move
        # from rank 20 of 112 to rank 7 of 46, and into the inspected set in
        # five of nine relocations instead of three.
        min_obs = int(self._container_min_obs)
        min_score = float(self._container_min_score)
        tracks = {
            tid: t for tid, t in tracks.items()
            if getattr(t, "n_obs", 0) >= min_obs
            and float(getattr(t, "best_score", 0.0)) >= min_score
        }

        comps: Dict[int, set] = {}
        for tid, track in tracks.items():
            linked = getattr(track, "linked_ids", None) or set()
            members = {tid} | {i for i in linked if i in tracks}
            comps.setdefault(min(members), set()).update(members)

        # 2. Which components qualify as containers.
        footprints: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
        for cid, members in comps.items():
            rep = tracks[cid]
            view = views.get(cid)
            if view is None:
                continue
            geoms = [
                (i, getattr(tracks[i], "ellipsoid", None))
                for i in sorted(members)
                if i in views
            ]
            geoms = [(i, e) for i, e in geoms if e is not None]
            if not geoms:
                continue
            # Top is the MAX over members (the tallest part of the surface);
            # area is the SUM (both halves of the sofa hold things).
            top = max(containers_mod.top_height(views[i].center, e) for i, e in geoms)
            area = sum(containers_mod.footprint_area(e) for _, e in geoms)
            if not containers_mod.qualifies(
                rep.label,
                top,
                area,
                top_h_m=self._container_top_h_m,
                min_area_m2=self._container_min_area_m2,
            ):
                continue
            shadows = [
                q
                for q in (
                    containers_mod.footprint_query(views[i].center, e) for i, e in geoms
                )
                if q is not None
            ]
            if not shadows:
                continue
            self.containers[cid] = ContainerNode(
                id=cid,
                label=rep.label,
                track_ids=[i for i, _ in geoms],
                center=view.center.copy(),
                top_h=top,
                area_m2=area,
                room_id=self._room_at(view.center[list(PLANE)], room_labels, costmap),
                floor_id=view.floor_id,
            )
            footprints[cid] = shadows

        # Merge duplicates of one piece of furniture. linking.relink cannot do
        # this: it requires co-observation, which is right for a movable object
        # and its own ghost but wrong for a bed seen from two rooms on two
        # different passes. Containers are the stable layer -- a bed does not
        # move -- so proximity and label are enough.
        merge_m = float(self._container_merge_m)
        if merge_m > 0.0 and len(self.containers) > 1:
            # Widest first, so the surviving node is the best-supported one.
            order = sorted(self.containers, key=lambda c: -self.containers[c].area_m2)
            keep: List[int] = []
            for cid in order:
                node = self.containers[cid]
                dup = None
                for kid in keep:
                    other = self.containers[kid]
                    if other.label != node.label:
                        continue
                    if float(np.linalg.norm(other.center - node.center)) <= merge_m:
                        dup = kid
                        break
                if dup is None:
                    keep.append(cid)
                else:
                    merged = self.containers[dup]
                    merged.track_ids = sorted(set(merged.track_ids) | set(node.track_ids))
                    merged.top_h = max(merged.top_h, node.top_h)
                    footprints[dup] = footprints[dup] + footprints[cid]
            dropped = [c for c in self.containers if c not in keep]
            for cid in dropped:
                self.containers.pop(cid, None)
                footprints.pop(cid, None)

        for cid in sorted(self.containers):
            room = self.rooms.get(self.containers[cid].room_id)
            if room is not None:
                room.container_ids.append(cid)

        # 3. Rest every object on at most one container. Containers themselves
        #    are never nested, which keeps the relation a forest.
        member_of_container = {
            tid for c in self.containers.values() for tid in c.track_ids
        }
        index = containers_mod.ShadowIndex.build(
            [
                (cid, self.containers[cid].top_h, mu, inv)
                for cid in sorted(self.containers)
                for mu, inv in footprints[cid]
            ]
        )
        tol = self._container_support_tol_m
        for obj in self.objects:
            if obj.track_id in member_of_container or index.cid.size == 0:
                continue
            ell = getattr(tracks.get(obj.track_id), "ellipsoid", None)
            if ell is None:
                continue
            bottom = containers_mod.bottom_height(obj.center, ell)
            supporting = index.supporting(bottom, obj.center[list(PLANE)], tol_m=tol)
            if supporting.size == 0:
                continue
            # Most specific surface wins: smallest footprint, then highest.
            cid = min(
                (int(c) for c in supporting),
                key=lambda c: (self.containers[c].area_m2, -self.containers[c].top_h),
            )
            rep_ell = getattr(tracks[cid], "ellipsoid", None)
            obj.container_id = cid
            obj.p_rel = containers_mod.relative_pose(
                self.containers[cid].center, rep_ell.R, obj.center
            )
            self.containers[cid].object_ids.append(obj.track_id)

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

    def containers_in_room(self, room_id: int) -> List[ContainerNode]:
        return [c for c in self.containers.values() if c.room_id == room_id]

    def objects_on(self, container_id: int) -> List[ObjectNodeView]:
        return [o for o in self.objects if o.container_id == container_id]

    def unlabeled_rooms(self) -> List[RoomNode]:
        return [r for r in self.rooms.values() if r.label is None and self.objects_in_room(r.id)]
