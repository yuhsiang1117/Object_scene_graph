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
from ..objects.linking import reported_geometry
from ..objects.object_layer import ObjectLayer
from . import containers as containers_mod

# `SceneGraph._member_geometry`: what it memoises, in index order.
_TOP, _AREA, _QUERY, _BOTTOM = range(4)
_GEOMETRY = (
    containers_mod.top_height,
    lambda center, e: containers_mod.footprint_area(e),
    containers_mod.footprint_query,
    containers_mod.bottom_height,
)
_GEOMETRY_MEMO_MAX = 50_000
_UNSET = object()
# `SceneGraph._nearest_room_indexed`: how far from the vectorised minimum a
# room is still re-measured exactly. Room centroids are metres; the two
# computations differ by ~1e-15 of that.
_ROOM_PREFILTER_TOL_M = 1e-6

ROOM_IDS_PER_FLOOR = 1_000_000


def _room_cells(room_labels: np.ndarray) -> List[Tuple[int, np.ndarray, int]]:
    """(label, centroid (row, col), n_cells) of every non-zero label, ascending.

    What `np.unique` + one `argwhere(labels == rid).mean(0)` per room gave,
    from three bincounts: the unique-sort of the whole grid was 39% of
    `rebuild_floor`, every keyframe (cProfile, 00862 s42). The centroid sums
    are sums of integers, exact in float64 either way, so every value is the
    one the per-room loop computed. Labels that are not non-negative integers
    take the per-room loop."""
    labels = np.asarray(room_labels)
    if labels.size == 0 or not np.issubdtype(labels.dtype, np.integer) or int(labels.min()) < 0:
        out = []
        for rid in np.unique(labels):
            if rid == 0:
                continue
            mask = labels == rid
            out.append((int(rid), np.argwhere(mask).mean(axis=0), int(mask.sum())))
        return out
    flat = labels.ravel()
    counts = np.bincount(flat)
    rows, cols = np.indices(labels.shape)
    sum_r = np.bincount(flat, weights=rows.ravel().astype(float), minlength=len(counts))
    sum_c = np.bincount(flat, weights=cols.ravel().astype(float), minlength=len(counts))
    return [
        (int(rid), np.array([sum_r[rid] / counts[rid], sum_c[rid] / counts[rid]]), int(counts[rid]))
        for rid in np.nonzero(counts)[0] if rid != 0
    ]


@dataclass
class FloorNode:
    """A storey. The missing middle of the advertised building -> room -> object
    hierarchy (docs/history/MULTI_FLOOR.md); ids come from mapping.floors.FloorEstimator
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
    floor: int = 0
    container_ids: List[int] = field(default_factory=list)

    @property
    def floor_id(self) -> int:
        return self.floor

    @floor_id.setter
    def floor_id(self, value: int) -> None:
        self.floor = int(value)


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
    floor: int = 0
    object_ids: List[int] = field(default_factory=list)

    @property
    def floor_id(self) -> int:
        return self.floor

    @floor_id.setter
    def floor_id(self, value: int) -> None:
        self.floor = int(value)


@dataclass
class ObjectNodeView:
    track_id: int
    label: str
    center: np.ndarray  # (3,) world
    room_id: int  # 0 = unassigned
    n_obs: int
    best_crop: Optional[np.ndarray] = None
    floor: int = 0
    # None = resting on no mapped surface. Deliberately NOT a reason to drop the
    # object: DualMap deletes a high-mobility detection with no supporting
    # anchor (utils/local_map_manager.py:316), which is why a mug on the floor
    # is unmappable there.
    container_id: Optional[int] = None
    p_rel: Optional[np.ndarray] = None  # container-relative centre, R_c^T (t_o - t_c)

    @property
    def floor_id(self) -> int:
        return self.floor

    @floor_id.setter
    def floor_id(self, value: int) -> None:
        self.floor = int(value)


class SceneGraph:
    def __init__(
        self,
        container_top_h_m: Tuple[float, float] = containers_mod.DEFAULT_TOP_H_M,
        container_min_area_m2: float = containers_mod.DEFAULT_MIN_AREA_M2,
        container_support_tol_m: float = containers_mod.DEFAULT_SUPPORT_TOL_M,
        container_min_obs: int = 1,
        container_min_score: float = 0.0,
        container_merge_m: float = 0.0,
        containers_floor_relative: bool = False,
        container_merge_sigma: float = 0.0,
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
        # Measure a support surface's height from ITS OWN floor, not from y=0.
        self._containers_floor_relative = bool(containers_floor_relative)
        self._container_merge_sigma = float(container_merge_sigma)
        # Per-member container geometry, memoised on the exact bytes it reads
        # (`_member_geometry`), and the per-floor room arrays `_nearest_room`
        # prefilters with, alive only while `rebuild_floor` holds the rooms
        # fixed.
        self._geometry_memo: Dict[tuple, list] = {}
        self._room_index: Optional[Dict] = None

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
        # Compatibility path for the original single-grid world model.  The
        # floor-aware agent calls rebuild_floor() directly; callers that still
        # pass a FloorEstimator retain the former height-inference behavior.
        prev_room_names = {rid: room.label for rid, room in self.rooms.items()}
        self.rooms = {}
        for rid in np.unique(room_labels):
            if rid == 0:
                continue
            mask = room_labels == rid
            rc = np.argwhere(mask).mean(axis=0)
            self.rooms[int(rid)] = RoomNode(
                id=int(rid), label=prev_room_names.get(int(rid)),
                centroid_xy=costmap.grid_to_world(rc), n_cells=int(mask.sum()),
            )
        self.objects = []
        for track in object_layer.tracks():
            center = object_layer.center_of(track)
            floor_id = (
                floors.floor_of_height(float(center[HEIGHT_AXIS]))
                if floors is not None and floors.levels else 0
            )
            self.objects.append(ObjectNodeView(
                track_id=track.id, label=track.label, center=center,
                room_id=self._room_at(center[list(PLANE)], room_labels, costmap),
                n_obs=track.n_obs, best_crop=track.best_crop, floor=floor_id,
            ))
        self._rebuild_containers(object_layer, room_labels, costmap, floor_key=None)
        self._rebuild_floors(floors)

    def rebuild_floor(
        self,
        room_labels: np.ndarray,
        costmap: Costmap2D,
        object_layer: ObjectLayer,
        floor_key: Optional[int] = 0,
        floor_height: float = 0.0,
    ) -> None:
        """Replace one storey's derived nodes while retaining every other.

        Room labels are local to a floor's segmenter, so their public IDs are
        namespaced by the stable floor key.  Floor zero deliberately keeps the
        historical IDs unchanged.
        """
        floor_key = int(floor_key)
        base = floor_key * ROOM_IDS_PER_FLOOR
        prev_room_names = {rid: r.label for rid, r in self.rooms.items()}
        self.rooms = {
            rid: room for rid, room in self.rooms.items()
            if room.floor_id != floor_key
        }
        for rid, rc, n_cells in _room_cells(room_labels):
            gid = base + int(rid)
            self.rooms[gid] = RoomNode(
                id=gid,
                label=prev_room_names.get(gid),
                centroid_xy=costmap.grid_to_world(rc),
                n_cells=n_cells,
                floor=floor_key,
            )

        self.objects = [o for o in self.objects if o.floor_id != floor_key]
        # The rooms are fixed from here to the end of the container rebuild.
        self._room_index = {}
        try:
            for track in object_layer.tracks():
                if int(getattr(track, "floor_key", 0)) != floor_key:
                    continue
                center = object_layer.center_of(track)
                room_id = self._room_at(
                    center[list(PLANE)], room_labels, costmap, floor_key=floor_key
                )
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

            self._rebuild_containers(
                object_layer, room_labels, costmap, floor_key=floor_key,
                floor_height=float(floor_height),
            )
        finally:
            self._room_index = None
        self.floors[floor_key] = FloorNode(
            id=floor_key,
            height_y=float(floor_height),
            label=self.floors.get(floor_key, FloorNode(floor_key, floor_height)).label,
            room_ids=sorted(
                rid for rid, room in self.rooms.items()
                if room.floor_id == floor_key
            ),
        )

    def _merge_pass(self, ids, footprints, merge_m: float, merge_sigma: float):
        """One sweep of duplicate-container merging; returns the survivors.

        Widest first, so the node that survives is the best-supported one. Two
        tests, either sufficient: the flat `container_merge_m` radius that every
        measured arm ran on, and -- when enabled -- the extent-aware shadow
        overlap, which scales itself to the object instead of asking one radius
        to serve a 2 m bed and a 0.5 m nightstand.
        """
        order = sorted(ids, key=lambda c: -self.containers[c].area_m2)
        keep: List[int] = []
        for cid in order:
            node = self.containers.get(cid)
            if node is None:
                continue
            dup = None
            for kid in keep:
                other = self.containers[kid]
                if other.label != node.label:
                    continue
                if merge_m > 0.0 and float(
                    np.linalg.norm(other.center - node.center)
                ) <= merge_m:
                    dup = kid
                    break
                # Extent-aware. Measured against HM3D's own semantic
                # annotations, the flat 1.0 m leaves 00829 with 16 `bed` nodes
                # in a house that has ONE bed; clustering those at 2 m collapses
                # them to 7, so over half the inflation is neighbouring
                # fragments this catches and the rest is the detector calling
                # sofas and benches `bed`, which no merge rule should touch.
                if merge_sigma > 0.0 and containers_mod.shadows_overlap(
                    footprints.get(kid, ()), footprints.get(cid, ()), merge_sigma
                ):
                    dup = kid
                    break
            if dup is None:
                keep.append(cid)
            else:
                merged = self.containers[dup]
                merged.track_ids = sorted(set(merged.track_ids) | set(node.track_ids))
                merged.top_h = max(merged.top_h, node.top_h)
                if merge_sigma > 0.0:
                    # The union's geometry, not the widest member's. Two halves
                    # of a bed merged under the old rule left the anchor on one
                    # half, which is what the search walks to and what the
                    # proximity term measures from.
                    wa, wb = max(merged.area_m2, 1e-9), max(node.area_m2, 1e-9)
                    merged.center = (merged.center * wa + node.center * wb) / (wa + wb)
                    merged.area_m2 = merged.area_m2 + node.area_m2
                footprints[dup] = list(footprints.get(dup, ())) + list(footprints.get(cid, ()))
                self.containers.pop(cid, None)
                footprints.pop(cid, None)
        return keep

    def _room_at(
        self, xy: np.ndarray, room_labels: np.ndarray, costmap: Costmap2D,
        floor_key: int = 0,
    ) -> int:
        rc = costmap.world_to_grid(xy)
        if costmap.in_bounds(rc) and room_labels[rc[0], rc[1]] > 0:
            base = 0 if floor_key is None else floor_key * ROOM_IDS_PER_FLOOR
            return base + int(room_labels[rc[0], rc[1]])
        return self._nearest_room(xy, floor_id=floor_key)

    def _rebuild_containers(
        self, object_layer: ObjectLayer, room_labels: np.ndarray, costmap: Costmap2D,
        floor_key: Optional[int] = 0, floor_height: float = 0.0,
    ) -> None:
        """Attach the container layer between rooms and objects.

        Recomputed from scratch every rebuild, like linking.relink -- there is no
        incremental state to drift. Tracks without an ellipsoid (the fakes in
        tests/unit/test_scene_graph_floors.py, and anything a future data source
        supplies pose-only) simply never qualify, so this is a no-op rather than
        an error on such a layer.
        """
        self.containers = ({
            cid: node for cid, node in self.containers.items()
            if node.floor_id != floor_key
        } if floor_key is not None else {})
        for room in self.rooms.values():
            if floor_key is None or room.floor_id == floor_key:
                room.container_ids = []

        tracks = {
            t.id: t for t in object_layer.tracks()
            if floor_key is None or int(getattr(t, "floor_key", 0)) == floor_key
        }
        views = {
            o.track_id: o for o in self.objects
            if floor_key is None or o.floor_id == floor_key
        }

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
                (i, reported_geometry(tracks[i]))
                for i in sorted(members)
                if i in views
            ]
            geoms = [(i, e) for i, e in geoms if e is not None]
            if not geoms:
                continue
            # Top is the MAX over members (the tallest part of the surface);
            # area is the SUM (both halves of the sofa hold things).
            top = max(self._member_geometry(views[i].center, e, _TOP) for i, e in geoms)
            area = sum(self._member_geometry(views[i].center, e, _AREA) for i, e in geoms)
            # `container_top_h_m` is a band ABOVE THE FLOOR -- 0.2 to 1.4 m, the
            # height range a thing can be set down on -- while `top_height` is an
            # absolute world height. On the ground floor the two coincide and
            # nobody noticed. On any storey above it they do not: 00808's upper
            # floor sits at y=2.86, so every table top there measures ~3.6 and
            # fails a 1.4 m ceiling. Measured on that scene's prior map, 367
            # tracks on floor 1 produced ZERO containers against floor 0's 76.
            #
            # The consequences run through the whole multi-storey line. The
            # container posterior can propose no surface on an upper floor, and
            # `_select_surface`'s per-floor mass -- the thing that chooses a
            # storey -- has an entry only for the ground floor, so an agent
            # standing upstairs computes floor_mass {'0': 48.2} and asks to go
            # down, every round, whatever it is looking for.
            #
            # `ContainerNode.top_h` stays absolute: `build_container_candidates`
            # subtracts the floor height itself, and the two must not both do it.
            if not containers_mod.qualifies(
                rep.label,
                top - float(floor_height) if self._containers_floor_relative else top,
                area,
                top_h_m=self._container_top_h_m,
                min_area_m2=self._container_min_area_m2,
            ):
                continue
            shadows = [
                q
                for q in (
                    self._member_geometry(views[i].center, e, _QUERY) for i, e in geoms
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
                room_id=self._room_at(
                    view.center[list(PLANE)], room_labels, costmap,
                    floor_key=floor_key,
                ),
                floor=view.floor_id,
            )
            footprints[cid] = shadows

        # Merge duplicates of one piece of furniture. linking.relink cannot do
        # this: it requires co-observation, which is right for a movable object
        # and its own ghost but wrong for a bed seen from two rooms on two
        # different passes. Containers are the stable layer -- a bed does not
        # move -- so proximity and label are enough.
        merge_m = float(self._container_merge_m)
        merge_sigma = float(self._container_merge_sigma)
        current_container_ids = [
            cid for cid, node in self.containers.items()
            if floor_key is None or node.floor_id == floor_key
        ]
        if (merge_m > 0.0 or merge_sigma > 0.0) and len(current_container_ids) > 1:
            for _ in range(8):
                before = len(current_container_ids)
                current_container_ids = self._merge_pass(
                    current_container_ids, footprints, merge_m, merge_sigma
                )
                if len(current_container_ids) == before:
                    break

        current_container_ids = [
            cid for cid, node in self.containers.items()
            if floor_key is None or node.floor_id == floor_key
        ]
        for cid in sorted(current_container_ids):
            room = self.rooms.get(self.containers[cid].room_id)
            if room is not None:
                room.container_ids.append(cid)

        # 3. Rest every object on at most one container. Containers themselves
        #    are never nested, which keeps the relation a forest.
        member_of_container = {
            tid for cid, c in self.containers.items()
            if cid in current_container_ids for tid in c.track_ids
        }
        index = containers_mod.ShadowIndex.build(
            [
                (cid, self.containers[cid].top_h, mu, inv)
                for cid in sorted(current_container_ids)
                for mu, inv in footprints[cid]
            ]
        )
        tol = self._container_support_tol_m
        for obj in self.objects:
            if floor_key is not None and obj.floor_id != floor_key:
                continue
            if obj.track_id in member_of_container or index.cid.size == 0:
                continue
            ell = reported_geometry(tracks.get(obj.track_id))
            if ell is None:
                continue
            bottom = self._member_geometry(obj.center, ell, _BOTTOM)
            supporting = index.supporting(bottom, obj.center[list(PLANE)], tol_m=tol)
            if supporting.size == 0:
                continue
            # Most specific surface wins: smallest footprint, then highest.
            cid = min(
                (int(c) for c in supporting),
                key=lambda c: (self.containers[c].area_m2, -self.containers[c].top_h),
            )
            rep_ell = reported_geometry(tracks[cid])
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

    def _member_geometry(self, center, ellipsoid, what: int):
        """containers_mod's `top_height` / `footprint_area` / `footprint_query`
        / `bottom_height` of one member, memoised across rebuilds on the exact
        bytes they read (the centre, the semi-axes, the rotation): most tracks'
        ellipsoids are unchanged from one rebuild to the next, and the four
        were recomputed for every member every time (38 s of a 75 s
        scene-graph profile). Pure functions of those bytes, so the values are
        the ones the direct calls return. A shadow is shared between rebuilds
        and so read-only.
        """
        axes, R = getattr(ellipsoid, "axes", None), getattr(ellipsoid, "R", None)
        if not all(isinstance(a, np.ndarray) for a in (center, axes, R)):
            return _GEOMETRY[what](center, ellipsoid)
        key = (center.dtype.str, center.shape, center.tobytes(), axes.dtype.str, axes.tobytes(),
               R.dtype.str, R.shape, R.tobytes())
        entry = self._geometry_memo.get(key)
        if entry is None:
            if len(self._geometry_memo) >= _GEOMETRY_MEMO_MAX:
                self._geometry_memo.clear()
            entry = self._geometry_memo[key] = [_UNSET] * len(_GEOMETRY)
        value = entry[what]
        if value is _UNSET:
            value = _GEOMETRY[what](center, ellipsoid)
            if what == _QUERY and value is not None:
                for a in value:
                    a.setflags(write=False)
            entry[what] = value
        return value

    def _nearest_room(
        self, xy: np.ndarray, max_dist: float = 3.0,
        floor_id: Optional[int] = None,
    ) -> int:
        if self._room_index is not None:
            return self._nearest_room_indexed(xy, max_dist, floor_id)
        best, best_d = 0, max_dist
        for rid, room in self.rooms.items():
            if floor_id is not None and room.floor_id != floor_id:
                continue
            d = float(np.linalg.norm(room.centroid_xy - xy))
            if d < best_d:
                best, best_d = rid, d
        return best

    def _nearest_room_indexed(self, xy: np.ndarray, max_dist: float,
                              floor_id: Optional[int]) -> int:
        """`_nearest_room` with one vectorised pass to find the candidates and
        the loop above run over those alone, in the same order with the same
        arithmetic -- so the same room. The pass only has to be close: every
        room within _ROOM_PREFILTER_TOL_M of its minimum is re-measured. It
        was the loop over every room for every object of every rebuild (640k
        calls in one profiled episode)."""
        entry = self._room_index.get(floor_id)
        if entry is None:
            items = [(rid, room) for rid, room in self.rooms.items()
                     if floor_id is None or room.floor_id == floor_id]
            C = (np.array([np.asarray(room.centroid_xy, dtype=float)[:2] for _, room in items])
                 if items else np.empty((0, 2)))
            entry = self._room_index[floor_id] = (items, C)
        items, C = entry
        if not items:
            return 0
        with np.errstate(all="ignore"):
            diff = C - np.asarray(xy, dtype=float)[None, :2]
            approx = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        approx[np.isnan(approx)] = np.inf
        lo = float(approx.min())
        if not lo < float(max_dist) + _ROOM_PREFILTER_TOL_M:
            return 0
        best, best_d = 0, max_dist
        for k in np.flatnonzero(approx <= lo + _ROOM_PREFILTER_TOL_M):
            rid, room = items[int(k)]
            d = float(np.linalg.norm(room.centroid_xy - xy))
            if d < best_d:
                best, best_d = rid, d
        return best

    def objects_near(
        self, xy: np.ndarray, radius_m: float, floor_key: Optional[int] = None
    ) -> List[ObjectNodeView]:
        out = []
        for obj in self.objects:
            if floor_key is not None and obj.floor_id != floor_key:
                continue
            if np.linalg.norm(obj.center[list(PLANE)] - xy) <= radius_m:
                out.append(obj)
        return out

    def objects_in_room(self, room_id: int) -> List[ObjectNodeView]:
        return [o for o in self.objects if o.room_id == room_id]

    def room_of_point(
        self, xy: np.ndarray, floor_key: Optional[int] = None
    ) -> Optional[RoomNode]:
        rid = self._nearest_room(xy, floor_id=floor_key)
        return self.rooms.get(rid)

    def containers_in_room(self, room_id: int) -> List[ContainerNode]:
        return [c for c in self.containers.values() if c.room_id == room_id]

    def objects_on(self, container_id: int) -> List[ObjectNodeView]:
        return [o for o in self.objects if o.container_id == container_id]

    def unlabeled_rooms(self) -> List[RoomNode]:
        return [r for r in self.rooms.values() if r.label is None and self.objects_in_room(r.id)]
