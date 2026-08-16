"""One occupancy map per storey.

The single shared `Costmap2D` is what confines the pipeline to one floor, and
it does so in two ways (docs/MULTI_FLOOR.md):

1. **Upper floors are never mapped.** `Costmap2D.update` bands points by height
   above a `floor_y` that was latched on the first frame, so once the agent
   climbs, every observation falls outside the band and is dropped. No cells,
   no frontiers, nothing to explore -- which is why `floor_transitions` was 0
   on all 24 cross-floor episodes of the last full run.
2. **Stair geometry pollutes the floor below.** Part-way up a staircase the
   agent's own surroundings land back inside the band, stamping treads and
   whatever is visible from them as obstacles at their (x, z) in the ground
   floor's grid -- and `_raycast_batch` never clears OCCUPIED, so it is
   permanent.

A `FloorStack` gives each storey its own `Costmap2D`, room segmenter and room
labels, keyed by the floor ids from `mapping.floors.FloorEstimator`. Crucially
each layer is still a plain 2D `Costmap2D` in the same (x, z) plane, so every
consumer -- planner, frontier extractor, viewpoint planner, controller, the
visualizers -- keeps working unchanged on whichever layer it is handed. Floor
is an index into a dict, never a change to the coordinate convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .costmap import Costmap2D
from .room_seg import RoomIdCounter, VoronoiRoomSegmenter


@dataclass
class FloorLayer:
    """Everything that is per-storey. `room_labels` is kept beside its costmap
    because the two must share a shape; the previous single `_room_labels` on
    the agent silently re-segmented whenever the grid grew."""

    floor_id: int
    costmap: Costmap2D
    segmenter: VoronoiRoomSegmenter
    room_labels: Optional[np.ndarray] = None
    entry_xy: Optional[np.ndarray] = None  # where the agent first arrived
    first_step: int = 0


@dataclass
class StairEdge:
    """A traversal between two storeys, recorded from the floor estimator's own
    committed transitions. Consumed by Stage 4/5 (cross-floor frontiers)."""

    from_floor: int
    to_floor: int
    entry_xy: Optional[np.ndarray] = None
    exit_xy: Optional[np.ndarray] = None
    step: int = 0
    n_traversals: int = 1


class FloorStack:
    def __init__(
        self,
        resolution_m: float = 0.05,
        room_seg_kwargs: Optional[dict] = None,
        current: int = 0,
        track_height: bool = False,
    ) -> None:
        self.resolution_m = float(resolution_m)
        self._track_height = bool(track_height)
        self._room_seg_kwargs = dict(room_seg_kwargs or {})
        self._room_ids = RoomIdCounter()
        self._layers: Dict[int, FloorLayer] = {}
        self.current_id = int(current)
        self.stair_edges: List[StairEdge] = []
        self.layer(self.current_id)  # floor 0 always exists

    # ------------------------------------------------------------------ access

    def layer(self, floor_id: int, step: int = 0) -> FloorLayer:
        """The layer for a storey, created on first visit."""
        fid = int(floor_id)
        if fid not in self._layers:
            self._layers[fid] = FloorLayer(
                floor_id=fid,
                costmap=Costmap2D(
                    resolution=self.resolution_m, track_height=self._track_height
                ),
                # One segmenter PER FLOOR: VoronoiRoomSegmenter._stabilize_ids
                # matches rooms to the previous call by 2D overlap, so a shared
                # segmenter would hand a room its predecessor's id -- and the
                # cached LLM room label with it -- purely for being directly
                # above it. Separate instances make that impossible, with no
                # change to _stabilize_ids itself.
                segmenter=VoronoiRoomSegmenter(
                    id_counter=self._room_ids, **self._room_seg_kwargs
                ),
                first_step=step,
            )
        return self._layers[fid]

    @property
    def current(self) -> FloorLayer:
        return self.layer(self.current_id)

    def visited_ids(self) -> List[int]:
        """Storeys the agent has actually stood on. Layers are created lazily on
        first visit, so the key set is the visit record."""
        return sorted(self._layers)

    @property
    def costmap(self) -> Costmap2D:
        """The seam: consumers receive an ordinary 2D costmap and never learn
        that floors exist."""
        return self.current.costmap

    def set_current(
        self, floor_id: int, step: int = 0, agent_xy: Optional[np.ndarray] = None
    ) -> bool:
        """Switch storeys. Returns True if this was a change."""
        fid = int(floor_id)
        if fid == self.current_id:
            return False
        prev = self.current_id
        layer = self.layer(fid, step=step)
        if layer.entry_xy is None and agent_xy is not None:
            layer.entry_xy = np.asarray(agent_xy, dtype=float).copy()
        self.stair_edges.append(
            StairEdge(
                from_floor=prev, to_floor=fid,
                # Where the agent was standing when it committed to the new
                # storey: the mouth of the staircase on the floor it left, and
                # its landing point on the floor it reached.
                entry_xy=None if agent_xy is None else np.asarray(agent_xy, float).copy(),
                exit_xy=layer.entry_xy, step=step,
            )
        )
        self.current_id = fid
        return True

    # ------------------------------------------------------------------ dunder

    def __len__(self) -> int:
        return len(self._layers)

    def __contains__(self, floor_id: int) -> bool:
        return int(floor_id) in self._layers

    def __iter__(self) -> Iterator[int]:
        return iter(self._layers)

    def items(self) -> List[Tuple[int, FloorLayer]]:
        return sorted(self._layers.items())

    def layers(self) -> List[FloorLayer]:
        return [l for _, l in self.items()]

    def reset(self) -> None:
        self._layers = {}
        self._room_ids = RoomIdCounter()
        self.current_id = 0
        self.stair_edges = []
        self.layer(0)
