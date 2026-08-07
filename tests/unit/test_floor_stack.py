"""Per-floor costmaps (osg.mapping.floor_stack).

Two defects these pin, both from sharing one Costmap2D across storeys:

1. An upper floor is never mapped -- its points fall outside the height band
   around the latched floor_y and are dropped, so it yields no frontiers.
2. Room ids stabilize by 2D overlap, so a room directly above another inherits
   its id (and its cached LLM label).
"""
import numpy as np
import pytest

from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN, Costmap2D
from osg.mapping.floor_stack import FloorStack
from osg.mapping.frontier import Frontier, FrontierExtractor
from osg.mapping.room_seg import RoomIdCounter, VoronoiRoomSegmenter
from osg.planning.voronoi_planner import HybridVoronoiPlanner


# ------------------------------------------------------------- the seam contract


def test_current_is_a_plain_costmap2d():
    """Everything downstream must keep receiving an ordinary 2D costmap."""
    st = FloorStack(resolution_m=0.05)
    assert isinstance(st.costmap, Costmap2D)


def test_existing_consumers_accept_a_layer_unmodified():
    """The seam is only real if the untouched consumers still work on it."""
    st = FloorStack(resolution_m=0.05)
    cm = st.costmap
    cm.grid[40:60, 40:60] = FREE
    cm.grid[60, 40:60] = OCCUPIED

    assert FrontierExtractor().extract(cm, np.array([0.0, 0.0])) is not None
    planner = HybridVoronoiPlanner(collision_m=0.25, goal_near_m=0.7, inflate_radius_m=0.25)
    assert planner.plan(cm, np.array([0.0, 0.0]), np.array([0.1, 0.1])) is not None
    assert VoronoiRoomSegmenter().segment(cm).shape == cm.grid.shape


def test_floor_zero_exists_immediately():
    st = FloorStack()
    assert len(st) == 1 and st.current_id == 0


# ------------------------------------------------------------- layer isolation


def test_layers_are_independent_grids():
    st = FloorStack(resolution_m=0.05)
    st.costmap.grid[10, 10] = OCCUPIED
    st.set_current(1)
    assert st.costmap.grid[10, 10] == UNKNOWN, "floor 1 inherited floor 0's geometry"
    st.set_current(0)
    assert st.costmap.grid[10, 10] == OCCUPIED, "floor 0 lost its geometry"


def test_upper_floor_gets_mapped_instead_of_being_dropped():
    """The headline fix: with one shared map and a latched floor_y, an upstairs
    observation lands outside the band and is discarded, so the floor stays
    entirely UNKNOWN and produces no frontiers."""
    pts_upper = np.array([[1.0, 2.9, 1.0], [1.1, 2.9, 1.0], [1.2, 3.4, 1.0]])
    frame = _fake_frame(cam=np.array([0.0, 3.68, 0.0]))

    shared = Costmap2D(resolution=0.05)
    shared.update(frame, floor_y=0.0, pts=pts_upper)  # latched to the ground floor
    # Every upstairs point is discarded. The one cell that does get written is
    # the agent's own, which `update` stamps FREE unconditionally -- so walking
    # around upstairs also punches FREE holes into the DOWNSTAIRS map, a second
    # (smaller) way the shared grid corrupts the floor below.
    touched = np.argwhere(shared.grid != UNKNOWN)
    cam_rc = shared.world_to_grid(frame.camera_position[[0, 2]])
    assert touched.shape[0] == 1 and touched[0].tolist() == cam_rc.tolist(), (
        f"expected only the agent's own cell, got {touched.shape[0]} cells"
    )

    st = FloorStack(resolution_m=0.05)
    st.set_current(1)
    st.costmap.update(frame, floor_y=2.8, pts=pts_upper)  # banded to ITS floor
    assert (st.costmap.grid != UNKNOWN).sum() > 0, "upper floor still not mapped"


def test_growth_is_per_layer():
    """Each layer auto-grows on its own; one floor's large extent must not
    force the allocation (or the origin) of another."""
    st = FloorStack(resolution_m=0.05)
    st.costmap.ensure_contains(np.array([50.0, 50.0]))
    grown = st.costmap.grid.shape
    st.set_current(1)
    assert st.costmap.grid.shape != grown
    assert st.layer(0).costmap.grid.shape == grown


# --------------------------------------------------------------- room id scope


def test_room_ids_are_unique_across_floors():
    """One flat SceneGraph.rooms dict and an id-keyed LLM label cache mean two
    floors must never mint the same room id."""
    st = FloorStack(resolution_m=0.05)
    ids = set()
    for fid in (0, 1, 2):
        st.set_current(fid)
        cm = st.costmap
        cm.grid[30:70, 30:70] = FREE
        labels = st.current.segmenter.segment(cm)
        ids_here = {int(v) for v in np.unique(labels) if v > 0}
        assert not (ids_here & ids), f"floor {fid} reused room ids {ids_here & ids}"
        ids |= ids_here
    assert len(ids) >= 3


def test_separate_segmenters_do_not_inherit_ids_across_floors():
    """A room directly above another must not take its id by 2D overlap."""
    counter = RoomIdCounter()
    a, b = (VoronoiRoomSegmenter(id_counter=counter) for _ in range(2))
    cm_a, cm_b = Costmap2D(resolution=0.05), Costmap2D(resolution=0.05)
    for cm in (cm_a, cm_b):
        cm.grid[30:70, 30:70] = FREE  # identical footprint, one above the other
    ids_a = {int(v) for v in np.unique(a.segment(cm_a)) if v > 0}
    ids_b = {int(v) for v in np.unique(b.segment(cm_b)) if v > 0}
    assert ids_a and ids_b and not (ids_a & ids_b)


def test_shared_counter_is_the_default_free_behaviour():
    """A lone segmenter still numbers from 1, as before."""
    cm = Costmap2D(resolution=0.05)
    cm.grid[30:70, 30:70] = FREE
    assert min(int(v) for v in np.unique(VoronoiRoomSegmenter().segment(cm)) if v > 0) == 1


# ------------------------------------------------------------------ transitions


def test_set_current_records_a_stair_edge():
    st = FloorStack()
    assert st.set_current(1, step=42, agent_xy=np.array([1.0, 2.0])) is True
    assert st.set_current(1, step=43) is False  # no-op
    assert len(st.stair_edges) == 1
    e = st.stair_edges[0]
    assert (e.from_floor, e.to_floor, e.step) == (0, 1, 42)


def test_entry_xy_records_first_arrival_only():
    st = FloorStack()
    st.set_current(1, step=10, agent_xy=np.array([1.0, 2.0]))
    st.set_current(0, step=20, agent_xy=np.array([9.0, 9.0]))
    st.set_current(1, step=30, agent_xy=np.array([5.0, 5.0]))
    assert st.layer(1).entry_xy.tolist() == [1.0, 2.0]


def test_reset_clears_everything():
    st = FloorStack()
    st.set_current(2, step=1)
    st.reset()
    assert len(st) == 1 and st.current_id == 0 and st.stair_edges == []


# ------------------------------------------------------------------ frontiers


def test_frontier_carries_its_floor():
    cm = Costmap2D(resolution=0.05)
    cm.grid[40:60, 40:60] = FREE
    fs = FrontierExtractor(min_cells=1).extract(cm, np.array([0.0, 0.0]), floor=3)
    assert fs and all(f.floor == 3 for f in fs)


def test_frontier_floor_defaults_to_zero():
    assert Frontier(id=0, centroid_xy=np.zeros(2), cells=np.zeros((0, 2), int), size=0).floor == 0


# ---------------------------------------------------------------------- helper


def _fake_frame(cam):
    from osg.core.types import CameraIntrinsics, FrameData

    T = np.eye(4)
    T[:3, 3] = cam
    return FrameData(
        frame_id=0,
        rgb=np.zeros((4, 4, 3), np.uint8),
        depth=np.zeros((4, 4), np.float32),
        T_wc=T,
        intrinsics=CameraIntrinsics.from_hfov(79.0, 4, 4),
    )
