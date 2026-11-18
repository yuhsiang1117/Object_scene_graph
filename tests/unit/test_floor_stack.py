"""Per-floor costmaps (osg.mapping.floor_stack).

Two defects these pin, both from sharing one Costmap2D across storeys:

1. An upper floor is never mapped -- its points fall outside the height band
   around the latched floor_y and are dropped, so it yields no frontiers.
2. Room ids stabilize by 2D overlap, so a room directly above another inherits
   its id (and its cached LLM label).
"""
import types

import numpy as np
import pytest

from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN, Costmap2D
from osg.mapping.floor_stack import FloorStack
from osg.mapping.value_map import ValueMap2D
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


def test_optional_value_maps_are_scoped_to_each_floor():
    st = FloorStack(value_map_factory=lambda cm: ValueMap2D(cm))
    lower = st.current.value_map
    st.set_current(1)
    upper = st.current.value_map
    assert lower is not None and upper is not None and lower is not upper
    lower.value[2, 3] = 0.9
    assert upper.value[2, 3] == 0.0


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


# --------------------------------------------------------------- FloorPolicy
#
# The multi-floor path has no end-to-end coverage: the mounted data is
# single-floor, so `floor.enabled=false` is the only branch a trajectory lock
# can reach. These are the cheap insurance for the branch that cannot be run --
# that the policy constructs, observes, and stays inert when it is switched off.


def _policy(**floor_overrides):
    from osg.agent.floor_policy import FloorPolicy
    from osg.core.config import OSGConfig

    cfg = OSGConfig()
    for key, value in floor_overrides.items():
        setattr(cfg.floor, key, value)
    return FloorPolicy(cfg, {})


def _frame_at(y: float, step_xy=(0.0, 0.0)):
    from osg.core.types import CameraIntrinsics

    import numpy as np

    from tests.unit.conftest import make_frame

    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = float(step_xy[0]), float(y), float(step_xy[1])
    k = CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0, width=640, height=480)
    return make_frame(k, T)


def test_the_policy_is_inert_when_floors_are_off():
    """Every default reproduces single-floor behaviour, so the whole file must
    be a no-op path -- it logs the estimate and hands back the height latched on
    frame 1, exactly as the agent did before there were storeys."""
    policy = _policy()
    first = policy.observe(_frame_at(1.5), step=1)
    later = policy.observe(_frame_at(1.9), step=2)  # a step up, but off
    assert first == later, "estimate_only must not move the obstacle band"
    assert policy.pursuing is False
    assert policy.switch_policy is None, "no cross-floor policy unless asked for"
    assert len(policy.floor_log) == 1


def test_a_live_floor_estimate_feeds_the_costmap_band():
    policy = _policy(enabled=True, estimate_only=False, per_floor_costmap=True)
    ground = policy.observe(_frame_at(1.5), step=1)
    assert ground == pytest.approx(1.5 - 0.88, abs=1e-6)
    # A storey up, held long enough to commit, opens its own grid.
    for step in range(2, 40):
        policy.observe(_frame_at(4.6), step=step)
    assert len(policy.levels) >= 2
    assert policy.floor_y_drift > 1.0, "the band moved, which is the point"


def test_a_switch_is_a_decision_the_caller_applies():
    """try_switch returns a PortalGoal or None and never touches FSM state --
    the policy decides which storey, the agent decides to move."""
    policy = _policy(enabled=True, estimate_only=False, per_floor_costmap=True,
                     cross_floor=True)
    assert policy.switch_policy is not None
    policy.observe(_frame_at(1.5), step=1)
    # Nothing mapped, no portals: the honest answer is "stay".
    assert policy.try_switch(
        _frame_at(1.5), step=300, best_path_cost=99.0,
        scene_graph=types.SimpleNamespace(objects=[], rooms={}),
        target="chair", reachable_fn=None,
    ) is None
    assert policy.pursuing is False


def test_a_directed_switch_uses_the_portal_toward_the_requested_stable_floor():
    """A cross-floor search posterior names a stable key, not merely "leave".
    With openings in both directions, that key must determine up versus down."""
    policy = _policy(enabled=True, estimate_only=False, per_floor_costmap=True,
                     cross_floor=True)
    policy.stack._layers = {}
    for key, height in ((4, 0.0), (9, 2.8), (12, 5.6)):
        policy.stack.set_height(key, height)
    policy.stack.current_id = 9
    policy.estimator._levels = {4: 0.0, 9: 2.8, 12: 5.6}
    policy.estimator.current = 9
    cm = policy.costmap
    cm.grid[:] = 0
    cm.height[100:200, 100:200] = 2.8
    cm.height[120:140, 120:140] = 5.6
    cm.height[160:180, 160:180] = 0.0

    goal = policy.try_switch(
        _frame_at(3.68), step=1, best_path_cost=1.0,
        scene_graph=types.SimpleNamespace(objects=[], rooms={}),
        target="chair", reachable_fn=None, target_floor=4,
    )

    assert goal is not None
    assert goal.target_y == pytest.approx(0.0, abs=0.05)
    assert policy.stats["directed_floor_switch_attempts"] == 1
