"""ASCENT's stair machinery in ascentnav, and the maps and instrumentation
around it.

The first ascentnav run scored 0.0% on all 21 cross-floor episodes because the
ObstacleMap was handed zeros for both stair masks -- everything downstream
existed and never fired. These pin the pieces that make it fire, and (S71)
the transcription of `Map_Controller`'s climb state machine that drives them.
"""
from __future__ import annotations

import numpy as np
import pytest

from ascentnav.stairs import (
    CLIMB_PAUSED_ABANDON,
    StairController,
    carrot_waypoint,
    ratchet_carrot,
    robot_on_stairs,
    stairs_in_upper_half,
)


# ------------------------------------------------------------ footprint test

def test_robot_on_stairs_uses_a_circular_footprint():
    m = np.zeros((100, 100), np.uint8)
    m[50, 55] = 1                       # 5 px to the right of (50,50)
    px = np.array([[50.0, 50.0]])
    assert robot_on_stairs(m, px, radius_px=6) is True
    assert robot_on_stairs(m, px, radius_px=3) is False


def test_robot_on_stairs_handles_an_empty_map():
    assert robot_on_stairs(np.zeros((10, 10), np.uint8), np.array([[5.0, 5.0]]), 3) is False
    assert robot_on_stairs(None, np.array([[5.0, 5.0]]), 3) is False


def test_footprint_is_clipped_at_the_map_edge():
    m = np.zeros((20, 20), np.uint8); m[0, 0] = 1
    assert robot_on_stairs(m, np.array([[0.0, 0.0]]), 2) is True
# ---------------------------------------------------------------- the carrot

def test_carrot_bearing_is_ccw_in_ascents_frame():
    """ASCENT SUBTRACTS the pixel offset because its episodic frame is
    CCW-positive -- the opposite sign to the OSG-plane port in nav_agent. The
    two are not interchangeable, so both are pinned."""
    d = np.zeros((10, 640), np.float32)
    d[:, 639] = 1.0                          # farthest thing on the RIGHT
    g = carrot_waypoint(d, np.zeros(2), heading=0.0, hfov=np.radians(79))
    assert g[1] < 0, "a far pixel on the right is a negative (clockwise) bearing"
    d2 = np.zeros((10, 640), np.float32); d2[:, 0] = 1.0
    assert carrot_waypoint(d2, np.zeros(2), 0.0, np.radians(79))[1] > 0


def test_carrot_is_placed_at_the_configured_distance():
    d = np.zeros((10, 640), np.float32); d[:, 320] = 1.0
    g = carrot_waypoint(d, np.array([2.0, -1.0]), 0.0, np.radians(79), carrot_m=0.8)
    assert np.linalg.norm(g - np.array([2.0, -1.0])) == pytest.approx(0.8)


def test_empty_depth_yields_no_carrot():
    assert carrot_waypoint(np.zeros((0, 0), np.float32), np.zeros(2), 0.0, 1.4) is None


# --------------------------------------------------------------- the ratchet

def _to_px(xy):
    return np.atleast_2d(np.asarray(xy, dtype=float) * 20.0)


def test_ratchet_keeps_the_carrot_closer_to_the_stair_end():
    end_px = np.array([200.0, 0.0])          # the stair end, 10 m along +x
    robot_px = np.array([[0.0, 0.0]])
    near, far = np.array([5.0, 0.0]), np.array([1.0, 0.0])
    kept = ratchet_carrot(near, far, end_px, robot_px, _to_px, 20.0, False)
    assert np.allclose(kept, near), "a carrot further from the end must be refused"
    took = ratchet_carrot(far, near, end_px, robot_px, _to_px, 20.0, False)
    assert np.allclose(took, near)


def test_disable_end_releases_the_ratchet():
    end_px = np.array([200.0, 0.0]); robot_px = np.array([[0.0, 0.0]])
    fresh = np.array([1.0, 0.0])
    assert np.allclose(
        ratchet_carrot(np.array([5.0, 0.0]), fresh, end_px, robot_px, _to_px, 20.0, True),
        fresh)


def test_no_recorded_end_means_no_ratchet():
    fresh = np.array([1.0, 0.0])
    assert np.allclose(
        ratchet_carrot(np.array([5.0, 0.0]), fresh, np.array([]), np.array([[0.0, 0.0]]),
                       _to_px, 20.0, False),
        fresh)

# ================================================================ fixtures
#
# These drive `AscentNavAgent` directly. The agent is built with a stub
# detector and a scripted mover, so no weights and no servers are touched.

from ascentnav.agent import AscentNavAgent  # noqa: E402
from osg.perception.detector import StubDetector  # noqa: E402

from .test_ascent_agent import _Driver  # noqa: E402
from .test_nav_agent import _StubScorer  # noqa: E402


def _agent(driver=None, **over):
    """The real OSGConfig, not the hand-rolled fixture: this agent reads
    `cfg.eval` for the camera model, which `make_cfg` does not carry."""
    from osg.core.config import OSGConfig
    from osg.exploration.async_scorer import AsyncScorer

    cfg = OSGConfig()
    cfg.agent.navigation = "pointnav"
    cfg.agent.initial_scan = False
    cfg.agent.stair_up_mode = "ascent"             # what the preset sets; passive entry needs it
    for k, v in over.items():
        setattr(cfg.agent, k, v)
    a = AscentNavAgent(cfg, StubDetector(), AsyncScorer(_StubScorer()),
                       None, "chair", pointnav=driver or _Driver())
    a.reset("chair")
    return a


def _paint_stairs(agent, direction=1, xy=(2.0, 0.0), r=6):
    """Put a staircase on the map where the ObstacleMap would have put one.

    Also moves the floor past the stairwell-reinitialisation window
    (`ascent_policy.py:709-713` fires within the first 50 steps of a floor
    whenever an unexplored direction has no frontier yet), so these tests
    exercise the branch AFTER it, which is where a real climb starts."""
    om = agent.obstacle_map
    om._floor_num_steps = 60
    m = om._up_stair_map if direction == 1 else om._down_stair_map
    c = om._xy_to_px(np.atleast_2d(np.asarray(xy, dtype=float)))[0]
    m[int(c[1]) - r:int(c[1]) + r, int(c[0]) - r:int(c[0]) + r] = 1
    f = np.atleast_2d(np.asarray(xy, dtype=float))
    if direction == 1:
        om._has_up_stair, om._up_stair_frontiers = True, f
    else:
        om._has_down_stair, om._down_stair_frontiers = True, f
    return om


DEPTH = np.zeros((48, 64), np.float32)
DEPTH[:, 32] = 1.0                                # far pixel dead ahead


# ============================================================== map bookkeeping
def test_each_map_owns_its_trajectory():
    """`BaseMap._camera_positions` is a CLASS attribute that `update_agent_traj`
    appends to in place. Two maps in one process would then share one path --
    which is what painted episode N-1's trajectory onto episode N's map, and
    floor 0's onto floor 1's."""
    from ascentnav.mapping.obstacle_map import ObstacleMap

    a = ObstacleMap(min_height=0.61, max_height=0.88, agent_radius=0.18, size=200)
    b = ObstacleMap(min_height=0.61, max_height=0.88, agent_radius=0.18, size=200)
    a.update_agent_traj(np.zeros(2), 0.0)
    assert len(a._camera_positions) == 1
    assert len(b._camera_positions) == 0, "a second map inherited the first's path"


def test_a_new_episode_starts_with_an_empty_trajectory():
    a = _agent()
    a.obstacle_map.update_agent_traj(np.zeros(2), 0.0)
    a.reset("chair")
    assert len(a.obstacle_map._camera_positions) == 0

def test_a_new_floor_starts_with_an_empty_trajectory():
    a = _agent()
    a.obstacle_map.update_agent_traj(np.zeros(2), 0.0)
    a._floors.append(a._new_floor())
    a._floor_idx = 1
    assert len(a.obstacle_map._camera_positions) == 0
    assert len(a._floors[0]["obstacle"]._camera_positions) == 1, "floor 0 keeps its own"


# ========================================================= is the purple right?
#
# The stair region on the debug video looked displaced. These separate the two
# candidate explanations: the RENDERER putting a correct map in the wrong place,
# or the MAP holding the wrong cells. They pin the renderer, so a displacement
# seen in a video is attributable to the map.

def test_a_painted_stair_cell_renders_where_the_world_says_it_is():
    from ascentnav.viz import _vis_px, obstacle_panel

    a = _agent()
    om = a.obstacle_map
    world = np.array([3.0, -2.0])
    px = om._xy_to_px(np.atleast_2d(world))[0]
    om._up_stair_map[px[1] - 2:px[1] + 3, px[0] - 2:px[0] + 3] = 1
    om.explored_area[:] = 1                       # so the crop covers the map

    img = om.visualize()                          # the vendored renderer
    purple = np.argwhere(np.all(img == (128, 0, 128), axis=-1))
    assert len(purple) > 0, "the stair cells were not drawn at all"
    centre = purple.mean(0)[::-1]                 # (row, col) -> (x, y)
    assert np.allclose(centre, _vis_px(om, world), atol=1.5), (
        "the renderer and the map disagree about where this cell is")


def test_the_agent_marker_and_the_map_share_one_frame():
    """A marker drawn 1 m ahead of a north-facing agent must land 20 px (one
    metre at 20 px/m) above it in the image, not beside it."""
    from ascentnav.viz import _vis_px

    a = _agent()
    om = a.obstacle_map
    here = _vis_px(om, np.array([0.0, 0.0]))
    ahead = _vis_px(om, np.array([1.0, 0.0]))     # +x
    left = _vis_px(om, np.array([0.0, 1.0]))      # +y
    assert ahead[1] == here[1] - om.pixels_per_meter and ahead[0] == here[0]
    assert left[0] == here[0] - om.pixels_per_meter and left[1] == here[1]


# ==================================================== where the stairs land
#
# The regression these guard is subtle and was invisible for three runs: the
# stair pixels were SELECTED correctly and projected at the wrong RANGE, so the
# staircase appeared along the right bearing at max_depth. It looked like a
# calibration shift, not a crash.

def _wall_frame(pos, look_at, range_m=3.0):
    from osg.core.types import CameraIntrinsics, FrameData
    from .conftest import make_camera
    intr = CameraIntrinsics.from_hfov(79.0, 640, 480)
    return FrameData(frame_id=0, rgb=np.zeros((480, 640, 3), np.uint8),
                     depth=np.full((480, 640), range_m, np.float32),
                     T_wc=make_camera(pos, look_at), intrinsics=intr)


def _paint(det_mask_dtype, range_m=3.0, patch=(200, 280, 280, 360)):
    """Project one patch of a fronto-parallel wall and return the painted xy."""
    from ascentnav.constants import STAIR_CLASS_ID
    from ascentnav.geometry import camera_pitch, normalise_depth, tf_camera_to_episodic
    from ascentnav.mapping.obstacle_map import ObstacleMap

    f = _wall_frame([0, 0.88, 0], [1, 0.88, 0], range_m)
    tf = tf_camera_to_episodic(f, 0.88)
    fx = fy = 640 / (2 * np.tan(np.radians(79.0) / 2))
    v0, v1, u0, u1 = patch
    m = np.zeros((480, 640), np.uint8)
    m[v0:v1, u0:u1] = 1
    seg = np.where(m.astype(bool), STAIR_CLASS_ID, 0).astype(np.uint8)

    om = ObstacleMap(min_height=0.61, max_height=0.88, agent_radius=0.18, size=800)
    om.update_map(normalise_depth(f.depth, 0.5, 5.0), tf, 0.5, 5.0, fx, fy,
                  np.radians(79.0), {}, np.zeros((480, 640), np.uint8),
                  m.astype(det_mask_dtype), seg,
                  float(np.degrees(-camera_pitch(f))), True, False, 0)
    px = np.argwhere(om._up_stair_map)
    return om._px_to_xy(px[:, ::-1].astype(float))


def test_stairs_are_painted_at_their_true_range():
    """A patch of wall 3 m ahead must be painted at 3 m, not at max_depth."""
    xy = _paint(np.uint8, range_m=3.0)
    assert len(xy) > 0, "nothing was painted at all"
    assert xy[:, 0].mean() == pytest.approx(3.0, abs=0.1), (
        f"painted at {xy[:, 0].mean():.2f} m instead of 3.0 m")


def test_a_uint8_detector_mask_projects_the_same_as_a_bool_one():
    """`uint8 & bool` promotes to uint8, and indexing a depth array with a uint8
    array is INTEGER ROW indexing, not masking -- which left every stair pixel
    at max_depth. ASCENT's own masks are bool, so its code never sees this."""
    a, b = _paint(np.uint8), _paint(bool)
    assert np.allclose(np.sort(a, axis=0), np.sort(b, axis=0))


def test_the_range_error_scaled_with_max_depth_not_the_scene():
    """The signature of the old bug: the painted range was pinned to max_depth,
    so a 2 m wall and a 3 m wall landed in the same place. They must not."""
    near = _paint(np.uint8, range_m=2.0)[:, 0].mean()
    far = _paint(np.uint8, range_m=3.5)[:, 0].mean()
    assert near == pytest.approx(2.0, abs=0.1)
    assert far == pytest.approx(3.5, abs=0.1)
    assert far - near > 1.0

# ============================================================ the state machine
#
# `Map_Controller._process_stair_climb_state` (`map_controller.py:259-316`)
# and the dispatch in `Ascent_Policy.act` (`ascent_policy.py:447-557`),
# transcribed. The reference's own inconsistencies are kept and pinned here.

def _px(agent, xy):
    return agent.obstacle_map._xy_to_px(np.atleast_2d(np.asarray(xy, dtype=float)))


def test_an_exhausted_floor_with_an_unexplored_storey_navigates_to_the_stairs():
    """`_explore` :716-728: no frontiers, an unexplored floor above -> drive
    at the up-stair frontier on this very step."""
    a = _agent()
    _paint_stairs(a)
    a._floors.append(a._new_floor())               # the storey the stairs lead to
    action = a._explore(DEPTH, np.zeros(2), 0.0)
    assert a.stairs.climbing and a.stairs.direction == 1
    assert action == "move_forward"


def test_a_floor_with_frontiers_does_not_climb():
    a = _agent()
    _paint_stairs(a)
    a.obstacle_map.frontiers = np.array([[3.0, 3.0], [-3.0, 1.0]])
    a._explore(DEPTH, np.zeros(2), 0.0)
    assert not a.stairs.climbing


def test_no_frontiers_and_no_unexplored_storey_is_a_terminal_stop():
    """`ascent_policy.py:725-726` -- the reference STOPs. The port turned left
    forever, which is one of the two ways its budget went into the ground."""
    a = _agent()
    a.obstacle_map._floor_num_steps = 60           # past the reinit window
    assert a._explore(DEPTH, np.zeros(2), 0.0) == "stop"
    assert a._state == "done" and a.approach_stop_reason == "explored_out"


def test_an_early_frontier_collapse_reinitialises_with_the_tight_threshold():
    """`_handle_stairwell_reinitialization` (:764-811): within the first 50
    steps of a floor, with an unexplored staircase that has no frontier yet,
    reset the maps, keep the stair fields, turn `_tight_search_thresh` on and
    start the opening scan again."""
    a = _agent()
    om = a.obstacle_map
    om._floor_num_steps = 10
    om._has_up_stair = True
    om._up_stair_map[100:110, 100:110] = 1
    om._explored_up_stair = False
    om._up_stair_frontiers = np.array([])
    a._done_initializing = True
    action = a._explore(DEPTH, np.zeros(2), 0.0)
    om = a.obstacle_map
    assert action == "turn_left"
    assert om._reinitialize_flag and om._tight_search_thresh
    assert om._has_up_stair and om._up_stair_map.sum() == 100, "stair fields survive the reset"
    assert a._done_initializing is False and a._initialize_step == 1


def test_reinitialisation_happens_at_most_once_per_floor():
    a = _agent()
    om = a.obstacle_map
    om._floor_num_steps = 10
    om._has_up_stair = True
    om._explored_up_stair = False
    om._up_stair_frontiers = np.array([])
    om._reinitialize_flag = True
    assert a._explore(DEPTH, np.zeros(2), 0.0) == "stop"


def test_the_approach_runs_until_the_footprint_touches_the_stairs():
    a = _agent()
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    a._explore(DEPTH, np.zeros(2), 0.0)
    # far away: still approaching
    a.stairs.pre_update(a, om, np.zeros(2), _px(a, (0.0, 0.0)))
    assert not a.stairs.reach_stair
    # on top of it: reached, start recorded
    a.stairs.pre_update(a, om, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))
    assert a.stairs.reach_stair
    assert np.allclose(om._up_stair_start, _px(a, (2.0, 0.0))[0])


def test_leaving_the_stairs_after_the_centroid_completes_the_transition():
    a = _agent()
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    a._explore(DEPTH, np.zeros(2), 0.0)
    a.stairs.pre_update(a, om, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))   # reached
    a.stairs.pre_update(a, om, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))   # centroid (<=0.3 m)
    assert a.stairs.reach_stair_centroid
    a.stairs.pre_update(a, om, np.array([9.0, 0.0]), _px(a, (9.0, 0.0)))   # off the map
    assert not a.stairs.climbing
    assert a._floor_idx == 1, "arrived on the storey above"
    assert a.stats["climb_ok"] == 1
    assert a._done_initializing is False, "a fresh storey gets the opening scan"


def test_arriving_upstairs_hands_the_flight_to_the_new_floor():
    """`_update_linked_stair_map` (:433-476): the staircase just climbed is the
    arrival floor's DOWN staircase, ends swapped, already explored -- or the
    new floor rediscovers it and climbs straight back down."""
    a = _agent()
    om0 = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    a._explore(DEPTH, np.zeros(2), 0.0)
    for xy in ((2.0, 0.0), (2.0, 0.0), (9.0, 0.0)):
        a.stairs.pre_update(a, om0, np.array(xy), _px(a, xy))
    om1 = a.obstacle_map
    assert om1._has_down_stair and om1._explored_down_stair
    assert om1._down_stair_map.sum() > 0
    assert np.allclose(om1._down_stair_start, om0._up_stair_end)


def test_a_stalled_flight_off_the_stairs_is_burned_and_the_storey_dropped():
    """`:279-297`: reached the centroid, paused >= 30, no longer on the stairs
    -> disable, burn the stair map, delete the neighbour floor."""
    a = _agent()
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    a._explore(DEPTH, np.zeros(2), 0.0)
    a.stairs.pre_update(a, om, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))
    a.stairs.pre_update(a, om, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))
    om._climb_stair_paused_step = CLIMB_PAUSED_ABANDON
    a.stairs.pre_update(a, om, np.array([9.0, 0.0]), _px(a, (9.0, 0.0)))
    assert not a.stairs.climbing and not om._has_up_stair
    assert len(a._floors) == 1 and a.stats["climb_fail"] == 1


def test_disabling_a_stair_frontier_does_not_burn_the_map_by_default():
    """F3: `_disable_stair_and_reset_state` (:328-380) zeroes the climb flag
    at :350 before testing it at :354/:367, so the burn never runs and a failed
    staircase is retried. The faithful default keeps that; `burn_on_disable`
    is the A/B that runs the code as written."""
    a = _agent()
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a.stairs.start_navigating(om, 1)
    a.stairs.disable_stair_and_reset(a, om, np.array([2.0, 0.0]))
    assert not a.stairs.climbing
    assert (2.0, 0.0) in om._disabled_frontiers
    assert om._has_up_stair and om._up_stair_map.sum() > 0, "the map is NOT burned"
    b = _agent(stair_disable_burns_map=True)
    om = _paint_stairs(b, xy=(2.0, 0.0))
    b.stairs.start_navigating(om, 1)
    b.stairs.disable_stair_and_reset(b, om, np.array([2.0, 0.0]))
    assert not om._has_up_stair and om._up_stair_map.sum() == 0


def test_the_stair_approach_retires_a_frontier_it_cannot_close_on():
    """`_get_close_to_stair` (:1015-1042): 30 stalled steps, or 60 in total."""
    a = _agent(driver=_Driver(action="turn_left"))
    om = _paint_stairs(a, xy=(5.0, 0.0))
    a._floors.append(a._new_floor())
    a._explore(DEPTH, np.zeros(2), 0.0)
    for _ in range(40):
        a._get_close_to_stair(np.zeros(2), 0.0)
    assert (5.0, 0.0) in om._disabled_frontiers and a.stats["climb_fail"] == 1
    # F3: the disable never clears `_has_up_stair`, and the same-step `_explore`
    # finds no frontier and an unexplored floor above, so the reference walks
    # straight back onto the same staircase.
    assert a.stairs.climbing and a.stats["climb_attempt"] == 2


def test_a_network_stop_on_the_stair_approach_retires_the_frontier():
    """`:1062-1065`."""
    a = _agent(driver=_Driver(action=None))
    om = _paint_stairs(a, xy=(5.0, 0.0))
    a._floors.append(a._new_floor())
    a.stairs.start_navigating(om, 1)
    action = a._get_close_to_stair(np.zeros(2), 0.0)
    assert (5.0, 0.0) in om._disabled_frontiers and a.stats["climb_fail"] == 1
    # ... and the F3 retry re-enters at once; on the re-entry the stub's STOP
    # is returned raw (`:847`).
    assert a.stairs.climbing and action == "stop"


def test_a_network_stop_on_the_flight_forces_forward_instead():
    """`:1136-1139`: a STOP on a staircase means the treads fill the view."""
    a = _agent(driver=_Driver(action=None))
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    a.stairs.start_navigating(om, 1)
    a.stairs.reach_stair = True
    a.stairs.reach_stair_centroid = True
    assert a._climb_stair(DEPTH, np.array([2.0, 0.0]), 0.0, _px(a, (2.0, 0.0))) == "move_forward"


def test_a_descent_looks_down_twice_before_the_centroid_and_back_up_after():
    """`act` :448-457 and `_climb_stair` :1119-1122, on the hand-tracked pitch."""
    a = _agent()
    om = _paint_stairs(a, direction=2, xy=(2.0, 0.0))
    a._floors.insert(0, a._new_floor()); a._floor_idx = 1
    a.stairs.start_navigating(om, 2)
    a.stairs.reach_stair = True
    assert a._stairs_dispatch(DEPTH, np.array([2.0, 0.0]), 0.0, _px(a, (2.0, 0.0)), None) == "look_down"
    assert a._pitch_angle == -30
    assert a._stairs_dispatch(DEPTH, np.array([2.0, 0.0]), 0.0, _px(a, (2.0, 0.0)), None) == "look_down"
    assert a._pitch_angle == -60
    a.stairs.reach_stair_centroid = True
    assert a._climb_stair(DEPTH, np.array([2.0, 0.0]), 0.0, _px(a, (2.0, 0.0))) == "look_up"
    assert a._pitch_angle == -30


def test_a_paused_flight_reinitialises_on_the_same_floor():
    """F5: the pause >= 30 branch (`act` :459-514) copies the flight to the
    neighbour floor and re-runs the opening scan HERE; the floor index does
    not change."""
    a = _agent()
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a._floors.append(a._new_floor())
    a.stairs.start_navigating(om, 1)
    a.stairs.reach_stair = True
    om._climb_stair_paused_step = 30
    action = a._stairs_dispatch(DEPTH, np.array([2.0, 0.0]), 0.0, _px(a, (2.0, 0.0)), None)
    assert action == "turn_left" and a._floor_idx == 0
    assert not a.stairs.climbing and a._done_initializing is False
    assert a._floors[1]["obstacle"]._has_down_stair


def test_passive_stair_entry_triggers_after_three_steps_on_the_treads():
    """`_detect_passive_stair_entry` (:626-672)."""
    a = _agent()
    om = _paint_stairs(a, xy=(2.0, 0.0))
    for _ in range(2):
        a.stairs.pre_update(a, om, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))
        assert not a.stairs.climbing
    a.stairs.pre_update(a, om, np.array([2.0, 0.0]), _px(a, (2.0, 0.0)))
    assert a.stairs.climbing and a.stairs.reach_stair and a.stats["passive_stair_entry"] == 1


def test_passive_entry_refuses_the_union_mask():
    with pytest.raises(ValueError):
        _agent(stair_up_mode="rednet", passive_stair_entry=True)
    a = _agent(stair_up_mode="rednet", passive_stair_entry=False)
    assert a.stairs.passive_entry is False


def test_upper_half_discriminator_needs_fifty_pixels():
    """`check_stairs_in_upper_50_percent` (`ascent/utils.py:163-183`)."""
    m = np.zeros((100, 100), bool)
    m[10, 10:40] = True                 # 30 px, top half
    assert stairs_in_upper_half(m) is False
    m[11, 10:40] = True                 # 60 px
    assert stairs_in_upper_half(m) is True
    m[:] = False; m[80, :] = True       # bottom half only
    assert stairs_in_upper_half(m) is False


def test_the_downstair_probe_tilts_then_drives_then_retires():
    """`_look_for_downstair` (:851-887)."""
    a = _agent(driver=_Driver(action=None))
    om = a.obstacle_map
    om._look_for_downstair_flag = True
    om._potential_stair_centroid = np.array([[2.0, 0.0]])
    om._down_stair_map[100:105, 100:105] = 1
    om._has_down_stair = True
    assert a._look_for_downstair(np.zeros(2), 0.0) == "look_down" and a._pitch_angle == -30
    # the mover refuses: retire the suspicion and tilt back up
    assert a._look_for_downstair(np.zeros(2), 0.0) == "look_up"
    assert a._pitch_angle == 0 and not om._has_down_stair and not om._look_for_downstair_flag
    assert (2.0, 0.0) in om._disabled_frontiers


def test_the_camera_is_levelled_before_anything_else_when_not_on_stairs():
    """`act` :559-566."""
    a = _agent()
    a._pitch_angle = 30
    f = _wall_frame([0, 0.88, 0], [1, 0.88, 0])
    assert a.act(f) == "look_down" and a._pitch_angle == 0


def _det(score=0.9, box=(0, 0, 60, 60), label="chair"):
    from osg.core.types import Detection
    x0, y0, x1, y1 = box
    m = np.zeros((480, 640), bool)
    m[y0:y1, x0:x1] = True
    return Detection(label=label, score=score, bbox_xyxy=np.array(box, float), mask=m)
# ================================================ where the DROP-OFF is marked
#
# The vendored code mirrored depth about (max+min)/2 and painted the drop-off at
# the mirrored range, so its position was set by whatever was visible THROUGH
# the hole: a lip at 1.0 m and one at 1.5 m both landed at 1.70 m, and a lip at
# 2 m or beyond produced nothing at all. These pin the replacement, which marks
# the place the floor should have been.

def _drop_off(edge_m, look_at=None, hole_range=3.8, size=800):
    """Render the depth image a camera really would see of a floor that stops at
    `edge_m`, run it through the map, and return the painted down-stair xy."""
    from osg.core.types import CameraIntrinsics, FrameData
    from ascentnav.geometry import camera_pitch, normalise_depth, tf_camera_to_episodic
    from ascentnav.mapping.obstacle_map import ObstacleMap
    from .conftest import make_camera

    W, H, cam_h = 640, 480, 0.88
    fx = fy = W / (2 * np.tan(np.radians(79.0) / 2))
    intr = CameraIntrinsics.from_hfov(79.0, W, H)
    T = make_camera([0, cam_h, 0], look_at or [1, cam_h, 0])
    blank = FrameData(frame_id=0, rgb=np.zeros((H, W, 3), np.uint8),
                      depth=np.zeros((H, W), np.float32), T_wc=T, intrinsics=intr)
    tf = tf_camera_to_episodic(blank, cam_h)

    v, u = np.meshgrid(np.arange(H) - H // 2, np.arange(W) - W // 2, indexing="ij")
    dirs = np.stack([np.ones_like(u, float), -u / fx, -v / fy], -1) @ tf[:3, :3].T
    C, dz = tf[:3, 3], dirs[..., 2]
    down = dz < -1e-9
    t = np.where(down, C[2] / np.where(down, -dz, 1.0), 1e9)
    on_floor = down & ((C + t[..., None] * dirs)[..., 0] <= edge_m) & (t < 5.0)
    depth = np.full((H, W), 5.0, np.float32)
    depth[on_floor] = np.clip(t[on_floor], 0.5, 5.0)
    depth[down & ~on_floor] = hole_range

    f = FrameData(frame_id=0, rgb=np.zeros((H, W, 3), np.uint8), depth=depth,
                  T_wc=T, intrinsics=intr)
    om = ObstacleMap(min_height=0.61, max_height=0.88, agent_radius=0.18, size=size)
    om._downstair_detector = "lip"                  # the A/B, not the reference trigger
    om.update_map(normalise_depth(depth, 0.5, 5.0), tf, 0.5, 5.0, fx, fy,
                  np.radians(79.0), {}, np.zeros((H, W), np.uint8),
                  np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8),
                  float(np.degrees(-camera_pitch(f))), True, False, 0)
    px = np.argwhere(om._down_stair_map)
    return om._px_to_xy(px[:, ::-1].astype(float)) if len(px) else np.zeros((0, 2))


@pytest.mark.parametrize("edge", [1.5, 2.0, 2.5, 3.0])
def test_the_drop_off_is_marked_at_the_lip(edge):
    xy = _drop_off(edge)
    assert len(xy) > 0, f"a hole starting at {edge} m was not detected at all"
    assert xy[:, 0].min() == pytest.approx(edge, abs=0.1), (
        f"marking starts at {xy[:, 0].min():.2f} m, the floor stops at {edge} m")


@pytest.mark.parametrize("edge", [1.5, 2.5])
def test_the_marking_does_not_spread_across_the_void(edge):
    """Every ray past the lip also misses the floor, so marking them all fills
    the whole visible void -- a 22 m^2 blob spilling onto the floor below and
    out through whatever the stairwell overlooks. Only the near edge is a place
    the agent can stand."""
    xy = _drop_off(edge)
    depth_of_band = xy[:, 0].max() - xy[:, 0].min()
    assert depth_of_band < 0.5, (
        f"the marked band is {depth_of_band:.2f} m deep; it should hug the lip")


def test_the_marking_tracks_the_lip_rather_than_the_far_surface():
    """The signature of the old bug: two different lips painted in the SAME
    place, because the position came from the range of the lower floor."""
    near, far = _drop_off(1.5)[:, 0].min(), _drop_off(3.0)[:, 0].min()
    assert far - near == pytest.approx(1.5, abs=0.2)


def test_an_unbroken_floor_marks_nothing():
    assert len(_drop_off(99.0)) == 0


def test_the_test_is_pitch_invariant():
    """A tilted camera sees a different image of the same hole and must reach
    the same conclusion -- the old below-ground test did not."""
    level = _drop_off(2.0)[:, 0].min()
    tilted = _drop_off(2.0, look_at=[1, 0.88 - np.tan(np.radians(30)), 0])[:, 0].min()
    assert level == pytest.approx(2.0, abs=0.1)
    assert tilted == pytest.approx(2.0, abs=0.1)
# ============================================================= behaviour log

def test_the_log_attributes_motion_to_the_action_that_caused_it():
    """`moved` is the realised displacement of the PREVIOUS row's action. A
    forward that moved nothing is the single most diagnostic event in the log,
    and the hand-rolled trace it replaces never recorded it."""
    from osg.eval.behaviour_log import BehaviourLog
    b = BehaviourLog()
    b.step(n=0, xy=[0, 0], yaw=0.0, height=0.9, action="move_forward")
    b.step(n=1, xy=[0.25, 0], yaw=0.0, height=0.9, action="move_forward")
    b.step(n=2, xy=[0.25, 0], yaw=0.0, height=0.9, action="turn_left")
    assert b.rows[0]["moved"] == 0.25 and "blocked" not in b.rows[0]
    assert b.rows[1]["moved"] == 0.0 and b.rows[1]["blocked"] == 1
    assert b.summary()["blocked_forwards"] == 1
    assert b.summary()["forwards"] == 2


def test_a_turn_is_not_counted_as_a_blocked_forward():
    from osg.eval.behaviour_log import BehaviourLog
    b = BehaviourLog()
    b.step(n=0, xy=[0, 0], yaw=0.0, height=0.9, action="turn_left")
    b.step(n=1, xy=[0, 0], yaw=float(np.radians(30)), height=0.9, action="turn_left")
    assert "blocked" not in b.rows[0]
    assert b.rows[0]["turned"] == pytest.approx(30.0, abs=0.1)
    assert b.summary()["blocked_forwards"] == 0


def test_absent_and_zero_stay_different():
    """An agent with no stair machinery must leave those keys ABSENT: absent
    and zero mean different things when you are counting opportunities."""
    from osg.eval.behaviour_log import BehaviourLog
    b = BehaviourLog()
    b.step(n=0, xy=[0, 0], yaw=0.0, height=0.9, action="stop", up_px=None, ndet=0)
    assert "up_px" not in b.rows[0] and b.rows[0]["ndet"] == 0


def test_the_log_can_be_switched_off_entirely():
    from osg.eval.behaviour_log import BehaviourLog
    b = BehaviourLog(enabled=False)
    b.step(n=0, xy=[0, 0], yaw=0.0, height=0.9, action="stop")
    assert b.rows == [] and b.summary() == {}


def test_the_agent_feeds_the_log_through_a_real_step():
    a = _agent()
    f = _wall_frame([0, 0.88, 0], [1, 0.88, 0])
    a.act(f)
    row = a.behaviour.rows[-1]
    for key in ("n", "xy", "yaw", "h", "state", "act", "ndet", "explored_m2", "on_stairs"):
        assert key in row, f"{key} missing from the behaviour row"
    assert a.step_trace is a.behaviour.rows, "the wire format must stay the same list"
# ============================================================ BLIP-2 value map

def test_the_blip2_scorer_speaks_ascents_wire_format(monkeypatch):
    """ASCENT's server decodes with cv2 and calls `Image.fromarray`, and its own
    client encodes the RGB array WITHOUT a channel swap
    (`server_wrapper_out.py:59`). Swapping here would feed BLIP-2
    channel-flipped images and still return plausible scores."""
    import base64, json as _json
    import cv2
    from osg.perception.image_text import Blip2ItmScorer

    sent = {}

    class _Resp:
        def __init__(self, body): self._b = body
        def read(self): return self._b
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        sent["body"] = _json.loads(req.data)
        return _Resp(_json.dumps({"response": 0.42}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[..., 0] = 255                       # pure RED in RGB terms
    out = Blip2ItmScorer().score(rgb, ["a sofa"])
    assert out[0] == pytest.approx(0.42)
    assert sent["body"]["method"] == "cosine" and sent["body"]["txt"] == "a sofa"
    back = cv2.imdecode(np.frombuffer(base64.b64decode(sent["body"]["image"]), np.uint8),
                        cv2.IMREAD_ANYCOLOR)
    assert back[..., 0].mean() > 200, "the array must round-trip unswapped"


def test_an_unreachable_value_model_is_visible_not_silent(monkeypatch, capsys):
    """A value map stuck at one value ranks nothing. That must show up as
    errors rather than quietly reshaping exploration."""
    from osg.perception.image_text import Blip2ItmScorer

    def boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    s = Blip2ItmScorer()
    out = s.score(np.zeros((8, 8, 3), np.uint8), ["a sofa", "a bed"])
    assert list(out) == [0.0, 0.0] and s.n_errors == 2 and s.n_calls == 0
    assert "unreachable" in capsys.readouterr().out


def test_the_factory_dispatches_on_value_model():
    from osg.core.config import OSGConfig
    from osg.perception.image_text import Blip2ItmScorer, build_image_text_scorer
    cfg = OSGConfig()
    cfg.exploration.value_map = True
    cfg.exploration.value_model = "blip2itm"
    assert isinstance(build_image_text_scorer(cfg), Blip2ItmScorer)
