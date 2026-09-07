"""ASCENT's stair machinery in ascentnav.

The first ascentnav run scored 0.0% on all 21 cross-floor episodes because the
ObstacleMap was handed zeros for both stair masks -- everything downstream
existed and never fired. These pin the pieces that make it fire.
"""
from __future__ import annotations

import numpy as np
import pytest

from ascentnav.stairs import (
    CLIMB_PAUSED_ABANDON,
    ClimbState,
    carrot_waypoint,
    ratchet_carrot,
    robot_on_stairs,
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


# ----------------------------------------------------------------- the state

def test_stuck_on_approach_resets_when_the_distance_closes():
    c = ClimbState(); c.start(1)
    for i in range(50):                      # closing 0.5 m each call
        assert c.stuck_on_approach(20.0 - 0.5 * i) is False


def test_stuck_on_approach_fires_when_it_does_not():
    c = ClimbState(); c.start(1)
    fired = any(c.stuck_on_approach(5.0) for _ in range(40))
    assert fired, "30 steps without closing 0.3 m must retire the staircase"


def test_a_long_but_progressing_approach_is_never_retired():
    """The budget is on stalling, not on walking (`ascent_policy.py:963-981`):
    both counters advance only on a step that failed to close 0.3 m, so a
    staircase 30 m away stays a valid target the whole way."""
    c = ClimbState(); c.start(1)
    d = 30.0
    for _ in range(90):
        d -= 0.31
        assert c.stuck_on_approach(d) is False


def test_the_total_budget_catches_an_intermittent_staller():
    """60 stalled steps retire it even when the agent creeps often enough to
    keep resetting the 30-step consecutive counter."""
    c = ClimbState(); c.start(1)
    d, fired = 30.0, False
    for i in range(400):
        if i % 20 == 19:
            d -= 0.31                        # one real step of progress
        if c.stuck_on_approach(d):
            fired = True
            break
    assert fired and c.stick_steps < 30


def test_start_clears_previous_state():
    c = ClimbState(); c.start(1)
    c.reached = c.reached_centroid = True; c.get_close_steps = 40
    c.start(2)
    assert c.direction == 2 and not c.reached and not c.reached_centroid
    assert c.get_close_steps == 0


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


# ============================================================ the state machine
#
# These drive `AscentNavAgent`'s climb branch directly. The agent is built with
# a stub detector and a scripted mover, so no weights are touched.

import numpy as np  # noqa: E402  (kept beside the agent tests for readability)

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
    for k, v in over.items():
        setattr(cfg.agent, k, v)
    a = AscentNavAgent(cfg, StubDetector(), AsyncScorer(_StubScorer()),
                       None, "chair", pointnav=driver or _Driver())
    a.reset("chair")
    return a


def _paint_stairs(agent, direction=1, xy=(2.0, 0.0), r=6):
    """Put a staircase on the map where the ObstacleMap would have put one."""
    om = agent.obstacle_map
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


def test_an_exhausted_floor_with_a_staircase_starts_climbing():
    a = _agent()
    _paint_stairs(a)
    action = a._explore(np.zeros(2), 0.0, DEPTH, 0.0)
    assert a.climb.climbing and a.climb.direction == 1
    assert a._state == "climb"
    # and it starts DRIVING on the same step rather than turning
    assert action == "move_forward"


def test_a_floor_with_frontiers_does_not_climb():
    """The trigger is 'no frontiers left', not 'stairs are visible'
    (`ascent_policy.py:648-680`)."""
    a = _agent()
    _paint_stairs(a)
    a.obstacle_map.frontiers = np.array([[5.0, 5.0]])
    a._explore(np.zeros(2), 0.0, DEPTH, 0.0)
    assert not a.climb.climbing


def test_a_direction_already_climbed_is_not_retried():
    a = _agent()
    om = _paint_stairs(a)
    om._explored_up_stair = True
    assert a._maybe_start_climb() is False


def test_the_approach_runs_until_the_footprint_touches_the_stairs():
    a = _agent()
    _paint_stairs(a, xy=(2.0, 0.0))
    a.climb.start(1)
    a._do_climb(DEPTH, np.array([-3.0, 0.0]), 0.0, 0.0)
    assert not a.climb.reached, "still 5 m away"
    a._do_climb(DEPTH, np.array([2.0, 0.0]), 0.0, 0.0)
    assert a.climb.reached


def test_a_network_stop_on_the_approach_retires_the_staircase():
    """`ascent_policy.py:1011-1015` -- the one place a STOP is authoritative."""
    a = _agent(driver=_Driver(reason="policy_stop", action=None))
    om = _paint_stairs(a)
    a.climb.start(1)
    a._do_climb(DEPTH, np.array([-3.0, 0.0]), 0.0, 0.0)
    assert not a.climb.climbing
    assert om._has_up_stair is False and om._up_stair_map.sum() == 0
    assert a.stats["climb_fail"] == 1


def test_a_network_stop_on_the_flight_forces_forward_instead():
    a = _agent(driver=_Driver(reason="policy_stop", action=None))
    _paint_stairs(a)
    a.climb.start(1)
    a.climb.reached = a.climb.reached_centroid = True
    assert a._do_climb(DEPTH, np.array([2.0, 0.0]), 0.0, 0.0) == "move_forward"
    assert a.climb.climbing, "a STOP mid-flight must not end the climb"
    assert a.stats["climb_forced_forward"] == 1


def test_leaving_the_stairs_after_the_centroid_completes_the_transition():
    a = _agent()
    _paint_stairs(a, xy=(2.0, 0.0))
    a.climb.start(1)
    a.climb.reached = a.climb.reached_centroid = True
    floors_before = len(a._floors)
    a._do_climb(DEPTH, np.array([9.0, 9.0]), 0.0, 0.0)      # far off the stair map
    assert not a.climb.climbing
    assert a._floor_idx == 1 and len(a._floors) == floors_before + 1
    assert a.stats["climb_ok"] == 1


def test_arriving_upstairs_does_not_bounce_straight_back_down():
    """The staircase is carried onto the new floor already marked explored
    (`map_controller.py:563-596`), which is what stops the up-down-up loop."""
    a = _agent()
    _paint_stairs(a, xy=(2.0, 0.0))
    a.climb.start(1)
    a.climb.reached = a.climb.reached_centroid = True
    a._do_climb(DEPTH, np.array([9.0, 9.0]), 0.0, 0.0)
    om = a.obstacle_map
    assert om._has_down_stair and om._explored_down_stair
    assert om._down_stair_map.sum() > 0
    assert a._maybe_start_climb() is False


def test_a_new_floor_is_rescanned_but_a_revisited_one_is_not():
    a = _agent(initial_scan=True)
    assert a._switch_floor(1) is True and a._init_left > 0
    a._init_left = 0
    assert a._switch_floor(2) is False, "floor 0 already exists"
    assert a._init_left == 0


def test_a_wedged_climb_is_abandoned_not_ridden_out():
    a = _agent()
    om = _paint_stairs(a, xy=(2.0, 0.0))
    a.climb.start(1)
    a.climb.reached = True
    for _ in range(CLIMB_PAUSED_ABANDON + 2):        # never moves
        a._do_climb(DEPTH, np.array([2.0, 0.0]), 0.0, 0.0)
        if not a.climb.climbing:
            break
    assert not a.climb.climbing and a.stats["climb_fail"] == 1
    assert om._disabled_stair_map.sum() > 0, "the bad flight is remembered"


def test_a_descent_tilts_down_before_the_centroid_and_back_up_after():
    a = _agent()
    _paint_stairs(a, direction=2, xy=(2.0, 0.0))
    a.climb.start(2)
    a.climb.reached = True
    assert a._do_climb(DEPTH, np.array([2.0, 0.0]), 0.0, pitch_deg=0.0) == "look_down"
    assert a._do_climb(DEPTH, np.array([2.0, 0.0]), 0.0, pitch_deg=-30.0) == "look_down"
    a.climb.reached_centroid = True
    assert a._do_climb(DEPTH, np.array([2.0, 0.0]), 0.0, pitch_deg=-60.0) == "look_up"


def test_an_ascent_never_tilts():
    """S14a measured tilting as no help going up, and a tilted frame feeds the
    map geometry it has to correct for."""
    a = _agent()
    _paint_stairs(a, xy=(2.0, 0.0))
    a.climb.start(1)
    a.climb.reached = True
    for pitch in (0.0, -30.0, 30.0):
        assert a._do_climb(DEPTH, np.array([2.0, 0.0]), 0.0, pitch) not in ("look_up", "look_down")


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
    a._switch_floor(1)
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


# ================================================== direction, and the probe
#
# The strict descent split ran 148 climb steps and every one was an ascent, on
# episodes whose goal is below the start. These pin the three pieces that fix
# that: the up/down discriminator, the tie-break that uses it, and the probe.

def test_upper_half_discriminator():
    from ascentnav.stairs import stairs_in_upper_half
    m = np.zeros((100, 100), bool)
    m[10:40] = True
    assert stairs_in_upper_half(m) is True          # treads above the horizon
    m2 = np.zeros((100, 100), bool)
    m2[60:90] = True
    assert stairs_in_upper_half(m2) is False        # a flight going down
    assert stairs_in_upper_half(None) is False
    assert stairs_in_upper_half(np.zeros((10, 10), bool)) is False


def test_down_is_preferred_when_the_treads_are_below_the_horizon():
    a = _agent()
    _paint_stairs(a, direction=1, xy=(2.0, 0.0))
    _paint_stairs(a, direction=2, xy=(-2.0, 0.0))
    a._seg_upper = False                            # stairs seen low in frame
    assert a._maybe_start_climb() is True
    assert a.climb.direction == 2


def test_up_still_wins_when_the_treads_are_above_the_horizon():
    a = _agent()
    _paint_stairs(a, direction=1, xy=(2.0, 0.0))
    _paint_stairs(a, direction=2, xy=(-2.0, 0.0))
    a._seg_upper = True
    assert a._maybe_start_climb() is True
    assert a.climb.direction == 1


def test_one_available_direction_is_taken_regardless_of_the_discriminator():
    a = _agent()
    _paint_stairs(a, direction=2, xy=(-2.0, 0.0))
    a._seg_upper = True                             # would prefer up, but there is none
    assert a._maybe_start_climb() is True
    assert a.climb.direction == 2


def test_the_downstair_probe_tilts_before_it_drives():
    a = _agent()
    a.obstacle_map._potential_stair_centroid = np.array([[3.0, 0.0]])
    assert a._look_for_downstair(np.zeros(2), 0.0, pitch_deg=0.0) == "look_down"
    assert a.stats["down_look"] == 1


def test_a_network_stop_on_the_probe_retires_the_suspicion():
    """`ascent_policy.py:632-643` -- the mover refusing to go is the evidence
    that there was no staircase there."""
    a = _agent(driver=_Driver(reason="policy_stop", action=None))
    om = a.obstacle_map
    om._potential_stair_centroid = np.array([[3.0, 0.0]])
    om._down_stair_map[100:110, 100:110] = 1
    om._has_down_stair = True
    om._look_for_downstair_flag = True
    assert a._look_for_downstair(np.zeros(2), 0.0, pitch_deg=-30.0) == "look_up"
    assert om._has_down_stair is False and om._down_stair_map.sum() == 0
    assert om._disabled_stair_map.sum() > 0 and om._look_for_downstair_flag is False


def test_standing_on_the_candidate_also_retires_it():
    a = _agent()
    a.obstacle_map._potential_stair_centroid = np.array([[0.1, 0.0]])
    a.obstacle_map._look_for_downstair_flag = True
    assert a._look_for_downstair(np.zeros(2), 0.0, pitch_deg=-30.0) == "look_up"
    assert a.stats["downstair_reject"] == 1


def test_the_camera_is_returned_to_level_when_not_on_stairs():
    """A tilt left standing would relabel every later staircase, because the map
    routes stair pixels by the SIGN of the pitch."""
    a = _agent()
    assert a._level_camera(-30.0) == "look_up"
    assert a._level_camera(30.0) == "look_down"
    assert a._level_camera(0.0) is None


def test_levelling_never_fights_the_climb_or_the_probe():
    a = _agent()
    a.climb.start(2)
    assert a._level_camera(-60.0) is None
    a.climb.reset()
    a.obstacle_map._look_for_downstair_flag = True
    assert a._level_camera(-60.0) is None
