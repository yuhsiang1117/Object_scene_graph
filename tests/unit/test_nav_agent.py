"""Unit tests for NavAgent.

Two areas covered so far:
1. The terminal APPROACH phase (P1a) — replaced three earlier distance-based
   stopping strategies, all of which stalled at dtg 0.107-0.147 m; see
   docs/DESIGN_AND_ROADMAP.md P0->P1 history. Tests call `_do_approach`
   directly rather than driving the full `act()` loop: its decision logic
   only depends on the detector, the costmap/planner, and a handful of
   `_approach_*` fields, so exercising it in isolation keeps this hermetic
   (no Hydra/habitat) and fast.
2. The frontier give-up progress timer (P1b) — an exploration round must
   reset the strategy's progress reference to the current step/pose
   whenever it starts pursuing a (new or re-selected) frontier; otherwise a
   stale reference from a prior pursuit can trigger a spurious give-up
   within the first step of the new one.

Uses a lightweight SimpleNamespace in place of a real Hydra config: only
the fields NavAgent.__init__/reset()/the methods under test actually read
are set.
"""
from __future__ import annotations

import numpy as np

from osg.agent.nav_agent import STOP_ACTION, NavAgent, State
from osg.core.config import OSGConfig
from osg.core.types import CameraIntrinsics, Detection
from osg.exploration.async_scorer import AsyncScorer
from osg.exploration.scorer import FrontierScorer
from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN
from osg.mapping.frontier import Frontier
from osg.perception.detector import StubDetector

from .conftest import make_camera, make_frame


class _StubScorer(FrontierScorer):
    """Deterministic no-LLM scorer for tests (all frontiers equally promising)."""

    def score(self, frontiers, sg, target, keyframes=None):
        return {f.id: 1.0 for f in frontiers}

APPROACH_BBOX_THRESHOLD = 40_000.0
_INTRINSICS = CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0, width=640, height=480)


def make_cfg(**agent_overrides):
    """The SHIPPED config, plus the handful of values these tests deliberately
    differ on.

    This used to be a forty-field `SimpleNamespace` mirroring whichever fields
    NavAgent happened to read. Two things went wrong with that. It had to be
    extended by hand every time the agent read a new field, and it silently
    drifted: `search_surface_mass` sat at 0.5 here against 1.0 shipped,
    `detector_absence_recall` at 0.5 against 0.8, `room_door_width_m` at 1.2
    against 2.0 -- so the tests were asserting on constants nothing runs with.

    `OSGConfig()` needs no Hydra and no yaml. Anything below is a test
    condition, stated in one visible line, and anything NOT below is exercised
    at the value the benchmark actually uses.
    """
    cfg = OSGConfig()
    # No 360-degree scan: these tests drive one state at a time and a twelve
    # step spin at the top of every episode is not the thing under test.
    cfg.agent.initial_scan = False
    # Hand-built costmaps here are filled with FREE and a few OCCUPIED cells;
    # the obstacle band that produced them is not part of any assertion.
    cfg.mapping.obstacle_low_m = 0.1
    # Candidate tests build tracks from one or two synthetic detections, which
    # cannot clear an evidence bar meant for real multi-view accumulation.
    cfg.verification.min_evidence = 0.0
    cfg.detector.vocabulary = ["chair", "bed"]
    for k, v in agent_overrides.items():
        setattr(cfg.agent, k, v)
    return cfg


def _frame_at(xy) -> "object":
    """A frame whose camera sits at this ground-plane point."""
    T = np.eye(4)
    T[0, 3], T[2, 3] = float(xy[0]), float(xy[1])
    return make_frame(_INTRINSICS, T)


def make_agent(cfg=None, target="chair") -> NavAgent:
    agent = NavAgent(cfg or make_cfg(), StubDetector(), AsyncScorer(_StubScorer()), None, target)
    agent.costmap.grid[:, :] = FREE
    return agent


def _det(label: str, bbox_wh: tuple, score: float = 0.8) -> Detection:
    w, h = bbox_wh
    return Detection(
        label=label, score=score,
        bbox_xyxy=np.array([0.0, 0.0, float(w), float(h)]),
        mask=np.zeros((480, 640), dtype=bool),
    )


def _frame(xy, frame_id=0):
    """xy is a ground-plane (x, z) position. depth_value is set beyond
    mapping.max_range_m (5.0) so a real `_act_inner` call's costmap.update()
    is a no-op on any grid a test carved by hand ahead of time."""
    T = make_camera([xy[0], 0.88, xy[1]], [xy[0] + 1.0, 0.88, xy[1]])
    return make_frame(_INTRINSICS, T, depth_value=100.0, frame_id=frame_id)


def test_bbox_fallback_stop_when_no_valid_depth():
    # Empty mask (_det uses np.zeros) -> no valid depth -> bbox fallback fires.
    agent = make_agent()
    agent.state = State.APPROACH
    agent._goal_xy = np.array([5.0, 0.0])
    agent.approach.steps_left = 5
    agent.detector.push([_det("chair", (300, 300))])  # area 90000 > 40000 threshold

    action = agent.approach.step(_frame([0.0, 0.0]))

    assert action == STOP_ACTION
    assert agent.state == State.DONE
    assert agent.approach.stop_reason == "bbox"
    assert agent.approach.bbox_log == [(0, 90000.0, None)]  # (step, bbox_px, depth)


def test_stops_when_within_depth_range():
    # Primary terminal: target visible AND within approach_stop_depth_m (1.0 m).
    agent = make_agent()
    agent.state = State.APPROACH
    agent._goal_xy = np.array([5.0, 0.0])
    agent.approach.steps_left = 5
    det = _det("chair", (50, 50))  # small bbox -- would NOT trip the bbox threshold
    det.mask[100:200, 100:200] = True  # populate mask so depth is sampled
    agent.detector.push([det])
    frame = make_frame(_INTRINSICS, make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0]),
                       depth_value=0.8)  # target 0.8 m away <= 1.0 m

    action = agent.approach.step(frame)

    assert action == STOP_ACTION
    assert agent.state == State.DONE
    assert agent.approach.stop_reason == "depth"


def test_advances_when_visible_but_too_far_by_depth():
    # Visible but beyond stop range -> keep advancing (bbox is large but depth rules).
    agent = make_agent()
    agent.state = State.APPROACH
    agent._goal_xy = np.array([5.0, 0.0])
    agent.approach.steps_left = 5
    det = _det("chair", (300, 300))  # large bbox
    det.mask[100:200, 100:200] = True
    agent.detector.push([det])
    frame = make_frame(_INTRINSICS, make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0]),
                       depth_value=3.0)  # 3 m > 1.0 m stop range

    action = agent.approach.step(frame)

    assert action != STOP_ACTION
    assert agent.state == State.APPROACH


def test_advances_when_visible_but_small():
    agent = make_agent()
    agent.state = State.APPROACH
    agent._goal_xy = np.array([5.0, 0.0])
    agent.approach.steps_left = 5
    agent.detector.push([_det("chair", (50, 50))])  # area 2500 < threshold

    action = agent.approach.step(_frame([0.0, 0.0]))

    assert action != STOP_ACTION
    assert agent.state == State.APPROACH
    assert agent.approach.steps_left == 4
    assert agent.approach.last_good_xy is not None
    assert np.allclose(agent.approach.last_good_xy, [0.0, 0.0])


def test_ignores_detections_of_other_labels():
    agent = make_agent(target="chair")
    agent.state = State.APPROACH
    agent._goal_xy = np.array([5.0, 0.0])
    agent.approach.steps_left = 5
    agent.detector.push([_det("sofa", (500, 500))])  # huge, but wrong label

    action = agent.approach.step(_frame([0.0, 0.0]))

    assert action != STOP_ACTION
    assert agent.state == State.APPROACH  # not fooled into stopping
    assert agent.approach.last_good_xy is None  # never confirmed visible


def test_retreats_when_visibility_lost():
    agent = make_agent()
    agent.state = State.APPROACH
    agent._goal_xy = np.array([5.0, 0.0])
    agent.approach.steps_left = 5
    agent.approach.last_good_xy = np.array([0.0, 0.0])  # a previously-visible pose
    agent.detector.push([])  # nothing visible from the current pose

    steps_before = agent.approach.steps_left
    action = agent.approach.step(_frame([1.0, 0.0]))  # 1 m away from the good pose

    assert action != STOP_ACTION  # heads back toward the good pose
    assert agent.state == State.APPROACH
    assert agent.approach.steps_left == steps_before  # retreat doesn't spend the advance budget
    assert agent.approach.stop_reason is None  # hasn't stopped yet


def test_falls_through_to_advance_when_never_visible():
    agent = make_agent()
    agent.state = State.APPROACH
    agent._goal_xy = np.array([5.0, 0.0])
    agent.approach.steps_left = 5
    agent.approach.last_good_xy = None  # never seen it
    agent.detector.push([])

    action = agent.approach.step(_frame([0.0, 0.0]))

    assert action != STOP_ACTION
    assert agent.state == State.APPROACH
    assert agent.approach.steps_left == 4  # advance budget spent (no retreat target)
    assert agent.approach.last_good_xy is None


def test_stops_at_step_budget():
    agent = make_agent()
    agent.state = State.APPROACH
    agent._goal_xy = np.array([5.0, 0.0])
    agent.approach.steps_left = 0  # budget exhausted
    agent.detector.push([_det("chair", (50, 50))])  # visible but small: would normally advance

    action = agent.approach.step(_frame([0.0, 0.0]))

    assert action == STOP_ACTION
    assert agent.state == State.DONE
    assert agent.approach.stop_reason == "deadline"


def test_stops_at_deadline():
    agent = make_agent()
    agent.state = State.APPROACH
    agent._goal_xy = np.array([5.0, 0.0])
    agent.approach.steps_left = 5
    agent._goto_deadline = agent.step_count - 1  # already past
    agent.detector.push([_det("chair", (50, 50))])

    action = agent.approach.step(_frame([0.0, 0.0]))

    assert action == STOP_ACTION
    assert agent.state == State.DONE
    assert agent.approach.stop_reason == "deadline"


def test_navigates_toward_vicinity_when_goal_cell_blocked():
    """GVG Voronoi navigation (ported from old) approaches a goal whose exact
    cell is blocked to its reachable medial-axis vicinity (a node within
    goal_near) instead of declaring the goal unreachable -- so it returns a
    navigation action; the stop then comes from the distance / step-budget
    rules, exercised by the other approach tests."""
    agent = make_agent()
    goal = np.array([5.0, 0.0])
    rc = agent.costmap.world_to_grid(goal)
    agent.costmap.grid[rc[0] - 5 : rc[0] + 6, rc[1] - 5 : rc[1] + 6] = OCCUPIED
    agent.state = State.APPROACH
    agent._goal_xy = goal
    agent.approach.steps_left = 5
    agent.detector.push([])

    action = agent.approach.step(_frame([0.0, 0.0]))
    assert action in ("move_forward", "turn_left", "turn_right")  # navigating, not crashed


def _carve_free_square(agent, half_width_cells=20):
    """UNKNOWN everywhere except a FREE square centered on the grid origin,
    so FrontierExtractor finds real frontiers along its border."""
    agent.costmap.grid[:, :] = UNKNOWN
    h, w = agent.costmap.grid.shape
    cy, cx = h // 2, w // 2
    agent.costmap.grid[
        cy - half_width_cells : cy + half_width_cells,
        cx - half_width_cells : cx + half_width_cells,
    ] = FREE


def test_progress_ref_resets_on_new_frontier_selection():
    """P1b regression: a stale progress-reference (left over from whatever
    pursuit preceded this selection) must not survive into a freshly
    started one — otherwise the 15-step give-up window can already be
    "expired" on step one of the new pursuit, judged against a position
    from an unrelated earlier pursuit."""
    agent = make_agent()
    _carve_free_square(agent)
    agent.state = State.EXPLORE
    agent.step_count = 50
    agent.exploration._last_select_step = -100  # bypass the 5-step selection throttle
    # Stale reference from a much earlier, unrelated pursuit.
    agent.exploration.progress_ref_step = 10
    agent.exploration.progress_ref_xy = np.array([-999.0, -999.0])

    agent._explore(_frame([0.0, 0.0], frame_id=agent.step_count))

    assert agent.state == State.GOTO_FRONTIER
    assert agent.exploration.progress_ref_step == agent.step_count
    assert np.allclose(agent.exploration.progress_ref_xy, [0.0, 0.0])


def test_giveup_logged_with_frontier_and_agent_position():
    agent = make_agent()
    _carve_free_square(agent)
    agent.state = State.GOTO_FRONTIER
    # A real Frontier, not a namespace: the blacklist is floor-scoped, so a
    # stub missing `.floor` diverges from anything production ever produces.
    agent.exploration.current_frontier = Frontier(
        id=0, centroid_xy=np.array([3.0, 4.0]), cells=np.zeros((0, 2), dtype=int), size=0
    )
    agent.step_count = 100
    agent.exploration.progress_ref_step = 80  # 20 steps ago, past the 15-step check window
    agent.exploration.progress_ref_xy = np.array([0.0, 0.0])  # same as current -> "no progress"
    agent._goal_xy = np.array([3.0, 4.0])
    agent._current_path = None

    agent._act_inner(_frame([0.0, 0.0], frame_id=agent.step_count))

    assert agent.stats["frontier_give_up"] == 1
    assert len(agent.giveup_log) == 1
    step, frontier_xy, agent_xy = agent.giveup_log[0]
    assert step == 100
    assert frontier_xy == [3.0, 4.0]
    assert agent_xy == [0.0, 0.0]


def test_follow_to_threads_configured_tolerances_to_planner_and_controller():
    """P1f: `_follow_to` (APPROACH-only) must pass agent.approach_goal_
    tolerance_m / approach_arrival_tol_m through to the planner and
    controller, not fall back to their loose frontier/verify-view
    defaults -- confirmed via spies rather than re-deriving the tolerance
    math already covered in test_planner.py / test_controller.py."""
    cfg = make_cfg(approach_goal_tolerance_m=0.12, approach_arrival_tol_m=0.1)
    agent = make_agent(cfg)
    _carve_free_square(agent)

    plan_calls = []
    orig_plan = agent.planner.plan

    def spy_plan(costmap, start_xy, goal_xy, goal_tolerance_m=None):
        plan_calls.append(goal_tolerance_m)
        return orig_plan(costmap, start_xy, goal_xy, goal_tolerance_m)

    act_calls = []
    orig_act = agent.controller.act

    def spy_act(T_wc, path, arrival_tol_m=0.2):
        act_calls.append(arrival_tol_m)
        return orig_act(T_wc, path, arrival_tol_m)

    agent.planner.plan = spy_plan
    agent.controller.act = spy_act

    agent.approach.follow_to(_frame([0.0, 0.0]), np.array([1.0, 0.0]))

    assert plan_calls == [0.12]
    assert act_calls == [0.1]


# --------------------------------------------------------- cross-floor goals


def test_portal_goal_keeps_its_target_floor_height():
    """A portal pursuit runs in GOTO_FRONTIER but targets another storey.

    Regression: `_follow_path` used to decide the snap height from the STATE
    ("frontier goals are on my floor"), which silently discarded the portal's
    target height and snapped its (x, z) onto the floor below it. The agent
    walked to a point under the mezzanine, arrived, never gained height, and
    the pursuit died as `no_vertical_progress` -- 26 of 35 endings on full v1.
    """
    calls = []

    def nav_fn(goal, floor_y=None):
        calls.append(floor_y)
        return "move_forward"

    cfg = make_cfg()
    cfg.agent.use_habitat_navmesh = True
    agent = NavAgent(cfg, StubDetector(), AsyncScorer(_StubScorer()), None, "chair",
                     nav_fn=nav_fn)
    agent.costmap.grid[:, :] = FREE

    # An ordinary frontier goal: own floor, default height.
    agent.state = State.GOTO_FRONTIER
    agent._goal_xy = np.array([2.0, 0.0])
    agent._goal_floor_y_cache = 2.8
    agent.floors.pursuing = False
    agent._follow_path(_frame((0.0, 0.0)))
    assert calls[-1] is None, "a same-floor frontier goal must not carry a height"

    # The same state, but pursuing a portal one storey up.
    agent.floors.pursuing = True
    agent._follow_path(_frame((0.0, 0.0)))
    assert calls[-1] == 2.8, "portal pursuit lost its target floor height"


def test_absence_does_not_erase_where_the_object_used_to_be():
    """Confirming the object is not at its old POSE does not refute "objects are
    moved short distances" -- it refutes one surface.

    `ExplorationStrategy._last_known_target_xy` used to return None once absence was confirmed, which
    flattened the search prior over every mapped surface in the house. With the
    proximity term unclipped the same evidence is better spent the other way:
    the object is usually a metre or two away, on a neighbouring surface, and
    the surface just ruled out is retired by the InspectionLog instead. Measured
    over 114 relocations, keeping the term takes the true destination into the
    top 5 in 29 of 57 in_anchor cases against 5 of 57 with it dropped.
    """
    from osg.objects.association import ObjectTrack
    from osg.objects.ellipsoid import Ellipsoid

    agent = make_agent(target="chair")
    track = ObjectTrack(
        id=7, label="chair",
        ellipsoid=Ellipsoid(center=np.array([2.0, 0.5, 3.0]),
                            axes=np.array([0.2, 0.2, 0.2]),
                            R=np.eye(3)),
    )
    agent.object_layer._tracks[track.id] = track

    # The agent goes there and the target is not there: the belief drops to the
    # floor and the identity channel records a rejection. Neither is allowed to
    # erase the fact that this is where the object was last seen.
    track.presence.log_odds = -6.0
    track.identity_rejections = 2

    where = agent.exploration._last_known_target_xy(agent._world(_frame_at(np.zeros(2))))
    assert where is not None, "absence must lower a belief, not delete the memory"
    assert np.allclose(where, np.array([2.0, 3.0]))


def test_a_frontier_the_agent_reached_is_not_called_unreachable():
    """`_frontier_reach_m` has to be derived from the planner, not picked.

    HybridVoronoi navigates the medial axis and stops at the graph node nearest
    the goal, within `voronoi_goal_near_m`; WaypointController then reports
    arrival within its own tolerance of that endpoint. So an ordinary, correct
    arrival can leave the agent `goal_near_m + arrival_tol` from the frontier
    goal. Against the old fixed 0.5 m that was classified as a degenerate stub:
    the frontier was blocked for 100 rounds and `_last_giveup_pt` was set --
    which also suppresses the all-frontiers-blocked fallback near that point --
    as the consequence of having got there.
    """
    from osg.exploration.strategy import FRONTIER_ARRIVAL_TOL_M

    cfg = make_cfg()
    agent = make_agent(cfg)
    worst_case_arrival = cfg.exploration.voronoi_goal_near_m + FRONTIER_ARRIVAL_TOL_M
    assert agent.exploration.frontier_reach_m > worst_case_arrival, (
        f"a correct arrival can leave the agent {worst_case_arrival} m from the "
        f"goal, but anything past {agent.exploration.frontier_reach_m} m is called a stub"
    )


def test_the_reach_threshold_follows_the_planner_it_is_derived_from():
    """Change the planner's stopping radius and the classification follows."""
    cfg = make_cfg()
    cfg.exploration.voronoi_goal_near_m = 1.5
    agent = make_agent(cfg)
    assert agent.exploration.frontier_reach_m > 1.5 + 0.2


def test_a_pursued_frontier_is_retired_however_the_pursuit_ended():
    """Ending a pursuit must make the frontier unselectable, arrival or not.

    The navigator returns None for arrived-or-unreachable alike and the FSM then
    drops GOTO_FRONTIER -> EXPLORE. If nothing blocks the frontier the next
    selection can pick the same one, and the agent freezes re-selecting it --
    give-up cannot save it, because it counts steps *inside* GOTO_FRONTIER and
    the re-entry resets the timer. Blocking used to be conditional on being far
    from the goal, which quietly did this job too; making the reach threshold
    correct removed the block for arrivals between 0.5 and 0.9 m and the
    livelock returned, at a cost of 0.073 SR over 96 episodes.
    """
    agent = make_agent()
    frame = _frame_at(np.zeros(2))
    f = Frontier(id=7, cells=np.zeros((0, 2), dtype=int),
                 centroid_xy=np.array([0.3, 0.0]), size=10)
    agent.exploration.current_frontier = f
    agent.state = State.GOTO_FRONTIER

    # An ordinary arrival: well inside the reach threshold.
    agent.exploration.retire_pursued(agent._world(frame), np.array([0.3, 0.0]))
    assert agent.exploration._blocked_pts, "an arrived-at frontier must still be retired"
    assert 7 in agent.exploration._blocked_ids([f], agent.step_count), "and must not be selectable again"
    assert agent.stats.get("frontier_reached") == 1
    assert agent.stats.get("frontier_stub_block", 0) == 0
    assert agent.exploration._last_giveup_pt is None, (
        "an ordinary arrival is not a give-up point -- marking it suppresses the "
        "all-frontiers-blocked fallback near somewhere already explored"
    )


def test_an_unreachable_frontier_is_retired_for_longer_and_marks_a_giveup():
    agent = make_agent()
    frame = _frame_at(np.zeros(2))
    f = Frontier(id=9, cells=np.zeros((0, 2), dtype=int),
                 centroid_xy=np.array([6.0, 0.0]), size=10)
    agent.exploration.current_frontier = f
    agent.state = State.GOTO_FRONTIER

    agent.exploration.retire_pursued(agent._world(frame), np.array([6.0, 0.0]))
    assert 9 in agent.exploration._blocked_ids([f], agent.step_count)
    assert agent.stats.get("frontier_stub_block") == 1
    assert agent.exploration._last_giveup_pt is not None


# ------------------------------- an arrival that is not one (condition S+1)
#
# On the navmesh the follower returns None for arrived AND for unreachable, and
# the approach has treated both as an arrival: it stops. Measured on 00848, the
# agent commits at step 1 to a track 0.81 m from the true object, is told None on
# step 5 while still 6.4 m away, stops, and repeats it for all three attempts --
# episode over at step 78 with 420 steps unspent. Four to six episodes per
# condition end that way and not one of them scores.


def _approach_agent(false_arrival_m, nav_returns=None):
    cfg = make_cfg()
    cfg.agent.use_habitat_navmesh = True
    cfg.agent.approach_false_arrival_m = false_arrival_m
    cfg.verification.absence_on_arrival = False
    agent = NavAgent(cfg, StubDetector(), AsyncScorer(_StubScorer()), None, "chair",
                     nav_fn=lambda goal, floor_y=None: nav_returns)
    agent.costmap.grid[:, :] = FREE
    from osg.objects.association import ObjectTrack
    from osg.objects.ellipsoid import Ellipsoid
    track = ObjectTrack(id=1, label="chair",
                        ellipsoid=Ellipsoid(center=np.array([8.0, 0.5, 0.0]),
                                            axes=np.array([0.2, 0.2, 0.2]), R=np.eye(3)))
    agent.object_layer._tracks[1] = track
    agent._candidate_id = 1
    agent.state = State.APPROACH
    agent._goal_xy = np.array([8.0, 0.0])     # the goal is 8 m away
    agent.approach.steps_left = 50
    agent._goto_deadline = 10 ** 9
    return agent, track


def test_stopping_eight_metres_from_the_goal_is_not_an_arrival():
    agent, track = _approach_agent(1.0)          # follower says None immediately
    action = agent.approach.step(_frame_at(np.zeros(2)))
    assert agent.state == State.EXPLORE, "it must go back to exploring, not stop"
    assert action != STOP_ACTION
    assert agent.stats["approach_false_arrival"] == 1
    assert track.identity_rejections == 1, "could not get there is identity evidence"
    assert not track.blacklisted, "and evidence is not a verdict"


def test_a_genuine_arrival_still_stops():
    agent, _ = _approach_agent(1.0)
    agent._goal_xy = np.array([0.2, 0.0])        # the agent is 0.2 m away
    action = agent.approach.step(_frame_at(np.zeros(2)))
    assert action == STOP_ACTION
    assert agent.state == State.DONE
    assert agent.stats.get("approach_false_arrival", 0) == 0


def test_the_check_is_off_by_default_and_stops_as_before():
    agent, _ = _approach_agent(0.0)
    action = agent.approach.step(_frame_at(np.zeros(2)))
    assert action == STOP_ACTION, "shipped behaviour: it stops eight metres out"
    assert agent.state == State.DONE


# --------------------------------------------------------------------------
# Re-aiming the approach at a refined centre (agent.approach_retarget_m).


def _approach_agent_with_track(centre, **agent_overrides):
    """An agent mid-APPROACH, committed to track 1 at `centre`."""
    from osg.objects.association import ObjectTrack
    from osg.objects.ellipsoid import Ellipsoid

    cfg = make_cfg(approach_to_viewpoint=False, approach_navigable_goal=False,
                   **agent_overrides)
    agent = make_agent(cfg, target="chair")
    track = ObjectTrack(
        id=1, label="chair",
        ellipsoid=Ellipsoid(center=np.asarray(centre, dtype=float),
                            axes=np.array([0.2, 0.2, 0.2]), R=np.eye(3)),
    )
    agent.object_layer._tracks[1] = track
    agent._candidate_id = 1
    agent.approach.start(np.asarray([centre[0], centre[2]], dtype=float),
                         agent_xy=np.zeros(2))
    agent.approach.steps_left = 50
    agent._goto_deadline = 10_000
    return agent, track


def test_a_refined_centre_moves_the_approach_goal():
    """The estimate improves most DURING the walk, which is the one time the
    shipped approach refuses to look at it again.

    Measured on 00848's red plate (in_anchor_02): committed to a centre 0.328 m
    from truth, drove to a viewpoint on that ring, stopped, scored nothing --
    and the same track ended the episode at 0.059 m. Success is scored at
    0.18 m from an authored viewpoint, so that refinement was the episode.
    """
    agent, track = _approach_agent_with_track([3.0, 0.5, 0.0],
                                              approach_retarget_m=0.15)
    before = agent._goal_xy.copy()

    # The ellipsoid refines 0.4 m -- what a few close keyframes buy.
    track.ellipsoid.center = np.array([3.4, 0.5, 0.0])
    agent.approach.step(_frame([0.0, 0.0]))

    assert agent.approach.retargets == 1
    assert not np.allclose(agent._goal_xy, before), "the goal must follow the object"
    assert np.allclose(agent._target_obj_xy, np.array([3.4, 0.0]))
    step, moved, goal_moved = agent.approach.retarget_log[0]
    assert moved == 0.4 and goal_moved > 0.0
    # The path in hand must lead to the NEW goal: `step()` replans within the
    # same call, so the check is where the path goes, not that it is empty.
    assert np.allclose(agent.approach.path_goal, agent._goal_xy)
    assert not np.allclose(agent.approach.path_goal, before)


def test_a_settled_track_never_retargets():
    """Jitter under the tolerance must not re-aim the walk; a goal that moves
    every step is a goal the follower never reaches."""
    agent, track = _approach_agent_with_track([3.0, 0.5, 0.0],
                                              approach_retarget_m=0.15)
    before = agent._goal_xy.copy()

    track.ellipsoid.center = np.array([3.05, 0.5, 0.02])  # 0.054 m of jitter
    agent.approach.step(_frame([0.0, 0.0]))

    assert agent.approach.retargets == 0
    assert np.allclose(agent._goal_xy, before)
    assert agent.stats.get("approach_retargeted", 0) == 0


def test_retargeting_is_off_unless_asked_for():
    """The shipped default is 0.0, and it has to stay a no-op: every earlier
    condition of the campaign ladder is reproducible only if it does."""
    agent, track = _approach_agent_with_track([3.0, 0.5, 0.0])
    before = agent._goal_xy.copy()

    track.ellipsoid.center = np.array([9.0, 0.5, 9.0])  # a huge move
    agent.approach.step(_frame([0.0, 0.0]))

    assert agent.approach.retargets == 0
    assert np.allclose(agent._goal_xy, before)


def test_a_track_that_will_not_settle_stops_being_chased():
    """The cap is a cap, not a budget: a centre that keeps moving is not
    converging, and the walk should end on the estimate it has."""
    agent, track = _approach_agent_with_track([3.0, 0.5, 0.0],
                                              approach_retarget_m=0.15,
                                              approach_retarget_max=2)
    for i in range(5):
        track.ellipsoid.center = np.array([3.0 + 0.5 * (i + 1), 0.5, 0.0])
        agent.approach.step(_frame([0.0, 0.0]))

    assert agent.approach.retargets == 2
    assert agent.stats["approach_retargeted"] == 2


def test_a_struck_off_candidate_is_recorded_with_the_belief_it_carried():
    """`unreachable_skip` counts events and names no track, and that ambiguity
    hides a real failure: on 00848's cross_anchor red plate the episode ends
    holding the true plate at 0.01 m with p=0.953 and never goes to it, while
    the counter reads 2 against a `max_identity_rejections` of 2. Whether those
    are the same track is the whole question, and the aggregate cannot say."""
    from osg.objects.association import ObjectTrack
    from osg.objects.ellipsoid import Ellipsoid

    agent = make_agent(target="chair")
    track = ObjectTrack(
        id=3, label="chair",
        ellipsoid=Ellipsoid(center=np.array([2.0, 0.5, 3.0]),
                            axes=np.array([0.2, 0.2, 0.2]), R=np.eye(3)),
    )
    track.presence.log_odds = 3.0  # p ~ 0.95: believed, and struck off anyway

    agent.candidates._log_reject(track, "unreachable")

    (rec,) = agent.candidate_reject_log
    assert rec["track_id"] == 3 and rec["reason"] == "unreachable"
    assert rec["p"] > 0.95, "the belief at the moment of the strike is the point"
    assert rec["rejections"] == 1
    assert rec["center"][:3] == [2.0, 0.5, 3.0]
