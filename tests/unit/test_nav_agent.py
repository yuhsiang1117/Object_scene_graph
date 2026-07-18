"""Unit tests for NavAgent.

Two areas covered so far:
1. The terminal APPROACH phase (P1a) — replaced three earlier distance-based
   stopping strategies, all of which stalled at dtg 0.107-0.147 m; see
   docs/DESIGN_AND_ROADMAP.md P0->P1 history. Tests call `_do_approach`
   directly rather than driving the full `act()` loop: its decision logic
   only depends on the detector, the costmap/planner, and a handful of
   `_approach_*` fields, so exercising it in isolation keeps this hermetic
   (no Hydra/habitat) and fast.
2. The frontier give-up progress timer (P1b) — `_select_new_frontier` must
   reset `_progress_ref_step`/`_progress_ref_xy` to the current step/pose
   whenever it starts pursuing a (new or re-selected) frontier; otherwise a
   stale reference from a prior pursuit can trigger a spurious give-up
   within the first step of the new one.

Uses a lightweight SimpleNamespace in place of a real Hydra config: only
the fields NavAgent.__init__/reset()/the methods under test actually read
are set.
"""
from __future__ import annotations

import types

import numpy as np

from osg.agent.nav_agent import STOP_ACTION, NavAgent, State
from osg.core.types import CameraIntrinsics, Detection
from osg.exploration.async_scorer import AsyncScorer
from osg.exploration.scorer import NearestScorer
from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN
from osg.perception.detector import StubDetector

from .conftest import make_camera, make_frame

APPROACH_BBOX_THRESHOLD = 40_000.0
_INTRINSICS = CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0, width=640, height=480)


def make_cfg(**agent_overrides) -> types.SimpleNamespace:
    cfg = types.SimpleNamespace(
        mapping=types.SimpleNamespace(
            resolution_m=0.05, inflate_margin_m=0.07,
            obstacle_low_m=0.1, obstacle_high_m=1.5, max_range_m=5.0, depth_stride=4,
        ),
        exploration=types.SimpleNamespace(
            frontier_min_cells=8, frontier_dedup_m=1.0,
            unscored_prior=0.3, min_path_cost_m=0.5, top_n_frontiers=5,
        ),
        scene_graph=types.SimpleNamespace(
            room_min_radius_m=0.9, room_door_width_m=1.2,
            assoc_score_thresh=0.4, assoc_depth_gate_m=0.5,
            min_obs_for_refine=3, refine_every=3, link_dist_m=1.0,
            keyframe_trans_m=0.25, keyframe_rot_deg=30.0, room_seg_every_kf=10,
            min_det_score=0.35, min_det_bbox_px=1500.0, confirm_baseline_m=0.0,
        ),
        agent=types.SimpleNamespace(
            agent_radius=0.18, forward_m=0.25, turn_deg=30.0, initial_scan=False,
            camera_height=0.88, approach_stop_bbox_px=APPROACH_BBOX_THRESHOLD,
            approach_max_steps=12,
            approach_goal_tolerance_m=0.12, approach_arrival_tol_m=0.1,
        ),
        verification=types.SimpleNamespace(
            ring_radii_m=[0.8, 1.2, 1.5, 2.0], min_obs=3, min_score=0.45, min_bbox_px=3000.0,
        ),
        detector=types.SimpleNamespace(vocabulary=["chair", "bed"]),
    )
    for k, v in agent_overrides.items():
        setattr(cfg.agent, k, v)
    return cfg


def make_agent(cfg=None, target="chair") -> NavAgent:
    agent = NavAgent(cfg or make_cfg(), StubDetector(), AsyncScorer(NearestScorer()), None, target)
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


def test_stops_when_bbox_large_enough():
    agent = make_agent()
    agent.state = State.APPROACH
    agent._goal_xy = np.array([5.0, 0.0])
    agent._approach_steps_left = 5
    agent.detector.push([_det("chair", (300, 300))])  # area 90000 > 40000 threshold

    action = agent._do_approach(_frame([0.0, 0.0]))

    assert action == STOP_ACTION
    assert agent.state == State.DONE
    assert agent.approach_stop_reason == "bbox"
    assert agent.approach_bbox_log == [(0, 90000.0)]  # calibration data for P1c


def test_advances_when_visible_but_small():
    agent = make_agent()
    agent.state = State.APPROACH
    agent._goal_xy = np.array([5.0, 0.0])
    agent._approach_steps_left = 5
    agent.detector.push([_det("chair", (50, 50))])  # area 2500 < threshold

    action = agent._do_approach(_frame([0.0, 0.0]))

    assert action != STOP_ACTION
    assert agent.state == State.APPROACH
    assert agent._approach_steps_left == 4
    assert agent._approach_last_good_xy is not None
    assert np.allclose(agent._approach_last_good_xy, [0.0, 0.0])


def test_ignores_detections_of_other_labels():
    agent = make_agent(target="chair")
    agent.state = State.APPROACH
    agent._goal_xy = np.array([5.0, 0.0])
    agent._approach_steps_left = 5
    agent.detector.push([_det("sofa", (500, 500))])  # huge, but wrong label

    action = agent._do_approach(_frame([0.0, 0.0]))

    assert action != STOP_ACTION
    assert agent.state == State.APPROACH  # not fooled into stopping
    assert agent._approach_last_good_xy is None  # never confirmed visible


def test_retreats_when_visibility_lost():
    agent = make_agent()
    agent.state = State.APPROACH
    agent._goal_xy = np.array([5.0, 0.0])
    agent._approach_steps_left = 5
    agent._approach_last_good_xy = np.array([0.0, 0.0])  # a previously-visible pose
    agent.detector.push([])  # nothing visible from the current pose

    steps_before = agent._approach_steps_left
    action = agent._do_approach(_frame([1.0, 0.0]))  # 1 m away from the good pose

    assert action != STOP_ACTION  # heads back toward the good pose
    assert agent.state == State.APPROACH
    assert agent._approach_steps_left == steps_before  # retreat doesn't spend the advance budget
    assert agent.approach_stop_reason is None  # hasn't stopped yet


def test_falls_through_to_advance_when_never_visible():
    agent = make_agent()
    agent.state = State.APPROACH
    agent._goal_xy = np.array([5.0, 0.0])
    agent._approach_steps_left = 5
    agent._approach_last_good_xy = None  # never seen it
    agent.detector.push([])

    action = agent._do_approach(_frame([0.0, 0.0]))

    assert action != STOP_ACTION
    assert agent.state == State.APPROACH
    assert agent._approach_steps_left == 4  # advance budget spent (no retreat target)
    assert agent._approach_last_good_xy is None


def test_stops_at_step_budget():
    agent = make_agent()
    agent.state = State.APPROACH
    agent._goal_xy = np.array([5.0, 0.0])
    agent._approach_steps_left = 0  # budget exhausted
    agent.detector.push([_det("chair", (50, 50))])  # visible but small: would normally advance

    action = agent._do_approach(_frame([0.0, 0.0]))

    assert action == STOP_ACTION
    assert agent.state == State.DONE
    assert agent.approach_stop_reason == "deadline"


def test_stops_at_deadline():
    agent = make_agent()
    agent.state = State.APPROACH
    agent._goal_xy = np.array([5.0, 0.0])
    agent._approach_steps_left = 5
    agent._goto_deadline = agent.step_count - 1  # already past
    agent.detector.push([_det("chair", (50, 50))])

    action = agent._do_approach(_frame([0.0, 0.0]))

    assert action == STOP_ACTION
    assert agent.state == State.DONE
    assert agent.approach_stop_reason == "deadline"


def test_stops_when_goal_unreachable():
    agent = make_agent()
    goal = np.array([5.0, 0.0])
    rc = agent.costmap.world_to_grid(goal)
    agent.costmap.grid[rc[0] - 5 : rc[0] + 6, rc[1] - 5 : rc[1] + 6] = OCCUPIED
    agent.state = State.APPROACH
    agent._goal_xy = goal
    agent._approach_steps_left = 5
    agent.detector.push([])  # never visible; falls through to the advance branch

    action = agent._do_approach(_frame([0.0, 0.0]))

    assert action == STOP_ACTION
    assert agent.state == State.DONE
    assert agent.approach_stop_reason == "path_consumed"


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
    agent._last_select_step = -100  # bypass the 5-step selection throttle
    # Stale reference from a much earlier, unrelated pursuit.
    agent._progress_ref_step = 10
    agent._progress_ref_xy = np.array([-999.0, -999.0])

    agent._select_new_frontier(_frame([0.0, 0.0], frame_id=agent.step_count))

    assert agent.state == State.GOTO_FRONTIER
    assert agent._progress_ref_step == agent.step_count
    assert np.allclose(agent._progress_ref_xy, [0.0, 0.0])


def test_giveup_logged_with_frontier_and_agent_position():
    agent = make_agent()
    _carve_free_square(agent)
    agent.state = State.GOTO_FRONTIER
    agent._current_frontier = types.SimpleNamespace(centroid_xy=np.array([3.0, 4.0]), id=0)
    agent.step_count = 100
    agent._progress_ref_step = 80  # 20 steps ago, past the 15-step check window
    agent._progress_ref_xy = np.array([0.0, 0.0])  # same as current -> "no progress"
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

    agent._follow_to(_frame([0.0, 0.0]), np.array([1.0, 0.0]))

    assert plan_calls == [0.12]
    assert act_calls == [0.1]
