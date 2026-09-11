"""The navigation switch: which mover drives, and what replaces the oracle."""
from __future__ import annotations

import numpy as np
import pytest

from osg.core.config import AgentConfig, resolve_navigation
from osg.objects.object_layer import ObjectLayer


# ------------------------------------------------------------ mode resolution


def test_unset_navigation_falls_back_to_the_legacy_flag():
    """Every preset written before agent.navigation existed sets
    use_habitat_navmesh, and those numbers must not move."""
    assert resolve_navigation(AgentConfig()) == "costmap"
    assert resolve_navigation(AgentConfig(use_habitat_navmesh=True)) == "navmesh"


def test_explicit_navigation_wins_when_the_legacy_flag_is_off():
    assert resolve_navigation(AgentConfig(navigation="pointnav")) == "pointnav"
    assert resolve_navigation(AgentConfig(navigation="costmap")) == "costmap"
    assert resolve_navigation(AgentConfig(navigation="navmesh")) == "navmesh"


def test_navmesh_and_pointnav_together_is_rejected():
    """Silently picking one would mean a preset that reads 'sensor-only' could
    quietly run on the navmesh -- exactly the confusion S8 exists to remove."""
    cfg = AgentConfig(navigation="pointnav", use_habitat_navmesh=True)
    with pytest.raises(ValueError, match="contradicts"):
        resolve_navigation(cfg)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="not one of"):
        resolve_navigation(AgentConfig(navigation="navmesh_but_faster"))


# ------------------------------------------------- spatial target rejection


def _layer_with_track_at(xy):
    """An ObjectLayer holding one hand-built track at a known ground position."""
    from osg.objects.association import ObjectTrack
    from osg.objects.ellipsoid import Ellipsoid

    layer = ObjectLayer()
    track = ObjectTrack(
        id=0,
        label="chair",
        ellipsoid=Ellipsoid(
            center=np.array([xy[0], 0.5, xy[1]]), axes=np.full(3, 0.2), R=np.eye(3)
        ),
    )
    track.evidence = 10.0
    track.best_score = 0.9
    track.best_bbox_px = 50_000.0
    layer._tracks[0] = track
    return layer


def test_disable_target_retires_a_place_not_an_id():
    """`blacklist` retires an identity, so the same object returns under a new
    track id. Abandoning an approach must retire the LOCATION -- and it does so
    through the same `_disabled_pts` list the false-positive retraction uses, so
    a re-detection dies at track birth instead of being filtered at every query.
    """
    layer = _layer_with_track_at((3.0, 4.0))
    assert [t.id for t in layer.candidates("chair", min_obs=0)] == [0]

    assert layer.disable_target(0) is True
    assert layer.candidates("chair", min_obs=0) == []

    # A newborn track of the same label at the same place is rejected on
    # arrival, whatever id association hands it.
    reborn = _layer_with_track_at((3.1, 4.1))._tracks[0]
    assert layer._in_disabled_region(reborn)


def test_disable_target_leaves_other_places_alone():
    layer = _layer_with_track_at((3.0, 4.0))
    layer.disable_target(0)
    elsewhere = _layer_with_track_at((-3.0, -4.0))._tracks[0]
    assert not layer._in_disabled_region(elsewhere)


def test_disable_target_is_label_scoped():
    """Retiring a mis-detected chair must not blind the agent to a real bed
    standing in the same spot."""
    layer = _layer_with_track_at((3.0, 4.0))
    layer.disable_target(0)
    other = _layer_with_track_at((3.0, 4.0))._tracks[0]
    other.label = "bed"
    assert not layer._in_disabled_region(other)


def test_disable_place_works_without_a_track():
    layer = _layer_with_track_at((3.0, 4.0))
    layer.disable_place(np.array([3.0, 4.0]), "chair")
    assert layer._in_disabled_region(layer._tracks[0])
    assert layer.disable_target(99) is False


# ------------------------------------------- abandoning instead of stopping


def _approaching_agent(**agent_overrides):
    """A NavAgent parked in APPROACH with a committed target, no oracle."""
    from osg.mapping.costmap import PLANE

    from .test_nav_agent import make_agent, make_cfg

    cfg = make_cfg(
        approach_abandon_steps=100,
        navmesh_approach_steps=200,
        **agent_overrides,
    )
    agent = make_agent(cfg)
    agent.state = agent.state.APPROACH
    agent._goal_xy = np.array([5.0, 0.0])
    agent._target_obj_xy = np.array([5.0, 0.0])
    agent._approach_steps_left = 10 ** 9
    agent._goto_deadline = 10 ** 9
    agent._approach_start_step = 0
    agent._candidate_id = None
    _ = PLANE
    return agent


def _frame_at(xy):
    from .test_nav_agent import _frame

    return _frame(xy)


def test_running_out_of_approach_budget_returns_to_exploring():
    """The whole substitute for is_reachable. A sensor-only agent cannot know a
    target is behind a sealed wall, so it commits and gives up -- and giving up
    must mean 'keep looking', not 'stop here and fail the episode'."""
    from osg.agent.nav_agent import STOP_ACTION, State

    agent = _approaching_agent()
    agent.step_count = 100
    action = agent._do_approach(_frame_at([0.0, 0.0]))

    assert agent.state is State.EXPLORE
    assert action != STOP_ACTION
    assert agent.stats.get("approach_abandon") == 1
    assert agent._goal_xy is None and agent._target_obj_xy is None


def test_the_abandoned_place_is_retired_not_just_the_track():
    """No candidate track here (the abandon happens without one), so the
    location has to be retired by position + label."""
    agent = _approaching_agent()
    agent.step_count = 100
    agent._do_approach(_frame_at([0.0, 0.0]))
    assert [lbl for _, lbl in agent.object_layer._disabled_pts] == ["chair"]
    xy, _ = agent.object_layer._disabled_pts[0]
    assert np.allclose(xy, [5.0, 0.0])


def test_within_budget_the_approach_continues():
    from osg.agent.nav_agent import State

    agent = _approaching_agent()
    agent.step_count = 99
    agent._do_approach(_frame_at([0.0, 0.0]))
    assert agent.state is State.APPROACH
    assert "approach_abandon" not in agent.stats


def test_the_budget_is_not_applied_when_an_oracle_exists():
    """navmesh mode rejects unreachable targets up front, so it must not also
    pay this cost -- every number measured on it has to stay reproducible."""
    from osg.agent.nav_agent import State

    agent = _approaching_agent()
    agent._reachable_fn = lambda xy: True
    agent.step_count = 500
    agent._do_approach(_frame_at([0.0, 0.0]))
    assert agent.state is State.APPROACH
    assert "approach_abandon" not in agent.stats


# --------------------------------------------- the escape guard, in the loop


def test_escape_guard_breaks_a_spin_through_act():
    from osg.agent.nav_agent import State

    from .test_nav_agent import _frame, make_agent, make_cfg

    agent = make_agent(make_cfg(escape_window=4))
    agent.state = State.VERIFYING
    # VERIFYING with no verifier and nothing centred keeps returning a turn.
    actions = [agent.act(_frame([0.0, 0.0], frame_id=i)) for i in range(6)]
    assert "move_forward" in actions, actions
    assert agent._escape.n_forced_forward >= 1


def test_the_terminal_stop_survives_the_escape_guard():
    """Deviation from ASCENT, and the reason for it: overriding the STOP that
    DONE has just committed would mean the episode can never end."""
    from osg.agent.nav_agent import STOP_ACTION, State

    from .test_nav_agent import _frame, make_agent, make_cfg

    agent = make_agent(make_cfg(escape_window=4))
    agent.state = State.VERIFYING
    for i in range(5):
        agent.act(_frame([0.0, 0.0], frame_id=i))
    # Force the next _act_inner to be a committed terminal stop.
    agent.state = State.DONE
    agent._escape._history.clear()
    agent._escape._history.extend(["turn_left"] * 4)
    assert agent.act(_frame([0.0, 0.0], frame_id=9)) == STOP_ACTION


# ------------------------------------------------ the two frontier-stall rules


def _goto_frontier_agent(rule, **overrides):
    from osg.agent.nav_agent import State
    from osg.mapping.frontier import Frontier

    from .test_nav_agent import make_agent, make_cfg

    agent = make_agent(make_cfg(
        frontier_stick_rule=rule,
        **overrides,
    ))
    agent.state = State.GOTO_FRONTIER
    agent._current_frontier = Frontier(
        id=1, centroid_xy=np.array([10.0, 0.0]), cells=np.zeros((0, 2), dtype=int), size=20
    )
    agent._progress_ref_step = 0
    agent._progress_ref_xy = np.zeros(2)
    return agent


def _stall(agent, positions):
    """Feed a trajectory; return the step at which the frontier was retired."""
    for i, xy in enumerate(positions, start=1):
        agent.step_count = i
        if agent._frontier_stalled(
            np.asarray(xy, dtype=float), agent._current_frontier, 0.3, 20
        ):
            return i
    return None


def test_displacement_rule_catches_a_motionless_agent():
    agent = _goto_frontier_agent("displacement")
    assert _stall(agent, [(0.0, 0.0)] * 40) is not None


def test_displacement_rule_misses_an_orbiting_agent():
    """The gap that made this a new rule rather than a retune: the agent moves
    every step, so displacement never looks stalled -- yet it never arrives.
    Observed as one frontier held from step 85 to 500 on a pointnav smoke run.
    """
    agent = _goto_frontier_agent("displacement")
    circle = [(10.0 + 4 * np.cos(t / 5.0), 4 * np.sin(t / 5.0)) for t in range(120)]
    assert _stall(agent, circle) is None


def test_closing_rule_catches_the_orbiting_agent():
    agent = _goto_frontier_agent("closing")
    circle = [(10.0 + 4 * np.cos(t / 5.0), 4 * np.sin(t / 5.0)) for t in range(120)]
    assert _stall(agent, circle) is not None


def test_closing_rule_lets_a_slow_approach_run():
    """5 cm of closing per step is under the 0.3 m threshold on any single step,
    but it accumulates -- the reference must reset on real progress or every
    approach would be retired."""
    agent = _goto_frontier_agent("closing")
    walk = [(10.0 - (100 - t) * 0.05, 0.0) for t in range(100)]
    assert _stall(agent, walk) is None


def test_closing_rule_resets_when_pushed_backwards():
    """ASCENT resets on a change in either direction: being shoved away from
    the frontier is a different failure from going nowhere."""
    agent = _goto_frontier_agent("closing")
    away = [(10.0 - 5.0 - t * 0.05, 0.0) for t in range(100)]
    assert _stall(agent, away) is None


def test_stick_steps_zero_disables_the_test():
    agent = _goto_frontier_agent("closing")
    agent.step_count = 500
    assert agent._frontier_stalled(np.zeros(2), agent._current_frontier, 0.3, 0) is False


def test_frontier_arrival_tolerance_covers_the_movers_stop_radius(monkeypatch):
    """A mover that reports arrival at 0.9 m must not have those arrivals
    classified as degenerate stub paths (0.5 m) and its frontiers retired."""
    from .test_nav_agent import make_agent, make_cfg

    class _P:
        num_recurrent_layers = 4

        def to(self, d):
            return self

    monkeypatch.setattr(
        "osg.planning.pointnav_driver.load_pointnav_policy", lambda _p: _P()
    )
    cfg = make_cfg(
        navigation="pointnav", pointnav_weights="unused.pth",
        pointnav_stop_radius=0.9, pointnav_depth_shape=[224, 224],
        pointnav_approach_creep_m=1.0,
    )
    cfg.eval = types_ns(depth_min_m=0.5, depth_max_m=5.0)
    agent = make_agent(cfg)
    assert agent.navigation == "pointnav"
    assert agent._driver is agent.pointnav
    assert agent._frontier_reach_m >= 0.9


def types_ns(**kw):
    import types

    return types.SimpleNamespace(**kw)


def test_approach_creep_does_not_leak_into_climb(monkeypatch):
    """`_follow_to` is shared by APPROACH and CLIMB. The creep must not
    suppress the `action is None` that ends an unreachable climb."""
    from osg.agent.nav_agent import State

    from .test_nav_agent import _frame, make_agent, make_cfg

    calls = []

    class _Driver:
        stop_radius = 0.9

        def observe(self, frame):
            pass

        def reset(self):
            pass

        def __call__(self, goal_xy, *, stop_radius=None, creep_below=0.0):
            calls.append(creep_below)
            return None

    cfg = make_cfg(navigation="pointnav", pointnav_approach_creep_m=1.0)
    agent = make_agent(cfg, pointnav=_Driver())
    frame = _frame([0.0, 0.0])

    agent.state = State.APPROACH
    agent._follow_to(frame, np.array([1.0, 0.0]))
    agent.state = State.CLIMB
    agent._follow_to(frame, np.array([1.0, 0.0]))

    assert calls == [1.0, 0.0]


# =========================================================== S30: the three fixes


def _approach_with_cloud(dist_m, **overrides):
    """An agent in APPROACH whose committed target has a surface cloud at a
    known ground-plane distance, and NO live detection this frame."""
    import numpy as np

    from osg.agent.nav_agent import State
    from osg.objects.association import ObjectTrack
    from osg.objects.ellipsoid import Ellipsoid

    from .test_nav_agent import make_agent, make_cfg

    cfg = make_cfg(
        terminal_rule="nearest_point", terminal_stop_m=0.4, terminal_engage_m=1.0,
        terminal_percentile=0.0, terminal_progress_eps=0.1, terminal_stall_steps=3,
        approach_abandon_steps=0, navmesh_approach_steps=200, **overrides,
    )
    agent = make_agent(cfg)
    track = ObjectTrack(
        id=0, label="chair",
        ellipsoid=Ellipsoid(center=np.array([dist_m, 0.5, 0.0]),
                            axes=np.full(3, 0.1), R=np.eye(3)),
    )
    track.points_w = np.array([[dist_m, 0.5, 0.0]])
    agent.object_layer._tracks[0] = track
    agent._candidate_id = 0
    agent.state = State.APPROACH
    agent._goal_xy = np.array([dist_m, 0.0])
    agent._target_obj_xy = np.array([dist_m, 0.0])
    agent._approach_steps_left = 10 ** 9
    agent._goto_deadline = 10 ** 9
    agent._approach_start_step = 0
    agent.detector.push([])  # nothing visible this frame
    return agent


def test_fix1_stops_on_the_map_when_the_target_is_not_visible():
    """The core S30 fix. The rule measures the accumulated cloud, so losing
    sight of the target must not stop it firing -- that is what left 28
    episodes parked a median 0.67 m from their goal."""
    from osg.agent.nav_agent import STOP_ACTION, State

    from .test_nav_agent import _frame

    agent = _approach_with_cloud(0.3, terminal_requires_detection=False)
    assert agent._do_approach(_frame([0.0, 0.0])) == STOP_ACTION
    assert agent.state is State.DONE
    assert agent.approach_stop_reason == "nearest_point"


def test_fix1_off_keeps_the_detection_gate():
    """Default behaviour, which every pre-S30 number was measured on."""
    from osg.agent.nav_agent import STOP_ACTION, State

    from .test_nav_agent import _frame

    agent = _approach_with_cloud(0.3, terminal_requires_detection=True)
    assert agent._do_approach(_frame([0.0, 0.0])) != STOP_ACTION
    assert agent.state is State.APPROACH


def test_fix1_does_not_stop_while_still_far():
    """Ungating must not mean stopping early -- the distance test still rules."""
    from osg.agent.nav_agent import STOP_ACTION, State

    from .test_nav_agent import _frame

    agent = _approach_with_cloud(4.0, terminal_requires_detection=False)
    assert agent._do_approach(_frame([0.0, 0.0])) != STOP_ACTION
    assert agent.state is State.APPROACH


def test_fix2_policy_stop_forces_forward_instead_of_retiring():
    """ASCENT overwrites a network STOP on an explore frontier with one forward
    step and keeps the target (ascent_policy.py:705-711)."""
    from osg.agent.nav_agent import State
    from osg.mapping.frontier import Frontier
    from osg.planning.pointnav_driver import NavStep

    from .test_nav_agent import _frame, make_agent, make_cfg

    class _Driver:
        stop_radius = 0.9

        def observe(self, frame):
            pass

        def reset(self):
            pass

        def step(self, goal_xy, **kw):
            return NavStep(None, "policy_stop")

        def __call__(self, goal_xy, **kw):
            return self.step(goal_xy, **kw).action

    for blocked, expect in ((False, "move_forward"), (True, None)):
        agent = make_agent(
            make_cfg(navigation="pointnav", pointnav_stop_means_blocked=blocked),
            pointnav=_Driver(),
        )
        agent.state = State.GOTO_FRONTIER
        agent._current_frontier = Frontier(
            id=1, centroid_xy=np.array([9.0, 0.0]),
            cells=np.zeros((0, 2), dtype=int), size=20,
        )
        assert agent._follow_path(_frame([0.0, 0.0])) == expect
        assert agent.stats.get("frontier_stub_block", 0) == (1 if blocked else 0)


def test_fix3_straight_line_planner_never_vetoes():
    from osg.mapping.costmap import OCCUPIED, Costmap2D
    from osg.planning.planner import AStarPlanner, StraightLinePlanner

    cm = Costmap2D(resolution=0.05, size_m=10.0)
    cm.grid[:, :] = OCCUPIED  # a goal no A* can reach
    start, goal = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
    assert AStarPlanner().plan(cm, start, goal).success is False
    r = StraightLinePlanner().plan(cm, start, goal)
    assert r.success and r.cost == pytest.approx(4.0)


def test_fix3_gate_off_selects_the_straight_line_planner():
    from osg.planning.planner import StraightLinePlanner

    from .test_nav_agent import make_agent, make_cfg

    on = make_agent(make_cfg(frontier_reachability_gate=True))
    assert on.selection_planner is on.planner

    class _D:
        stop_radius = 0.9

        def observe(self, f):
            pass

        def reset(self):
            pass

    off = make_agent(
        make_cfg(navigation="pointnav", frontier_reachability_gate=False), pointnav=_D()
    )
    assert isinstance(off.selection_planner, StraightLinePlanner)


def test_fix3_gate_is_forced_on_without_a_self_planning_mover():
    """The costmap arm really does drive on the plan; the gate must stay."""
    from .test_nav_agent import make_agent, make_cfg

    a = make_agent(make_cfg(navigation="costmap", frontier_reachability_gate=False))
    assert a.selection_planner is a.planner


def test_fix3_releases_a_frontier_that_was_explored_away():
    from osg.mapping.costmap import FREE, UNKNOWN
    from osg.mapping.frontier import Frontier

    from .test_nav_agent import make_agent, make_cfg

    class _D:
        stop_radius = 0.9

        def observe(self, f):
            pass

        def reset(self):
            pass

    agent = make_agent(
        make_cfg(navigation="pointnav", frontier_reachability_gate=False), pointnav=_D()
    )
    f = Frontier(id=1, centroid_xy=np.array([2.0, 0.0]),
                 cells=np.zeros((0, 2), dtype=int), size=20)

    agent.costmap.grid[:, :] = FREE  # nothing unknown left anywhere
    assert agent._frontier_consumed(f) is True

    rc = agent.costmap.world_to_grid(np.array([2.0, 0.0]))
    agent.costmap.grid[rc[0], rc[1] + 1] = UNKNOWN  # boundary restored
    assert agent._frontier_consumed(f) is False


# ========================================= S34: per-step frontier re-selection


def _reselect_agent(reselect_every, select_every=1, frontiers=(), **over):
    """An agent in GOTO_FRONTIER whose extractor returns `frontiers`."""
    from osg.agent.nav_agent import State
    from osg.mapping.costmap import FREE
    from osg.mapping.frontier import Frontier

    from .test_nav_agent import make_agent, make_cfg

    cfg = make_cfg(**over)
    cfg.exploration.select_every = select_every
    cfg.exploration.reselect_every = reselect_every
    agent = make_agent(cfg)
    agent.costmap.grid[:, :] = FREE
    agent.frontier_extractor.extract = lambda *a, **k: [
        Frontier(id=i, centroid_xy=np.asarray(xy, dtype=float),
                 cells=np.zeros((0, 2), dtype=int), size=30)
        for i, xy in enumerate(frontiers)
    ]
    agent.state = State.GOTO_FRONTIER
    return agent


def _seed_pursuit(agent, xy):
    from osg.mapping.frontier import Frontier

    f = Frontier(id=99, centroid_xy=np.asarray(xy, dtype=float),
                 cells=np.zeros((0, 2), dtype=int), size=30)
    f.path_cost = 1.0
    agent._current_frontier = f
    return f


def test_reselect_defaults_off():
    a = _reselect_agent(0, select_every=5)
    assert a._reselect_every == 0 and a._select_every == 5


def test_reselecting_the_same_frontier_keeps_the_stall_clock_running():
    """The stall rule needs 20 CONSECUTIVE steps of not closing. If a
    re-selection that lands on the same frontier reset its clock, the guard
    could never accumulate them and would silently stop existing."""
    a = _reselect_agent(1, frontiers=[(4.0, 0.0)])
    _seed_pursuit(a, (4.0, 0.0))
    a._progress_ref_step, a._frontier_ref_dist = 3, 2.5
    a.step_count = 30
    a._last_select_step = 0

    a._select_new_frontier(_frame_at([0.0, 0.0]))

    assert a._progress_ref_step == 3, "same target must not restart the clock"
    assert a._frontier_ref_dist == 2.5
    assert a.stats.get("frontier_switch", 0) == 0


def test_switching_to_a_different_frontier_restarts_the_clock():
    a = _reselect_agent(1, frontiers=[(-6.0, 2.0)])
    _seed_pursuit(a, (4.0, 0.0))
    a._progress_ref_step, a._frontier_ref_dist = 3, 2.5
    a.step_count = 30
    a._last_select_step = 0

    a._select_new_frontier(_frame_at([0.0, 0.0]))

    assert a._current_frontier is not None
    assert np.allclose(a._current_frontier.centroid_xy, [-6.0, 2.0])
    assert a._progress_ref_step == 30, "a new pursuit gets a fresh window"
    assert a._frontier_ref_dist is None
    assert a.stats.get("frontier_switch", 0) == 1


def test_select_every_gates_how_often_selection_runs():
    """The throttle is what makes per-step selection affordable; it must be the
    configured value rather than a hardcoded 5."""
    for every, expect in ((1, True), (5, False)):
        a = _reselect_agent(1, select_every=every, frontiers=[(4.0, 0.0)])
        seen = []
        inner = a.frontier_extractor.extract
        a.frontier_extractor.extract = lambda *x, **k: (seen.append(1) or inner())
        a._last_select_step, a.step_count = 0, 1
        a._select_new_frontier(_frame_at([0.0, 0.0]))
        assert bool(seen) is expect, f"select_every={every} gated wrongly"


def test_in_pursuit_reselection_only_fires_when_enabled():
    """With reselect off, GOTO_FRONTIER must not call selection at all -- that
    is the behaviour every number before S34 was measured on."""
    from osg.agent.nav_agent import State

    for reselect, expect in ((0, 0), (1, 1)):
        a = _reselect_agent(reselect, frontiers=[(4.0, 0.0)])
        _seed_pursuit(a, (4.0, 0.0))
        a.state = State.GOTO_FRONTIER
        a.step_count, a._last_select_step = 50, 0
        # Keep the pursuit alive on BOTH exits. A consumed path or a fired
        # stall rule drops the FSM into EXPLORE, whose post-transition selects
        # too -- either would count here and hide what the test is about.
        a._progress_ref_step, a._progress_ref_xy = a.step_count, np.zeros(2)
        a._follow_path = lambda frame: "move_forward"
        calls = []
        a._select_new_frontier = lambda frame: calls.append(1)
        assert a._act_inner(_frame_at([0.0, 0.0])) == "move_forward"
        assert len(calls) == expect, f"reselect_every={reselect}"


# =============================================== S40: the arrival signal

def test_approach_arrival_ends_the_approach_like_the_navmesh_did():
    """The navmesh ends 46 of 100 episodes by reporting arrival
    (`path_consumed`); the pointnav arm ended ZERO because the creep swallowed
    the check. With an arrival radius the same terminal path fires."""
    from osg.agent.nav_agent import STOP_ACTION, State
    from osg.planning.pointnav_driver import NavStep

    from .test_nav_agent import _frame, make_agent, make_cfg

    class _Driver:
        stop_radius = 0.9

        def __init__(self):
            self.seen = []

        def observe(self, frame):
            pass

        def reset(self):
            pass

        def step(self, goal_xy, *, stop_radius=None, creep_below=0.0):
            self.seen.append((stop_radius, creep_below))
            # inside the arrival radius
            return NavStep(None, "arrived")

        def __call__(self, goal_xy, **kw):
            # `_follow_to` uses the action-only form, as the navmesh call site
            # does -- both report arrival by returning None.
            return self.step(goal_xy, **kw).action

    d = _Driver()
    cfg = make_cfg(navigation="pointnav", pointnav_arrival_m=0.3,
                   pointnav_approach_creep_m=1.0, terminal_rule="depth",
                   approach_abandon_steps=0)
    a = make_agent(cfg, pointnav=d)
    a.state = State.APPROACH
    a._goal_xy = np.array([0.2, 0.0])
    a._approach_steps_left = 10 ** 9
    a._goto_deadline = 10 ** 9
    a._approach_start_step = 0
    a.detector.push([])  # nothing visible: only the navigator can conclude

    action = a._do_approach(_frame([0.0, 0.0]))
    assert d.seen == [(0.3, 1.0)], "the arrival radius must reach the driver"
    assert action == STOP_ACTION
    assert a.state is State.DONE
    assert a.approach_stop_reason == "path_consumed"


def test_arrival_defaults_off_so_prior_numbers_stand():
    from .test_nav_agent import make_cfg

    assert float(getattr(make_cfg().agent, "pointnav_arrival_m", 0.0)) == 0.0
