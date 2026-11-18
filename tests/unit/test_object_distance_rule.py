"""The authored benchmark scored by the released rule (sim/ycb_env.py,
`ObjectDistanceRule`): a STOP within 1 m horizontal of the object, within three
attempts, SPL against the geodesic to the object; off by default, and a None
verdict from the env hands the attempt back to habitat's viewpoint rule."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from osg.eval.attempts import attempt_succeeded
from osg.sim.ycb_env import ObjectDistanceRule


def test_disabled_by_default():
    from osg.core.config import OSGConfig
    cfg = OSGConfig()
    assert cfg.ycb.score_by_object_distance is False
    assert cfg.ycb.object_success_distance_m == 1.0


def test_a_stop_within_a_metre_scores_and_a_far_one_does_not():
    rule = ObjectDistanceRule(enabled=True)
    target = np.array([3.0, 0.9, 1.0])
    assert rule.record_stop(10, np.array([2.4, 0.1, 1.5]), target) is True   # 0.78 m
    assert rule.record_stop(20, np.array([1.5, 0.1, 1.0]), target) is False  # 1.5 m
    assert [a["success"] for a in rule.attempts] == [True, False]
    assert abs(rule.attempts[0]["distance_horizontal_m"] - 0.781) < 0.01


def test_height_does_not_count():
    rule = ObjectDistanceRule(enabled=True)
    assert rule.record_stop(1, np.array([0.5, 0.1, 0.0]), np.array([0.0, 2.5, 0.0])) is True


def test_a_stop_on_another_storey_does_not_score():
    """outputs/RERUN/mf5 00862 cracker box: 0.62 m horizontal, a storey up."""
    rule = ObjectDistanceRule(enabled=True, max_floor_dy_m=1.0)
    rule.target_floor_y = 0.106
    target = np.array([-1.0, 1.0, 4.0])
    assert rule.record_stop(141, np.array([-1.2, 3.306, 4.6]), target) is False
    attempt = rule.attempts[-1]
    assert attempt["on_target_floor"] is False
    assert attempt["success_horizontal_only"] is True
    assert abs(attempt["floor_dy_m"] - 3.2) < 1e-6
    summary = rule.summary(np.array([-1.2, 3.306, 4.6]), target)
    assert summary["success"] == 0 and summary["success_horizontal_only"] == 1


def test_the_same_stop_on_the_objects_storey_scores():
    rule = ObjectDistanceRule(enabled=True, max_floor_dy_m=1.0)
    rule.target_floor_y = 0.106
    # A split-level step under a metre is still the object's storey.
    assert rule.record_stop(5, np.array([-1.2, 0.7, 4.6]), np.array([-1.0, 1.0, 4.0])) is True
    assert rule.attempts[-1]["on_target_floor"] is True


def test_without_a_target_floor_the_rule_stays_horizontal_and_says_so():
    rule = ObjectDistanceRule(enabled=True, max_floor_dy_m=1.0)
    assert rule.record_stop(1, np.array([0.5, 3.0, 0.0]), np.array([0.0, 0.0, 0.0])) is True
    assert rule.attempts[-1]["on_target_floor"] is None


def test_zero_tolerance_is_the_horizontal_only_rule():
    rule = ObjectDistanceRule(enabled=True, max_floor_dy_m=0.0)
    rule.target_floor_y = 0.0
    assert rule.record_stop(1, np.array([0.5, 3.0, 0.0]), np.array([0.0, 0.0, 0.0])) is True
    assert "storey" not in rule.definition()


def test_the_shipped_default_checks_the_storey():
    from osg.core.config import OSGConfig
    assert OSGConfig().ycb.object_success_max_floor_dy_m == 1.0


def test_target_floor_comes_from_the_manifest_or_its_first_view_point():
    from osg.sim.ycb_env import _target_floor_y
    assert _target_floor_y({"destination_floor_y": 3.16}) == 3.16
    assert _target_floor_y({"viewpoints": [{"position": [1.0, 0.16, 2.0]}]}) == 0.16
    assert _target_floor_y({}) is None


def test_the_terminal_stop_at_the_same_step_is_one_attempt():
    rule = ObjectDistanceRule(enabled=True)
    target = np.array([0.0, 0.0, 0.0])
    rule.record_stop(7, np.array([0.5, 0.0, 0.0]), target)
    rule.record_stop(7, np.array([0.5, 0.0, 0.0]), target)
    assert len(rule.attempts) == 1


def test_success_needs_a_stop_within_the_first_three_attempts():
    rule = ObjectDistanceRule(enabled=True)
    target = np.array([0.0, 0.0, 0.0])
    for step in (1, 2, 3):
        rule.record_stop(step, np.array([5.0, 0.0, 0.0]), target)
    rule.record_stop(4, np.array([0.2, 0.0, 0.0]), target)  # a fourth stop, right there
    summary = rule.summary(np.array([0.2, 0.0, 0.0]), target)
    assert summary["success"] == 0 and summary["attempt_count"] == 4
    assert abs(summary["final_distance_horizontal_m"] - 0.2) < 1e-9


def test_spl_uses_the_geodesic_to_the_object_and_the_distance_travelled():
    rule = ObjectDistanceRule(enabled=True)
    rule.shortest_m = 4.0
    for x in (0.0, 2.0, 4.0, 6.0, 8.0):  # 8 m walked, twice the shortest
        rule.travelled(np.array([x, 0.0, 0.0]))
    rule.record_stop(5, np.array([8.0, 0.0, 0.0]), np.array([8.5, 0.0, 0.0]))
    summary = rule.summary(np.array([8.0, 0.0, 0.0]), np.array([8.5, 0.0, 0.0]))
    assert summary["success"] == 1 and abs(summary["spl"] - 0.5) < 1e-9
    assert summary["shortest_path_to_object_m"] == 4.0


def _fake_env(verdict, geodesic_m):
    """An env whose `attempt_scored` answers `verdict` and whose habitat
    fallback would find a view point `geodesic_m` away."""
    class _PF:
        def find_path(self, path):
            path.geodesic_distance = geodesic_m
            return True
    vp = SimpleNamespace(agent_state=SimpleNamespace(position=np.zeros(3, dtype=np.float32)))
    return SimpleNamespace(
        attempt_scored=lambda frame, cfg: verdict,
        current_episode=SimpleNamespace(goals=[SimpleNamespace(view_points=[vp])]),
        env=SimpleNamespace(sim=SimpleNamespace(
            pathfinder=_PF(),
            get_agent_state=lambda: SimpleNamespace(position=np.zeros(3, dtype=np.float32)),
        )),
    )


def test_a_none_verdict_falls_through_to_the_viewpoint_rule():
    """An env that has the hook but has it switched off must not turn every
    attempt into a failure."""
    cfg = SimpleNamespace(agent=SimpleNamespace(success_distance=0.18))
    assert attempt_succeeded(_fake_env(None, geodesic_m=0.1), SimpleNamespace(), cfg) is True
    assert attempt_succeeded(_fake_env(None, geodesic_m=0.5), SimpleNamespace(), cfg) is False
    # A real verdict is final, whatever habitat would have said.
    assert attempt_succeeded(_fake_env(False, geodesic_m=0.1), SimpleNamespace(), cfg) is False
    assert attempt_succeeded(_fake_env(True, geodesic_m=9.0), SimpleNamespace(), cfg) is True


def test_the_authored_env_overrides_step_for_the_terminal_stop_and_travel():
    """The step override was once lost to an editing slip and SPL came out 1.0
    on every success (no travel accumulated) with the relocation hook gone."""
    from osg.sim.habitat_env import HabitatObjectNavEnv
    from osg.sim.ycb_env import YCBAuthoredNavEnv
    assert YCBAuthoredNavEnv.step is not HabitatObjectNavEnv.step
    import inspect
    src = inspect.getsource(YCBAuthoredNavEnv.step)
    assert "_record_stop" in src and "_maybe_relocate" in src and "travelled" in src
