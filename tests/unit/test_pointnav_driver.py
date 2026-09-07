"""The sensor-only mover's geometry, which is where it can go silently wrong.

A mirrored heading convention produces an agent that turns confidently in the
wrong direction and still looks, frame by frame, like it is navigating. These
tests pin the convention against the SAME fixture the costmap controller is
pinned against (test_controller.py), so the two movers cannot drift apart.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.planning.controller import agent_heading
from osg.planning.pointnav_driver import PointNavDriver, rho_theta, to_ccw_frame

from .conftest import make_camera, make_frame


class _StubPolicy:
    """Records what it was asked and replies with a fixed action."""

    num_recurrent_layers = 4

    def __init__(self, action: int = 1) -> None:
        self.action = action
        self.calls: list = []

    def to(self, device):
        return self

    def act(self, obs, hidden, prev_actions, masks, deterministic=True):
        import torch

        self.calls.append(
            {
                "rho": float(obs["pointgoal_with_gps_compass"][0, 0]),
                "theta": float(obs["pointgoal_with_gps_compass"][0, 1]),
                "depth": obs["depth"],
                "mask": bool(masks.view(-1)[0]),
                "prev_action": int(prev_actions.view(-1)[0]),
            }
        )
        return torch.tensor([[self.action]], dtype=torch.long), hidden


def _driver(monkeypatch, action: int = 1, **kwargs) -> tuple:
    policy = _StubPolicy(action)
    monkeypatch.setattr(
        "osg.planning.pointnav_driver.load_pointnav_policy", lambda _p: policy
    )
    return PointNavDriver("unused.pth", device="cpu", **kwargs), policy


# ------------------------------------------------------------------ geometry


def test_ccw_frame_flips_the_second_axis():
    assert np.allclose(to_ccw_frame(np.array([2.0, 3.0])), [2.0, -3.0])


def test_rho_theta_is_ccw_positive():
    # Facing +x in the CCW frame; goal to the left (+y) must be a positive
    # theta, i.e. "turn left by this much".
    rho, theta = rho_theta(np.zeros(2), 0.0, np.array([0.0, 1.0]))
    assert rho == pytest.approx(1.0)
    assert theta == pytest.approx(np.pi / 2)


def test_theta_sign_matches_the_costmap_controller(monkeypatch):
    """The convention pin.

    test_controller.py asserts that, facing world +x, a waypoint at +z (plane
    axis 1) requires TURN_RIGHT. The pointgoal handed to the network must agree:
    a right turn is a NEGATIVE theta.
    """
    driver, policy = _driver(monkeypatch)
    T = make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0])  # facing world +x
    assert agent_heading(T) == pytest.approx(0.0)
    driver.observe(make_frame(_intr(), T))

    driver(np.array([0.0, 3.0]))  # goal at +z -> TURN_RIGHT in test_controller
    assert policy.calls[-1]["theta"] < 0

    driver(np.array([0.0, -3.0]))  # goal at -z -> TURN_LEFT
    assert policy.calls[-1]["theta"] > 0


def test_rho_is_the_straight_line_distance(monkeypatch):
    driver, policy = _driver(monkeypatch)
    T = make_camera([1.0, 0.88, 2.0], [2.0, 0.88, 2.0])
    driver.observe(make_frame(_intr(), T))
    driver(np.array([4.0, 6.0]))
    assert policy.calls[-1]["rho"] == pytest.approx(5.0)


# ------------------------------------------------------- arrival / giving up


def test_inside_stop_radius_reports_arrival(monkeypatch):
    driver, policy = _driver(monkeypatch, stop_radius=0.9)
    T = make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0])
    driver.observe(make_frame(_intr(), T))
    assert driver(np.array([0.5, 0.0])) is None
    assert policy.calls == []  # never even asked the network


def test_creep_forces_forward_between_arrival_and_the_creep_radius(monkeypatch):
    """The approach band: arrive tight, creep in between, ask the network beyond.

    Arrival is tested FIRST. That ordering is the S40 fix -- with the creep
    checked first, a 1.0 m creep radius swallowed the 0.9 m arrival and the
    driver could never report arriving at all, which is why the pointnav arm
    produced `path_consumed` zero times against the navmesh arm's 46.
    """
    driver, policy = _driver(monkeypatch, stop_radius=0.9)
    T = make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0])
    driver.observe(make_frame(_intr(), T))
    # inside the arrival radius -> arrived, network never consulted
    assert driver(np.array([0.2, 0.0]), stop_radius=0.3, creep_below=1.0) is None
    assert policy.calls == []
    # between arrival and creep -> forced forward
    assert driver(np.array([0.6, 0.0]), stop_radius=0.3, creep_below=1.0) == "move_forward"
    assert policy.calls == []
    # beyond the creep radius -> the network decides
    driver(np.array([4.0, 0.0]), stop_radius=0.3, creep_below=1.0)
    assert len(policy.calls) == 1


def test_arrival_radius_zero_never_arrives(monkeypatch):
    """The default. 0 reproduces the signal-less behaviour every pre-S40 number
    was measured on: the creep runs all the way in and nothing concludes."""
    driver, _ = _driver(monkeypatch, stop_radius=0.9)
    T = make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0])
    driver.observe(make_frame(_intr(), T))
    for d in (0.05, 0.2, 0.9):
        assert driver(np.array([d, 0.0]), stop_radius=0.0, creep_below=1.0) == "move_forward"


def test_policy_stop_while_far_reports_no_path(monkeypatch):
    """A STOP from the network outside the goal radius means "I can't get
    there" -- the caller must retire the goal, not stop the episode."""
    driver, _ = _driver(monkeypatch, action=0)
    T = make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0])
    driver.observe(make_frame(_intr(), T))
    assert driver(np.array([5.0, 0.0])) is None


def test_action_index_maps_to_habitat_names(monkeypatch):
    T = make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0])
    for idx, name in [(1, "move_forward"), (2, "turn_left"), (3, "turn_right")]:
        driver, _ = _driver(monkeypatch, action=idx)
        driver.observe(make_frame(_intr(), T))
        assert driver(np.array([5.0, 0.0])) == name


# ------------------------------------------------------------ recurrent state


def test_goal_change_resets_the_recurrent_state(monkeypatch):
    driver, policy = _driver(monkeypatch)
    T = make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0])
    driver.observe(make_frame(_intr(), T))

    driver(np.array([5.0, 0.0]))
    assert policy.calls[-1]["mask"] is False  # first step of this goal
    driver(np.array([5.0, 0.0]))
    assert policy.calls[-1]["mask"] is True  # same goal: memory carries over
    driver(np.array([5.02, 0.0]))  # 2 cm: noise, not a new goal
    assert policy.calls[-1]["mask"] is True
    driver(np.array([5.0, 4.0]))  # a real change
    assert policy.calls[-1]["mask"] is False


def test_reset_clears_the_last_goal(monkeypatch):
    driver, policy = _driver(monkeypatch)
    T = make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0])
    driver.observe(make_frame(_intr(), T))
    driver(np.array([5.0, 0.0]))
    driver(np.array([5.0, 0.0]))
    assert policy.calls[-1]["mask"] is True
    driver.reset()
    driver.observe(make_frame(_intr(), T))
    driver(np.array([5.0, 0.0]))
    assert policy.calls[-1]["mask"] is False


def test_acting_without_observe_is_an_error(monkeypatch):
    driver, _ = _driver(monkeypatch)
    with pytest.raises(RuntimeError):
        driver(np.array([5.0, 0.0]))


# -------------------------------------------------------------------- depth


def test_depth_is_normalised_and_resized(monkeypatch):
    driver, policy = _driver(monkeypatch, depth_min_m=0.5, depth_max_m=5.0)
    T = make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0])
    driver.observe(make_frame(_intr(), T, depth_value=5.0))
    driver(np.array([5.0, 0.0]))
    depth = policy.calls[-1]["depth"]
    assert tuple(depth.shape) == (1, 224, 224, 1)  # channels last, as trained
    assert float(depth.max()) == pytest.approx(1.0)

    driver.observe(make_frame(_intr(), T, depth_value=0.5))
    driver(np.array([5.0, 0.0]))
    assert float(policy.calls[-1]["depth"].max()) == pytest.approx(0.0)


def test_depth_out_of_range_is_clipped_not_wrapped(monkeypatch):
    """Habitat clips to [min, max] itself, but a 0.0 "no return" must land at
    0.0 normalised rather than going negative and off-distribution."""
    driver, policy = _driver(monkeypatch, depth_min_m=0.5, depth_max_m=5.0)
    T = make_camera([0.0, 0.88, 0.0], [1.0, 0.88, 0.0])
    driver.observe(make_frame(_intr(), T, depth_value=0.0))
    driver(np.array([5.0, 0.0]))
    depth = policy.calls[-1]["depth"]
    assert float(depth.min()) >= 0.0 and float(depth.max()) <= 1.0


def _intr():
    from osg.core.types import CameraIntrinsics

    return CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0, width=640, height=480)
