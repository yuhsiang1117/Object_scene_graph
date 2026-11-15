"""`navigation/search/stop_geometry.py` and the sampling half of
scripts/eval_stop_geometry.py (the half that needs no simulator)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from navigation.search.stop_geometry import horizontal_radius, stop_point

_spec = importlib.util.spec_from_file_location(
    "eval_stop_geometry", Path(__file__).resolve().parents[2] / "scripts" / "eval_stop_geometry.py")
esg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(esg)


def test_radius_is_the_largest_semi_axis_and_zero_when_unreadable():
    assert horizontal_radius([0.2, 0.9, 0.4]) == pytest.approx(0.9)
    assert horizontal_radius([-0.5, 0.1, 0.1]) == pytest.approx(0.5)
    assert horizontal_radius(None) == 0.0
    assert horizontal_radius([float("nan"), 1.0, 1.0]) == 0.0


def test_stop_point_backs_off_radius_plus_standoff_toward_the_agent():
    p = stop_point([0.0, 0.0], 0.5, [4.0, 0.0], 0.75)
    assert p == pytest.approx([1.25, 0.0])
    p = stop_point([1.0, 1.0], 0.0, [1.0, 5.0], 0.75)
    assert p == pytest.approx([1.0, 1.75])


def test_a_centre_error_moves_the_stop_and_a_size_error_moves_it_along_the_ray():
    agent = [5.0, 0.0]
    true = stop_point([0.0, 0.0], 0.4, agent, 0.75)
    # Off by 0.3 m along the ray: the stop moves 0.3 m.
    assert stop_point([-0.3, 0.0], 0.4, agent, 0.75) - true == pytest.approx([-0.3, 0.0])
    # Off by 0.3 m across it: the ray swings with the centre, so the stop
    # moves 0.3 * (1 - 1.15 / D).
    off = stop_point([0.0, 0.3], 0.4, agent, 0.75)
    d = float(np.hypot(5.0, 0.3))
    assert np.linalg.norm(off - true) == pytest.approx(0.3 * (1 - 1.15 / d), abs=0.01)
    # The size is double: the stop sits that much further back, same bearing.
    big = stop_point([0.0, 0.0], 0.8, agent, 0.75)
    assert big - true == pytest.approx([0.4, 0.0])


def test_an_agent_inside_the_standoff_is_not_sent_backwards():
    assert stop_point([0.0, 0.0], 0.5, [0.6, 0.0], 0.75) == pytest.approx([0.6, 0.0])
    assert stop_point([2.0, 3.0], 0.5, [2.0, 3.0], 0.75) == pytest.approx([2.0, 3.0])


# ------------------------------------------------- scripts/eval_stop_geometry

def _goal(x, y_plane, height=0.4):
    pos = np.array([x, height, -y_plane])
    return {"pos": pos, "xy": esg.plane(pos), "y": height, "object_id": 0}


def _step(n, xy, yaw=0.0, try_nav=False, goal_xy=None, h=0.88):
    return {"n": n, "xy": list(xy), "yaw": yaw, "h": h, "try_nav": try_nav, "goal_xy": goal_xy}


def test_plane_is_x_and_minus_z():
    assert esg.plane([1.0, 9.0, 2.0]) == pytest.approx([1.0, -2.0])


def test_departure_is_the_first_commit_to_that_instance_else_the_first_sighting():
    near, far = _goal(3.0, 0.0), _goal(-6.0, 0.0)
    trace = [
        _step(0, (0.0, 0.0), yaw=np.pi),                                   # facing away: not in view
        _step(1, (0.5, 0.0), yaw=0.0),                                     # in view of `near`
        _step(2, (1.0, 0.0), try_nav=True, goal_xy=[9.0, 9.0]),            # a commit to something else
        _step(3, (1.5, 0.0), try_nav=True, goal_xy=[2.8, 0.2]),            # the commit to `near`
        _step(4, (2.0, 0.0), try_nav=True, goal_xy=[2.8, 0.2]),
    ]
    out = esg.departures(trace, [near, far])
    assert out == [{"goal": 0, "step": 3, "kind": "commit"}]
    # No commit: the first step the instance is inside 3 m and +-40 deg.
    out = esg.departures(trace[:3], [near, far])
    assert out == [{"goal": 0, "step": 1, "kind": "sight"}]


def test_a_commit_belongs_to_the_nearest_instance_and_only_on_its_storey():
    a, b = _goal(3.0, 0.0), _goal(3.0, 1.0)
    trace = [_step(0, (0.0, 0.0), yaw=np.pi, try_nav=True, goal_xy=[3.0, 0.9])]
    assert esg.departures(trace, [a, b]) == [{"goal": 1, "step": 0, "kind": "commit"}]
    # The same goal point seen from the storey above is not a commit to it.
    upstairs = [_step(0, (0.0, 0.0), yaw=np.pi, try_nav=True, goal_xy=[3.0, 0.9], h=0.88 + 3.0)]
    assert esg.departures(upstairs, [a, b]) == []


def _track(x, y_plane, height=0.4, axes=(0.3, 0.2, 0.5)):
    return {"id": 1, "center": [x, height, -y_plane], "axes": list(axes)}


def test_track_near_takes_the_closest_within_reach_on_the_same_storey():
    g = _goal(3.0, 0.0)
    close, closer = _track(3.5, 0.0), _track(3.1, 0.0)
    assert esg.track_near([close, closer], g) is closer
    assert esg.track_near([_track(5.0, 0.0)], g) is None                   # 2 m away
    assert esg.track_near([_track(3.1, 0.0, height=3.5)], g) is None       # the storey above


def test_track_at_reads_the_keyframe_at_the_departure_and_flags_a_later_one():
    g = _goal(3.0, 0.0)
    early, late, last = _track(3.4, 0.0), _track(3.2, 0.0), _track(3.05, 0.0)
    log = [{"frame_id": 5, "tracks": [early]}, {"frame_id": 20, "tracks": [late]},
           {"frame_id": 40, "tracks": [last]}]
    at, when, final = esg.track_at(log, g, 25)
    assert at is late and when == "commit" and final is last
    # Nothing near the instance yet at the departure: the first keyframe that has it.
    log[0]["tracks"] = []
    at, when, _ = esg.track_at(log, g, 10)
    assert at is late and when == "late"
    assert esg.track_at([], g, 10) == (None, None, None)


def test_geometry_of_a_track_is_its_plane_centre_and_largest_axis():
    gm = esg.geometry_of(_track(3.0, -2.0, axes=(0.3, 0.2, 0.7)))
    assert gm["centre"] == pytest.approx([3.0, -2.0]) and gm["radius"] == pytest.approx(0.7)


def test_shifted_centres_sit_on_a_ring():
    ring = esg.shifted(np.array([1.0, 2.0]), 0.3)
    assert len(ring) == esg.DIRECTIONS
    assert [float(np.linalg.norm(c - [1.0, 2.0])) for c in ring] == pytest.approx([0.3] * esg.DIRECTIONS)
    assert np.mean(ring, axis=0) == pytest.approx([1.0, 2.0])


def test_mcnemar_exact_is_the_two_sided_binomial():
    assert esg.mcnemar_exact(0, 0) == 1.0
    assert esg.mcnemar_exact(5, 5) == 1.0
    assert esg.mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    assert esg.mcnemar_exact(1, 7) == pytest.approx(2 * (1 + 8) / 256)


def test_habitat_rotation_faces_the_trace_yaw():
    # Habitat's agent looks down -z; the trace's plane is (x, -z), yaw counter-clockwise.
    for yaw in (0.0, np.pi / 2, np.pi, -np.pi / 3, 2.1):
        x, y, z, w = esg.habitat_rotation(yaw)
        assert (x, z) == (0.0, 0.0) and y * y + w * w == pytest.approx(1.0)
        phi = 2.0 * np.arctan2(y, w)                      # a rotation about +y
        forward = np.array([-np.sin(phi), 0.0, -np.cos(phi)])
        assert esg.plane(forward) == pytest.approx([np.cos(yaw), np.sin(yaw)], abs=1e-9)
