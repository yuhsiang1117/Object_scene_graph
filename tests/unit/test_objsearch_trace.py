"""`agent.search.trace` (S83): logging-only columns on the behaviour log.

The flag adds nothing to the control flow, so what is pinned here is that the
off state writes the row it always wrote, that the storey identity survives
the index shift a storey inserted below causes, and that a sub-threshold box
is recorded without being ingested.
"""
from __future__ import annotations

import numpy as np
import pytest

from navigation.search.trace import weak_points

from .test_ascentnav_stairs import _agent, _wall_frame

NEW_KEYS = ("floor_uid", "n_floors", "has_up", "has_dn", "exp_up", "exp_dn",
            "floor_done", "nfront_live")


def _traced():
    from osg.core.config.agent import AgentSearchConfig
    return _agent(search=AgentSearchConfig(trace=True))


def test_off_the_row_is_the_row_it_always_was():
    a = _agent()
    a.act(_wall_frame([0, 0.88, 0], [1, 0.88, 0]))
    assert not set(NEW_KEYS + ("weak", "goal_xy")) & set(a.behaviour.rows[-1])


def test_on_the_row_carries_the_storey_and_frontier_state():
    a = _traced()
    a.act(_wall_frame([0, 0.88, 0], [1, 0.88, 0]))
    row = a.behaviour.rows[-1]
    for key in NEW_KEYS:
        assert key in row, f"{key} missing"
    assert row["floor_uid"] == 0 and row["n_floors"] == 1
    assert "weak" not in row and "goal_xy" not in row      # absent, not empty


def test_the_trace_does_not_change_the_action():
    f = _wall_frame([0, 0.88, 0], [1, 0.88, 0])
    a, b = _agent(), _traced()
    assert [a.act(f) for _ in range(5)] == [b.act(f) for _ in range(5)]


def test_a_storey_keeps_its_uid_when_one_is_inserted_below():
    a = _traced()
    first = a._floors[a._floor_idx]["uid"]
    a._floors.insert(0, a._new_floor())                    # what `_has_down_stair` does
    a._floor_idx += 1
    assert a._floors[a._floor_idx]["uid"] == first
    assert a._floors[0]["uid"] != first
    a.reset("chair")
    assert a._floors[0]["uid"] == 0


def test_a_weak_box_lands_where_its_depth_puts_it():
    """A box centred on the optical axis, 2 m of range, robot at (1, 2) facing
    +x in the episodic frame: the point is 2 m ahead."""
    from navigation.geometry import xyz_yaw_pitch_roll_to_tf_matrix

    h, w, lo, hi = 480, 640, 0.5, 5.0
    depth_n = np.full((h, w), (2.0 - lo) / (hi - lo), np.float32)
    tf = xyz_yaw_pitch_roll_to_tf_matrix(np.array([1.0, 2.0, 0.88]), 0.0, 0.0, 0.0)
    fx = w / (2 * np.tan(np.radians(79.0) / 2))
    weak = [("chair", 0.55, np.array([300.0, 220.0, 340.0, 260.0])),
            ("bed", 0.7, np.array([0.0, 0.0, 50.0, 50.0]))]            # not the target
    out = weak_points(weak, "chair", depth_n, tf, lo, hi, fx, fx)
    assert len(out) == 1 and out[0][0] == 0.55
    assert out[0][1:] == pytest.approx([3.0, 2.0], abs=0.02)


def test_a_weak_box_with_no_depth_keeps_only_its_score():
    out = weak_points([("chair", 0.5, np.array([10.0, 10.0, 40.0, 40.0]))], "chair",
                      np.zeros((48, 64), np.float32), np.eye(4), 0.5, 5.0, 50.0, 50.0)
    assert out == [[0.5]]


def test_dfine_records_what_it_rejects_without_masking_it(monkeypatch):
    from osg.perception.detector import DFineDetector

    det = DFineDetector(conf=0.8)
    det.set_vocabulary(["chair"])
    posts = []

    def post(url, payload):
        posts.append(url)
        if url == det.url:
            return {"boxes": [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]],
                    "logits": [0.9, 0.5], "phrases": ["chair", "chair"]}
        return None                                        # SAM: fall back to the box

    monkeypatch.setattr(det, "_post", post)
    out = det.detect(np.zeros((100, 200, 3), np.uint8))
    assert [d.score for d in out] == [0.9]
    assert len(det.last_weak) == 1
    label, score, box = det.last_weak[0]
    assert (label, score) == ("chair", 0.5)
    assert box == pytest.approx([100.0, 50.0, 140.0, 70.0])
    assert posts.count(det.sam_url) == 1, "no mask is requested for a rejected box"
    det.detect(np.zeros((100, 200, 3), np.uint8))
    assert len(det.last_weak) == 1, "per call, not accumulated"
