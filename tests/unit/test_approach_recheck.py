"""ASCENT's dense approach re-check, ported.

ASCENT scores the live frame against the target prompt on every step of the
approach and latches a flag once the score clears a threshold
(map_controller.py:770-776, reading the same cosine the value map uses at
map_controller.py:562). At the stop moment, if the flag never latched it wipes
the object clouds, disables the region and returns to exploring
(ascent_policy.py:910-922) rather than stopping on a false positive.

The first test here is the important one. A previous diagnostic in this repo
produced an all-None column across a full 100-episode run because it read a
value that was never written on the branch the evaluations actually take, and
the "SR unchanged" sanity check passed anyway. So this pins the *wiring* --
that a real `act()` step during APPROACH moves the score -- separately from the
decision rule that consumes it.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.agent.nav_agent import STOP_ACTION, NavAgent, State
from osg.exploration.async_scorer import AsyncScorer
from osg.perception.detector import StubDetector

from .conftest import make_camera, make_frame
from .test_nav_agent import _StubScorer, make_cfg


@pytest.fixture
def frame(intrinsics):
    return make_frame(intrinsics, make_camera([0.0, 0.88, 0.0], [0.0, 0.88, 3.0]),
                      depth_value=3.0)


class _StubImageText:
    """Stands in for CLIP; returns a fixed cosine and counts calls."""

    def __init__(self, score: float = 0.5) -> None:
        self.score_value = score
        self.calls = 0

    def score(self, rgb, prompts):
        self.calls += 1
        return [self.score_value] * len(prompts)


def _agent(score: float, thresh: float, armed: bool = True) -> NavAgent:
    cfg = make_cfg()
    cfg.verification.approach_recheck = armed
    cfg.verification.approach_recheck_thresh = thresh
    return NavAgent(cfg, StubDetector(), AsyncScorer(_StubScorer()), None, "chair",
                    image_text=_StubImageText(score))


# --------------------------------------------------------------- the wiring


def test_a_real_step_during_approach_moves_the_score(frame):
    """The value map runs inside act(); APPROACH steps must feed the latch."""
    agent = _agent(score=0.42, thresh=0.0)
    agent.act(frame)  # first step builds the map and sets the floor
    agent.state = State.APPROACH
    agent._approach_itm_max = 0.0
    agent.act(frame)

    assert agent.image_text.calls > 0, "the image-text model never ran"
    assert agent._approach_itm_max == pytest.approx(0.42), (
        "APPROACH steps are not reaching _update_value_map -- the latch would "
        "stay at 0.0 for every episode and reject everything"
    )


def test_the_latch_keeps_the_best_view_not_the_last(frame):
    """One good look is enough, exactly as ASCENT latches _double_check_goal."""
    agent = _agent(score=0.42, thresh=0.0)
    agent.act(frame)
    agent.state = State.APPROACH
    agent._approach_itm_max = 0.0
    agent.act(frame)
    agent.image_text.score_value = 0.01  # target now occluded
    agent.act(frame)
    assert agent._approach_itm_max == pytest.approx(0.42)


def test_each_approach_starts_from_zero(frame):
    """A fresh candidate must earn its own score, not inherit the last one's."""
    agent = _agent(score=0.42, thresh=0.0)
    agent.act(frame)
    agent._approach_itm_max = 0.9
    agent._start_approach(np.array([1.0, 1.0]))
    assert agent._approach_itm_max == 0.0


# ---------------------------------------------------------- the decision rule


def _stop_with(score: float, thresh: float, armed: bool = True):
    """Drive the agent to the stop condition and return (action, agent)."""
    agent = _agent(score=score, thresh=thresh, armed=armed)
    agent._approach_itm_max = score
    agent._approach_itm_n = 5  # the score did run; see the no-obs test below
    agent._candidate_id = 7
    agent.state = State.APPROACH
    return agent


def test_a_vouched_target_is_committed(frame, monkeypatch):
    agent = _stop_with(score=0.42, thresh=0.15)
    action = _run_stop(agent, frame, monkeypatch)
    assert action == STOP_ACTION
    assert agent.state is State.DONE
    assert agent.stats.get("recheck_pass") == 1
    assert agent.stats.get("recheck_reject", 0) == 0


def test_a_target_the_score_never_vouched_for_is_abandoned(frame, monkeypatch):
    """ASCENT: 'Might false positive, change to look for the true goal.'"""
    agent = _stop_with(score=0.02, thresh=0.15)
    blacklisted = []
    monkeypatch.setattr(agent.object_layer, "blacklist", blacklisted.append)
    action = _run_stop(agent, frame, monkeypatch)

    assert action != STOP_ACTION, "stopped on a target nothing ever vouched for"
    assert agent.state is State.EXPLORE
    assert blacklisted == [7], "the false positive was not blacklisted"
    assert agent._candidate_id is None
    assert agent.stats.get("recheck_reject") == 1


def test_disarmed_it_records_but_never_rejects(frame, monkeypatch):
    """How the calibration run is collected: the score is written to every
    episode record while the threshold is inert."""
    agent = _stop_with(score=0.02, thresh=0.15, armed=False)
    action = _run_stop(agent, frame, monkeypatch)
    assert action == STOP_ACTION
    assert agent.stats.get("recheck_reject", 0) == 0
    assert agent.approach_recheck_max == pytest.approx(0.02), (
        "the score must be recorded even when disarmed, or the threshold "
        "cannot be calibrated from a run"
    )


def _run_stop(agent, frame, monkeypatch):
    """Enter _do_approach with the target in view and the stop rule satisfied."""
    from osg.core.types import Detection

    det = Detection(label="chair", score=0.9, bbox_xyxy=(280, 200, 360, 280),
                    mask=None)
    monkeypatch.setattr(agent, "_best_target_detection", lambda f: det)
    monkeypatch.setattr(agent, "_detection_depth", lambda d, f: 0.3)
    monkeypatch.setattr(agent, "_nearest_point_stop", lambda xy: "nearest_point")
    agent.cfg.agent.terminal_rule = "nearest_point"
    return agent._do_approach(frame)


def test_every_approach_exit_is_gated():
    """The leak the smoke test found.

    Gating only the terminal rule left `retreat`, `deadline` and
    `path_consumed` free to commit STOP on a target the score had just
    rejected -- with the threshold forced to 0.99, all 12 rejections over 4
    episodes were absorbed by `path_consumed` and the agent stopped 0.02-0.05 m
    from where it would have anyway. ASCENT has no ungated stop in its approach
    (ascent_policy.py:913 is the only one), so neither may this.
    """
    import inspect

    src = inspect.getsource(NavAgent._do_approach)
    stops = src.count("return STOP_ACTION")
    gates = src.count("_recheck_rejects(")
    assert stops == gates, (
        f"_do_approach has {stops} STOP exits but {gates} re-check gates; "
        "every exit that commits STOP must consult the gate first"
    )


def test_an_approach_the_score_never_ran_on_is_not_rejected(frame, monkeypatch):
    """Absence of evidence is not evidence against.

    The value map is skipped on frames outside the current floor's plane, so an
    approach can contain no scored frame at all. Rejecting on the resulting 0.0
    would blacklist targets the gate never looked at -- and would show up in the
    calibration as a large low-scoring population that isn't real.
    """
    agent = _stop_with(score=0.0, thresh=0.15)
    agent._approach_itm_n = 0
    action = _run_stop(agent, frame, monkeypatch)
    assert action == STOP_ACTION
    assert agent.stats.get("recheck_reject", 0) == 0
    assert agent.stats.get("recheck_no_obs") == 1
