"""The down-look stair probe (S32).

Down stairs are pure geometry -- a point that back-projects below the standing
floor is a hole in it -- so the only question is whether the hole is inside the
frame. At 0.88 m and 79 degrees a level camera stops seeing floor about two
metres out, and a stairwell beyond that is never sampled.

Strictly DOWN. S14a already measured the up probe and killed it (up-stair recall
19% level -> 0% at +30 deg), so nothing here tilts up to search; `look_up` exists
only to undo a `look_down`.
"""
from __future__ import annotations

import numpy as np

from osg.agent.nav_agent import State

from .test_nav_agent import _frame, make_agent, make_cfg


def _agent(every=10, **over):
    a = make_agent(make_cfg(down_look_every=every, **over))
    a.state = State.EXPLORE
    return a


def _run(agent, n, start=0):
    """Drive `act` for n steps and return the action sequence."""
    return [agent.act(_frame([0.0, 0.0], frame_id=start + i)) for i in range(n)]


def test_disabled_by_default_emits_no_pitch_actions():
    a = make_agent(make_cfg())
    assert a._down_look_every == 0
    acts = _run(a, 25)
    assert "look_down" not in acts and "look_up" not in acts


def test_probe_fires_on_the_interval():
    a = _agent(every=5)
    acts = _run(a, 22)
    assert acts.count("look_down") >= 2, acts
    assert a.stats["down_look"] >= 2


def test_every_look_down_is_followed_immediately_by_a_look_up():
    """The camera must never be left tilted -- the agent would drive with a
    depth image aimed at the floor."""
    a = _agent(every=4)
    acts = _run(a, 30)
    downs = [i for i, x in enumerate(acts) if x == "look_down"]
    assert downs, acts
    for i in downs:
        assert i + 1 < len(acts), "a look_down must not be the last action"
        assert acts[i + 1] == "look_up", f"step {i+1} was {acts[i+1]!r}, not look_up"
    assert acts.count("look_up") == acts.count("look_down")


def test_agent_never_moves_while_tilted():
    a = _agent(every=4)
    acts = _run(a, 30)
    for i, x in enumerate(acts):
        if x == "look_down":
            assert acts[i + 1] not in ("move_forward", "turn_left", "turn_right", "stop")


def test_probe_is_suppressed_in_committed_states():
    """APPROACH, VERIFYING and CLIMB read the live frame for their own
    decisions; a frame pointed at the floor would corrupt them."""
    for st in (State.APPROACH, State.VERIFYING, State.CLIMB):
        a = _agent(every=1)
        a.state = st
        assert a._down_look(_frame([0.0, 0.0]), a.floors.current(), False) is None


def test_restore_wins_even_in_a_committed_state():
    """If the state changed while tilted, the restore must still happen."""
    a = _agent(every=1)
    a._pitch_ticks = 1
    a.state = State.APPROACH
    assert a._down_look(_frame([0.0, 0.0]), a.floors.current(), False) == "look_up"
    assert a._pitch_ticks == 0


def test_tilted_frame_feeds_the_stair_detector_without_a_detector_call():
    """The tilted frame is read directly rather than via _on_keyframe, so the
    probe cannot depend on a 30 deg tilt happening to count as a keyframe."""
    a = _agent(every=1)
    seen = []

    class _Stairs:
        def accumulate(self, frame, layer, dets, seg_stair_mask=None):
            seen.append(dets)

        def disable(self, *_a):
            pass

    a.stair_detector = _Stairs()
    a._pitch_ticks = 1
    assert a._down_look(_frame([0.0, 0.0]), a.floors.current(), False) == "look_up"
    assert seen == [None], "down-stair geometry only; no detections passed"


def test_off_map_frames_are_not_accumulated():
    """A frame captured between storeys belongs to no floor's grid."""
    a = _agent(every=1)
    seen = []

    class _Stairs:
        def accumulate(self, frame, layer, dets, seg_stair_mask=None):
            seen.append(dets)

        def disable(self, *_a):
            pass

    a.stair_detector = _Stairs()
    a._pitch_ticks = 1
    assert a._down_look(_frame([0.0, 0.0]), a.floors.current(), True) == "look_up"
    assert seen == []


def test_probe_never_tilts_up_to_search():
    """S14a: tilting up moved up-stair recall 19% -> 0%. look_up must only ever
    be a restore, never a probe of its own."""
    a = _agent(every=3)
    acts = _run(a, 30)
    ups = [i for i, x in enumerate(acts) if x == "look_up"]
    for i in ups:
        assert acts[i - 1] == "look_down", "a look_up not preceded by look_down is a search tilt"
