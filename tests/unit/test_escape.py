from __future__ import annotations

from osg.planning.escape import ActionHistoryEscape

TURN_L, TURN_R, FWD, STOP = "turn_left", "turn_right", "move_forward", "stop"


def _feed(esc, action, n):
    out = None
    for _ in range(n):
        out = esc(action)
    return out


def test_spinning_is_broken_by_a_forced_forward():
    esc = ActionHistoryEscape(window=30)
    assert _feed(esc, TURN_L, 30) == TURN_L  # the window only just filled
    assert esc(TURN_L) == FWD
    assert esc.n_forced_forward == 1


def test_mixed_turns_still_count_as_spinning():
    esc = ActionHistoryEscape(window=4)
    for a in (TURN_L, TURN_R, TURN_L, TURN_R):
        esc(a)
    assert esc(TURN_L) == FWD


def test_grinding_forward_is_broken_by_a_forced_turn():
    esc = ActionHistoryEscape(window=4)
    _feed(esc, FWD, 4)
    assert esc(FWD) == TURN_R
    assert esc.n_forced_turn == 1


def test_a_mixed_history_passes_through():
    esc = ActionHistoryEscape(window=4)
    for a in (FWD, TURN_L, FWD, TURN_R):
        esc(a)
    assert esc(TURN_L) == TURN_L
    assert esc.n_forced_forward == 0 and esc.n_forced_turn == 0


def test_the_forced_action_is_what_gets_remembered():
    """The override must enter the history, or the guard fires every step
    forever once a degenerate window forms."""
    esc = ActionHistoryEscape(window=4)
    _feed(esc, TURN_L, 4)
    assert esc(TURN_L) == FWD
    assert esc(TURN_L) == TURN_L  # the forced FORWARD broke the run


def test_reset_forgets_the_history():
    esc = ActionHistoryEscape(window=4)
    _feed(esc, TURN_L, 4)
    esc.reset()
    assert esc(TURN_L) == TURN_L


def test_stop_is_not_special_cased_here():
    """The DONE exemption lives in NavAgent.act, not in the guard: a STOP that
    is NOT the terminal one should still be able to break a spin."""
    esc = ActionHistoryEscape(window=4)
    _feed(esc, TURN_L, 4)
    assert esc(STOP) == FWD
