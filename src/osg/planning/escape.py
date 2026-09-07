"""Anti-deadlock guard on the executed action stream.

Port of ASCENT's action-history override (`ascent/ascent_policy.py:595-606`).
A reactive mover with no global plan has two failure modes that look nothing
alike from inside the control loop but are both terminal: spinning on the spot,
and grinding into an obstacle the map cannot see. Both are invisible to the
policy itself -- it keeps choosing a locally sensible action -- and both are
obvious in the last thirty actions.

This is what ASCENT uses instead of the costmap's stuck detector
(`planning/controller.py:observe_progress`), which needs a map to mark.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

FORWARD = "move_forward"
TURN_LEFT = "turn_left"
TURN_RIGHT = "turn_right"
TURNS = (TURN_LEFT, TURN_RIGHT)


class ActionHistoryEscape:
    """Rewrite the next action when the recent history is degenerate.

    all turns    -> force forward   (spinning in place)
    all forwards -> force turn right (pushing into something unmapped)
    """

    def __init__(self, window: int = 30) -> None:
        self.window = int(window)
        self._history: Deque[str] = deque(maxlen=self.window)
        self.n_forced_forward = 0
        self.n_forced_turn = 0

    def reset(self) -> None:
        self._history.clear()

    def __call__(self, action: str) -> str:
        """Return the action to execute, and record it."""
        forced = self._override()
        if forced is not None:
            action = forced
        self._history.append(action)
        return action

    def _override(self) -> Optional[str]:
        if len(self._history) < self.window:
            return None
        if all(a in TURNS for a in self._history):
            self.n_forced_forward += 1
            return FORWARD
        if all(a == FORWARD for a in self._history):
            self.n_forced_turn += 1
            return TURN_RIGHT
        return None
