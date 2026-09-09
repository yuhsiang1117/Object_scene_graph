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


class DisplacementEscape:
    """Escape a wedge, judged on what the body DID rather than what was asked.

    `ActionHistoryEscape` below reads the COMMANDED action stream and fires only
    when the last N are all turns or all forwards. Measured over 100 episodes
    (S51): 793 forwards produced no displacement, in 69 of the 100 episodes, and
    that guard fired **zero** times -- the real stream alternates turn, turn,
    blocked-forward, so neither predicate ever holds. A guard that cannot see
    the failure it exists for is not a conservative guard, it is an absent one.

    This one is fed the realised displacement. When forward has been commanded
    and gone nowhere `patience` times inside `window` steps, the agent is
    against geometry it cannot see, and the answer is to turn away decisively
    rather than keep pressing: `turn_burst` turns in one direction, which is
    what breaks the local minimum that a reactive mover cannot plan out of
    (S42: habitat's own planner escaped the same pocket in 13 steps).
    """

    def __init__(self, patience: int = 4, window: int = 12, turn_burst: int = 3) -> None:
        self.patience = int(patience)
        self.window = int(window)
        self.turn_burst = int(turn_burst)
        self.reset()

    def reset(self) -> None:
        self._recent: Deque[int] = deque(maxlen=self.window)
        self._burst = 0
        self.n_escapes = 0
        self.n_forced = 0

    def observe(self, action: Optional[str], moved: float) -> None:
        """Record what the last action actually achieved."""
        if action == FORWARD:
            self._recent.append(1 if moved < 0.01 else 0)

    def __call__(self, action: str) -> str:
        """Override `action` while an escape is in progress, or start one."""
        if self.patience <= 0:
            return action
        if self._burst > 0:
            self._burst -= 1
            self.n_forced += 1
            return TURN_RIGHT
        if sum(self._recent) >= self.patience:
            self._recent.clear()          # one escape per accumulation
            self._burst = self.turn_burst - 1
            self.n_escapes += 1
            self.n_forced += 1
            return TURN_RIGHT
        return action


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
