"""The FSM's alphabet: its states and the two literal actions it emits.

In its own module so the state handlers (`approach.py`, `candidate.py`) can name
a state without importing the agent that runs them.

    INIT (360 scan) -> EXPLORE <-> GOTO_FRONTIER
                           |  candidate found
                           v
                  GOTO_VERIFY_VIEW -> VERIFYING --accept--> APPROACH -> STOP
                           ^                |                  |
                           |                +--reject--> blacklist, EXPLORE
                           +---------------------(retreat if visibility lost)

Under `agent.use_habitat_navmesh` -- which every YCB run sets -- GOTO_VERIFY_VIEW
and VERIFYING are skipped: the candidate check verifies in place and goes
straight to APPROACH, mirroring the old ROS stack's /goal_object.
"""
from __future__ import annotations

from enum import Enum

STOP_ACTION = "stop"
TURN_ACTION = "turn_left"
FORWARD_ACTION = "move_forward"


class State(Enum):
    INIT = "init"
    EXPLORE = "explore"
    GOTO_FRONTIER = "goto_frontier"
    GOTO_VERIFY_VIEW = "goto_verify_view"
    VERIFYING = "verifying"
    APPROACH = "approach"
    CLIMB = "climb"
    DONE = "done"
