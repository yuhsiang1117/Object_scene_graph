"""One row per step, for asking why an episode failed rather than that it did.

Every diagnosis in docs/AB_RESULTS since S47 was reconstructed after the fact
from whatever happened to be in `episodes.jsonl` -- and the reconstructions kept
running out of road. "The detector fires on 12% of the frames where it should"
needed pose and detection count per step; "62% of the time the object is outside
the FOV" needed the same; "132 forwards produced no motion" needed the REALISED
displacement, which nothing recorded. Each answer cost a bespoke re-run.

This records those quantities as a matter of course, in six groups:

    pose        where the agent is, where it is looking, and where it GOT to
    perception  what the detector, the segmenter and the value model said
    maps        how much is known, and what the stair maps hold
    decision    which state, which frontier, which goal, and the mover's answer
    commit      the evidence behind a goal, and what the verifier did with it
    stairs      the climb state machine

The three rules that make it worth keeping:

* Realised, not commanded. `action` is what the agent asked for; `moved` is what
  the simulator gave it. A forward that moved 0 m is the single most diagnostic
  event in the log and no earlier version recorded it.
* Cheap enough to leave on. About 40 numbers a step, appended to a list, no I/O
  until the episode ends. Measured at well under 1% of step time, against a
  YOLOE forward pass.
* Ground truth is NOT in here. Goal positions and view-points are read from the
  dataset offline by `scripts/analyse_behaviour.py`. An agent that can see the
  answer is an agent that can leak it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _r(v, n=2):
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return [round(float(x), n) for x in np.asarray(v).ravel()[:3]]
    return round(float(v), n)


class BehaviourLog:
    """Per-step behaviour trace for one episode.

    The agent calls `step(...)` once per control step with whatever it knows;
    every field is optional, so an agent that has no stair machinery simply
    never passes those keys and they stay absent rather than zero -- absent and
    zero mean different things when you are counting opportunities.
    """

    VERSION = 1

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.rows: List[Dict[str, Any]] = []
        self._prev_xy: Optional[np.ndarray] = None
        self._prev_yaw: Optional[float] = None

    # ------------------------------------------------------------------ write

    def step(
        self,
        *,
        n: int,
        xy,
        yaw: float,
        height: float,
        pitch: float = 0.0,
        state: str = "",
        action: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Record one step. `xy`/`yaw` are the pose BEFORE `action` executes."""
        if not self.enabled:
            return
        xy = np.asarray(xy, dtype=float)[:2]
        row: Dict[str, Any] = {
            "n": int(n), "xy": _r(xy), "yaw": _r(yaw), "h": _r(height),
            "pitch": _r(pitch, 1), "state": str(state), "act": action,
        }
        # Realised motion, attributed to the PREVIOUS row's action: this is the
        # only place the log can tell a commanded forward from a moved one.
        if self._prev_xy is not None and self.rows:
            d = float(np.linalg.norm(xy - self._prev_xy))
            dyaw = abs((yaw - self._prev_yaw + np.pi) % (2 * np.pi) - np.pi)
            self.rows[-1]["moved"] = round(d, 3)
            self.rows[-1]["turned"] = round(float(np.degrees(dyaw)), 1)
            if self.rows[-1].get("act") == "move_forward" and d < 0.01:
                self.rows[-1]["blocked"] = 1
        self._prev_xy, self._prev_yaw = xy.copy(), float(yaw)
        for k, v in extra.items():
            if v is None:
                continue
            row[k] = _r(v) if isinstance(v, (float, np.floating, list, tuple, np.ndarray)) else v
        self.rows.append(row)

    def annotate(self, **fields: Any) -> None:
        """Attach to the row being built -- for facts a caller learns after
        `step` (the mover's answer, a verifier verdict)."""
        if self.enabled and self.rows:
            for k, v in fields.items():
                if v is not None:
                    self.rows[-1][k] = _r(v) if isinstance(v, (float, np.floating)) else v

    # ----------------------------------------------------------------- report

    def summary(self) -> Dict[str, Any]:
        """Per-episode aggregates that do not need ground truth.

        These are the numbers that turned out to matter, computed once here
        instead of by a fresh script each time.
        """
        if not self.rows:
            return {}
        acts = [r.get("act") for r in self.rows]
        moved = [r.get("moved", 0.0) or 0.0 for r in self.rows]
        fwd = [i for i, a in enumerate(acts) if a == "move_forward"]
        blocked = sum(1 for r in self.rows if r.get("blocked"))
        xy = np.array([r["xy"] for r in self.rows], dtype=float)
        det = [r.get("ndet", 0) or 0 for r in self.rows]
        return {
            "log_version": self.VERSION,
            "steps": len(self.rows),
            "path_len_m": round(float(sum(moved)), 2),
            "bbox_m": [round(float(xy[:, 0].ptp()), 2), round(float(xy[:, 1].ptp()), 2)],
            "forwards": len(fwd),
            # The wedge signature: forward commanded, nothing happened.
            "blocked_forwards": blocked,
            "blocked_frac": round(blocked / max(len(fwd), 1), 3),
            "turns": sum(1 for a in acts if a in ("turn_left", "turn_right")),
            "tilts": sum(1 for a in acts if a in ("look_up", "look_down")),
            "det_steps": sum(1 for d in det if d),
            "det_rate": round(sum(1 for d in det if d) / len(self.rows), 3),
            "states": {s: sum(1 for r in self.rows if r.get("state") == s)
                       for s in sorted({r.get("state", "") for r in self.rows}) if s},
        }
