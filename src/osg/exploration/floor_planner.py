"""Coarse level of the cascade: which storey is the target on?

ASCENT asks this question and, when the answer is a different floor, **never
asks the fine one** -- it returns a sentinel and heads for the stairs
(`llm_planner.py:216-236`). That pruning is the part worth copying: each level
sees only information at its own granularity, rather than one prompt carrying
the whole map.

What differs here is what the coarse level gets to look at. ASCENT describes a
floor with `this_floor_rooms` and `this_floor_objects` -- two flat sets of
strings (`object_point_cloud_map.py:44-45`), so a chair seen in five rooms is
one entry and there is no room-object containment anywhere. OSG has a real
floor -> room -> object graph, so a floor can be described by its rooms and what
is in each of them. That is the extension being measured; it is not a port,
because ASCENT has no scene graph to port from.

The decision is deliberately coarse in its effect too: it sets a DIRECTION, and
the existing stair machinery executes it. Nothing here plans a path.

Measured context for why this is worth trying at all: on `dev50` a room-typed
LLM choosing between *frontiers* is a regression (S11, net -3, override rate
49% -> 66%). But 86% of `dev50` episodes never change floor, so that experiment
used the model for the one job ASCENT does not give it. This gives it the job it
does have.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ..graph.scene_graph import SceneGraph
from ..llm import prompts
from ..llm.client import ChatClient

log = logging.getLogger(__name__)

# ascent/constants.py:237-238
ASK_EVERY_STEPS = 60
MIN_STEPS_ON_FLOOR = 100


def _pct_lines(probs: Dict, key_fmt) -> str:
    """`        "Floor 2": 30.0%`, descending, ties broken by name."""
    if not probs:
        return ' ' * 8 + '"Unknown": 0.0%'
    ordered = sorted(probs.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return ',\n'.join(
        f'{" " * 8}"{key_fmt(k)}": {v * 100.0:.1f}%' for k, v in ordered
    )


def describe_floor(sg: SceneGraph, floor_key: int, order: int, is_current: bool,
                   explored: bool, max_rooms: int = 6, max_objects: int = 10) -> str:
    """One floor, described through the room level rather than as a flat set.

    Mirrors the shape of ASCENT's floor description (`llm_planner.py:538-541`),
    including the "already explored" clause, which is what lets the model rule a
    storey out rather than merely rank it.
    """
    rooms = [r for r in sg.rooms.values() if r.floor == floor_key and r.label]
    rooms = sorted(rooms, key=lambda r: -r.n_cells)[:max_rooms]
    objects = [o for o in sg.objects if o.floor == floor_key]

    room_names = ", ".join(dict.fromkeys(r.label.replace("_", " ") for r in rooms))
    obj_names = ", ".join(dict.fromkeys(o.label for o in objects))[:400]
    status = "Current floor" if is_current else "Other floor"
    text = (f'{status}. There are room types: {room_names or "unknown rooms"}, '
            f'containing objects: {obj_names or "unknown objects"}')
    if explored:
        text += ". You do not need to explore this floor again"
    return text


class FloorDecisionPlanner:
    def __init__(
        self,
        client: ChatClient,
        floor_prior=None,
        kg=None,
        ask_every_steps: int = ASK_EVERY_STEPS,
        min_steps_on_floor: int = MIN_STEPS_ON_FLOOR,
    ) -> None:
        self.client = client
        self.floor_prior = floor_prior
        self.kg = kg
        self.ask_every_steps = ask_every_steps
        self.min_steps_on_floor = min_steps_on_floor
        self._last_ask = -10_000
        self.asks = 0
        self.moves = 0  # decisions that named a floor other than the current one
        # Why the gate refused, so an inert arm is distinguishable from an
        # ineffective one -- the failure mode that let S10 and S11 ship results
        # about mechanisms that had barely run.
        self.blocked_one_floor = 0
        self.blocked_throttle = 0
        self.blocked_too_soon = 0

    def reset(self) -> None:
        self._last_ask = -10_000
        self.asks = 0
        self.moves = 0
        self.blocked_one_floor = 0
        self.blocked_throttle = 0
        self.blocked_too_soon = 0

    def should_ask(self, floors, step: int, has_up: bool, has_down: bool) -> bool:
        """ASCENT's gate (llm_planner.py:217), plus its notion of how many
        floors exist.

        ASCENT allocates a storey the moment it DETECTS a staircase, not when it
        reaches one: seeing an up-flight creates the floor above and seeing a
        down-flight creates the floor below (map_controller.py:530-537), so
        `floor_num` counts INFERRED floors. OSG's FloorStack allocates on
        observed height, with hysteresis, precisely so a climb does not leave a
        phantom floor behind -- correct for mapping, but it means a question
        about "which floor" could only ever be asked after the climb it was
        supposed to motivate. A smoke run showed exactly that: 1 of 4 cross-floor
        episodes ever reached the model.

        So candidate floors = the ones stood on, plus the ones a detected
        staircase implies.
        """
        if floors.n_floors() + int(has_up) + int(has_down) <= 1:
            self.blocked_one_floor += 1
            return False
        if step - self._last_ask < self.ask_every_steps:
            self.blocked_throttle += 1
            return False
        if floors.current().steps_on_floor < self.min_steps_on_floor:
            self.blocked_too_soon += 1
            return False
        return True

    def decide(self, target: str, floors, sg: SceneGraph, step: int,
               has_up: bool = False, has_down: bool = False) -> Optional[int]:
        """Direction to travel: -1 down, 0 stay, +1 up. None = not asked.

        A direction rather than an absolute storey index, because that is all
        the caller can act on -- the stair machinery takes a direction -- and
        because the candidate list mixes visited floors with ones only implied
        by a staircase, where an absolute index is an easy thing to get wrong.

        Every failure returns None, i.e. "keep exploring this floor", matching
        ASCENT's habit of falling back to the fine level rather than acting on a
        malformed answer (`llm_planner.py:222-223, 625-628`).
        """
        if not self.should_ask(floors, step, has_up, has_down):
            return None
        self._last_ask = step

        # Candidate storeys, lowest first: an implied one below, the visited
        # ones, an implied one above. Implied floors have no map, and
        # describe_floor renders them as "unknown rooms / unknown objects" --
        # which is the honest description and what makes them selectable at all.
        visited = floors.layers()
        cur_order = floors.order_of(floors.current().key)
        entries = [(l.key, l.explored) for l in visited]
        offset = 0
        if has_down:
            entries.insert(0, (None, False))
            offset = 1
        if has_up:
            entries.append((None, False))
        cur_idx = cur_order + offset

        descriptions = ',\n'.join(
            f'{" " * 8}"Floor {i + 1}": '
            f'"{describe_floor(sg, key, i, i == cur_idx, explored) if key is not None else describe_floor(sg, -999, i, False, False)}"'
            for i, (key, explored) in enumerate(entries)
        )
        floor_probs = (self.floor_prior.probabilities(target, len(entries))
                       if self.floor_prior is not None else {})
        room_probs = (self.kg.room_probabilities(target)
                      if self.kg is not None else {})

        user = prompts.FLOOR_DECISION_USER.format(
            example=prompts.FLOOR_DECISION_EXAMPLE,
            target=target,
            floor_priors=_pct_lines(floor_probs, lambda k: f"Floor {int(k) + 1}"),
            room_priors=_pct_lines(room_probs, lambda k: str(k).replace("_", " ").capitalize()),
            floors=descriptions,
        )
        try:
            self.asks += 1
            resp = self.client.chat(prompts.FLOOR_DECISION_SYSTEM, user)
            idx = int(str(resp.get("Index", "0")).strip()) - 1  # 1-based
        except Exception as e:  # noqa: BLE001
            log.warning("floor planner failed, staying on this floor: %s", e)
            return None
        if not (0 <= idx < len(entries)):
            log.warning("floor planner returned out-of-range floor %d", idx + 1)
            return None
        direction = (idx > cur_idx) - (idx < cur_idx)
        if direction:
            self.moves += 1
        return direction
