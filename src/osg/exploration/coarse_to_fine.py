"""ASCENT's coarse-to-fine reasoning: an LLM picks the STOREY, then the AREA.

Port of `ascent/llm_planner.py` (arXiv 2505.23019). The idea that makes it
affordable is not the prompting but the **gating**: the LLM is consulted only
when the cheap rules cannot decide. ASCENT reports 2.0-2.7 calls per episode
against 35-149 for per-step VLM frontier scoring, at higher SR.

Two levels, each a single JSON question:

  coarse (floor)  "which storey is the target on?"  -- asked only when more
                  than one storey is known, the current one has been searched
                  for `floor_min_steps_on_floor`, and we last asked at least
                  `floor_ask_interval` steps ago. It may answer "stay".
  fine (area)     "which frontier region should I explore?" -- asked only when
                  no frontier is within `nearby_m`, because a frontier you can
                  reach in three metres is not worth a network round-trip.

The two prior tables are ASCENT's, transcribed here rather than imported so the
runtime has no dependency on `relative_works/ascent/`:

  ROOM_PRIORS   P(room type | goal object), from their `knowledge_graph.json`
                (edge weights x100, over their 10 REFERENCE_ROOMS).
  FLOOR_PRIORS  P(goal object on storey k | building has N storeys), from their
                `statistic_priors/hm3d_floor_object_possibility.xlsx`, counted
                over the HM3D **train** split -- so it is a legitimate prior and
                not val leakage.

Both are keyed by our own goal-category spelling (`tv_monitor`, `plant`), which
is why the transcription is spelled out rather than looked up by their names.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..llm import prompts

log = logging.getLogger(__name__)

# ASCENT's REFERENCE_ROOMS, in their order.
REFERENCE_ROOMS = [
    "bathroom", "bedroom", "dining_room", "garage", "hall",
    "kitchen", "laundry_room", "living_room", "office", "rec_room",
]

# P(room | goal) x100. Their knowledge graph has no "potted plant" node, so
# `plant` uses the "plant" node and `tv_monitor` uses "tv", exactly the synonym
# resolution their `get_room_probabilities` does for couch/sofa.
ROOM_PRIORS: Dict[str, Dict[str, float]] = {
    "bed": {"bathroom": 4.7, "bedroom": 75.1, "dining_room": 0.2, "garage": 0.2,
            "hall": 0.7, "kitchen": 1.4, "laundry_room": 0.1, "living_room": 4.0,
            "office": 12.9, "rec_room": 0.8},
    "chair": {"bathroom": 7.7, "bedroom": 21.8, "dining_room": 1.2, "garage": 1.4,
              "hall": 3.3, "kitchen": 32.1, "laundry_room": 1.4, "living_room": 13.8,
              "office": 15.9, "rec_room": 1.4},
    "sofa": {"bathroom": 3.2, "bedroom": 10.2, "dining_room": 1.3, "garage": 0.3,
             "hall": 4.3, "kitchen": 19.4, "laundry_room": 0.0, "living_room": 50.5,
             "office": 8.1, "rec_room": 2.7},
    "toilet": {"bathroom": 97.0, "bedroom": 1.2, "dining_room": 0.0, "garage": 0.0,
               "hall": 0.2, "kitchen": 0.5, "laundry_room": 0.6, "living_room": 0.1,
               "office": 0.4, "rec_room": 0.0},
    "tv_monitor": {"bathroom": 4.2, "bedroom": 27.4, "dining_room": 3.0, "garage": 0.2,
                   "hall": 2.4, "kitchen": 23.7, "laundry_room": 0.5, "living_room": 26.4,
                   "office": 9.8, "rec_room": 2.4},
    "plant": {"bathroom": 27.0, "bedroom": 14.2, "dining_room": 1.0, "garage": 0.2,
              "hall": 4.2, "kitchen": 26.9, "laundry_room": 0.8, "living_room": 16.4,
              "office": 8.7, "rec_room": 0.6},
}

# P(goal on storey k | N storeys) x100, HM3D train. Storeys are 1-indexed and
# ordered bottom-up. Their table covers N in 2..4; above that they fall back to
# the largest N they have, which `floor_priors` reproduces.
FLOOR_PRIORS: Dict[str, Dict[int, List[float]]] = {
    "bed": {2: [41.0, 59.0], 3: [22.2, 29.6, 48.1], 4: [0.0, 0.0, 50.0, 50.0]},
    "chair": {2: [52.6, 47.4], 3: [34.3, 37.1, 28.6], 4: [40.0, 20.0, 20.0, 20.0]},
    "plant": {2: [50.0, 50.0], 3: [30.8, 30.8, 38.5], 4: [0.0, 100.0, 0.0, 0.0]},
    "sofa": {2: [55.8, 44.2], 3: [44.4, 38.9, 16.7], 4: [0.0, 50.0, 0.0, 50.0]},
    "toilet": {2: [45.3, 54.7], 3: [30.0, 36.7, 33.3], 4: [50.0, 0.0, 25.0, 25.0]},
    "tv_monitor": {2: [49.2, 50.8], 3: [25.0, 41.7, 33.3], 4: [20.0, 20.0, 20.0, 40.0]},
}


def _norm(target: str) -> str:
    return str(target).lower().strip().replace(" ", "_")


def room_priors(target: str) -> Dict[str, float]:
    """P(room | goal) x100 for every reference room; empty for an unknown goal."""
    return dict(ROOM_PRIORS.get(_norm(target), {}))


def floor_priors(target: str, total_floors: int) -> Dict[int, float]:
    """P(goal on storey k | N storeys) x100, keyed by 1-indexed storey."""
    table = FLOOR_PRIORS.get(_norm(target))
    if not table or total_floors < 1:
        return {k: 0.0 for k in range(1, max(1, total_floors) + 1)}
    if total_floors == 1:
        return {1: 100.0}
    n = total_floors if total_floors in table else max(table)
    row = table[n]
    return {k + 1: (row[k] if k < len(row) else 0.0) for k in range(total_floors)}


def _fmt_probs(probs: Dict, indent: str = " " * 8) -> str:
    """ASCENT formats priors as an indented, probability-descending JSON-ish
    block. Descending order matters: it puts the answer they want the model to
    reach for at the top of the list."""
    items = sorted(probs.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return ",\n".join(
        f'{indent}"{k.replace("_", " ").capitalize() if isinstance(k, str) else f"Floor {k}"}"'
        f": {v:.1f}%"
        for k, v in items
    )


@dataclass
class Area:
    """One frontier region as the LLM sees it."""

    room: str  # room type, or "unknown room"
    objects: Sequence[str]

    def describe(self) -> str:
        objs = ", ".join(self.objects) if self.objects else "no visible objects"
        return f"a {self.room.replace('_', ' ')} containing objects: {objs}"


@dataclass
class FloorDesc:
    """One storey as the LLM sees it."""

    index: int  # 1-indexed, bottom-up
    is_current: bool
    rooms: Sequence[str]
    objects: Sequence[str]
    fully_explored: bool = False

    def describe(self) -> str:
        status = "Current floor" if self.is_current else "Other floor"
        rooms = ", ".join(self.rooms) if self.rooms else "unknown rooms"
        objs = ", ".join(self.objects) if self.objects else "unknown objects"
        tail = ". You do not need to explore this floor again" if self.fully_explored else ""
        return f"{status}. There are room types: {rooms}, containing objects: {objs}{tail}"


class CoarseToFinePlanner:
    """Holds the LLM client and the ask-interval state for one episode.

    Every method degrades to the caller's geometric choice on any failure --
    no client, no priors, unparseable reply, out-of-range index. That matters
    more than it looks: a hosted endpoint that starts refusing mid-run must not
    change the trajectory distribution, or an A/B measures the outage.
    """

    def __init__(
        self,
        client=None,
        nearby_m: float = 3.0,
        topk: int = 3,
        floor_ask_interval: int = 60,
        floor_min_steps_on_floor: int = 100,
    ) -> None:
        self.client = client
        self.nearby_m = float(nearby_m)
        self.topk = int(topk)
        self.floor_ask_interval = int(floor_ask_interval)
        self.floor_min_steps_on_floor = int(floor_min_steps_on_floor)
        self.calls = 0
        self.errors = 0
        self.last_floor_ask_step = -10 ** 9
        self.log: List[tuple] = []

    def reset(self) -> None:
        self.calls = 0
        self.errors = 0
        self.last_floor_ask_step = -10 ** 9
        self.log = []

    # ------------------------------------------------------------------ fine

    def should_ask_area(self, best_path_cost: Optional[float], n_areas: int) -> bool:
        """ASCENT's two cheap outs: a single candidate needs no reasoning, and
        a frontier already within `nearby_m` is taken on the spot."""
        if self.client is None or n_areas < 2:
            return False
        return best_path_cost is None or best_path_cost > self.nearby_m

    def choose_area(self, target: str, areas: Sequence[Area], step: int = -1) -> int:
        """Index into `areas` of the region to explore; 0 (the geometric best)
        on any failure."""
        if self.client is None or len(areas) < 2:
            return 0
        entries = ",\n".join(
            f'{" " * 8}"Area {i + 1}": "{a.describe()}"' for i, a in enumerate(areas)
        )
        priors = room_priors(target)
        idx = self._ask(
            prompts.AREA_CHOICE_SYSTEM,
            prompts.AREA_CHOICE_USER.format(
                goal=target,
                room_priors=_fmt_probs(priors) if priors else f'{" " * 8}"Unknown": 0.0%',
                areas=entries,
            ),
            n_options=len(areas),
        )
        if idx is None:
            return 0
        # The descriptions go in the log, not just the index: whether this
        # method can work at all depends on how much context the areas carry,
        # and "unknown room containing objects: no visible objects" is a
        # different experiment from ASCENT's Places365-labelled rooms.
        self.log.append((step, "area", idx, [a.describe() for a in areas]))
        return idx

    # ---------------------------------------------------------------- coarse

    def should_ask_floor(self, step: int, n_floors: int, steps_on_floor: int) -> bool:
        if self.client is None or n_floors < 2:
            return False
        if steps_on_floor < self.floor_min_steps_on_floor:
            return False
        return step - self.last_floor_ask_step >= self.floor_ask_interval

    def choose_floor(
        self,
        target: str,
        floors: Sequence[FloorDesc],
        current_index: int,
        step: int = -1,
    ) -> int:
        """1-indexed storey to search next; `current_index` (i.e. stay) on any
        failure. Asking is what costs, so the interval is stamped here even if
        the reply is unusable."""
        if self.client is None or len(floors) < 2:
            return current_index
        self.last_floor_ask_step = step
        entries = ",\n".join(
            f'{" " * 8}"Floor {f.index}": "{f.describe()}"' for f in floors
        )
        rp = room_priors(target)
        idx = self._ask(
            prompts.FLOOR_CHOICE_SYSTEM,
            prompts.FLOOR_CHOICE_USER.format(
                goal=target,
                floor_priors=_fmt_probs(floor_priors(target, len(floors))),
                room_priors=_fmt_probs(rp) if rp else f'{" " * 8}"Unknown": 0.0%',
                floors=entries,
            ),
            n_options=len(floors),
        )
        if idx is None:
            return current_index
        self.log.append((step, "floor", idx + 1, [f.describe() for f in floors]))
        return idx + 1

    # ------------------------------------------------------------------ misc

    def _ask(self, system: str, user: str, n_options: int) -> Optional[int]:
        """Returns a 0-based index, or None if the model gave us nothing usable.
        ASCENT accepts `{"Index": "2", "Reason": ...}` with the index 1-based."""
        try:
            resp = self.client.chat(system, user)
            self.calls += 1
        except Exception as e:  # network, timeout, unparseable JSON
            self.errors += 1
            log.warning("coarse-to-fine LLM call failed: %s", e)
            return None
        raw = resp.get("Index", resp.get("index"))
        try:
            idx = int(str(raw).strip())
        except (TypeError, ValueError):
            log.warning("coarse-to-fine index not an integer: %r", raw)
            return None
        if not 1 <= idx <= n_options:
            log.warning("coarse-to-fine index %d out of range 1..%d", idx, n_options)
            return None
        return idx - 1
