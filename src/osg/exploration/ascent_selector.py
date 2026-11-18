"""ASCENT's frontier selection, ported from `ascent/llm_planner.py`.

`selector.select_frontier` ranks by `score / path_cost`. ASCENT never divides by
distance: it takes the argmax of the value map, and distance enters exactly once,
as a hard shortcut for frontiers within `nearby_distance` metres. The difference
is not cosmetic -- dividing by path cost systematically favours near frontiers
and can drown a semantic signal that spans a few hundredths, which is why
`value_weight` had to be raised to 4 before the value map moved anything.

The second half ASCENT has and OSG does not is commitment. OSG re-selects every
five steps with no memory, using `continuity_weight` as a soft proxy. ASCENT
sticks to a frontier, and retires one that it keeps choosing without getting
closer, or keeps returning to. All of that bookkeeping is keyed on **position**,
because `Frontier.id` is reassigned on every extraction.

The LLM branch (`_decide_frontier_with_llm`) is deliberately not ported. S7
established that the knowledge-graph prior is the part that reaches selection,
and a synchronous model call per round would dominate a 50-episode A/B. Without
it the top-k step degenerates to "take the highest-valued frontier", which is
what this does.

**One deliberate deviation, marked here and in the A/B log**: ASCENT hands the
chosen waypoint to a learned PointNav policy, which will make progress toward
almost anything. OSG plans with A*, so an unreachable frontier would be
re-selected every round until the repeat counter retires it 20 rounds later,
burning ~100 steps. Selection therefore falls through to the next candidate when
planning fails. Reachability is the only thing this adds; the ranking is
ASCENT's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..mapping.costmap import Costmap2D
from ..mapping.frontier import Frontier
from ..planning.planner import Planner
from .selector import frontier_goal_xy

# ascent/constants.py:234-236
STICKY_DISTANCE_M = 0.3
STICKY_STEPS = 20
REPEATED_SELECTION_THRESHOLD = 20


@dataclass
class FrontierCommitState:
    """Cross-round frontier bookkeeping, keyed on position rather than id.

    `Frontier.id` is handed out fresh by every `extract` call, so an id-keyed
    set forgets everything the moment the map grows. ASCENT keys on the
    quantised waypoint (`tuple(frontier)`), which survives re-extraction as long
    as the waypoint lands in the same place -- so that is what is reproduced,
    with an explicit quantisation instead of relying on float equality.
    """

    quantise_m: float = 0.5
    sticky_distance_m: float = STICKY_DISTANCE_M
    sticky_steps: int = STICKY_STEPS
    repeat_disable: int = REPEATED_SELECTION_THRESHOLD

    disabled: Set[Tuple[int, int]] = field(default_factory=set)
    selection_count: Dict[Tuple[int, int], int] = field(default_factory=dict)
    last_xy: Optional[np.ndarray] = None
    last_distance: float = 0.0
    stick_steps: int = 0
    force_xy: Optional[np.ndarray] = None
    # ASCENT gates the nearby shortcut on this and clears it when no nearby
    # frontier exists, which forces one committed exploration pick next round
    # (llm_planner.py:98-108, 124-126).
    finish_first_explore: bool = False
    neighbor_search: bool = False

    def key(self, xy: np.ndarray) -> Tuple[int, int]:
        q = self.quantise_m
        return (int(round(float(xy[0]) / q)), int(round(float(xy[1]) / q)))

    def is_disabled(self, xy: np.ndarray) -> bool:
        return self.key(xy) in self.disabled

    def reset(self) -> None:
        self.disabled.clear()
        self.selection_count.clear()
        self.last_xy = None
        self.last_distance = 0.0
        self.stick_steps = 0
        self.force_xy = None
        self.finish_first_explore = False
        self.neighbor_search = False

    def observe(self, chosen: Frontier, agent_xy: np.ndarray) -> None:
        """Port of `_handle_frontier_stick_and_disable` (llm_planner.py:239-270)."""
        xy = chosen.centroid_xy
        k = self.key(xy)
        same = self.last_xy is not None and self.key(self.last_xy) == k

        if same:
            if self.stick_steps == 0:
                self.last_distance = float(np.linalg.norm(xy - agent_xy))
                self.stick_steps = 1
            else:
                cur = float(np.linalg.norm(xy - agent_xy))
                closing = abs(self.last_distance - cur) > self.sticky_distance_m
                # `and not neighbor_search`: a frontier taken by the nearby
                # shortcut is metres away at most, so re-selecting it round after
                # round means stuck, even while the distance wobbles.
                if closing and not self.neighbor_search:
                    self.stick_steps = 0
                    self.last_distance = cur
                elif self.stick_steps >= self.sticky_steps:
                    self.disabled.add(k)
                    self.stick_steps = 0
                else:
                    self.stick_steps += 1
        else:
            self.stick_steps = 0
            self.last_distance = 0.0
            if k in self.selection_count:
                # Returned to a frontier chosen before: commit to it this time.
                self.force_xy = xy.copy()

        self.selection_count.setdefault(k, 0)
        if not same:
            self.selection_count[k] += 1
            if self.selection_count[k] >= self.repeat_disable:
                self.disabled.add(k)

        self.last_xy = xy.copy()


def _first_plannable(
    candidates: Sequence[Frontier],
    planner: Planner,
    costmap: Costmap2D,
    agent_xy: np.ndarray,
    min_path_cost_m: float,
    failed_out: Optional[Set[int]],
) -> Optional[Frontier]:
    """The reachability deviation. See the module docstring."""
    for f in candidates:
        result = planner.plan(costmap, agent_xy, frontier_goal_xy(f, costmap))
        if result.success:
            f.path_cost = max(result.cost, min_path_cost_m)
            return f
        f.path_cost = None
        if failed_out is not None:
            failed_out.add(f.id)
    return None


def select_frontier_ascent(
    frontiers: List[Frontier],
    values: Dict[int, float],
    planner: Planner,
    costmap: Costmap2D,
    agent_xy: np.ndarray,
    state: FrontierCommitState,
    nearby_distance_m: float = 3.0,
    min_path_cost_m: float = 0.5,
    top_n: int = 5,
    blocked: Optional[Set[int]] = None,
    failed_out: Optional[Set[int]] = None,
) -> Optional[Frontier]:
    """Port of `_get_best_frontier_with_llm` (llm_planner.py:57-129), LLM aside.

    `values` is the semantic value per frontier id; frontiers absent from it
    score 0, which sorts them last exactly as ASCENT's `-1` sentinel does.
    """
    blocked = blocked or set()
    live = [f for f in frontiers if f.id not in blocked]
    if not live:
        return None

    # 0. A single frontier needs no reasoning at all (:79-81).
    if len(live) == 1:
        chosen = _first_plannable(live, planner, costmap, agent_xy,
                                  min_path_cost_m, failed_out)
        if chosen is not None:
            state.observe(chosen, agent_xy)
        return chosen

    # 1. Rank by value, drop retired frontiers (_sort_frontiers_by_value, :131-148).
    ranked = [f for f in live if not state.is_disabled(f.centroid_xy)]
    ranked.sort(key=lambda f: -float(values.get(f.id, 0.0)))
    if not ranked:
        return None
    for f in ranked:
        f.score = float(values.get(f.id, 0.0))

    chosen: Optional[Frontier] = None
    state.neighbor_search = False

    # 2. A frontier we committed to earlier (_try_force_frontier, :157-164).
    if state.force_xy is not None:
        fk = state.key(state.force_xy)
        forced = [f for f in ranked if state.key(f.centroid_xy) == fk]
        chosen = _first_plannable(forced, planner, costmap, agent_xy,
                                  min_path_cost_m, failed_out)

    # 3. Anything within reach wins outright, whatever its value (:96-108, :166-179).
    if chosen is None and state.finish_first_explore:
        near = [f for f in ranked
                if float(np.linalg.norm(f.centroid_xy - agent_xy)) <= nearby_distance_m]
        near.sort(key=lambda f: float(np.linalg.norm(f.centroid_xy - agent_xy)))
        chosen = _first_plannable(near, planner, costmap, agent_xy,
                                  min_path_cost_m, failed_out)
        if chosen is not None:
            state.neighbor_search = True
        else:
            # Nothing nearby: drop back to a committed exploration pick (:106-108).
            state.finish_first_explore = False

    # 4. Otherwise the highest-valued frontier. No division by path cost -- this
    #    is the whole point of the arm.
    if chosen is None:
        chosen = _first_plannable(ranked[:top_n], planner, costmap, agent_xy,
                                  min_path_cost_m, failed_out)

    if chosen is None:
        return None

    # 5. Bookkeeping, then arm the shortcut for subsequent rounds (:118-126).
    state.observe(chosen, agent_xy)
    if not state.finish_first_explore:
        state.finish_first_explore = True
        state.force_xy = chosen.centroid_xy.copy()
    return chosen
