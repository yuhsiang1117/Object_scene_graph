"""Which storey is the target on? A belief over the storeys ASCENT has mapped.

ASCENT changes storey in two places: when a storey runs out of frontiers
(`ascent_policy.py:707-731`), and when its LLM is asked the multi-floor
question and answers GO_UP / GO_DOWN (`llm_planner.py`, reached on a fraction
of the frontier decisions). Neither is a belief. Of the 22 cross-floor
episodes of `scenes20_ep0to4`, 6 mapped a staircase toward the goal's storey
and never started a climb (docs/history/AB_RESULTS.md S71): the storey still
had frontiers, and the question was not asked, or not answered that way.

This scores every known storey `k` each step, the form OSG's own
`exploration/storey_belief.py` uses:

    P_k  oc  pi_k * (1 - d * c_k)
    pi_k  =  w * prior_k + (1 - w) / K
    c_k   =  A_k / (A_k + a_f * F_k)

`prior_k` is ASCENT's own table, P(storey | category, storeys in the building)
(`AscentLLMPlanner.get_floor_probabilities`); `c_k` is how much of storey `k`
has been searched -- its explored area `A_k` against what its `F_k` live
frontiers still promise at `a_f` m^2 each, 0 for a storey never stood on and 1
for one ASCENT has marked explored; `d` is the chance that having searched a
place means having seen the object there. The agent is asked to leave when
another storey it can reach scores `switch_ratio` times this one, and the
request is held until this storey overtakes it again, so it does not flicker
while the agent walks to the stairs.

Pure bookkeeping: it reads numbers ASCENT's maps already hold and returns a
direction. Whether a staircase that way exists, whether enough of the episode
is left to climb it, and whether that staircase has been given up on are the
caller's to say (`can_up` / `can_down`), so the same class replays a recorded
trace offline (scripts/analyze_objectnav_search.py) and runs in the loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Storey:
    """One storey as ASCENT's own maps describe it, low to high in the list."""
    visited: bool = False          # the agent has stood on it this episode
    explored_m2: float = 0.0       # its obstacle map's explored area
    live_frontiers: int = 0        # frontiers not disabled
    exhausted: bool = False        # `_this_floor_explored`


class FloorSearchBelief:
    def __init__(self, *, switch_ratio: float, detect_prob: float, prior_weight: float,
                 frontier_area_m2: float, stats: Optional[dict] = None) -> None:
        self.switch_ratio = float(switch_ratio)
        self.detect_prob = float(detect_prob)
        self.prior_weight = float(prior_weight)
        self.frontier_area_m2 = float(frontier_area_m2)
        self.stats = stats if stats is not None else {}
        self.reset()

    def reset(self) -> None:
        self._requested: Optional[str] = None
        self._requested_from: Optional[int] = None
        self.log: List[dict] = []
        self._last_logged: Optional[tuple] = None

    # ----------------------------------------------------------- posterior

    def coverage(self, s: Storey) -> float:
        if not s.visited:
            return 0.0
        if s.exhausted:
            return 1.0
        promised = self.frontier_area_m2 * max(int(s.live_frontiers), 0)
        total = float(s.explored_m2) + promised
        return float(s.explored_m2) / total if total > 0.0 else 0.0

    def prior(self, table: Optional[Dict[int, float]], n: int) -> List[float]:
        """`table` is `{1..n: weight}` (ASCENT's percentages), storey 1 lowest.
        An empty or all-zero table is no information: uniform."""
        raw = [max(float((table or {}).get(i, 0.0)), 0.0) for i in range(1, n + 1)]
        total = sum(raw)
        if total <= 0.0:
            self.stats["storey_prior_empty"] = self.stats.get("storey_prior_empty", 0) + 1
            return [1.0 / n] * n
        w = self.prior_weight
        return [w * v / total + (1.0 - w) / n for v in raw]

    def posterior(self, storeys: Sequence[Storey], table: Optional[Dict[int, float]]) -> dict:
        n = len(storeys)
        pi = self.prior(table, n)
        cover = [self.coverage(s) for s in storeys]
        raw = [pi[k] * (1.0 - self.detect_prob * cover[k]) for k in range(n)]
        total = sum(raw) or 1.0
        return {"P": [v / total for v in raw], "prior": pi, "cover": cover}

    # ------------------------------------------------------------ decision

    def decide(self, step: int, here: int, storeys: Sequence[Storey],
               table: Optional[Dict[int, float]], *, can_up: bool, can_down: bool,
               here_id: Optional[int] = None) -> Optional[str]:
        """"up", "down", or None to keep searching this storey.

        `here` indexes `storeys`. Only storeys on a side the caller says is
        reachable compete; the request is for the best of them. `here_id` names
        the storey across calls -- its index moves when ASCENT inserts a storey
        below -- and defaults to the index.
        """
        post = self.posterior(storeys, table)
        P = post["P"]
        ident = here if here_id is None else here_id
        if self._requested_from != ident:         # arrived somewhere: nothing is held
            self._requested = None
        self._requested_from = ident
        reachable = [k for k in range(len(storeys))
                     if (k > here and can_up) or (k < here and can_down)]
        choice = None
        if reachable:
            best = max(reachable, key=lambda k: (P[k], -abs(k - here)))
            direction = "up" if best > here else "down"
            held = self._requested == direction and P[best] > P[here]
            if P[best] >= self.switch_ratio * P[here] or held:
                choice = direction
        self._requested = choice
        self.stats["storey_belief_rounds"] = self.stats.get("storey_belief_rounds", 0) + 1
        if choice is not None:
            self.stats["storey_belief_requests"] = self.stats.get("storey_belief_requests", 0) + 1
        key = (here, choice, tuple(round(v, 2) for v in P))
        if key != self._last_logged:
            self._last_logged = key
            self.log.append({
                "step": int(step), "here": int(here), "request": choice,
                "P": [round(float(v), 3) for v in P],
                "prior": [round(float(v), 3) for v in post["prior"]],
                "cover": [round(float(v), 3) for v in post["cover"]],
            })
        return choice


# ------------------------------------------------------------------ inputs

def floor_table(prior: Optional[dict], category: str, n: int) -> Dict[int, float]:
    """P(storey | category, an `n`-storey building) as `{1..n: percent}`.

    `AscentLLMPlanner.get_floor_probabilities` on the same json
    (`scripts/make_priors.py`), keyed by the HM3D category: the planner is
    handed the COCO name, which the json does not carry for sofa / tv_monitor /
    plant, and a building taller than the table gets the tallest column.
    """
    table = (prior or {}).get(str(category).lower().replace(" ", "_"))
    if not table:
        return {}
    tallest = max(int(k) for k in table)
    col = table.get(str(min(int(n), tallest)), {})
    return {int(k): float(v) for k, v in col.items()}


def storeys_of(agent) -> List[Storey]:
    """`AscentNavAgent._floors`, low to high, as the belief reads them."""
    out = []
    for floor in agent._floors:
        om = floor["obstacle"]
        frontiers = np.atleast_2d(np.asarray(om.frontiers)).reshape(-1, 2)
        out.append(Storey(
            visited=floor["standing_y_n"] > 0,
            explored_m2=float(om.explored_area.sum()) / (om.pixels_per_meter ** 2),
            live_frontiers=sum(1 for f in frontiers if tuple(f) not in om._disabled_frontiers),
            exhausted=bool(om._this_floor_explored),
        ))
    return out


class TraceStoreys:
    """The same list rebuilt from a recorded trace (`agent.search.trace`).

    A row describes only the storey the agent is on; the others keep what they
    were when it left, which is also all the agent's own maps know. ASCENT
    appends a storey when a staircase up is mapped from the top one and
    inserts one at index 0 when a staircase down is mapped from the bottom one
    (`agent.py`, `:531-535`), so a storey is followed by `floor_uid`, not by
    its index.
    """

    def __init__(self) -> None:
        self.uids: List[Optional[int]] = []
        self.storeys: List[Storey] = []

    def update(self, row: dict) -> Tuple[int, List[Storey]]:
        idx, uid, n = int(row["floor"]), int(row["floor_uid"]), int(row["n_floors"])
        if uid in self.uids:
            for _ in range(max(idx - self.uids.index(uid), 0)):   # inserted below
                self.uids.insert(0, None)
                self.storeys.insert(0, Storey())
        while len(self.uids) < n:                                 # appended above
            self.uids.append(None)
            self.storeys.append(Storey())
        self.uids[idx] = uid
        s = self.storeys[idx]
        s.visited = True
        s.explored_m2 = float(row.get("explored_m2") or 0.0)
        s.live_frontiers = int(row.get("nfront_live") or 0)
        s.exhausted = bool(row.get("floor_done"))
        return idx, self.storeys


def unexplored_beyond(storeys: Sequence[Storey], here: int, direction: str) -> bool:
    """`_navigate_stair_if_unexplored_floor`'s own condition (`ascent_policy.py
    :813-849`): some storey that way is not marked explored."""
    side = storeys[here + 1:] if direction == "up" else storeys[:here]
    return any(not s.exhausted for s in side)
