"""Where to look next, once "not here" is an answer the map can give.

DualMap's reaction to a failed candidate is to take the next-highest similarity
and add the failed one to an ignore list that is discarded when the query ends.
That is the discrete search problem with the belief thrown away: no model of
where the object went, no cost of getting there, no memory of what was already
searched.

The classical result for that problem is that the optimal ORDER is by the index

    b(x) * d(x) / c(x)

-- belief times per-visit detection probability over cost -- and that after an
unsuccessful look the belief updates multiplicatively, b(x) <- b(x) * (1 - d(x)).
Greedy is optimal there under stated assumptions (independent locations, cost
per look); what makes it approximate in practice is travel-order coupling, since
this is really a profitable-tour problem. DualMap's ignore list is the degenerate
case b <- 0 after a single look, discarded at query end.

The pleasing part is that `select_frontier` already computes exactly this index
over frontiers (score / path_cost). Phase 3 does not add an objective or a
planner; it widens the candidate set to include the surfaces an object could
have been moved to, so exploring and re-searching stop being two subsystems --
which they still are in DualMap, whose retry loop cannot decide to go explore
instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..graph.priors import affinity_scores, affords


@dataclass
class SearchCandidate:
    """One place worth looking, of either kind."""

    kind: str  # "container" | "frontier"
    ref_id: int
    goal_xy: np.ndarray
    prior: float  # b(x) before this round's cost is known
    detect_prob: float  # d(x): chance a visit here would find the target
    label: str = ""
    path_cost: Optional[float] = None
    utility: Optional[float] = None


@dataclass
class InspectionLog:
    """What has already been searched, and how well.

    A visit does not settle a place -- it multiplies its belief by (1 - d). A
    surface glanced at from four metres stays plausible; one inspected from
    close range mostly stops being. Persisting this is the whole difference
    from an ignore list, which cannot tell those two apart and forgets both.
    """

    survived: Dict[int, float] = field(default_factory=dict)

    def factor(self, ref_id: int) -> float:
        return float(self.survived.get(int(ref_id), 1.0))

    def searched(self, ref_id: int, detect_prob: float) -> float:
        f = self.factor(ref_id) * (1.0 - float(np.clip(detect_prob, 0.0, 0.99)))
        self.survived[int(ref_id)] = f
        return f


def container_prior(
    target: str,
    label: str,
    top_h: float,
    area_m2: float,
    centre_xy: np.ndarray,
    last_known_xy: Optional[np.ndarray] = None,
    proximity_len_m: float = 4.0,
    proximity_floor: float = 0.2,
    affinity_source=None,
) -> float:
    """b(x) for a mapped surface: affordance x affinity x proximity.

    Affordance is binary and comes first because it is the cheapest way to be
    certain: a surface that cannot hold the class is not a worse place to look,
    it is not a place to look. Proximity encodes that things are moved by
    someone doing a task, so displacements are short far more often than long --
    it is measured from where the object was last believed to be, not from the
    agent, because the agent's distance is already the cost term.
    """
    if not affords(target, top_h, area_m2):
        return 0.0
    scores = affinity_scores(target, source=affinity_source)
    key = str(label).lower().replace("_", " ").strip()
    if not scores:
        # No prior at all for this class: every surface is equally plausible.
        affinity = 0.5
    else:
        # We DO have a ranking and this category is not in it. That is weak
        # evidence against, and it has to sit BELOW the lowest ranked entry --
        # at a middling 0.5 a bed and a sofa tied with a sink as places to look
        # for a bowl, and the agent duly went to both.
        affinity = scores.get(key, 0.25)
    prox = 1.0
    if last_known_xy is not None:
        d = float(np.linalg.norm(np.asarray(centre_xy, float) - np.asarray(last_known_xy, float)))
        # A pure exponential says an object that moved 7 m is almost impossible,
        # which is false -- cross-anchor moves in this benchmark average 5 m.
        # The floor makes this a mixture of "moved nearby" and "moved anywhere",
        # which also stops distance being punished twice: once in the prior and
        # again in the cost term that already divides by path length.
        prox = max(float(np.exp(-d / max(proximity_len_m, 1e-6))), float(proximity_floor))
    return float(affinity * prox)


def select_candidate(
    candidates: Sequence[SearchCandidate],
    planner,
    costmap,
    agent_xy: np.ndarray,
    top_n: int = 6,
    min_path_cost_m: float = 0.5,
    failed_out: Optional[set] = None,
) -> Optional[SearchCandidate]:
    """argmax prior * detect_prob / path_cost over the top-N by prior.

    Same shape as `select_frontier`: rank cheaply, cut to top-N, then pay for
    real path costs only on the survivors, because planning is the expensive
    part and ranking by Euclidean distance would defeat the point of using a
    geodesic cost at all.
    """
    ranked = sorted(
        (c for c in candidates if c.prior > 0.0 and c.detect_prob > 0.0),
        key=lambda c: -(c.prior * c.detect_prob),
    )[:top_n]
    best, best_util = None, -1.0
    for cand in ranked:
        result = planner.plan(costmap, agent_xy, cand.goal_xy)
        if not result.success:
            cand.path_cost = None
            if failed_out is not None:
                failed_out.add((cand.kind, cand.ref_id))
            continue
        cand.path_cost = max(result.cost, min_path_cost_m)
        cand.utility = cand.prior * cand.detect_prob / cand.path_cost
        if cand.utility > best_util:
            best, best_util = cand, cand.utility
    return best


def build_container_candidates(
    scene_graph,
    target: str,
    log: InspectionLog,
    detect_prob: float = 0.8,
    last_known_xy: Optional[np.ndarray] = None,
    proximity_len_m: float = 4.0,
    proximity_floor: float = 0.2,
    plane=(0, 2),
    affinity_source=None,
) -> List[SearchCandidate]:
    out: List[SearchCandidate] = []
    for node in getattr(scene_graph, "containers", {}).values():
        centre_xy = np.asarray(node.center, dtype=float)[list(plane)]
        prior = container_prior(
            target, node.label, node.top_h, node.area_m2, centre_xy,
            last_known_xy=last_known_xy, proximity_len_m=proximity_len_m,
            proximity_floor=proximity_floor, affinity_source=affinity_source,
        )
        if prior <= 0.0:
            continue
        out.append(
            SearchCandidate(
                kind="container",
                ref_id=int(node.id),
                goal_xy=centre_xy,
                prior=prior * log.factor(node.id),
                detect_prob=float(detect_prob),
                label=str(node.label),
            )
        )
    return out
