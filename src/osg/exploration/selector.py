"""Frontier selection: argmax P_i / d_i with d_i the true path cost from the
planner (paper formula) — relevance alone would chase distant frontiers.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np

from ..mapping.costmap import Costmap2D
from ..mapping.frontier import Frontier
from ..planning.planner import Planner


def select_frontier(
    frontiers: List[Frontier],
    scores: Dict[int, float],
    planner: Planner,
    costmap: Costmap2D,
    agent_xy: np.ndarray,
    unscored_prior: float = 0.3,
    min_path_cost_m: float = 0.5,
    top_n: int = 5,
    blocked: Optional[Set[int]] = None,
) -> Optional[Frontier]:
    blocked = blocked or set()
    candidates = [f for f in frontiers if f.id not in blocked]
    if not candidates:
        return None

    for f in candidates:
        f.score = scores.get(f.id, unscored_prior)
    candidates.sort(key=lambda f: -(f.score or 0.0))
    candidates = candidates[:top_n]

    best, best_util = None, -1.0
    for f in candidates:
        result = planner.plan(costmap, agent_xy, f.centroid_xy)
        if not result.success:
            f.path_cost = None
            continue
        f.path_cost = max(result.cost, min_path_cost_m)
        util = (f.score or 0.0) / f.path_cost
        if util > best_util:
            best, best_util = f, util
    return best
