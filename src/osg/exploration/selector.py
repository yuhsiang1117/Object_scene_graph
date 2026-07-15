"""Frontier selection: argmax P_i / d_i with d_i the true path cost from the
planner (paper formula) — relevance alone would chase distant frontiers.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np

from ..mapping.costmap import Costmap2D
from ..mapping.frontier import Frontier
from ..planning.planner import Planner


def frontier_goal_xy(f: Frontier, costmap: Costmap2D) -> np.ndarray:
    """Plan to the frontier cell nearest the centroid, not the raw centroid:
    a concave component's centroid can fall in unreachable or occupied
    space."""
    if f.cells.shape[0] == 0:
        return f.centroid_xy
    cells_xy = np.stack([costmap.grid_to_world(rc) for rc in f.cells])
    d = np.linalg.norm(cells_xy - f.centroid_xy, axis=1)
    return cells_xy[int(np.argmin(d))]


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
    failed_out: Optional[Set[int]] = None,
) -> Optional[Frontier]:
    """Best frontier by P_i / d_i among the top-N scored candidates.
    Candidates whose path planning failed are added to `failed_out` so the
    caller can block just those — blocking every frontier on one bad round
    deadlocked exploration."""
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
        result = planner.plan(costmap, agent_xy, frontier_goal_xy(f, costmap))
        if not result.success:
            f.path_cost = None
            if failed_out is not None:
                failed_out.add(f.id)
            continue
        f.path_cost = max(result.cost, min_path_cost_m)
        util = (f.score or 0.0) / f.path_cost
        if util > best_util:
            best, best_util = f, util
    return best
