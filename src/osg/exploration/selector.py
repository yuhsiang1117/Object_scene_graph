"""Frontier selection: argmax P_i / d_i with d_i the true path cost from the
planner (paper formula) — relevance alone would chase distant frontiers.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np

from ..mapping.costmap import OCCUPIED, UNKNOWN, Costmap2D
from ..mapping.frontier import Frontier
from ..planning.planner import Planner


def _has_line_of_sight(costmap: Costmap2D, a_xy: np.ndarray, b_xy: np.ndarray) -> bool:
    """True if no OCCUPIED cell lies on the straight segment a->b. A frontier
    with clear line of sight from the agent has no wall between them, so it is
    (almost always) in the same room -- less valuable to explore than an
    occluded, behind-a-doorway frontier that opens a new room."""
    a = costmap.world_to_grid(a_xy).astype(float)
    b = costmap.world_to_grid(b_xy).astype(float)
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    t = np.linspace(0.0, 1.0, 2 * n + 1)[:, None]
    rc = np.rint(a[None, :] + t * (b - a)[None, :]).astype(int)
    h, w = costmap.grid.shape
    inb = (rc[:, 0] >= 0) & (rc[:, 0] < h) & (rc[:, 1] >= 0) & (rc[:, 1] < w)
    rc = rc[inb]
    if rc.shape[0] <= 2:
        return True
    # drop the last sample (the frontier boundary itself borders unknown/occupied)
    return not (costmap.grid[rc[:-1, 0], rc[:-1, 1]] == OCCUPIED).any()


def frontier_goal_xy(f: Frontier, costmap: Costmap2D) -> np.ndarray:
    """Plan to the frontier cell nearest the centroid, not the raw centroid:
    a concave component's centroid can fall in unreachable or occupied
    space."""
    if f.cells.shape[0] == 0:
        return f.centroid_xy
    cells_xy = np.stack([costmap.grid_to_world(rc) for rc in f.cells])
    d = np.linalg.norm(cells_xy - f.centroid_xy, axis=1)
    return cells_xy[int(np.argmin(d))]


def _info_gains(frontiers, costmap: Costmap2D, radius_m: float) -> Dict[int, int]:
    """Per-frontier unknown-area estimate: number of UNKNOWN costmap cells in a
    square window of `radius_m` around each frontier centroid (a cheap proxy for
    how much new space observing from there would reveal)."""
    unknown = costmap.grid == UNKNOWN
    h, w = unknown.shape
    rad = max(1, int(radius_m / costmap.resolution))
    out: Dict[int, int] = {}
    for f in frontiers:
        rc = costmap.world_to_grid(f.centroid_xy)
        r0, r1 = max(0, rc[0] - rad), min(h, rc[0] + rad + 1)
        c0, c1 = max(0, rc[1] - rad), min(w, rc[1] + rad + 1)
        out[f.id] = int(unknown[r0:r1, c0:c1].sum())
    return out


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
    info_gain_weight: float = 0.0,
    info_gain_radius_m: float = 2.5,
    los_visibility_penalty: float = 1.0,
    heading_xy: Optional[np.ndarray] = None,
    continuity_weight: float = 0.0,
    frontier_values: Optional[Dict[int, float]] = None,
    value_weight: float = 1.0,
    value_argmax: bool = False,
) -> Optional[Frontier]:
    """Best frontier by P_i / d_i among the top-N scored candidates.
    Candidates whose path planning failed are added to `failed_out` so the
    caller can block just those — blocking every frontier on one bad round
    deadlocked exploration.

    Information gain: when `info_gain_weight > 0`, each frontier's score is
    boosted by how much unknown area it exposes -- the count of UNKNOWN costmap
    cells within `info_gain_radius_m` of the frontier, normalized against the
    best candidate this round: score *= (1 + info_gain_weight * gain/gain_max).
    Folding it in before the top-N cut makes exploration commit to frontiers
    that open large unexplored regions instead of the nearest small one."""
    blocked = blocked or set()
    candidates = [f for f in frontiers if f.id not in blocked]
    if not candidates:
        return None

    gain = _info_gains(candidates, costmap, info_gain_radius_m) if info_gain_weight > 0.0 else None
    gmax = max(gain.values()) if gain else 0
    for f in candidates:
        # A semantic value from the top-down map replaces the flat prior where
        # one exists: it is a per-frontier estimate of how much that direction
        # looks like the target's habitat, which a constant by definition is
        # not. value_weight sharpens it, because the geometric boosts below are
        # multiplicative and can otherwise swamp a value range of a few
        # hundredths.
        if frontier_values is not None and f.id in frontier_values:
            base = max(float(frontier_values[f.id]), 0.0) ** value_weight
        else:
            base = scores.get(f.id, unscored_prior)
        boost = 1.0 + info_gain_weight * (gain[f.id] / gmax) if (gain and gmax > 0) else 1.0
        # Continuity / momentum: boost frontiers that lie AHEAD of the agent's
        # current heading, so consecutive frontier goals form a continuous sweep
        # instead of the greedy argmax ping-ponging across the map (which spends
        # ~30 steps travelling between far-apart frontiers). align in [0,1] =
        # how forward the frontier direction is; behind-the-agent frontiers get
        # no bonus (align clamped at 0), so the agent finishes the current
        # direction before reversing.
        if continuity_weight > 0.0 and heading_xy is not None:
            d = f.centroid_xy - agent_xy
            n = float(np.linalg.norm(d))
            align = max(0.0, float(d @ heading_xy) / n) if n > 1e-6 else 0.0
            boost *= 1.0 + continuity_weight * align
        f.score = base * boost
        # Down-weight frontiers with clear line of sight from the agent: no wall
        # between => same room => less likely to open a new room with the target.
        if los_visibility_penalty < 1.0 and _has_line_of_sight(costmap, agent_xy, f.centroid_xy):
            f.score *= los_visibility_penalty
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
        # Dividing by path cost systematically favours near frontiers, which can
        # drown out a semantic signal spanning only a few hundredths. ASCENT
        # takes the argmax of the value instead, with a nearby-frontier
        # shortcut; value_argmax makes that comparable here. Reachability is
        # still required -- only the ranking changes.
        util = (f.score or 0.0) if value_argmax else (f.score or 0.0) / f.path_cost
        if util > best_util:
            best, best_util = f, util
    return best
