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


def frontier_goal_xy(
    f: Frontier, costmap: Costmap2D, prefer_free: bool = False
) -> np.ndarray:
    """Where to drive in order to explore this frontier.

    The historical default returns the frontier CELL nearest the centroid,
    because "a concave component's centroid can fall in unreachable or occupied
    space". That reasoning predates `FrontierExtractor.extract`, which now snaps
    `centroid_xy` to the nearest FREE cell and rejects the frontier outright if
    an obstacle lies within the collision radius of it -- so a surviving
    frontier's centroid is already free and clear.

    `prefer_free` returns that snapped centroid instead. It matters on the
    navmesh: `Frontier.cells` are UNKNOWN cells by construction, and handing
    unknown space to `pathfinder.snap_point` yields whatever navigable point
    happens to be nearest -- possibly behind the agent or through a wall. The
    follower then reports arrived-or-unreachable almost at once. Measured over
    100 episodes: 289 stub-blocks against only 24 give-ups, 53% of selections
    landing within 1.5 m of an earlier one, and twice the planned distance
    actually walked.
    """
    if prefer_free or f.cells.shape[0] == 0:
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
    goal_prefer_free: bool = False,
    cost_prefer_free: bool = False,
    ranked_out: Optional[List[Frontier]] = None,
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
    reachable: List[tuple] = []
    for f in candidates:
        # The RANKING cost may be measured to a different point than the agent
        # will drive to, and usually should be. Planning to an UNKNOWN cell
        # fails often enough that whole selection rounds collapse (measured:
        # select_none 29 -> 2 over 100 episodes when costed to the free
        # centroid). But DRIVING to that same free centroid stops the agent at
        # the edge of known space instead of pushing into the frontier, cutting
        # coverage -- single-floor explore-failures went 0 -> 3 and mean steps
        # 136 -> 165. The two points are a cell or two apart, so the ranking is
        # barely affected; only the planner's success rate is.
        result = planner.plan(
            costmap, agent_xy,
            frontier_goal_xy(f, costmap, goal_prefer_free or cost_prefer_free),
        )
        if not result.success:
            f.path_cost = None
            if failed_out is not None:
                failed_out.add(f.id)
            continue
        f.path_cost = max(result.cost, min_path_cost_m)
        util = (f.score or 0.0) / f.path_cost
        reachable.append((util, f))
        if util > best_util:
            best, best_util = f, util
    if ranked_out is not None:
        # Utility order, best first -- the candidate list a semantic re-rank
        # (exploration/coarse_to_fine) chooses among. Only path-reachable
        # frontiers appear, so a choice from this list is always drivable.
        ranked_out.extend(f for _, f in sorted(reachable, key=lambda uf: -uf[0]))
    return best
