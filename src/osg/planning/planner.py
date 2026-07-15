"""Grid path planning. A* on the inflated costmap; unknown cells are
traversable at a penalty (frontier goals sit at the unknown boundary by
definition). The Planner ABC keeps a Voronoi/FMM planner pluggable later.
"""
from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..mapping.costmap import FREE, UNKNOWN, Costmap2D


@dataclass
class PlanResult:
    success: bool
    path: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # (M, 2) world xy
    cost: float = float("inf")  # meters (unknown penalty excluded)


class Planner(ABC):
    @abstractmethod
    def plan(self, costmap: Costmap2D, start_xy: np.ndarray, goal_xy: np.ndarray) -> PlanResult: ...


_SQRT2 = float(np.sqrt(2.0))
_NEIGHBORS: List[Tuple[int, int, float]] = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2),
]


class AStarPlanner(Planner):
    def __init__(
        self,
        inflate_radius_m: float = 0.25,
        unknown_penalty: float = 3.0,
        goal_tolerance_m: float = 0.3,
        max_expansions: int = 60_000,  # caps worst-case spikes (~12 s at 200k)
    ) -> None:
        self.inflate_radius_m = inflate_radius_m
        self.unknown_penalty = unknown_penalty
        self.goal_tolerance_m = goal_tolerance_m
        self.max_expansions = max_expansions

    def plan(self, costmap: Costmap2D, start_xy: np.ndarray, goal_xy: np.ndarray) -> PlanResult:
        blocked = costmap.inflated(self.inflate_radius_m)
        grid = costmap.grid
        h, w = grid.shape
        start = tuple(costmap.world_to_grid(start_xy))
        goal = tuple(costmap.world_to_grid(goal_xy))
        if not costmap.in_bounds(np.array(start)):
            return PlanResult(False)
        goal = (min(max(goal[0], 0), h - 1), min(max(goal[1], 0), w - 1))
        start = self._nudge_free(start, blocked, grid)
        if start is None:
            return PlanResult(False)

        tol_cells = max(1, int(self.goal_tolerance_m / costmap.resolution))
        res = costmap.resolution

        g = {start: 0.0}
        came: dict = {}
        pq: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
        visited = set()
        expansions = 0
        found = None
        while pq and expansions < self.max_expansions:
            _, cur = heapq.heappop(pq)
            if cur in visited:
                continue
            visited.add(cur)
            expansions += 1
            if abs(cur[0] - goal[0]) <= tol_cells and abs(cur[1] - goal[1]) <= tol_cells:
                found = cur
                break
            for dr, dc, step in _NEIGHBORS:
                nr, nc = cur[0] + dr, cur[1] + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if blocked[nr, nc]:
                    continue
                cell = grid[nr, nc]
                mult = self.unknown_penalty if cell == UNKNOWN else 1.0
                ng = g[cur] + step * mult
                nxt = (nr, nc)
                if ng < g.get(nxt, float("inf")):
                    g[nxt] = ng
                    came[nxt] = cur
                    hcost = np.hypot(nr - goal[0], nc - goal[1])
                    heapq.heappush(pq, (ng + hcost, nxt))

        if found is None:
            return PlanResult(False)

        # Reconstruct and convert to world coords; cost in meters over the
        # actual geometric path (penalties guide search, not the reported d_i).
        cells = [found]
        while cells[-1] in came:
            cells.append(came[cells[-1]])
        cells.reverse()
        path = np.array([costmap.grid_to_world(np.array(c)) for c in cells])
        seg = np.diff(path, axis=0)
        cost = float(np.sum(np.linalg.norm(seg, axis=1))) if len(path) > 1 else 0.0
        return PlanResult(True, path=path, cost=max(cost, res))

    @staticmethod
    def _nudge_free(start, blocked, grid, max_r: int = 6):
        """If the start cell is inside the inflation radius, find the nearest
        plannable cell within a small window (the agent is never truly stuck
        in a wall; inflation just swallowed its cell)."""
        if not blocked[start] and grid[start] == FREE:
            return start
        h, w = grid.shape
        best, best_d = None, float("inf")
        for r in range(max(0, start[0] - max_r), min(h, start[0] + max_r + 1)):
            for c in range(max(0, start[1] - max_r), min(w, start[1] + max_r + 1)):
                if blocked[r, c] or grid[r, c] != FREE:
                    continue
                d = (r - start[0]) ** 2 + (c - start[1]) ** 2
                if d < best_d:
                    best, best_d = (r, c), d
        return best
