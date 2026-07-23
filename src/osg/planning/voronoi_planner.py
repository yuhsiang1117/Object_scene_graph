"""Generalized Voronoi Graph (GVG) planner, in the spirit of ObjectSceneGraph_old
(voronoi_diagram.py). Navigation follows the medial axis of the free space:

  1. Euclidean distance field to the nearest obstacle (collision field)
  2. the medial axis is the SKELETON of the clearance-eroded free space -- this
     is connected wherever the free space is (crucially, through doorways), which
     scipy.spatial.Voronoi-over-obstacle-cells did NOT reliably give in a pure-
     Python port without Nav2's local connection. Same GVG/medial-axis idea.
  3. graph = skeleton cells (nodes) + 8-neighbour adjacency (edges)
  4. A* on the robot's connected component; for object goals, end at a graph node
     within `goal_near_m` of the goal (stop near, not on, it)

Implements the `Planner` interface so it drops into the existing agent. The
graph is cached and rebuilt only when the costmap changes materially.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from ..mapping.costmap import FREE, OCCUPIED, UNKNOWN, Costmap2D
from .planner import AStarPlanner, PlanResult, Planner


class HybridVoronoiPlanner(Planner):
    """GVG Voronoi (medial-axis) navigation with a grid-A* fallback for when the
    Voronoi skeleton has no coverage near the agent yet (early exploration /
    tiny free pockets). Keeps navigation on the medial axis whenever possible."""

    def __init__(self, collision_m: float = 0.2, goal_near_m: float = 0.7,
                 inflate_radius_m: float = 0.25) -> None:
        self.voronoi = VoronoiPlanner(collision_m=collision_m, goal_near_m=goal_near_m)
        self.astar = AStarPlanner(inflate_radius_m=inflate_radius_m)

    def plan(self, costmap, start_xy, goal_xy, goal_tolerance_m=None) -> PlanResult:
        r = self.voronoi.plan(costmap, start_xy, goal_xy, goal_tolerance_m)
        if r.success:
            return r
        return self.astar.plan(costmap, start_xy, goal_xy, goal_tolerance_m)


class VoronoiPlanner(Planner):
    def __init__(
        self,
        collision_m: float = 0.2,        # min clearance of a graph edge to an obstacle
        check_m: float = 0.3,            # edge-sampling step for the clearance test
        goal_near_m: float = 0.7,        # object-goal: stop at a node within this radius
        boundary_spacing_m: float = 0.5,  # map-border Voronoi seed spacing
        min_obstacle_cells: int = 20,
        min_component: int = 6,           # skip tiny Voronoi corner-artifact components
    ) -> None:
        self.collision_m = collision_m
        self.check_m = check_m
        self.goal_near_m = goal_near_m
        self.boundary_spacing_m = boundary_spacing_m
        self.min_obstacle_cells = min_obstacle_cells
        self.min_component = min_component
        self.last_failure = ""
        # cached graph: nodes (M,2 rc float), adjacency (list of neighbor idx),
        # and the signature it was built for.
        self._nodes: Optional[np.ndarray] = None
        self._adj: List[List[int]] = []
        self._sig: Optional[Tuple] = None
        self._dist_cells: Optional[np.ndarray] = None

    # ------------------------------------------------------------- graph build

    def _signature(self, costmap: Costmap2D) -> Tuple:
        g = costmap.grid
        return (g.shape, int((g == OCCUPIED).sum()), int((g != UNKNOWN).sum()))

    def _rebuild(self, costmap: Costmap2D) -> None:
        from skimage.morphology import skeletonize

        grid = costmap.grid
        h, w = grid.shape
        res = costmap.resolution
        occ = grid == OCCUPIED
        self._dist_cells = ndimage.distance_transform_edt(~occ)
        collision_c = self.collision_m / res

        # medial axis = skeleton of the free space that keeps `collision_m`
        # clearance from obstacles (so the robot can follow it safely)
        free_safe = (grid == FREE) & (self._dist_cells >= collision_c)
        if free_safe.sum() < self.min_obstacle_cells:
            self._nodes, self._adj = np.zeros((0, 2)), []
            return
        skel = skeletonize(free_safe)
        rc = np.argwhere(skel)
        if rc.shape[0] == 0:
            self._nodes, self._adj = np.zeros((0, 2)), []
            return

        idx = -np.ones((h, w), dtype=np.int64)
        idx[rc[:, 0], rc[:, 1]] = np.arange(rc.shape[0])
        adj: List[List[int]] = [[] for _ in range(rc.shape[0])]
        # 8-neighbour edges among skeleton cells (vectorized per offset)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            r2, c2 = rc[:, 0] + dr, rc[:, 1] + dc
            ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
            j = np.where(ok, idx[np.clip(r2, 0, h - 1), np.clip(c2, 0, w - 1)], -1)
            for k in np.nonzero(j >= 0)[0]:
                adj[k].append(int(j[k]))
        self._nodes, self._adj = rc.astype(float), adj

    # ------------------------------------------------------------- graph query

    def _component(self, seed_idx: int) -> set:
        seen = {seed_idx}
        stack = [seed_idx]
        while stack:
            u = stack.pop()
            for v in self._adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    def _nearest(self, rc: np.ndarray, allowed: Optional[set] = None) -> Optional[int]:
        if self._nodes is None or len(self._nodes) == 0:
            return None
        d = np.linalg.norm(self._nodes - rc, axis=1)
        if allowed is not None:
            mask = np.full(len(d), np.inf)
            for i in allowed:
                mask[i] = d[i]
            d = mask
        idx = int(np.argmin(d))
        return idx if np.isfinite(d[idx]) else None

    def _clear_segment(self, a_rc: np.ndarray, b_rc: np.ndarray, grid) -> bool:
        """True if the straight line a->b hits no occupied cell (the robot has
        no local obstacle avoidance, so its connector to the graph must be clear)."""
        h, w = grid.shape
        n = int(np.hypot(b_rc[0] - a_rc[0], b_rc[1] - a_rc[1])) + 1
        ts = np.linspace(0.0, 1.0, max(2, n))
        rs = np.clip(np.round(a_rc[0] + ts * (b_rc[0] - a_rc[0])).astype(int), 0, h - 1)
        cs = np.clip(np.round(a_rc[1] + ts * (b_rc[1] - a_rc[1])).astype(int), 0, w - 1)
        return not (grid[rs, cs] == OCCUPIED).any()

    def _nearest_clear(self, rc: np.ndarray, grid, allowed: set) -> Optional[int]:
        """Nearest node whose straight connector from rc is obstacle-free."""
        if self._nodes is None or len(self._nodes) == 0:
            return None
        order = sorted(allowed, key=lambda i: np.linalg.norm(self._nodes[i] - rc))
        for i in order[:40]:
            if self._clear_segment(rc, self._nodes[i], grid):
                return i
        return None

    def _astar(self, start_i: int, goal_i: int) -> Optional[List[int]]:
        nodes = self._nodes
        pq = [(0.0, start_i)]
        g = {start_i: 0.0}
        came: Dict[int, int] = {}
        goal_pt = nodes[goal_i]
        while pq:
            _, u = heapq.heappop(pq)
            if u == goal_i:
                path = [u]
                while path[-1] in came:
                    path.append(came[path[-1]])
                return path[::-1]
            for v in self._adj[u]:
                ng = g[u] + float(np.linalg.norm(nodes[u] - nodes[v]))
                if ng < g.get(v, np.inf):
                    g[v] = ng
                    came[v] = u
                    heapq.heappush(pq, (ng + float(np.linalg.norm(nodes[v] - goal_pt)), v))
        return None

    # ------------------------------------------------------------------- plan

    def plan(
        self,
        costmap: Costmap2D,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        goal_tolerance_m: Optional[float] = None,
    ) -> PlanResult:
        sig = self._signature(costmap)
        if sig != self._sig or self._nodes is None:
            self._rebuild(costmap)
            self._sig = sig
        if self._nodes is None or len(self._nodes) == 0:
            self.last_failure = "empty_graph"
            return PlanResult(False)

        start_rc = costmap.world_to_grid(start_xy).astype(float)
        goal_rc = costmap.world_to_grid(goal_xy).astype(float)
        # Snap the robot onto the nearest medial-axis node reachable by a clear
        # straight connector whose component is substantial (>= min_component) --
        # this joins the robot's actual room/corridor while skipping the tiny
        # Voronoi corner artifacts, without forcing a single global component.
        all_nodes = set(range(len(self._nodes)))
        start_i, comp = None, set()
        order = sorted(all_nodes, key=lambda i: np.linalg.norm(self._nodes[i] - start_rc))
        for i in order[:60]:
            if not self._clear_segment(start_rc, self._nodes[i], costmap.grid):
                continue
            c = self._component(i)
            if len(c) >= self.min_component:
                start_i, comp = i, c
                break
        if start_i is None:
            self.last_failure = "no_clear_start_node"
            return PlanResult(False)

        # object goals: end at a node within goal_near_m of the goal (stop near it)
        near_m = self.goal_near_m if goal_tolerance_m is None else goal_tolerance_m
        near_c = near_m / costmap.resolution
        cand = [i for i in comp if np.linalg.norm(self._nodes[i] - goal_rc) <= near_c]
        if cand:
            # choose the reachable candidate giving the shortest path
            best_path, best_len = None, np.inf
            for gi in sorted(cand, key=lambda i: np.linalg.norm(self._nodes[i] - goal_rc))[:8]:
                p = self._astar(start_i, gi)
                if p is None:
                    continue
                L = sum(float(np.linalg.norm(self._nodes[p[k]] - self._nodes[p[k + 1]]))
                        for k in range(len(p) - 1))
                if L < best_len:
                    best_path, best_len = p, L
            path_idx = best_path
        else:
            # No skeleton node within goal_near_m of the goal: the goal is in a
            # different medial-axis component than the robot (clearance erosion
            # severed the passage, even though the free space is 4-connected).
            # Navigating to the nearest in-component node then only makes sense
            # if it gets the robot MEANINGFULLY closer to the goal; otherwise
            # that node is ~= the robot's own cell and the "path" is a stub that
            # reads as a false arrival (goal 1-3 m away, path ends at the start),
            # freezing the agent. Require real progress; if there is none, fail
            # so HybridVoronoiPlanner falls back to grid-A*, which routes through
            # the full free grid and does reach these goals.
            goal_i = self._nearest(goal_rc, allowed=comp)
            if goal_i is None:
                self.last_failure = "no_goal_node"
                return PlanResult(False)
            start_to_goal = float(np.linalg.norm(start_rc - goal_rc))
            node_to_goal = float(np.linalg.norm(self._nodes[goal_i] - goal_rc))
            if node_to_goal > start_to_goal - near_c:
                self.last_failure = "goal_unreachable_via_graph"
                return PlanResult(False)
            path_idx = self._astar(start_i, goal_i)

        if not path_idx:
            self.last_failure = "no_path"
            return PlanResult(False)

        # node rc -> world path; prepend the true start so following is smooth
        pts = np.array([costmap.grid_to_world(self._nodes[i]) for i in path_idx])
        pts = np.vstack([start_xy, pts])
        seg = np.diff(pts, axis=0)
        cost = float(np.sum(np.linalg.norm(seg, axis=1))) if len(pts) > 1 else 0.0
        return PlanResult(True, path=pts, cost=max(cost, costmap.resolution))
