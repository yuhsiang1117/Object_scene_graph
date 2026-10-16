"""A walkable route up or down a staircase, read off the height layer.

`stairs.find_flights` finds a staircase as a BLOB of cells whose heights sit
between two storeys and hands the climb its lowest or highest cell. A blob has
no notion of walking: furniture beside the foot, the void seen over a banister
and both legs of a switchback join it, and its extreme cell is wherever the
merge put it. Measured on Benchmark v2 seed 42 (outputs/mfv2/diag_climb): on
00821 the agent stood 1.1 m from the true foot of a switchback and the blob's
foot was 1.5 m to the side of it; the climb held that carrot for 300 steps,
twice, and rose 0.01 m. On 00800 the descent's "top" was 1.9 m down the
flight.

This asks the question a staircase answers by construction: which cells can be
WALKED to from where the agent stands, one riser at a time? Neighbouring cells
are joined when their surface heights differ by at most `max_step_m`, and
cells a wall stands on at this storey's floor are left out. Treads chain;
a sofa's edge, the drop over a banister and the gap between the two legs of a
switchback are steps no one takes, so they split the graph where a walker
would be stopped. The route is the cheapest walk from the agent to FLAT
reachable floor near the storey asked for -- not a landing, which is flat but
a storey-half off, and not the next storey down, which is a whole one off.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .costmap import OCCUPIED, block_min


@dataclass
class StairRoute:
    """A route in world coordinates, the agent's end first."""

    xy: np.ndarray            # (N, 2)
    heights: np.ndarray       # (N,) surface height of each point
    cost_m: np.ndarray        # (N,) walking distance from the start
    reaches_target: bool      # ends within reach of the storey asked for
    gain_m: float             # height gained in the wanted direction

    @property
    def end_xy(self) -> np.ndarray:
        return self.xy[-1]

    def leaves_floor_after_m(self, floor_y: float, tol_m: float = 0.15) -> float:
        """Walking distance before the route first leaves this floor's height."""
        off = np.flatnonzero(np.abs(self.heights - float(floor_y)) > tol_m)
        return float(self.cost_m[off[0]]) if len(off) else float(self.cost_m[-1])


def plan_stair_route(
    costmap,
    floor_y: float,
    agent_xy,
    standing_y: float,
    direction: int,
    target_y: Optional[float] = None,
    cell_m: float = 0.1,
    max_step_m: float = 0.25,
    radius_m: float = 12.0,
    wall_band_m: float = 0.15,
    min_gain_m: float = 1.0,
    seed_m: float = 0.6,
    flat_tol_m: float = 0.04,
    flat_window_m: float = 0.5,
    storey_tol_m: float = 0.6,
    storey_m: float = 1.8,
) -> Optional[StairRoute]:
    """The cheapest walk from the agent to the storey asked for.

    `direction` is +1 up, -1 down. The goal is reachable FLAT floor: within
    `storey_tol_m` of `target_y`, or without one the nearest at least
    `storey_m` away. When no such floor is mapped yet, the walk goes as far
    in `direction` as the map allows (at least `min_gain_m`, never more than
    `storey_tol_m` past `target_y`), with `reaches_target` False. None when
    there is no height layer, nowhere to start from, or no such walk.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    height = getattr(costmap, "height", None)
    if height is None:
        return None
    res = float(costmap.resolution)
    block = max(1, int(round(float(cell_m) / res)))
    cell = block * res
    rc = costmap.world_to_grid(np.asarray(agent_xy, dtype=float))
    rad = int(np.ceil(float(radius_m) / res))
    h_full, w_full = costmap.grid.shape
    r0, c0 = max(0, int(rc[0]) - rad), max(0, int(rc[1]) - rad)
    r1, c1 = min(h_full, int(rc[0]) + rad + 1), min(w_full, int(rc[1]) + rad + 1)
    # Align the window to whole blocks.
    r1 = r0 + ((r1 - r0) // block) * block
    c1 = c0 + ((c1 - c0) // block) * block
    if r1 <= r0 or c1 <= c0:
        return None
    hw = np.asarray(height[r0:r1, c0:c1], dtype=float)
    top = getattr(costmap, "height_top", None)
    if top is not None:
        tw = np.asarray(top[r0:r1, c0:c1], dtype=float)
        hw = np.where(np.isfinite(tw), tw, hw)
    grid = costmap.grid[r0:r1, c0:c1]
    # A wall standing on THIS floor: occupied, and its lowest seen point at the
    # floor. Treads above or below the floor are occupied in a floor-relative
    # grid too, by their own surface, and must stay in.
    wall_fine = (grid == OCCUPIED) & (np.abs(hw - float(floor_y)) < float(wall_band_m))
    H = block_min(hw.astype(np.float32), block).astype(float)
    nr, nc = H.shape
    wall = wall_fine.reshape(nr, block, nc, block).any(axis=(1, 3))
    valid = np.isfinite(H) & ~wall
    if not valid.any():
        return None

    idx = -np.ones((nr, nc), dtype=np.int64)
    idx[valid] = np.arange(int(valid.sum()))
    rows, cols, wts = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        rs, rt = slice(0, nr - dr), slice(dr, nr)
        cs, ct = (slice(0, nc - dc), slice(dc, nc)) if dc >= 0 else (slice(-dc, nc), slice(0, nc + dc))
        ok = valid[rs, cs] & valid[rt, ct] & (np.abs(H[rs, cs] - H[rt, ct]) <= float(max_step_m))
        ia, ib = idx[rs, cs][ok], idx[rt, ct][ok]
        rows.append(ia)
        cols.append(ib)
        wts.append(np.full(ia.shape, cell * (np.sqrt(2.0) if dr and dc else 1.0)))
    n = int(valid.sum())
    rows_a, cols_a, w_a = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
    graph = coo_matrix((w_a, (rows_a, cols_a)), shape=(n, n)).tocsr()

    # Start from the cells under and around the agent at its own height: the
    # camera never sees the cell it stands on.
    ii, jj = np.nonzero(valid)
    centres = np.stack([r0 + (ii + 0.5) * block, c0 + (jj + 0.5) * block], axis=1)
    world = costmap.origin[None, :] + centres * res
    d_agent = np.linalg.norm(world - np.asarray(agent_xy, dtype=float)[None, :], axis=1)
    near_h = np.abs(H[ii, jj] - float(standing_y)) <= 0.2
    sources = np.flatnonzero((d_agent <= float(seed_m)) & near_h)
    if not len(sources):
        return None
    dist, pred, _ = dijkstra(graph, directed=False, indices=sources,
                             min_only=True, return_predecessors=True)
    reach = np.isfinite(dist)
    h_node = H[ii, jj]
    gain = float(direction) * (h_node - float(standing_y))
    # FLOOR, as opposed to treads: flat over a `flat_window_m` neighbourhood.
    # A tread is ~0.28 m deep, so a 0.5 m window always straddles a riser (a
    # 0.3 m one fits inside a tread once cells are 0.1 m).
    from scipy import ndimage
    size = max(3, 2 * int(round(float(flat_window_m) / (2.0 * cell))) + 1)
    hi_f = ndimage.maximum_filter(np.where(np.isfinite(H), H, np.inf), size=size, mode="nearest")
    lo_f = ndimage.minimum_filter(np.where(np.isfinite(H), H, -np.inf), size=size, mode="nearest")
    flat = ((hi_f - lo_f) <= float(flat_tol_m))[ii, jj]
    goal, reaches = None, False
    if target_y is not None:
        # The storey asked for, by its height -- which a prior map can have
        # wrong by half a metre (00800: 0.64 for a floor at 0.16), so the
        # nearest FLAT reachable height within `storey_tol_m` of it, first
        # arrival first.
        want = float(direction) * (float(target_y) - float(standing_y))
        near = reach & flat & (np.abs(gain - want) <= float(storey_tol_m))
        if near.any():
            sel = np.flatnonzero(near)
            order = np.lexsort((dist[sel], np.round(np.abs(h_node[sel] - float(target_y)) / 0.1)))
            goal, reaches = int(sel[order[0]]), True
    else:
        near = reach & flat & (gain >= float(storey_m))
        if near.any():
            sel = np.flatnonzero(near)
            order = np.lexsort((dist[sel], np.round(gain[sel] / 0.1)))
            goal, reaches = int(sel[order[0]]), True
    if goal is None:
        # No floor within reach yet -- typically the far end of the flight is
        # still unmapped. Walk as far as the map goes; the caller re-plans.
        cap = np.inf if target_y is None else abs(float(target_y) - float(standing_y)) + float(storey_tol_m)
        cand = reach & (gain <= cap)
        if not cand.any():
            return None
        best = float(gain[cand].max())
        if best < float(min_gain_m):
            return None
        near_best = cand & (gain >= best - 0.1)
        goal = int(np.flatnonzero(near_best)[np.argmin(dist[near_best])])

    chain = [goal]
    while pred[chain[-1]] >= 0:
        chain.append(int(pred[chain[-1]]))
    chain = chain[::-1]
    return StairRoute(xy=world[chain], heights=h_node[chain], cost_m=dist[chain],
                      reaches_target=bool(reaches), gain_m=float(gain[goal]))
