"""A walkable route up or down a staircase, read off the height layers.

`stairs.find_flights` finds a staircase as a BLOB of cells whose heights sit
between two storeys and hands the climb its lowest or highest cell. A blob has
no notion of walking: furniture beside the foot, the void seen over a banister
and both legs of a switchback join it, and its extreme cell is wherever the
merge put it. Measured on Benchmark v2 seed 42 (outputs/mfv2/diag_climb): on
00821 the agent stood 1.1 m from the true foot of a switchback and the blob's
foot was 1.5 m to the side of it; the climb held that carrot for 300 steps,
twice, and rose 0.01 m. On 00800 the descent's "top" was 1.9 m down the
flight.

This asks the question a staircase answers by construction: which surfaces
can be WALKED to from where the agent stands, one riser at a time?

A cell can hold two surfaces that matter, and no single height per cell is
right on every staircase. Seen from below, an open flight shows the floor
beneath its treads, which wins the running minimum (00821's switchback); seen
from the storey above, the slab over a lower flight is below the camera and
wins a running maximum (00800's descent read 3.4-4.0 m where the treads are
0.5-1.4); and a single frame through open risers sees both the tread and the
floor behind it. So every cell offers up to two LEVELS -- the lowest surface
seen (`Costmap2D.height`) and the highest seen below the observing camera
(`height_top`) when they differ by more than a riser -- and neighbouring
levels are joined wherever they differ by at most `max_step_m`. Treads chain
from the floor up the top level; the floor beneath them, a slab and a ceiling
are islands no riser reaches. Cells a wall stands on at this storey's floor,
and cells where something stands inside a 0.1 m block, are left out.

The route is the cheapest walk from the agent to FLAT reachable floor near the
storey asked for -- not a landing, flat but a storey-half off, and not the next
storey, a whole one off -- or, when no such floor is mapped yet, as far in the
wanted direction as the map goes, to be re-planned as more comes into view.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .costmap import FREE, OCCUPIED


@dataclass
class StairRoute:
    """A route in world coordinates, the agent's end first."""

    xy: np.ndarray            # (N, 2)
    heights: np.ndarray       # (N,) surface height of each point
    cost_m: np.ndarray        # (N,) walked distance from the start (metres, unpenalised)
    reaches_target: bool      # ends on flat floor near the storey asked for
    gain_m: float             # height gained in the wanted direction

    @property
    def end_xy(self) -> np.ndarray:
        return self.xy[-1]

    def leaves_floor_after_m(self, floor_y: float, tol_m: float = 0.15) -> float:
        """Walking distance before the route first leaves this floor's height."""
        off = np.flatnonzero(np.abs(self.heights - float(floor_y)) > tol_m)
        return float(self.cost_m[off[0]]) if len(off) else float(self.cost_m[-1])


def _blocks(fine: np.ndarray, block: int) -> np.ndarray:
    nr, nc = fine.shape[0] // block, fine.shape[1] // block
    return fine[: nr * block, : nc * block].reshape(nr, block, nc, block).transpose(0, 2, 1, 3) \
        .reshape(nr, nc, block * block)


def _coarse(fine: np.ndarray, block: int):
    """(median, span) of each block's finite values; NaN where it has none.

    The median, because a few stray readings -- a baluster, a gap onto the
    floor below -- must not decide a cell the way a minimum lets them.
    """
    b = _blocks(fine, block)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(b, axis=2)
        span = np.nanmax(b, axis=2) - np.nanmin(b, axis=2)
    return med, span


def _flat(H: np.ndarray, size: int, tol: float) -> np.ndarray:
    from scipy import ndimage

    hi = ndimage.maximum_filter(np.where(np.isfinite(H), H, np.inf), size=size, mode="nearest")
    lo = ndimage.minimum_filter(np.where(np.isfinite(H), H, -np.inf), size=size, mode="nearest")
    return (hi - lo) <= float(tol)


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
    clearance_m: float = 0.2,
    edge_cost: float = 5.0,
    bumps_xy=None,
    blind_fill_m: float = 1.5,
    gap_fill_m: float = 0.0,
    storey_tol_m: float = 0.6,
    storey_m: float = 1.8,
    standing_span_m: float = 0.45,
    debug: Optional[dict] = None,
    explore: bool = False,
    toward_xy=None,
) -> Optional[StairRoute]:
    """The cheapest walk from the agent to the storey asked for.

    `direction` is +1 up, -1 down. The goal is reachable FLAT floor: within
    `storey_tol_m` of `target_y`, or without one the nearest at least
    `storey_m` away. When no such floor is mapped yet, the walk goes as far
    in `direction` as the map allows (at least `min_gain_m`, never more than
    `storey_tol_m` past `target_y`), with `reaches_target` False. None when
    there is no height layer, nowhere to start from, or no such walk.

    `explore` asks for something else: the farthest reachable point at the
    agent's own level (within 0.3 m of it, at most 4 m of walking) -- the far
    end of a landing, from where the next flight can be seen. With
    `toward_xy` it is instead the reachable point at that level nearest to
    `toward_xy` (a known stair mouth), None once the agent is within 0.3 m of
    walking of it.
    """
    from scipy import ndimage
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    height = getattr(costmap, "height", None)
    if height is None:
        return None
    res = float(costmap.resolution)
    block = max(1, int(round(float(cell_m) / res)))
    cell = block * res
    agent_xy = np.asarray(agent_xy, dtype=float)
    rc = costmap.world_to_grid(agent_xy)
    rad = int(np.ceil(float(radius_m) / res))
    h_full, w_full = costmap.grid.shape
    r0, c0 = max(0, int(rc[0]) - rad), max(0, int(rc[1]) - rad)
    r1, c1 = min(h_full, int(rc[0]) + rad + 1), min(w_full, int(rc[1]) + rad + 1)
    r1 = r0 + ((r1 - r0) // block) * block
    c1 = c0 + ((c1 - c0) // block) * block
    if r1 <= r0 or c1 <= c0:
        return None
    grid = costmap.grid[r0:r1, c0:c1]
    low = np.asarray(height[r0:r1, c0:c1], dtype=float)
    top_layer = getattr(costmap, "height_top", None)
    high = (np.asarray(top_layer[r0:r1, c0:c1], dtype=float) if top_layer is not None
            else np.full(low.shape, np.nan))
    # The camera never sees the floor within ~1.4 m of where it stands (0.88 m
    # up, a 31-degree half field of view), so a map built by turning in place
    # has no surface under or around the agent to start from -- measured at
    # the foot of 00821's flight: 71 of 1681 cells within a metre had a height
    # and the planner found nothing, 60 times out of 60. Cells the occupancy
    # grid calls FREE inside that blind ring are floor the agent is standing
    # on or walked across: give them its standing height. Only that close --
    # further out a FREE cell can be a ray passing over a stairwell.
    on_stairs = abs(float(standing_y) - float(floor_y)) > 0.15
    if gap_fill_m > 0.0 and on_stairs and int(direction) < 0:
        # Going down from the stairs, nothing more than a metre above the
        # agent's feet is a surface it will walk on -- but the storey above
        # mapped what overhangs the flight from up there, and a block holding
        # both that and the treads below it "has something standing in it".
        # Measured at the foot of 00873's switchback (standing -2.45): the
        # soffit over the last treads (-0.6 .. 0.9 m) walled off the whole
        # width of the flight 0.5 m ahead and the route was lost two treads
        # from the floor below.
        over = float(standing_y) + 1.0
        low = np.where(low > over, np.nan, low)
        high = np.where(high > over, np.nan, high)
    if gap_fill_m > 0.0:
        # While climbing, only `height_top` is recorded (`ClimbPolicy.
        # _record_climb_surfaces`; the occupancy update is suppressed on a
        # flight), so the treads seen from the stairs have no LOWEST reading.
        # Where the top layer is all there is, it is the surface.
        low = np.where(~np.isfinite(low) & np.isfinite(high), high, low)
    if blind_fill_m > 0.0 and not (gap_fill_m > 0.0 and on_stairs):
        # ...and only from the floor. FREE is this storey's floor-relative
        # grid: rays from the floor above pass over a stairwell and mark its
        # treads FREE, so on the stairs this rule paved the flight ahead at
        # the agent's own height. Measured on 00862's second leg (standing
        # 1.20): every tread ahead read 1.20, the real ones at 0.95 / 0.76 /
        # 0.57 were dropped as the same surface or left unreachable, and the
        # climb stalled 2.2 m down. With `gap_fill_m` on, the stairs get the
        # bridging below instead.
        rr, cc = np.mgrid[r0:r1, c0:c1]
        dx = costmap.origin[0] + (rr + 0.5) * res - float(agent_xy[0])
        dz = costmap.origin[1] + (cc + 0.5) * res - float(agent_xy[1])
        blind = (~np.isfinite(low)) & (grid == FREE) & (dx * dx + dz * dz <= float(blind_fill_m) ** 2)
        low = np.where(blind, float(standing_y), low)
    # On the treads themselves that rule has nothing to work with: they are
    # not FREE in this storey's floor-relative grid, and the tread just ahead
    # stays unseen -- the camera cannot look under itself. Measured descending
    # 00862's switchback (scratchpad trace, the second leg at 1.12 m): the
    # treads at 0.62 and 0.42 m were mapped 0.3-0.5 m ahead, the band between
    # them and the agent was not, and the planner found no route 3 times out
    # of 3 until the climb stalled. Bridge such a band from both sides: an
    # unseen, unoccupied cell within `gap_fill_m` takes the mean of the seen
    # cells around it, a few cells deep, which makes the missing tread a ramp
    # between its neighbours.
    if gap_fill_m > 0.0:
        rr, cc = np.mgrid[r0:r1, c0:c1]
        dx = costmap.origin[0] + (rr + 0.5) * res - float(agent_xy[0])
        dz = costmap.origin[1] + (cc + 0.5) * res - float(agent_xy[1])
        near = (dx * dx + dz * dz <= float(gap_fill_m) ** 2) & (grid != OCCUPIED)
        filled = low.copy()
        for _ in range(max(1, int(np.ceil(0.25 / res)))):
            seen = np.isfinite(filled)
            todo = near & ~seen
            if not todo.any():
                break
            total = ndimage.uniform_filter(np.where(seen, filled, 0.0), size=3, mode="constant")
            count = ndimage.uniform_filter(seen.astype(float), size=3, mode="constant")
            new = todo & (count * 9.0 >= 3.0 - 1e-6)
            if not new.any():
                break
            filled[new] = total[new] / count[new]
        low = filled
    # A wall standing on THIS floor: occupied, and its lowest seen point at the
    # floor. Treads above or below the floor are occupied in a floor-relative
    # grid too, by their own surface, and must stay in.
    wall_fine = (grid == OCCUPIED) & (np.abs(low - float(floor_y)) < float(wall_band_m))

    H0, span0 = _coarse(low, block)
    H1, span1 = _coarse(high, block)
    nr, nc = H0.shape
    wall = _blocks(wall_fine, block).any(axis=2)
    # A block whose readings span more than a riser plus depth noise has
    # something STANDING in it -- a baluster, a newel post, a wall's edge
    # (00800's probe flight: the route cut the inner corner of the L at its
    # foot and pressed into the post for 50 steps).
    valid0 = np.isfinite(H0) & ~wall & ~(np.isfinite(span0) & (span0 > float(standing_span_m)))
    valid1 = np.isfinite(H1) & ~(np.isfinite(span1) & (span1 > float(standing_span_m)))
    # The upper level is a separate surface only where it is more than a riser
    # above the lower one; otherwise the cell has one surface.
    with np.errstate(invalid="ignore"):
        valid1 &= ~(valid0 & (np.abs(H1 - H0) <= float(max_step_m)))
    # Where a forward step was refused (`bumps_xy`): something is there that
    # the layers do not show -- the inner corner of a turning flight.
    if bumps_xy is not None and len(bumps_xy):
        b = np.asarray(bumps_xy, dtype=float).reshape(-1, 2)
        brc = np.floor(((b - costmap.origin[None, :]) / res - np.array([r0, c0])) / block).astype(int)
        inside = (brc[:, 0] >= 0) & (brc[:, 0] < nr) & (brc[:, 1] >= 0) & (brc[:, 1] < nc)
        valid0[brc[inside, 0], brc[inside, 1]] = False
        valid1[brc[inside, 0], brc[inside, 1]] = False
    levels = [(H0, valid0), (H1, valid1)]
    if not (valid0.any() or valid1.any()):
        return None

    # Nodes: every valid (cell, level).
    idx = [-np.ones((nr, nc), dtype=np.int64) for _ in levels]
    n = 0
    for (H, valid), ix in zip(levels, idx):
        k = int(valid.sum())
        ix[valid] = np.arange(n, n + k)
        n += k
    node_i = np.empty(n, dtype=np.int64)
    node_j = np.empty(n, dtype=np.int64)
    node_h = np.empty(n, dtype=float)
    for (H, valid), ix in zip(levels, idx):
        ii, jj = np.nonzero(valid)
        node_i[ix[valid]] = ii
        node_j[ix[valid]] = jj
        node_h[ix[valid]] = H[valid]

    # Keep clear of edges: a cell within `clearance_m` of anything a walker
    # cannot step onto costs `edge_cost` times as much, so the route runs down
    # the middle of a flight rather than along its banister.
    blocked = ~(valid0 | valid1)
    Hf = np.where(valid0, H0, np.where(valid1, H1, np.nan))
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        rs, rt = slice(0, nr - dr), slice(dr, nr)
        cs, ct = (slice(0, nc - dc), slice(dc, nc)) if dc >= 0 else (slice(-dc, nc), slice(0, nc + dc))
        with np.errstate(invalid="ignore"):
            jump = np.abs(Hf[rs, cs] - Hf[rt, ct]) > float(max_step_m)
        blocked[rs, cs] |= jump
        blocked[rt, ct] |= jump
    near_edge = ndimage.binary_dilation(blocked, iterations=max(1, int(np.ceil(float(clearance_m) / cell))))
    penalty = np.where(near_edge, float(edge_cost), 1.0)

    rows, cols, wts = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        rs, rt = slice(0, nr - dr), slice(dr, nr)
        cs, ct = (slice(0, nc - dc), slice(dc, nc)) if dc >= 0 else (slice(-dc, nc), slice(0, nc + dc))
        step = cell * (np.sqrt(2.0) if dr and dc else 1.0)
        w_pair = step * 0.5 * (penalty[rs, cs] + penalty[rt, ct])
        for (Ha, va), ia_map in zip(levels, idx):
            for (Hb, vb), ib_map in zip(levels, idx):
                with np.errstate(invalid="ignore"):
                    ok = va[rs, cs] & vb[rt, ct] & (np.abs(Ha[rs, cs] - Hb[rt, ct]) <= float(max_step_m))
                if not ok.any():
                    continue
                rows.append(ia_map[rs, cs][ok])
                cols.append(ib_map[rt, ct][ok])
                wts.append(w_pair[ok])
    if not rows:
        return None
    graph = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n, n)).tocsr()

    world = costmap.origin[None, :] + np.stack(
        [r0 + (node_i + 0.5) * block, c0 + (node_j + 0.5) * block], axis=1) * res
    d_agent = np.linalg.norm(world - agent_xy[None, :], axis=1)
    sources = np.flatnonzero((d_agent <= float(seed_m)) & (np.abs(node_h - float(standing_y)) <= 0.2))
    if not len(sources):
        return None
    dist, pred, _ = dijkstra(graph, directed=False, indices=sources,
                             min_only=True, return_predecessors=True)
    if debug is not None:
        debug.update(H0=H0, H1=H1, valid0=valid0, valid1=valid1, node_i=node_i,
                     node_j=node_j, node_h=node_h, sources=sources, dist=dist,
                     n_edges=int(sum(len(r) for r in rows)))
    reach = np.isfinite(dist)
    gain = float(direction) * (node_h - float(standing_y))
    size = max(3, 2 * int(round(float(flat_window_m) / (2.0 * cell))) + 1)
    flat = np.zeros(n, dtype=bool)
    for (H, valid), ix in zip(levels, idx):
        f = _flat(np.where(valid, H, np.nan), size, flat_tol_m)
        flat[ix[valid]] = f[valid]

    goal, reaches = None, False
    if explore:
        level = reach & (np.abs(node_h - float(standing_y)) <= 0.3) & (dist <= 4.0)
        if not level.any():
            return None
        if toward_xy is not None:
            sel = np.flatnonzero(level)
            off = np.linalg.norm(world[sel] - np.asarray(toward_xy, dtype=float)[None, :], axis=1)
            goal = int(sel[np.lexsort((dist[sel], np.round(off / 0.1)))[0]])
            if float(dist[goal]) < 0.3:
                return None
        else:
            goal = int(np.flatnonzero(level)[np.argmax(dist[level])])
            if float(dist[goal]) < 0.8:
                return None
    elif target_y is not None:
        # The storey asked for, by its height -- which a prior map can have
        # wrong by half a metre (00800: 0.64 for a floor at 0.16), so the
        # nearest FLAT reachable height within `storey_tol_m` of it, first
        # arrival first.
        want = float(direction) * (float(target_y) - float(standing_y))
        near = reach & flat & (np.abs(gain - want) <= float(storey_tol_m))
        if near.any():
            sel = np.flatnonzero(near)
            order = np.lexsort((dist[sel], np.round(np.abs(node_h[sel] - float(target_y)) / 0.1)))
            goal, reaches = int(sel[order[0]]), True
    else:
        near = reach & flat & (gain >= float(storey_m))
        if near.any():
            sel = np.flatnonzero(near)
            order = np.lexsort((dist[sel], np.round(gain[sel] / 0.1)))
            goal, reaches = int(sel[order[0]]), True
    if goal is None and not explore:
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
    xy = world[chain]
    walked = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    return StairRoute(xy=xy, heights=node_h[chain], cost_m=walked,
                      reaches_target=bool(reaches), gain_m=float(gain[goal]))
