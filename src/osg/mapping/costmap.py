"""2D occupancy costmap built from depth frames.

World plane axes are (x, z) with y as height (habitat convention, y-up).
Grid values: -1 unknown, 0 free, 100 occupied. The grid auto-grows.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from ..core.geometry import backproject
from ..core.types import FrameData

UNKNOWN, FREE, OCCUPIED = -1, 0, 100
PLANE = (0, 2)  # world axes forming the ground plane
HEIGHT_AXIS = 1


class Costmap2D:
    def __init__(
        self, resolution: float = 0.05, size_m: float = 20.0, track_height: bool = False,
        track_top: bool = False,
    ) -> None:
        self.resolution = resolution
        n = int(size_m / resolution)
        self.grid = np.full((n, n), UNKNOWN, dtype=np.int8)
        self.origin = np.array([-size_m / 2.0, -size_m / 2.0])  # world xy of grid[0, 0]
        # Lowest observed surface height per cell, world y, NaN where unseen.
        # The MINIMUM (not the mean) because we want the surface you could
        # stand on: a table top or a chair back in the same cell must not hide
        # the floor beneath it. Only allocated when asked for -- it is float32,
        # 4x the grid, and only the stair detector needs it.
        self.height: Optional[np.ndarray] = (
            np.full((n, n), np.nan, dtype=np.float32) if track_height else None
        )
        # Cells confirmed as traversable stairs. Written FREE and, crucially,
        # exempted from the OCCUPIED endpoint stamp in _raycast_batch -- without
        # that exemption the next frame re-stamps them and the relabel is a
        # no-op, because OCCUPIED is never otherwise cleared.
        self.stair_mask: Optional[np.ndarray] = None
        # Which height cells are INVENTED -- the linear ramp a pasted ASCENT
        # map gets in place of the heights it never recorded. A real depth
        # reading replaces one of these outright instead of being min'd against
        # it (see `_record_heights`).
        self.height_synthetic: Optional[np.ndarray] = (
            np.zeros((n, n), dtype=bool) if track_height else None
        )
        # Highest surface seen per cell BELOW the camera that saw it, world y,
        # NaN where unseen -- the stair route's layer (`mapping/stair_route.py`).
        # `height` keeps the minimum, which is the right surface to stand on
        # everywhere except under an open staircase: seen from below, the floor
        # beneath the treads wins the minimum and the upper flight is gone for
        # good. The maximum below the camera keeps the treads, and makes walls
        # and railings tall instead of floor-high. Only with `track_top`.
        self.height_top: Optional[np.ndarray] = (
            np.full((n, n), np.nan, dtype=np.float32) if (track_height and track_top) else None
        )
        self._grow_listeners: List[Callable[[int, int, int, int], None]] = []

    def add_grow_listener(self, fn: Callable[[int, int, int, int], None]) -> None:
        self._grow_listeners.append(fn)

    # ------------------------------------------------------------- transforms

    def world_to_grid(self, xy: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(xy) - self.origin) / self.resolution).astype(int)

    def grid_to_world(self, rc: np.ndarray) -> np.ndarray:
        return self.origin + (np.asarray(rc, dtype=float) + 0.5) * self.resolution

    def in_bounds(self, rc: np.ndarray) -> bool:
        return 0 <= rc[0] < self.grid.shape[0] and 0 <= rc[1] < self.grid.shape[1]

    def clear_footprint(self, xy: np.ndarray, radius_m: float) -> int:
        """The robot is standing here, so these cells are free. Returns how
        many cells changed.

        What Nav2's costmaps do as a matter of course and the simulator never
        needed: its agent sees the floor at its feet, a head camera at 1.3 m
        does not, and a glossy floor or the robot's own mast in a tilted view
        puts obstacle speckle exactly where the base is. Measured on the
        Stretch (outputs/20260922_232853): 23 of 101 poses on cells the map
        called OCCUPIED, 57 on free islands under 50 cells, 98 free
        components -- and wavefront frontier detection, which grows from the
        robot's own component, found nothing from a 9-cell island.
        """
        r = int(np.ceil(float(radius_m) / self.resolution))
        if r <= 0:
            return 0
        rc = self.world_to_grid(xy)
        if not self.in_bounds(rc):
            return 0
        h, w = self.grid.shape
        r0, r1 = max(0, rc[0] - r), min(h, rc[0] + r + 1)
        c0, c1 = max(0, rc[1] - r), min(w, rc[1] + r + 1)
        rr, cc = np.ogrid[r0:r1, c0:c1]
        disc = (rr - rc[0]) ** 2 + (cc - rc[1]) ** 2 <= r * r
        block = self.grid[r0:r1, c0:c1]
        changed = int((disc & (block != FREE)).sum())
        block[disc] = FREE
        return changed

    # The parallel layers -- one entry per cell of `grid`, and meaningless if
    # their shapes ever diverge from it.
    _PARALLEL_LAYERS = (("height", np.nan, np.float32),
                        ("stair_mask", False, bool),
                        ("height_synthetic", False, bool),
                        ("height_top", np.nan, np.float32))

    def conform_layers(self) -> None:
        """Re-shape every parallel layer to match `grid`, blanking what cannot
        be carried over.

        `ensure_contains` keeps them together when the map GROWS. Restoring a
        snapshot is the other way a grid changes shape -- it is replaced
        wholesale, at whatever extent the mapping pass had grown to -- and
        nothing was keeping the layers with it. A 400x400 `height_synthetic`
        behind a restored 576x576 grid indexes out of bounds on the first depth
        frame (`_record_heights`), which is an IndexError on episode 1 of every
        run that loads a prior map with occupancy.
        """
        shape = self.grid.shape
        for name, fill, dtype in self._PARALLEL_LAYERS:
            layer = getattr(self, name, None)
            if layer is not None and layer.shape != shape:
                setattr(self, name, np.full(shape, fill, dtype=dtype))

    def ensure_contains(self, xy: np.ndarray, margin_m: float = 2.0) -> None:
        rc = self.world_to_grid(xy)
        m = int(margin_m / self.resolution)
        h, w = self.grid.shape
        if 0 + m <= rc[0] < h - m and 0 + m <= rc[1] < w - m:
            return
        # Double the grid, keeping content centered
        new = np.full((h * 2, w * 2), UNKNOWN, dtype=np.int8)
        new[h // 2 : h // 2 + h, w // 2 : w // 2 + w] = self.grid
        self.grid = new
        # Every parallel layer must grow with it or the shapes silently diverge
        for name, fill, dtype in self._PARALLEL_LAYERS:
            old = getattr(self, name, None)
            if old is not None:
                grown = np.full((h * 2, w * 2), fill, dtype=dtype)
                grown[h // 2 : h // 2 + h, w // 2 : w // 2 + w] = old
                setattr(self, name, grown)
        self.origin = self.origin - np.array([h // 2, w // 2]) * self.resolution
        for listener in self._grow_listeners:
            listener(h, w, h // 2, w // 2)
        self.ensure_contains(xy, margin_m)

    # ---------------------------------------------------------------- update

    def update(
        self,
        frame: FrameData,
        floor_y: float,
        obstacle_low: float = 0.2,
        obstacle_high: float = 1.5,
        max_range: float = 5.0,
        stride: int = 4,
        pts: Optional[np.ndarray] = None,
        height_span: float = 3.5,
    ) -> None:
        """Fold one frame into the grid, banding points by height above
        `floor_y`. Points outside the band are dropped, so with a per-floor
        stack each storey only ever absorbs its own geometry.

        `pts` accepts an already-backprojected (N, 3) world point cloud. The
        backprojection dominates mapping cost, so a caller feeding several
        floors from one frame must share it rather than recompute per floor.

        `height_span` is how far from `floor_y` the optional height layer
        records, in BOTH directions and independent of the obstacle band. Above,
        it holds a whole staircase; below, it catches the floor seen down a
        stairwell -- which is what makes a descending portal visible at all,
        since the occupancy band stops 0.3 m under the agent's feet.
        """
        cam_xy = frame.camera_position[list(PLANE)]
        self.ensure_contains(cam_xy, margin_m=max_range + 1.0)

        if pts is None:
            pts = backproject(
                frame.depth, frame.intrinsics, frame.T_wc, stride=stride, max_depth=max_range
            )
        if pts.shape[0] == 0:
            return
        rel_h = pts[:, HEIGHT_AXIS] - floor_y
        xy = pts[:, list(PLANE)]

        floor_mask = (rel_h > -0.3) & (rel_h < obstacle_low)
        obst_mask = (rel_h >= obstacle_low) & (rel_h < obstacle_high)

        # The height layer is a TRAVERSABILITY layer, not an occupancy one, so
        # it must not inherit the occupancy band. A staircase climbs a whole
        # storey (2.5-3.4 m in HM3D); banding its heights at obstacle_high
        # (1.5 m) truncates it to less rise than it takes to recognise one, and
        # the detector then finds nothing even while the agent is walking up it.
        # Taking the per-cell MINIMUM makes the wider range safe: ceilings and
        # tall furniture never displace the surface underneath them.
        if self.height is not None:
            self._record_heights(pts[np.abs(rel_h) < height_span])
        if self.height_top is not None:
            self.record_top(pts[np.abs(rel_h) < height_span],
                            cap_y=float(frame.camera_position[HEIGHT_AXIS]))

        cam_rc = self.world_to_grid(cam_xy)
        self._raycast_batch(cam_rc, xy[floor_mask], xy[obst_mask])
        # The agent's own cell is free by construction
        if self.in_bounds(cam_rc):
            self.grid[cam_rc[0], cam_rc[1]] = FREE

    def _raycast_batch(
        self, cam_rc: np.ndarray, floor_xy: np.ndarray, obst_xy: np.ndarray
    ) -> None:
        """Vectorized raycasting: all rays sampled as a (M, N) grid of cells.
        Intermediate unknown/free cells become FREE; rays stop at the first
        already-occupied cell (never carve walls); obstacle endpoints are
        stamped OCCUPIED last. The per-point python Bresenham this replaces
        dominated the control loop (~1.6 s/frame)."""
        ends = []
        occ_flags = []
        for pts, occ in ((floor_xy, False), (obst_xy, True)):
            if pts.shape[0] == 0:
                continue
            rc = np.floor((pts - self.origin) / self.resolution).astype(np.int64)
            rc, idx = np.unique(rc, axis=0, return_index=True)
            ends.append(rc)
            occ_flags.append(np.full(rc.shape[0], occ))
        if not ends:
            return
        end_rc = np.concatenate(ends)  # (N, 2)
        end_occ = np.concatenate(occ_flags)  # (N,)
        h, w = self.grid.shape
        inb = (end_rc[:, 0] >= 0) & (end_rc[:, 0] < h) & (end_rc[:, 1] >= 0) & (end_rc[:, 1] < w)
        end_rc, end_occ = end_rc[inb], end_occ[inb]
        if end_rc.shape[0] == 0:
            return

        delta = end_rc - cam_rc[None, :]
        n_steps = int(np.abs(delta).max()) + 1
        # 2x supersampling closes diagonal gaps a true Bresenham would fill
        m = min(2 * n_steps + 1, 4096)
        t = np.linspace(0.0, 1.0, m)[:, None, None]
        samples = np.rint(cam_rc[None, None, :] + t * delta[None, :, :]).astype(np.int64)
        rr = np.clip(samples[..., 0], 0, h - 1)
        cc = np.clip(samples[..., 1], 0, w - 1)

        vals = self.grid[rr, cc]  # (M, N)
        blocked = vals == OCCUPIED
        any_blocked = blocked.any(axis=0)
        first_block = np.where(any_blocked, blocked.argmax(axis=0), m)  # (N,)
        step_idx = np.arange(m)[:, None]
        visible = step_idx < first_block[None, :]  # strictly before the wall

        self.grid[rr[visible], cc[visible]] = FREE
        # Obstacle endpoints whose ray was not blocked earlier
        reached = ~any_blocked | (first_block >= m - 1)
        stamp = end_occ & reached
        if self.stair_mask is not None:
            # Confirmed stairs are traversable, so their rising treads must not
            # be stamped as obstacles. This exemption is what makes the FREE
            # relabel stick: OCCUPIED is never cleared once written, so without
            # it the next frame would restore the wall across the staircase.
            stamp = stamp & ~self.stair_mask[end_rc[:, 0], end_rc[:, 1]]
        self.grid[end_rc[stamp, 0], end_rc[stamp, 1]] = OCCUPIED

    def _record_heights(self, pts: np.ndarray) -> None:
        """Keep the lowest surface height seen in each cell.

        EXCEPT where the height is SYNTHETIC. A pasted ASCENT map carries no
        heights, so `map_store._ramp_stair_heights` invents a linear ramp over
        the stair cells -- and the invention runs LOW: measured on 00821 it
        spans 3.17 m where the true surface under the same cells spans 1.34,
        reaching -3.26 where the truth is -1.21. Under a running minimum the
        agent's own depth reading, which is higher and correct, can never
        displace it, so the climber follows the fiction for the whole episode.
        A first real observation therefore REPLACES a synthetic height and
        clears the flag; from then on the cell behaves normally.
        """
        if pts.shape[0] == 0:
            return
        rc = np.floor((pts[:, list(PLANE)] - self.origin) / self.resolution).astype(np.int64)
        h, w = self.grid.shape
        inb = (rc[:, 0] >= 0) & (rc[:, 0] < h) & (rc[:, 1] >= 0) & (rc[:, 1] < w)
        rc, ys = rc[inb], pts[inb, HEIGHT_AXIS].astype(np.float32)
        if rc.shape[0] == 0:
            return
        flat = rc[:, 0] * w + rc[:, 1]
        cur = self.height.reshape(-1)
        synthetic = getattr(self, "height_synthetic", None)
        if synthetic is not None:
            flags = synthetic.reshape(-1)
            fake = flags[flat]
            if fake.any():
                # Last real reading wins over an invented one, then the cell is
                # real and rejoins the running minimum below.
                cur[flat[fake]] = ys[fake]
                flags[flat[fake]] = False
                keep = ~fake
                flat, ys = flat[keep], ys[keep]
                if flat.size == 0:
                    return
        # np.fmin.at ignores NaN on the accumulator side, so first-write cells
        # take the value and later writes keep the running minimum.
        np.fmin.at(cur, flat, ys)

    def record_top(self, pts: np.ndarray, cap_y: float) -> None:
        """Fold world points into `height_top`: per cell, the highest point at
        or below `cap_y` (the observing camera's height). A no-op without the
        layer. Public because the climb records it while the occupancy update
        is suppressed on the stairs."""
        if self.height_top is None or pts.shape[0] == 0:
            return
        pts = pts[pts[:, HEIGHT_AXIS] <= float(cap_y)]
        if pts.shape[0] == 0:
            return
        rc = np.floor((pts[:, list(PLANE)] - self.origin) / self.resolution).astype(np.int64)
        h, w = self.grid.shape
        inb = (rc[:, 0] >= 0) & (rc[:, 0] < h) & (rc[:, 1] >= 0) & (rc[:, 1] < w)
        if not inb.any():
            return
        flat = rc[inb, 0] * w + rc[inb, 1]
        np.fmax.at(self.height_top.reshape(-1), flat, pts[inb, HEIGHT_AXIS].astype(np.float32))

    def height_gradient(self, cell_m: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """ZONDA's traversability signal, coarsened to `cell_m`.

        Returns `(dh, valid)` on the coarse grid: `dh[i, j]` is the largest
        height difference between a cell and its 8 neighbours. Flat floor gives
        ~0, a wall gives a large value, and a staircase sits in between at
        roughly its riser height -- which is what separates "step up" from
        "cannot pass" without any learned model.
        """
        from scipy import ndimage

        if self.height is None:
            raise ValueError("costmap was built without track_height=True")
        block = max(1, int(round(cell_m / self.resolution)))
        coarse = block_min(self.height, block)

        valid = np.isfinite(coarse)
        filled = np.where(valid, coarse, np.nan)
        # min/max filters propagate NaN, so fill with +-inf per direction and
        # mask the result back to cells that had real data on both sides.
        lo = ndimage.minimum_filter(np.where(valid, filled, np.inf), size=3, mode="nearest")
        hi = ndimage.maximum_filter(np.where(valid, filled, -np.inf), size=3, mode="nearest")
        dh = np.where(valid & np.isfinite(lo) & np.isfinite(hi), hi - lo, np.nan)
        return dh, valid

    # ------------------------------------------------------------------ views

    def inflated(self, radius_m: float) -> np.ndarray:
        """Boolean obstacle map dilated by radius (for planning)."""
        from scipy import ndimage

        obst = self.grid == OCCUPIED
        r = max(1, int(round(radius_m / self.resolution)))
        struct = _disk(r)
        return ndimage.binary_dilation(obst, structure=struct)

    def free_mask(self) -> np.ndarray:
        return self.grid == FREE

    def coverage_cells(self) -> int:
        return int((self.grid != UNKNOWN).sum())


def grow_aligned(
    arr: np.ndarray, old_h: int, old_w: int, off_r: int, off_c: int, fill=0
) -> np.ndarray:
    """Grow a grid-aligned side array after its costmap doubles in size."""
    new = np.full((old_h * 2, old_w * 2), fill, dtype=arr.dtype)
    new[off_r : off_r + old_h, off_c : off_c + old_w] = arr
    return new


def block_min(height, block):
    """Block-reduce a height map by minimum, NaN-safe and warning-free.

    np.nanmin raises "All-NaN slice encountered" for blocks that are entirely
    unobserved, which is the common case early in an episode. Substituting +inf
    and mapping it back to NaN avoids both the warning and the per-slice check.
    """
    import numpy as np

    h, w = height.shape
    hh, ww = (h // block) * block, (w // block) * block
    view = height[:hh, :ww].reshape(hh // block, block, ww // block, block)
    filled = np.where(np.isnan(view), np.inf, view)
    out = filled.min(axis=3).min(axis=1)
    return np.where(np.isfinite(out), out, np.nan)


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return x * x + y * y <= r * r


def nearest_free_xy(costmap: "Costmap2D", xy: np.ndarray) -> np.ndarray:
    """Nearest FREE cell to a (possibly occupied) world point -- the closest
    pose the agent can actually stand at.

    Used wherever a goal is derived from an OBJECT rather than from free space:
    a tabletop object's centre is an occupied cell inside the furniture, and
    driving to it strands the follower against the desk. Measured over 42
    episodes, the 10 that ended on such a goal scored SR 0.100 against 0.516 for
    the rest.
    """
    rc = costmap.world_to_grid(xy)
    h, w = costmap.grid.shape
    rad = int(1.5 / costmap.resolution)
    r0, r1 = max(0, rc[0] - rad), min(h, rc[0] + rad + 1)
    c0, c1 = max(0, rc[1] - rad), min(w, rc[1] + rad + 1)
    free = np.argwhere(costmap.grid[r0:r1, c0:c1] == FREE)
    if free.shape[0] == 0:
        return xy
    free_world = costmap.grid_to_world(free + np.array([r0, c0]))
    d = np.linalg.norm(free_world - xy, axis=1)
    return free_world[int(np.argmin(d))]


def cell_status(costmap: "Costmap2D", xy: np.ndarray) -> str:
    """Costmap classification of a world point: free/occupied/unknown/oob."""
    rc = costmap.world_to_grid(xy)
    h, w = costmap.grid.shape
    if not (0 <= rc[0] < h and 0 <= rc[1] < w):
        return "oob"
    v = costmap.grid[rc[0], rc[1]]
    return {FREE: "free", OCCUPIED: "occupied", UNKNOWN: "unknown"}.get(int(v), str(int(v)))
