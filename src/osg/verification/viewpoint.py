"""Improvement C, part 1: approach-viewpoint planning for the last-mile
problem — pick an unobstructed, well-placed pose from which to verify (and
declare) the candidate target.
"""
from __future__ import annotations

from typing import List, Optional  # noqa: F401, Sequence

import numpy as np
from scipy import ndimage

from ..core.geometry import bresenham
from ..mapping.costmap import FREE, OCCUPIED, Costmap2D


def nearest_clear_xy(costmap: Costmap2D, obj_xy: np.ndarray, min_clearance_m: float,
                     max_radius_m: float = 2.0) -> Optional[np.ndarray]:
    """Nearest cell to `obj_xy` the agent actually fits in, from the costmap alone.

    The sensor-only answer to the question the closing walk asks the pathfinder:
    where is the nearest point to this object that I could stand on? The
    navmesh answers it from ground-truth floor geometry; this answers it from
    the agent's own depth-built occupancy grid, by requiring `min_clearance_m`
    of distance to the nearest OCCUPIED cell.

    Returns None when nothing within `max_radius_m` qualifies -- an honest
    answer, and the caller then simply does not walk.
    """
    clearance = ndimage.distance_transform_edt(costmap.grid != OCCUPIED) * costmap.resolution
    best, best_d = None, float("inf")
    step = max(costmap.resolution, 0.05)
    r = step
    while r <= max_radius_m + 1e-9:
        n = max(8, int(round(2.0 * np.pi * r / step)))
        for k in range(n):
            ang = 2.0 * np.pi * k / n
            cand = obj_xy + r * np.array([np.cos(ang), np.sin(ang)])
            rc = costmap.world_to_grid(cand)
            if not costmap.in_bounds(rc):
                continue
            if costmap.grid[rc[0], rc[1]] != FREE:
                continue
            if float(clearance[rc[0], rc[1]]) < min_clearance_m:
                continue
            d = float(np.linalg.norm(cand - obj_xy))
            if d < best_d:
                best, best_d = cand, d
        if best is not None:
            return best  # rings grow outward, so the first hit is the nearest
        r += step
    return None


class ViewpointPlanner:
    def __init__(self, ring_radii_m: Optional[List[float]] = None, n_samples: int = 16,
                 min_clearance_m: float = 0.0) -> None:
        self.ring_radii = ring_radii_m or [0.8, 1.2, 1.5, 2.0]
        self.n_samples = n_samples
        # A cell marked FREE is not a cell the agent fits in. `clearance` (the
        # distance transform below) was computed and then used only to SCORE
        # candidates, so a pose 5 cm from an occupied cell could win a ring and
        # become the approach goal. Habitat's navmesh excludes such a pose, and
        # measuring the sensor arm's stranded approaches against it, 27 of 44
        # goals were non-navigable against 17 of 17 navigable among those that
        # arrived (docs/history/WHY_THE_SENSOR_ARM_LOSES.md).
        #
        # The navmesh follower hid this: it SNAPPED the goal to the mesh, so an
        # unstandable goal silently became a standable one. The PointNav mover
        # takes the goal as a bearing and walks into it.
        #
        # Requiring clearance is the sensor-only half of that snap: the costmap
        # is the agent's own depth map, so this asks nothing of the simulator.
        # 0.0 keeps the old behaviour, which every measured arm ran on.
        self.min_clearance_m = float(min_clearance_m)

    def approach_viewpoint(
        self,
        obj_xy: np.ndarray,
        costmap: Costmap2D,
        exclude: Optional[List[np.ndarray]] = None,
        exclude_radius_m: float = 0.5,
        require_line_of_sight: bool = True,
        allow_unknown: bool = False,
        obj_radius_m: float = 0.0,
        radii: Optional[Sequence[float]] = None,
    ) -> Optional[np.ndarray]:
        """Best world-xy pose to observe the object from, or None if the
        object is not yet observable from mapped free space. `exclude` lists
        previously tried viewpoints (e.g., where the detector could not see
        the object due to 3D occlusion the 2D map misses).

        `obj_radius_m` measures the rings from the object's estimated SURFACE
        rather than its centre, and `radii` replaces the configured ladder for
        one call (a close look wants a single 1.5 m ring). The radii were chosen
        for tabletop YCB objects,
        whose extent is a few centimetres and so does not matter; on a couch or a
        counter the centre can be a metre inside the furniture, and a ring drawn
        around it puts every sample inside the object. Passing the track's
        horizontal semi-axis pushes the whole ladder outwards by that much, which
        is what lets the innermost ring be small enough to score without the goal
        landing in an occupied cell. 0.0 reproduces the original behaviour.

        The two relaxations exist for the fallback in `_start_approach`, and
        both are about the difference between "I know this is bad" and "I do not
        know yet". `allow_unknown` accepts UNKNOWN cells: unmapped is not
        unstandable, and the navmesh follower will find out for real. Dropping
        `require_line_of_sight` accepts a pose whose ray to the object crosses
        occupied cells, which on a 2D costmap includes the object's own
        supporting table. A relaxed viewpoint is a worse place to stand than a
        strict one; it is a far better place than the object's own centre, which
        is what the fallback used to be."""
        exclude = exclude or []
        clearance = ndimage.distance_transform_edt(costmap.grid != OCCUPIED) * costmap.resolution
        best, best_score = None, -1.0
        for ring in (self.ring_radii if radii is None else [float(r) for r in radii]):
            radius = float(obj_radius_m) + float(ring)
            for k in range(self.n_samples):
                ang = 2.0 * np.pi * k / self.n_samples
                cand = obj_xy + radius * np.array([np.cos(ang), np.sin(ang)])
                if any(np.linalg.norm(cand - e) < exclude_radius_m for e in exclude):
                    continue
                rc = costmap.world_to_grid(cand)
                if not costmap.in_bounds(rc):
                    continue
                cell = costmap.grid[rc[0], rc[1]]
                if cell == OCCUPIED or (cell != FREE and not allow_unknown):
                    continue
                if (self.min_clearance_m > 0.0
                        and float(clearance[rc[0], rc[1]]) < self.min_clearance_m):
                    continue  # the agent does not fit here
                if require_line_of_sight and not self._line_of_sight(costmap, cand, obj_xy):
                    continue
                # Prefer clearance and a mid-range viewing distance (~1.2 m)
                score = float(clearance[rc[0], rc[1]]) - 0.3 * abs(radius - 1.2)
                if score > best_score:
                    best, best_score = cand, score
            if best is not None:
                return best  # nearest ring with a valid view wins
        return None

    @staticmethod
    def _line_of_sight(costmap: Costmap2D, from_xy: np.ndarray, to_xy: np.ndarray) -> bool:
        """The ray must not cross occupied cells (cells adjacent to the
        object itself are excluded — the object is an obstacle)."""
        rc0 = costmap.world_to_grid(from_xy)
        rc1 = costmap.world_to_grid(to_xy)
        r1, c1 = int(rc1[0]), int(rc1[1])
        skip_near = max(2, int(0.3 / costmap.resolution))  # cells near the object
        for r, c in bresenham(int(rc0[0]), int(rc0[1]), r1, c1):
            if np.hypot(r1 - r, c1 - c) <= skip_near:
                continue
            if costmap.in_bounds(np.array([r, c])) and costmap.grid[r, c] == OCCUPIED:
                return False
        return True
