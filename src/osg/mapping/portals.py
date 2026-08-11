"""Cross-floor portals: where can the agent see another storey from here?

The measured failure this addresses: on the 100-episode v1 run, **all 24
cross-floor episodes failed and not one of them ever attempted a floor
change**. The agent could not fail to climb the stairs -- it never tried.
Stage 2 showed the mechanics work once a transition is committed (the Habitat
navmesh walks the stairs by itself), so the missing piece is purely the
decision, and the decision needs two things: knowing another floor exists, and
having somewhere to head for.

Both fall out of the height layer that Stage 4 added. Wherever the agent can
see another storey -- up a stairwell, over a mezzanine rail, down an opening --
the cells it observes record a surface roughly a storey away from the floor it
is standing on. Those cells are a **portal**: not the staircase itself, but a
place to walk toward, after which the navmesh handles the climb.

This deliberately does NOT depend on `mapping.stairs`, which does not detect
staircases (see docs/MULTI_FLOOR.md). It needs only the height layer, and it
works for descending portals too, which are invisible to a forward-facing
obstacle band.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .costmap import HEIGHT_AXIS, PLANE, Costmap2D


@dataclass
class Portal:
    """A patch of another storey, visible from the current one."""

    centroid_xy: np.ndarray   # world (x, z) to head toward
    target_y: float           # median surface height of the patch
    n_cells: int
    delta_y: float            # signed: + is up, - is down

    @property
    def going_up(self) -> bool:
        return self.delta_y > 0


def find_portals(
    costmap: Costmap2D,
    floor_y: float,
    min_delta_m: float = 1.8,
    max_delta_m: float = 4.0,
    min_cells: int = 20,
    merge_m: float = 1.5,
) -> List[Portal]:
    """Patches of the height layer that sit a storey away from `floor_y`.

    `min_delta_m` matches the floor estimator's `new_level_m`: below it the
    surface is a split level or furniture, not another storey. `max_delta_m`
    rejects readings two floors off, which are geometrically possible down a
    tall atrium but useless as a next goal.
    """
    from scipy import ndimage

    if costmap.height is None:
        return []
    rel = costmap.height - floor_y
    other = np.isfinite(rel) & (np.abs(rel) >= min_delta_m) & (np.abs(rel) <= max_delta_m)
    if not other.any():
        return []

    labels, n = ndimage.label(other, structure=np.ones((3, 3)))
    portals: List[Portal] = []
    for lbl in range(1, n + 1):
        rc = np.argwhere(labels == lbl)
        if rc.shape[0] < min_cells:
            continue
        ys = costmap.height[labels == lbl]
        ys = ys[np.isfinite(ys)]
        if ys.size == 0:
            continue
        target = float(np.median(ys))
        portals.append(
            Portal(
                centroid_xy=costmap.grid_to_world(rc.mean(axis=0)),
                target_y=target,
                n_cells=int(rc.shape[0]),
                delta_y=target - float(floor_y),
            )
        )
    return _merge(sorted(portals, key=lambda p: -p.n_cells), merge_m)


def portals_from_stair_points(
    points: np.ndarray,
    floor_y: float,
    min_points: int = 60,
    min_rise_m: float = 0.35,
    storey_guess_m: float = 2.8,
    cluster_m: float = 1.5,
) -> List[Portal]:
    """Turn observed staircase surface points into portals.

    The height-layer portal (`find_portals`) needs to see the *other floor's
    surface*, which usually means standing where you can already look onto it.
    A staircase is visible from much further away and from far more poses --
    measured, the height-layer route left `portals_seen == 0` in 14 of 24
    cross-floor episodes.

    The target height is an estimate, not an observation: from the bottom of a
    flight you see the first metre or so, not the landing. We take the highest
    point actually seen and, if that is less than a storey up, extrapolate to
    `storey_guess_m`. `snap_point` then puts the goal on whatever navmesh
    surface is nearest that query, so a wrong guess costs an imprecise goal
    rather than a wrong floor. Descending flights (highest point below the
    floor) extrapolate downward the same way.
    """
    if points is None or len(points) < min_points:
        return []
    pts = np.asarray(points, dtype=float)
    rel = pts[:, HEIGHT_AXIS] - floor_y
    up = float(rel.max())
    down = float(rel.min())
    going_up = abs(up) >= abs(down)
    rise = up if going_up else down
    if abs(rise) < min_rise_m:
        return []

    # Cluster in the ground plane so two staircases do not merge into one goal.
    out: List[Portal] = []
    for centre, n in _cluster_xy(pts[:, list(PLANE)], cluster_m):
        delta = rise if abs(rise) >= storey_guess_m else (
            storey_guess_m if going_up else -storey_guess_m
        )
        out.append(
            Portal(
                centroid_xy=centre,
                target_y=float(floor_y + delta),
                n_cells=int(n),
                delta_y=float(delta),
            )
        )
    return out


def find_descent_portals(
    points: np.ndarray,
    floor_y: float,
    min_points: int = 60,
    storey_guess_m: float = 2.8,
    cluster_m: float = 1.5,
) -> List[Portal]:
    """Portals from points lying below the current floor (see stairs.descent_points).

    A descending opening shows only its first half-metre from a few metres back,
    so the drop is extrapolated to a storey rather than trusted as measured.
    """
    if points is None or len(points) < min_points:
        return []
    pts = np.asarray(points, dtype=float)
    drop = float((pts[:, HEIGHT_AXIS] - floor_y).min())
    delta = drop if abs(drop) >= storey_guess_m else -storey_guess_m
    return [
        Portal(centroid_xy=centre, target_y=float(floor_y + delta),
               n_cells=int(n), delta_y=float(delta))
        for centre, n in _cluster_xy(pts[:, list(PLANE)], cluster_m)
    ]


def _cluster_xy(xy: np.ndarray, radius_m: float):
    """Single-linkage ground-plane clustering; (centroid, count), largest first.

    Single linkage rather than a fixed radius from a seed, because a staircase
    is ELONGATED: a 2 m flight clustered at a 1.5 m radius splits into two
    goals, and a longer one into more. Stair points form a dense chain, so
    linking neighbours within `radius_m` walks the whole flight into one
    cluster while still separating two staircases in different rooms.
    """
    from scipy import ndimage

    if xy.shape[0] == 0:
        return []
    # Link on a GRID rather than pairwise. A pairwise neighbour query is
    # quadratic when the points are dense inside the link radius, which is
    # exactly the case here -- measured 33 s for 40k points, enough to hang a
    # run. Occupied cells of side radius_m/2, 8-connected, is the same
    # single-linkage relation at grid resolution and is linear in the points.
    cell = max(radius_m / 2.0, 1e-6)
    ij = np.floor(xy / cell).astype(np.int64)
    lo = ij.min(axis=0)
    ij -= lo
    shape = (int(ij[:, 0].max()) + 3, int(ij[:, 1].max()) + 3)
    if shape[0] * shape[1] > 4_000_000:  # pathological spread: fall back to one cluster
        return [(xy.mean(axis=0), int(xy.shape[0]))]
    occ = np.zeros(shape, dtype=bool)
    occ[ij[:, 0] + 1, ij[:, 1] + 1] = True
    lbl, n = ndimage.label(occ, structure=np.ones((3, 3)))
    if n == 0:
        return []
    point_labels = lbl[ij[:, 0] + 1, ij[:, 1] + 1]
    clusters = [
        (xy[point_labels == k].mean(axis=0), int((point_labels == k).sum()))
        for k in range(1, n + 1)
    ]
    return sorted(clusters, key=lambda c: -c[1])


def _merge(portals: Sequence[Portal], merge_m: float) -> List[Portal]:
    """Collapse portals onto the same opening -- a stairwell seen from several
    poses yields several patches of one thing."""
    kept: List[Portal] = []
    for p in portals:
        if any(
            np.linalg.norm(p.centroid_xy - k.centroid_xy) < merge_m
            and (p.delta_y > 0) == (k.delta_y > 0)
            for k in kept
        ):
            continue
        kept.append(p)
    return kept


# graph.priors adds this much when the target CATEGORY itself is mapped on the
# floor, so any evidence at or above it means "the target is already here".
_TARGET_PRESENT = 10


class FloorSwitchPolicy:
    """When may the agent leave the storey it is on?

    ASCENT's gate rather than MFNP's weighted score, for reasons specific to
    this repo: MFNP's score needs an LLM term, and `docs/INVESTIGATION.md`
    records that LLM frontier guidance produced byte-identical trajectories to
    the geometric heuristic here across all 35 episodes. Its other two terms
    need new accumulators whose weights would have to be tuned at n=65, where
    the standard error is ~6 points. ASCENT's rule reuses state this code
    already computes and has the better published number (65.4 vs 58.3 SR).

    The rule: switch only when nothing near is left to explore on this floor,
    never twice in quick succession, and not at the very start or the very end
    of the budget (MFNP's two free guards -- early on the current floor is
    barely mapped, late there is no budget to recover from a wrong choice).
    """

    def __init__(
        self,
        max_steps: int = 500,
        near_frontier_m: float = 4.0,
        min_interval_steps: int = 50,
        no_switch_before: int = 50,
        no_switch_after_frac: float = 0.7,
        use_target_evidence: bool = False,
        early_switch_step: int = 30,
        min_objects_to_judge: int = 8,
        strong_evidence: int = 2,
        evidence_patience_steps: int = 120,
    ) -> None:
        self.near_frontier_m = float(near_frontier_m)
        self.min_interval_steps = int(min_interval_steps)
        self.no_switch_before = int(no_switch_before)
        self.no_switch_after = int(no_switch_after_frac * max_steps)
        self.use_target_evidence = bool(use_target_evidence)
        self.early_switch_step = int(early_switch_step)
        self.min_objects_to_judge = int(min_objects_to_judge)
        self.strong_evidence = int(strong_evidence)
        self.evidence_patience_steps = int(evidence_patience_steps)
        self.last_switch_step = -10 ** 9

    def may_switch(
        self,
        step: int,
        best_path_cost: Optional[float],
        evidence: Optional[int] = None,
        n_objects: int = 0,
        steps_on_floor: int = 0,
    ) -> bool:
        """`evidence` is how many of the target's usual companions are on this
        floor (see graph.priors); None means no usable prior for the category.

        Two hard guards always apply -- the late cutoff and the interval. Within
        those, the target evidence can both bring a switch FORWARD (this floor
        has been looked at and shows no sign of the target) and hold one BACK
        (it looks like the right kind of floor, so keep searching here).
        """
        if step > self.no_switch_after:
            return False
        if step - self.last_switch_step < self.min_interval_steps:
            return False

        if self.use_target_evidence and evidence is not None:
            # A mapped instance of the target category itself (evidence carries
            # a large bonus for it) settles the question: never leave a floor
            # that has the thing we are looking for on it.
            if evidence >= _TARGET_PRESENT:
                return False

            searched = (
                n_objects >= self.min_objects_to_judge and step >= self.early_switch_step
            )
            promising = evidence >= self.strong_evidence
            # Context said "the target's kind of floor", but a long search here
            # has turned up nothing. The evidence is stale: a bathroom on this
            # storey does not mean THIS storey's bathroom has the toilet.
            stale = (
                self.evidence_patience_steps > 0
                and steps_on_floor >= self.evidence_patience_steps
            )

            # Go now, while there is budget to search the next floor. Requiring
            # evidence to be exactly zero was far too strict -- almost any floor
            # has one incidental companion object, so this fired on 3 of 24
            # cross-floor episodes where the looser rule fired on 14.
            if searched and (not promising or stale):
                return True
            if promising and not stale:
                return False

        if step < self.no_switch_before:
            return False
        # "No near frontier": either nothing selectable at all, or the best
        # thing left on this floor is further than a portal is worth.
        return best_path_cost is None or best_path_cost > self.near_frontier_m

    def note_switch(self, step: int) -> None:
        self.last_switch_step = step
