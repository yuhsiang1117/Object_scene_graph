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

from .costmap import Costmap2D


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
    ) -> None:
        self.near_frontier_m = float(near_frontier_m)
        self.min_interval_steps = int(min_interval_steps)
        self.no_switch_before = int(no_switch_before)
        self.no_switch_after = int(no_switch_after_frac * max_steps)
        self.use_target_evidence = bool(use_target_evidence)
        self.early_switch_step = int(early_switch_step)
        self.min_objects_to_judge = int(min_objects_to_judge)
        self.strong_evidence = int(strong_evidence)
        self.last_switch_step = -10 ** 9

    def may_switch(
        self,
        step: int,
        best_path_cost: Optional[float],
        evidence: Optional[int] = None,
        n_objects: int = 0,
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
            # Enough of this floor mapped to trust the absence, and nothing that
            # belongs with the target: go now, while there is budget to search
            # the next floor. This is the whole point -- the geometric rule
            # cannot fire until the floor is exhausted, which is too late.
            if (
                n_objects >= self.min_objects_to_judge
                and evidence <= 0
                and step >= self.early_switch_step
            ):
                return True
            # Conversely, this looks like the target's kind of floor. Stay, even
            # if the nearest frontier is far -- leaving now would abandon the
            # most promising storey in the building.
            if evidence >= self.strong_evidence:
                return False

        if step < self.no_switch_before:
            return False
        # "No near frontier": either nothing selectable at all, or the best
        # thing left on this floor is further than a portal is worth.
        return best_path_cost is None or best_path_cost > self.near_frontier_m

    def note_switch(self, step: int) -> None:
        self.last_switch_step = step
