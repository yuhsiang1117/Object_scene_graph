"""Per-floor map stack.

A single 2D costmap cannot represent a multi-storey house: geometry from an
upper floor lands on the lower floor's grid (see NavAgent._act_inner). This
keeps one costmap -- and one planner, value map, room segmentation and set of
object tracks -- per floor, and decides which of them is current.

Floor identification is height clustering, not a stair state machine. habitat
reports the sensor pose exactly and mounts it at [0, camera_height, 0] on an
agent of that height (sim/habitat_env.py), so `camera_position[1] -
camera_height` IS the walkable surface height, not an estimate. Clustering on
it is both simpler and more robust than inferring floor changes from stair
detections. (Deciding to *go* to another floor is a separate problem -- that
lives in the agent's CLIMB state.)

**key vs order.** `key` is allocated once and never reused or renumbered;
`order` is the rank by height, 0 = lowest. Anything persistent -- an object
track's floor, a frontier's floor, room-id namespacing, a score cache -- must
use `key`. Anything semantic -- "go up", "the ground floor", a per-storey prior
-- must use `order`. Without the split, discovering a basement halfway through
an episode renumbers every floor reference already stored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from .costmap import Costmap2D


@dataclass
class FloorLayer:
    """Everything scoped to one storey."""

    key: int
    floor_y: float
    costmap: Costmap2D
    planner: object  # HybridVoronoiPlanner; typed loosely to avoid the import
    n_y_samples: int = 1
    value_map: object = None  # S4: ValueMap2D
    room_labels: Optional[np.ndarray] = None
    up_stair_hits: Optional[np.ndarray] = None  # S3
    down_stair_hits: Optional[np.ndarray] = None  # S3
    # Cells of staircases that failed to carry the agent anywhere. Retiring the
    # CELLS (not just a centroid) is what stops the same unusable stairwell
    # being rediscovered as the hit grid grows and its centroid drifts.
    disabled_stair: Optional[np.ndarray] = None  # S3
    track_ids: Set[int] = field(default_factory=set)
    # Where the agent arrived on this floor. Used to suppress a "down stairs"
    # detection of the staircase just climbed.
    entry_xy: Optional[np.ndarray] = None
    steps_on_floor: int = 0
    visits: int = 1
    explored: bool = False


class FloorStack:
    """Allocates and selects FloorLayers from observed standing heights."""

    def __init__(
        self,
        make_layer: Callable[[int, float], FloorLayer],
        band_m: float = 0.9,
        commit_steps: int = 4,
        settle_m: float = 0.2,
    ) -> None:
        """
        Args:
            make_layer: ``(key, floor_y) -> FloorLayer``; builds the costmap,
                planner and anything else scoped to a floor.
            band_m: a height within this of a known floor belongs to it,
                otherwise the agent is somewhere new. Storeys are ~2.5 m, so
                0.9 separates them while tolerating ramps and landings.
                ``inf`` pins the stack to a single floor, which is how
                multi-floor support is switched off.
            commit_steps: consecutive observations required before `current`
                switches, and before an unrecognised height becomes a new
                floor. Stops one noisy sample, or one step onto a landing,
                from swapping the active map.
            settle_m: how still the agent must be, vertically, for an
                unrecognised height to count as a new floor: the spread over
                the last `commit_steps` samples must be under this. Without it
                a staircase -- which passes through heights far from BOTH real
                floors -- would have a phantom floor allocated part-way up, and
                then another at the top. A stair rises ~0.15-0.2 m per step, so
                four samples span ~0.5 m and never settle; a real floor spans
                ~0.
        """
        self._make_layer = make_layer
        self.band_m = band_m
        self.commit_steps = commit_steps
        self.settle_m = settle_m
        self._layers: List[FloorLayer] = []
        self._next_key = 0
        self._current: Optional[FloorLayer] = None
        self._pending: Optional[FloorLayer] = None
        self._pending_count = 0
        # Recent heights that matched no known floor, for the settle test.
        self._unassigned: List[float] = []
        self.switches = 0

    # ------------------------------------------------------------------ core

    def observe(self, y_obs: float, step: int = 0, frozen: bool = False) -> FloorLayer:
        """Feed the agent's standing height; returns the current floor."""
        if frozen and self._current is not None:
            # Mid-traversal: ASCENT changes its floor index in exactly one
            # place, when the agent leaves the staircase
            # (map_controller.py:299), so nothing can commit a floor while a
            # climb is in progress. OSG clusters height continuously, which on
            # a staircase both ALLOCATES a layer at a mid-flight height and then
            # switches to it -- measured on the cross-floor split as climbs
            # ending after 72 cm of gain against a 90 cm threshold, i.e. via the
            # floor-changed branch rather than the height one.
            #
            # The current floor is still charged for the step, so
            # steps_on_floor and the explored/stair-prior logic behave as
            # before; only allocation and switching are suspended.
            self._current.steps_on_floor += 1
            self._pending, self._pending_count = None, 0
            # Keep _unassigned populated, because in_transit() reads it and
            # NavAgent drops frames from the costmap while in transit. Being
            # frozen MEANS being on a staircase, which is exactly when a frame
            # belongs to no floor's map -- clearing it here reported "settled"
            # and let mid-staircase geometry be written against a stale
            # floor_y. Measured: single-floor SR 68.4% -> 62.0%, six episodes
            # going from dtg ~0.03 m to 1.5-9.0 m, on a split where the agent
            # merely walks PAST a staircase.
            self._unassigned.append(y_obs)
            del self._unassigned[: -self.commit_steps]
            return self._current

        cand = self._nearest(y_obs)
        if cand is None and self._current is None:
            # Bootstrap: the starting height defines the first floor. There is
            # nothing to be "in transit" from, so no settle test applies.
            cand = self._allocate(y_obs)
        if cand is not None:
            self._unassigned.clear()
        else:
            # An unrecognised height. Only becomes a floor once the agent has
            # actually settled there -- otherwise every staircase would leave a
            # phantom floor behind it (see settle_m).
            self._unassigned.append(y_obs)
            del self._unassigned[: -self.commit_steps]
            settled = (
                len(self._unassigned) >= self.commit_steps
                and max(self._unassigned) - min(self._unassigned) < self.settle_m
            )
            if settled:
                cand = self._allocate(float(np.mean(self._unassigned)))
                self._unassigned.clear()
                # Persistence has already been demonstrated; switch now rather
                # than waiting out a second commit window.
                if self._current is not None:
                    self.switches += 1
                self._current = cand
                self._pending, self._pending_count = None, 0
            elif self._current is not None:
                # In transit: hold the current floor so nothing is written to
                # the wrong map. (NavAgent additionally drops these frames --
                # see mapping.floor_reject_m.)
                cand = self._current

        if self._current is None:
            self._current = cand
        elif cand is self._current:
            self._pending, self._pending_count = None, 0
        else:
            # Hysteresis: require the same alternative to win repeatedly.
            if cand is self._pending:
                self._pending_count += 1
            else:
                self._pending, self._pending_count = cand, 1
            if self._pending_count >= self.commit_steps:
                self._current = cand
                self._current.visits += 1
                self.switches += 1
                self._pending, self._pending_count = None, 0

        cur = self._current
        # Refine the floor height only while genuinely settled on this floor.
        # Updating mid-transition would drag the estimate up the staircase and
        # eventually merge two floors into one.
        if cand is cur and abs(y_obs - cur.floor_y) < 0.15:
            cur.n_y_samples += 1
            cur.floor_y += (y_obs - cur.floor_y) / cur.n_y_samples
        cur.steps_on_floor += 1
        return cur

    def _nearest(self, y_obs: float) -> Optional[FloorLayer]:
        if not self._layers:
            return None
        best = min(self._layers, key=lambda l: abs(l.floor_y - y_obs))
        return best if abs(best.floor_y - y_obs) < self.band_m else None

    def _allocate(self, floor_y: float) -> FloorLayer:
        layer = self._make_layer(self._next_key, floor_y)
        self._next_key += 1
        self._layers.append(layer)
        return layer

    # ---------------------------------------------------------------- access

    def current(self) -> FloorLayer:
        if self._current is None:  # before the first observe()
            self._current = self._allocate(0.0)
        return self._current

    def in_transit(self) -> bool:
        """True when the last observed height matched no known floor and has
        not settled into a new one -- i.e. the agent is on a staircase.

        This, not a fixed height threshold, is the right condition for
        dropping frames from the costmap. A raw threshold has to be smaller
        than a storey, so it also fires on a 0.4 m step or ramp *within* a
        floor and then blinds the agent for the rest of the episode (observed:
        471 of 500 steps dropped on a 0.42 m rise). Being between known floors
        is exactly the case where a frame belongs to no map.
        """
        return bool(self._unassigned)

    def layers(self) -> List[FloorLayer]:
        """All floors, lowest first."""
        return sorted(self._layers, key=lambda l: l.floor_y)

    def n_floors(self) -> int:
        return len(self._layers)

    def order_of(self, key: int) -> int:
        """Rank of a floor by height, 0 = lowest. Recomputed on demand because
        it changes whenever a lower floor is discovered -- which is exactly why
        nothing persistent may store it."""
        for i, layer in enumerate(self.layers()):
            if layer.key == key:
                return i
        raise KeyError(f"unknown floor key {key}")

    def by_order(self, order: int) -> Optional[FloorLayer]:
        ordered = self.layers()
        return ordered[order] if 0 <= order < len(ordered) else None

    def by_key(self, key: int) -> Optional[FloorLayer]:
        return next((l for l in self._layers if l.key == key), None)

    def up(self) -> Optional[FloorLayer]:
        """The known floor immediately above the current one, if any."""
        return self.by_order(self.order_of(self.current().key) + 1)

    def down(self) -> Optional[FloorLayer]:
        order = self.order_of(self.current().key)
        return self.by_order(order - 1) if order > 0 else None

    def stats(self) -> Dict[str, int]:
        return {"n_floors": self.n_floors(), "floor_switches": self.switches}
