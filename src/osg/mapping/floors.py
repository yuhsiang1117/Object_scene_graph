"""Online floor estimation.

Which storey is the agent on? Habitat is y-up, so this is a question about one
scalar. Two sources, in order of trust:

1. **The agent's own standing height** (`camera_position[1] - camera_height`).
   Direct, per-step, and noise-free in sim -- and the agent only ever stands on
   floors, so every sample is by construction a floor observation. This is the
   primary signal.
2. **A histogram over observed floor-plane points** (HOV-SG's recipe, see
   `height_histogram_peaks`). Optional and secondary: it can pre-register a
   floor the agent has *seen* but not yet visited (looking up a stairwell), but
   a depth cloud also peaks on ceilings and table tops, so it never overrides
   (1).

Floor ids are **stable**: a level discovered later (e.g. a basement, below
everything known) gets a fresh id rather than renumbering the existing ones.
Anything keyed by floor id -- per-floor costmaps, room ids, frontiers,
blacklists -- would otherwise be silently re-pointed at a different storey.

The hard case is a **multi-flight staircase landing**: a genuine, sustained
height plateau midway between two storeys, indistinguishable from a new floor
to any dwell-based rule. Registering one splits the map mid-staircase onto a
"floor" with no reachable frontiers -- strictly worse than today's collapse.

This is not hypothetical: it fired on the first multi-floor episode measured
(XB4GS9ShBRE ep3), where a single descent from 2.796 to 0.196 registered FOUR
floors, inventing landings at 1.879 and 1.253. A dwell rule cannot fix it --
the agent lingers on a landing for 25+ steps while turning. What separates the
two cleanly is the gap itself: HM3D storeys are 2.5-3.4 m apart, landings
<=1.1 m from the floor below. Hence `new_level_m = 1.8`.

The trade is deliberate: a real mezzanine less than 1.8 m up is missed and
folded into the floor below. A missed floor degrades to today's behaviour; a
phantom floor is worse than today's behaviour. See docs/MULTI_FLOOR.md.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HEIGHT_AXIS = 1


def height_histogram_peaks(
    ys: Sequence[float],
    bin_m: float = 0.01,
    peak_window_m: float = 0.2,
    peak_frac: float = 0.2,
    merge_m: float = 0.5,
) -> List[float]:
    """Floor heights from a set of point heights (HOV-SG).

    Discretize at `bin_m`, keep local maxima within +-`peak_window_m` that
    exceed `peak_frac` of the global maximum, then merge responses closer than
    `merge_m`. The merge is a 1D gap-merge, which is what DBSCAN reduces to for
    `min_samples=1` in one dimension -- no sklearn dependency needed.

    `peak_frac` deliberately departs from HOV-SG's 0.9. That threshold assumes
    a complete, uniformly reconstructed point cloud where every floor plane
    accumulates a comparable bin count. Ours is incremental and view-dependent,
    and even two *identically sized* synthetic modes differ by ~14% from
    sampling noise alone -- at 0.9 the second floor is silently dropped. 0.2
    keeps genuine storeys while still rejecting furniture-height clutter.
    """
    ys = np.asarray(list(ys), dtype=float)
    if ys.size == 0:
        return []
    lo, hi = float(ys.min()), float(ys.max())
    if hi - lo < bin_m:
        return [round(float(ys.mean()), 3)]

    edges = np.arange(lo, hi + bin_m, bin_m)
    counts, _ = np.histogram(ys, bins=edges)
    centers = edges[:-1] + bin_m / 2.0
    if counts.max() == 0:
        return []

    w = max(1, int(peak_window_m / bin_m))
    peaks = [
        i for i in range(len(counts))
        if counts[i] == counts[max(0, i - w):i + w + 1].max()
        and counts[i] >= peak_frac * counts.max()
    ]

    merged: List[int] = []
    for i in peaks:
        if merged and centers[i] - centers[merged[-1]] < merge_m:
            if counts[i] > counts[merged[-1]]:
                merged[-1] = i
        else:
            merged.append(i)
    return [round(float(centers[i]), 3) for i in merged]


class FloorEstimator:
    """Tracks which floor the agent is on, and the set of known floors.

    With a single floor -- every single-floor scene, and every multi-floor
    scene until the agent actually climbs -- `current` stays 0 and `height_of(0)`
    is identically the value `nav_agent` used to latch on the first frame. The
    single-floor code path is therefore unchanged by construction, not by
    tuning.
    """

    def __init__(
        self,
        camera_height: float,
        level_tol_m: float = 0.35,
        merge_m: float = 0.6,
        new_level_m: float = 1.8,
        min_dwell_steps: int = 6,
        min_horizontal_run_m: float = 2.5,
    ) -> None:
        self.camera_height = float(camera_height)
        self.level_tol_m = float(level_tol_m)
        self.merge_m = float(merge_m)
        # Separation a candidate must have from every known level before it is
        # registered as a new floor. 1.8 m is measured, not guessed: HM3D
        # storeys sit 2.5-3.4 m apart, while staircase landings sit <=1.1 m
        # from the floor below (see the docstring and test_landings_*).
        self.new_level_m = float(new_level_m)
        self.min_dwell_steps = int(min_dwell_steps)
        # Second, independent route to committing a level: how far the agent
        # gets HORIZONTALLY from where it first stood at a candidate height.
        # This is what separates a storey from a staircase landing when the two
        # are at similar heights -- a 1.2 m landing pins you within 1.2 m no
        # matter how long you stay, while any real floor lets you walk away.
        # Without it, `new_level_m` alone has to be both high enough to reject
        # landings (measured up to 1.4 m) and low enough to accept a climb the
        # agent aborts partway (measured 1.6 m), which is not satisfiable.
        # Displacement, not path length -- see update(). 0 disables.
        self.min_horizontal_run_m = float(min_horizontal_run_m)
        # Wide enough to close the registration error, well short of the
        # separation that would make it a different storey.
        self.capture_m = self.new_level_m / 2.0

        self._levels: Dict[int, float] = {}
        self._samples: Dict[int, List[float]] = {}
        self._max_samples = 400
        self._next_id = 0
        self.current: int = 0
        self.on_stairs: bool = False
        # (step, from_floor, to_floor, entry_xy, exit_xy) -- filled by the
        # caller's stair bookkeeping in a later stage; recorded here because
        # this is where the transition is detected.
        self.transitions: List[Tuple[int, int, int]] = []

        self._cand: Optional[float] = None
        self._cand_floor: Optional[int] = None
        self._dwell = 0
        self._cand_xy: Optional[np.ndarray] = None
        self._cand_run = 0.0

    def reset(self) -> None:
        """Clear all levels. Floors are per-episode: ids and heights from one
        scene mean nothing in the next."""
        self._levels = {}
        self._samples = {}
        self._next_id = 0
        self.current = 0
        self.on_stairs = False
        self.transitions = []
        self._reset_candidate()

    # ------------------------------------------------------------------ state

    @property
    def levels(self) -> Dict[int, float]:
        return dict(self._levels)

    def sorted_levels(self) -> List[Tuple[int, float]]:
        """(id, height) ascending by height. Ids are creation-ordered, not
        height-ordered, so never infer "upstairs" from the id."""
        return sorted(self._levels.items(), key=lambda kv: kv[1])

    def height_of(self, floor_id: int) -> float:
        return self._levels[floor_id]

    def floor_of_height(self, y: float) -> int:
        """Nearest known floor to a world height. Used to place objects (whose
        ellipsoid centres sit above their floor) and goals."""
        if not self._levels:
            return self.current
        return min(self._levels, key=lambda k: abs(self._levels[k] - float(y)))

    def _nearest(self, y: float) -> Tuple[int, float]:
        fid = min(self._levels, key=lambda k: abs(self._levels[k] - y))
        return fid, abs(self._levels[fid] - y)

    # ----------------------------------------------------------------- update

    def update(self, cam_y: float, step: int = 0, xy: Optional[Sequence[float]] = None) -> int:
        """Feed one frame's camera height; returns the committed floor id.

        `xy` is the agent's ground-plane position. Optional, but without it the
        horizontal-run route to committing a level is unavailable and only
        `new_level_m` applies.
        """
        y = float(cam_y) - self.camera_height

        if not self._levels:  # bootstrap: the starting floor is floor 0
            self.current = self._add_level(y)
            return self.current

        # Refine before classifying. A level is first registered at whatever
        # height cleared new_level_m, which is typically part-way down the
        # stairs (measured: XB4GS9ShBRE ep3 registered 0.832 for a floor whose
        # true height is 0.196). Left uncorrected that 0.64 m error exceeds
        # level_tol_m, so the agent reads as permanently on-stairs once it
        # reaches the real floor, and any per-floor obstacle band derived from
        # it sits a half-metre too high. The median over samples within the
        # capture radius converges to the modal standing height -- the flat
        # floor, where the agent spends nearly all its steps -- rather than the
        # handful of treads above it.
        self._refine(y)

        fid, dist = self._nearest(y)
        self.on_stairs = dist > self.level_tol_m

        if not self.on_stairs:
            if fid == self.current:
                self._reset_candidate()
                return self.current
            # Standing on a known, different floor: commit after the dwell.
            if self._cand_floor == fid:
                self._dwell += 1
            else:
                self._cand_floor, self._cand, self._dwell = fid, None, 1
            if self._dwell >= self.min_dwell_steps:
                self.transitions.append((step, self.current, fid))
                self.current = fid
                self._reset_candidate()
            return self.current

        # Off every known level. The floor id stays FROZEN at the last
        # committed value -- a half-mapped staircase must not become its own
        # floor, and the caller keeps writing into the origin floor's map.
        if dist <= self.merge_m:  # a threshold or a sunken room, never a storey
            self._reset_candidate()
            return self.current

        if self._cand is not None and abs(y - self._cand) <= self.level_tol_m:
            self._dwell += 1
            if xy is not None and self._cand_xy is not None:
                # DISPLACEMENT from where this candidate height was first stood
                # on, not path length. Path length would be fooled by an agent
                # pacing on the spot: a 1.2 m landing racks up tens of metres of
                # path while never getting more than 1.2 m from anywhere.
                self._cand_run = max(
                    self._cand_run,
                    float(np.linalg.norm(np.asarray(xy, float) - self._cand_xy)),
                )
        else:
            self._cand, self._cand_floor, self._dwell = y, None, 1
            self._cand_run = 0.0
            self._cand_xy = np.asarray(xy, dtype=float).copy() if xy is not None else None

        # Two independent routes to a new storey, either sufficient:
        #   * a full storey of separation (`new_level_m`), or
        #   * enough horizontal room at this height that it cannot be a landing.
        far_enough = dist >= self.new_level_m
        roomy = self.min_horizontal_run_m > 0 and self._cand_run >= self.min_horizontal_run_m
        if self._dwell >= self.min_dwell_steps and (far_enough or roomy):
            new_id = self._add_level(y)
            self.transitions.append((step, self.current, new_id))
            self.current = new_id
            self.on_stairs = False
            self._reset_candidate()
        return self.current

    def observe_points(self, ys: Sequence[float], **kwargs) -> List[int]:
        """Secondary source: pre-register floors seen but not yet stood on.

        Only ADDS levels that are `merge_m` clear of every known one; never
        switches `current` and never removes a level, so it cannot override the
        agent's own height trace. Returns the ids it created.
        """
        added = []
        for peak in height_histogram_peaks(ys, merge_m=self.merge_m, **kwargs):
            if not self._levels or self._nearest(peak)[1] > self.merge_m:
                added.append(self._add_level(peak))
        return added

    # -------------------------------------------------------------- internals

    def _refine(self, y: float) -> None:
        """Pull the nearest level toward the heights actually stood on.

        Only levels within `capture_m` are touched, so a genuinely different
        storey is never dragged. Capped sample buffer keeps this O(1) per step
        amortized and lets a level follow slow drift.
        """
        if not self._levels:
            return
        fid, dist = self._nearest(y)
        if dist > self.capture_m:
            return
        buf = self._samples.setdefault(fid, [])
        buf.append(float(y))
        if len(buf) > self._max_samples:
            del buf[: len(buf) - self._max_samples]
        self._levels[fid] = float(np.median(buf))

    def _add_level(self, y: float) -> int:
        fid = self._next_id
        self._levels[fid] = float(y)
        self._samples[fid] = [float(y)]
        self._next_id += 1
        return fid

    def _reset_candidate(self) -> None:
        self._cand, self._cand_floor, self._dwell = None, None, 0
        self._cand_xy, self._cand_run = None, 0.0
