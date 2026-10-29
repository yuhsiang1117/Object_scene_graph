"""Which storey is the target on? A belief over the storeys the agent knows.

Before this, the storey question had three answers, none of them a belief. The
container posterior averages affinity per storey, and the per-storey means agree
to within half a percent (`strategy._select_surface`), so it abstains by design.
`floor_disproved_after_failed_attempts` sends the agent to the nearest other
storey after two failed attempts -- and on Benchmark v2 seed 42 those attempts
were false-positive STOPs in 7 of the 11 episodes that left the object's own
storey. And when neither fires, the "nothing near is left here" gate takes the
NEAREST FLIGHT IN EITHER DIRECTION toward a storey it invents 1.8 m away: 9 of
the 28 episodes that had to go down pursued only flights going up (00821 from
0.13 m toward a storey at 1.93 m that does not exist).

This module scores each storey `k` by

    P_k  oc  pi_k * (1 - d * c_k)
    pi_k  =  w * a_k + (1 - w) / K
    a_k   =  1 - prod_t (1 - p_t)       over target-labelled tracks t on k

`a_k` is the chance that at least one target-labelled track on `k` is the
object, from each track's presence belief `p_t` -- prior-map anchors and live
detections alike, so a stale anchor stops pulling once the agent has looked at
its pose and found nothing. `c_k` is the fraction of `k`'s mapped floor this
episode has been near, and `d` how likely being near it means seeing the object.
The agent asks for the best other storey when its belief is `switch_ratio`
times this one's, and never leaves on the geometric gate alone while it knows a
second storey: the undirected switch is kept only for a building it knows one
storey of.

Two facts about the prior maps shape the bookkeeping:

* a LANDING is stored as a storey. 00800's pass-1 graph holds 3.16, 1.74 and
  0.64 m, its real floors are 3.16 and 0.16; 00821's holds -2.40 beside -3.47.
  Levels closer than one storey (`floor.new_level_m`) are one storey here, and
  the one with more mapped floor speaks for it;
* a track seen FROM a landing is filed under the landing. So a track's storey is
  read off its height -- the highest storey at or below its centre -- and not
  off the layer it was mapped on.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from ..core.labels import same_label
from ..mapping.costmap import FREE

#: An object rests on its storey's floor or on furniture above it, so its centre
#: sits up to this far BELOW the storey it belongs to only through noise.
_BELOW_FLOOR_M = 0.3


class StoreyBelief:
    """Owned by `ExplorationStrategy`; fed one position a step by the agent."""

    def __init__(self, cfg, new_level_m: float, stats: dict) -> None:
        self.cfg = cfg
        self.new_level_m = float(new_level_m)
        self.stats = stats
        self.reset()

    def reset(self) -> None:
        self._covered: Dict[int, set] = {}
        self._requested: Optional[int] = None
        self.log: list = []
        self._last_logged: Optional[tuple] = None

    # ------------------------------------------------------------ coverage

    @property
    def _block_m(self) -> float:
        return float(getattr(self.cfg, "storey_block_m", 0.5))

    def note(self, floor_key: int, agent_xy) -> None:
        """The agent stood here, on this layer: mark the blocks it was near."""
        r = float(getattr(self.cfg, "storey_cover_radius_m", 2.0))
        b = self._block_m
        xy = np.asarray(agent_xy, dtype=float)
        lo = np.floor((xy - r) / b).astype(int)
        hi = np.floor((xy + r) / b).astype(int)
        cells = self._covered.setdefault(int(floor_key), set())
        for i in range(lo[0], hi[0] + 1):
            for j in range(lo[1], hi[1] + 1):
                centre = (np.array([i, j], dtype=float) + 0.5) * b
                if float(np.linalg.norm(centre - xy)) <= r:
                    cells.add((i, j))

    def _free_blocks(self, costmap) -> set:
        rc = np.argwhere(costmap.grid == FREE)
        if not len(rc):
            return set()
        xy = costmap.origin[None, :] + (rc.astype(float) + 0.5) * float(costmap.resolution)
        blocks = np.unique(np.floor(xy / self._block_m).astype(int), axis=0)
        return {(int(i), int(j)) for i, j in blocks}

    # ------------------------------------------------------------- storeys

    def storeys(self, layers: Dict[int, Tuple[float, object]]) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, set]]:
        """`(storeys, representative, free)`.

        `layers` is `{key: (height, costmap)}`. Layers are taken in order of
        mapped floor, most first (ties: the lower key); one becomes a storey
        when it is at least `new_level_m` from every storey already taken, and
        otherwise belongs to the nearest of them. Greedy by AREA and not by
        height, because landings chain: 00800's 0.64, 1.74 and 3.16 are each
        within a storey of the next, and a height sweep makes them one.
        `representative` maps every layer key to the storey key that speaks for
        it; `free` holds each storey's free blocks, its member layers' pooled.
        """
        free = {int(k): (self._free_blocks(cm) if cm is not None else set())
                for k, (_, cm) in layers.items()}
        height = {int(k): float(h) for k, (h, _) in layers.items()}
        storeys: Dict[int, float] = {}
        rep: Dict[int, int] = {}
        for k in sorted(height, key=lambda k: (-len(free[k]), k)):
            near = [s for s in storeys if abs(height[k] - storeys[s]) < self.new_level_m]
            if near:
                rep[k] = min(near, key=lambda s: abs(height[k] - storeys[s]))
            else:
                storeys[k] = height[k]
                rep[k] = k
        pooled = {s: set() for s in storeys}
        for k, s in rep.items():
            pooled[s] |= free[k]
        return storeys, rep, pooled

    @staticmethod
    def storey_of_height(y: float, storeys: Dict[int, float]) -> int:
        """The highest storey at or below `y` (with noise), else the lowest."""
        below = [k for k, h in storeys.items() if h <= float(y) + _BELOW_FLOOR_M]
        if below:
            return max(below, key=lambda k: storeys[k])
        return min(storeys, key=lambda k: storeys[k])

    # ----------------------------------------------------------- posterior

    def posterior(self, layers, tracks: Iterable[Tuple[float, float]], current_key: int) -> dict:
        """The belief, and everything it was computed from.

        `tracks` are `(centre_y, presence)` of the target-labelled tracks.
        """
        storeys, rep, free = self.storeys(layers)
        anchor_miss = {k: 1.0 for k in storeys}
        for y, p in tracks:
            k = self.storey_of_height(y, storeys)
            anchor_miss[k] *= 1.0 - float(np.clip(p, 0.0, 1.0))
        anchor = {k: 1.0 - v for k, v in anchor_miss.items()}
        covered = {k: set() for k in storeys}
        for key, blocks in self._covered.items():
            head = rep.get(int(key))
            if head is not None:
                covered[head] |= blocks
        cover = {}
        for k in storeys:
            n = len(free[k])
            cover[k] = (len(covered[k] & free[k]) / n) if n else 0.0
        w = float(getattr(self.cfg, "storey_anchor_weight", 0.8))
        d = float(getattr(self.cfg, "storey_detect_prob", 0.85))
        n_storeys = max(len(storeys), 1)
        raw = {k: (w * anchor[k] + (1.0 - w) / n_storeys) * (1.0 - d * cover[k]) for k in storeys}
        total = sum(raw.values()) or 1.0
        return {
            "P": {k: v / total for k, v in raw.items()},
            "anchor": anchor,
            "cover": cover,
            "storeys": storeys,
            "current": rep.get(int(current_key), int(current_key)),
        }

    def decide(self, step: int, layers, tracks, current_key: int) -> Optional[int]:
        """The LAYER KEY of the storey to ask for, or None to stay here.

        Asks once another storey's belief reaches `switch_ratio` times this
        one's, and keeps asking for it until this one overtakes it again, so a
        request does not flicker on the boundary while the agent walks to the
        stairs.
        """
        post = self.posterior(layers, tracks, current_key)
        P, here = post["P"], post["current"]
        others = [k for k in P if k != here]
        choice = None
        if others and here in P:
            best = max(others, key=lambda k: (P[k], -k))
            ratio = float(getattr(self.cfg, "storey_switch_ratio", 1.3))
            back = float(getattr(self.cfg, "storey_return_ratio", 0.0) or 0.0)
            if back > 0.0 and post["cover"].get(best, 0.0) > 0.0:
                # A storey stood on this episode is one a staircase walked this
                # episode leads back to: the climb's cost is that climb, not
                # the expected ~250 steps of an unknown one.
                ratio = min(ratio, back)
            held = self._requested == best and P[best] > P[here]
            if P[best] >= ratio * P[here] or held:
                choice = best
        self._requested = choice
        self.stats["storey_belief_rounds"] = self.stats.get("storey_belief_rounds", 0) + 1
        if choice is not None:
            self.stats["storey_belief_requests"] = self.stats.get("storey_belief_requests", 0) + 1
        key = (here, choice, tuple(sorted((k, round(v, 2)) for k, v in P.items())))
        if key != self._last_logged:
            self._last_logged = key
            self.log.append({
                "step": int(step),
                "here": int(here),
                "request": None if choice is None else int(choice),
                "P": {str(k): round(float(v), 3) for k, v in P.items()},
                "anchor": {str(k): round(float(v), 3) for k, v in post["anchor"].items()},
                "cover": {str(k): round(float(v), 3) for k, v in post["cover"].items()},
                "height": {str(k): round(float(v), 2) for k, v in post["storeys"].items()},
            })
        return choice

    def knows_another_storey(self, layers) -> bool:
        """More than one storey once landings are merged: the undirected switch
        has nothing to add, because every storey it could find is scored here."""
        storeys, _, _ = self.storeys(layers)
        return len(storeys) >= 2


def target_tracks(object_layer, target: str, max_rejections: int = 2,
                  rejection_discount: float = 0.3, retirement=None):
    """`(centre_y, weight)` of every live, target-labelled track.

    The weight is the chance the track is the target where it is: its
    presence belief times what its IDENTITY evidence leaves. Presence alone
    answers "is something still there", and a false positive the detector
    keeps firing on stays present -- measured on 00862 (seed 42, same-floor
    pitcher): three STOP-gate rejections later its storey's anchor still read
    1.0. So a track the agent's own candidate gate has given up on (a failed
    attempt, or `max_rejections` identity rejections) weighs nothing, and each
    rejection short of that multiplies its weight by `rejection_discount`.
    Blacklisted tracks are left out; a track without a presence filter counts
    at 0.5. With a `retirement` (objects/retirement.py), a stale twin the
    candidate gate strikes off is left out too.
    """
    out = []
    for track in object_layer.tracks():
        if not same_label(getattr(track, "label", ""), target):
            continue
        rejected = int(getattr(track, "identity_rejections", 0) or 0)
        if int(getattr(track, "failed_attempts", 0) or 0) > 0 or rejected >= int(max_rejections):
            continue
        if retirement is not None and retirement.twin(track):
            retirement.n_twins_dropped += 1
            continue
        centre = np.asarray(object_layer.center_of(track), dtype=float)
        presence = getattr(track, "presence", None)
        p = float(presence.p) if presence is not None else 0.5
        out.append((float(centre[1]), p * float(rejection_discount) ** rejected))
    return out
