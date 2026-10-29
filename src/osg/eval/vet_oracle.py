"""The ground-truth stand-in for the STOP gate's recogniser (`verification.stop_gate_vet_oracle`).

A DIAGNOSTIC, never an arm to report: it reads the authored layout. It answers
how much success the commit-time vetting could buy at all, before more is spent
on the recogniser -- on osg_landing_vet_live (3 seeds) 212 of the goal commits
went to false positives, and the live margin that admits 79% of the true
tracks admits 11% of the false ones.

A track is the object when its centre is where the goal commits are scored
TRUE or STALE (scripts/analyze_multifloor.py, fp_after_vet): within the success
radius horizontally and FLOOR_TOL_M vertically of the object's true position or
its prior (pass-1) one. STALE is admitted on purpose: a prior track where the
object used to be is the presence filter's to retire, not the recogniser's.

`refutes` is the stricter question the "sweep" mode asks before retiring a
track for good: is it farther than IDENTITY_BAND_M from both, or on another
storey? Between the success radius and the band a track may be the object
itself, placed off by a far-range ellipsoid, and a perfect recogniser judges
what it sees, not where the map put it -- so it is left to the arm.
"""
import math
from typing import Optional, Sequence

# The analyzers' vertical tolerance for "the same storey's object".
FLOOR_TOL_M = 1.5
# osg_landing_vet_live (3 seeds): 30 of the 212 false-positive commits sat
# within 2.5 m of the object or its prior on the same storey, 182 beyond.
IDENTITY_BAND_M = 2.5


class VetOracle:
    def __init__(self, target_position: Optional[Sequence[float]],
                 prior_position: Optional[Sequence[float]], radius_m: float) -> None:
        self.anchors = [tuple(float(v) for v in p) for p in (target_position, prior_position)
                        if p is not None]
        if not self.anchors:
            raise ValueError("verification.stop_gate_vet_oracle needs an authored layout "
                             "(the episode has no target_position)")
        self.radius_m = float(radius_m)

    def _within(self, centre, radius_m: float) -> bool:
        x, y, z = (float(v) for v in centre[:3])
        return any(math.hypot(x - a[0], z - a[2]) <= radius_m and abs(y - a[1]) <= FLOOR_TOL_M
                   for a in self.anchors)

    def admits(self, centre) -> bool:
        if centre is None:
            return False
        return self._within(centre, self.radius_m)

    def refutes(self, centre) -> bool:
        if centre is None:
            return False                          # nothing to judge: the arm's call
        return not self._within(centre, max(IDENTITY_BAND_M, self.radius_m))
