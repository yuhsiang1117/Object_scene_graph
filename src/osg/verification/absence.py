"""Arriving and finding nothing is an observation. C5.

Walking to where the map said an object was, finding nothing, and stopping there
is how a stale map turns a success into a confident failure -- and, worse, it
teaches the map nothing, so the next episode makes the same trip. DualMap papers
over this with a per-query ignore list discarded when the query ends. The map
never learns.

Two sensors answer the question, with different error rates, and the filter does
not care which produced the reading -- that is the payoff of writing presence as
a Bayes filter rather than as detector bookkeeping:

    detector silence over a whole approach   r = 0.8   (6790 logged expectations
                                                        give 0.812 in the regime
                                                        the visibility gate admits)
    one VLM answer on a zoomed crop          r = 0.9, q = 0.2   (measured on 20
                                                        real present/absent cases:
                                                        17/20 overall)

so one trusted "no" is worth about two detector misses -- log(0.15/0.98) against
log(0.5/0.95) -- and no fusion code is needed.

Three things this deliberately does NOT do, each of them a mistake that was made
and measured:

  A failed call is NO INFORMATION, never absence. Treating a network error, or a
  "blocked" answer about an obstructed view, as evidence would quietly delete
  objects behind doors.

  The detector's silence only counts where a detection was EXPECTED -- in
  frustum, in range, big enough, unoccluded. An unexpected miss says something
  about the view, not about the world. A VLM that answered about the region has
  already looked, so its answer stands on its own.

  Abandoning does not BLACKLIST. Blacklisting is absorbing and C1's premise is
  that no state is: the belief carries the information and `min_presence` keeps a
  disbelieved track out of the candidate list until evidence brings it back.
  Measured cost of getting this wrong: on a CORRECT map the agent abandoned the
  bowl, wandered, and finished the episode standing 0.088 m from the goal --
  inside the success radius -- unable to stop, because the only track that could
  have been the answer had been struck off for good.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AbsenceVerdict:
    """The outcome of one arrival reading.

    `abandon=False` still carries a reading that was applied -- the belief moved,
    the agent simply still believes enough to stop. The caller reads `abandon`
    and nothing else; `p` and `event` are for the record.
    """

    abandon: bool
    p: float
    event: Optional[dict] = None


class AbsenceSensor:
    def __init__(self, cfg, verifier, profiler, stats: dict) -> None:
        self.cfg = cfg.verification
        self.verifier = verifier
        self.profiler = profiler
        self.stats = stats

    def observe(
        self, track, target: str, frame, presence_filter,
        scan_expected: int, reason: str,
    ) -> Optional[AbsenceVerdict]:
        """Apply the arrival as evidence. None means no reading was taken.

        None and `abandon=False` are both "let the stop stand"; they differ in
        whether the belief moved, which is what the record needs to distinguish.
        """
        vc = self.cfg
        if not vc.absence_on_arrival or presence_filter is None or track is None:
            return None
        # The VLM is a second sensor with its own (r, q); when it is available,
        # ask it about the target's own footprint rather than trusting the
        # detector's silence alone. A failed call returns None and is treated as
        # no information, never as absence.
        recall, q = float(vc.detector_absence_recall), None
        asked_vlm = False
        if self.verifier is not None and vc.absence_use_vlm:
            proj = track.ellipsoid.project(frame.intrinsics.K(), frame.T_cw)
            if proj is not None:
                with self.profiler.timeit("absence_vlm"):
                    still = self.verifier.verify_still_there(
                        frame.rgb, proj.bbox(), target
                    )
                if still is not None:
                    asked_vlm = True
                    recall = float(vc.vlm_recall)
                    q = float(vc.vlm_q)
                    if still:
                        # It IS there and the detector merely missed it. Let the
                        # stop stand -- this is the case that made a correct map
                        # abandon a bowl 0.8 m in front of it.
                        presence_filter.apply_reading(track, True, recall, q)
                        return None
        # The expectation gate is the DETECTOR's precondition: its silence only
        # means something where a detection was likely (frustum, range, apparent
        # size, occlusion -- C1 already answers this). A VLM that answered about
        # the region has already looked, so its answer stands on its own.
        if not asked_vlm and vc.absence_requires_expectation:
            # A sweep that expected to see it at ANY heading has looked at it.
            if scan_expected == 0 and presence_filter.expectation(
                track, frame, center_only=True
            ) is None:
                self.stats["absence_not_expected"] = (
                    self.stats.get("absence_not_expected", 0) + 1
                )
                return None

        p = presence_filter.apply_reading(track, False, recall, q)
        self.stats["absence_checks"] = self.stats.get("absence_checks", 0) + 1
        if asked_vlm:
            self.stats["absence_vlm"] = self.stats.get("absence_vlm", 0) + 1
        if p >= float(vc.abandon_below_p):
            return AbsenceVerdict(abandon=False, p=p)  # still believed
        self.stats["absence_abandon"] = self.stats.get("absence_abandon", 0) + 1
        # Walking to a mapped pose and not finding the TARGET says something the
        # belief cannot carry, because a false positive is an object that is
        # genuinely there and will be re-detected on the very next keyframe.
        # That is the identity channel, and it is a separate count from p.
        track.identity_rejections += 1
        return AbsenceVerdict(
            abandon=True,
            p=p,
            event={"cause": f"absent_on_arrival:{reason}"},
        )
