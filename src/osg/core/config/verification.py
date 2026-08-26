"""`verification` group: the VLM gate, and absence as an observation.

Two distinct mechanisms share this group. The candidate gate asks "is this the
target"; the absence block asks "is it still there", which is the dynamic-scene
question -- the VLM enters as a second sensor with its own measured (r, q)
rather than as an override. See `osg/verification/absence.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class VerificationConfig:
    enabled: bool = True
    # Arriving at a committed target without ever seeing it is an OBSERVATION,
    # not just a failed trip (docs/DYNAMIC_SCENES.md, C5). Applying it as
    # negative evidence is what stops a stale map sending the agent back to the
    # same empty spot next episode.
    absence_on_arrival: bool = True
    # Only treat a non-detection as absence where the presence filter says a
    # detection was EXPECTED (in frame, in range, big enough, unoccluded). An
    # unexpected miss says nothing about the world, only about the view.
    absence_requires_expectation: bool = True
    # Effective recall of "the detector saw nothing during the WHOLE approach".
    # Measured, not guessed: 6790 logged expectations from real episodes give a
    # 0.812 detection rate in the regime the visibility gate admits, and an
    # approach is dozens of frames from many poses rather than one look. Kept
    # just under that so a failed approach is strong evidence without being
    # decisive on its own.
    detector_absence_recall: float = 0.8
    # The VLM as a second sensor, with its own error rates. One trusted "no" is
    # worth about two detector misses: log(0.15/0.98) vs log(0.5/0.95).
    absence_use_vlm: bool = True
    # Measured on 20 real present/absent cases at the agent's own bounding box,
    # forced choice on a zoomed crop: 17/20 overall, saying the object is there
    # 9 times out of 10 when it is, and "bare" 8 times out of 10 when it is not.
    # These are those rates, not a guess.
    vlm_recall: float = 0.9
    vlm_q: float = 0.2
    # Build the verifier for the ABSENCE check only, leaving the pre-approach
    # candidate gate off, so a run isolates one variable.
    absence_only: bool = False
    # Enumerating a long list is where VLMs are least reliable, and an absence
    # you cannot trust is worse than no absence at all.
    absence_categories_max: int = 5
    # Below this belief the agent abandons the candidate instead of stopping on
    # it. 1.0 = always abandon, which is the right default once you notice what
    # the alternative actually is: NOT "keep believing and look again later" but
    # "STOP here and end the episode". Measured on the batch, two cross-anchor
    # episodes arrived at an empty spot, dropped the belief to 0.64, and -- being
    # above a 0.45 threshold -- stopped and failed with 450 steps unspent. An
    # approach that never saw its target has no reason to stop at it while steps
    # remain; the belief arithmetic still does its work in the ranking. Lower
    # this only to A/B the stricter behaviour.
    abandon_below_p: float = 1.0
    # Is "I cannot reach that" a permanent verdict?
    #
    # True is the shipped behaviour and it blacklists, which is absorbing --
    # the one thing this pipeline says everywhere else that no state may be.
    # The absence path, the map loader and the attempt protocol each had to have
    # a blacklist removed for the same reason; this is the fourth site and the
    # only one still holding one. Measured on condition K, `unreachable_skip`
    # fired in 18 of 53 failing episodes and 2 of 43 successful ones.
    #
    # False routes it to the identity channel instead: one unreachable verdict
    # is evidence, two retire the track (max_identity_rejections), and a track
    # that becomes reachable later can come back.
    unreachable_is_absorbing: bool = True
    min_obs: int = 3
    # Candidate quality gates: sliver/fragment detections (a chair edge seen
    # through furniture) must not trigger the expensive approach+verify loop.
    min_score: float = 0.45
    min_bbox_px: int = 3000
    # The same exemption one stage later. Without it the deadlock simply moves:
    # a track seeded from a distant sighting can only grow its best box by being
    # approached, and it can only be approached by being proposed. Of the 11
    # deadlocked episodes, 8 clear this gate once admitted and 3 do not.
    target_bypasses_bbox_gate: bool = False
    # Rank candidates by belief, tie-broken on evidence, instead of by
    # `best_score * presence.p`.
    #
    # Measured over the 170 within-episode pairs of K, L and M where a correct
    # and a wrong BELIEVED track compete, the chance the key puts the correct one
    # first: best_score alone 0.635, best_score * p 0.729 (shipped), p alone
    # 0.800. Multiplying by detector confidence hurts, because a confident false
    # positive is precisely a distant object that really does look like the
    # target -- best_score is highest where it misleads. Per episode with a real
    # choice, the correct track is chosen 64/93 shipped and 73/93 this way.
    rank_candidates_by_presence: bool = False
    # Evidence-score gate (P1i follow-up, 2026-07-19): threshold picked from
    # a real 8-episode/1343-track measurement (scripts/orphan_node_check.py)
    # of evidence separated by whether a track ever reached candidate
    # quality -- non-candidate tracks: p75=0.84 p90=1.27; candidate-quality
    # tracks: min=0.64 p10=1.18 p25=1.56. 1.0 sits between the non-candidate
    # p75/p90 (filtering roughly 75-80% of low-evidence noise) and just
    # under the candidate p10 (sacrificing only ~6-7% of genuine candidates,
    # erring toward not rejecting real targets over aggressively filtering).
    min_evidence: float = 1.0
    ring_radii_m: List[float] = field(default_factory=lambda: [0.8, 1.2, 1.5, 2.0])
    accept_confidence: float = 0.5
    # Forced-choice verification: instead of asking the VLM "is this a <target>?"
    # (which it tends to agree with), show it the object and the FULL category
    # list and make it pick the single best-matching category; accept only if it
    # picks the target. This catches detector mislabels -- a table YOLOE called a
    # chair -> VLM picks "table" -> reject -- that a yes/no question waves through.
    choice_mode: bool = True
    # Center-then-verify: when a VLM verifier is active and the target is
    # visible in the live view, turn to bring its detection to the middle of
    # the camera before calling the VLM, then verify that well-framed live frame
    # (whole image + red box). Centering gives the VLM a clear, unambiguous view
    # instead of a target at the frame edge. center_tol_deg is "close enough to
    # centered" -- >= half the turn angle so a single turn does not overshoot.
    center_before_verify: bool = True
    center_tol_deg: float = 16.0
    center_max_turns: int = 6
    # Terminal-view verification: instead of (or in addition to) verifying the
    # track's historical best_crop before APPROACH, verify the LIVE close-up
    # frame at the moment the agent decides to STOP. The pre-approach best_crop
    # is category-correct even for false positives (a distant chair-like object
    # really looks like a chair), so verifying it accepts ~97% and does not
    # move SR; the terminal close-up is the decisive view and can reject a
    # false positive right before the commit. When terminal=True the
    # pre-approach VLM call is skipped (accept) so this isolates the terminal
    # gate. A rejected terminal STOP blacklists the track and resumes exploring.
    terminal: bool = False
    # Verification is rare (1-3 calls/episode) and precision-critical: the 3B
    # VLM rejected clear true positives in prompt-lab tests; 7B passed all.
    vlm_model: str = "qwen2.5vl:7b"


