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
    # ASCENT treats a rejection as retryable; zero preserves the legacy
    # permanent rejection behavior used by existing dynamic presets.
    reject_cooldown_steps: int = 0
    # Arriving at a committed target without ever seeing it is an OBSERVATION,
    # not just a failed trip (docs/history/DYNAMIC_SCENES.md, C5). Applying it as
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
    # Beyond this range, "I did not see it" is not evidence of absence. 0.0
    # disables the check, which is the shipped behaviour.
    #
    # In navmesh mode the follower cannot distinguish arrived from unreachable,
    # so an unreachable goal ends the approach exactly as an arrival does and a
    # reading is taken from wherever the agent happens to stand. Measured on
    # 00848: a track 0.38 m from the true object was read as absent from 6.4 m
    # away, on a VLM answer about a handful of pixels, and collapsed 0.82 ->
    # 0.36. All six pitcher episodes on that scene are byte-identical as a
    # result, and its per-target SR is 0.056.
    #
    # 3.0 m is the split the ground-truth instrument already measures: in-situ
    # recall is 0.52-0.63 within three metres and 0.24-0.26 beyond, so past that
    # the detector's silence is close to uninformative -- and the viewpoint rings
    # the agent stops on top out at 2.0 m, so a genuine arrival is always inside
    # it.
    absence_max_range_m: float = 0.0
    # Build the verifier for the ABSENCE check only, leaving the pre-approach
    # candidate gate off, so a run isolates one variable.
    absence_only: bool = False
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
    # A sighting of the target during the approach farther than this (median
    # mask depth, m) does not count as "seen this approach", so the absence
    # reading is still taken at arrival. 0 = any range counts (the shipped
    # rule). Measured on the Stretch (outputs/20260922_191444): a single
    # 0.36-score detection at 4.9 m protected a stop at an empty spot while
    # the belief had fallen to 0.44; stretch3_map sets 3.0, the range past
    # which in-situ detector recall is 0.24-0.26 (see absence_max_range_m).
    seen_counts_within_m: float = 0.0
    # Two rules about PRIOR-MAP tracks of the target's label, measured on the
    # released benchmark (docs/history/SR_PROPOSAL_CLOSE_LOOK.md, "The stale anchor"):
    # the static pass names the target in 39 of 54 in-anchor and 39 of 53
    # cross-anchor trials, and in-anchor moves it a median 0.7 m, so the stale
    # track is the answer more often than not -- yet a silent arrival there
    # abandons it. And every prior-map track of the label OTHER than the one
    # the object sat at is a false positive by construction (one instance per
    # scene), yet the agent spends 37% (in-anchor) and 52% (cross-anchor) of
    # its steps walking to them, a median 90 steps each.
    #
    # `stop_at_stale_anchor_once`: the first silent arrival at a prior-map
    # track never seen live this episode STOPs instead of consulting the
    # absence sensor. It costs at most one of the protocol's three attempts;
    # a failed attempt still applies the negative reading through
    # eval/attempts.rearm_after_failed_attempt.
    # `retire_stale_twins_after_absence`: once any prior-map track of the label
    # has been refuted in place (an absence arrival or a failed attempt at it),
    # the remaining prior-map tracks of that label that have not been seen live
    # are no longer candidates, so the search runs instead of the ghost tour.
    stop_at_stale_anchor_once: bool = False
    retire_stale_twins_after_absence: bool = False
    # Where the stale stop stands. Measured on the first run of the policy: the
    # stop was taken on the approach's own ring, 0.65-0.8 m from the track
    # centre, and the object had moved a median 0.7 m the other way -- nine
    # stops at a median 1.47 m from the object, one inside the metre. With this
    # on, a granted stop first walks to the nearest navigable point to the
    # track centre and stops there, which is as close to the old position as
    # the furniture allows.
    stale_stop_at_nearest_free: bool = False
    # After a stale stop FAILS, the object is still most likely on that same
    # surface (in-anchor moves it a median 0.7 m along its anchor), and the
    # remaining failures under the island fix stop at 1.0-1.9 m of it. With
    # this on, the attempt after a failed stale stop begins with a close look
    # at the stale track's container from a pose at least a metre from the
    # one just stood on, before the search moves to other surfaces.
    relook_after_stale_stop: bool = False
    # A failed attempt refutes the PLACE, not just the track. On 00848 the
    # pitcher's ghost is three fragment tracks 0.2 m apart; each failed stop
    # struck one of them and the next commit took the next, three attempts on
    # one wrong spot. With this on, eval/attempts.rearm_after_failed_attempt
    # also disables every track of the target label within
    # scene_graph.fp_disable_radius_m of the stop's committed centre.
    failed_attempt_disables_place: bool = False
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
    # A track far ABOVE the storey the agent stands on needs a second look
    # before it may take an attempt. 0.0 disables it, which is the shipped
    # behaviour; the multi-floor arm sets it.
    #
    # Measured on 00800 (outputs/mf5_pass2_p1500): the toy-airplane episode
    # spent both of its early attempts, by step 95, on tracks whose centres sat
    # at y 5.64 and 5.09 -- 1.9 and 2.5 m ABOVE the 3.16 m storey it was on,
    # each with a single observation. The true target was on the other storey
    # at 0.96. v11-v13 record the same shape: every wrong commit was more than
    # 0.8 m above its storey.
    #
    # A height alone must not be absorbing -- a real object CAN sit high on a
    # shelf -- so this asks for corroboration rather than refusing: at
    # `commit_high_min_obs` observations the track competes normally again.
    commit_max_above_storey_m: float = 0.0
    commit_high_min_obs: int = 2
    # Candidate quality gates: sliver/fragment detections (a chair edge seen
    # through furniture) must not trigger the expensive approach+verify loop.
    min_score: float = 0.70
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
    # Reject a viewpoint whose distance to the nearest OCCUPIED costmap cell is
    # under this. The planner already computes that clearance and used it only
    # to rank candidates, so it could hand back a pose the agent does not
    # physically fit in: on the sensor arm 27 of 44 stranded approach goals were
    # non-navigable, against 17 of 17 navigable among approaches that arrived.
    # The navmesh follower hid it by snapping the goal to the mesh; the PointNav
    # mover walks into it instead. Set to roughly the agent radius plus a cell.
    # 0.0 keeps the old behaviour.
    viewpoint_min_clearance_m: float = 0.0
    ring_radii_m: List[float] = field(default_factory=lambda: [0.8, 1.2, 1.5, 2.0])
    # Measure the rings from the object's estimated surface rather than its
    # centre. The radii above were tuned against HM3D ObjectNav, which scores
    # distance to a sampled goal VIEWPOINT, so the innermost ring at 0.8 m is a
    # success by construction. A benchmark that scores distance to the OBJECT at
    # 1 m has no such slack: measured over the 15 released-benchmark trials that
    # committed to the right object and still failed, the goal was placed well
    # (median 0.82 m from truth, 10/15 already inside 1 m) and the agent stopped
    # a further 0.37 m short of it, ending at a median 1.27 m. Shrinking the
    # innermost ring is what recovers those, and it is only safe if the ring
    # clears the object: on a couch or a counter the track centre is a metre
    # inside the furniture, and every sample on a small ring lands in an
    # occupied cell -> approach_viewpoint returns None -> the fallback goal that
    # scored 0.100 against 0.516. Off by default: every condition in
    # docs/history/AUTHORED_BENCHMARK_LADDER.md was measured with centred rings.
    ring_radius_extent_aware: bool = False
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
    approach_recheck: bool = False
    approach_recheck_thresh: float = 0.0
    # Verification is rare (1-3 calls/episode) and precision-critical: the 3B
    # VLM rejected clear true positives in prompt-lab tests; 7B passed all.
    vlm_model: str = "qwen2.5vl:7b"
    # Endpoint for the VLM, when it differs from the text model's. Empty means
    # "use cfg.llm", which is how this has always worked. It exists because the
    # two roles can live on different providers: as of 2026-09 the NIM TEXT
    # models return 410 Gone while its vision model still answers, so the
    # frontier ranker has to move to a local model without dragging the
    # verifier -- measured at 0.850 on NIM against 0.750 locally -- with it.
    base_url: str = ""
    api_key: str = ""
    # The STOP gate (agent/stop_gate.py): when the agent is about to STOP on a
    # live detection of the target, ask a second recogniser whether it IS the
    # target, and if not, release the candidate instead of spending an attempt
    # on it. "off" (the shipped behaviour), "vlm" or "clip".
    #
    # On the YCB benchmarks most commits are wrong -- 31 of 40 on the
    # multi-floor set, 49 of 73 live commits on DualMap's static condition --
    # and a false positive is a real object, so the detector re-detects it on
    # arrival, the absence sensor has nothing to say, and the attempt is spent.
    # HM3D's verdict that verification is a dead end (docs/history/
    # INVESTIGATION.md) was about category-correct false positives; these are
    # a lamp called "toy airplane". Measure the recogniser with
    # scripts/eval_verifiers.py before switching this on.
    stop_gate: str = "off"
    # The gate's VLM; empty = `vlm_model`. And forced choice between the query
    # and the detector's background vocabulary, instead of yes/no.
    stop_gate_model: str = ""
    stop_gate_choice: bool = False
    # `clip`: reject when cos(crop, query) does not beat the best confuser by
    # this margin (MobileCLIP-S2, the region proposer's encoder).
    stop_gate_clip_margin: float = 0.0
    # What the query must beat (and what the VLM's forced choice lists):
    # "generic" -- the indoor background vocabulary plus the 80 COCO classes,
    # fixed and independent of any benchmark -- or "detector", the arm's own
    # detector vocabulary, which on the YCB arms already names eight of the
    # benchmark's own objects. Measured on outputs/verifier_set (<= 1.5 m, 250
    # targets / 131 false positives, scripts/eval_verifiers.py), margin 0:
    #     generic   r 0.95  q 0.46
    #     detector  r 0.90  q 0.31, but a true coffee can kept 9 times of 19:
    #               "tin can" is in that vocabulary and "coffee can" is not
    # Most of the rejecting is done by naming other small objects; "generic"
    # does it without naming this benchmark's.
    stop_gate_confusers: str = "generic"
    # A rejection is identity evidence with these rates, applied through the
    # presence filter exactly as a failed attempt applies the VLM's
    # (eval/attempts.py); set them to what eval_verifiers.py measured.
    stop_gate_recall: float = 0.9
    stop_gate_q: float = 0.2
    # Bounds on what a wrong recogniser can cost: a track rejected this many
    # times is let stop, and an episode rejects at most this many stops.
    stop_gate_max_per_track: int = 2
    stop_gate_max_per_episode: int = 4
    # WHAT is judged. "frame": the target detection in the STOP frame, and
    # nothing when there is none. On Benchmark v2 (outputs/mfv2/osg*_s42) that
    # was 17 STOPs of 66: the approach ends where the object is out of frame.
    # "approach": with nothing in the frame, the most confident sighting of
    # the approach within `stop_gate_close_m` -- the distance the recogniser
    # was measured at. 37 of the 79 STOPs on a false positive had one.
    stop_gate_view: str = "frame"
    stop_gate_close_m: float = 1.5
    # A STOP with nothing to judge -- not in frame, never seen this approach
    # within `stop_gate_close_m`: "allow" (the shipped behaviour) or "reject",
    # released without an attempt exactly as a recogniser's "no". On the same
    # runs those STOPs were 42 false positives, 0 successes; the 3 true-target
    # STOPs whose last sighting was 1.5-3 m away succeeded. A stale-anchor
    # STOP (a prior track never seen live; `stop_at_stale_anchor_once`) is
    # that rule's deliberate gamble and is let through.
    stop_gate_unseen: str = "allow"
    # Vet the PRIOR map before the episode: every pass-1 track carrying the
    # target's label has its saved best crop judged by the gate's recogniser,
    # read through the presence filter at the gate's (r, q) -- a "no" leaves it
    # under the commit bar until a live sighting brings it back, nothing is
    # deleted. Pass 1 bakes false positives into the map: over Benchmark v2's
    # 37 (scene, label) pairs, 90 target-labelled prior tracks, 52 of them not
    # the object; the margin at 0 kept 34 of the 38 true ones and 9 of the 52
    # false (outputs/maps_v2_osg). On seed 42, static episodes starting on the
    # other storey went 4 of 17, most failing on a first commit to a wrong
    # same-label track on the start storey.
    stop_gate_vet_prior: bool = False
    # The same reading for a LIVE target-labelled track, once, the first time it
    # is a commit candidate (`StopGate.vet_live`): before the approach walk,
    # not at the STOP. Measured on osg_landing (3 seeds): the on-storey
    # failures committed 129 times to false positives, 18 to the object; the
    # labelled false commits are a toy airplane on doors, handrails and stair
    # railings, a coffee can on walls and paintings, a cracker box on baskets.
    # The crops are not the <= 1.5 m ones the recogniser was measured on, so a
    # "no" is one presence reading, never a strike-off, and every verdict is
    # logged with its range (`live_vet_log`). False = no live vetting.
    stop_gate_vet_live: bool = False
    # The pass-1 crop of the target INSTANCE as the clip recogniser's reference
    # (`StopGate._clip_reading`): the benchmark moves the object pass 1 saw, so
    # the prior map holds pictures of the very thing being looked for. The
    # exemplars are the prior target-labelled tracks whose crop the text margin
    # accepts, never the track being judged. "off" = the text margin alone;
    # "log" = also record the exemplar cosine beside every verdict, deciding
    # nothing; "on" = with at least one exemplar, the verdict is cosine >=
    # `stop_gate_exemplar_tau`, else the text margin. Offline, on the 90
    # target-labelled pass-1 crops (true crops against the same class in the
    # other four houses, false ones against their own house's exemplars): AUC
    # 0.975 against the text margin's 0.947; at 95% of true kept, cosine 0.79
    # rejects 98% of the false, and 6 of the 7 the margin passes.
    # OFF for a measured reason: on LIVE crops it does not separate. Nine seed-42
    # episodes in log mode (outputs/mfv2/_smoke_exemplar): live false positives
    # sit at a median cosine of 0.784, the true sightings at 0.715 and 0.562,
    # while the text margin accepts 4 of 30 false and 4 of 5 true. The exemplar
    # set holds pass 1's false positives too (on 4 of the 37 (scene, label)
    # pairs it is ONLY false positives, e.g. a toy airplane on 00878's
    # handrail), and pass 2 re-detects the very same thing: cosine 0.94. An
    # exemplar remembers the mistake, not the object.
    stop_gate_exemplar: str = "off"
    stop_gate_exemplar_tau: float = 0.79
