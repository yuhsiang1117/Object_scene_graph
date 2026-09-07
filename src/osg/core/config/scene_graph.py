"""`scene_graph` group: the object layer, the container layer, and presence.

Three layers of gate live here and they answer different questions: admission
(is this detection worth a track), container qualification (is this track a
surface something could be put on), and presence (is the object still there).
`PresenceConfig` is the dynamic-scene half -- see `osg/objects/presence.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PresenceConfig:
    """Presence belief per object (objects/presence.py, docs/DYNAMIC_SCENES.md).

    Off by default: enabling it changes which candidate the agent proposes
    first, so it must be an explicit, measurable A/B rather than a silent
    default change.
    """

    enabled: bool = False
    # P(detected | present, view). `recall_model_path` wins when it exists;
    # otherwise every view gets `recall_constant`, which makes negative updates
    # uniform -- wrong, but unbiased, and it lets the filter run before any fit.
    recall_constant: float = 0.6
    recall_model_path: str = ""
    # P(detection at a mapped pose | object gone). Only bounds the size of a
    # POSITIVE step; the system is insensitive to it.
    q_false_alarm: float = 0.05
    # Belief clamp, both signs. Never remove it: it is what keeps an object that
    # was wrongly disbelieved resurrectable by a single later detection.
    l_clamp: float = 6.0
    # Asymmetric on purpose: a sighting is worth +2.5 and a miss -0.9, so a
    # symmetric clamp saturates after three sightings and then needs seven clean
    # misses to undo. Believing an object's PRESENCE that hard is unjustified --
    # the world changes while you are not looking -- while disbelief keeps the
    # deeper floor so a genuinely gone object stays gone.
    l_clamp_pos: float = 3.0
    # Ceiling on a belief restored from a snapshot built in an earlier session.
    reload_max_log_odds: float = 1.5
    # Expected-depth band tolerance. Generous, because a mask-moment ellipsoid
    # fitted from a partial view is a coarse estimate of where a surface is.
    depth_tol_m: float = 0.15
    occ_ratio_max: float = 0.30
    range_m: Tuple[float, float] = (0.4, 6.0)
    img_inside_frac: float = 0.5
    min_depth_samples: int = 12
    max_samples: int = 256
    max_tracks: int = 64
    z_overlap_iou: float = 0.05
    # Minimum belief for a track to be proposed as a navigation candidate.
    # This is what replaces blacklisting when an approach finds nothing: the
    # track stays in the map, drops below the bar, and comes back if it is seen
    # again. The value is set by the arithmetic -- a belief reloaded at 0.82
    # lands at 0.485 after one detector-strength absence reading and 0.36 after
    # a VLM one, while a freshly detected object starts at 0.82. 0.45 sits
    # between those two: ONE detector-strength absence is not enough to retire a
    # track (its silence at close range is measurably unreliable -- it misses a
    # bowl 0.8 m in front of it), a second one is, and a single VLM answer is
    # decisive on its own. That asymmetry is the whole point of having two
    # sensors with different error rates.
    min_presence: float = 0.45
    # Retire a candidate the agent has walked to and found was not the target
    # this many times (0 disables). This is the IDENTITY channel, and it exists
    # because presence cannot answer the question: a false positive is an object
    # that really is there, so each look that disproves it as the target also
    # re-detects it as an object and pins its belief at the positive clamp.
    # Measured with this off: one episode committed to the same wrong track 251
    # times in 500 steps. Two visits is the setting -- one arrival can end on a
    # bad heading or a consumed path, two is a decision.
    max_identity_rejections: int = 2
    # JSONL of expectation features per keyframe, for scripts/fit_recall_model.py.
    log_path: str = ""


@dataclass
class SceneGraphConfig:
    keyframe_trans_m: float = 0.25
    keyframe_rot_deg: float = 30.0
    min_obs_for_refine: int = 3
    refine_every: int = 3
    # Reject a Wasserstein refine that moves the ellipsoid centre further than
    # this (m) from its pre-refine value -- the reprojection objective is
    # parallax-limited and otherwise drifts the 3D centre metres under the
    # narrow ObjectNav view arc (see analyze_refine_accuracy). 0 disables.
    refine_max_center_move_m: float = 0.5
    link_dist_m: float = 1.0
    # Two tracks may only be merged if they were observed at about the same
    # time. Linking exists to reunite fragments of one object that a single
    # ellipsoid cannot cover, and those are seen together; an object and its own
    # ghost are not (docs/DYNAMIC_SCENES.md, the in-anchor ghosting bug).
    link_max_frame_gap: int = 50
    near_edge_dist_m: float = 1.5
    assoc_score_thresh: float = 0.4
    assoc_depth_gate_m: float = 0.5
    # Wasserstein data association requires the detection label to match the
    # track label. The ported VOOM matcher had no label check, but for SR eval
    # (navigate to a target CATEGORY) cross-category merges corrupt labels and
    # starve target candidates -- so gate on category by default.
    assoc_category_gate: bool = True
    room_seg_every_kf: int = 10
    room_erode_iters: int = 6
    min_room_cells: int = 60
    room_min_radius_m: float = 0.9
    # 1.2 caused universal 1-room collapse on real HM3D scans: the merge
    # condition is clearance > door_width_m/2, so a wider value RAISES the
    # threshold and preserves more boundaries. 2.0 is where a 10-episode
    # sweep on real explored costmaps saturates (matches the measured
    # 0.85m/0.934m boundary clearances in the multi-room episodes).
    room_door_width_m: float = 2.0
    # Node-creation quality gate (P1h/orphan-node follow-up): a real diagnostic
    # run showed ~228 tracks/episode with 36% never re-observed and 49% never
    # reaching min_obs_for_refine -- most of the scene graph's memory was
    # spent on throwaway single-sighting noise that every downstream consumer
    # (frontier scoring, room segmentation, relinking) still had to pay for.
    min_det_score: float = 0.35
    min_det_bbox_px: float = 1500.0
    # Admit the episode's target on the DETECTOR's terms, bypassing the two
    # gates above.
    #
    # Those gates price "is this worth remembering" for a scene full of
    # furniture the agent is not looking for. Applied to the target they
    # deadlock: measured over 96 episodes of condition L, 33% of the times the
    # detector named the target the map discarded it, and in 11 episodes it
    # discarded EVERY naming -- no track, so no candidate, so no approach, so
    # the detection never got closer, bigger or more confident. All 11 failed.
    #
    # Five of those 11 are a contradiction rather than a threshold. Condition H
    # lowered detector.class_conf to 0.20 for the pitcher, tin can, banana and
    # red plate on a 900-pose false-positive census; min_det_score 0.35 then
    # discards everything those classes gained between 0.20 and 0.35. Their
    # boxes were 5146, 5077, 3102, 2808 and 1258 px -- far above the size gate,
    # thrown away on score alone. It is why H moved the population by one
    # episode.
    #
    # The other six are the size gate on genuinely small, genuinely confident
    # sightings: a bowl named 20 times at 3.26 m, score 0.91, 608 px box.
    #
    # Admission is not candidacy: evidence, observation count, presence and the
    # identity channel all still decide whether a track may become a goal.
    target_bypasses_gates: bool = False
    # Foveated second look at container surfaces (perception/foveate.py). One
    # extra detector call per keyframe per region, off by default.
    #
    # The whole-frame pass is budgeted for furniture and misses the objects this
    # benchmark asks for. Measured on 00848's tin can over the 21 keyframes the
    # GT instrument calls in-view and unoccluded: the run's own settings name it
    # 0/21, removing every furniture class that outscored it names it 0/21, and
    # ten alternative names name it at most 1/21 -- while a 320 px window around
    # it, upscaled, names it 5/21. Not the name, not class competition, scale.
    foveate_containers: bool = False
    # Beyond this the second look recovers nothing, measured IN THE LOOP rather
    # than in the probe. Over six tin can episodes under condition F, every
    # recovered detection is within 3 m -- 6/9 and 7/8 close, 5/5 and 10/16
    # close -- and the far bands are 0/6, 0/3, 0/2, 0/1, 0/0.
    #
    # The probe's own recoveries sat at 2.6-3.4 m with a FIXED 320 px window and
    # nothing under 2.4 m, which would argue for a lower bound too. It does not
    # transfer, and the reason is the design: the runtime window is the
    # surface's projection, which grows as the agent approaches, so the
    # magnification adapts and close range works where a fixed window failed. A
    # 2 m floor read off the probe would have cut the detections that convert.
    foveate_max_range_m: float = 3.0
    # A surface projecting smaller than this is too far or too oblique to be
    # worth a second inference.
    foveate_min_bbox_px: float = 20_000.0
    # Regions per keyframe. Each one is a full detector call, and the control
    # loop runs at ~3 Hz with one.
    foveate_max_regions: int = 1
    # Fraction of the surface's own size to pad the window by. The probe's 192 px
    # window scored 0/21 -- WORSE than no crop -- because an object that fills
    # its crop has lost the context the detector needs to call it an object.
    foveate_pad: float = 0.15
    # Foveate ONLY the surface the search posterior is currently inspecting,
    # rather than every container in view. The arm's cost is one detector call
    # per keyframe per region and its yield is concentrated on the surface the
    # agent came to look at: 3897 fires bought 16 target detections. Restricting
    # it there is the difference between paying the 18% control-loop cost all
    # episode and paying it while inspecting.
    foveate_active_only: bool = False
    # Evidence-score corroboration (P1i, FUS3DMaps-inspired 2026-07-19): a
    # detection that only re-matches an existing track from nearly the same
    # camera position adds little real corroborating evidence (no parallax)
    # -- it's still consistent with a one-off misdetection that happened to
    # repeat within the current keyframe's dwell. Earlier this was a hard
    # confirmed/tentative visibility gate (a track was hidden from tracks()/
    # candidates() entirely until re-observed from far enough away), but a
    # 30-episode eval showed that starved scene_graph.rebuild() of objects
    # early in exploration -- frontier scoring got "(no objects mapped yet)"
    # prompts and SR/SPL roughly halved (agent_stats: stop_reason=None,
    # select_none 0->22-24). Replaced with a soft evidence weight instead:
    # tracks are visible immediately from creation (ObjectTrack.evidence
    # accumulates every observation's det.score, discounted by
    # repeat_view_discount when the camera hasn't moved this far from the
    # track's first sighting). 0 disables the discount (every observation
    # gets full weight, matching pre-feature behavior).
    confirm_baseline_m: float = 0.15
    repeat_view_discount: float = 0.2
    fp_retraction: bool = False
    fp_disable_radius_m: float = 0.5
    target_every_step: bool = False
    cloud_stride: int = 4
    cloud_cap: int = 2000
    room_classifier: str = "none"  # none | place365
    # Container (anchor) layer -- floor -> room -> container -> object
    # (docs/DYNAMIC_SCENES.md). A track qualifies as a support surface when its
    # label is in graph.containers.CONTAINER_CATEGORIES AND its world top height
    # falls in this band with at least this much ground footprint. The geometry
    # half is what a DualMap-style word list alone cannot do: reject the
    # mis-segmented sliver labelled "table", and the "shelf" whose top lands
    # at 1.9 m where nothing is ever put down.
    container_top_h_m: Tuple[float, float] = (0.2, 1.4)
    container_min_area_m2: float = 0.06
    # How far an object's underside may sit from a surface and still count as
    # resting on it. Generous, because a mask-moment ellipsoid fitted from a
    # partial view is a coarse estimate of where an object's bottom is.
    container_support_tol_m: float = 0.15
    # A search candidate has to be a surface that is really there. Measured on
    # the accumulated map: 112 containers for a six-object hotel suite, 29 of
    # them "bed", 34 seen exactly once, 46 scoring under 0.5 -- against a search
    # budget of seven to nine inspections per episode. These gates cut it to 46
    # and take a cross-anchor destination from rank 20 to rank 7.
    container_min_obs: int = 2
    container_min_score: float = 0.5
    # Same-label surfaces this close are one piece of furniture. relink cannot
    # merge them because it requires co-observation -- right for an object and
    # its ghost, wrong for a bed seen on two different passes.
    container_merge_m: float = 1.0
    presence: PresenceConfig = field(default_factory=PresenceConfig)

