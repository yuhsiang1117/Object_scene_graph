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
    """Presence belief per object (objects/presence.py, docs/history/DYNAMIC_SCENES.md).

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
class EllipsoidConfig:
    """How an object's ellipsoid is built from depth (docs/history/AB_RESULTS.md,
    the ellipsoid sections; objects/point_buffer.py, objects/merge.py).

    Every default reproduces the layer as it was measured before this group
    existed -- mask-moment initialisation, a 2D-only refine, projection-only
    association, no buffer -- so no arm changes unless it asks to.
    """

    # M1 initialisation. "moments" = Ellipsoid.init_from_detection (2D ellipse
    # moments + the mean of 50 sampled depths, third axis (a+b)/2, camera-
    # aligned rotation). "points" = Ellipsoid.init_from_points: the mask's
    # depth pixels cleaned (erosion, flying pixels, MAD) and fitted in a
    # gravity-aligned frame.
    init: str = "moments"
    erode_px: int = 2
    # A pixel whose depth jumps by more than this fraction of its own depth to
    # a 4-neighbour sits on an occlusion boundary: mixed foreground/background.
    flying_rel: float = 0.05
    # Depths further than k robust sigmas (1.4826 * MAD) from the median.
    mad_k: float = 3.0
    # Extents from the [p, 100-p] percentiles, rescaled as for a uniform spread.
    extent_pct: float = 5.0
    # Added to every half-extent of a points fit. A buffer keeps one centroid
    # per voxel, so its outermost points sit up to half a voxel inside the
    # surface; with mask erosion that shrank 0-25 cm objects by ~22% (log
    # ratio -0.25) and YCB targets by ~27% (S78).
    extent_pad_m: float = 0.0
    # The axis along the view, which one depth image cannot see the back of:
    # "lateral" = the horizontal half-width (objects are about as deep as they
    # are wide), "mean_ab" = (lateral + vertical)/2 as the moments init does,
    # "thickness" = what the visible surface spans, at least `min_axis_m`.
    depth_axis: str = "lateral"
    # Where a points fit takes its horizontal axes from. "fit" = the line of
    # sight (one view) or the points' principal axis (several); "manhattan" =
    # the room's dominant wall direction from depth normals
    # (point_buffer.ManhattanFrame), the nearest of its four; "world" = the
    # world axes, which in HM3D happen to be the walls' (a reference, not a
    # deployable choice). Refitting the dev buffers in the world axes lifted
    # IoU 0.170 -> 0.267 (S78).
    yaw_mode: str = "fit"
    min_axis_m: float = 0.02
    # Pixels per detection the cleaning keeps at most (evenly strided, no RNG).
    max_pixels: int = 6000
    # A bbox within this many px of an image edge is a truncated view: its
    # extent is a lower bound, so it seeds a provisional track and may not
    # enter the buffer.
    border_px: int = 4
    # Re-initialise a provisional track from its first full view.
    reinit_truncated: bool = False

    # M2 buffer: per-track voxel hash of cleaned points from good views only.
    buffer: bool = False
    voxel_m: float = 0.03
    buffer_cap: int = 6000
    view_range_m: Tuple[float, float] = (0.3, 4.5)
    min_valid_frac: float = 0.5
    # Let a truncated view (bbox at the image edge) add its points too. Its
    # surface is real, only incomplete, and a buffer accumulates pieces; a
    # large object touches the edge in nearly every view, so the rule kept 91
    # dev tracks (median 1.8 m) out of the buffer entirely: IoU 0.036 (S78).
    buffer_truncated: bool = False

    # M3 refine. "2d" = WassersteinRefiner as measured. "joint" = 2D residuals
    # plus point-to-surface distances from the buffer, weighted `depth_weight`
    # px per metre and balanced by residual count. "points" = refit from the
    # buffer directly, then nothing else.
    refine_mode: str = "2d"
    depth_weight: float = 200.0
    depth_max_pts: int = 300
    # Centre-move reject for refines that see depth; < 0 = refine_max_center_move_m.
    depth_max_center_move_m: float = -1.0
    # Buffer views a "points" refit needs; fewer falls back to the 2D refine,
    # whose ellipsoids scored IoU 0.006-0.060 on the dev first episodes (S78).
    refit_min_views: int = 2
    # Accept a points refit whatever its centre move when the CURRENT
    # ellipsoid holds less than this share of its own buffered points: it no
    # longer describes its own data. 0 = off (the move limit always applies).
    # The move limit kept 43 dev objects at their initial ellipsoid (IoU 0.045)
    # while 2+ views of points said otherwise (S78).
    refit_override_overlap: float = 0.0
    # A points refit the move limit refuses still becomes the track's
    # REPORTED geometry; association keeps projecting the last ellipsoid the
    # limit accepted (ObjectTrack.assoc_ellipsoid), so what a detection joins
    # is unchanged. Every relaxation of the limit cost purity, and that cost
    # came through association (S78).
    decouple_geometry: bool = False

    # M4 association.
    normalize_labels: bool = False
    # An unmatched detection joins the same-label track holding at least this
    # share of its cleaned points (inside the ellipsoid inflated by
    # `assoc_3d_inflate`, or in the track's buffer voxels). 0 = off.
    assoc_3d_overlap: float = 0.0
    assoc_3d_inflate: float = 1.25
    # What "overlap" means for 3D association and the finalise merge.
    # "ellipsoid": inside the inflated ellipsoid OR within a voxel of the
    # buffer. "voxel": within a voxel of the buffer only -- a wrong, oversized
    # ellipsoid then cannot swallow its same-label neighbours (dev round 1:
    # "ellipsoid" took purity 0.924 -> 0.905 at V4 and -> 0.525 at V5).
    overlap_mode: str = "ellipsoid"
    # Voxels of slack around the buffer when counting a point as overlapping.
    overlap_dilate: int = 1
    # Leave a detection unmatched when its points overlap two or more
    # same-label tracks: adjacent same-label objects (a row of cabinets, the
    # pillows on a bed) are where 3D association took the wrong one.
    assoc_3d_unique: bool = False
    # Same-frame, same-label detections whose points overlap this much are
    # one object split in two masks: merged before association. 0 = off.
    merge_split_overlap: float = 0.0
    # Veto a projection match when less than this share of the detection's
    # cleaned points lies in the track's ellipsoid (inflated by
    # `assoc_3d_inflate`) or near its buffer. 0 = off. With room-aligned,
    # visible-depth ellipsoids (S78 round 4) adjacent same-label chairs
    # started trading observations through the 1%-bbox-overlap gate.
    assoc_2d_min_overlap: float = 0.0
    # Reassign instead of vetoing: a projection match keeping under this share
    # of the detection's points goes to the unmatched same-label track holding
    # at least `assoc_3d_overlap` of them, if there is one. 0 = off.
    assoc_2d_reassign: float = 0.0

    # M5 finalise: once the camera is `finalize_dist_m` away and the track has
    # gone unobserved for `finalize_unseen_kf` keyframes, refit it from its
    # buffer, merge same-label tracks that hold `merge_overlap` of each
    # other's points, and compress the buffer to `summary_voxels`.
    finalize: bool = False
    finalize_dist_m: float = 3.0
    finalize_unseen_kf: int = 5
    merge_overlap: float = 0.3
    # Require `merge_overlap` BOTH ways (min of the one-way overlaps) instead
    # of either way: "either" lets a track whose buffer once took a
    # neighbour's points absorb that neighbour, and the merged buffer the
    # next one (dev round 2: purity 0.92 -> 0.68 without compression).
    merge_symmetric: bool = False
    summary_voxels: int = 256

    # Save-time consolidation (ObjectLayer.consolidate, run by pass 1 just
    # before the snapshot is written): a same-label track with at most
    # `consolidate_max_obs` observations is folded into the nearest same-label
    # track with `consolidate_ratio` times as many whose world AABB is within
    # `consolidate_gap_m` of its own, unless the two were ever observed in the
    # same keyframe (two masks at once are two objects). The kept track's
    # ellipsoid is not refit. 0 = off.
    # Measured on I6r1 (AB_RESULTS S79), not adopted: max_obs 2 / 5 cut the
    # pass-1 snapshot's extra tracks per object by 8.5% / 14%, but pass 2's
    # by only 1.3% / 2.9%, with IoU lower (p = 3e-6 at 5) and purity -0.6 pt.
    consolidate_max_obs: int = 0
    consolidate_gap_m: float = 0.2
    consolidate_ratio: float = 3.0


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
    # ghost are not (docs/history/DYNAMIC_SCENES.md, the in-anchor ghosting bug).
    link_max_frame_gap: int = 50
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
    # (docs/history/DYNAMIC_SCENES.md). A track qualifies as a support surface when its
    # label is in graph.containers.CONTAINER_CATEGORIES AND its world top height
    # falls in this band with at least this much ground footprint. The geometry
    # half is what a DualMap-style word list alone cannot do: reject the
    # mis-segmented sliver labelled "table", and the "shelf" whose top lands
    # at 1.9 m where nothing is ever put down.
    # Measure a container's top from its OWN floor rather than from y=0.
    # `container_top_h_m` is a band above the floor; `top_height` returns an
    # absolute world height. They coincide on the ground floor and diverge on
    # every storey above it, where the band then rejects everything -- 00808's
    # upper floor holds 367 mapped tracks and produced zero containers. Off by
    # default because it changes which surfaces exist on a multi-storey map.
    containers_floor_relative: bool = False
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
    # Extent-aware companion to `container_merge_m`, in shadow sigmas. Two
    # same-label containers merge when one ground shadow's centre lies within
    # this many of the other's radii -- a test that scales itself to the object,
    # which a flat radius cannot: it has to serve a 2 m bed and a 0.5 m
    # nightstand at the same time.
    #
    # Measured against HM3D's own semantic annotations, the flat 1.0 m leaves
    # 00829 with 16 `bed` container nodes in a house that has ONE bed. Single-
    # link clustering at 2 m collapses those to 7 in 5 rooms, so over half the
    # inflation is neighbouring fragments -- what this catches -- and the rest is
    # the detector calling sofas and benches `bed` across the house, which no
    # merge rule should touch.
    #
    # 0 keeps the flat-only behaviour every measured arm ran on.
    container_merge_sigma: float = 0.0
    presence: PresenceConfig = field(default_factory=PresenceConfig)
    ellipsoid: EllipsoidConfig = field(default_factory=EllipsoidConfig)

