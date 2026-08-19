"""Structured configs registered with Hydra's ConfigStore so that typos in
yaml/CLI overrides fail fast instead of silently creating new keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# ~40 common indoor categories used as the fixed detector vocabulary in
# addition to the episode target. HM3D ObjectNav v2 targets are a subset.
DEFAULT_VOCABULARY: List[str] = [
    "chair", "sofa", "armchair", "plant", "bed", "toilet", "tv_monitor",
    "table", "desk", "cabinet", "shelf", "dresser", "wardrobe", "nightstand",
    "lamp", "pillow", "cushion", "picture", "mirror", "window", "door",
    "sink", "bathtub", "shower", "towel", "counter", "stool", "bench",
    "refrigerator", "oven", "microwave", "washing machine", "stove",
    "fireplace", "stairs", "rug", "curtain", "clothes", "book", "box", "basket",
]


@dataclass
class AgentConfig:
    max_steps: int = 500
    forward_m: float = 0.25
    turn_deg: float = 30.0
    success_distance: float = 0.1  # paper mode: 0.13
    initial_scan: bool = True  # 360 deg spin at episode start to seed the map
    camera_height: float = 0.88
    agent_radius: float = 0.18
    # Terminal APPROACH phase: walk toward the verified object while a
    # detection stays visible, stopping once its bbox is large enough (a
    # borderline "object recognizable but distant" crop measured ~25k px^2
    # in verify_debug samples; this threshold asks for a noticeably closer
    # view than that before considering the approach complete).
    # Terminal stop is primarily DEPTH-based: RGB-D gives the real metric range
    # to the detected target, so we stop every object at the same distance
    # regardless of its pixel size -- unlike a bbox-area threshold, which trips
    # a 2 m sofa at ~2.6 m but a chair at ~1 m. Stop once the target's median
    # mask depth falls to approach_stop_depth_m (agent is close and the object
    # is visible -> inside the densely-tiled viewpoint region). bbox is only a
    # fallback for when the mask has no valid depth.
    approach_stop_depth_m: float = 1.0
    approach_stop_bbox_px: float = 40_000.0
    # Detection-based terminal stop (depth-stop, + bbox fallback). When False the
    # approach relies solely on navmesh-arrival / deadline to terminate.
    approach_depth_stop: bool = True
    # Drive to a VIEW POINT and stop there, rather than closing on the object
    # until its depth crosses a threshold. HM3D scores success against the
    # nearest sampled goal viewpoint, and those sit on rings at fixed radii; a
    # depth stop at 1.0 m lands between the 0.8 m and 1.2 m rings. Navmesh mode
    # only -- the costmap path already pre-positions at a viewpoint.
    approach_to_viewpoint: bool = False
    # Turns allowed on arriving at a viewpoint, sweeping in place until the
    # target is seen. The follower arrives on the path's heading, which need not
    # point at the target, and one frame from one heading is a thin basis for
    # deciding an object is gone. 12 x 30 deg is a full circle.
    approach_scan_turns: int = 12
    # Turns allowed on arriving at a viewpoint, sweeping in place until the
    # target is seen. The navmesh follower arrives on the path's heading, which
    # need not point at the target, and one frame from one heading is a thin
    # basis for deciding an object is gone. 12 x 30 deg is a full circle.
    approach_scan_turns: int = 12
    approach_max_steps: int = 12  # ~3 m of travel at forward_m=0.25
    # Tighter-than-default planner/controller stopping precision for the
    # final APPROACH segment only (P1f). HM3D success is a geodesic
    # distance to a view_point; the general 0.3 m (planner) / 0.2 m
    # (controller) tolerances used for frontier/verify-view travel left
    # enough slack that a short geodesic detour around a nearby thin
    # obstacle (wall corner, furniture edge) blew the 0.13 m success
    # radius on episodes where we were already 5-8 cm away in a straight
    # line. Kept above the 0.05 m costmap resolution to stay robust to
    # grid discretization.
    approach_goal_tolerance_m: float = 0.12
    approach_arrival_tol_m: float = 0.1
    # Approach-goal selection. The default (_nearest_free_xy) snaps the goal to
    # the nearest RAW-free cell to the object center (~0.05 m away), which often
    # lands in a pocket walled by the object's OCCUPIED cells: the Voronoi
    # medial axis keeps 0.25 m clearance so it has no node there, and A* (only
    # OCCUPIED is hard-blocked) cannot enter the enclosed pocket -> planner_no_
    # path, agent strands ~1.9 m out (scripts/analyze_approach.py: 17/18
    # path_consumed = planner_no_path, 81% stall >1 m from a free, object-
    # adjacent goal on a real viewpoint). When approach_navigable_goal is set,
    # the goal is placed at approach_standoff_m from the object ALONG THE RAY
    # TOWARD THE AGENT -- the side the object was actually observed from, so it
    # sits in open, reachable space at roughly the distance successes stop at
    # (~0.9 m; the depth-stop still fires en route at <=1 m).
    approach_navigable_goal: bool = False
    approach_standoff_m: float = 0.75
    # Drive on Habitat's own navmesh (ShortestPathFollower) instead of the
    # from-scratch costmap planner + waypoint controller -- mirroring the OLD
    # ObjectSceneGraph stack, which publishes a goal point and lets Habitat plan
    # and execute. Perception / scene graph / frontier selection are unchanged;
    # only path planning + execution (and the terminal approach: navigate to the
    # object position, then STOP on arrival, like the old /goal_object) switch
    # to the navmesh. Removes the self-built-costmap failure modes (planner_no_
    # path, stuck-give-up) that the old system never had. See docs/INVESTIGATION.
    use_habitat_navmesh: bool = False
    navmesh_goal_radius: float = 0.1
    # Max steps to reach a committed target on the navmesh before giving up the
    # approach. Large because navmesh drives the full distance to the object
    # (no viewpoint pre-positioning); the 12-step short-leg cap used in costmap
    # mode would otherwise cut the approach off while the target is still in view.
    navmesh_approach_steps: int = 200
    # Snap navmesh goals at the TARGET's floor instead of substituting the
    # agent's own height. Without this, a candidate one storey up snaps to
    # whatever lies under the agent, so is_reachable reports it unreachable and
    # _check_candidates blacklists it -- every cross-floor target is discarded.
    # Needs floor.enabled for the floor heights. See docs/MULTI_FLOOR.md.
    navmesh_3d_goals: bool = False


@dataclass
class DetectorConfig:
    name: str = "yoloe"
    weights: str = "data/weights/yoloe-11s-seg.pt"
    conf: float = 0.3
    imgsz: int = 512
    half: bool = True
    device: str = "cuda"
    vocabulary: List[str] = field(default_factory=lambda: list(DEFAULT_VOCABULARY))


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


@dataclass
class ExplorationConfig:
    scorer: str = "vlm"  # vlm | llm_text | nearest | random
    top_n_frontiers: int = 5
    frontier_dedup_m: float = 1.0
    frontier_min_cells: int = 8
    subgraph_radius_m: float = 3.0
    images_per_frontier: int = 1  # each image costs ~1-2k ctx tokens
    max_frontiers_per_call: int = 4
    unscored_prior: float = 0.3
    min_path_cost_m: float = 0.5
    # Search posterior (docs/DYNAMIC_SCENES.md, C3). Off by default: it changes
    # where the agent goes, so it must be an explicit A/B. When on, mapped
    # surfaces compete with frontiers under ONE index, b*d/c, so exploring and
    # re-searching stop being separate subsystems.
    search_posterior: bool = False
    # d(x): chance a visit to a surface would find the target if it is there.
    # Taken from the same measured detection rate as the presence filter's
    # absence recall (0.812 over 6790 logged expectations).
    search_detect_prob: float = 0.8
    # Length scale for "things are moved short distances": b decays as
    # exp(-d/L) from where the object was last believed to be.
    search_proximity_len_m: float = 4.0
    # Scales a frontier's utility against a surface's, i.e. the price of
    # preferring unmapped space over a plausible surface. Must be non-zero or
    # the agent stops exploring once its surfaces are exhausted.
    search_frontier_weight: float = 1.0
    # Ask the text LLM where a class of object gets put down, for targets the
    # static table in graph/priors.py does not cover (every YCB target). Cached
    # to disk, so a run is deterministic after the first and the priors used are
    # inspectable afterwards.
    # How close counts as having inspected a surface, and how long to stay
    # committed to reaching one before giving up on it.
    # A surface in plain view counts as searched without driving to it: the
    # binding budget is inspections (about fifty steps each), not travel.
    search_glance_detect_prob: float = 0.35
    search_glance_range_m: float = 4.0
    # Finishing the room you are in beats crossing the house and coming back.
    search_same_room_bonus: float = 4.0
    search_arrival_m: float = 1.2
    search_max_steps: int = 60
    # Credit for a surface the agent set off towards but never reached: it has
    # barely been ruled out, and spending full belief on it would retire the
    # very surfaces that were never inspected.
    search_unreached_credit: float = 0.25
    affinity_llm: bool = False
    affinity_cache: str = "outputs/affinity_cache.json"
    # Information-gain weighting: boost frontiers that expose more unknown area
    # (estimated as the count of UNKNOWN costmap cells within info_gain_radius_m
    # of the frontier), so exploration commits to directions that open large
    # unexplored regions instead of crawling the nearest small frontier. A
    # frontier's score is multiplied by (1 + info_gain_weight * gain/gain_max),
    # normalized against the best candidate each round. 0 weight disables it.
    info_gain_weight: float = 2.0
    info_gain_radius_m: float = 2.5
    # Continuity / momentum bonus: prefer the next frontier to lie AHEAD of the
    # agent's current heading, so exploration sweeps continuously instead of the
    # greedy argmax ping-ponging between far-apart frontiers (~30 steps/trip).
    # 0 = off; higher = stronger preference for staying the course.
    continuity_weight: float = 0.0
    # Line-of-sight visibility down-weighting: multiply the score of frontiers
    # the agent has clear line of sight to (no wall between => same room) by this
    # factor, so exploration prefers occluded, behind-a-doorway frontiers that
    # open new rooms. 1.0 = off; <1.0 penalizes visible/same-room frontiers.
    # Drive to the frontier's free-snapped centroid instead of an UNKNOWN
    # frontier cell. On the navmesh, unknown space is snapped by snap_point to
    # an arbitrary nearby navigable point, so the follower reports
    # arrived-or-unreachable at once: measured 289 stub-blocks vs 24 give-ups
    # over 100 episodes, 53% of selections repeating an earlier one, and 2x the
    # planned distance walked. See exploration/selector.frontier_goal_xy.
    frontier_goal_free_cell: bool = False
    # Measure the RANKING path cost to the free centroid while still driving to
    # the frontier cell. Fixes the planner failures without the coverage loss
    # that moving the drive goal causes -- see select_frontier.
    frontier_cost_free_cell: bool = False
    los_visibility_penalty: float = 1.0


@dataclass
class LLMConfig:
    base_url: str = "${oc.env:OLLAMA_HOST,http://localhost:11434}/v1"
    api_key: str = "ollama"
    text_model: str = "qwen2.5vl:3b"
    vlm_model: str = "qwen2.5vl:3b"
    timeout_s: float = 120.0
    max_image_px: int = 512
    # Some OpenAI-compatible providers (e.g. NVIDIA NIM vision models) return
    # malformed output when sent response_format=json_object; setting this false
    # omits that param and parses the JSON out of the plain-text reply instead.
    send_response_format: bool = True


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
    min_obs: int = 3
    # Candidate quality gates: sliver/fragment detections (a chair edge seen
    # through furniture) must not trigger the expensive approach+verify loop.
    min_score: float = 0.45
    min_bbox_px: int = 3000
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


@dataclass
class MappingConfig:
    resolution_m: float = 0.05
    # Lower bound 0.15 (was 0.1): more floor tolerance before slightly-raised
    # ground (thresholds, rugs, ramps, floor_y drift) reads as an obstacle at
    # the robot's feet. Ceiling kept at 1.5 (aligning the FULL band to the old
    # [0.15, 0.88] regressed SR 40% -> 28.6% -- the 0.88 m ceiling, not the
    # lower bound, was the culprit; see docs/INVESTIGATION.md).
    obstacle_low_m: float = 0.15
    obstacle_high_m: float = 1.5
    max_range_m: float = 5.0
    depth_stride: int = 4
    inflate_margin_m: float = 0.07


@dataclass
class FloorConfig:
    """Multi-floor support (docs/MULTI_FLOOR.md). Every default reproduces the
    current single-floor behaviour, so `floor.enabled=false` is a no-op path."""

    enabled: bool = False
    # Log the estimated floor but keep using the latched floor_y. Lets the
    # estimator be validated against the per-scene navmesh ground truth from
    # scripts/scene_floors.py before any behaviour depends on it.
    estimate_only: bool = True
    # Within this of a known level -> standing on it; beyond it -> on stairs,
    # and the floor id freezes until a level is committed.
    level_tol_m: float = 0.35
    # Minimum separation between distinct levels. Sits in the gap between a
    # real storey (>=2.2 m) and a split-level / sunken room (<=0.5 m).
    merge_m: float = 0.6
    # Separation required to register a NEW level. 1.8 is measured: HM3D
    # storeys are 2.5-3.4 m apart, staircase landings <=1.1 m from the floor
    # below. At 0.6 a single descent in XB4GS9ShBRE registered 4 floors.
    new_level_m: float = 1.8
    # Consecutive steps at a height before a switch or a new level commits
    # (6 steps ~= 1.5 m at forward_m=0.25). Prevents id thrash mid-staircase.
    min_dwell_steps: int = 6
    # Secondary source: pre-register floors seen but not yet visited from the
    # depth cloud. Off by default -- a depth histogram also peaks on ceilings.
    point_cloud_peaks: bool = False
    # One Costmap2D + room segmenter + room labels per storey, instead of one
    # shared map. Implies enabled=true and estimate_only=false. Without it an
    # upper floor is never mapped at all (its points fall outside the band
    # around the latched floor_y), so it yields no frontiers and the agent has
    # nothing to explore there. See docs/MULTI_FLOOR.md.
    per_floor_costmap: bool = False
    # --- stair detection (Stage 4) -------------------------------------------
    # Find steppable regions and mark them traversable, so the staircase stops
    # reading as a wall. Implies per_floor_costmap (a stair only means anything
    # once each storey has its own map).
    stairs: bool = False
    # Max height change between neighbouring cells the embodiment can step
    # over. Matches Habitat's navmesh max_climb so the costmap and the navmesh
    # agree on what is passable.
    climb_limit_m: float = 0.2
    # Below this a cell is flat floor, not a step -- without it every cell in
    # the map qualifies as a "staircase".
    stair_min_dh_m: float = 0.03
    stair_cell_m: float = 0.1        # ZONDA's coarse grid
    stair_min_cells: int = 12        # fine cells; rejects speckle
    # Minimum vertical rise for a region to count as a staircase. 1.0 is
    # measured: on the single-floor gate every false positive rose 0.33-0.75 m,
    # while a real HM3D storey is 2.5-3.4 m up. At the old 0.3 the detector
    # fired in 28/35 single-floor episodes.
    stair_min_rise_m: float = 1.0
    stair_min_obs: int = 2           # semantic corroboration bar (object layer)
    stair_min_evidence: float = 1.0
    # Require a YOLOE `stairs` track to corroborate. Default False: measured at
    # only 15% coverage on multi-floor episodes, so requiring it would discard
    # most real staircases. See docs/MULTI_FLOOR.md.
    stair_require_semantic: bool = False
    # Cap on relabelled area as a fraction of the known map. The FREE relabel
    # is permanent, so a runaway mask could carve through real obstacles.
    stair_max_area_frac: float = 0.05
    stair_detect_every_kf: int = 5
    # --- cross-floor exploration (Stage 5) -----------------------------------
    # Let the agent decide to LEAVE its storey. This is the binding constraint:
    # on the 100-episode v1 run all 24 cross-floor episodes failed and not one
    # ever attempted a transition. Implies per_floor_costmap; uses the height
    # layer to find portals, and does NOT depend on floor.stairs (which does
    # not work -- see docs/MULTI_FLOOR.md).
    cross_floor: bool = False
    # ASCENT's gate: only reason about storeys when the best frontier left on
    # this floor is further away than this.
    near_frontier_m: float = 4.0
    # ASCENT's T/10 -- stops the agent oscillating between floors.
    switch_min_interval: int = 50
    # MFNP's two free guards: too early the current floor is barely mapped, too
    # late there is no budget left to recover from a wrong choice.
    no_switch_before: int = 50
    no_switch_after_frac: float = 0.7
    # A portal is a patch of another storey visible from this one. Its lower
    # bound is new_level_m (below that it is a split level, not a floor).
    portal_max_delta_m: float = 4.0
    portal_min_cells: int = 20
    # Use the target CATEGORY to time the switch, not just geometry. The
    # geometric rule cannot fire until the floor is exhausted (~step 200
    # measured), leaving too little budget to search the next one. Seeing none
    # of the target's usual companions on a floor that HAS been mapped is
    # evidence to leave early; seeing several is reason to stay.
    # LLM-free -- a fixed co-occurrence table, see graph/priors.py.
    use_target_evidence: bool = True
    # Earliest step the evidence rule may fire (the geometric no_switch_before
    # still guards the geometry-only path).
    early_switch_step: int = 30
    # Objects mapped on this floor before zero evidence means "not here"
    # rather than "not looked yet".
    min_objects_to_judge: int = 8
    # Distinct context categories that make a floor worth staying on.
    strong_evidence: int = 2
    # How long strong context may hold the agent on a floor before it is
    # treated as stale. A bathroom on this storey does not mean THIS storey's
    # bathroom holds the toilet, and without expiry the "stay" rule suppressed
    # cross-floor switching almost entirely (3 of 24 episodes, was 14). 0 = never.
    evidence_patience_steps: int = 120
    portal_deadline_steps: int = 120
    # Vertical travel that counts as "the climb is under way", so the portal
    # goal is held against same-floor frontier re-selection.
    portal_progress_m: float = 0.25
    # Grace before abandoning a portal that produces no vertical movement at
    # all. Must cover walking ACROSS the floor to reach the stairs, not just
    # the climb: at 0.25 m/step, 40 steps was only 10 m and abandoned 21 of 28
    # attempts before the agent had even arrived. The deadline still caps the
    # total pursuit, so this only controls how patient the "not started yet"
    # check is.
    portal_grace_steps: int = 100
    # Horizontal distance at a candidate height that proves it is a storey and
    # not a staircase landing. See FloorEstimator; 0 disables.
    min_horizontal_run_m: float = 2.5


@dataclass
class EvalConfig:
    # Navigation attempts per query. DualMap allows several: a failed attempt
    # updates the map and the agent goes again. Scoring one attempt is a
    # STRICTER protocol than the system being compared against, so this exists
    # to match theirs rather than to flatter ours. 1 keeps the old behaviour.
    attempts: int = 1
    # `objectnav` loads the standard HM3D episode dataset. `ycb_authored`
    # discovers scene-layout JSON files written by habitat-data-collector and
    # builds equivalent ObjectNav episodes from the placed YCB objects.
    mode: str = "objectnav"
    split: str = "val"
    dataset_version: str = "v2"  # HM3D-semantics v0.2, 6 categories
    episodes_path: str = "data/datasets/objectnav/hm3d/v2/{split}/{split}.json.gz"
    scenes_dir: str = "data/scene_datasets/"
    num_episodes: int = -1  # -1 = all
    # >0 forces habitat to move to a new scene after this many episodes, so a
    # fixed-size subset spans the split instead of draining one scene first.
    # -1 = habitat default (group by scene, ~10000-step budget per scene).
    max_scene_repeat_episodes: int = -1
    episode_ids: Optional[List[str]] = None
    # Restrict the eval to specific scene ids (None/["*"] = all). Used by the
    # single-floor preset since the 2D scene graph cannot represent stairs.
    content_scenes: Optional[List[str]] = None
    save_viz: bool = True
    # Per-step debug video: for each episode write viz/debug/ep<ID>.mp4 whose
    # frames are [live RGB + YOLOE segmentation overlay | top-down costmap] at
    # every step. The detector is re-run per step FOR VISUALIZATION ONLY (it
    # does not feed the object layer -- keyframe detection is unchanged), so SR
    # is unaffected; it roughly doubles detector load, hence off by default.
    debug_frames: bool = False
    rgb_width: int = 640
    rgb_height: int = 480
    hfov_deg: float = 79.0


YCB_TARGET_LABELS: Dict[str, str] = {
    "003_cracker_box": "cracker box",
    "005_tomato_soup_can": "tomato soup can",
    "011_banana": "banana",
    "019_pitcher_base": "pitcher",
    "024_bowl": "bowl",
    "025_mug": "mug",
    "029_plate": "plate",
    "037_scissors": "scissors",
}


@dataclass
class YCBAuthoredConfig:
    """Runtime discovery and deterministic episode generation for authored YCB layouts."""

    data_root: str = "/datasets/habitat-data-collector/data"
    layout_root: str = "/datasets/habitat-data-collector/outputs/dualmap_authoring"
    scenes: List[str] = field(default_factory=lambda: ["*"])
    layout_types: List[str] = field(default_factory=lambda: ["static"])
    layout_indices: List[int] = field(default_factory=lambda: [1, 2, 3])
    # Restrict episodes to these targets (YCB handle or label; empty = all).
    # Several assets in the collector's dataset render in a way the open-vocab
    # detector cannot recognise at any authored viewpoint, so their episodes
    # measure asset coverage rather than dynamic-scene handling. The layouts
    # themselves are DualMap's original data and are never edited.
    targets: List[str] = field(default_factory=list)
    starts_per_target: int = 1
    seed: int = 42
    manifest_cache_dir: str = "outputs/ycb_manifests"
    # Mid-episode relocation (docs/DYNAMIC_SCENES.md, Phase 2). -1 disables it
    # and every episode behaves exactly as before. When enabled, an episode
    # whose layout is a dynamic one starts the world in the paired STATIC
    # layout and moves the objects to the episode's own poses at this step --
    # so the goals are where the object ends up, and the change is something
    # the agent can witness rather than wake up to.
    relocate_at_step: int = -1
    # `in_view` waits until the target is actually visible from the current
    # pose, `out_of_view` waits until it is not, `any` fires immediately. The
    # two conditions measure different things: in_view is the clean test of
    # negative evidence, out_of_view tests whether the search recovers.
    relocate_when: str = "any"
    # If the visibility condition never comes true, relocate anyway this many
    # steps later, rather than silently turning the episode into a static one.
    relocate_deadline_steps: int = 120
    # Two-pass benchmark (docs/DYNAMIC_SCENES.md, Phase 2). Pass 1 explores the
    # STATIC layout and writes one snapshot per scene to `map_out`; pass 2 runs
    # the moved layout and starts from `map_in`, so the map the agent navigates
    # with is genuinely stale. The staleness IS the experiment -- an agent that
    # rebuilds from scratch is never wrong about anything and measures nothing.
    map_out: str = ""
    map_in: str = ""
    target_labels: Dict[str, str] = field(
        default_factory=lambda: dict(YCB_TARGET_LABELS)
    )
    viewpoint_radii_m: List[float] = field(
        default_factory=lambda: [0.8, 1.2, 1.5, 2.0]
    )
    viewpoint_angular_samples: int = 24
    viewpoint_max_snap_m: float = 0.5
    viewpoint_dedup_m: float = 0.2
    viewpoint_min_visible_pixels: int = 20
    start_min_geodesic_m: float = 3.0
    start_sample_attempts: int = 2000


@dataclass
class OSGConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    scene_graph: SceneGraphConfig = field(default_factory=SceneGraphConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    floor: FloorConfig = field(default_factory=FloorConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    ycb: YCBAuthoredConfig = field(default_factory=YCBAuthoredConfig)
    seed: int = 42
    output_dir: str = "outputs/${now:%Y%m%d_%H%M%S}"


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=OSGConfig)
    cs.store(group="agent", name="base_default", node=AgentConfig)
    cs.store(group="detector", name="base_yoloe", node=DetectorConfig)
    cs.store(group="scene_graph", name="base_default", node=SceneGraphConfig)
    cs.store(group="exploration", name="base_vlm", node=ExplorationConfig)
    cs.store(group="llm", name="base_ollama", node=LLMConfig)
    cs.store(group="verification", name="base_on", node=VerificationConfig)
    cs.store(group="mapping", name="base_default", node=MappingConfig)
    cs.store(group="floor", name="base_default", node=FloorConfig)
    cs.store(group="eval", name="base_hm3d", node=EvalConfig)
    cs.store(group="ycb", name="base_authored", node=YCBAuthoredConfig)
