"""Structured configs registered with Hydra's ConfigStore so that typos in
yaml/CLI overrides fail fast instead of silently creating new keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

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
    # path, stuck-give-up) that the old system never had. See docs/AB_RESULTS.
    use_habitat_navmesh: bool = False
    navmesh_goal_radius: float = 0.1
    # Which mover drives the agent. This is THE sensor-only switch:
    #
    #   costmap   from-scratch A*/Voronoi planner + WaypointController
    #   navmesh   habitat's ShortestPathFollower -- PRIVILEGED (ground-truth
    #             geometry, including areas never observed), plus the
    #             is_reachable connectivity oracle
    #   pointnav  ASCENT's own mover: a frozen PointNav ResNet policy reading
    #             (rho, theta) + depth (ascent/ascent_policy.py:837). No map,
    #             no oracle, nothing the sensors did not supply.
    #
    # Left as None so `use_habitat_navmesh` keeps working unchanged for every
    # preset written before this existed; resolve_navigation() below is the
    # single place the two spellings are reconciled.
    navigation: Optional[str] = None
    pointnav_weights: str = "data/weights/pointnav_weights.pth"
    # VLFM's default (base_objectnav_policy.py:380), which ascent inherits.
    pointnav_stop_radius: float = 0.9
    pointnav_depth_shape: List[int] = field(default_factory=lambda: [224, 224])
    # Inside this range of a committed target the mover stops consulting the
    # network and creeps forward, because the 0.9 m stop radius is far outside
    # the success distance. ASCENT's rule (ascent_policy.py:920-927); the
    # terminal stop rule is what actually ends the approach.
    pointnav_approach_creep_m: float = 1.0
    # ---------------------------------------------------------------- S40
    # Radius at which the mover reports ARRIVING at an approach goal, the
    # signal `ShortestPathFollower` gives free via `navmesh_goal_radius`.
    # Measured: the navmesh arm ends 46 of 100 episodes on that signal
    # (`path_consumed`) and the pointnav arm ends ZERO, because PointNav emits
    # motion only. 8 of the 20 episodes navmesh solves and pointnav times out
    # on are ones where it reached the object and could not conclude it.
    # 0 disables, reproducing every pre-S40 number.
    pointnav_arrival_m: float = 0.0
    # Reachability, without a reachability oracle. `navmesh` mode rejects a
    # target on a disconnected navmesh island before committing
    # (unreachable_skip). A sensor-only agent cannot know that in advance, so it
    # commits, and abandons after this many steps -- marking the region so the
    # same false target is not re-acquired as a fresh track. ASCENT's
    # _try_to_navigate_step limit (ascent_policy.py:929).
    #
    # OFF by default. It is only asked when there is no oracle, which is true of
    # `costmap` as well as `pointnav` -- and in costmap mode it would land on
    # the same step as the existing approach deadline and convert its
    # stop-where-you-are into a return-to-exploring. That is a real behaviour
    # change to an arm every pre-S8 number was measured on, so it is opt-in
    # (the sensor presets set 100) rather than a default.
    approach_abandon_steps: int = 0
    # Frontier give-up: no more than this much displacement over this many
    # steps retires the frontier. Defaults reproduce the behaviour measured
    # up to S28; ASCENT uses 0.3 m / 20 steps (ascent/constants.py:234-235).
    frontier_stick_m: float = 0.2
    frontier_stick_steps: int = 15
    # What "stuck on a frontier" means. "displacement" asks whether the AGENT
    # moved and is what every pre-S8 number was measured on; it catches the
    # motionless push against an unmapped obstacle. "closing" is ASCENT's rule
    # (llm_planner.py:239-257) and asks whether the DISTANCE TO THE FRONTIER
    # changed -- which is the failure a reactive mover actually has, since it
    # orbits rather than freezing and so never trips the displacement test.
    frontier_stick_rule: str = "displacement"  # displacement | closing
    # Window for the action-history escape (planning/escape.py). 0 disables it.
    escape_window: int = 0
    # ---------------------------------------------------------------- S30
    # Three divergences from ASCENT found by decomposing the S8 result. All
    # default to the pre-S30 behaviour so no measured number moves; the
    # sensor presets turn them on.
    #
    # 1. Must the target be VISIBLE THIS FRAME for the terminal rule to fire?
    #    `terminal_rule: nearest_point` measures against the accumulated surface
    #    cloud, so a live detection was never one of its inputs. ASCENT asks the
    #    same question every step regardless (ascent_policy.py:434 computes
    #    cur_dis_to_goal, :910 tests it). Navmesh mode hid the difference by
    #    reporting arrival; a sensor-only mover has no arrival signal, and 28 of
    #    100 episodes closed to a median 0.67 m and never stopped.
    terminal_requires_detection: bool = True
    # 2. Does a PointNav STOP short of the goal retire the frontier? ASCENT
    #    overwrites it with one forward step and keeps the target
    #    (ascent_policy.py:705-711); it disables only on the stair paths.
    pointnav_stop_means_blocked: bool = True
    # 3. Must a frontier be A*-reachable on the costmap before it is pursued?
    #    A self-planning mover discards the path, so this lets a conservative
    #    costmap veto frontiers the mover could reach. ASCENT gates on nothing.
    #    Forced ON whenever the planner actually drives.
    frontier_reachability_gate: bool = True
    # ---------------------------------------------------------------- S31
    # ASCENT's stair traversal (ascent_policy.py:1075-1112): instead of driving
    # at a fixed point past the staircase, re-aim every step at the FARTHEST
    # thing in the depth image. On a flight that bearing points along the well,
    # so the agent walks through it; a fixed overshoot goal is a straight line
    # through whatever wall the stairwell turns around. Ratcheted so the
    # waypoint only ever moves closer to the stair end, and a network STOP is
    # overridden with a forward step rather than ending the climb.
    #
    # Off by default -- it replaces the overshoot goal outright, and every
    # cross-floor number so far was measured on that goal.
    climb_carrot: bool = False
    climb_carrot_m: float = 0.8  # ascent_policy.py:1075
    # ---------------------------------------------------------------- S32
    # Tilt the camera down every N steps for one frame, to look for stairwells
    # the level frustum never covers. Down stairs are PURE GEOMETRY
    # (mapping/stairs.py), so this needs no detector -- it only needs the hole
    # in the floor to be inside the frame, which at 0.88 m and 79 deg it stops
    # being about two metres out.
    #
    # Strictly down. S14a measured the UP probe and killed it: up-stair recall
    # went 19% at level pitch to 0% at +30 deg, because tilting up moves treads
    # out of frame. `look_up` is emitted here only to undo a `look_down`.
    #
    # 0 disables. Each probe costs two steps (tilt, observe, restore) out of
    # 500, and the sensor-only arm already times out in 43 of 100 episodes, so
    # the interval is a real trade rather than a free win.
    down_look_every: int = 0
    # ---------------------------------------------------------------- S37
    # Which control flow decides the action. `nav_agent` is OSG's FSM;
    # `ascent` is ASCENT's stateless per-step dispatch on the same perception
    # (agent/ascent_agent.py). The differences are structural rather than
    # parametric -- no committed states, the object goal re-aimed every step,
    # the frontier re-chosen every step with its damping -- which is why
    # porting them one at a time kept measuring null (S31, S33, S34).
    # `ascentnav` is the clean-room reimplementation: ASCENT's own ObstacleMap /
    # ValueMap / ObjectPointCloudMap under ASCENT's control flow, in src/ascentnav.
    policy: str = "nav_agent"  # nav_agent | ascent | ascentnav
    # ASCENT's obstacle band (VLFMConfig defaults), used only by `ascentnav`.
    ascent_min_obstacle_h: float = 0.61
    ascent_max_obstacle_h: float = 0.88
    # ---------------------------------------------------------------- S33
    # Which up-stair signal to believe. Measured on 250 stair poses, control
    # false-positive in brackets:
    #
    #   detector  YOLOE `stairs` + geometric gate            10% [0%]
    #   ascent    ASCENT's exact fusion: that mask INTERSECTED with RedNet's,
    #             no geometric gate (obstacle_map.py:520-524)  10% [0%]
    #   rednet    RedNet alone through the geometric gate      see AB_RESULTS
    #
    # `ascent` is the faithful port and it buys nothing HERE, because an
    # intersection cannot beat its weaker input and ASCENT's second opinion is
    # GroundingDINO, which this repo does not have. RedNet alone finds 53.6%.
    stair_up_mode: str = "detector"  # detector | ascent | rednet
    # Load RedNet at all. Required by the ascent and rednet modes; 626 MB.
    rednet_stairs: bool = False
    rednet_weights: str = "data/weights/rednet_semmap_mp3d_40.pth"
    # Max steps to reach a committed target on the navmesh before giving up the
    # approach. Large because navmesh drives the full distance to the object
    # (no viewpoint pre-positioning); the 12-step short-leg cap used in costmap
    # mode would otherwise cut the approach off while the target is still in view.
    navmesh_approach_steps: int = 200
    # Distance at which arriving at a stair frontier hands over to CLIMB.
    stair_reach_m: float = 0.6
    # How far past the staircase centroid to aim, so the agent walks THROUGH
    # the flight instead of stopping on the first tread.
    stair_overshoot_m: float = 1.5
    # Steps allowed for one floor transition before it is abandoned and the
    # staircase blacklisted. A failed climb is pure loss out of 500.
    climb_max_steps: int = 80
    # Hand over to a dedicated CLIMB state on reaching a stair frontier, whose
    # goal is a point PAST the flight so the agent walks through it rather than
    # stopping on the first tread.
    #
    # Probe on 6 cross-floor episodes (docs/AB_RESULTS.md): stair frontiers
    # alone lift SR 0/6 -> 1/6, and adding CLIMB takes it to 2/6 while making
    # the shared success far more direct (207 steps / SPL 0.17 -> 51 / 0.77).
    #
    # Note what it does NOT do: climb_ok stayed 0, i.e. the floor change always
    # completed a dozen steps AFTER the CLIMB state exited, during ordinary
    # frontier navigation. So the overshoot repositions the agent into the
    # stairwell rather than carrying it up, and CLIMB is better understood as an
    # approach behaviour than as a traversal. n=6 is a probe, not an A/B -- see
    # docs/AB_RESULTS.md for the pending 50-episode measurement.
    stair_climb_state: bool = True
    # Height change that counts as having changed floor during a climb.
    floor_gap_min_m: float = 0.9
    # When a climb counts as finished. "height" declares success on
    # floor_gap_min_m of gain; "topological" is ASCENT's rule -- the climb is
    # over when the agent is no longer on the staircase
    # (map_controller.py:299, `not is_robot_in_stair_map_fast(...)`), with the
    # height gain kept as a guard so stepping back off the bottom is not a
    # success.
    #
    # Why it matters: measured on the cross-floor split, 4 of 7 partial climbs
    # stall at 50-75% of a 2.7-3.2 m storey. 0.9 m is satisfied at a mid-flight
    # landing, so CLIMB declares victory there and hands back to EXPLORE. The
    # cross-floor episodes that DO succeed are the ones with a 1.15-1.60 m gap,
    # where 0.9 m happens to be most of the way.
    climb_exit_rule: str = "height"  # height | topological
    # How close to a staircase cell still counts as "on it". ASCENT uses the
    # agent radius (map_controller.py:198); OSG's cells come from detections of
    # treads AHEAD of the agent, so its own cell can sit behind the component.
    stair_exit_m: float = 0.5
    # Also look for a target candidate while travelling to a verify viewpoint,
    # not just while exploring.
    check_candidates_all_states: bool = False
    # Terminal stop rule: "depth" (median mask depth, the default) or
    # "nearest_point" (distance to the object's nearest accumulated surface
    # point). HM3D scores geodesic distance to a view_point, and view points
    # are tiled around the SURFACE -- median mask depth is the distance to the
    # middle of whatever the mask covers, so a 2 m sofa and a chair stop at
    # very different distances from their near edge.
    terminal_rule: str = "depth"
    # Engage the rule inside this range; stop at terminal_stop_m, or earlier if
    # closing stalls by less than terminal_progress_eps (blocked by the object
    # itself or by furniture in front of it).
    terminal_engage_m: float = 1.0
    terminal_stop_m: float = 0.6
    terminal_progress_eps: float = 0.1
    # Consecutive MOVING steps without closing before the approach is
    # abandoned. One is noise: an oblique approach barely changes the
    # distance to the nearest surface.
    terminal_stall_steps: int = 3
    # Percentile of point-cloud distances to treat as "the near surface".
    # 0 = the exact minimum, which is what ASCENT uses. Measured on dev50 (see
    # AB_RESULTS): the min of a cloud accumulated across hundreds of noisy
    # frames is set by its worst stray point, so a single bad pixel stops the
    # approach a metre early -- net -8 at stop_m=0.4, -12 at 0.6, with the
    # median distance to goal among surface stops at 0.401 m and 1.008 m.
    # At percentile 5 the same thresholds give 0.060 m and 0.044 m and net
    # goes to +1 and -3. Defaulting to 5 so that selecting terminal_rule=
    # "nearest_point" gets the version that works; pass 0 to reproduce the
    # ASCENT-faithful runs. Note this only runs under that rule -- the default
    # terminal rule is still "depth", which the fixed surface rule only
    # matched rather than beat.
    terminal_percentile: float = 5.0


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
    near_edge_dist_m: float = 1.5
    assoc_score_thresh: float = 0.4
    assoc_depth_gate_m: float = 0.5
    # Wasserstein data association requires the detection label to match the
    # track label. The ported VOOM matcher had no label check, but for SR eval
    # (navigate to a target CATEGORY) cross-category merges corrupt labels and
    # starve target candidates -- so gate on category by default.
    assoc_category_gate: bool = True
    room_seg_every_kf: int = 10
    # Free-space erosion (3x3 passes) used to sever doorways before the cores
    # are regrown into rooms. THIS is the parameter that decides how many rooms
    # exist; room_min_radius_m and room_door_width_m below are read by
    # VoronoiRoomSegmenter's constructor and then ignored.
    #
    # Was 12 (0.6 m). Swept over 12 real explored costmaps
    # (scripts/measure_room_seg.py):
    #
    #   erode  median rooms  >1 room  500-step episodes with >=3 rooms
    #      12           2.0     7/12                              0/2
    #      10           4.0     9/12                              0/2
    #       8           5.5    12/12                              1/2
    #       6           5.5    11/12                              2/2
    #       4           7.0    12/12                              2/2
    #
    # At 12 the room level of the scene graph carried almost no information --
    # half the episodes had a single room, so "which room is this frontier in"
    # had one answer. A partially explored costmap's free space is a narrow
    # region carved along the trajectory, and 0.6 m of erosion destroys every
    # core except the widest; the regrow step then assigns the whole map to it.
    #
    # 6 rather than 4: without ground-truth room labels there is no way to tell
    # "correctly found 7 rooms" from "over-segmented one room into 7", so this
    # takes the largest erosion that still clears the bar.
    room_erode_iters: int = 6
    min_room_cells: int = 60  # measured to have almost no effect over 30..120
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
    # Geometric false-positive retraction. A track born from a detection hard
    # against an image edge or at the far end of the depth range is a
    # hypothesis; when the agent later has that spot well inside its view cone
    # at close range and sees nothing of that category, the hypothesis is
    # refuted and the track is retracted. Ported from ASCENT, where it exists
    # because the dominant ObjectNav failure is walking across the house to a
    # confidently-detected object that is not there. Off by default (A/B-able).
    fp_retraction: bool = False
    # Re-detecting the same false positive within this of a retracted one, with
    # the same label, blacklists it on arrival instead of re-committing.
    fp_disable_radius_m: float = 0.5
    # Feed target-category detections to the object layer on EVERY step, not
    # only at keyframes (0.25 m / 30 deg apart). A target glimpsed while
    # crossing a doorway is otherwise missed entirely. Target-only by design --
    # the full detection set every step would inflate ObjectTrack.evidence and
    # invalidate the calibrated min_evidence / confirm_baseline_m thresholds.
    target_every_step: bool = False
    # Surface-point cloud for the terminal "nearest_point" rule: depth pixel
    # stride and per-track point cap. Only the episode target accumulates one.
    cloud_stride: int = 4
    cloud_cap: int = 2000
    # Room typing for the scene graph. "none" leaves RoomNode.label as None --
    # which, since the only other writer is the LLM scorer that no measured arm
    # enables, means rooms carry geometry but no semantics.
    room_classifier: str = "none"  # none | place365


@dataclass
class ExplorationConfig:
    # Whether the legacy per-frontier LLM *text* scorer runs (LLMTextScorer:
    # one chat call describing several frontiers from the scene graph, scoring
    # each 0-1). "off" leaves every frontier on `unscored_prior`, so this term
    # is constant and drops out of the ranking.
    #
    # This does NOT turn semantics off. Two other switches feed the same
    # selection and both override the flat prior: `value_map` (the CLIP value
    # map, whose score is read at selector.py:106-108 *before* the prior is
    # consulted) and `ranker` (ASCENT's forced-choice LLM ranker). A genuinely
    # geometry-only arm needs all three off.
    #
    # Renamed from `scorer`, whose "nearest"/"geometric"/"none" values read as
    # "no semantics" while the value map was in fact still running -- and whose
    # declared "vlm"/"random" values had no dispatch branch at all and silently
    # built the text scorer. The dataclass default was one of those phantoms
    # and never ran, because all four exploration groups set it explicitly.
    # "disabled", not "off": YAML 1.1 parses off/on/yes/no as booleans, and the
    # field is typed str, so `frontier_text_scorer: off` silently became the
    # string "False" and no longer matched anything.
    frontier_text_scorer: str = "disabled"  # disabled | llm_text
    top_n_frontiers: int = 5
    frontier_dedup_m: float = 1.0
    frontier_min_cells: int = 8
    # "wfd" clusters UNKNOWN cells and takes their centroid; "contour" is
    # ASCENT's method -- the contour of the explored region, split where it
    # stops bordering unexplored space, waypoint at each arc's midpoint.
    # The substantive difference is what the size filter measures: wfd drops
    # frontiers shorter than frontier_min_cells, contour drops frontiers whose
    # adjacent unexplored region is under area_thresh_m2. A 1 m doorway is kept
    # or dropped by what lies behind it rather than by its own width.
    extractor: str = "wfd"  # wfd | contour
    area_thresh_m2: float = 1.5  # ascent experiments/eval_ascent_hm3d.yaml:27
    # "utility" ranks by score/path_cost; "ascent" takes the value argmax with a
    # nearby shortcut and retires frontiers it keeps choosing without reaching.
    # Dividing by path cost systematically favours near frontiers, which is why
    # value_weight had to reach 4 before the value map moved anything.
    selector: str = "utility"  # utility | ascent
    nearby_distance_m: float = 3.0  # ascent_policy.py:99
    # ASCENT's commitment half, independently of its ranking half: retire a
    # frontier chosen 20 rounds running without the agent closing on it, or
    # returned to 20 times. selector="ascent" implies this; setting it alone
    # bolts commitment onto the score/path_cost ranking, which is the only way
    # to tell the two apart -- the "ascent" selector changes both at once.
    frontier_commit: bool = False
    # ASCENT's LLM: one synchronous forced choice among the top-k frontiers,
    # with the room-to-goal priors stated in the prompt. "none" keeps the
    # existing async per-frontier scorer, whose results are keyed by
    # Frontier.id and therefore almost never reach a decision.
    ranker: str = "none"  # none | ascent
    ranker_topk: int = 3
    # Call throttle. ASCENT gets the same effect from its 3 m nearby shortcut,
    # which skips the model entirely whenever anything is close by.
    ranker_every_steps: int = 20
    # Coarse level of the cascade: ask an LLM which STOREY the target is on, and
    # let the existing stair machinery execute the direction. This is the job
    # ASCENT actually gives its LLM (llm_planner.py:216-236); the frontier-level
    # ranker above is a job it does not, and measured as a regression on dev50.
    floor_llm: bool = False
    floor_ask_every: int = 60      # ascent MULTI_FLOOR_ASK_STEP_THRESHOLD
    floor_min_steps: int = 100     # ascent FLOOR_EXP_STEP_THRESHOLD
    # How hard a floor decision pushes the stairs that lead there. Stair
    # frontiers compete on the same score axis as explore frontiers, so a
    # direction with no weight behind it changes nothing.
    floor_llm_boost: float = 5.0
    subgraph_radius_m: float = 3.0
    # Where a frontier's description to the LLM comes from.
    #   "graph" -- spatial query against the accumulated scene graph: the room
    #              the point falls in and the tracks mapped near it.
    #   "frame_objects" -- only the object list comes from that frame; the room
    #              stays with the graph. Isolates the channel RAM++ would
    #              replace, since S27's "frame" moved both halves at once and
    #              66% of what it did to the room half was overwriting an
    #              explicit "unknown room".
    #   "frame" -- ASCENT's source: the room and object labels seen in the frame
    #              that first revealed this frontier (map_controller.py:800-830
    #              feeding llm_planner.py:418-419).
    # The two answer different questions. A frontier is a frontier because what
    # lies beyond it is unknown, so "already mapped near this point" tends to
    # describe the room the agent is standing in rather than the opening.
    # ---------------------------------------------------------------- S34
    # How often selection may run, and how often a pursuit already under way is
    # reconsidered. ASCENT does both EVERY STEP: it rebuilds the frontier list
    # (`map_controller.py:528`) and re-runs the whole selection
    # (`ascent_policy.py:684`) on every call to act(). Its "commitment" is
    # bookkeeping inside the selector -- sticky counters, a forced frontier --
    # not an FSM state the agent is stuck in.
    #
    # OSG commits for a median 23 steps. Behind a mover that closes 0.046 m per
    # step (measured against the navmesh follower's 0.076), that means holding a
    # target chosen from a map two dozen steps out of date.
    #
    # `select_every` is affordable to lower only with the reachability gate off:
    # selection runs A* over the top-N candidates, which ASCENT's does not.
    # 0 for reselect_every keeps the commit-until-exit behaviour every number
    # before S34 was measured on.
    select_every: int = 5
    reselect_every: int = 0
    frontier_desc: str = "graph"
    # How far a re-extracted frontier may drift and still count as the same
    # opening. Frontiers are rebuilt from scratch each round and their centroids
    # move as the map grows; matching keeps the ORIGINAL frame's description,
    # which is the whole point of the mechanism.
    frontier_desc_match_m: float = 1.0
    images_per_frontier: int = 1  # each image costs ~1-2k ctx tokens
    max_frontiers_per_call: int = 4
    unscored_prior: float = 0.3
    min_path_cost_m: float = 0.5
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
    los_visibility_penalty: float = 1.0
    # Score given to a staircase frontier. A stairwell has no meaningful
    # image-text or info-gain value -- the only question it answers is whether
    # this floor is still worth searching -- hence a flat prior, multiplied by
    # stair_explored_boost once the floor is exhausted.
    #
    # 0.6 measured on 50 paired cross-floor episodes (docs/AB_RESULTS.md):
    # SR 4% -> 10%, net +3 with nothing lost, floor changes 5/50 -> 13/50.
    # Inert unless mapping.multi_floor is on, since no stair detector is built
    # without per-floor maps. 0 disables stair frontiers entirely.
    stair_prior: float = 0.6
    stair_explored_boost: float = 3.0
    # Steps on a floor after which it counts as explored (ASCENT uses 100).
    # When a floor counts as exhausted, which is when stair frontiers get
    # stair_explored_boost. "no_frontiers" is ASCENT's rule
    # (ascent_policy.py:655): nothing left to explore here. "steps" is the old
    # behaviour and coincides with the floor-LLM gate below, which made the two
    # mechanisms fire on the same step (S12: 45/50 episodes bit-identical).
    stair_explored_rule: str = "no_frontiers"  # no_frontiers | steps
    floor_exp_steps: int = 100  # only read when stair_explored_rule == "steps"
    # Frames that must independently see a cell before it can be stair.
    # Frames a cell must be seen in before it can belong to a staircase, and
    # the minimum component size. Swept offline over 38 cross-floor episodes
    # (scripts/measure_stair_accumulation.py), measuring per-STAIRCASE recall
    # after accumulation rather than per-pose detection:
    #
    #   min_hits  min_cells   found   false components
    #          3         25    8/38                 38   <- was the default
    #          2         25   12/38                 37
    #          1         25   14/38                 26
    #          1         10   16/38                 36
    #
    # 1 dominates 3 on BOTH axes -- more staircases found AND fewer spurious
    # components. Keeping more cells lets a flight merge into one component
    # instead of fragmenting into pieces that each fail min_cells, and the
    # merged component is the one that contains the real staircase.
    #
    # This is the threshold that gates cross-floor behaviour, not the detector:
    # per-POSE recall went 0% -> 19% with a 2.5x larger model and produced no
    # extra climb attempts (S15).
    stair_min_hits: int = 1
    stair_min_cells: int = 25
    # On a failed climb, retire the staircase's CELLS (plus a margin) as well as
    # its centroid. Cuts repeat attempts by 89% but measured net -1 on 50
    # cross-floor episodes: the repeats were nearly free, and permanently
    # retiring a real staircase after one bad approach loses more than it saves.
    stair_retire_cells: bool = False
    # Semantic value map: paint each frame's image-text similarity to the target
    # prompt over the ground it observed, and rank frontiers by it instead of by
    # a flat prior. This is the mechanism VLFM/ASCENT use and the main thing
    # purely geometric exploration lacks. Off by default (A/B-able).
    # Measured on 50 paired episodes (docs/AB_RESULTS.md): with value_weight>=4
    # the map lifts SR 52% -> 56-58% and cuts steps-to-first-candidate by ~9 and
    # total steps by ~20. At value_weight=1 it is provably inert -- see below.
    value_map: bool = True
    value_model: str = "clip"  # clip | constant
    value_clip_name: str = "ViT-B/32"
    # Staged host-side: the nav container has no outbound network, and
    # data/weights is a named volume the host cannot write to.
    value_clip_root: str = "data/clip"
    # Score every Nth step: the view changes slowly at 0.25 m per step.
    value_stride: int = 1
    # Exponent on the value before the geometric boosts. NOT cosmetic: the
    # boosts (1+2*gain/gmax)*(1+2*align) span 1->9, while CLIP cosines across
    # indoor frames span roughly 0.20-0.24, so at weight 1 the geometric terms
    # out-range the semantic one about 7:1 and the value can only break
    # near-ties. Measured: weight 1 net 0, weight 4 net +2, weight 8 net +3, and
    # the mechanism metric (steps to first candidate) only moves at weight >= 4.
    # 4 and 8 differ by one episode -- noise -- so 4 is chosen on the efficiency
    # metrics, which are more stable at this sample size.
    value_weight: float = 4.0
    # Rank by value alone rather than value / path_cost, as ASCENT does.
    value_argmax: bool = False
    value_radius_m: float = 0.5
    value_prompt: str = "Seems like there is a {target} ahead."
    # Room prior from ASCENT's object-room knowledge graph, applied WITHOUT an
    # LLM: objects already mapped near a frontier imply a room type, and its
    # overlap with the goal's room distribution scores the frontier. Needs
    # data/priors (scripts/make_priors.py); silently inert if absent.
    knowledge_prior: bool = False
    knowledge_prior_path: Optional[str] = None
    knowledge_radius_m: float = 3.0
    knowledge_weight: float = 1.0



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
    # ---------------------------------------------------------------- S38
    # How long a VLM rejection sets a track aside. 0 = forever, which is what
    # every number before S38 was measured on.
    #
    # ASCENT's equivalent is a retryable gate, not a blacklist: the BLIP-2
    # double-check is re-evaluated every step until it passes
    # (map_controller.py:770-776), and a target is only abandoned at close
    # range where the view is best (ascent_policy.py:910-922). OSG asked once,
    # from wherever the agent stood, and a NO was final -- measured at 3 of 22
    # episodes ending within 0.5 m of the goal with no candidate left to stop
    # on, two of them at 3-5 cm.
    reject_cooldown_steps: int = 0
    min_obs: int = 3
    # Candidate quality gates: sliver/fragment detections (a chair edge seen
    # through furniture) must not trigger the expensive approach+verify loop.
    # 0.45 -> 0.70 on the strength of the 100-episode aligned run: 30 of 45
    # failures were "committed to something, walked to it (median 0.43 m), and
    # it was 7.48 m from any real goal". Those commits carried a median score of
    # 0.698 against 0.853 for successes. Raising the gate took far-commit
    # failures 30 -> 23 and SR 55% -> 58%, and every gained episode shows the
    # committed track's score rising from 0.40-0.57 to 0.79-0.97 -- the agent
    # skipping a weak detection and committing to a strong one later.
    #
    # It costs cross-floor episodes (23.8% -> 19.0%): they have less opportunity
    # to accumulate a confident detection inside the step budget.
    min_score: float = 0.70
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
    # Continuous approach re-check, ported from ASCENT. Every step of the
    # approach ASCENT scores the live frame against the target prompt and
    # latches a flag once the score clears a threshold
    # (map_controller.py:770-776); at the stop moment, if the flag never
    # latched it wipes the object clouds, disables the region and goes back to
    # exploring (ascent_policy.py:910-922). The signal costs nothing: it is the
    # same cosine that already drives the value map (map_controller.py:562), so
    # OSG re-uses its own value-map score the same way.
    #
    # Unlike `terminal` above this is a DENSE signal -- the whole approach gets
    # a vote, not one frame -- so a target that never once looks like the
    # category from any range is rejected even when the final close-up is
    # ambiguous.
    approach_recheck: bool = False
    # NOT ASCENT's 0.15. That number is a BLIP-2 ITM cosine; OSG's value map is
    # CLIP, whose cosines occupy a different range entirely, so the threshold
    # has to be calibrated against OSG's own distribution rather than copied.
    # Left at 0.0 (never rejects) until measured -- see
    # scripts/calibrate_approach_recheck.py.
    approach_recheck_thresh: float = 0.0
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
    # lower bound, was the culprit; see docs/AB_RESULTS.md).
    obstacle_low_m: float = 0.15
    obstacle_high_m: float = 1.5
    max_range_m: float = 5.0
    depth_stride: int = 4
    inflate_margin_m: float = 0.07
    # Cross-floor frame rejection: drop a depth frame from the costmap once the
    # agent is standing more than this far (m) above/below the floor the map was
    # started on. The costmap is a single ground plane, so an upper storey's
    # geometry otherwise lands on the lower one and corrupts frontier
    # extraction. 0 disables (= behaviour before this existed).
    #
    # 0.4 m is the shipped experiment value: comfortably above thresholds,
    # ramps and floor_y noise, well below a ~2.5 m storey. Only a stopgap for
    # TRANSIENT excursions (a stair landing) -- see NavAgent._act_inner.
    floor_reject_m: float = 0.0
    # Per-floor maps: one costmap/planner/value-map per storey, selected by
    # clustering the agent's standing height. Off reproduces the single-costmap
    # agent exactly (the stack is pinned to one layer). See mapping/floor_stack.
    multi_floor: bool = False
    # Heights within this of a known floor belong to it. Storeys are ~2.5 m.
    floor_band_m: float = 0.9
    # Consecutive observations before the active floor switches, and before an
    # unrecognised height becomes a new floor.
    floor_commit_steps: int = 4
    # Suspend floor allocation and switching while the CLIMB state is active.
    #
    # ASCENT changes its floor index in exactly one place -- when the agent
    # leaves the staircase (map_controller.py:299) -- so nothing can commit
    # mid-flight. OSG clusters height continuously, which on a staircase
    # allocates a layer at a mid-flight height and then switches to it: measured
    # on the cross-floor split, climbs end after 72 cm of gain against a 90 cm
    # threshold, so the exit comes from the floor-changed branch rather than the
    # height one. One episode records 18 switches between "two" floors while
    # covering only 1.34 m of height, which two real storeys 2.66 m apart cannot
    # produce.
    freeze_floor_in_climb: bool = False
    # The same freeze, but scoped to standing on a staircase at all rather than
    # to the CLIMB state. This is what ASCENT gates on. Measured need: the
    # episode with 18 floor switches recorded ONE climb attempt, so its
    # oscillation happened during ordinary frontier navigation and the
    # state-machine version could not reach it.
    freeze_floor_on_stairs: bool = False


@dataclass
class EvalConfig:
    split: str = "val"
    dataset_version: str = "v2"  # HM3D-semantics v0.2, 6 categories
    episodes_path: str = "data/datasets/objectnav/hm3d/v2/{split}/{split}.json.gz"
    scenes_dir: str = "data/scene_datasets/"
    num_episodes: int = -1  # -1 = all
    # >0 forces habitat to move to a new scene after this many episodes, so a
    # fixed-size subset spans the split instead of draining one scene first.
    # -1 = habitat default (group by scene, ~10000-step budget per scene).
    max_scene_repeat_episodes: int = -1
    # Habitat episode-iterator / simulator settings that habitat's own defaults
    # get WRONG relative to the ASCENT baseline we compare against. They are
    # pinned here (rather than inherited from benchmark/nav/objectnav/
    # objectnav_hm3d.yaml) so a protocol drift shows up as a config diff:
    #   shuffle          habitat default True;  ascent sets False
    #   max_scene_repeat_steps  habitat default 1e4; ascent sets 50000
    #   allow_sliding    habitat default True; objectnav_hm3d.yaml already sets
    #                    False, matching ascent -- pinned so it cannot silently
    #                    change if the benchmark yaml is ever swapped.
    # See docs: the ascent config is experiments/eval_ascent_hm3d.yaml:19-21.
    shuffle_episodes: bool = False
    max_scene_repeat_steps: int = 50_000
    allow_sliding: bool = False
    # Run only these episodes, as SCENE-QUALIFIED ids "<scene>:<episode_id>"
    # (e.g. "4ok3usBNeis.basis.glb:12"). habitat restarts episode_id at "0" in
    # every per-scene content file, so a bare id is ambiguous. Generated by
    # scripts/make_dev_split.py; used by the dev50 / dev50_mf A/B splits.
    episode_ids: Optional[List[str]] = None
    # Restrict the eval to specific scene ids (None/["*"] = all). Used by the
    # single-floor preset since the 2D scene graph cannot represent stairs.
    content_scenes: Optional[List[str]] = None
    save_viz: bool = True
    # Persist each episode's final occupancy grid, so room segmentation
    # parameters can be swept offline (scripts/measure_room_seg.py) instead of
    # re-running an episode per setting. ~40 KB each.
    save_costmap: bool = False
    # Per-step debug video: for each episode write viz/debug/ep<ID>.mp4 whose
    # frames are [live RGB + YOLOE segmentation overlay | top-down costmap] at
    # every step. The detector is re-run per step FOR VISUALIZATION ONLY (it
    # does not feed the object layer -- keyframe detection is unchanged), so SR
    # is unaffected; it roughly doubles detector load, hence off by default.
    debug_frames: bool = False
    rgb_width: int = 640
    rgb_height: int = 480
    hfov_deg: float = 79.0
    # Depth sensor range. These already come from objectnav_hm3d.yaml, so
    # pinning them is behaviourally a no-op -- but agent.navigation=pointnav
    # normalises depth against them before handing it to a frozen network, so
    # a silent drift here would be a silent input-distribution shift. Same
    # reasoning as allow_sliding / shuffle_episodes above.
    depth_min_m: float = 0.5
    depth_max_m: float = 5.0


@dataclass
class OSGConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    scene_graph: SceneGraphConfig = field(default_factory=SceneGraphConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42
    output_dir: str = "outputs/${now:%Y%m%d_%H%M%S}"


NAVIGATION_MODES = ("costmap", "navmesh", "pointnav")


def resolve_navigation(agent_cfg) -> str:
    """The single place `navigation` and `use_habitat_navmesh` are reconciled.

    `use_habitat_navmesh` predates the three-way switch and is set by every
    preset written before it (full_v1_navmesh, ascent_aligned, matched_navmesh,
    ...). Those must keep behaving exactly as measured, so an unset
    `navigation` falls back to it. Setting both to conflicting values is a
    config bug, not a precedence question -- say so rather than silently
    picking one.
    """
    mode = getattr(agent_cfg, "navigation", None)
    legacy = bool(getattr(agent_cfg, "use_habitat_navmesh", False))
    if mode is None:
        return "navmesh" if legacy else "costmap"
    mode = str(mode)
    if mode not in NAVIGATION_MODES:
        raise ValueError(
            f"agent.navigation={mode!r} is not one of {NAVIGATION_MODES}"
        )
    if legacy and mode != "navmesh":
        raise ValueError(
            f"agent.navigation={mode!r} contradicts agent.use_habitat_navmesh=true. "
            "Drop use_habitat_navmesh -- navigation=navmesh is the same thing."
        )
    return mode


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
    cs.store(group="eval", name="base_hm3d", node=EvalConfig)
