"""`agent` group: embodiment, the terminal approach, and how it drives.

The approach constants are the most heavily measured block in the config: three
distance-based stopping strategies stalled at dtg 0.107-0.147 m before the
depth stop, and `approach_to_viewpoint` exists because HM3D scores against
sampled goal viewpoints on fixed rings rather than against the object itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


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
    # target is seen. The navmesh follower arrives on the path's heading, which
    # need not point at the target, and one frame from one heading is a thin
    # basis for deciding an object is gone. 12 x 30 deg is a full circle.
    approach_scan_turns: int = 12
    # Beyond this distance from its own goal, a reported arrival is not one.
    # 0.0 disables the check, which is the shipped behaviour.
    #
    # On the navmesh the follower returns None for arrived AND unreachable, and
    # the approach treats both as an arrival: it stops. Measured on 00848, the
    # agent commits at step 1 to a track 0.81 m from the true object, is told
    # None on step 5 while still 6.4 m away, stops, and repeats it for all three
    # attempts -- episode over at step 78 with 420 steps unspent. Four to six
    # episodes per condition end that way and not one of them scores.
    #
    # 1.0 m is generous: the goal IS a viewpoint on a 0.8-2.0 m ring, so a real
    # arrival puts the agent on the goal itself, and the frontier side already
    # allows 0.9 m for the planner's own stopping radius.
    approach_false_arrival_m: float = 0.0
    # Re-derive the approach goal when the candidate's ellipsoid refines under
    # it. 0.0 disables the check, which is the shipped behaviour.
    #
    # `start()` computes a viewpoint on the ring around the object's centre AS
    # ESTIMATED AT COMMIT TIME, and never looks at it again -- but the estimate
    # is at its worst exactly then, and improves fastest during the approach,
    # when the agent is walking toward the object and every new keyframe is
    # closer and better framed than the last. Measured on 00848's red plate,
    # in_anchor_02: committed to a centre 0.328 m from truth, drove to a
    # viewpoint on THAT ring, stopped, and scored nothing -- while the same
    # track ended the episode at 0.059 m, a 5.6x refinement that arrived after
    # the only decision it could have changed. Success is scored at 0.18 m from
    # an authored viewpoint, so 0.328 m of centre error cannot score and 0.059 m
    # comfortably can.
    #
    # 0.15 m is half the error that lost that episode and comfortably above the
    # refiner's own step-to-step jitter, so a settled track never retargets.
    # A track already ruled unreachable from where the agent stands is not
    # ruled unreachable twice. 0.0 disables the check, which is the shipped
    # behaviour.
    #
    # `candidates.check()` runs every step, re-picks the same top candidate and
    # asks the pathfinder the same question from the same pose. Measured on
    # 00848's cross_anchor_02 red plate: the agent builds the REAL plate at
    # 0.04 m from truth with p=0.818, strikes it at step 150 and again at step
    # 151, hits `max_identity_rejections` (2) and spends the remaining 350 steps
    # not going to an object it had correctly mapped. Two strikes are meant to
    # be two separate failures to find it, not one verdict counted twice.
    #
    # 0.5 m is a real change of vantage and half the agent's own turning circle,
    # so a second strike means the pathfinder was asked from somewhere new.
    # Stop on ARRIVING at the approach viewpoint with the target in view.
    # 0.0 disables the check, which is the shipped behaviour.
    #
    # In viewpoint mode the depth stop is deliberately off -- it would fire en
    # route and leave the agent short of the ring success is measured on -- so
    # the only stop left was the follower reporting arrival, and the follower
    # does not report arrival while the agent is sitting on the goal. Measured
    # on 00848's cross_anchor_01 tin can, an episode where everything upstream
    # worked: viewpoint reached to 0.111 m, then 84 steps of the same 8640 px
    # detection at the same 0.769 m depth every 14 steps -- a 12-turn
    # revolution, spinning on the goal until the step budget ran out.
    #
    # 0.3 m is above the approach controller's own stopping tolerance and well
    # inside the 0.18 m-from-a-viewpoint success radius plus the ring's angular
    # sampling error, so arriving this close is arriving.
    viewpoint_stop_m: float = 0.0
    unreachable_restrike_m: float = 0.0
    approach_retarget_m: float = 0.0
    # Retargets allowed per approach. A cap, not a budget: a track that moves
    # this many times is not converging and the walk should end on the estimate
    # it has rather than chase one.
    approach_retarget_max: int = 3
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
    # Ask whether the pose the agent would DRIVE TO is reachable, not whether
    # the object's own position is.
    #
    # `_check_candidates` queries Habitat with the object's (x, z). For anything
    # resting on furniture that point is inside the furniture, and this is the
    # same fact that made the approach goal unwinnable until it was moved onto a
    # viewpoint ring: "a tabletop object's centre is an occupied cell inside the
    # furniture, so the follower stalls against it".
    #
    # Measured over the 36 authored target poses of 00829, with no detector
    # involved: the object's own position is off the navmesh in 6 of them, and
    # in ALL SIX an authored viewpoint is reachable. The benchmark defines
    # success as standing at such a viewpoint, so those episodes are solvable by
    # construction and the agent was giving up on them. Worst hit are exactly
    # the two lowest-SR targets -- the pitcher (3 of 6 poses) and the bleach
    # bottle (2 of 6).
    reachable_via_viewpoint: bool = False

    # Canonical mover selection.  ``None`` preserves the historical
    # ``use_habitat_navmesh`` spelling; resolve_navigation() is the only place
    # the two settings are reconciled.
    navigation: Optional[str] = None  # costmap | navmesh | pointnav
    # Control flow is independent of the mover.  The standard OSG FSM remains
    # the default; the two ASCENT policies are explicit alternatives.
    policy: str = "nav_agent"  # nav_agent | ascent | ascentnav

    # Sensor-only PointNav mover (ASCENT/VLFM compatible defaults).
    pointnav_weights: str = "data/weights/pointnav_weights.pth"
    pointnav_stop_radius: float = 0.9
    pointnav_depth_shape: List[int] = field(default_factory=lambda: [224, 224])
    pointnav_approach_creep_m: float = 1.0
    pointnav_arrival_m: float = 0.0
    pointnav_stop_means_blocked: bool = True

    # Navigation/termination controls imported with the ASCENT behavior
    # snapshot.  Defaults are intentionally inert for legacy presets.
    approach_abandon_steps: int = 0
    frontier_stick_m: float = 0.2
    frontier_stick_steps: int = 15
    frontier_stick_rule: str = "displacement"  # displacement | closing
    escape_window: int = 0
    # S47/S48: refuse to walk to a detection until it clears the evidence bar
    # `object_layer.candidates` applies (verification.min_score / min_obs /
    # min_bbox_px). `ascentnav` otherwise writes every detection above the
    # detector's own conf into its object cloud and treats a cloud as a goal.
    # MEASURED NULL on SR at both 0.70 and 0.60 (S48, S49) -- far-commits fall
    # 22 -> 8 and SR does not move -- so it stays off; the flag is kept because
    # the mechanism is real and n=100 cannot resolve a 2-3 episode effect.
    # S51: escape a wedge on REALISED displacement. `escape_window` below reads
    # the commanded action stream and fired zero times across 100 episodes while
    # 793 forwards produced no motion in 69 of them -- the stream alternates
    # turn/turn/blocked-forward, so neither of its predicates ever holds. This
    # counts forwards that went nowhere. 0 disables.
    stuck_escape_patience: int = 0
    commit_gate: bool = False
    # S50: turn in place on ARRIVING at a frontier. 62% of the steps the agent
    # spends within 3 m of the target object have it outside the FOV. MEASURED
    # NULL: 288 scans cost 17% of all steps and SR moved -2 (p = 0.75).
    scan_on_arrival: int = 0
    terminal_requires_detection: bool = True
    frontier_reachability_gate: bool = True
    check_candidates_all_states: bool = False
    terminal_rule: str = "depth"  # depth | nearest_point
    terminal_engage_m: float = 1.0
    terminal_stop_m: float = 0.6
    terminal_progress_eps: float = 0.1
    terminal_stall_steps: int = 3
    terminal_percentile: float = 5.0

    # Stair sensing and traversal.  RedNet is loaded only when explicitly
    # enabled by an imported or combined multi-floor preset.
    ascent_min_obstacle_h: float = 0.61
    ascent_max_obstacle_h: float = 0.88
    stair_up_mode: str = "detector"  # detector | ascent | rednet
    rednet_stairs: bool = False
    rednet_weights: str = "data/weights/rednet_semmap_mp3d_40.pth"
    stair_reach_m: float = 0.6
    stair_overshoot_m: float = 1.5
    climb_max_steps: int = 80
    stair_climb_state: bool = True
    floor_gap_min_m: float = 0.9
    climb_exit_rule: str = "height"  # height | topological
    stair_exit_m: float = 0.5
    climb_carrot: bool = False
    climb_carrot_m: float = 0.8
    down_look_every: int = 0

