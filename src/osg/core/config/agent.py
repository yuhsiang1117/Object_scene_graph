"""`agent` group: embodiment, the terminal approach, and how it drives.

The approach constants are the most heavily measured block in the config: three
distance-based stopping strategies stalled at dtg 0.107-0.147 m before the
depth stop, and `approach_to_viewpoint` exists because HM3D scores against
sampled goal viewpoints on fixed rings rather than against the object itself.
"""
from __future__ import annotations

from dataclasses import dataclass


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


