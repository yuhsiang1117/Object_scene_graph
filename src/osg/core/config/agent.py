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
    # Close the last metre. With `approach_to_viewpoint` the walk ends on the
    # innermost viewpoint ring that has a FREE costmap cell, and beside a bed
    # or a desk the inflated costmap has none nearer than about a metre: on
    # DualMap's released benchmark 20 failed trials stopped 1.0-1.6 m from an
    # object whose track was within 0.6 m of it, having reached the goal to
    # 0.1 m, while the navmesh floor came within 0.9 m of the object in 13 of
    # them. When the viewpoint is reached and the agent is still farther than
    # this from the track centre, the approach asks the navmesh for the
    # nearest navigable point to the centre and walks there before it stops.
    # 0 keeps the viewpoint as the stopping pose.
    approach_close_last_metre_m: float = 0.0
    # Stop at the CLOSEST pose the approach reached, not the pose it happens to
    # be in when its terminal rule fires. Measured over 81 trials the agent
    # gives up a median 0.18 m between the two; on the trials it loses, 0.04 to
    # 1.72 m, and ten of sixteen losses had already come inside 1.0 m and then
    # stopped outside it. When the current pose is worse than the best by more
    # than this, the approach walks back before stopping. Distance is to the
    # agent's own estimate of the track centre, so nothing is asked of the
    # simulator. 0 disables it.
    approach_stop_at_best_m: float = 0.0
    # Where the closing walk gets its goal. `navmesh` asks the pathfinder, which
    # is ground-truth geometry and privileged in a sensor-only arm; `costmap`
    # asks the agent's own depth-built grid for the nearest cell to the object
    # with `approach_close_clearance_m` of clearance (0 = agent_radius + 0.05).
    # The walk's advantage is a goal that is standable by construction, and that
    # does not require the mesh.
    approach_close_source: str = "navmesh"
    approach_close_clearance_m: float = 0.0
    # Arrival by distance on the navmesh. The follower reports arrival only
    # inside its 0.1 m goal radius, and an agent that moves in 0.25 m steps
    # can circle a goal at 0.11-0.13 m for a hundred steps without ever
    # landing in it: on DualMap's released benchmark the three 00829 cracker
    # boxes were committed from 0.03 m, driven to within 0.13 m of the
    # viewpoint, and spun there to the deadline. A consumed path is declared
    # once the agent is within this distance of its approach goal. 0 leaves
    # arrival to the follower.
    approach_arrival_m: float = 0.0
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
    # Needs floor.enabled for the floor heights. See docs/history/MULTI_FLOOR.md.
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
    # A third answer to "can the agent get to this candidate": the nearest free
    # costmap cell to the object, which is where `_aim` sends the approach when
    # no ring pose exists. Read on the released benchmark's in-anchor failures:
    # every candidate rejection was "unreachable", including live tracks 0.05 m
    # from the object with belief 0.95 -- a plate on a desk against a wall,
    # whose ring poses were all occupied or unknown at that moment and whose
    # own position is off the navmesh. The place the approach would actually
    # drive to was reachable all along.
    reachable_via_nearest_free: bool = False
    # Snap navmesh goals and reachability queries onto the AGENT'S island.
    # HM3D navmeshes carry furniture tops as tiny separate islands (00880:
    # one floor of 63 m2 and three tops of 2 m2), and `snap_point` from the
    # agent's height lands a goal beside a desk on the desk-top island because
    # it is nearer in 3D than the floor next to the desk. Every such goal then
    # reads as unreachable: 37 of 37 candidate rejections on the released
    # benchmark's in-anchor failures, including a live track 0.05 m from the
    # object. With this on, the query is constrained to the island the agent
    # stands on (habitat_sim snap_point(..., island_index)), and falls back to
    # the plain snap only when that returns nothing.
    navmesh_snap_on_agent_island: bool = False

    # Canonical mover selection.  ``None`` preserves the historical
    # ``use_habitat_navmesh`` spelling; resolve_navigation() is the only place
    # the two settings are reconciled.
    navigation: Optional[str] = None  # costmap | navmesh | pointnav
    # Control flow is independent of the mover.  The standard OSG FSM remains
    # the default; the two ASCENT policies are explicit alternatives.
    policy: str = "nav_agent"  # nav_agent | ascent | ascentnav
    # Build OSG's world model ALONGSIDE whichever policy is driving, so one
    # mapping pass leaves both artifacts: the policy's own map and the scene
    # graph `graph/map_store.save_map` writes. Only useful with `ascentnav`,
    # which otherwise leaves no object layer at all -- and measured on 00800 it
    # reaches both storeys inside 500 steps where `nav_agent` needs 1500 to
    # reach the second. Fed after the action is decided, so it cannot change it.
    osg_world_model: bool = False
    # The world model's OWN detector. ASCENT drives on D-FINE, which is
    # closed-set COCO and cannot see a cracker box; the scene graph needs the
    # open-vocabulary head.
    osg_world_model_weights: str = "data/weights/yoloe-11s-seg.pt"

    # Sensor-only PointNav mover (ASCENT/VLFM compatible defaults).
    pointnav_weights: str = "data/weights/pointnav_weights.pth"
    pointnav_stop_radius: float = 0.9
    pointnav_depth_shape: List[int] = field(default_factory=lambda: [224, 224])
    pointnav_approach_creep_m: float = 1.0
    pointnav_arrival_m: float = 0.0
    # The creep (`pointnav_approach_creep_m`) is a blind forward with no
    # obstacle test. It assumes the goal is walkable; an approach goal is a free
    # cell in a DEPTH-BUILT costmap, and on the sensor arm 27 of 44 stranded
    # approaches had a goal the navmesh calls non-navigable. Habitat refuses the
    # forward, rho never falls, and the press runs to the budget -- a median 262
    # approach steps, ending 33 of those episodes.
    #
    # With this many consecutive creep steps that fail to beat the closest
    # approach so far by `pointnav_creep_stall_eps`, the driver reports arrival:
    # the agent is as close to the goal as the geometry allows. 0 keeps the
    # unconditional press, which is what every measured ascent arm ran on.
    pointnav_creep_stall_steps: int = 0
    pointnav_creep_stall_eps: float = 0.05
    # Steer the creep (planning/creep.py, OSG's approach only): turn toward the
    # goal when it is more than this many degrees off the heading, and sweep
    # headings round a refused forward, instead of the blind forward above.
    # 0 keeps the blind creep. Benchmark v2 seed 42: 00862's banana stopped
    # 1.15 m from the object after ten forwards into the table it sat on, and
    # 00873's pitcher at 1.10 m in all four OSG arms.
    approach_creep_steer_deg: float = 0.0
    # With the steered creep: forwards that fail to close on the goal by
    # `pointnav_creep_stall_eps` before it reports a stall (turns do not count).
    approach_creep_stall_steps: int = 6
    pointnav_stop_means_blocked: bool = True

    # Navigation/termination controls imported with the ASCENT behavior
    # snapshot.  Defaults are intentionally inert for legacy presets.
    approach_abandon_steps: int = 0
    frontier_stick_rule: str = "displacement"  # displacement | closing
    escape_window: int = 0
    # --- ASCENT control-flow port (S71) ------------------------------------
    # `_double_check_goal` latches when the value map's BLIP-2 cosine clears
    # this (`map_controller.py:774`). Read by ascentnav only.
    blip_gate_threshold: float = 0.15
    # `_detect_passive_stair_entry` (`map_controller.py:626-672`). Only valid
    # with the strict stair mask (`stair_up_mode: ascent`); the constructor
    # refuses the union.
    passive_stair_entry: bool = True
    # ASCENT's `_initialize` returns TURN_LEFT until `_initialize_step > 11`,
    # which is 13 calls (`ascent_policy.py:689-697`).
    initialize_turns: int = 13
    # `ascent` = REF's mirrored-depth down-stair trigger
    # (`obstacle_map.py:549-562`); `lip` = OSG's rewritten "missing floor lip"
    # test. The flag sets `_look_for_downstair_flag`, and every frame it is up
    # is spent tilting at a possible phantom drop-off.
    downstair_detector: str = "ascent"
    # S75: run OSG's VLM verifier on the frame at the arrival STOP and take the
    # give-up path when it refuses. An ADDITION -- the reference has no
    # verifier -- so it is off by default and the default arm is unaffected.
    # Needs `verification.enabled=true`; the verifier fails open on a transport
    # error, so check `verify_errors` against `verify_calls` before believing a
    # null.
    verify_on_stop: bool = False
    # Which view the stop check is asked about. `live` is the frame at the
    # moment of stopping (S75, blind on the 42% of stops where the detection
    # has left the frame); `stored` is the best look the agent had at the cloud
    # it is stopping on, dropped whenever that cloud is burned (S76).
    verify_stop_view: str = "live"
    terminal_requires_detection: bool = True
    frontier_reachability_gate: bool = True
    terminal_rule: str = "depth"  # depth | nearest_point
    terminal_engage_m: float = 1.0
    terminal_stop_m: float = 0.6
    terminal_progress_eps: float = 0.1
    terminal_stall_steps: int = 3
    terminal_percentile: float = 5.0

    # A close look at a surface (agent/close_look.py, docs/history/SR_PROPOSAL_CLOSE_LOOK.md):
    # drive to a facing pose on `close_look_ring_m`, turn until the surface is
    # inside 15 degrees, hold `close_look_hold_steps` with the detector on, and
    # only then conclude anything. 1.5 m is where in-situ recall peaks (0.58,
    # against 0.50 inside 1.5 m and 0.28 beyond 2.5 m) and where a 79-degree
    # frame is 2.4 m wide, so an object that moved 0.7 m along its surface is
    # still in it. Both uses are off by default; each is its own A/B arm.
    #
    # `opportunistic`: on a keyframe in EXPLORE / GOTO_FRONTIER, detour to an
    # affording container within `close_look_trigger_range_m` that has not had
    # its look, then resume the pursuit. Measured before this existed: across
    # the 32 cross-anchor failures without a detection the search reached the
    # object's own surface in 6 (five of them detector walls), and 12 had the
    # object in frame only at 2-4 m in passing.
    # `before_absence`: the approach is about to conclude absence at a track it
    # never saw live; look from the ring first, and re-aim if the target shows.
    close_look_opportunistic: bool = False
    close_look_before_absence: bool = False
    close_look_ring_m: float = 1.5
    close_look_trigger_range_m: float = 2.5
    # Cost, measured on the three-trial smoke of the first cut: a look was 15-49
    # steps (drive + up to six facing turns + a four-step hold), and eight of
    # them took 218 of a 481-step episode. The budget is the thing cross-anchor
    # is already short of, so the defaults below bound a look at ~15 steps and
    # an episode at ~90: 25 steps of driving, then look from wherever that got
    # to; a two-step hold (left, right) that ends on the start heading.
    close_look_max_per_episode: int = 6
    close_look_max_steps: int = 25   # drive budget per look, then look from here
    close_look_face_turns: int = 6
    close_look_hold_steps: int = 2   # an even count returns to the start heading
    # Spend looks by belief, not by order of encounter. Measured (A/B, arm
    # "opportunistic"): the nearest-first look hit its six-look cap by step ~155
    # in 66 of 107 episodes on the surfaces around the stale position, and in
    # 17 of the 32 cross-anchor failures the object's own surface came inside
    # trigger range only after the budget was gone. With this on, a surface in
    # view earns a look only when its search belief -- the same prior and
    # survived factor the selection round uses, relative to the best surface on
    # the floor -- is at least `close_look_min_belief`, and the highest belief
    # in view wins over the nearest.
    close_look_by_belief: bool = False
    close_look_min_belief: float = 0.5

    # Stair sensing and traversal.  RedNet is loaded only when explicitly
    # enabled by an imported or combined multi-floor preset.
    ascent_min_obstacle_h: float = 0.61
    ascent_max_obstacle_h: float = 0.88
    stair_up_mode: str = "detector"  # detector | ascent | rednet; `ascentnav` sets `ascent` (the strict RedNet AND GroundingDINO fusion), `rednet` (the union) is its A/B
    rednet_stairs: bool = False
    rednet_weights: str = "data/weights/rednet_semmap_mp3d_40.pth"
    stair_reach_m: float = 0.6
    climb_max_steps: int = 80
    stair_exit_m: float = 0.5
    # Let a floor pursuit become a climb. Until now State.CLIMB was never
    # entered: the carrot, the on-stairs test and the exit rule were all here
    # and nothing assigned the state, so the sensor-only agent had no way up a
    # staircase at all. Off by default because it changes what the agent does
    # on arrival at a portal.
    climb_enabled: bool = False
    # AN ABLATION, NOT A MOVER: once a climb has started, Habitat's navmesh
    # follower -- privileged, `agent.navigation=navmesh`'s own -- drives the
    # staircase to the pursued flight's far end at the target storey's height.
    # Everything that decides to climb, finds the stairs and gets to them stays
    # sensor-only. It prices the climb: on Benchmark v2 seed 42 about 15 of 25
    # cross-storey OSG episodes chose to change storey and never rose more than
    # ~0.3 m, spending ~370 steps on a flight. Never compare an arm with this on
    # against a sensor-only one as if it were one.
    climb_oracle: bool = False
    # Climb along a WALKED route through the height layer (mapping/stair_route.py)
    # instead of at a carrot on a flight blob: cells joined only where their
    # surfaces differ by a riser, walls on this floor left out, the goal flat
    # floor near the storey asked for; followed by turning toward a point
    # `climb_route_lookahead_m` ahead and walking within the deadband, not by
    # the PointNav policy. A route from where the agent stands that leaves the
    # floor within `climb_route_start_m` starts the climb there. Measured before
    # it (outputs/mfv2/diag_climb): on 00821 the blob's foot sat 1.5 m beside
    # the real one and two 300-step climbs rose 0.01 m; on the probe flight
    # with exact tread heights the carrot climb took 275 steps for 2.8 m.
    climb_route: bool = False
    climb_route_start_m: float = 2.5
    climb_route_lookahead_m: float = 0.5
    climb_route_deadband_deg: float = 20.0
    climb_route_replan_steps: int = 10
    climb_route_stall_steps: int = 40
    # Extra steps the stall rule allows before the climb first gains 0.1 m: a
    # route climb starts up to `climb_route_start_m` out on the flat floor, and
    # that walk plus one 12-turn look-round is more than 40 steps. 00873 (seed
    # 42, static 50002, mfv2_osg_descent) ended "stalled" at step 41 heading
    # for the lip 1 m away. 0 = as measured.
    climb_route_approach_steps: int = 0
    # With no route onto the stairs in view from the start floor, walk to the
    # climb's goal (the stair mouth pursued) and look round there, before the
    # far-end walk meant for landings. 00862 (seed 42, static 50007,
    # mfv2_osg_mouth): two climbs began 1.2 m short of the remembered mouth and
    # the far-end walk took the agent 4 m along the upper floor, twice.
    climb_route_mouth_first: bool = False
    # The least height a route must gain to be walked when it does not reach
    # flat floor. From the foot of a flight the camera sees no tread surface
    # above its own height (~0.88 m), only risers: on 00821 the mapped treads
    # ended 0.85 m up, so a metre was never on offer. The walk is re-planned
    # as the eyes rise.
    climb_route_min_gain_m: float = 0.3
    # A climb STARTS on a route only if it reaches the storey or gains this
    # much: 00878's 0.3 m route up something step-like began three climbs.
    climb_route_start_gain_m: float = 0.5
    # A route probe starts a climb only where the route leaves the floor
    # within this distance of the staircase being pursued -- its cells for a
    # flight, else the pursuit's goal. 0 = anywhere, as measured. 00821 (seed
    # 42, cross 50006, mfv2_osg_mouth_walk): walking to the flight at (-1.77,
    # 1.43), the probe found a 0.5 m "route" up something step-like at (3,
    # -0.5), 4 m away; the climb pressed into it 31 times and stalled at 0.0 m,
    # and its failure was then filed against the flight it never reached.
    # Portal pursuits are left free: a portal is a sightline, not a staircase.
    climb_route_start_near_m: float = 0.0
    # How often a storey request with nothing to drive to tries a route.
    climb_route_probe_steps: int = 10
    # Bridge an unseen band of treads within this distance of the agent
    # (`stair_route.plan_stair_route`): the tread just ahead is under the
    # camera. 0 = off. 00862's second leg stalled on it.
    climb_route_gap_fill_m: float = 0.0
    # Stamp the detector's `stairs` masks into the stair-evidence grids on every
    # keyframe. Until now that happened only on the single frame of a periodic
    # look-down, and `down_look_every` defaults to 0, so on every ycb preset the
    # evidence was never accumulated and `StairDetector.extract` had no caller.
    stair_evidence_every_kf: bool = False
    # Steer the climb at the detector's stamped stair cells (farthest within
    # 3 m ascending, nearest descending) rather than at the farthest depth ray.
    # The depth ray is ASCENT's carrot and works from ON the flight; ours starts
    # within reach of a target that is often beside it -- on 00821 seven climbs
    # pushed forward 238 times and rose 0.00 m.
    # Steer the climb at the next tread of the flight the pursuit chose, from
    # the height layer. Keeps the goal ON the flight, which is what the PointNav
    # mover climbs (slowly). Falls through to the other carrots when the target
    # was not a flight.
    # Carry a climb through a half-landing onto the next flight, instead of
    # handing back to exploration part-way up a multi-flight staircase.
    climb_relink_flights: bool = False
    climb_flight_carrot: bool = False
    # Steps of real exploration to guarantee after a failed attempt, before a
    # candidate commit may pre-empt the round again. 0 is the shipped
    # behaviour.
    #
    # `rearm` already returns the agent to EXPLORE, but EXPLORE is not where
    # the exploration ROUND lives: `_act_inner` runs `candidates.check` first,
    # every step, and a commit there jumps straight back to APPROACH. The
    # round -- and with it `floor_switch`, the only call site of
    # `_try_floor_switch` -- is additionally rate-limited to one run per
    # `exploration.select_every` steps.
    #
    # Measured on 00800 cross_anchor_01, both episodes: attempt 1 fails at step
    # 70, the next commit lands at step 74, and the four EXPLORE steps between
    # them are all inside the 5-step rate limit -- so `select` never runs.
    # `frontier_select_log` and `search_log_events` are both EMPTY over 180 and
    # 362 steps, and `flights_seen`/`portals_seen` never appear in
    # `agent_stats`. Three attempts went to three same-label tracks on the
    # starting storey while the goal sat on the other one.
    #
    # Holding the commit for a window forces one honest round: the frontier
    # selector runs, the search posterior gets to name a storey, and the floor
    # switch gets its chance -- after which candidates resume normally. It is a
    # hold, not a ban: nothing is blacklisted and the next window commits.
    explore_after_failed_attempt_steps: int = 0
    # Take the climb's direction from the flight being pursued rather than
    # from the difference of two storey heights. 0/False is shipped behaviour.
    #
    # The height difference is unreliable when the estimator holds two levels a
    # few centimetres apart: `target_y > here_y` is False at equality and the
    # tie silently means DOWN. Measured on 00800 cross_anchor_01
    # (outputs/mf5_pass2_v5): flight kind `up`, `climb_here_y_x100` 16,
    # `climb_target_y_x100` 16, `climb_start_down` 1, 200 steps on the flight,
    # `climb_max_dy_x100` 0.
    climb_direction_from_flight: bool = False
    # `_flight_carrot` never aims at a cell closer than this, horizontally.
    # 0.0 is the shipped behaviour (nearest in-band cell, wherever it is).
    #
    # Measured with scripts/probe_climb_osg.py on 00800's navmesh flight,
    # descending: with TRUE tread heights the carrot's goal sits 0.66-1.01 m
    # ahead and the agent arrives (-1.80 m, 28 steps); with the pasted RAMP
    # the goal shrinks 0.68 -> 0.48 -> 0.36 -> 0.10 m and the climb stalls at
    # -0.12 m. The mouth of that flight is a flat landing: the ramp says the
    # floor drops at once, the agent walks 0.5 m without descending, and the
    # nearest cell "0.35 m below" is now the one under its feet. A ramp cannot
    # know where the landings are; the carrot can refuse to turn toward its
    # own feet, which on a staircase is never the way.
    climb_carrot_min_ahead_m: float = 0.0
    # The carrot must also be AHEAD, not merely far enough away. `min_ahead`
    # prices distance; this prices direction, and on a pasted map they are not
    # the same thing. A stored flight carries a LINEAR RAMP for heights
    # (`map_store._ramp_stair_heights`), so an iso-height contour is a line
    # across the whole blob and "the nearest cell 0.35-1.0 m further along in
    # height" can sit BEHIND the agent.
    #
    # Measured with scripts/probe_climb_osg.py on 00821's descent, same flight,
    # same climb loop, only the height field differing:
    #   exact  (true tread heights)  ARRIVED  -3.32 m in  60 steps, 31 fwd / 28 turns
    #   ramp   (linear ramp)                  -0.93 m in 300 steps, 50 fwd / 249 turns
    #                                         -- 20 of 55 steps ran BACKWARDS
    #   pasted (the stored map)               +0.00 m, stalled at 52 steps
    # The climb loop is not the defect; what it is aimed at is.
    #
    # Keeps only candidates in the half-plane toward the far end of the flight,
    # and falls back to the unfiltered band if that would leave nothing -- a
    # direction rule may not make the climb impossible.
    climb_carrot_forward_only: bool = False
    # When every remaining tread is nearer than `climb_carrot_min_ahead_m`,
    # take the nearest one anyway instead of returning no carrot at all.
    # OFF by default = the shipped behaviour. The arm turns it on: measured on
    # 00873 ep50005, the ascent stalls at 1.80 m of a 3.20 m flight with 109
    # cells in band and no carrot picked, because the last-resort branch
    # applied the spacing rule as a hard filter with nothing behind it.
    climb_carrot_relax_min_ahead: bool = False
    # Choose the next waypoint by walking distance ALONG the flight instead of
    # by height band / euclidean nearest. OFF by default = shipped behaviour.
    # A switchback has no straight axis: on 00873, 28 of the flight's 57
    # ground-truth steps run backwards along its own foot->top chord, so both
    # the band rule and the "highest tread" fallback aim across the banister.
    climb_carrot_follow_path: bool = False
    # How far off the flight the agent may be and still follow its path.
    # Beyond this the older carrots take over and head for the mouth.
    climb_carrot_path_max_offset_m: float = 1.0
    # Hold the flight carrot until the agent REACHES it (this many metres) or
    # climbs past its height, instead of re-picking the nearest tread every
    # step. 0 keeps the shipped per-step pick.
    #
    # Why: `_flight_carrot` picks the nearest cell in a height band, so the
    # goal moves every step -- and `PointNavDriver` wipes its recurrent state
    # whenever the goal moves more than 0.1 m, which is then EVERY step. A
    # point-goal policy reset every step cannot build momentum: it turns
    # toward the new goal, takes one action, and starts again. Measured
    # (outputs/mf5_pass2_v16 ep1): 300 steps for 2.56 m of descent, with no
    # `climb_forced_forward` and no `climb_blocked_turn` in the whole climb --
    # the mover always had an action, so the agent was never pressed against
    # anything, it was turning. The same production climb on the same pasted
    # flight, started aligned with it by `scripts/probe_climb_osg.py`, covers
    # 2.74 m in 50 steps.
    climb_carrot_hold_m: float = 0.0
    # Do not turn on a staircase when the carrot is already within this many
    # degrees of dead ahead: one turn is `turn_deg` (30), so a correction
    # smaller than half of that OVERSHOOTS, and the next step corrects back.
    # Measured on the climb traces (outputs/mf5_pass2_v18): on the descent 25%
    # of turns immediately reversed the previous turn, on the ascent 43%, and
    # the agent covered 6.6 m of net displacement along a 28.1 m path (8.9 m
    # along 41.7 m on the ascent) -- three quarters of the motion undone. 0
    # keeps the mover's own turns.
    climb_turn_deadband_deg: float = 0.0
    # Once locked forward, the error has to exceed THIS to turn again -- the
    # hysteresis half of the lock, without which the agent sits on the
    # deadband edge and chatters. Defaults to one full turn step.
    climb_turn_release_deg: float = 0.0
    # The lock must YIELD when forward is not working. Measured
    # (outputs/mf5_pass2_v19): with the deadband alone, 328 of 332 turns were
    # suppressed and the agent issued 340 forward actions that produced 5.4 m
    # of path -- it was pressing into the banister, because the mover's turns
    # are not only alignment, they are how it gets around things, and the
    # action alone does not say which. So: if the last forward step did not
    # move the agent this far, the next turn goes through.
    climb_turn_stuck_eps_m: float = 0.05
    # And never suppress more than this many turns in a row, whatever the
    # geometry says. 0 for no cap.
    climb_turn_suppress_max: int = 0
    # Turn in place at the start of a climb until the first tread is within
    # this many degrees of dead ahead, before taking a single step. 0 skips it.
    #
    # Why: `scripts/probe_climb_osg.py` drives the SAME production climb up
    # the SAME pasted flight in 50 steps for 2.74 m, and the one thing it does
    # differently is start the agent aligned with the flight. A run arrives at
    # the mouth from a walk, facing wherever the approach left it, and then
    # asks a point-goal policy to rotate and translate at once on stairs.
    # Capped at a full revolution so it can never spin in place forever.
    climb_align_first_deg: float = 0.0
    # How many floor switches may FAIL from one storey before the agent stops
    # asking to leave it and searches instead. 0 keeps asking forever.
    #
    # Measured (outputs/mf5_pass2_v18, the cracker box): it climbed to the
    # correct storey at step 161, searched three containers, and from step 333
    # to 841 made SIX descent attempts, every one ending in
    # `portal_end_no_vertical_progress` -- 508 steps, half the episode, trying
    # to leave the storey the target was actually on. It selected a frontier
    # three times in 1000 steps. A storey the agent cannot leave is a storey
    # it should be searching.
    max_failed_switches_per_storey: int = 0
    # How long that ban lasts, in steps; 0 is the rest of the episode. A
    # permanent ban is too absolute: in outputs/mf5_pass2_v19 the cracker box
    # failed twice while the turn lock was breaking its climb, hit the limit,
    # and could then never reach the target storey at all (banned 16 times,
    # `goal_floor_reached` false). The ban should cost the agent the next
    # stretch of the episode, not the episode.
    switch_ban_steps: int = 0
    # End a climb by height only once the KNOWN gap to the target storey is
    # closed (minus this tolerance), rather than at `floor.new_level_m`.
    # 0.0 disables, which is the shipped behaviour.
    #
    # Measured on 00800 cross_anchor_01 (outputs/mf5_pass2_v11 ep2): the
    # storeys are 3.0 m apart and `new_level_m` is 1.8, so the first
    # successful climb in any run -- +1.83 m in 64 steps -- was declared a
    # storey at 1.83 m with the agent standing on the treads. The estimator
    # never committed the upper storey, the posterior picked a cabinet on the
    # LOWER one, and the agent walked back down. The prior map carries both
    # storey heights, and `_goal_floor_y_cache` is the one being climbed to.
    climb_to_target_storey_tol_m: float = 0.0
    # After this many failed attempts on ONE storey, that storey is disproved:
    # its target-labelled tracks stop vetoing a floor switch, and the nearest
    # other known storey is requested directly. 0 is the shipped behaviour.
    #
    # A failed attempt is the strongest signal the pipeline gets that THIS
    # storey is not it, and until now it lowered exactly one track. Measured
    # on 00800 cross_anchor_01 (outputs/mf5_pass2_v6): the upper storey held
    # eight "toy airplane" tracks -- a ceiling fixture at 2.6 m and its kin,
    # detector scores up to 0.82 -- with 0 presence events for the label in
    # 200 steps, because a false positive IS present and every look re-detects
    # it. Three attempts went to three of them; `floor_target_evidence` then
    # added _TARGET_PRESENT for the believed fakes and `may_switch` refused to
    # leave "a floor that has the thing we are looking for on it". The true
    # target was one storey down and never once in view.
    #
    # Every failed attempt on the storey counts, the stale anchor included: an
    # attempt at the prior's own pose that finds nothing is the single
    # strongest "it moved" reading there is.
    floor_disproved_after_failed_attempts: int = 0
    # On the robot only (floor.source=external): when the storey the agent is
    # on is finished -- no frontier left on it, `_after_steps` spent on it, or
    # disproved by failed attempts with no other storey known -- ask the
    # operator for the lowest storey key nobody has stood on, instead of
    # idling until max_steps. The posterior and the rule above can only name
    # storeys the map already holds, and the search pass may restore a
    # one-storey map (docs/THOR.md, the operator's protocol). Off is every
    # simulator arm; stretch3_map turns it on. `_after_steps` 0 is no budget;
    # it counts from the first exploration round on the storey (after the
    # remembered spot has been walked to and checked) and is checked every
    # step, so it interrupts a frontier drive.
    request_new_storey_when_exhausted: bool = False
    request_new_storey_after_steps: int = 0
    # While a floor switch is being walked (`floors.pursuing`), only a
    # candidate seen LIVE within `range_m` with detector score >= `min_score`
    # may pre-empt it. Off (False) is the shipped behaviour: any candidate
    # pre-empts. Measured on 00800 ep1, every run: the down-flight is chosen
    # at step 172 and a same-floor detection commits at step 181, nine steps
    # later, abandoning it; `candidates.check` runs in GOTO_FRONTIER and
    # nothing protected a switch that was decided but not yet walked.
    protect_floor_switch: bool = False
    protect_floor_switch_range_m: float = 1.5
    protect_floor_switch_min_score: float = 0.6
    climb_cell_carrot: bool = False
    # After this many consecutive mover STOPs with no height gained, turn to
    # re-aim instead of pressing into the wall again. 0 disables.
    climb_blocked_turn_after: int = 0
    climb_carrot: bool = False
    climb_carrot_m: float = 0.8
    down_look_every: int = 0
    # Where the look-down is allowed to fire, in metres from a cell the stair
    # map calls stairs. The look-down exists to DISCOVER a staircase down, and
    # it is purely periodic: every `down_look_every` steps the agent pitches
    # down, accumulates stair evidence from whatever is in front of it, and
    # pitches back -- two steps, wherever it happens to be standing. When a
    # prior ASCENT map has already been pasted, where the stairs are is known,
    # so looking for them in the middle of a bedroom is two wasted steps and a
    # chance to plant stair evidence on furniture. Measured (v13/v14, 00800):
    # 6-16 look-downs an episode, none of them at the staircase. 0 disables
    # the gate, which is the shipped behaviour; with no stair map at all the
    # gate never applies, because then there IS something to discover.
    down_look_near_stairs_m: float = 0.0

