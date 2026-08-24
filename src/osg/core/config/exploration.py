"""`exploration` group: frontier selection and the search posterior.

Both halves are scored under ONE index, b*d/c, so exploring and re-searching are
not separate subsystems. The `search_*` block is the dynamic-scene half: where an
object could have been moved to, and what a look at a surface is worth. See
`osg/exploration/search_belief.py`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExplorationConfig:
    scorer: str = "vlm"  # vlm | llm_text | nearest | random
    top_n_frontiers: int = 5
    # HybridVoronoiPlanner navigates the medial axis and stops at the graph node
    # nearest the goal, within this radius -- it stops NEAR a goal, not on it.
    # Read by NavAgent both to build the planner and to derive
    # `_frontier_reach_m`; those two must agree, or an ordinary arrival at a
    # frontier is misread as a degenerate stub and the frontier is blocked for
    # having been reached.
    voronoi_goal_near_m: float = 0.7
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
    #
    # 1.0 m, not the 4.0 m this started at, because the benchmark's own
    # displacements say so: in_anchor relocations move a median 0.72 m and
    # cross_anchor ones 6.06 m, which is a short mode plus a long tail rather
    # than one exponential with a 4 m scale. Scored offline against the true
    # destination over 114 (scene, layout, target) combinations
    # (scripts/rank_search_surfaces.py), share of cases where the true surface
    # lands in the top 5 -- what one episode can afford to inspect:
    #
    #                                    overall   in_anchor  cross_anchor
    #   proximity dropped after absence   12/114      5/57        7/57
    #   L=4.0 with a 0.2 floor            20/114     19/57        1/57
    #   L=1.0, no floor                   36/114     29/57        7/57
    search_proximity_len_m: float = 1.0
    # The floor used to be 0.2, and it was doing damage. `max(exp(-d/L), floor)`
    # clips every candidate past ~6 m to the SAME value, so all of them tie and
    # their order falls to whatever `sorted` does with equal keys -- track id.
    # For a cross-anchor move, where the destination is 6 m away by
    # construction, that discards the only signal left. Without the floor the
    # far candidates stay ordered by distance and cross_anchor recovers from
    # 1/57 to 7/57. Keep it at 0.0 unless something needs a genuine mixture.
    search_proximity_floor: float = 0.0
    # Belief carried by the single most plausible mapped surface. The candidate
    # priors are affinity x proximity normalised so the best of them equals this,
    # which separates the ORDERING (what the proximity model is for) from the
    # SCALE (what `search_frontier_weight` prices unexplored space against).
    # 0.5 because that is where the previously tuned model sat -- median top
    # prior 0.479 over 19 (scene, target) pairs -- so sharpening proximity does
    # not silently re-tune the search-versus-explore trade at the same time.
    #
    # Raised from 0.5 to 1.0 after condition E. Matching the previously tuned
    # model's TOP prior (median 0.479) turned out to under-fund the search,
    # because the old model's tail was held up by its 0.2 floor and the new
    # one's is not: measured over 96 episodes, E ran the search in 38 episodes
    # for 1.24 inspections each against C0's 53 and 3.34. The ranking is the
    # part that improved -- E arrived at the surface the object was moved to
    # six times against C0's zero -- so the search deserves to outbid
    # unexplored space more often, not less.
    search_surface_mass: float = 1.0
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
    # Turns spent looking AT a surface on arrival, before its belief is scored.
    # `_mark_surface_searched` multiplies belief by (1 - search_detect_prob) on
    # a single frame taken at whatever heading the follower stopped on. In
    # condition D the search reached the true surface five times and converted
    # one. A few turns are cheap against the ~50 steps an inspection costs.
    search_face_turns: int = 8
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


