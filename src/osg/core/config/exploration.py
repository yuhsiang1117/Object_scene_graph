"""`exploration` group: frontier selection and the search posterior.

Both halves are scored under ONE index, b*d/c, so exploring and re-searching are
not separate subsystems. The `search_*` block is the dynamic-scene half: where an
object could have been moved to, and what a look at a surface is worth. See
`osg/exploration/search_belief.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExplorationConfig:
    frontier_text_scorer: str = "disabled"  # disabled | llm_text
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
    area_thresh_m2: float = 1.5
    selector: str = "utility"  # utility | ascent
    nearby_distance_m: float = 3.0
    frontier_commit: bool = False
    ranker: str = "none"  # none | ascent
    ranker_topk: int = 3
    ranker_every_steps: int = 20
    floor_llm: bool = False
    floor_ask_every: int = 60
    floor_min_steps: int = 100
    floor_llm_boost: float = 5.0
    subgraph_radius_m: float = 3.0
    select_every: int = 5
    reselect_every: int = 0
    frontier_desc: str = "graph"  # graph | frame_objects | frame
    frontier_desc_match_m: float = 1.0
    max_frontiers_per_call: int = 4
    unscored_prior: float = 0.3
    min_path_cost_m: float = 0.5
    # Search posterior (docs/history/DYNAMIC_SCENES.md, C3). Off by default: it changes
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
    # (scripts/legacy/rank_search_surfaces.py), share of cases where the true surface
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
    # Stop anchoring the search on the last known pose once the map has stopped
    # believing the object is there. 0.0 keeps the anchor forever (shipped).
    #
    # The two halves of the benchmark want opposite models and no mixture serves
    # both -- swept offline, flat and two-scale alike, every setting lands on one
    # frontier. Inspections a greedy search needs to reach the true destination:
    #
    #                       reaches it   median   within 10
    #     in_anchor   prox    23/57         2        23
    #     in_anchor   flat    15/57        17         5
    #     cross       prox    12/57        22         4
    #     cross       flat    20/57        16         9
    #
    # The agent does not have to guess which half it is in: its own presence
    # belief answers -- but the right event is an ARRIVAL, not a belief level.
    # A first attempt keyed this on `presence.p < min_presence` and fired in 56%
    # of in_anchor episodes against the 30% predicted, because presence also
    # decays from ordinary missed expectations while the agent walks past. It
    # dropped the anchor on the half that needs it and cost three episodes on
    # the first scene. `ObjectTrack.absence_arrivals` counts only "I went to
    # look and it was gone".
    search_drop_proximity_after_absence: bool = False
    # ---------------------------------------------------------------- fusion
    # Order the two systems: test the stale anchor BEFORE choosing a storey.
    #
    # `_select_surface` aggregates container mass per floor and asks FloorPolicy
    # for the argmax. With a flat proximity prior that mass is just "which floor
    # has more mapped surfaces", so on a two-storey prior map it fires on the
    # FIRST selection round -- measured on outputs/osg_authored_15, the first
    # cross-floor request came at step 13 in 17 of 30 episodes, including 10 of
    # the 19 whose object never left the start floor. The agent leaves before it
    # has been anywhere, and the presence filter -- the mechanism that knows the
    # object moved -- never gets to speak: 4 absence checks in 30 episodes.
    #
    # With this on, a cross-floor request is held while a target-labelled track
    # on THIS floor is still believed, i.e. no absence arrival yet and presence
    # still above the candidate gate. The held round falls through to same-floor
    # selection, so the agent searches where the map says the object is, and the
    # storey question is asked once that belief has actually been tested.
    search_floor_requires_anchor_test: bool = False
    # Weight `graph.priors.floor_target_evidence` by presence instead of
    # counting labels. A mapped instance of the target category scores +10 and
    # closes the switch gate for good ("never leave a floor that has the thing
    # we are looking for on it") -- but on a stale prior map that instance is
    # exactly the object that has since been moved. Presence already knows: the
    # track sits below `min_presence` after an absence arrival. Off by default
    # because it changes when a single-storey run may leave a floor at all.
    floor_evidence_by_presence: bool = False
    # A storey request that cannot be executed must not veto searching this one.
    #
    # `_select_surface` returns None when it wants another floor, and the round
    # then falls through to a frontier. If `try_switch` finds no portal -- 405
    # requests produced 12 directed attempts on outputs/osg_authored_15, so 97%
    # of them found nothing to drive to -- the container posterior is switched
    # off for as long as the request stands, on the floor the agent is on and
    # among the surfaces it can actually reach. With this on, a failed switch
    # falls back to same-floor surface selection instead of to frontiers alone.
    search_surface_when_floor_unreachable: bool = False
    # How a storey's candidate mass is scored: "sum" (every mapped surface adds
    # to it, the shipped behaviour) or "mean" (how good this floor's surfaces
    # are, independent of how many there are). See the measurement in
    # `_select_surface`: on 00808 the two floors' means agree to within half a
    # percent while their sums differ 2.4x, purely because one floor holds more
    # furniture.
    floor_mass_rule: str = "sum"
    # How much better another storey must score before the agent pays for the
    # stairs. 0.0 disables the test, which is the shipped behaviour; 1.15 means
    # "15% better or stay". With `floor_mass_rule: mean` this is what lets the
    # posterior answer "no opinion" instead of breaking a tie by floor area.
    floor_mass_margin: float = 0.0
    # Steps after ARRIVING on a storey during which an empty container
    # posterior there means "unmapped", not "the target is not here". 0 is the
    # shipped rule: with nothing at all here, anywhere else is better at once
    # (tests/unit/test_floor_anchor_order.py). Measured on 00800
    # (outputs/mf5_pass2_v12 ep1): three steps after climbing down, the
    # storey had no containers yet -- they are rebuilt from tracks at
    # keyframes -- so `floor_mass={'0': 11.8}` named only the storey just
    # left, and the agent climbed straight back up. The margin rule cannot
    # see this: 0 * margin is 0. Within this window the round explores here
    # instead; containers appear as the storey is mapped.
    empty_storey_settle_steps: int = 0
    # A belief over storeys decides whether, and where, to change storey
    # (`exploration/storey_belief.py`): target-labelled tracks by height and
    # presence, discounted by this episode's coverage of each storey. On, it
    # replaces the container posterior's storey vote, and the geometric
    # "nothing near left" gate may no longer switch undirected while a second
    # storey is known. Measured on Benchmark v2 seed 42 without it: 9 of 28
    # episodes that had to go down only ever pursued flights going UP, toward a
    # storey invented 1.8 m above (00821: 1.93 m from 0.13).
    storey_belief: bool = False
    # pi_k = w * anchor_k + (1 - w) / K: how much the target-labelled tracks
    # outweigh "it could be on any storey".
    storey_anchor_weight: float = 0.8
    # How likely having been within `storey_cover_radius_m` of a stretch of
    # floor means having seen the object there (the discount 1 - d * cover).
    storey_detect_prob: float = 0.85
    # Ask for another storey once its belief is this many times this one's.
    storey_switch_ratio: float = 1.3
    storey_cover_radius_m: float = 2.0
    # Each identity rejection (STOP gate, unreachable verdict) multiplies a
    # track's weight in the storey belief by this; a failed attempt or the
    # candidate gate's own limit removes it (`storey_belief.target_tracks`).
    storey_rejection_discount: float = 0.3
    storey_block_m: float = 0.5
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
    # The floor glancing alone may not push a surface past. 0.0 is the shipped
    # behaviour: unbounded compounding, once per keyframe, for every container
    # in view. `_scan_at_viewpoint` refuses exactly this multiplication and says
    # why -- "twelve looks at the same object from the same pose are not twelve
    # independent observations ... measured: doing it dropped SR from 0.429 to
    # 0.286" -- and the glance path was never given the same treatment.
    # 0.2 is not a taste: it is what a real arrival and inspection is worth
    # (1 - search_detect_prob), so it makes the docstring's claim true, that a
    # passing look is weaker evidence than standing there.
    search_glance_floor: float = 0.0
    search_glance_range_m: float = 4.0
    # Finishing the room you are in beats crossing the house and coming back.
    search_same_room_bonus: float = 4.0
    # ...but that bonus is a PRIOR ("the object is in this room") that never got
    # a likelihood update, and an un-updated prior in a container-dense room is
    # an absorbing state. Scene 00848 is a kitchen holding 35 cabinets, 20
    # shelves and 13 refrigerators; every one keeps its x4 however many come up
    # empty, and since utility divides by path cost the nearest of the 68 always
    # outbids a frontier discounted to 0.3. Measured over the six 00848 pitcher
    # episodes under condition N: 11 surfaces inspected, ZERO frontier
    # selections in 286 steps, the target never once in view, SR 0.15.
    #
    # Each fruitless ARRIVAL in a room multiplies its bonus by (1 - rate) --
    # glances and give-ups on the way do not count, the same distinction
    # InspectionLog already makes between a look and a visit. 0.0 disables it
    # and reproduces every earlier condition exactly.
    search_room_saturation: float = 0.0
    # Arrivals a room gets for free before any decay starts.
    #
    # Without this, condition V decayed the bonus from the FIRST fruitless
    # arrival and cost 6 episodes over 90 matched (in_anchor 0.822 -> 0.733).
    # The reason is in the distribution: over condition N's 96 episodes NO
    # success ever needed more than 7 surface inspections (median 0, p90 5),
    # while the stuck 00848 failures need 11-13. Decaying from the first
    # arrival therefore weakens the room prior precisely inside the window
    # where successes happen, and the agent -- pushed out of a room it had not
    # finished -- enters the next one with a fresh x4 and scatters, doing MORE
    # inspections than before (10 against 2 on the episodes V lost).
    #
    # So the room keeps its full bonus until it has already given more looks
    # than any success has ever needed. Beyond that the room is, empirically,
    # not the room.
    search_room_saturation_free: int = 0
    # How far the bonus may fall. 1.0 means saturation can cancel the bonus but
    # never invert it into a penalty: a room that has disappointed becomes
    # ordinary, not worse than one never visited. Below 1.0 actively pushes the
    # agent out of a room it has been failing in.
    search_room_saturation_floor: float = 1.0
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
    # The hand-written table (`graph.priors.CONTAINER_AFFINITY`) is
    # authoritative where it has the class, and the LLM answers only for the
    # rest -- on the YCB benchmarks the table covers DualMap's targets (bowl,
    # cracker box, tin can, banana, ...) and the LLM the multi-floor ones. The
    # review asked for the prior to come from the model for every episode:
    # false skips the table, so with `affinity_llm` the LLM ranks every class
    # (grounded on this map's containers under `affinity_grounded`), and
    # without it there is no affinity at all -- the flat-prior ablation.
    affinity_table: bool = True
    # Rank affinity over the container categories the MAP actually contains,
    # rather than over all of CONTAINER_CATEGORIES.
    #
    # A prior whose top choices do not exist in this house is not a prior. The
    # LLM ranks "blue plastic pitcher" as counter > table > shelf > ..., and
    # scene 00848 contains ZERO counters and ZERO tables out of 343 tracks --
    # its counter runs are labelled `cabinet`, which is unlisted and so takes
    # UNLISTED_AFFINITY, below shelf. The 35 cabinets that ARE the counters
    # therefore rank beneath everything, and what reaches the search posterior
    # is a near-flat prior over 68 containers.
    #
    # Grounding is surgical here by construction: 00829 and 00880 contain all
    # 14 categories, so their grounded option set IS the full set and their
    # ranking cannot change. Only 00848 is affected.
    affinity_grounded: bool = False
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
    # `maybe_give_up`: a frontier pursuit that has not moved the agent
    # `giveup_min_progress_m` in `giveup_after_steps` steps is abandoned and
    # blocked (an obstacle the map cannot see, glass, a sim collision). The
    # values are the constants the rule shipped with. On a robot a step is
    # about a second and Nav2 spends several planning and 10-20 s in a
    # recovery at a doorway, so the Stretch presets widen the window: measured
    # (outputs/20260922_234724), goals Nav2 had accepted were cancelled after
    # 4-18 s and the doorway frontier blocked for 100 steps.
    giveup_after_steps: int = 15
    giveup_min_progress_m: float = 0.2
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
    stair_prior: float = 0.6
    stair_explored_rule: str = "no_frontiers"  # no_frontiers | steps
    floor_exp_steps: int = 100
    stair_min_hits: int = 1
    stair_min_cells: int = 25
    value_map: bool = False
    # ASCENT constructs its ValueMap with `use_max_confidence=False`
    # (`map_controller.py:79-83`): a confidence-weighted running average. The
    # vendored default is True (single most-confident view wins). Read by
    # ascentnav only.
    value_use_max_confidence: bool = False
    # Fail loud on the value model, see `osg/perception/ascent_models.py`.
    value_strict: bool = False
    # RAM++ per-step scene tags for the LLM prompt (`map_controller.py:800-813`).
    ram_url: str = "http://localhost:13185/ram"
    ram_tags: bool = False
    # ASCENT's multi-floor prompt is dead in the reference run: `_explore`
    # never passes `floor_num`, so the `> 1` branch never fires. Kept behind a
    # flag so the reference-effective behaviour is the default.
    llm_multi_floor: bool = False
    # clip | constant | blip2itm. `blip2itm` is ASCENT's own value model,
    # reached over HTTP because lavis cannot be installed in this environment;
    # see `perception/image_text.py::Blip2ItmScorer` and
    # `scripts/serve_perception.sh`, which serves it on this port.
    value_model: str = "clip"
    value_blip2_url: str = "http://localhost:13182/blip2itm"
    value_blip2_timeout_s: float = 10.0
    value_clip_name: str = "ViT-B/32"
    value_clip_root: str = "data/clip"
    value_stride: int = 1
    value_weight: float = 4.0
    value_argmax: bool = False
    value_radius_m: float = 0.5
    value_prompt: str = "Seems like there is a {target} ahead."
    knowledge_prior: bool = False
    knowledge_radius_m: float = 3.0
    knowledge_weight: float = 1.0
