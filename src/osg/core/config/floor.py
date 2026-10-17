"""`floor` group: multi-floor support (docs/history/MULTI_FLOOR.md).

Not part of the dynamic-scene line: every default here reproduces single-floor
behaviour, so `floor.enabled=false` -- which every YCB run uses -- is a no-op
path. See `osg/agent/floor_policy.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class FloorConfig:
    """Multi-floor support (docs/history/MULTI_FLOOR.md). Every default reproduces the
    current single-floor behaviour, so `floor.enabled=false` is a no-op path."""

    enabled: bool = False
    # Where the current storey comes from.
    #
    #   estimator  the agent's own height trace (docs/history/MULTI_FLOOR.md). The only
    #              source in habitat, and the default, so nothing below moves.
    #   external   an authority outside the agent says which storey it is on:
    #              the operator's `/osg/floor` topic on the robot, or
    #              `external_schedule` in a simulated test. The estimator is
    #              not consulted at all -- its primary signal is
    #              `camera_position[1] - camera_height`, and a real robot on a
    #              2D Nav2 map has no trustworthy vertical at all. A phantom
    #              floor is strictly worse than no floor (docs/history/MULTI_FLOOR.md
    #              "the landing problem"), which is the argument for taking an
    #              authoritative id when one exists.
    source: str = "estimator"  # estimator | external
    # "step:key" pairs applying an external switch at a given step, so the
    # external path is exercisable in habitat with no ROS in the room.
    external_schedule: List[str] = field(default_factory=list)
    # How far apart external storeys are placed in the world's vertical.
    #
    # The robot's `map` frame is 2D: every floor is reported at the same
    # height, so two storeys would occupy one band of the costmap, one set of
    # `floor_of_height` answers, and one position in the LLM prompt's
    # height-ordered floor list. `sim/ros2_env.py` therefore lifts the pose by
    # `key * virtual_storey_m` -- the switch IS the height sensor. Roughly a
    # storey (HM3D measures 2.5-3.4 m) so the geometry the rest of the
    # pipeline sees is the geometry it was calibrated on. 0 disables the lift,
    # which collapses the storeys back into one band.
    virtual_storey_m: float = 3.0
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
    # One Costmap2D + room segmenter + room labels per storey, instead of one
    # shared map. Implies enabled=true and estimate_only=false. Without it an
    # upper floor is never mapped at all (its points fall outside the band
    # around the latched floor_y), so it yields no frontiers and the agent has
    # nothing to explore there. See docs/history/MULTI_FLOOR.md.
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
    # most real staircases. See docs/history/MULTI_FLOOR.md.
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
    # not work -- see docs/history/MULTI_FLOOR.md).
    cross_floor: bool = False
    # An UNDIRECTED storey switch (the geometric "nothing near left here" gate,
    # not a directed request for a known target floor) is attempted only when
    # the estimator already knows >= 2 levels. Without it, a single-floor scene
    # still hunts portals once its frontiers get far, and `find_portals` reads a
    # tall bookcase as a portal: measured on the DualMap released benchmark,
    # 1-2 spurious `floor_switch_attempts` per episode that walked the agent off
    # to a false portal and perturbed a run the floor group should not touch at
    # all. A schema-v2 prior map seeds every floor into the estimator on load
    # (graph/map_store.apply_map), so a genuine multi-floor scored run has >= 2
    # levels from step 0 and its cross-floor switches are unaffected; only a
    # truly single-floor scene, which has one level and nowhere to go, is gated.
    # Also gates the periodic stair-hunting look-down (agent/stair_sense.py),
    # which likewise has no job on a one-level scene.
    # Default off so no existing preset changes.
    switch_requires_second_level: bool = False
    # ASCENT's gate: only reason about storeys when the best frontier left on
    # this floor is further away than this.
    near_frontier_m: float = 4.0
    # Here we set a minimum interval between floor switches, to stop the agent
    # oscillating between floors. NOT inherited from ASCENT: its reference
    # implementation gates switching on events, not on time -- when a floor
    # runs out of frontiers it is marked explored and the agent leaves
    # (`ascent_policy.py:656-670`), with no interval guard anywhere in
    # `relative_work/ascent/ascent/`. The "T/10" this comment used to cite is
    # a related-work note about the ASCENT PAPER (docs/history/MULTI_FLOOR.md:152),
    # not about the code we transcribe.
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
    # Drive to the staircase the PRIOR MAP already walked when no portal is
    # visible. `save_map` records every committed transition in `connectivity`
    # and `apply_map` restores it into `FloorStack.stair_edges`; until now
    # nothing read it back, so an agent that knew exactly where the stairs were
    # still had to rediscover them by chance. Measured on
    # outputs/osg_authored_15: 4 of 11 cross-floor episodes never saw a portal
    # and never attempted a switch, and 405 storey requests produced 12
    # attempts. Off by default -- it adds a goal the agent would not otherwise
    # have had.
    use_prior_stairs: bool = False
    # Use the remembered staircase AHEAD of a detected portal, not only when no
    # portal is visible. `find_portals` finds a patch of another storey visible
    # from here, which over a balcony rail is a sightline and not a way up:
    # 00821's cracker box produced 55 such goals, arrived at one, and rose
    # 0.17 m in 500 steps. `connectivity` records where pass 1 actually changed
    # floor, which is a staircase by construction.
    prefer_prior_stairs: bool = False
    # Which point of a remembered traversal is this storey's stair mouth.
    # "edge" (as measured): `_remembered_stair`'s reading -- `entry_xy` of an
    # edge leaving this storey, `exit_xy` of one arriving. But `set_current`
    # runs when the estimator COMMITS the new storey, so both points of every
    # saved edge lie on the storey ARRIVED at (outputs/maps_v2_osg: entry ==
    # exit on 13 of 23 edges, the rest a later arrival beside the first).
    # "edge" sent a descent from 00873's upper storey to (-3.27, -4.96), a
    # point on the lower one, ranked ahead of the arrival on this storey 0.3 m
    # from the true lip. "arrival": only edges that ARRIVED on this storey,
    # from a storey on the target's side of it (a dropped landing counts by
    # its height). On seed 42 (mfv2_osg_climb) 13 of the 18 cross-storey
    # episodes that never reached the object's storey had one, 0.3-0.8 m from
    # the true mouth; the agent chased sightline portals instead.
    prior_stair_mouth: str = "edge"
    # Skip a remembered mouth once this many pursuits failed within
    # `portal_failure_radius_m` of it (0: never, as measured -- the branch had
    # no failure check). Not 1: a climb that stalls at the right staircase
    # (00873, seed 42: 40 steps on the flat floor beside its rail) is an
    # execution failure, not evidence against the place.
    prior_stair_max_failures: int = 0
    # With `prefer_prior_stairs`, try the remembered staircase BEFORE the flights
    # read off the height layer (`climb_targets: flights_first`), not only
    # before portals. 00800 (seed 42, static 50001, mfv2_osg_mouth): the flight's
    # foot was taken 3 m from the true one, beside the treads, and both climbs
    # from it stalled at 0.0 m; pass 1 had arrived on this storey 0.8-1.0 m from it.
    prior_stairs_before_flights: bool = False
    # Let a remembered staircase satisfy the switch gate when the geometric rule
    # will not. `may_switch`'s last clause is "nothing near is left on this
    # floor", which a large storey never satisfies, so `try_switch` returns
    # before `find_portals` is called and the remembered staircase is never
    # reached: 3 of 7 cross-floor episodes never made one attempt. Every timing
    # guard still applies.
    prior_stairs_override_gate: bool = False
    # Remember where a portal pursuit failed, so the same patch is not proposed
    # again. `find_portals` is recomputed from scratch every selection round and
    # keeps no state, so a patch that did not lead anywhere is re-chosen
    # immediately: 00821's cracker box made 55 switch attempts, the last 50 at
    # one 28-cell patch, for 0.17 m of ascent in 500 steps.
    portal_failure_memory: bool = False
    # What a floor pursuit aims at. "portals": a patch of another storey visible
    # from here (the shipped behaviour). "stairs_first": a `stairs` track the
    # detector has actually seen on this storey, nearest first, and only then a
    # portal. Measured on 00821, the portal the agent chased 55 times sat about
    # 10 m from the flight the navmesh uses.
    # "flights_first": a run of intermediate-height cells in the height layer
    # whose heights span at least `flight_min_span_m` -- a staircase, read
    # geometrically; its lowest tread is the foot. Tried before stairs tracks
    # and portals. See mapping.stairs.find_flights.
    climb_targets: str = "portals"
    flight_min_span_m: float = 1.0
    flight_min_cells: int = 150
    # Where a DESCENDING flight's pursuit goes (`mapping.stairs.mouth_xy`).
    # "lowest_tread" is what every arm through 2026-10-17 ran: `mouth_xy` read
    # `Flight.foot_xy` as every flight's lowest tread, but `find_flights` stores
    # a descent's HIGHEST tread there, so a descent was pursued to the lowest
    # tread seen -- the far end, under this storey. Benchmark v2 seed 42: no
    # episode on 00862 or 00873 that had to go down ever did. "mouth" is the
    # highest tread seen; "extrapolated" walks up the flight's own slope from
    # it to where the slope meets this floor (`descent_mouth_xy`), because
    # from a distance the top treads are hidden behind the lip.
    descent_goal: str = "lowest_tread"
    # Rank flights by whether the STAIR MAP says they are stairs, and drop
    # flights that lead where no storey is known to be. 0/False is the shipped
    # behaviour: nearest foot wins, whatever it is.
    #
    # `find_flights` reads the height layer for cells sitting between two
    # storeys, and furniture qualifies -- a sofa back, a counter, a stair-free
    # mezzanine lip. Measured on 00800 cross_anchor_01 (outputs/mf5_pass2_v3),
    # the real staircase is at world (-8.1, -13.0) and the flights actually
    # pursued were:
    #
    #   ep1  step 172  (-0.67, -4.73)   210 cells   ~10 m from the stairs
    #   ep2  step  68  (-3.32, -0.97)  1166 cells
    #   ep2  step 192  (-9.42, -3.32)   652 cells
    #   ep2  step 245  (-11.02, -3.82)  778 cells
    #   ep2  step 296  (-8.32, -12.67)  596 cells   <- the staircase, at last
    #
    # Selection is `min(usable, key=distance to foot)`, so furniture near the
    # agent beats a staircase across the house every time; ep2 reached the real
    # one on its fourth try at step 296 of 430. Worse, ep1 was on the TOP
    # storey pursuing a flight_up -- there is nothing above the top storey, so
    # that flight cannot be a staircase by construction.
    #
    # The stair mask carried over from ASCENT is exactly the evidence that
    # settles this, and nothing was reading it. Corroborated flights sort
    # first; the direction filter needs two known storeys before it will drop
    # anything, so a single-storey costmap at episode start still behaves as
    # before.
    flights_prefer_stair_mask: bool = False
    # Do not re-issue a floor pursuit that is still in flight. A directed
    # request recurs every selection round and each one restarted the pursuit,
    # resetting its progress clock, so a pursuit never ended, no failure was
    # ever remembered and the same target was chosen 55-82 times.
    hold_pursuit: bool = False
    # The geometric switch rule must not fire on a floor the agent has only
    # just reached: with almost no map, "no near frontier" is trivially true and
    # the agent leaves again before looking. Measured on 00808's yellow bottle:
    # arrived downstairs at step 314, left at 347, back upstairs by 418. Off by
    # default because it changes when an agent may leave a storey.
    dwell_on_arrival: bool = False
    # Refuse to create a NEW level while the agent is standing on a staircase.
    # A half-landing is off every known level and roomy enough to walk 2.5 m
    # across, so the horizontal-run route makes it a storey of its own and the
    # climb ends there: measured on 00808, a descent of 2.3 m out of 3.2
    # committed the landing at y=0.946 as a floor. Arriving on a floor the stack
    # already knows is unaffected.
    no_level_on_flight: bool = False
    # A continuation flight, found from a landing, is half a storey rather than
    # a whole one; `find_flights` is asked for a smaller span when relinking.
    flight_relink_span_m: float = 0.5
    # `find_flights` keeps cells within `new_level_m` of the floor, and that is
    # a constant. On 00800 the storeys are 3.0 m apart and the ramp pasted
    # from the ASCENT map spans all of it, so the flight the carrot follows
    # stopped 1.6 m up (measured, outputs/mf5_pass2_v13: the ascent stalled at
    # +1.97 m and the descent at -1.51 m, both at the end of the truncated
    # flight, and the depth-ray fallback then drove the agent into the
    # balustrade for the rest of the 200-step budget). With this on, the span
    # asked of `find_flights` is the gap to the nearest OTHER known level when
    # there is one -- the prior map seeds every storey's height at episode
    # start -- and `new_level_m` otherwise.
    flight_span_from_levels: bool = False
    # How close counts as the same failed place.
    portal_failure_radius_m: float = 1.5
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

