"""`floor` group: multi-floor support (docs/MULTI_FLOOR.md).

Not part of the dynamic-scene line: every default here reproduces single-floor
behaviour, so `floor.enabled=false` -- which every YCB run uses -- is a no-op
path. See `osg/agent/floor_policy.py`.
"""
from __future__ import annotations

from dataclasses import dataclass


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


