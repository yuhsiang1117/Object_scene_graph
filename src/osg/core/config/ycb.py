"""`ycb` group: the authored dynamic-scene benchmark.

Runtime discovery of the collector's layouts, deterministic episode generation,
and the two knobs the dynamic protocol turns on: `relocate_at_step` for a move
the agent can witness, and `map_out`/`map_in` for the two-pass protocol where
pass 2 navigates from a map that pass 1 built and the world has since
invalidated. The staleness IS the experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..paths import collector_data_root, hm3d_scene_root, ycb_authoring_root


# Handle -> the class name the detector is asked for. Not always the object's
# common name: YOLOE's text head scores "pitcher" at 0.00 on this asset at every
# resolution and "blue plastic pitcher" at 0.71, so the descriptive phrase is the
# label (scripts/probe_ycb_detection.py, mode=labels).
YCB_TARGET_LABELS: Dict[str, str] = {
    "002_master_chef_can": "coffee can",
    "003_cracker_box": "cracker box",
    "005_tomato_soup_can": "tin can",
    # "mustard bottle" scored 0.31-0.39 on the asset at 0.6-2 m and 0/5 in situ
    # on the 15-scene runs; "yellow bottle" scores 0.50-0.93 on the same frames
    # (scripts/render_asset_views.py, render.names probe, 2026-09-10).
    "006_mustard_bottle": "yellow bottle",
    "011_banana": "banana",
    "019_pitcher_base": "blue plastic pitcher",
    "021_bleach_cleanser": "bleach bottle",
    "024_bowl": "bowl",
    "025_mug": "mug",
    "029_plate": "red plate",
    "037_scissors": "scissors",
    "053_mini_soccer_ball": "soccer ball",
    "077_rubiks_cube": "rubiks cube",
    # The 15-scene authoring root places this in 14 of its 15 scenes; the name
    # is the detector query, and this is the plain one.
    "072-a_toy_airplane": "toy airplane",
}



@dataclass
class YCBAuthoredConfig:
    """Runtime discovery and deterministic episode generation for authored YCB layouts."""

    data_root: str = field(default_factory=lambda: str(collector_data_root()))
    layout_root: str = field(default_factory=lambda: str(ycb_authoring_root()))
    # Full canonical HM3D v0.2 scene tree.  Authored JSON may still contain
    # the collector's historical ``scene_datasets/hm3d`` path; the loader
    # rebases that spelling here rather than following the mutable symlink.
    hm3d_root: str = field(default_factory=lambda: str(hm3d_scene_root()))
    scenes: List[str] = field(default_factory=lambda: ["*"])
    layout_types: List[str] = field(default_factory=lambda: ["static"])
    layout_indices: List[int] = field(default_factory=lambda: [1, 2, 3])
    # Restrict episodes to these targets (YCB handle or label; empty = all).
    # Several assets in the collector's dataset render in a way the open-vocab
    # detector cannot recognise at any authored viewpoint, so their episodes
    # measure asset coverage rather than dynamic-scene handling. The layouts
    # themselves are DualMap's original data and are never edited.
    targets: List[str] = field(default_factory=list)
    # Combined dynamic/multi-floor benchmark selectors.  Disabled by default so
    # every existing authored manifest and experiment keeps its source behavior.
    cross_floor_relocations_only: bool = False
    relocation_directions: List[str] = field(
        default_factory=lambda: ["upward", "downward"]
    )
    start_on_prior_floor: bool = False
    # Start on the storey the object is ACTUALLY on, instead of the one the
    # prior claims. This is the control that separates the two halves of a
    # cross-floor episode: with it the agent begins on the goal storey and the
    # staircase is removed from the problem, so what remains is the search and
    # the terminal approach. Measured against the normal start (on the prior
    # storey) it says whether a scene's failures are stairs or approach.
    start_on_target_floor: bool = False
    # Draw each start's STOREY uniformly from the scene's navmesh storeys, then
    # the point on it (`_starts_on_uniform_storeys`). With neither flag above,
    # a start is an area-uniform navigable point, which lets the bigger storey
    # decide how often the stairs are needed and can land on a stair tread.
    # Benchmark v2 (configs/experiment/mfv2_dataset.yaml) uses this so that the
    # agent must work out whether to change storey at all: about half its
    # episodes start on the object's storey.
    start_storey_uniform: bool = False
    relocation_floor_tolerance_m: float = 0.5
    # Explicit scene lists normally fail on any absent requested slot. Combined
    # campaigns span heterogeneous authoring coverage and opt into recording
    # those gaps while keeping every available layout.
    skip_incomplete_layouts: bool = False
    starts_per_target: int = 1
    seed: int = 42
    manifest_cache_dir: str = "outputs/ycb_manifests"
    # Mid-episode relocation (docs/history/DYNAMIC_SCENES.md, Phase 2). -1 disables it
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
    # Two-pass benchmark (docs/history/DYNAMIC_SCENES.md, Phase 2). Pass 1 explores the
    # STATIC layout and writes one snapshot per scene to `map_out`; pass 2 runs
    # the moved layout and starts from `map_in`, so the map the agent navigates
    # with is genuinely stale. The staleness IS the experiment -- an agent that
    # rebuilds from scratch is never wrong about anything and measures nothing.
    map_out: str = ""
    map_in: str = ""
    # The same two passes, for the OTHER map. `navigation/` builds ASCENT's
    # per-storey `ObstacleMap` while it explores; these store that stack and
    # give it back, so pass 2's presence-filter pipeline plans over occupancy
    # ASCENT's navigation built rather than rebuilding its own from scratch.
    # Kept separate from `map_out`/`map_in` because the two artifacts are
    # written in different frames and either can be used without the other
    # (src/navigation/mapping/map_store.py).
    obstacle_map_out: str = ""
    obstacle_map_in: str = ""
    # Whether `map_in` also restores the OCCUPANCY it stored, or only the
    # scene graph (tracks, presence beliefs, storey heights and connectivity).
    # False is the "ASCENT's map is the sole occupancy" arm: the snapshot keeps
    # what the objects hang off, and every cell the planner reads comes from
    # `obstacle_map_in` instead of from the pass that built the scene graph.
    map_in_occupancy: bool = True
    # Whether a loaded obstacle map may overwrite cells the live map has
    # already witnessed for itself. False keeps this session's own evidence
    # and fills in only what it has not seen.
    obstacle_map_overwrite: bool = False
    # Keep EVERY mapping episode's snapshot, not the single best one, and in
    # pass 2 paste all of a scene's snapshots into the same costmaps.
    #
    # Why: the protocol runs one mapping episode per authored object and the
    # writer keeps whichever saw the most (`save_obstacle_map_for_scene`), so
    # five of six explorations are thrown away. Measured on 00800 against the
    # scene's navmesh, the kept 500-step episode covers 51% of the lower
    # storey's navigable area and 80% of the upper, and the cross-anchor
    # target for episode 1 sits 5.1 m outside it -- the agent cannot search a
    # room its prior does not contain. The explorations start from different
    # poses and see different rooms, so their UNION is strictly more map for
    # no extra simulation.
    #
    # Only snapshots whose explored-storey count matches the agent's stack are
    # merged: `ObstacleMap` records no world height, floors are matched BY
    # ORDER (see `load_obstacle_map`), and a one-storey snapshot pasted onto a
    # two-storey stack would land a whole floor on the wrong storey. Skipped
    # ones are reported in `prior_obstacle_map.union_skipped`.
    obstacle_map_union: bool = False
    # Take the STOREY HEIGHTS from the obstacle snapshots, adding any storey
    # the scene-graph prior missed.
    #
    # Why: the two prior artifacts come from mapping episodes, and an episode
    # that never climbed has no upper storey to record. Measured on 00808,
    # where NO single mapping episode visited both storeys: the scene graph
    # kept one whose upper "floor" is the stairs themselves, recorded at
    # 1.03 m, when the storeys are 0.06 and 2.86. Pass 2 reads its storey
    # heights from that stack, so the upper floor's occupancy would be pasted
    # 1.8 m below where it belongs and the stair ramp built between the wrong
    # pair. The obstacle snapshots DO record 2.86 (`floor_y`, the height the
    # mapping agent stood at), so they can fill the gap.
    #
    # A storey is added only when the stack has none within
    # `storey_seed_tol_m`; existing layers are never moved.
    seed_storeys_from_obstacle_map: bool = False
    # After the obstacle paste, take the prior's landings out of the storeys
    # (eval/prior_map.py:drop_prior_landings): 00800's pass-1 graph holds a
    # landing at 1.74 m that a descent "arrives" on.
    drop_prior_landings: bool = False
    storey_seed_tol_m: float = 1.0
    # Score a STOP by horizontal distance to the OBJECT, the released DualMap
    # benchmark's rule, instead of habitat's geodesic distance to an authored
    # viewpoint (whose rings at 0.8-2.0 m make the effective tolerance about
    # 2 m; docs/history/DUALMAP_PROTOCOL_EARLY_ARMS.md). Attempts are spent by the same rule.
    # Habitat's own numbers are kept under habitat_success / habitat_spl.
    # Pass 1 only (needs `map_out`): after an episode is scored, show its
    # target to the world model from this many of the layout's goal view
    # points, so the prior holds every object the robot is supposed to have
    # seen (eval/prior_map.py `tour_target_views`). 0 is the shipped behaviour.
    # On 00800 two of six ASCENT mapping walks never had their target in view.
    map_tour_views: int = 0
    score_by_object_distance: bool = False
    object_success_distance_m: float = 1.0
    # ...and the stop must be on the object's storey: the agent's floor height
    # within this of the navigable floor at the object. The released rule is
    # single-storey and horizontal distance alone cannot tell a stop beside the
    # object from one on the floor above it -- 2 of the 11 successes in
    # outputs/RERUN/mf5 stopped a storey away (00862 / 00873 cracker box, 2.33
    # / 2.11 m in 3D). 1.0 m because storeys here are >= 1.8 m apart
    # (`floor.new_level_m`) and a split-level step is under a metre. A no-op on
    # a single storey; 0 restores the horizontal-only rule.
    object_success_max_floor_dy_m: float = 1.0
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
