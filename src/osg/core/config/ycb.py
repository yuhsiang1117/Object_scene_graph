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


# Handle -> the class name the detector is asked for. Not always the object's
# common name: YOLOE's text head scores "pitcher" at 0.00 on this asset at every
# resolution and "blue plastic pitcher" at 0.71, so the descriptive phrase is the
# label (scripts/probe_ycb_detection.py, mode=labels).
YCB_TARGET_LABELS: Dict[str, str] = {
    "002_master_chef_can": "coffee can",
    "003_cracker_box": "cracker box",
    "005_tomato_soup_can": "tin can",
    "006_mustard_bottle": "mustard bottle",
    "011_banana": "banana",
    "019_pitcher_base": "blue plastic pitcher",
    "021_bleach_cleanser": "bleach bottle",
    "024_bowl": "bowl",
    "025_mug": "mug",
    "029_plate": "red plate",
    "037_scissors": "scissors",
    "053_mini_soccer_ball": "soccer ball",
    "077_rubiks_cube": "rubiks cube",
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
    # Combined dynamic/multi-floor benchmark selectors.  Disabled by default so
    # every existing authored manifest and experiment keeps its source behavior.
    cross_floor_relocations_only: bool = False
    relocation_directions: List[str] = field(
        default_factory=lambda: ["upward", "downward"]
    )
    start_on_prior_floor: bool = False
    relocation_floor_tolerance_m: float = 0.5
    # Explicit scene lists normally fail on any absent requested slot. Combined
    # campaigns span heterogeneous authoring coverage and opt into recording
    # those gaps while keeping every available layout.
    skip_incomplete_layouts: bool = False
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
