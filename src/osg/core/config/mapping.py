"""`mapping` group: the 2D occupancy costmap built from depth."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MappingConfig:
    resolution_m: float = 0.05
    # Lower bound 0.15 (was 0.1): more floor tolerance before slightly-raised
    # ground (thresholds, rugs, ramps, floor_y drift) reads as an obstacle at
    # the robot's feet. Ceiling kept at 1.5 (aligning the FULL band to the old
    # [0.15, 0.88] regressed SR 40% -> 28.6% -- the 0.88 m ceiling, not the
    # lower bound, was the culprit; see docs/history/INVESTIGATION.md).
    obstacle_low_m: float = 0.15
    obstacle_high_m: float = 1.5
    max_range_m: float = 5.0
    depth_stride: int = 4
    inflate_margin_m: float = 0.07
    # Cells within this radius of the robot are FREE because the robot is
    # there (Costmap2D.clear_footprint). 0 is off: the simulator agent sees
    # the floor at its feet and never needed it. The Stretch presets set it
    # to the footprint plus a margin -- a glossy floor and the robot's own
    # mast in a tilted view put obstacle speckle under the base, and frontier
    # detection then grew from a 9-cell island and found nothing.
    footprint_clear_m: float = 0.0
    floor_reject_m: float = 0.0
    multi_floor: bool = False
    freeze_floor_in_climb: bool = False
    freeze_floor_on_stairs: bool = False

