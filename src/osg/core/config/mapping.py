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
    # lower bound, was the culprit; see docs/INVESTIGATION.md).
    obstacle_low_m: float = 0.15
    obstacle_high_m: float = 1.5
    max_range_m: float = 5.0
    depth_stride: int = 4
    inflate_margin_m: float = 0.07


