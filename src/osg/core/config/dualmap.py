"""`dualmap` group: our stack on DualMap's released benchmark.

Nothing here defines the benchmark. The trials, the targets and the success
rule live in `osg.eval.dualmap_release`, which the DualMap harness imports too,
precisely so a comparison cannot drift by editing a config. These are the
selectors for one run: which slice of the 186 trials to drive, and which
measured DualMap run supplies the start poses that make the two runs paired.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class DualMapProtocolConfig:
    """Selectors for a run against the released DualMap benchmark."""

    # The measured DualMap run whose per-trial `start_position` this run reuses.
    # Required: without it the two systems answer the same query from different
    # places and per-trial agreement means nothing.
    reference_run: str = ""
    scenes: List[str] = field(default_factory=lambda: ["*"])
    conditions: List[str] = field(
        default_factory=lambda: ["static", "in_anchor", "cross_anchor"]
    )
    # Restrict to specific queries or trial ids (empty = the whole selection).
    queries: List[str] = field(default_factory=list)
    trial_ids: List[str] = field(default_factory=list)
    max_trials: int = -1
    # The released benchmark records where each trial started but not which way
    # the agent faced, so the heading is drawn per trial from this seed.
    start_yaw_seed: int = 12
