"""One normalisation for category names, used everywhere they are compared.

The vocabulary, the detector's output, the episode target, the container table
and the affinity priors all name categories, and they disagree about spelling:
Habitat's ObjectNav targets are `tv_monitor`, the detector answers to
`tv monitor`, and an authored layout may carry either. Every comparison in the
codebase therefore has to normalise first, and it was being written out by hand
in fourteen places across nine modules -- five of which had also defined their
own private `_norm`, three subtly different in whether they coerced to `str`.

A category comparison that silently fails does not raise; it produces an empty
candidate list and an episode that looks like a navigation failure. That is
worth exactly one function.
"""
from __future__ import annotations


def normalize_label(label) -> str:
    """Lower-case, underscores to spaces, stripped. `str()` first, because
    labels arrive from JSON, numpy scalars and OmegaConf as well as from code."""
    return str(label).lower().replace("_", " ").strip()


def same_label(a, b) -> bool:
    """Do these two names refer to the same category?"""
    return normalize_label(a) == normalize_label(b)
