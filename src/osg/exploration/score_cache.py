"""Frontier scores keyed by position, not by frontier id.

`FrontierExtractor` allocates a fresh id for every frontier on every
extraction, so a score computed in one round is looked up under an id that no
longer exists in the next. That is why the asynchronous LLM scorer was measured
to have never influenced a single selection: its results always arrived keyed to
frontiers that had already been renumbered.

Positions survive renumbering, so scores are stored and retrieved spatially, and
expire after a while because the map they were computed against keeps changing.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class SpatialScoreCache:
    def __init__(self, radius_m: float = 0.75, ttl_steps: int = 60) -> None:
        """
        Args:
            radius_m: a query within this of a stored point reuses its score.
                Roughly the frontier dedup distance -- close enough that it is
                the same opening, far enough to survive centroid jitter as the
                frontier grows.
            ttl_steps: entries older than this are dropped. A score describes
                what was known then; the map moves on.
        """
        self.radius_m = radius_m
        self.ttl_steps = ttl_steps
        self._entries: List[Tuple[np.ndarray, float, int]] = []

    def put(self, xy: np.ndarray, score: float, step: int) -> None:
        self._entries.append((np.asarray(xy, dtype=float).copy(), float(score), int(step)))

    def get(self, xy: np.ndarray, step: int) -> Optional[float]:
        """Score of the nearest live entry within `radius_m`, else None."""
        self._expire(step)
        xy = np.asarray(xy, dtype=float)
        best, best_d = None, self.radius_m
        for pos, score, _ in self._entries:
            d = float(np.linalg.norm(pos - xy))
            if d < best_d:
                best, best_d = score, d
        return best

    def _expire(self, step: int) -> None:
        if self.ttl_steps <= 0:
            return
        cutoff = step - self.ttl_steps
        self._entries = [e for e in self._entries if e[2] > cutoff]

    def reset(self) -> None:
        """Per episode: positions are episode-relative and would otherwise leak
        a previous, unrelated scene's scores into this one."""
        self._entries = []

    def __len__(self) -> int:
        return len(self._entries)
