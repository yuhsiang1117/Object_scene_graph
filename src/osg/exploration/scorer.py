"""Frontier scorers. The ablation ladder (random -> nearest -> text LLM ->
multimodal VLM) is a config switch; all scorers share this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

from ..graph.scene_graph import SceneGraph
from ..mapping.frontier import Frontier
from ..perception.keyframe import KeyframeStore


class FrontierScorer(ABC):
    @abstractmethod
    def score(
        self,
        frontiers: List[Frontier],
        sg: SceneGraph,
        target: str,
        keyframes: Optional[KeyframeStore] = None,
    ) -> Dict[int, float]:
        """Returns {frontier_id: P in [0, 1]}. May score a subset."""


class RandomScorer(FrontierScorer):
    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def score(self, frontiers, sg, target, keyframes=None):
        return {f.id: float(self._rng.random()) for f in frontiers}


class NearestScorer(FrontierScorer):
    """P_i = 1 for all: selection reduces to argmax 1/d_i (nearest frontier)."""

    def score(self, frontiers, sg, target, keyframes=None):
        return {f.id: 1.0 for f in frontiers}
