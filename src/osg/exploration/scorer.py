"""Frontier scorer interface. The old-algorithm pipeline uses the text-LLM
scorer (LLMTextScorer) -- a scene-graph subgraph ranking, matching
ObjectSceneGraph_old's frontiers_ranking.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

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

    def reset(self) -> None:
        """Clear any per-episode state before a new episode starts. A scorer
        instance is built once and reused across every episode in a run
        (see eval/runner.py); subclasses that key a cache by frontier.id or
        room.id (both small integers that restart from 0/1 each episode --
        see FrontierExtractor/RoomSegmenter) must override this, or stale
        entries from a previous, unrelated scene silently leak into the
        current one whenever an id happens to collide. No-op by default."""
