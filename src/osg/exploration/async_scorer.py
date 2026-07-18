"""Async wrapper: LLM/VLM scoring must never block the control loop — the
real-time claim concerns the mapping/control loop; decision latency is
reported separately. Single worker; a request is dropped if one is in flight;
the agent keeps acting on the latest (possibly stale) scores.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from ..graph.scene_graph import SceneGraph
from ..mapping.frontier import Frontier
from ..perception.keyframe import KeyframeStore
from .scorer import FrontierScorer


class AsyncScorer:
    def __init__(self, scorer: FrontierScorer, error_backoff_s: float = 30.0) -> None:
        self.scorer = scorer
        self.error_backoff_s = error_backoff_s
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._latest: Dict[int, float] = {}
        self._in_flight = False
        self._retry_after = 0.0
        self.last_latency_s: Optional[float] = None
        self.last_error: Optional[str] = None
        self.n_calls = 0
        self.n_errors = 0

    def request(
        self,
        frontiers: List[Frontier],
        sg: SceneGraph,
        target: str,
        keyframes: Optional[KeyframeStore] = None,
    ) -> bool:
        """Submit a scoring request; returns False if one is already running
        or the scorer is in error backoff."""
        with self._lock:
            if self._in_flight or time.monotonic() < self._retry_after:
                return False
            self._in_flight = True
        self._executor.submit(self._run, list(frontiers), sg, target, keyframes)
        return True

    def _run(self, frontiers, sg, target, keyframes) -> None:
        t0 = time.perf_counter()
        try:
            scores = self.scorer.score(frontiers, sg, target, keyframes)
            with self._lock:
                self._latest.update(scores)
                self.last_error = None
        except Exception as e:
            with self._lock:
                self.n_errors += 1
                self.last_error = f"{type(e).__name__}: {e}"[:300]
                self._retry_after = time.monotonic() + self.error_backoff_s
        finally:
            with self._lock:
                self._in_flight = False
                self.n_calls += 1
                self.last_latency_s = time.perf_counter() - t0

    def latest(self) -> Dict[int, float]:
        with self._lock:
            return dict(self._latest)

    def reset(self) -> None:
        """Call at the start of each episode: frontier.id restarts from 0
        each episode (fresh FrontierExtractor per NavAgent), so a stale
        _latest entry would otherwise hand select_frontier() a score from a
        different frontier in a previous, unrelated scene the moment a new
        episode's id happens to collide with an old one."""
        with self._lock:
            self._latest = {}
        self.scorer.reset()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
