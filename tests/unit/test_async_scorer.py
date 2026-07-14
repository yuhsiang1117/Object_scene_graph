from __future__ import annotations

import time

import numpy as np

from osg.exploration.async_scorer import AsyncScorer
from osg.exploration.scorer import FrontierScorer
from osg.graph.scene_graph import SceneGraph
from osg.mapping.frontier import Frontier


class SlowScorer(FrontierScorer):
    def __init__(self, delay=0.2):
        self.delay = delay
        self.calls = 0

    def score(self, frontiers, sg, target, keyframes=None):
        self.calls += 1
        time.sleep(self.delay)
        return {f.id: 0.7 for f in frontiers}


def _frontiers():
    return [Frontier(id=1, centroid_xy=np.zeros(2), cells=np.zeros((1, 2)), size=1)]


def test_request_does_not_block():
    a = AsyncScorer(SlowScorer(delay=0.3))
    t0 = time.perf_counter()
    assert a.request(_frontiers(), SceneGraph(), "bed")
    assert time.perf_counter() - t0 < 0.1  # returned immediately
    assert a.latest() == {}  # no scores yet
    time.sleep(0.5)
    assert a.latest() == {1: 0.7}
    a.shutdown()


def test_in_flight_requests_dropped():
    scorer = SlowScorer(delay=0.3)
    a = AsyncScorer(scorer)
    assert a.request(_frontiers(), SceneGraph(), "bed")
    assert not a.request(_frontiers(), SceneGraph(), "bed")  # dropped
    time.sleep(0.5)
    assert scorer.calls == 1
    a.shutdown()


def test_error_is_contained():
    class Boom(FrontierScorer):
        def score(self, *a, **k):
            raise RuntimeError("llm down")

    a = AsyncScorer(Boom())
    a.request(_frontiers(), SceneGraph(), "bed")
    time.sleep(0.2)
    assert a.n_errors == 1
    assert a.latest() == {}
    a.shutdown()
