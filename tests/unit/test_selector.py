from __future__ import annotations

import numpy as np

from osg.mapping.costmap import FREE, Costmap2D
from osg.mapping.frontier import Frontier
from osg.exploration.selector import select_frontier
from osg.planning.planner import AStarPlanner


def _setup():
    cm = Costmap2D(resolution=0.1, size_m=10.0)
    cm.grid[:, :] = FREE
    planner = AStarPlanner(inflate_radius_m=0.05)
    f_near = Frontier(id=1, centroid_xy=np.array([1.0, 0.0]), cells=np.zeros((5, 2)), size=5)
    f_far = Frontier(id=2, centroid_xy=np.array([4.0, 0.0]), cells=np.zeros((5, 2)), size=5)
    return cm, planner, f_near, f_far


def test_score_per_distance_tradeoff():
    cm, planner, f_near, f_far = _setup()
    # Far frontier has higher P but not enough to beat 4x the distance
    scores = {1: 0.5, 2: 0.6}
    best = select_frontier([f_near, f_far], scores, planner, cm, np.array([0.0, 0.0]))
    assert best.id == 1
    # Overwhelming relevance wins despite distance
    scores = {1: 0.1, 2: 0.9}
    best = select_frontier([f_near, f_far], scores, planner, cm, np.array([0.0, 0.0]))
    assert best.id == 2


def test_blocked_frontiers_skipped():
    cm, planner, f_near, f_far = _setup()
    best = select_frontier(
        [f_near, f_far], {1: 0.9, 2: 0.9}, planner, cm, np.array([0.0, 0.0]), blocked={1}
    )
    assert best.id == 2


def test_unscored_prior_used():
    cm, planner, f_near, f_far = _setup()
    best = select_frontier([f_near, f_far], {}, planner, cm, np.array([0.0, 0.0]))
    assert best is not None  # prior lets selection proceed without LLM scores
    assert best.id == 1
