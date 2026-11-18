"""`eval/prior_map.drop_prior_landings`: a pass-1 landing is not a storey.

00800's pass-1 graph holds 3.16, 1.74 and 0.64 m; the 1.74 layer is a landing,
and a descent with the navmesh walking the stairs "arrived" on it. 00878's top
storey has no pasted occupancy at all and is real -- the rule must keep it.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from osg.eval.prior_map import drop_prior_landings
from osg.mapping.costmap import FREE
from osg.mapping.floor_stack import FloorStack


def _agent(heights_area, current=0, tracks=()):
    stack = FloorStack()
    for key, (h, free_m) in heights_area.items():
        layer = stack.layer(key)
        stack.set_height(key, h)
        n = int(free_m / layer.costmap.resolution)
        c = layer.costmap.grid.shape[0] // 2
        layer.costmap.grid[c - n // 2:c + n // 2, c - n // 2:c + n // 2] = FREE
    stack.current_id = current
    estimator = SimpleNamespace(_levels={k: h for k, (h, _) in heights_area.items()},
                                _samples={k: [h] for k, (h, _) in heights_area.items()})
    layer = SimpleNamespace(
        tracks=lambda include_blacklisted=False, include_proposals=False: list(tracks),
        center_of=lambda t: np.array(t.c))
    return SimpleNamespace(_floor_stack=stack, floors=SimpleNamespace(estimator=estimator),
                           object_layer=layer)


def _cfg():
    return SimpleNamespace(exploration=SimpleNamespace(storey_block_m=0.5),
                           floor=SimpleNamespace(new_level_m=1.8))


def test_the_00800_landing_goes_and_its_tracks_are_refiled():
    on_landing = SimpleNamespace(floor_key=1, c=(0.0, 3.5, 0.0))   # an object upstairs, seen from the landing
    agent = _agent({0: (3.16, 8.0), 1: (1.74, 1.0), 2: (0.64, 8.0)}, tracks=[on_landing])
    assert drop_prior_landings(_cfg(), agent) == [1.74]
    assert sorted(agent._floor_stack._layers) == [0, 2]
    assert sorted(agent.floors.estimator._levels) == [0, 2]
    assert on_landing.floor_key == 0
    # The stair edges still name it (`FloorPolicy._remembered_arrival`).
    assert agent._floor_stack.dropped_levels == {1: 1.74}


def test_an_unwalked_storey_is_kept():
    agent = _agent({0: (-2.80, 8.0), 1: (0.07, 8.0), 2: (2.80, 0.0)}, current=1)
    assert drop_prior_landings(_cfg(), agent) == []
    assert sorted(agent._floor_stack._layers) == [0, 1, 2]


def test_the_storey_the_agent_starts_on_is_never_dropped():
    agent = _agent({0: (3.16, 8.0), 1: (1.74, 1.0), 2: (0.64, 8.0)}, current=1)
    assert drop_prior_landings(_cfg(), agent) == []
