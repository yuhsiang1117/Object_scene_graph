"""The storey belief (`exploration/storey_belief.py`).

What it must get right, each from a measured failure on Benchmark v2 seed 42:
a landing stored as a storey is not a storey (00800's prior holds 3.16, 1.74 and
0.64 m); a track is on the storey its HEIGHT says, not the layer it was filed
under; coverage, not a failed STOP, is what moves belief off a storey; and with
two storeys known, the undirected "nearest flight, invented storey" switch never
fires (9 of 28 descents pursued only flights going up).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from osg.exploration.storey_belief import StoreyBelief, target_tracks
from osg.mapping.costmap import FREE, Costmap2D


def _cfg(**kw):
    base = dict(storey_anchor_weight=0.8, storey_detect_prob=0.85, storey_switch_ratio=1.3,
                storey_cover_radius_m=2.0, storey_block_m=0.5)
    base.update(kw)
    return SimpleNamespace(**base)


def _floor(free_m: float) -> Costmap2D:
    """A costmap whose central `free_m` x `free_m` square is free floor."""
    cm = Costmap2D(resolution=0.1, size_m=20.0)
    n = int(free_m / 0.1)
    c = cm.grid.shape[0] // 2
    cm.grid[c - n // 2:c + n // 2, c - n // 2:c + n // 2] = FREE
    return cm


def _belief(**kw):
    return StoreyBelief(_cfg(**kw), new_level_m=1.8, stats={})


def test_a_landing_is_merged_into_the_storey_with_more_floor():
    b = _belief()
    layers = {0: (3.16, _floor(10.0)), 1: (1.74, _floor(1.0)), 2: (0.64, _floor(8.0))}
    storeys, rep, _ = b.storeys(layers)
    assert storeys == {0: 3.16, 2: 0.64}
    assert rep == {0: 0, 1: 2, 2: 2}  # 1.74 sits within a storey of 0.64


def test_a_track_belongs_to_the_storey_its_height_says():
    storeys = {0: 3.16, 2: 0.16}
    assert StoreyBelief.storey_of_height(0.9, storeys) == 2     # on a table downstairs
    assert StoreyBelief.storey_of_height(2.7, storeys) == 2     # a tall shelf downstairs
    assert StoreyBelief.storey_of_height(3.0, storeys) == 0     # on the upper floor, within noise
    assert StoreyBelief.storey_of_height(-1.0, storeys) == 2    # below everything: the lowest


def test_an_anchor_on_the_other_storey_is_asked_for_at_once():
    b = _belief()
    layers = {0: (3.16, _floor(10.0)), 1: (0.16, _floor(10.0))}
    assert b.decide(5, layers, [(0.9, 0.9)], current_key=0) == 1
    assert b.log[-1]["request"] == 1


def test_a_believed_anchor_here_holds_the_agent_however_much_is_covered():
    b = _belief()
    layers = {0: (0.16, _floor(6.0)), 1: (3.16, _floor(6.0))}
    for x in np.arange(-3.0, 3.01, 1.0):
        for y in np.arange(-3.0, 3.01, 1.0):
            b.note(0, (x, y))
    post = b.posterior(layers, [(0.9, 0.9)], current_key=0)
    assert post["cover"][0] > 0.95
    assert b.decide(5, layers, [(0.9, 0.9)], current_key=0) is None


def test_coverage_moves_belief_once_the_anchor_is_disbelieved():
    b = _belief()
    layers = {0: (0.16, _floor(6.0)), 1: (3.16, _floor(6.0))}
    stale = [(0.9, 0.1)]  # the agent looked at the anchor's pose and found nothing
    assert b.decide(5, layers, stale, current_key=0) is None   # nothing covered yet
    for x in np.arange(-3.0, 3.01, 1.0):
        for y in np.arange(-3.0, 3.01, 1.0):
            b.note(0, (x, y))
    assert b.decide(10, layers, stale, current_key=0) == 1


def test_no_anchor_anywhere_and_nothing_covered_means_stay():
    b = _belief()
    layers = {0: (0.16, _floor(6.0)), 1: (3.16, _floor(6.0))}
    assert b.decide(5, layers, [], current_key=0) is None


def test_a_request_is_held_while_the_other_storey_stays_ahead():
    b = _belief(storey_switch_ratio=2.0)
    layers = {0: (0.16, _floor(6.0)), 1: (3.16, _floor(6.0))}
    assert b.decide(5, layers, [(3.5, 0.6)], current_key=0) == 1     # 0.58 vs 0.10: asked
    # A weaker pull, under the ratio but still ahead (0.18 vs 0.10): it stands.
    assert b.decide(10, layers, [(3.5, 0.1)], current_key=0) == 1
    # Behind: dropped.
    assert b.decide(15, layers, [(0.9, 0.5)], current_key=0) is None


def test_standing_on_a_landing_counts_as_its_storey():
    b = _belief()
    layers = {0: (3.16, _floor(10.0)), 1: (1.0, _floor(1.0)), 2: (0.16, _floor(10.0))}
    # On the landing (layer 1, nearer the lower storey), the object downstairs:
    # nothing to ask for.
    assert b.decide(5, layers, [(0.9, 0.9)], current_key=1) is None
    assert b.log[-1]["here"] == 2


def test_one_storey_known_leaves_the_undirected_switch_alone():
    b = _belief()
    assert not b.knows_another_storey({0: (0.16, _floor(6.0))})
    assert not b.knows_another_storey({0: (0.16, _floor(6.0)), 1: (1.2, _floor(1.0))})
    assert b.knows_another_storey({0: (0.16, _floor(6.0)), 1: (3.16, _floor(6.0))})


def test_target_tracks_reads_label_height_and_presence():
    tracks = [
        SimpleNamespace(label="tin can", presence=SimpleNamespace(p=0.7), c=(1.0, 0.9, 2.0)),
        SimpleNamespace(label="bowl", presence=SimpleNamespace(p=0.9), c=(1.0, 0.9, 2.0)),
        SimpleNamespace(label="tin can", presence=None, c=(1.0, 3.9, 2.0)),
    ]
    layer = SimpleNamespace(tracks=lambda: tracks, center_of=lambda t: np.array(t.c))
    assert target_tracks(layer, "tin can") == [(0.9, 0.7), (3.9, 0.5)]


def test_identity_evidence_takes_a_present_false_positive_out():
    """Presence says something is there; a rejection says it is not the target."""
    def t(**kw):
        base = dict(label="pitcher", presence=SimpleNamespace(p=0.9), c=(0.0, 0.5, 0.0),
                    identity_rejections=0, failed_attempts=0)
        base.update(kw)
        return SimpleNamespace(**base)
    tracks = [t(), t(identity_rejections=1), t(identity_rejections=2), t(failed_attempts=1)]
    layer = SimpleNamespace(tracks=lambda: tracks, center_of=lambda tr: np.array(tr.c))
    got = target_tracks(layer, "pitcher", max_rejections=2, rejection_discount=0.3)
    assert [round(w, 3) for _, w in got] == [0.9, 0.27]


# ----------------------------------------------------------- in the strategy

def _strategy(belief: bool):
    from osg.exploration.strategy import ExplorationStrategy

    cfg = SimpleNamespace(
        exploration=SimpleNamespace(
            storey_belief=belief, frontier_min_cells=5, frontier_dedup_m=0.5,
            frontier_goal_free_cell=False, search_drop_proximity_after_absence=False,
            frontier_cost_free_cell=False, voronoi_goal_near_m=0.7, select_every=1,
            search_posterior=False, **vars(_cfg())),
        floor=SimpleNamespace(new_level_m=1.8),
    )
    return ExplorationStrategy(cfg, None, None, None, None, stats={},
                               profiler=SimpleNamespace(timeit=lambda name: _NullCtx()))


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _world_two_storeys(strategy, tracks):
    from osg.exploration.strategy import WorldView

    layer = SimpleNamespace(tracks=lambda: tracks, center_of=lambda t: np.array(t.c))
    empty = Costmap2D(resolution=0.1, size_m=4.0)       # no frontier anywhere
    return WorldView(
        frame=None, step=10, agent_xy=np.zeros(2), costmap=empty, scene_graph=None,
        object_layer=layer, keyframes=None, target="tin can", goal_xy=None, floor_id=0,
        storeys={0: (0.16, _floor(6.0)), 1: (3.16, _floor(6.0))},
    )


def test_with_two_storeys_known_the_switch_is_directed_or_not_at_all():
    calls = []

    def floor_switch(cost, target_floor=None):
        calls.append(target_floor)
        return False

    s = _strategy(belief=True)
    s.select(_world_two_storeys(s, []), floor_switch)
    assert calls == []                   # nothing believed elsewhere: no undirected switch
    up = [SimpleNamespace(label="tin can", presence=SimpleNamespace(p=0.9), c=(1.0, 3.9, 2.0))]
    s._last_select_step = -100
    s.select(_world_two_storeys(s, up), floor_switch)
    assert calls == [1]                  # directed, at the storey the anchor is on


def test_off_the_strategy_is_unchanged():
    calls = []
    s = _strategy(belief=False)
    assert s.storey_belief is None
    s.select(_world_two_storeys(s, []), lambda cost, target_floor=None: calls.append(target_floor))
    assert calls == [None]               # the legacy undirected switch
