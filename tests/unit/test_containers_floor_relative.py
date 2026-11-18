"""A support surface is measured from its own floor, not from y=0.

`container_top_h_m` is a band ABOVE THE FLOOR -- 0.2 to 1.4 m, the height range
a thing can be set down on -- and `containers.top_height` returns an absolute
world height. On the ground floor the two coincide. On any storey above it they
do not, and the band then rejects every surface there.

Measured on 00808's prior map: floor 1 holds 367 mapped tracks and produced ZERO
containers, against 76 on floor 0. The container posterior could therefore
propose no surface upstairs, and the per-floor mass that chooses a storey had an
entry for the ground floor alone -- so an agent standing upstairs computed
floor_mass {'0': 48.2} and asked to go down, every round, whatever its target.
"""
from __future__ import annotations

import numpy as np

from osg.graph.scene_graph import SceneGraph
from osg.mapping.costmap import Costmap2D
from osg.objects.association import ObjectTrack
from osg.objects.ellipsoid import Ellipsoid

UPPER_FLOOR_Y = 2.86


class _Layer:
    def __init__(self, tracks):
        self._t = {t.id: t for t in tracks}

    def tracks(self, include_blacklisted: bool = False):
        return list(self._t.values())

    def center_of(self, track):
        return np.asarray(track.ellipsoid.center, dtype=float)


def _table(tid, floor_y, floor_key):
    """A table whose top sits 0.75 m above the given floor."""
    centre = np.array([1.0, floor_y + 0.6, 2.0])
    track = ObjectTrack(
        id=tid, label="table",
        ellipsoid=Ellipsoid(center=centre, axes=np.array([0.6, 0.15, 0.6]), R=np.eye(3)),
    )
    track.best_score = 0.9
    track.floor_key = floor_key
    for _ in range(5):
        track.observations.append(None)
    return track


def _graph(floor_relative):
    return SceneGraph(container_top_h_m=(0.2, 1.4), container_min_area_m2=0.0,
                      container_min_obs=1, container_min_score=0.0,
                      containers_floor_relative=floor_relative)


def _rebuild(graph, track, floor_key, floor_y):
    costmap = Costmap2D(resolution=0.05, size_m=20.0)
    labels = np.zeros(costmap.grid.shape, dtype=np.int32)
    graph.rebuild_floor(labels, costmap, _Layer([track]),
                        floor_key=floor_key, floor_height=floor_y)


def test_the_ground_floor_is_unaffected():
    """Where the bug never showed, the behaviour must not move."""
    for relative in (False, True):
        graph = _graph(relative)
        _rebuild(graph, _table(1, 0.0, 0), floor_key=0, floor_y=0.0)
        assert len(graph.containers) == 1, f"floor_relative={relative}"


def test_an_upstairs_table_is_invisible_without_the_fix():
    graph = _graph(False)
    _rebuild(graph, _table(1, UPPER_FLOOR_Y, 1), floor_key=1, floor_y=UPPER_FLOOR_Y)
    assert graph.containers == {}


def test_an_upstairs_table_is_a_container_with_it():
    graph = _graph(True)
    _rebuild(graph, _table(1, UPPER_FLOOR_Y, 1), floor_key=1, floor_y=UPPER_FLOOR_Y)
    assert len(graph.containers) == 1
    node = next(iter(graph.containers.values()))
    assert node.floor_id == 1


def test_top_h_stays_an_absolute_height():
    """`build_container_candidates` subtracts the floor height itself. If the
    stored value were already relative the two would both subtract and every
    upstairs surface would come out below its floor."""
    graph = _graph(True)
    _rebuild(graph, _table(1, UPPER_FLOOR_Y, 1), floor_key=1, floor_y=UPPER_FLOOR_Y)
    node = next(iter(graph.containers.values()))
    assert node.top_h > UPPER_FLOOR_Y


def test_a_ceiling_light_upstairs_is_still_not_a_surface():
    """The band must still reject things out of reach on its own floor."""
    graph = _graph(True)
    high = _table(1, UPPER_FLOOR_Y + 1.6, 1)
    _rebuild(graph, high, floor_key=1, floor_y=UPPER_FLOOR_Y)
    assert graph.containers == {}
