"""Everything downstream of the per-floor map stack must respect the floor.

Two floors of a house occupy the SAME (x, z) footprint and differ only in
height, so any component that reasons on the ground plane will happily merge,
match or mislabel across storeys unless it is told not to.
"""
from __future__ import annotations

import numpy as np

from osg.core.types import Detection
from osg.graph.scene_graph import SceneGraph
from osg.mapping.costmap import FREE, Costmap2D
from osg.objects.association import ObjectTrack
from osg.objects.ellipsoid import Ellipsoid
from osg.objects.linking import object_center, relink
from osg.objects.object_layer import ObjectLayer

from .conftest import draw_ellipse_mask, make_frame

STOREY = 2.7


def _track(tid, label, center, floor_key=0):
    return ObjectTrack(
        id=tid,
        label=label,
        ellipsoid=Ellipsoid(
            center=np.asarray(center, dtype=float),
            axes=np.array([0.3, 0.3, 0.3]),
            R=np.eye(3),
        ),
        floor_key=floor_key,
    )


def _det(label, center_px=(320, 240), semi_px=(60, 40), score=0.8):
    mask = draw_ellipse_mask(480, 640, center_px, semi_px)
    x1, y1 = center_px[0] - semi_px[0], center_px[1] - semi_px[1]
    x2, y2 = center_px[0] + semi_px[0], center_px[1] + semi_px[1]
    return Detection(label=label, score=score, bbox_xyxy=np.array([x1, y1, x2, y2]), mask=mask)


# --------------------------------------------------------------------- linking


def test_relink_does_not_join_floors():
    """A mezzanine puts two same-category objects within link_dist_m in 3D. If
    they merge, object_center returns the midpoint -- a position on neither
    floor, which the agent then walks to."""
    below = _track(0, "sofa", [0.0, 0.0, 0.0], floor_key=0)
    above = _track(1, "sofa", [0.0, 0.8, 0.0], floor_key=1)
    relink([below, above], link_dist_m=1.0)
    assert below.linked_ids == set()
    assert above.linked_ids == set()


def test_relink_still_joins_within_a_floor():
    """The floor gate must not break normal linking (an L-shaped sofa split
    across two ellipsoids)."""
    a = _track(0, "sofa", [0.0, 0.0, 0.0], floor_key=0)
    b = _track(1, "sofa", [0.5, 0.0, 0.0], floor_key=0)
    relink([a, b], link_dist_m=1.0)
    assert a.linked_ids == {1}
    assert object_center(a, {0: a, 1: b})[0] == 0.25


# ----------------------------------------------------------------- association


def test_association_ignores_tracks_on_another_floor(intrinsics):
    """A track a storey below projects into the lower image rows and would
    otherwise capture a detection of a different object on this floor."""
    layer = ObjectLayer(min_det_score=0.0, min_det_bbox_px=0.0)
    frame = make_frame(intrinsics, np.eye(4), depth_value=3.0)

    layer.update(frame, [_det("chair")], floor_key=0)
    assert len(layer.tracks()) == 1

    # Same view, same detection, but the agent is now on floor 1.
    layer.update(frame, [_det("chair")], floor_key=1)
    tracks = layer.tracks()
    assert len(tracks) == 2, "the floor-0 track absorbed a floor-1 detection"
    assert {t.floor_key for t in tracks} == {0, 1}


def test_candidates_can_be_restricted_to_a_floor(intrinsics):
    layer = ObjectLayer(min_det_score=0.0, min_det_bbox_px=0.0)
    frame = make_frame(intrinsics, np.eye(4), depth_value=3.0)
    layer.update(frame, [_det("chair")], floor_key=0)
    layer.update(frame, [_det("chair")], floor_key=1)

    assert len(layer.candidates("chair", min_obs=1)) == 2
    assert len(layer.candidates("chair", min_obs=1, floor_key=1)) == 1
    assert layer.candidates("chair", min_obs=1, floor_key=1)[0].floor_key == 1


def test_candidates_unfiltered_by_default_so_other_floors_stay_visible(intrinsics):
    """A target mapped upstairs is not somewhere to walk to, but it is the
    signal that should send the agent up the stairs -- so the default must not
    hide it."""
    layer = ObjectLayer(min_det_score=0.0, min_det_bbox_px=0.0)
    frame = make_frame(intrinsics, np.eye(4), depth_value=3.0)
    layer.update(frame, [_det("chair")], floor_key=1)
    assert len(layer.candidates("chair", min_obs=1)) == 1


# ----------------------------------------------------------------- scene graph


def _costmap_with_room():
    cm = Costmap2D(resolution=0.1, size_m=10.0)
    cm.grid[40:60, 40:60] = FREE
    labels = np.zeros(cm.grid.shape, dtype=np.int32)
    labels[40:60, 40:60] = 1
    return cm, labels


def test_rebuild_floor_keeps_other_floors():
    """Re-segmenting one storey must not drop what is mapped elsewhere: the
    agent still needs that for frontier scoring and for knowing where it has
    been."""
    cm, labels = _costmap_with_room()
    x, z = cm.grid_to_world(np.array([50, 50]))
    layer = ObjectLayer()
    # Same ground-plane position, different storeys -- the case that breaks
    # anything reasoning in 2D.
    layer._tracks[0] = _track(0, "bed", [x, 0.0, z], floor_key=0)
    layer._tracks[1] = _track(1, "sofa", [x, STOREY, z], floor_key=1)

    sg = SceneGraph()
    sg.rebuild_floor(labels, cm, layer, floor_key=0)
    sg.rebuild_floor(labels, cm, layer, floor_key=1)

    assert {o.label for o in sg.objects} == {"bed", "sofa"}
    assert {o.floor for o in sg.objects} == {0, 1}

    # Re-running floor 0 must leave floor 1 alone.
    sg.rebuild_floor(labels, cm, layer, floor_key=0)
    assert {o.label for o in sg.objects} == {"bed", "sofa"}


def test_room_ids_are_namespaced_per_floor():
    """Every floor's segmenter emits a room 1; without namespacing they merge
    into one room spanning two storeys."""
    cm, labels = _costmap_with_room()
    layer = ObjectLayer()
    sg = SceneGraph()
    sg.rebuild_floor(labels, cm, layer, floor_key=0)
    sg.rebuild_floor(labels, cm, layer, floor_key=1)

    assert len(sg.rooms) == 2
    assert {r.floor for r in sg.rooms.values()} == {0, 1}
    assert len(set(sg.rooms.keys())) == 2
