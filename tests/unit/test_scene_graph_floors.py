"""Floor layer in the scene graph (osg.graph.scene_graph / serialize).

The defect these guard: `rebuild` assigned objects to rooms from their (x, z)
alone, discarding `center[1]`. A bed upstairs and a bed directly below it
therefore landed in the same room, and every consumer -- LLM prompt, nearest
-object lookup, candidate selection -- saw one indistinguishable pair.
"""
import numpy as np
import pytest

from osg.graph.scene_graph import SceneGraph
from osg.graph.serialize import to_json, to_prompt_text
from osg.mapping.costmap import Costmap2D
from osg.mapping.floors import FloorEstimator

CAM_H = 0.88


class _Track:
    def __init__(self, tid, label, center):
        self.id, self.label = tid, label
        self.center = np.asarray(center, float)
        self.n_obs, self.best_crop = 5, None


class _Layer:
    def __init__(self, tracks):
        self._t = tracks

    def tracks(self, include_blacklisted=False):
        return self._t

    def center_of(self, t):
        return t.center


def two_floor_estimator():
    e = FloorEstimator(camera_height=CAM_H)
    for step, y in enumerate([0.0] * 30 + [2.8] * 30):
        e.update(y + CAM_H, step)
    return e


def make(labels_value=1):
    cm = Costmap2D(resolution=0.05, size_m=20.0)
    labels = np.zeros(cm.grid.shape, dtype=np.int32)
    labels[:] = labels_value
    return cm, labels


# ------------------------------------------------------------ the real defect


def test_objects_stacked_vertically_get_different_floors():
    cm, labels = make()
    layer = _Layer([_Track(1, "bed", [1.0, 0.2, 1.0]),
                    _Track(2, "bed", [1.0, 3.0, 1.0])])  # same x,z -- one storey up
    sg = SceneGraph()
    sg.rebuild(labels, cm, layer, floors=two_floor_estimator())
    floors = {o.track_id: o.floor_id for o in sg.objects}
    assert floors[1] != floors[2], "vertically stacked objects collapsed onto one floor"


def test_without_floors_everything_is_floor_zero():
    """The no-op path: omitting `floors` reproduces the old behaviour exactly."""
    cm, labels = make()
    layer = _Layer([_Track(1, "bed", [1.0, 0.2, 1.0]), _Track(2, "bed", [1.0, 3.0, 1.0])])
    sg = SceneGraph()
    sg.rebuild(labels, cm, layer)
    assert {o.floor_id for o in sg.objects} == {0}
    assert sg.floors == {}


def test_floor_nodes_carry_their_height():
    cm, labels = make()
    layer = _Layer([_Track(1, "bed", [1.0, 0.2, 1.0])])
    sg = SceneGraph()
    sg.rebuild(labels, cm, layer, floors=two_floor_estimator())
    heights = sorted(f.height_y for f in sg.floors.values())
    assert heights[0] == pytest.approx(0.0, abs=0.15)
    assert heights[1] == pytest.approx(2.8, abs=0.15)


def test_room_takes_the_storey_of_most_of_its_objects():
    cm, labels = make()
    layer = _Layer([_Track(1, "bed", [1.0, 3.0, 1.0]),
                    _Track(2, "lamp", [1.2, 3.0, 1.0]),
                    _Track(3, "chair", [1.4, 0.2, 1.4])])
    sg = SceneGraph()
    sg.rebuild(labels, cm, layer, floors=two_floor_estimator())
    upper = max(sg.floors.values(), key=lambda f: f.height_y).id
    assert sg.rooms[1].floor_id == upper


# ------------------------------------------------------------- serialization


def test_prompt_text_unchanged_when_not_grouping():
    cm, labels = make()
    layer = _Layer([_Track(1, "bed", [1.0, 0.2, 1.0])])
    sg = SceneGraph()
    sg.rebuild(labels, cm, layer, floors=two_floor_estimator())
    assert to_prompt_text(sg) == "Room 1: bed x1"


def test_prompt_text_grouped_by_floor_orders_by_height():
    cm, labels = make()
    layer = _Layer([_Track(1, "bed", [1.0, 3.0, 1.0]), _Track(2, "sofa", [1.0, 0.2, 1.0])])
    sg = SceneGraph()
    # Two rooms so each storey has its own; label the grid in halves.
    labels[: labels.shape[0] // 2] = 1
    labels[labels.shape[0] // 2:] = 2
    sg.rebuild(labels, cm, layer, floors=two_floor_estimator())
    text = to_prompt_text(sg, group_by_floor=True)
    assert text.startswith("Floor "), text
    ys = [float(l.split("y=")[1].split("]")[0]) for l in text.splitlines() if l.startswith("Floor")]
    assert ys == sorted(ys), f"floors not ordered by height: {ys}"


def test_grouping_falls_back_when_no_floor_layer():
    cm, labels = make()
    sg = SceneGraph()
    sg.rebuild(labels, cm, _Layer([_Track(1, "bed", [1.0, 0.2, 1.0])]))
    assert to_prompt_text(sg, group_by_floor=True) == to_prompt_text(sg)


def test_to_json_carries_floors():
    cm, labels = make()
    layer = _Layer([_Track(1, "bed", [1.0, 3.0, 1.0])])
    sg = SceneGraph()
    sg.rebuild(labels, cm, layer, floors=two_floor_estimator())
    d = to_json(sg)
    assert d["floors"] and "floor_id" in d["objects"][0] and "floor_id" in d["rooms"][0]
    assert [f["height_y"] for f in d["floors"]] == sorted(f["height_y"] for f in d["floors"])
