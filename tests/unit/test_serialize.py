from __future__ import annotations

import numpy as np

from osg.graph.scene_graph import ObjectNodeView, RoomNode, SceneGraph
from osg.graph.serialize import to_json, to_prompt_text


def _graph() -> SceneGraph:
    sg = SceneGraph()
    sg.rooms = {
        1: RoomNode(id=1, label="bedroom", centroid_xy=np.array([1.0, 1.0]), n_cells=400),
        2: RoomNode(id=2, label=None, centroid_xy=np.array([5.0, 1.0]), n_cells=300),
    }
    sg.objects = [
        ObjectNodeView(track_id=0, label="bed", center=np.array([1.0, 0.5, 1.0]), room_id=1, n_obs=5),
        ObjectNodeView(track_id=1, label="lamp", center=np.array([1.3, 0.8, 1.2]), room_id=1, n_obs=3),
        ObjectNodeView(track_id=2, label="lamp", center=np.array([0.7, 0.8, 0.9]), room_id=1, n_obs=2),
        ObjectNodeView(track_id=3, label="sink", center=np.array([5.0, 0.9, 1.0]), room_id=2, n_obs=4),
        ObjectNodeView(track_id=4, label="towel", center=np.array([9.0, 0.9, 9.0]), room_id=0, n_obs=1),
    ]
    return sg


def test_prompt_text_golden():
    text = to_prompt_text(_graph())
    assert "Room 1 (bedroom): bed x1 (near lamp), lamp x2 (near bed)" in text
    assert "Room 2: sink x1" in text
    assert "Hallway/other: towel x1" in text


def test_empty_graph():
    assert to_prompt_text(SceneGraph()) == "(no objects mapped yet)"


def test_json_roundtrip():
    j = to_json(_graph())
    assert len(j["rooms"]) == 2
    assert len(j["objects"]) == 5
    assert j["objects"][0]["label"] == "bed"
    import json

    json.dumps(j)  # must be serializable
