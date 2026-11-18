"""`scripts/flip_cross_anchor.py` -- reversing the storey an episode must cross.

The arm measured 0/11 on ascents against 5/15 on descents, and the direction an
episode demands is `sign(cross_anchor_floor - static_floor)` because the agent
starts on the storey the PRIOR puts the object on. Reversing one therefore
means exchanging the two authored poses, and the exchange has to be exact: an
off-by-one in which fields move would leave an object floating off its anchor.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "flip_cross_anchor",
    Path(__file__).resolve().parents[2] / "scripts" / "flip_cross_anchor.py",
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def _obj(sid, xyz, anchor, floor_y, floor_index):
    return {"semantic_id": sid, "translation": list(xyz), "rotation": [0, 0, 0, 1],
            "object_id": sid, "anchor": {"object_id": anchor, "category": "table",
                                         "floor_height": floor_y,
                                         "floor_index": floor_index}}


def _scene(tmp_path, statics, crosses):
    root = tmp_path / "root"
    d = root / "00xxx-scene"
    (d / "dynamic_scene_config" / "cross_anchor").mkdir(parents=True)
    (d / "dynamic_scene_config" / "in_anchor").mkdir(parents=True)
    mapping = {str(o["semantic_id"]): f"h{o['semantic_id']}" for o in statics}
    (d / "static_scene_config.json").write_text(json.dumps(
        {"scene": "s", "id_handle_mapping": mapping, "objects": statics,
         "authoring": {}}))
    (d / "dynamic_scene_config" / "cross_anchor" / "layout_01.json").write_text(
        json.dumps({"scene": "s", "id_handle_mapping": mapping,
                    "objects": crosses, "authoring": {}}))
    return root


def test_flip_reverses_direction_and_moves_every_pose_field(tmp_path):
    statics = [_obj(50001, (0, 0.2, 0), "low_table", 0.16, 0)]
    crosses = [_obj(50001, (5, 3.3, 1), "high_table", 3.16, 1)]
    root = _scene(tmp_path, statics, crosses)

    r = mod.flip_scene(root, "00xxx-scene", 1, "upward", dry_run=False)
    assert [f["before"] for f in r["flipped"]] == ["up"]
    assert [f["after"] for f in r["flipped"]] == ["down"]

    d = root / "00xxx-scene"
    s = json.loads((d / "static_scene_config.json").read_text())["objects"][0]
    c = json.loads((d / "dynamic_scene_config" / "cross_anchor"
                    / "layout_01.json").read_text())["objects"][0]
    # The pose and the anchor travel TOGETHER; splitting them floats the object.
    assert s["translation"] == [5, 3.3, 1] and s["anchor"]["object_id"] == "high_table"
    assert c["translation"] == [0, 0.2, 0] and c["anchor"]["object_id"] == "low_table"


def test_flip_is_an_involution(tmp_path):
    """Flipping twice restores the file byte for byte.

    This is what makes a capacity-driven revert safe: seven of eleven flips had
    to be undone after `replace_placement` refused the static end.
    """
    statics = [_obj(50001, (0, 0.2, 0), "low", 0.16, 0)]
    crosses = [_obj(50001, (5, 3.3, 1), "high", 3.16, 1)]
    root = _scene(tmp_path, statics, crosses)
    before = (root / "00xxx-scene" / "static_scene_config.json").read_text()
    mod.flip_scene(root, "00xxx-scene", 1, {50001}, dry_run=False)
    mod.flip_scene(root, "00xxx-scene", 1, {50001}, dry_run=False)
    after = json.loads((root / "00xxx-scene" / "static_scene_config.json").read_text())
    assert after["objects"] == json.loads(before)["objects"]


def test_same_floor_targets_are_never_flipped(tmp_path):
    """`--auto-upward` must leave a same-storey relocation alone."""
    statics = [_obj(50002, (0, 0.2, 0), "a", 0.16, 0)]
    crosses = [_obj(50002, (3, 0.3, 0), "b", 0.16, 0)]
    root = _scene(tmp_path, statics, crosses)
    assert mod.flip_scene(root, "00xxx-scene", 1, "upward", dry_run=False)["flipped"] == []


def test_dry_run_writes_nothing(tmp_path):
    statics = [_obj(50001, (0, 0.2, 0), "low", 0.16, 0)]
    crosses = [_obj(50001, (5, 3.3, 1), "high", 3.16, 1)]
    root = _scene(tmp_path, statics, crosses)
    before = (root / "00xxx-scene" / "static_scene_config.json").read_text()
    r = mod.flip_scene(root, "00xxx-scene", 1, "upward", dry_run=True)
    assert r["flipped"]
    assert (root / "00xxx-scene" / "static_scene_config.json").read_text() == before


def test_the_partial_swap_hazard_is_reported(tmp_path):
    """A flip mixes two independently validated layouts.

    Each pose was cleared against the other objects of ITS OWN layout, so the
    exchange can put two targets inside the 0.35 m keep-out or onto one anchor.
    Measured on the real dataset: gaps of 0.02-0.33 m and three on one anchor.
    """
    statics = [_obj(50001, (0, 0.2, 0), "low", 0.16, 0),
               _obj(50002, (5.05, 3.3, 1), "high", 3.16, 1)]
    crosses = [_obj(50001, (5, 3.3, 1), "high", 3.16, 1),
               _obj(50002, (9, 0.2, 0), "far", 0.16, 0)]
    root = _scene(tmp_path, statics, crosses)
    r = mod.flip_scene(root, "00xxx-scene", 1, "upward", dry_run=True)
    assert any(c["layout"] == "static" and c["gap_m"] < 0.35 for c in r["collisions"])
    assert any(d["layout"] == "static" and d["anchor"] == "high"
               for d in r["double_anchored"])


def test_cross_floor_semantic_ids_follow_the_anchors(tmp_path):
    """The field is validated AGAINST the anchors, so it must be recomputed."""
    statics = [_obj(50001, (0, 0.2, 0), "low", 0.16, 0),
               _obj(50002, (1, 0.2, 0), "low2", 0.16, 0)]
    crosses = [_obj(50001, (5, 3.3, 1), "high", 3.16, 1),
               _obj(50002, (2, 0.3, 0), "low3", 0.16, 0)]
    root = _scene(tmp_path, statics, crosses)
    mod.flip_scene(root, "00xxx-scene", 1, "upward", dry_run=False)
    ca = json.loads((root / "00xxx-scene" / "dynamic_scene_config" / "cross_anchor"
                     / "layout_01.json").read_text())
    assert ca["authoring"]["cross_floor_semantic_ids"] == [50001]
