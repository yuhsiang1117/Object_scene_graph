"""Importing the collector's raw layouts into the authoring schema.

The collector writes bare pose lists: no `authoring` block, no per-object anchor,
and a scene path from a machine that no longer exists. The validator needs all
three, and relaxing it would throw away the checks that make a layout
trustworthy -- so the importer converts, and these tests pin the conversion.

Habitat-free: this is JSON in, JSON out.
"""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "import_collector_layouts",
    Path(__file__).resolve().parents[2] / "scripts" / "import_collector_layouts.py",
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def raw(objects):
    return {
        "scene": {"scene_path": "/home/eku/whatever/QaLdnwvtxbs.basis.glb"},
        "id_handle_mapping": {"95": "003_cracker_box", "56": "024_bowl"},
        "objects": objects,
    }


STATIC = raw([
    {"object_id": 1, "semantic_id": 95, "translation": [0.0, 0.9, 0.0],
     "rotation": [0.0, 0.0, 0.0, 1.0]},
    {"object_id": 2, "semantic_id": 56, "translation": [5.0, 0.9, 5.0],
     "rotation": [0.0, 0.0, 0.0, 1.0]},
])


def test_scene_path_is_rewritten_to_the_mounted_form():
    """The collector's own path has no `data` component, so rebase_collector_path
    cannot map it and discovery would reject the layout outright."""
    out = mod.convert_static(STATIC, "00829-QaLdnwvtxbs")
    assert out["scene"]["scene_path"] == (
        "/app/data/scene_datasets/hm3d/val/00829-QaLdnwvtxbs/QaLdnwvtxbs.basis.glb"
    )
    assert "hm3d_annotated_basis" in out["scene"]["scene_dataset_config"]


def test_static_gets_one_anchor_per_object():
    out = mod.convert_static(STATIC, "s")
    anchors = [o["anchor"]["object_id"] for o in out["objects"]]
    assert anchors == ["surface_003_cracker_box", "surface_024_bowl"]
    assert out["authoring"]["layout_type"] == "static"
    assert out["authoring"]["relocated_semantic_ids"] == []


def test_in_anchor_keeps_every_object_on_its_own_anchor():
    moved = raw([
        {"object_id": 1, "semantic_id": 95, "translation": [0.6, 0.9, 0.2],
         "rotation": [0.0, 0.0, 0.0, 1.0]},
        {"object_id": 2, "semantic_id": 56, "translation": [5.4, 0.9, 5.1],
         "rotation": [0.0, 0.0, 0.0, 1.0]},
    ])
    out = mod.convert_dynamic(moved, STATIC, "s", "in_anchor", 1, "0116.json")
    static_anchors = {o["semantic_id"]: o["anchor"]["object_id"]
                      for o in mod.convert_static(STATIC, "s")["objects"]}
    for obj in out["objects"]:
        assert obj["anchor"]["object_id"] == static_anchors[obj["semantic_id"]]
    assert out["authoring"]["layout_type"] == "in_anchor"
    assert sorted(out["authoring"]["relocated_semantic_ids"]) == [56, 95]


def test_cross_anchor_gives_every_object_a_different_anchor():
    """The schema rejects a cross_anchor layout where anything kept its anchor,
    so the conversion has to actually change all of them."""
    swapped = raw([
        {"object_id": 1, "semantic_id": 95, "translation": [5.05, 0.9, 5.0],
         "rotation": [0.0, 0.0, 0.0, 1.0]},
        {"object_id": 2, "semantic_id": 56, "translation": [0.05, 0.9, 0.0],
         "rotation": [0.0, 0.0, 0.0, 1.0]},
    ])
    out = mod.convert_dynamic(swapped, STATIC, "s", "cross_anchor", 1, "0128-1.json")
    static_anchors = {o["semantic_id"]: o["anchor"]["object_id"]
                      for o in mod.convert_static(STATIC, "s")["objects"]}
    for obj in out["objects"]:
        assert obj["anchor"]["object_id"] != static_anchors[obj["semantic_id"]]
    # Landing on another object's surface should be NAMED as that surface.
    by_id = {o["semantic_id"]: o["anchor"]["object_id"] for o in out["objects"]}
    assert by_id[95] == "surface_024_bowl"
    assert by_id[56] == "surface_003_cracker_box"


def test_an_object_that_lands_nowhere_known_gets_its_own_anchor():
    far = raw([
        {"object_id": 1, "semantic_id": 95, "translation": [40.0, 0.9, 40.0],
         "rotation": [0.0, 0.0, 0.0, 1.0]},
        {"object_id": 2, "semantic_id": 56, "translation": [0.05, 0.9, 0.0],
         "rotation": [0.0, 0.0, 0.0, 1.0]},
    ])
    out = mod.convert_dynamic(far, STATIC, "s", "cross_anchor", 1, "x.json")
    by_id = {o["semantic_id"]: o["anchor"]["object_id"] for o in out["objects"]}
    assert by_id[95] == "surface_moved_003_cracker_box"


def test_displacement_disagreeing_with_the_declared_type_is_reported_not_silent():
    """The collector's directory is the author's intent and stays authoritative,
    but a 9 m 'in_anchor' move is worth saying out loud."""
    far = raw([
        {"object_id": 1, "semantic_id": 95, "translation": [9.0, 0.9, 0.0],
         "rotation": [0.0, 0.0, 0.0, 1.0]},
        {"object_id": 2, "semantic_id": 56, "translation": [5.1, 0.9, 5.0],
         "rotation": [0.0, 0.0, 0.0, 1.0]},
    ])
    lines = mod.displacement_report(STATIC, far, "in_anchor")
    assert any("disagrees" in line for line in lines)

    # The same layout declared cross_anchor: object 95 moved 9 m and agrees,
    # object 56 moved 0.1 m and does not -- the report names the object, not
    # just the layout.
    cross = mod.displacement_report(STATIC, far, "cross_anchor")
    note = next(line for line in cross if "disagrees" in line)
    assert "024_bowl" in note and "003_cracker_box" not in note

    both_far = raw([
        {"object_id": 1, "semantic_id": 95, "translation": [9.0, 0.9, 0.0],
         "rotation": [0.0, 0.0, 0.0, 1.0]},
        {"object_id": 2, "semantic_id": 56, "translation": [-4.0, 0.9, 5.0],
         "rotation": [0.0, 0.0, 0.0, 1.0]},
    ])
    assert not any("disagrees" in line
                   for line in mod.displacement_report(STATIC, both_far, "cross_anchor"))


# ------------------------------------------------- what the detector can see


# The handles the collector actually places in 00829-QaLdnwvtxbs, after the
# scissors substitution. `YCB_TARGET_LABELS` is a catalogue of every handle the
# benchmark can name, which is a larger set -- an entry there is not a promise
# that the object is in any scene.
PLACED_HANDLES = (
    "003_cracker_box",
    "005_tomato_soup_can",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "029_plate",
)


def _experiment_config():
    import yaml
    from pathlib import Path

    return yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "configs/experiment/ycb_authored_nav.yaml")
        .read_text(encoding="utf-8")
    )


def test_the_ycb_experiment_offers_every_placed_target_as_a_class():
    """The detector's vocabulary is DEFAULT_VOCABULARY plus the EPISODE's
    target, so a mapping run for one object has no class for the others and
    cannot map them at all -- which is why a map built while chasing the bowl
    contained no soup can. A multi-target benchmark needs every target present
    from the start."""
    from osg.core.config import YCB_TARGET_LABELS

    vocab = {v.lower() for v in _experiment_config()["detector"]["vocabulary"]}
    missing = sorted(
        YCB_TARGET_LABELS[handle].lower()
        for handle in PLACED_HANDLES
        if YCB_TARGET_LABELS[handle].lower() not in vocab
    )
    assert not missing, f"targets absent from the detector vocabulary: {missing}"


def test_the_ycb_experiment_trades_recall_against_a_promiscuous_label():
    """imgsz is a precision decision, not only a recall one.

    Recall at the 0.30 gate over 20 authored viewpoints rises with resolution --
    the cracker box goes 0.20 / 0.70 / 0.90 at 512 / 768 / 1280 -- but so does
    the number of things the detector calls a cracker box. Over 300 random
    navigable poses it fires above the gate on 0 / 2 / 28 non-boxes, and at 1280
    the accumulated map held fifteen "cracker box" tracks for a house with one.
    Anything at or above 960 reintroduces that; 512 costs too much recall.
    """
    assert 640 <= _experiment_config()["detector"]["imgsz"] <= 768
