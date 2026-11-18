"""Swapping an undetectable asset into an authored slot.

The scissors score 0.00 against eight candidate labels at every resolution
(scripts/probe_ycb_detection.py), so their episodes measure asset coverage
rather than dynamic-scene handling. The substitution keeps the pose and swaps
the mesh, and these tests pin the parts that need no simulator.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "author_substitute_layout",
    Path(__file__).resolve().parents[2] / "scripts" / "author_substitute_layout.py",
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_replacements_parse_as_old_equals_new():
    assert mod.parse_replacements(["037_scissors=021_bleach_cleanser"]) == {
        "037_scissors": "021_bleach_cleanser"
    }
    with pytest.raises(SystemExit):
        mod.parse_replacements(["037_scissors"])


def test_every_layout_of_the_scene_is_rewritten_not_only_the_static_one(tmp_path):
    """A dynamic layout that still carries the old handle would relocate a
    different object from the one the static map holds -- the two passes would
    disagree about what the scene contains."""
    scene = tmp_path / "00829-QaLdnwvtxbs"
    (scene / "dynamic_scene_config" / "in_anchor").mkdir(parents=True)
    (scene / "dynamic_scene_config" / "cross_anchor").mkdir(parents=True)
    (scene / "static_scene_config.json").write_text("{}")
    for kind in ("in_anchor", "cross_anchor"):
        for index in (1, 2, 3):
            (scene / "dynamic_scene_config" / kind / f"layout_{index:02d}.json").write_text("{}")

    files = mod.layout_files(scene)
    assert len(files) == 7
    assert files[0].name == "static_scene_config.json"
    assert {path.name for path in files[1:]} == {
        f"layout_{index:02d}.json" for index in (1, 2, 3)
    }


def test_the_source_root_is_never_the_destination(tmp_path, monkeypatch, capsys):
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "static_scene_config.json").write_text(json.dumps({"id_handle_mapping": {}}))
    monkeypatch.setattr(
        "sys.argv",
        ["author_substitute_layout.py", "--source", str(tmp_path), "--out", str(tmp_path),
         "--scene", "scene", "--replace", "a=b"],
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert "refusing to write over the source" in str(excinfo.value)
