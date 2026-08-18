from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from osg.core.config import YCB_TARGET_LABELS
from osg.sim.ycb_env import manifest_cache_key, prepare_ycb_benchmark
from osg.sim.ycb_layouts import (
    YCBLayoutError,
    discover_authored_layouts,
    load_authored_layout,
    rebase_collector_path,
)


HANDLE = "003_cracker_box"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture_roots(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    layout_root = tmp_path / "authoring"
    (data_root / "scene_datasets/hm3d/val").mkdir(parents=True)
    (data_root / "scene_datasets/hm3d/hm3d.scene_dataset_config.json").write_text(
        "{}", encoding="utf-8"
    )
    configs = data_root / "objects/ycb/configs"
    configs.mkdir(parents=True)
    for handle in YCB_TARGET_LABELS:
        (configs / f"{handle}.object_config.json").write_text("{}", encoding="utf-8")
    layout_root.mkdir()
    return data_root, layout_root


def _layout_value(
    scene_hash: str,
    *,
    layout_type: str = "static",
    layout_index: int | None = None,
    handle: str = HANDLE,
    semantic_id: int = 50001,
) -> dict:
    return {
        "scene": {
            "scene_path": (
                f"/app/data/scene_datasets/hm3d/val/00001-{scene_hash}/"
                f"{scene_hash}.basis.glb"
            ),
            "scene_dataset_config": (
                "/app/data/scene_datasets/hm3d/hm3d.scene_dataset_config.json"
            ),
        },
        "id_handle_mapping": {str(semantic_id): handle},
        "objects": [
            {
                "semantic_id": semantic_id,
                "translation": [1.0, 0.8, 2.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "anchor": {
                    "object_id": (
                        "other_table" if layout_type == "cross_anchor" else "table"
                    ),
                    "category": "table",
                },
            }
        ],
        "authoring": {
            "layout_type": layout_type,
            "layout_index": layout_index,
            "reference_static_config": (
                None if layout_type == "static" else "../../static_scene_config.json"
            ),
            "relocated_semantic_ids": [] if layout_type == "static" else [semantic_id],
        },
    }


def _add_scene(
    data_root: Path,
    layout_root: Path,
    scene_name: str,
    *,
    dynamic: tuple[str, int] | None = None,
) -> None:
    scene_hash = scene_name.split("-", 1)[1]
    mesh = data_root / f"scene_datasets/hm3d/val/00001-{scene_hash}/{scene_hash}.basis.glb"
    mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.write_bytes(b"mesh")
    static = _layout_value(scene_hash)
    _write_json(layout_root / scene_name / "static_scene_config.json", static)
    if dynamic:
        layout_type, index = dynamic
        value = _layout_value(scene_hash, layout_type=layout_type, layout_index=index)
        value["objects"][0]["translation"][0] = 3.0
        _write_json(
            layout_root
            / scene_name
            / "dynamic_scene_config"
            / layout_type
            / f"layout_{index:02d}.json",
            value,
        )


def _discover(data_root: Path, layout_root: Path, **overrides):
    values = {
        "scenes": ["*"],
        "layout_types": ["static"],
        "layout_indices": [1, 2, 3],
    }
    values.update(overrides)
    return discover_authored_layouts(
        data_root=data_root,
        layout_root=layout_root,
        target_labels=YCB_TARGET_LABELS,
        **values,
    )


def _benchmark_cfg(data_root: Path, layout_root: Path, cache_root: Path):
    return SimpleNamespace(
        ycb=SimpleNamespace(
            data_root=str(data_root),
            layout_root=str(layout_root),
            scenes=["*"],
            layout_types=["static"],
            layout_indices=[1, 2, 3],
            starts_per_target=1,
            seed=42,
            manifest_cache_dir=str(cache_root),
            target_labels=YCB_TARGET_LABELS,
            viewpoint_radii_m=[0.8, 1.2, 1.5, 2.0],
            viewpoint_angular_samples=24,
            viewpoint_max_snap_m=0.5,
            viewpoint_dedup_m=0.2,
            viewpoint_min_visible_pixels=20,
            start_min_geodesic_m=3.0,
            start_sample_attempts=2000,
        ),
        eval=SimpleNamespace(rgb_width=640, rgb_height=480, hfov_deg=79.0),
        agent=SimpleNamespace(camera_height=0.88, agent_radius=0.18),
    )


def test_rebase_collector_paths_and_canonical_labels(tmp_path):
    root = tmp_path / "mounted-data"
    assert rebase_collector_path(
        "/app/data/scene_datasets/hm3d/scene.glb", root
    ) == root / "scene_datasets/hm3d/scene.glb"
    assert rebase_collector_path("objects/ycb/configs", root) == root / "objects/ycb/configs"
    assert YCB_TARGET_LABELS == {
        "003_cracker_box": "cracker box",
        "005_tomato_soup_can": "tomato soup can",
        "011_banana": "banana",
        "019_pitcher_base": "pitcher",
        "024_bowl": "bowl",
        "025_mug": "mug",
        "029_plate": "plate",
        "037_scissors": "scissors",
    }


def test_wildcard_discovers_future_scenes_and_reports_incomplete(tmp_path):
    data_root, layout_root = _fixture_roots(tmp_path)
    _add_scene(data_root, layout_root, "00001-SceneOne")
    _add_scene(data_root, layout_root, "00001-SceneTwo")
    (layout_root / "00000-Incomplete").mkdir()

    found = _discover(data_root, layout_root)

    assert [layout.scene_name for layout in found.layouts] == [
        "00001-SceneOne",
        "00001-SceneTwo",
    ]
    assert found.skipped == (
        {"scene": "00000-Incomplete", "reason": "missing static_scene_config.json"},
    )


@pytest.mark.parametrize(
    ("selectors", "expected"),
    [
        (["SceneOne"], ["00001-SceneOne"]),
        (["00001-SceneOne"], ["00001-SceneOne"]),
        (["SceneOne", "SceneTwo"], ["00001-SceneOne", "00001-SceneTwo"]),
    ],
)
def test_single_and_list_scene_selectors(tmp_path, selectors, expected):
    data_root, layout_root = _fixture_roots(tmp_path)
    _add_scene(data_root, layout_root, "00001-SceneOne")
    _add_scene(data_root, layout_root, "00001-SceneTwo")
    found = _discover(data_root, layout_root, scenes=selectors)
    assert [layout.scene_name for layout in found.layouts] == expected


def test_explicit_incomplete_scene_and_layout_fail(tmp_path):
    data_root, layout_root = _fixture_roots(tmp_path)
    (layout_root / "00000-Incomplete").mkdir()
    with pytest.raises(YCBLayoutError, match="missing static_scene_config"):
        _discover(data_root, layout_root, scenes=["00000-Incomplete"])

    _add_scene(data_root, layout_root, "00001-SceneOne")
    with pytest.raises(YCBLayoutError, match="missing in_anchor layout index 2"):
        _discover(
            data_root,
            layout_root,
            scenes=["SceneOne"],
            layout_types=["in_anchor"],
            layout_indices=[2],
        )


def test_dynamic_layout_selection_and_static_consistency(tmp_path):
    data_root, layout_root = _fixture_roots(tmp_path)
    _add_scene(
        data_root,
        layout_root,
        "00001-SceneOne",
        dynamic=("cross_anchor", 2),
    )
    found = _discover(
        data_root,
        layout_root,
        scenes=["SceneOne"],
        layout_types=["cross_anchor"],
        layout_indices=[2],
    )
    assert [(item.layout_type, item.layout_index, item.layout_id) for item in found.layouts] == [
        ("cross_anchor", 2, "cross_anchor_02")
    ]

    dynamic_path = found.layouts[0].layout_path
    value = json.loads(dynamic_path.read_text(encoding="utf-8"))
    value["id_handle_mapping"] = {"50001": "025_mug"}
    _write_json(dynamic_path, value)
    static = load_authored_layout(
        layout_root / "00001-SceneOne/static_scene_config.json",
        layout_root=layout_root,
        data_root=data_root,
        target_labels=YCB_TARGET_LABELS,
    )
    with pytest.raises(YCBLayoutError, match="dynamic target set differs"):
        load_authored_layout(
            dynamic_path,
            layout_root=layout_root,
            data_root=data_root,
            target_labels=YCB_TARGET_LABELS,
            static_layout=static,
        )


def test_static_in_anchor_and_cross_anchor_layout_types(tmp_path):
    data_root, layout_root = _fixture_roots(tmp_path)
    _add_scene(
        data_root,
        layout_root,
        "00001-SceneOne",
        dynamic=("in_anchor", 1),
    )
    _add_scene(
        data_root,
        layout_root,
        "00001-SceneOne",
        dynamic=("cross_anchor", 1),
    )
    found = _discover(
        data_root,
        layout_root,
        scenes=["SceneOne"],
        layout_types=["static", "in_anchor", "cross_anchor"],
        layout_indices=[1],
    )
    assert {(item.layout_type, item.layout_index) for item in found.layouts} == {
        ("static", None),
        ("in_anchor", 1),
        ("cross_anchor", 1),
    }


def test_invalid_pose_is_rejected(tmp_path):
    data_root, layout_root = _fixture_roots(tmp_path)
    _add_scene(data_root, layout_root, "00001-SceneOne")
    path = layout_root / "00001-SceneOne/static_scene_config.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["objects"][0]["rotation"] = [0.0, 0.0, 0.0, 2.0]
    _write_json(path, value)
    with pytest.raises(YCBLayoutError, match="quaternion norm"):
        _discover(data_root, layout_root, scenes=["SceneOne"])


def test_manifest_cache_reuse_and_layout_hash_invalidation(tmp_path, monkeypatch):
    data_root, layout_root = _fixture_roots(tmp_path)
    _add_scene(data_root, layout_root, "00001-SceneOne")
    cfg = _benchmark_cfg(data_root, layout_root, tmp_path / "cache")
    calls = []

    def fake_manifest(layout, generator_cfg):
        calls.append(layout.layout_sha256)
        return {
            "schema_version": 1,
            "generator_version": 1,
            "cache_key": manifest_cache_key(layout, generator_cfg),
            "layout": {"scene": layout.scene_name, "layout_id": layout.layout_id},
            "episodes": [],
        }

    monkeypatch.setattr("osg.sim.ycb_env.generate_manifest", fake_manifest)
    first = prepare_ycb_benchmark(cfg)
    second = prepare_ycb_benchmark(cfg)
    assert len(calls) == 1
    assert first.cache_files == second.cache_files

    layout_path = layout_root / "00001-SceneOne/static_scene_config.json"
    changed = json.loads(layout_path.read_text(encoding="utf-8"))
    changed["objects"][0]["translation"][0] = 1.25
    _write_json(layout_path, changed)
    third = prepare_ycb_benchmark(cfg)
    assert len(calls) == 2
    assert third.cache_files != first.cache_files
