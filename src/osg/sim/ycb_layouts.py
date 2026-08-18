"""Discovery and validation for habitat-data-collector YCB authoring layouts.

This module deliberately has no Habitat dependency. Scene selection, schema
validation, path rebasing, and manifest cache decisions stay unit-testable on
machines without the simulator or licensed HM3D assets.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


VALID_LAYOUT_TYPES = ("static", "in_anchor", "cross_anchor")


class YCBLayoutError(ValueError):
    """An authored layout or selection cannot be used for the benchmark."""


@dataclass(frozen=True)
class AuthoredObject:
    semantic_id: int
    handle: str
    label: str
    translation: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]
    anchor_object_id: str
    anchor_category: str


@dataclass(frozen=True)
class AuthoredLayout:
    scene_name: str
    layout_type: str
    layout_index: int | None
    layout_path: Path
    layout_relative_path: str
    layout_sha256: str
    layout_id: str
    scene_mesh: Path
    scene_mesh_relative_path: str
    scene_dataset_config: Path
    scene_dataset_config_relative_path: str
    objects_dir: Path
    id_handle_mapping: Tuple[Tuple[int, str], ...]
    objects: Tuple[AuthoredObject, ...]


@dataclass(frozen=True)
class LayoutDiscovery:
    layouts: Tuple[AuthoredLayout, ...]
    skipped: Tuple[Dict[str, str], ...]
    wildcard: bool


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise YCBLayoutError(f"layout does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise YCBLayoutError(f"invalid layout JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise YCBLayoutError(f"layout root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebase_collector_path(serialized: str, data_root: Path) -> Path:
    """Map a serialized collector path (usually `/app/data/...`) to a mount.

    Authoring files intentionally retain the path used by the collector. The
    nav image uses a different mount point, so retain only the suffix below the
    collector's `data` directory. Already-relative paths are also supported.
    """
    if not serialized or not isinstance(serialized, str):
        raise YCBLayoutError("serialized asset path must be a non-empty string")
    path = Path(serialized)
    parts = path.parts
    if "data" in parts:
        # Use the last `data` component in case a parent directory also happens
        # to contain that word.
        index = max(i for i, part in enumerate(parts) if part == "data")
        suffix = parts[index + 1 :]
        if suffix:
            return data_root.joinpath(*suffix)
    if not path.is_absolute():
        return data_root / path
    raise YCBLayoutError(
        f"cannot rebase asset path outside a collector data tree: {serialized}"
    )


def _relative_to(path: Path, root: Path, description: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise YCBLayoutError(f"{description} is outside {root}: {path}") from exc


def _float_tuple(value: Any, length: int, field: str, path: Path) -> tuple:
    if not isinstance(value, list) or len(value) != length:
        raise YCBLayoutError(f"{path}: {field} must contain {length} numbers")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise YCBLayoutError(f"{path}: {field} contains a non-number") from exc
    if not all(math.isfinite(item) for item in result):
        raise YCBLayoutError(f"{path}: {field} contains a non-finite value")
    return result


def _layout_identity(data: Mapping[str, Any], path: Path) -> Tuple[str, int | None]:
    authoring = data.get("authoring")
    if not isinstance(authoring, dict):
        raise YCBLayoutError(f"{path}: missing authoring metadata")
    layout_type = str(authoring.get("layout_type", ""))
    if layout_type not in VALID_LAYOUT_TYPES:
        raise YCBLayoutError(f"{path}: unsupported layout type {layout_type!r}")
    raw_index = authoring.get("layout_index")
    layout_index = None if raw_index is None else int(raw_index)
    if layout_type == "static" and layout_index is not None:
        raise YCBLayoutError(f"{path}: static layout cannot have an index")
    if layout_type != "static" and layout_index not in (1, 2, 3):
        raise YCBLayoutError(f"{path}: dynamic layout index must be 1, 2, or 3")
    if layout_type == "static" and path.name != "static_scene_config.json":
        raise YCBLayoutError(f"{path}: static layout has an unexpected filename")
    if layout_type != "static":
        expected_name = f"layout_{layout_index:02d}.json"
        if path.parent.name != layout_type or path.name != expected_name:
            raise YCBLayoutError(
                f"{path}: authoring identity requires {layout_type}/{expected_name}"
            )
    return layout_type, layout_index


def load_authored_layout(
    path: Path,
    *,
    layout_root: Path,
    data_root: Path,
    target_labels: Mapping[str, str],
    static_layout: AuthoredLayout | None = None,
) -> AuthoredLayout:
    data = _load_json(path)
    # Dynamic paths are <scene>/dynamic_scene_config/<type>/<file>.
    try:
        scene_name = path.relative_to(layout_root).parts[0]
    except ValueError as exc:
        raise YCBLayoutError(f"layout is outside layout_root {layout_root}: {path}") from exc

    layout_type, layout_index = _layout_identity(data, path)
    scene = data.get("scene")
    if not isinstance(scene, dict):
        raise YCBLayoutError(f"{path}: missing scene object")
    mesh_raw = scene.get("scene_path")
    dataset_raw = scene.get("scene_dataset_config") or scene.get("scene_config_path")
    scene_mesh = rebase_collector_path(mesh_raw, data_root)
    dataset_config = rebase_collector_path(dataset_raw, data_root)
    objects_dir = data_root / "objects" / "ycb" / "configs"
    required_paths = (
        (scene_mesh, "HM3D scene mesh"),
        (dataset_config, "HM3D scene dataset config"),
        (objects_dir, "YCB object config directory"),
    )
    for required, description in required_paths:
        if not required.exists():
            raise YCBLayoutError(f"{path}: missing {description}: {required}")

    mapping_raw = data.get("id_handle_mapping")
    objects_raw = data.get("objects")
    if not isinstance(mapping_raw, dict) or not isinstance(objects_raw, list):
        raise YCBLayoutError(f"{path}: id_handle_mapping and objects are required")
    try:
        id_to_handle = {int(key): str(value) for key, value in mapping_raw.items()}
    except (TypeError, ValueError) as exc:
        raise YCBLayoutError(f"{path}: semantic IDs must be integers") from exc
    unsupported_handles = sorted(set(id_to_handle.values()) - set(target_labels))
    if unsupported_handles:
        raise YCBLayoutError(
            f"{path}: unsupported handles in id_handle_mapping: {unsupported_handles}"
        )
    if len(set(id_to_handle.values())) != len(id_to_handle):
        raise YCBLayoutError(f"{path}: id_handle_mapping contains duplicate handles")

    objects: List[AuthoredObject] = []
    seen_ids: set[int] = set()
    for index, value in enumerate(objects_raw):
        if not isinstance(value, dict) or "semantic_id" not in value:
            raise YCBLayoutError(f"{path}: objects[{index}] is malformed")
        semantic_id = int(value["semantic_id"])
        if semantic_id in seen_ids:
            raise YCBLayoutError(f"{path}: duplicate semantic ID {semantic_id}")
        seen_ids.add(semantic_id)
        handle = id_to_handle.get(semantic_id)
        if handle is None:
            raise YCBLayoutError(f"{path}: semantic ID {semantic_id} has no handle")
        if handle not in target_labels:
            raise YCBLayoutError(f"{path}: unsupported YCB handle {handle!r}")
        anchor = value.get("anchor")
        if not isinstance(anchor, dict):
            raise YCBLayoutError(f"{path}: objects[{index}] is missing anchor metadata")
        anchor_object_id = str(anchor.get("object_id", "")).strip()
        anchor_category = str(anchor.get("category", "")).strip()
        if not anchor_object_id or not anchor_category:
            raise YCBLayoutError(f"{path}: objects[{index}] has invalid anchor metadata")
        object_config = objects_dir / f"{handle}.object_config.json"
        if not object_config.is_file():
            raise YCBLayoutError(f"{path}: missing object config {object_config}")
        rotation = _float_tuple(value.get("rotation"), 4, f"objects[{index}].rotation", path)
        rotation_norm = math.sqrt(sum(component * component for component in rotation))
        if abs(rotation_norm - 1.0) > 1e-3:
            raise YCBLayoutError(
                f"{path}: objects[{index}].rotation quaternion norm is {rotation_norm:.6f}"
            )
        objects.append(
            AuthoredObject(
                semantic_id=semantic_id,
                handle=handle,
                label=str(target_labels[handle]).strip().lower(),
                translation=_float_tuple(
                    value.get("translation"),
                    3,
                    f"objects[{index}].translation",
                    path,
                ),
                rotation=rotation,
                anchor_object_id=anchor_object_id,
                anchor_category=anchor_category,
            )
        )
    if not objects:
        raise YCBLayoutError(f"{path}: layout contains no placed YCB objects")
    object_ids = {obj.semantic_id for obj in objects}

    authoring = data["authoring"]
    relocated_raw = authoring.get("relocated_semantic_ids")
    if not isinstance(relocated_raw, list):
        raise YCBLayoutError(f"{path}: authoring.relocated_semantic_ids must be an array")
    try:
        relocated_ids = {int(value) for value in relocated_raw}
    except (TypeError, ValueError) as exc:
        raise YCBLayoutError(
            f"{path}: authoring.relocated_semantic_ids contains an invalid ID"
        ) from exc
    reference = authoring.get("reference_static_config")
    if layout_type == "static":
        if relocated_ids or reference is not None:
            raise YCBLayoutError(
                f"{path}: static layout cannot reference or relocate another layout"
            )

    if static_layout is not None:
        static_pairs = {(obj.semantic_id, obj.handle) for obj in static_layout.objects}
        dynamic_pairs = {(obj.semantic_id, obj.handle) for obj in objects}
        if dynamic_pairs != static_pairs:
            raise YCBLayoutError(
                f"{path}: dynamic target set differs from {static_layout.layout_path}"
            )
        if tuple(sorted(id_to_handle.items())) != static_layout.id_handle_mapping:
            raise YCBLayoutError(
                f"{path}: dynamic semantic-ID mapping differs from static layout"
            )
        if scene_mesh.resolve() != static_layout.scene_mesh.resolve():
            raise YCBLayoutError(f"{path}: dynamic scene mesh differs from static layout")
        if dataset_config.resolve() != static_layout.scene_dataset_config.resolve():
            raise YCBLayoutError(
                f"{path}: dynamic scene dataset config differs from static layout"
            )
        if relocated_ids != object_ids:
            raise YCBLayoutError(
                f"{path}: dynamic relocated IDs must exactly match placed object IDs"
            )
        if not isinstance(reference, str) or not reference.strip():
            raise YCBLayoutError(f"{path}: dynamic layout is missing reference_static_config")
        referenced_path = (path.parent / reference).resolve()
        if referenced_path != static_layout.layout_path.resolve():
            raise YCBLayoutError(
                f"{path}: reference_static_config does not point to "
                f"{static_layout.layout_path}"
            )
        static_by_id = {obj.semantic_id: obj for obj in static_layout.objects}
        for obj in objects:
            same_anchor = obj.anchor_object_id == static_by_id[obj.semantic_id].anchor_object_id
            if layout_type == "in_anchor" and not same_anchor:
                raise YCBLayoutError(
                    f"{path}: {obj.semantic_id} moved to a different anchor"
                )
            if layout_type == "cross_anchor" and same_anchor:
                raise YCBLayoutError(
                    f"{path}: {obj.semantic_id} did not move to a different anchor"
                )

    layout_id = "static" if layout_type == "static" else f"{layout_type}_{layout_index:02d}"
    return AuthoredLayout(
        scene_name=scene_name,
        layout_type=layout_type,
        layout_index=layout_index,
        layout_path=path,
        layout_relative_path=_relative_to(path, layout_root, "layout"),
        layout_sha256=sha256_file(path),
        layout_id=layout_id,
        scene_mesh=scene_mesh,
        scene_mesh_relative_path=_relative_to(scene_mesh, data_root, "scene mesh"),
        scene_dataset_config=dataset_config,
        scene_dataset_config_relative_path=_relative_to(
            dataset_config, data_root, "dataset config"
        ),
        objects_dir=objects_dir,
        id_handle_mapping=tuple(sorted(id_to_handle.items())),
        objects=tuple(sorted(objects, key=lambda obj: obj.semantic_id)),
    )


def _resolve_scene_names(layout_root: Path, selectors: Sequence[str]) -> Tuple[List[str], bool]:
    available = sorted(path.name for path in layout_root.iterdir() if path.is_dir())
    wildcard = len(selectors) == 1 and selectors[0] == "*"
    if wildcard:
        return available, True
    if not selectors or "*" in selectors:
        raise YCBLayoutError("scenes must be ['*'] or an explicit non-empty list")
    resolved: List[str] = []
    for selector in selectors:
        matches = [
            name for name in available
            if name == selector or name.startswith(f"{selector}-") or name.endswith(f"-{selector}")
        ]
        if len(matches) != 1:
            detail = "not found" if not matches else f"ambiguous: {matches}"
            raise YCBLayoutError(f"scene {selector!r} {detail} under {layout_root}")
        if matches[0] not in resolved:
            resolved.append(matches[0])
    return resolved, False


def _dynamic_file(root: Path, layout_index: int) -> Path | None:
    preferred = root / f"layout_{layout_index:02d}.json"
    if preferred.is_file():
        return preferred
    candidates: List[Path] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = _load_json(path)
            _, found_index = _layout_identity(data, path)
        except YCBLayoutError:
            continue
        if found_index == layout_index:
            candidates.append(path)
    if len(candidates) > 1:
        raise YCBLayoutError(f"multiple layout files claim index {layout_index}: {candidates}")
    return candidates[0] if candidates else None


def discover_authored_layouts(
    *,
    layout_root: Path,
    data_root: Path,
    scenes: Sequence[str],
    layout_types: Sequence[str],
    layout_indices: Sequence[int],
    target_labels: Mapping[str, str],
) -> LayoutDiscovery:
    if not layout_root.is_dir():
        raise YCBLayoutError(f"layout root does not exist: {layout_root}")
    if not data_root.is_dir():
        raise YCBLayoutError(f"collector data root does not exist: {data_root}")
    invalid_types = sorted(set(layout_types) - set(VALID_LAYOUT_TYPES))
    if invalid_types or not layout_types:
        raise YCBLayoutError(f"invalid layout_types: {invalid_types or list(layout_types)}")
    invalid_indices = sorted(index for index in layout_indices if index not in (1, 2, 3))
    if invalid_indices:
        raise YCBLayoutError(f"layout indices must be 1, 2, or 3: {invalid_indices}")

    scene_names, wildcard = _resolve_scene_names(layout_root, list(scenes))
    layouts: List[AuthoredLayout] = []
    skipped: List[Dict[str, str]] = []
    for scene_name in scene_names:
        scene_root = layout_root / scene_name
        static_path = scene_root / "static_scene_config.json"
        if not static_path.is_file():
            message = "missing static_scene_config.json"
            if wildcard:
                skipped.append({"scene": scene_name, "reason": message})
                continue
            raise YCBLayoutError(f"{scene_name}: {message}")
        try:
            static_layout = load_authored_layout(
                static_path,
                layout_root=layout_root,
                data_root=data_root,
                target_labels=target_labels,
            )
        except YCBLayoutError as exc:
            if wildcard:
                skipped.append({"scene": scene_name, "reason": str(exc)})
                continue
            raise
        if "static" in layout_types:
            layouts.append(static_layout)
        for layout_type in ("in_anchor", "cross_anchor"):
            if layout_type not in layout_types:
                continue
            dynamic_root = scene_root / "dynamic_scene_config" / layout_type
            for layout_index in layout_indices:
                dynamic_path = (
                    _dynamic_file(dynamic_root, int(layout_index))
                    if dynamic_root.is_dir()
                    else None
                )
                if dynamic_path is None:
                    message = f"missing {layout_type} layout index {layout_index}"
                    if wildcard:
                        skipped.append({"scene": scene_name, "reason": message})
                        continue
                    raise YCBLayoutError(f"{scene_name}: {message}")
                layouts.append(
                    load_authored_layout(
                        dynamic_path,
                        layout_root=layout_root,
                        data_root=data_root,
                        target_labels=target_labels,
                        static_layout=static_layout,
                    )
                )
    if not layouts:
        reasons = "; ".join(f"{item['scene']}: {item['reason']}" for item in skipped)
        suffix = f": {reasons}" if reasons else ""
        raise YCBLayoutError(f"no usable authored layouts found{suffix}")
    layouts.sort(key=lambda item: (item.scene_name, item.layout_type, item.layout_index or 0))
    return LayoutDiscovery(tuple(layouts), tuple(skipped), wildcard)
