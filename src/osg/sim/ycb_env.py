"""Habitat environment and deterministic episode cache for authored YCB scenes."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .habitat_env import HabitatObjectNavEnv, make_objectnav_config
from .ycb_layouts import (
    AuthoredLayout,
    LayoutDiscovery,
    YCBLayoutError,
    discover_authored_layouts,
)


MANIFEST_SCHEMA_VERSION = 1
MANIFEST_GENERATOR_VERSION = 1


@dataclass(frozen=True)
class PreparedYCB:
    discovery: LayoutDiscovery
    manifests: Tuple[Dict[str, Any], ...]
    cache_files: Tuple[Path, ...]


def _plain(value: Any) -> Any:
    """Convert OmegaConf/list-like values to JSON-stable Python values."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def manifest_cache_payload(layout: AuthoredLayout, cfg) -> Dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_version": MANIFEST_GENERATOR_VERSION,
        "scene": layout.scene_name,
        "layout_id": layout.layout_id,
        "layout_sha256": layout.layout_sha256,
        "seed": int(cfg.ycb.seed),
        "starts_per_target": int(cfg.ycb.starts_per_target),
        "target_labels": _plain(cfg.ycb.target_labels),
        "viewpoint_radii_m": [float(x) for x in cfg.ycb.viewpoint_radii_m],
        "viewpoint_angular_samples": int(cfg.ycb.viewpoint_angular_samples),
        "viewpoint_max_snap_m": float(cfg.ycb.viewpoint_max_snap_m),
        "viewpoint_dedup_m": float(cfg.ycb.viewpoint_dedup_m),
        "viewpoint_min_visible_pixels": int(cfg.ycb.viewpoint_min_visible_pixels),
        "start_min_geodesic_m": float(cfg.ycb.start_min_geodesic_m),
        "start_sample_attempts": int(cfg.ycb.start_sample_attempts),
        "camera": {
            "width": int(cfg.eval.rgb_width),
            "height": int(cfg.eval.rgb_height),
            "hfov_deg": float(cfg.eval.hfov_deg),
            "height_m": float(cfg.agent.camera_height),
        },
        "agent_radius": float(cfg.agent.agent_radius),
    }


def manifest_cache_key(layout: AuthoredLayout, cfg) -> str:
    encoded = json.dumps(
        manifest_cache_payload(layout, cfg), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_cache_path(layout: AuthoredLayout, cfg) -> Path:
    key = manifest_cache_key(layout, cfg)
    return (
        Path(str(cfg.ycb.manifest_cache_dir))
        / layout.scene_name
        / layout.layout_id
        / f"{key}.json"
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _load_cached_manifest(path: Path, expected_key: str) -> Dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("cache_key") != expected_key:
        return None
    if value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return None
    return value


def _yaw_facing(source: Sequence[float], target: Sequence[float]) -> List[float]:
    dx = float(target[0]) - float(source[0])
    dz = float(target[2]) - float(source[2])
    yaw = math.atan2(-dx, -dz)
    return [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]


def _random_yaw(rng: np.random.Generator) -> List[float]:
    yaw = float(rng.uniform(-math.pi, math.pi))
    return [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]


def _finite_point(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    point = np.asarray(value, dtype=np.float32)
    if point.shape != (3,) or not np.isfinite(point).all():
        return None
    return point


def _geodesic_distance(
    pathfinder, start: Sequence[float], goals: Iterable[Sequence[float]]
) -> float:
    import habitat_sim

    best = math.inf
    for goal in goals:
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(start, dtype=np.float32)
        path.requested_end = np.asarray(goal, dtype=np.float32)
        if pathfinder.find_path(path):
            best = min(best, float(path.geodesic_distance))
    return best


def _template_handle(manager, desired: str) -> str:
    candidates = []
    for handle in manager.get_file_template_handles():
        basename = Path(str(handle)).name
        stem = basename.split(".object_config", 1)[0].split(".", 1)[0]
        if stem == desired or desired in basename:
            candidates.append(str(handle))
    if not candidates:
        raise YCBLayoutError(f"YCB template {desired!r} was not loaded")
    candidates.sort(key=lambda item: (len(item), item))
    return candidates[0]


def inject_layout_objects(sim, layout: AuthoredLayout) -> List[Any]:
    """Replace rigid objects in a Habitat simulator with one authored layout."""
    import habitat_sim
    import magnum as mn

    object_manager = sim.get_rigid_object_manager()
    object_manager.remove_all_objects()
    template_manager = sim.get_object_template_manager()
    template_manager.load_configs(str(layout.objects_dir))

    added = []
    for authored in layout.objects:
        source_handle = _template_handle(template_manager, authored.handle)
        attributes = template_manager.get_template_by_handle(source_handle)
        attributes.semantic_id = int(authored.semantic_id)
        runtime_handle = f"ycb_authored::{authored.handle}::{authored.semantic_id}"
        template_manager.register_template(attributes, runtime_handle)
        rigid = object_manager.add_object_by_template_handle(runtime_handle)
        rigid.translation = mn.Vector3(*authored.translation)
        rigid.rotation = mn.Quaternion(
            mn.Vector3(*authored.rotation[:3]), authored.rotation[3]
        )
        try:
            rigid.semantic_id = int(authored.semantic_id)
        except AttributeError:
            pass
        rigid.motion_type = habitat_sim.physics.MotionType.STATIC
        added.append(rigid)
    return added


class _ManifestSimulator:
    def __init__(self, layout: AuthoredLayout, cfg) -> None:
        import habitat_sim
        import magnum as mn

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(layout.scene_mesh)
        sim_cfg.scene_dataset_config_file = str(layout.scene_dataset_config)
        sim_cfg.gpu_device_id = 0
        sim_cfg.enable_physics = True

        semantic = habitat_sim.CameraSensorSpec()
        semantic.uuid = "semantic"
        semantic.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        semantic.resolution = [int(cfg.eval.rgb_height), int(cfg.eval.rgb_width)]
        semantic.position = [0.0, float(cfg.agent.camera_height), 0.0]
        semantic.hfov = mn.Deg(float(cfg.eval.hfov_deg))

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = float(cfg.agent.camera_height)
        agent_cfg.radius = float(cfg.agent.agent_radius)
        agent_cfg.sensor_specifications = [semantic]
        self.sim = habitat_sim.Simulator(
            habitat_sim.Configuration(sim_cfg, [agent_cfg])
        )
        self.agent = self.sim.initialize_agent(0)
        if not self.sim.pathfinder.is_loaded:
            self.sim.close()
            raise YCBLayoutError(f"scene has no loaded navmesh: {layout.scene_mesh}")
        inject_layout_objects(self.sim, layout)

    def close(self) -> None:
        self.sim.close()

    def semantic_at(self, position: Sequence[float], rotation: Sequence[float]) -> np.ndarray:
        from habitat_sim.utils.common import quat_from_coeffs

        state = self.agent.get_state()
        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = quat_from_coeffs(np.asarray(rotation, dtype=np.float32))
        state.sensor_states = {}
        self.agent.set_state(state, reset_sensors=True)
        return np.asarray(self.sim.get_sensor_observations()["semantic"])


def _viewpoints_for_object(simulator: _ManifestSimulator, authored, cfg) -> List[Dict[str, Any]]:
    pathfinder = simulator.sim.pathfinder
    target = np.asarray(authored.translation, dtype=np.float32)
    viewpoints: List[Dict[str, Any]] = []
    samples = int(cfg.ycb.viewpoint_angular_samples)
    if samples < 4:
        raise YCBLayoutError("ycb.viewpoint_angular_samples must be at least 4")
    for radius in [float(value) for value in cfg.ycb.viewpoint_radii_m]:
        for sample in range(samples):
            angle = 2.0 * math.pi * sample / samples
            requested = np.array(
                [
                    target[0] + radius * math.cos(angle),
                    target[1],
                    target[2] + radius * math.sin(angle),
                ],
                dtype=np.float32,
            )
            snapped = _finite_point(pathfinder.snap_point(requested))
            if snapped is None:
                continue
            if float(np.linalg.norm(snapped[[0, 2]] - requested[[0, 2]])) > float(
                cfg.ycb.viewpoint_max_snap_m
            ):
                continue
            if any(
                float(np.linalg.norm(snapped - np.asarray(item["position"])))
                < float(cfg.ycb.viewpoint_dedup_m)
                for item in viewpoints
            ):
                continue
            rotation = _yaw_facing(snapped, target)
            semantic = simulator.semantic_at(snapped, rotation)
            visible_pixels = int(np.count_nonzero(semantic == int(authored.semantic_id)))
            if visible_pixels < int(cfg.ycb.viewpoint_min_visible_pixels):
                continue
            viewpoints.append(
                {
                    "position": [float(x) for x in snapped],
                    "rotation": rotation,
                    "visible_pixels": visible_pixels,
                    "requested_radius_m": radius,
                }
            )
    viewpoints.sort(
        key=lambda item: (
            -int(item["visible_pixels"]),
            item["requested_radius_m"],
            item["position"],
        )
    )
    if not viewpoints:
        raise YCBLayoutError(
            f"{authored.handle} ({authored.semantic_id}) has no target-visible navigable viewpoint"
        )
    return viewpoints


def _starts_for_object(
    pathfinder, viewpoints: Sequence[Mapping[str, Any]], authored, cfg
) -> List[Dict[str, Any]]:
    goals = [item["position"] for item in viewpoints]
    starts: List[Dict[str, Any]] = []
    for start_index in range(int(cfg.ycb.starts_per_target)):
        derived_seed = int(cfg.ycb.seed) + int(authored.semantic_id) * 1009 + start_index
        pathfinder.seed(derived_seed)
        rng = np.random.default_rng(derived_seed)
        for _ in range(int(cfg.ycb.start_sample_attempts)):
            point = _finite_point(pathfinder.get_random_navigable_point())
            if point is None:
                continue
            distance = _geodesic_distance(pathfinder, point, goals)
            if math.isfinite(distance) and distance >= float(cfg.ycb.start_min_geodesic_m):
                starts.append(
                    {
                        "position": [float(x) for x in point],
                        "rotation": _random_yaw(rng),
                        "initial_geodesic_distance": distance,
                    }
                )
                break
        else:
            raise YCBLayoutError(
                f"could not sample a start >= {cfg.ycb.start_min_geodesic_m} m "
                f"from {authored.handle} after {cfg.ycb.start_sample_attempts} attempts"
            )
    return starts


def generate_manifest(layout: AuthoredLayout, cfg) -> Dict[str, Any]:
    cache_key = manifest_cache_key(layout, cfg)
    simulator = _ManifestSimulator(layout, cfg)
    try:
        episodes: List[Dict[str, Any]] = []
        for authored in layout.objects:
            viewpoints = _viewpoints_for_object(simulator, authored, cfg)
            starts = _starts_for_object(simulator.sim.pathfinder, viewpoints, authored, cfg)
            for start_index, start in enumerate(starts):
                episodes.append(
                    {
                        "episode_id": (
                            f"{layout.scene_name}__{layout.layout_id}__"
                            f"{authored.semantic_id}__s{start_index}"
                        ),
                        "target": {
                            "semantic_id": authored.semantic_id,
                            "handle": authored.handle,
                            "label": authored.label,
                            "position": list(authored.translation),
                        },
                        "start": start,
                        "viewpoints": viewpoints,
                    }
                )
    finally:
        simulator.close()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_version": MANIFEST_GENERATOR_VERSION,
        "cache_key": cache_key,
        "cache_parameters": manifest_cache_payload(layout, cfg),
        "layout": {
            "scene": layout.scene_name,
            "layout_type": layout.layout_type,
            "layout_index": layout.layout_index,
            "layout_id": layout.layout_id,
            "path": layout.layout_relative_path,
            "sha256": layout.layout_sha256,
            "scene_mesh": layout.scene_mesh_relative_path,
            "scene_dataset_config": layout.scene_dataset_config_relative_path,
        },
        "episodes": episodes,
    }


def prepare_ycb_benchmark(cfg, *, force: bool = False) -> PreparedYCB:
    discovery = discover_authored_layouts(
        layout_root=Path(str(cfg.ycb.layout_root)),
        data_root=Path(str(cfg.ycb.data_root)),
        scenes=[str(value) for value in cfg.ycb.scenes],
        layout_types=[str(value) for value in cfg.ycb.layout_types],
        layout_indices=[int(value) for value in cfg.ycb.layout_indices],
        target_labels={str(key): str(value) for key, value in cfg.ycb.target_labels.items()},
    )
    manifests: List[Dict[str, Any]] = []
    cache_files: List[Path] = []
    for layout in discovery.layouts:
        cache_path = manifest_cache_path(layout, cfg)
        cache_key = manifest_cache_key(layout, cfg)
        manifest = None if force else _load_cached_manifest(cache_path, cache_key)
        if manifest is None:
            manifest = generate_manifest(layout, cfg)
            _atomic_write_json(cache_path, manifest)
        manifests.append(manifest)
        cache_files.append(cache_path)
    return PreparedYCB(discovery, tuple(manifests), tuple(cache_files))


def _make_dataset(prepared: PreparedYCB, layout_by_key: Mapping[Tuple[str, str], AuthoredLayout]):
    from habitat.core.simulator import AgentState
    from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
    from habitat.tasks.nav.object_nav_task import (
        ObjectGoal,
        ObjectGoalNavEpisode,
        ObjectViewLocation,
    )

    dataset = ObjectNavDatasetV1()
    dataset.episodes = []
    labels = sorted(
        {
            str(episode["target"]["label"])
            for manifest in prepared.manifests
            for episode in manifest["episodes"]
        }
    )
    dataset.category_to_task_category_id = {label: index for index, label in enumerate(labels)}
    dataset.category_to_scene_annotation_category_id = dict(dataset.category_to_task_category_id)
    dataset.goals_by_category = {}

    for manifest in prepared.manifests:
        layout_meta = manifest["layout"]
        layout_key = (str(layout_meta["scene"]), str(layout_meta["layout_id"]))
        layout = layout_by_key[layout_key]
        for item in manifest["episodes"]:
            target = item["target"]
            view_points = [
                ObjectViewLocation(
                    agent_state=AgentState(
                        position=[float(x) for x in vp["position"]],
                        rotation=[float(x) for x in vp["rotation"]],
                    ),
                    iou=None,
                )
                for vp in item["viewpoints"]
            ]
            goal = ObjectGoal(
                position=[float(x) for x in target["position"]],
                object_id=str(target["semantic_id"]),
                object_name=str(target["handle"]),
                object_name_id=int(target["semantic_id"]),
                object_category=str(target["label"]),
                view_points=view_points,
            )
            info = {
                "ycb": {
                    **layout_meta,
                    "target_handle": str(target["handle"]),
                    "target_semantic_id": int(target["semantic_id"]),
                    "target_position": [float(x) for x in target["position"]],
                    "start": _plain(item["start"]),
                    "viewpoints": _plain(item["viewpoints"]),
                    "manifest_cache_key": str(manifest["cache_key"]),
                }
            }
            episode = ObjectGoalNavEpisode(
                episode_id=str(item["episode_id"]),
                scene_id=str(layout.scene_mesh),
                scene_dataset_config=str(layout.scene_dataset_config),
                additional_obj_config_paths=[str(layout.objects_dir)],
                start_position=[float(x) for x in item["start"]["position"]],
                start_rotation=[float(x) for x in item["start"]["rotation"]],
                goals=[goal],
                object_category=str(target["label"]),
                info=info,
            )
            dataset.episodes.append(episode)
            dataset.goals_by_category[episode.goals_key] = [goal]
    return dataset


class YCBAuthoredNavEnv(HabitatObjectNavEnv):
    """ObjectNav-compatible environment backed by authored rigid-object layouts."""

    def __init__(self, cfg) -> None:
        import habitat

        self.prepared = prepare_ycb_benchmark(cfg)
        self._layout_by_key = {
            (layout.scene_name, layout.layout_id): layout
            for layout in self.prepared.discovery.layouts
        }
        dataset = _make_dataset(self.prepared, self._layout_by_key)
        self._hab_cfg = make_objectnav_config(cfg)
        self.env = habitat.Env(config=self._hab_cfg, dataset=dataset)
        from ..core.types import CameraIntrinsics

        self.intrinsics = CameraIntrinsics.from_hfov(
            cfg.eval.hfov_deg, cfg.eval.rgb_width, cfg.eval.rgb_height
        )
        self._frame_id = 0
        self._follower = None
        self._action_name = {value: key for key, value in self.ACTIONS.items()}
        self._navmesh_goal_radius = float(getattr(cfg.agent, "navmesh_goal_radius", 0.1))
        self._active_objects: List[Any] = []

    def reset(self):
        self.env.reset()
        info = (getattr(self.current_episode, "info", None) or {}).get("ycb", {})
        key = (str(info.get("scene")), str(info.get("layout_id")))
        layout = self._layout_by_key.get(key)
        if layout is None:
            raise YCBLayoutError(f"episode references unknown authored layout {key}")
        self._active_objects = inject_layout_objects(self.env.sim, layout)
        observations = self.env.sim.get_observations_at()
        if observations is None:
            raise RuntimeError("failed to refresh observations after YCB object injection")
        self._frame_id = 0
        return self._to_frame(observations)

    def episode_metadata(self) -> Dict[str, Any]:
        return dict((getattr(self.current_episode, "info", None) or {}).get("ycb", {}))

    def benchmark_metadata(self) -> Dict[str, Any]:
        return {
            "mode": "ycb_authored",
            "selected_layouts": [
                {
                    "scene": layout.scene_name,
                    "layout_id": layout.layout_id,
                    "layout_type": layout.layout_type,
                    "layout_index": layout.layout_index,
                    "layout_path": layout.layout_relative_path,
                    "layout_sha256": layout.layout_sha256,
                    "num_targets": len(layout.objects),
                }
                for layout in self.prepared.discovery.layouts
            ],
            "skipped": list(self.prepared.discovery.skipped),
            "manifest_cache_files": [str(path) for path in self.prepared.cache_files],
        }
