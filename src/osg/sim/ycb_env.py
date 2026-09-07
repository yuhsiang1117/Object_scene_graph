"""Habitat environment and deterministic episode cache for authored YCB scenes."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .habitat_env import HabitatObjectNavEnv, make_objectnav_config
from .ycb_layouts import (
    AuthoredLayout,
    LayoutDiscovery,
    RelocationPair,
    YCBLayoutError,
    discover_authored_layouts,
    relocation_pairs,
)


# Above this share of the frame, a "target" mask is scene geometry with a
# colliding semantic id rather than a tabletop object. See _viewpoints_for_object.
SEMANTIC_COLLISION_FRAC = 0.2

MANIFEST_SCHEMA_VERSION = 2
MANIFEST_GENERATOR_VERSION = 2


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


def manifest_cache_payload(
    layout: AuthoredLayout, cfg, source_layout: Optional[AuthoredLayout] = None
) -> Dict[str, Any]:
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
        "cross_floor_relocations_only": bool(
            getattr(cfg.ycb, "cross_floor_relocations_only", False)
        ),
        "relocation_directions": [
            str(x) for x in getattr(
                cfg.ycb, "relocation_directions", ["upward", "downward"]
            )
        ],
        "start_on_prior_floor": bool(getattr(cfg.ycb, "start_on_prior_floor", False)),
        "relocation_floor_tolerance_m": float(
            getattr(cfg.ycb, "relocation_floor_tolerance_m", 0.5)
        ),
        "skip_incomplete_layouts": bool(
            getattr(cfg.ycb, "skip_incomplete_layouts", False)
        ),
        "source_layout_sha256": (
            str(source_layout.layout_sha256) if source_layout is not None else None
        ),
        "camera": {
            "width": int(cfg.eval.rgb_width),
            "height": int(cfg.eval.rgb_height),
            "hfov_deg": float(cfg.eval.hfov_deg),
            "height_m": float(cfg.agent.camera_height),
        },
        "agent_radius": float(cfg.agent.agent_radius),
    }


def manifest_cache_key(
    layout: AuthoredLayout, cfg, source_layout: Optional[AuthoredLayout] = None
) -> str:
    encoded = json.dumps(
        manifest_cache_payload(layout, cfg, source_layout),
        sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_cache_path(
    layout: AuthoredLayout, cfg, source_layout: Optional[AuthoredLayout] = None
) -> Path:
    key = manifest_cache_key(layout, cfg, source_layout)
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


def apply_layout_transforms(objects: List[Any], layout: AuthoredLayout) -> List[int]:
    """Move already-placed rigid objects to another layout's poses.

    The same object ids, moved -- not a fresh injection. That distinction is the
    whole point of a mid-episode relocation: the simulator state, the agent, its
    map and its beliefs all survive, so the change is something the robot can
    WITNESS rather than something it wakes up to.

    Returns the semantic ids that actually moved.
    """
    import magnum as mn

    by_id = {}
    for rigid in objects:
        try:
            by_id[int(rigid.semantic_id)] = rigid
        except (AttributeError, TypeError, ValueError):
            continue

    import habitat_sim

    moved = []
    for authored in layout.objects:
        rigid = by_id.get(int(authored.semantic_id))
        if rigid is None:
            raise YCBLayoutError(
                f"cannot relocate {authored.semantic_id}: not present in the running scene"
            )
        before = np.array([rigid.translation.x, rigid.translation.y, rigid.translation.z])
        # inject_layout_objects marks objects STATIC, and a STATIC object
        # SILENTLY IGNORES a new translation -- it even reads back the old pose
        # afterwards, so nothing in the calling code can tell. Every mid-episode
        # relocation before this was a no-op that reported success. KINEMATIC
        # objects hold their pose exactly the same way and can be moved.
        rigid.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        rigid.translation = mn.Vector3(*authored.translation)
        rigid.rotation = mn.Quaternion(
            mn.Vector3(*authored.rotation[:3]), authored.rotation[3]
        )
        after = np.array([rigid.translation.x, rigid.translation.y, rigid.translation.z])
        target = np.asarray(authored.translation, float)
        if float(np.linalg.norm(after - target)) > 1e-3:
            raise YCBLayoutError(
                f"relocating {authored.semantic_id} did not take: asked for "
                f"{target.tolist()}, object reports {after.tolist()}"
            )
        if float(np.linalg.norm(before - target)) > 1e-6:
            moved.append(int(authored.semantic_id))
    return moved


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
            # A YCB object is at most ~25 cm across and these viewpoints stand at
            # 0.8 m or further, so it cannot occupy a fifth of the frame -- 25 cm
            # at 0.8 m is under 5%. A mask that large means the authored semantic
            # id is also some SCENE instance's id, and every number derived from
            # this mask is then about a wall. The collector's ids are 26..95 and
            # this HM3D scene annotates 252 instances as 0..251, so the collision
            # is real; it stays invisible only because the manifest simulator
            # carries a semantic sensor alone and habitat then renders the scene
            # as a single blob. Attach a colour sensor here and the ids appear.
            if visible_pixels > SEMANTIC_COLLISION_FRAC * semantic.size:
                raise YCBLayoutError(
                    f"{authored.handle} (semantic id {authored.semantic_id}) covers "
                    f"{visible_pixels}/{semantic.size} px at {radius:.1f} m -- that is "
                    "scene geometry sharing the id, not the object. Re-author the "
                    "layout with ids outside the scene's instance range."
                )
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
    pathfinder, viewpoints: Sequence[Mapping[str, Any]], authored, cfg,
    required_floor_y: Optional[float] = None,
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
            if required_floor_y is not None and abs(
                float(point[1]) - float(required_floor_y)
            ) > float(getattr(cfg.ycb, "relocation_floor_tolerance_m", 0.5)):
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


def generate_manifest(
    layout: AuthoredLayout, cfg, source_layout: Optional[AuthoredLayout] = None
) -> Dict[str, Any]:
    cache_key = manifest_cache_key(layout, cfg, source_layout)
    simulator = _ManifestSimulator(layout, cfg)
    try:
        episodes: List[Dict[str, Any]] = []
        skipped_targets: List[Dict[str, Any]] = []
        source_by_id = {
            int(obj.semantic_id): obj for obj in getattr(source_layout, "objects", ())
        }
        for authored in layout.objects:
            try:
                viewpoints = _viewpoints_for_object(simulator, authored, cfg)
            except YCBLayoutError as exc:
                # Combined campaigns may explicitly retain heterogeneous
                # authored layouts while omitting individual targets that do
                # not have a valid navigable/viewpoint construction.  The
                # same opt-in must apply to the static map pass; otherwise a
                # single unobservable object aborts map generation before the
                # dynamic suite can run.
                if not (
                    bool(getattr(cfg.ycb, "cross_floor_relocations_only", False))
                    or bool(getattr(cfg.ycb, "skip_incomplete_layouts", False))
                ):
                    raise
                skipped_targets.append({
                    "semantic_id": int(authored.semantic_id),
                    "handle": str(authored.handle),
                    "reason": str(exc),
                })
                continue
            prior = source_by_id.get(int(authored.semantic_id))
            prior_floor_y = None
            destination_floor_y = float(viewpoints[0]["position"][1])
            direction = None
            if prior is not None:
                snapped = _finite_point(
                    simulator.sim.pathfinder.snap_point(
                        np.asarray(prior.translation, dtype=np.float32)
                    )
                )
                if snapped is not None:
                    prior_floor_y = float(snapped[1])
                    delta = destination_floor_y - prior_floor_y
                    tolerance = float(getattr(cfg.ycb, "relocation_floor_tolerance_m", 0.5))
                    direction = (
                        "same_floor" if abs(delta) <= tolerance
                        else "upward" if delta > 0.0 else "downward"
                    )
            if bool(getattr(cfg.ycb, "cross_floor_relocations_only", False)):
                allowed = {
                    str(value) for value in getattr(
                        cfg.ycb, "relocation_directions", ["upward", "downward"]
                    )
                }
                if direction not in {"upward", "downward"} or direction not in allowed:
                    continue
            required_y = (
                prior_floor_y
                if bool(getattr(cfg.ycb, "start_on_prior_floor", False))
                and prior_floor_y is not None
                else None
            )
            try:
                starts = _starts_for_object(
                    simulator.sim.pathfinder, viewpoints, authored, cfg,
                    required_floor_y=required_y,
                )
            except YCBLayoutError as exc:
                if not (
                    bool(getattr(cfg.ycb, "cross_floor_relocations_only", False))
                    or bool(getattr(cfg.ycb, "skip_incomplete_layouts", False))
                ):
                    raise
                skipped_targets.append({
                    "semantic_id": int(authored.semantic_id),
                    "handle": str(authored.handle),
                    "reason": str(exc),
                })
                continue
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
                            "prior_position": (
                                list(prior.translation) if prior is not None else None
                            ),
                            "prior_floor_y": prior_floor_y,
                            "destination_floor_y": destination_floor_y,
                            "relocation_direction": direction,
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
        "cache_parameters": manifest_cache_payload(layout, cfg, source_layout),
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
        "skipped_targets": skipped_targets,
    }


def prepare_ycb_benchmark(cfg, *, force: bool = False) -> PreparedYCB:
    discovery = discover_authored_layouts(
        layout_root=Path(str(cfg.ycb.layout_root)),
        data_root=Path(str(cfg.ycb.data_root)),
        scenes=[str(value) for value in cfg.ycb.scenes],
        layout_types=[str(value) for value in cfg.ycb.layout_types],
        layout_indices=[int(value) for value in cfg.ycb.layout_indices],
        target_labels={str(key): str(value) for key, value in cfg.ycb.target_labels.items()},
        allow_incomplete=bool(getattr(cfg.ycb, "skip_incomplete_layouts", False)),
    )
    manifests: List[Dict[str, Any]] = []
    cache_files: List[Path] = []
    static_by_scene = {
        layout.scene_name: layout
        for layout in discovery.layouts
        if layout.layout_type == "static"
    }
    if bool(getattr(cfg.ycb, "cross_floor_relocations_only", False)):
        missing = sorted({
            layout.scene_name for layout in discovery.layouts
            if layout.layout_type != "static" and layout.scene_name not in static_by_scene
        })
        if missing:
            raise YCBLayoutError(
                "cross-floor relocation manifests require the static source layout for: "
                + ", ".join(missing)
            )
    for layout in discovery.layouts:
        if (
            bool(getattr(cfg.ycb, "cross_floor_relocations_only", False))
            and layout.layout_type == "static"
        ):
            # The static layout is the relocation source and snapshot pass, not
            # an episode in the cross-floor destination suite.
            continue
        source_layout = (
            static_by_scene.get(layout.scene_name)
            if layout.layout_type != "static" else None
        )
        cache_path = manifest_cache_path(layout, cfg, source_layout)
        cache_key = manifest_cache_key(layout, cfg, source_layout)
        manifest = None if force else _load_cached_manifest(cache_path, cache_key)
        if manifest is None:
            manifest = generate_manifest(layout, cfg, source_layout)
            _atomic_write_json(cache_path, manifest)
        manifests.append(manifest)
        cache_files.append(cache_path)
    return PreparedYCB(discovery, tuple(manifests), tuple(cache_files))


def select_targets(episodes: Sequence[Mapping[str, Any]], wanted: Sequence[str]):
    """Keep only episodes whose target is in `wanted` (handle or label).

    Several YCB assets in the collector's dataset cannot be detected at all --
    probed at each object's best authored viewpoint, the pitcher, plate,
    scissors and cracker box return nothing even as the detector's only class,
    because of how those meshes render. Running their episodes measures the
    detector's asset coverage, not dynamic-scene handling. The layouts are
    DualMap's original data and are never edited; this only chooses which
    episodes to run.

    Empty `wanted` keeps everything.
    """
    if not wanted:
        return list(episodes)
    keys = {str(w).strip().lower() for w in wanted}
    kept = [
        ep for ep in episodes
        if str(ep["target"]["handle"]).lower() in keys
        or str(ep["target"]["label"]).lower() in keys
    ]
    return kept


def _make_dataset(
    prepared: PreparedYCB,
    layout_by_key: Mapping[Tuple[str, str], AuthoredLayout],
    cfg,
    targets: Sequence[str] = (),
):
    from habitat.core.simulator import AgentState
    from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
    from habitat.tasks.nav.object_nav_task import (
        ObjectGoal,
        ObjectGoalNavEpisode,
        ObjectViewLocation,
    )

    dataset = ObjectNavDatasetV1()
    dataset.episodes = []
    active_manifests = [
        manifest for manifest in prepared.manifests
        if not (
            bool(getattr(cfg.ycb, "cross_floor_relocations_only", False))
            and str(manifest["layout"].get("layout_type")) == "static"
        )
    ]
    selected = {
        id(manifest): select_targets(manifest["episodes"], targets)
        for manifest in active_manifests
    }
    labels = sorted(
        {
            str(episode["target"]["label"])
            for manifest in active_manifests
            for episode in selected[id(manifest)]
        }
    )
    if not labels:
        raise YCBLayoutError(
            f"ycb.targets={list(targets)} matched no episode; targets are matched "
            "against the YCB handle or its label"
        )
    dataset.category_to_task_category_id = {label: index for index, label in enumerate(labels)}
    dataset.category_to_scene_annotation_category_id = dict(dataset.category_to_task_category_id)
    dataset.goals_by_category = {}

    for manifest in active_manifests:
        layout_meta = manifest["layout"]
        layout_key = (str(layout_meta["scene"]), str(layout_meta["layout_id"]))
        layout = layout_by_key[layout_key]
        for item in selected[id(manifest)]:
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
                    "prior_position": target.get("prior_position"),
                    "prior_floor_y": target.get("prior_floor_y"),
                    "destination_floor_y": target.get("destination_floor_y"),
                    "relocation_direction": target.get("relocation_direction"),
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


class _RelocationPolicy:
    """When, and under what visibility, a relocation fires."""

    def __init__(self, cfg, layouts: Sequence[AuthoredLayout]) -> None:
        ycb = cfg.ycb
        self.at_step = int(ycb.relocate_at_step)
        self.when = str(ycb.relocate_when)
        self.deadline_steps = int(ycb.relocate_deadline_steps)
        # Firing mid-episode is opt-in, but KNOWING the pair is not: the
        # two-pass protocol changes the world between runs, and the metrics
        # still need to know what moved and where it moved from.
        self.enabled = self.at_step >= 0
        self._pairs: Dict[Tuple[str, str], RelocationPair] = {}
        try:
            for pair in relocation_pairs(layouts):
                self._pairs[(pair.after.scene_name, pair.after.layout_id)] = pair
        except YCBLayoutError:
            # A dynamic-only selection is legitimate for pass 2 -- the static
            # layout lives in the snapshot, not in this run.
            if self.enabled:
                raise

    def pair_for(self, key: Tuple[str, str]) -> Optional[RelocationPair]:
        return self._pairs.get(key)

    def condition_met(self, in_view: bool) -> bool:
        if self.when == "in_view":
            return in_view
        if self.when == "out_of_view":
            return not in_view
        return True


class YCBAuthoredNavEnv(HabitatObjectNavEnv):
    """ObjectNav-compatible environment backed by authored rigid-object layouts."""

    def __init__(self, cfg) -> None:
        import habitat

        self.prepared = prepare_ycb_benchmark(cfg)
        self._layout_by_key = {
            (layout.scene_name, layout.layout_id): layout
            for layout in self.prepared.discovery.layouts
        }
        self._relocation = _RelocationPolicy(cfg, self.prepared.discovery.layouts)
        dataset = _make_dataset(
            self.prepared, self._layout_by_key, cfg,
            targets=[str(t) for t in cfg.ycb.targets or []],
        )
        self._hab_cfg = make_objectnav_config(cfg)
        self.env = habitat.Env(config=self._hab_cfg, dataset=dataset)
        from ..core.types import CameraIntrinsics

        self.intrinsics = CameraIntrinsics.from_hfov(
            cfg.eval.hfov_deg, cfg.eval.rgb_width, cfg.eval.rgb_height
        )
        self._frame_id = 0
        self._follower = None
        self._action_name = {value: key for key, value in self.ACTIONS.items()}
        self._navmesh_goal_radius = float(cfg.agent.navmesh_goal_radius)
        self._active_objects: List[Any] = []

    def reset(self):
        self.nav_reasons.clear()
        self.env.reset()
        info = (getattr(self.current_episode, "info", None) or {}).get("ycb", {})
        key = (str(info.get("scene")), str(info.get("layout_id")))
        layout = self._layout_by_key.get(key)
        if layout is None:
            raise YCBLayoutError(f"episode references unknown authored layout {key}")

        # A relocation episode starts the world in the BEFORE layout while its
        # goals sit at the AFTER poses: the agent maps the old world, watches
        # (or misses) the change, and is scored on finding the object where it
        # now is. Injecting the episode's own layout would make the change
        # unobservable, which is exactly DualMap's protocol.
        self._pair = self._relocation.pair_for(key)
        # Only the mid-episode variant starts the world in the BEFORE layout.
        # The two-pass benchmark runs the moved world from step 0 and gets its
        # staleness from the snapshot it loads, not from a live change.
        start_layout = (
            self._pair.before
            if (self._pair is not None and self._relocation.enabled)
            else layout
        )
        self._active_objects = inject_layout_objects(self.env.sim, start_layout)
        self._relocated_at = None
        self._relocated_in_view = None
        self._relocated_ids: List[int] = []

        observations = self.env.sim.get_observations_at()
        if observations is None:
            raise RuntimeError("failed to refresh observations after YCB object injection")
        self._frame_id = 0
        return self._to_frame(observations)

    def step(self, action: str):
        frame = super().step(action)
        self._maybe_relocate(frame)
        return frame

    # ------------------------------------------------------------ relocation

    def _maybe_relocate(self, frame) -> None:
        policy = self._relocation
        # Knowing the pair is not permission to fire: with the two-pass protocol
        # the pair exists for metadata only, and relocating here would re-apply
        # poses the world is already in -- recording a live change that never
        # happened and masking the offline one.
        if not policy.enabled or self._pair is None or self._relocated_at is not None:
            return
        if self._frame_id < policy.at_step:
            return
        in_view = self._target_in_view(frame)
        overdue = self._frame_id >= policy.at_step + policy.deadline_steps
        if not overdue and not policy.condition_met(in_view):
            return
        self._relocated_ids = apply_layout_transforms(self._active_objects, self._pair.after)
        self._relocated_at = int(self._frame_id)
        self._relocated_in_view = bool(in_view)

    def _target_in_view(self, frame) -> bool:
        """Is the episode's target visible from the current pose right now?

        Deliberately the same question the presence filter asks, answered from
        ground truth: frustum, range, then a depth read at the object's pixel so
        an object behind a wall does not count as witnessed.
        """
        info = (getattr(self.current_episode, "info", None) or {}).get("ycb", {})
        target_id = info.get("target_semantic_id")
        position = None
        for authored in (self._pair.before.objects if self._pair else ()):
            if target_id is not None and int(authored.semantic_id) == int(target_id):
                position = np.asarray(authored.translation, dtype=float)
                break
        if position is None:
            return False

        T_cw = frame.T_cw
        p_cam = T_cw[:3, :3] @ position + T_cw[:3, 3]
        z = float(p_cam[2])
        if not (0.3 <= z <= 6.0):
            return False
        K = frame.intrinsics.K()
        uvw = K @ p_cam
        u, v = float(uvw[0] / z), float(uvw[1] / z)
        h, w = frame.depth.shape
        if not (0 <= u < w and 0 <= v < h):
            return False
        measured = float(frame.depth[int(v), int(u)])
        if measured <= 1e-3:
            return False
        return abs(measured - z) <= 0.5

    def episode_metadata(self) -> Dict[str, Any]:
        meta = dict((getattr(self.current_episode, "info", None) or {}).get("ycb", {}))
        if self._pair is not None:
            target_id = meta.get("target_semantic_id")
            origin = destination = None
            for authored in self._pair.before.objects:
                if target_id is not None and int(authored.semantic_id) == int(target_id):
                    origin = list(authored.translation)
                    break
            for authored in self._pair.after.objects:
                if target_id is not None and int(authored.semantic_id) == int(target_id):
                    destination = list(authored.translation)
                    break
            meta["relocation"] = {
                "kind": self._pair.kind,
                "from_layout": self._pair.before.layout_id,
                "to_layout": self._pair.after.layout_id,
                "step": self._relocated_at,
                "in_view": self._relocated_in_view,
                "moved_semantic_ids": list(self._relocated_ids),
                # Where the target USED to be: ghost rate is "does the map still
                # believe it is here", so the metric needs the old pose.
                "origin_position": origin,
                "destination_position": destination,
            }
        return meta

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
