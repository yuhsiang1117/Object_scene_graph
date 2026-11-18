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
        # The draw's rule, not just the flag: a manifest built while the draw
        # still included object-less storeys must not satisfy the current one.
        "start_storey_uniform": (
            "object_storeys" if bool(getattr(cfg.ycb, "start_storey_uniform", False))
            else False),
        # Missing until now: a manifest built with the control arm's starts was
        # a cache hit for the normal arm, and vice versa.
        "start_on_target_floor": bool(getattr(cfg.ycb, "start_on_target_floor", False)),
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


def _target_floor_y(info: Dict[str, Any]) -> Optional[float]:
    """The navigable floor height at the target, from the episode's manifest.

    `destination_floor_y` is the height of the target's first goal view point,
    which generation already requires to be navigable; a manifest cached
    before that field existed still carries the view points themselves."""
    value = info.get("destination_floor_y")
    if value is None:
        viewpoints = info.get("viewpoints") or []
        if viewpoints and viewpoints[0].get("position") is not None:
            value = viewpoints[0]["position"][1]
    return None if value is None else float(value)


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


def _starts_on_uniform_storeys(
    pathfinder, goals, authored, cfg, storey_heights: Sequence[float],
) -> List[Dict[str, Any]]:
    """One start per `starts_per_target`, its storey drawn uniformly.

    The storey is drawn first and the point second, so a two-storey scene
    starts half its episodes on the object's storey whatever the two storeys'
    areas are -- an area-uniform point would let the bigger floor decide how
    often the stairs are needed. A storey with no point far enough from the
    object (or not connected to it) is skipped for the next one in the draw.
    The point is never on the stairs: it must be within
    `relocation_floor_tolerance_m` of a storey's height."""
    tol = float(getattr(cfg.ycb, "relocation_floor_tolerance_m", 0.5))
    starts: List[Dict[str, Any]] = []
    for start_index in range(int(cfg.ycb.starts_per_target)):
        derived_seed = int(cfg.ycb.seed) + int(authored.semantic_id) * 1009 + start_index
        pathfinder.seed(derived_seed)
        rng = np.random.default_rng(derived_seed)
        found = None
        for storey in rng.permutation(len(storey_heights)):
            height = float(storey_heights[int(storey)])
            for _ in range(int(cfg.ycb.start_sample_attempts)):
                point = _finite_point(pathfinder.get_random_navigable_point())
                if point is None or abs(float(point[1]) - height) > tol:
                    continue
                distance = _geodesic_distance(pathfinder, point, goals)
                if math.isfinite(distance) and distance >= float(cfg.ycb.start_min_geodesic_m):
                    found = {
                        "position": [float(x) for x in point],
                        "rotation": _random_yaw(rng),
                        "initial_geodesic_distance": distance,
                        "start_storey_y": height,
                    }
                    break
            if found is not None:
                break
        if found is None:
            raise YCBLayoutError(
                f"could not sample a start >= {cfg.ycb.start_min_geodesic_m} m "
                f"from {authored.handle} on any of the storeys {list(storey_heights)}"
            )
        starts.append(found)
    return starts


def _benchmark_storeys(pathfinder, layout, source_layout, cfg) -> List[float]:
    """The storeys a start may be drawn on: the scene's navmesh storeys that
    hold an authored object, before or after the move.

    Not every navmesh storey. 00878's basement is navigable, holds no object
    in any layout, and is out of the prior map's scope (the coverage audit
    exempts it); drawing over it would start a third of that scene's episodes
    in a room nothing in the benchmark is about."""
    from ..eval.floors import navmesh_storeys

    tol = float(getattr(cfg.ycb, "relocation_floor_tolerance_m", 0.5))
    heights = [
        float(s["height"]) for s in navmesh_storeys(
            pathfinder, n_points=4000, seed=int(cfg.ycb.seed))["storeys"]
        if int(s["n_points"]) > 0
    ]
    floors = []
    for source in (layout, source_layout):
        for obj in getattr(source, "objects", ()) or ():
            snapped = _finite_point(pathfinder.snap_point(
                np.asarray(obj.translation, dtype=np.float32)))
            if snapped is not None:
                floors.append(float(snapped[1]))
    used = [h for h in heights if any(abs(h - f) <= tol for f in floors)]
    return used or heights


def _starts_for_object(
    pathfinder, viewpoints: Sequence[Mapping[str, Any]], authored, cfg,
    required_floor_y: Optional[float] = None,
    storey_heights: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    goals = [item["position"] for item in viewpoints]
    if required_floor_y is None and storey_heights:
        return _starts_on_uniform_storeys(pathfinder, goals, authored, cfg, storey_heights)
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
        storey_heights = None
        if bool(getattr(cfg.ycb, "start_storey_uniform", False)):
            storey_heights = _benchmark_storeys(
                simulator.sim.pathfinder, layout, source_layout, cfg)
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
            # The control arm: begin on the storey the object is really on, so
            # the episode measures search and approach with the staircase taken
            # out. Takes precedence, because asking for both is asking for the
            # destination.
            if bool(getattr(cfg.ycb, "start_on_target_floor", False)) \
                    and destination_floor_y is not None:
                required_y = destination_floor_y
            try:
                starts = _starts_for_object(
                    simulator.sim.pathfinder, viewpoints, authored, cfg,
                    required_floor_y=required_y,
                    storey_heights=storey_heights,
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
    discovery_kwargs = dict(
        layout_root=Path(str(cfg.ycb.layout_root)),
        data_root=Path(str(cfg.ycb.data_root)),
        scenes=[str(value) for value in cfg.ycb.scenes],
        layout_types=[str(value) for value in cfg.ycb.layout_types],
        layout_indices=[int(value) for value in cfg.ycb.layout_indices],
        target_labels={str(key): str(value) for key, value in cfg.ycb.target_labels.items()},
        allow_incomplete=bool(getattr(cfg.ycb, "skip_incomplete_layouts", False)),
    )
    # ``hm3d_root`` was added after the original YCB helper.  Keep lightweight
    # test/integration callers that provide only the old fields working, while
    # every registered Hydra config resolves legacy scene paths to v0.2.
    configured_hm3d_root = getattr(cfg.ycb, "hm3d_root", None)
    if configured_hm3d_root:
        discovery_kwargs["hm3d_root"] = Path(str(configured_hm3d_root))
    discovery = discover_authored_layouts(**discovery_kwargs)
    manifests: List[Dict[str, Any]] = []
    cache_files: List[Path] = []
    static_by_scene = {
        layout.scene_name: layout
        for layout in discovery.layouts
        if layout.layout_type == "static"
    }
    # The static layout is the RELOCATION SOURCE, not only an episode type. The
    # nav pass runs `layout_types=[in_anchor,cross_anchor]`, which kept static
    # out of discovery entirely -- so `source_layout` was None, and with it
    # every relocation field: `prior_position`, `prior_floor_y` and
    # `relocation_direction` are null in all 30 episodes of
    # outputs/osg_authored_15. Two things silently did not happen. The record
    # could not say which floor the object came FROM, and `start_on_prior_floor`
    # -- the combined preset's headline, "every deterministic start is sampled
    # on the object's prior floor" -- had no floor to require, so starts were
    # sampled anywhere and `floor_class` was an accident of sampling.
    #
    # Discover the static layouts separately when the episode types exclude
    # them. This changes the manifest cache key, and therefore the sampled
    # starts, for any run that relocates: that is the point.
    if "static" not in set(discovery_kwargs["layout_types"]):
        source_kwargs = dict(discovery_kwargs, layout_types=["static"])
        try:
            for layout in discover_authored_layouts(**source_kwargs).layouts:
                static_by_scene.setdefault(layout.scene_name, layout)
        except YCBLayoutError:
            # A root with no static layout at all is a valid static-only or
            # legacy corpus; the fields simply stay null, as before.
            pass
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


class ObjectDistanceRule:
    """A STOP within `threshold_m` horizontal of the object, within
    `max_attempts` attempts: the released DualMap benchmark's rule, applied to
    an authored episode. Pure bookkeeping, so it is testable without habitat.

    Distance is to the object's position (zero extent), which is what the
    released harness uses for its YCB targets too.

    The released rule is single-storey, and horizontal distance alone cannot
    tell a stop beside the object from a stop on the floor above it. When
    `max_floor_dy_m` > 0 and the env has set `target_floor_y` (the navigable
    floor height at the object, `destination_floor_y` in the manifest), a stop
    also has to stand within that height of the object's floor. Measured on
    outputs/RERUN/mf5: 00862's and 00873's cracker boxes both "succeeded" from
    the wrong storey, 0.62 / 0.57 m horizontal and 2.33 / 2.11 m in 3D, each a
    stop at a false positive directly above or below the object."""

    def __init__(self, enabled: bool, threshold_m: float = 1.0, max_attempts: int = 3,
                 max_floor_dy_m: float = 0.0) -> None:
        self.enabled = bool(enabled)
        self.threshold_m = float(threshold_m)
        self.max_attempts = int(max_attempts)
        self.max_floor_dy_m = float(max_floor_dy_m)
        #: Set by the env: attempts -> geodesic from the episode start to the
        #: winning stop, or None. The rule is pure bookkeeping and holds no
        #: simulator, so the one thing it cannot compute is a path length.
        self.shortest_fallback = None
        self.reset()

    def reset(self) -> None:
        self.attempts: List[Dict[str, Any]] = []
        self._travelled = 0.0
        self._previous: Optional[np.ndarray] = None
        self.shortest_m: Optional[float] = None
        #: The navigable floor height at the object, set by the env per episode.
        #: None leaves the rule horizontal-only, and every attempt records
        #: `on_target_floor: None` so that case is visible in the record.
        self.target_floor_y: Optional[float] = None

    @property
    def floor_checked(self) -> bool:
        return self.max_floor_dy_m > 0.0 and self.target_floor_y is not None

    def on_target_floor(self, position: np.ndarray) -> Optional[bool]:
        """Is this pose on the object's storey? None when the rule cannot say."""
        if not self.floor_checked:
            return None
        return bool(abs(float(position[1]) - float(self.target_floor_y)) <= self.max_floor_dy_m)

    def travelled(self, position: np.ndarray) -> None:
        if self._previous is not None:
            self._travelled += float(np.linalg.norm((position - self._previous)[[0, 2]]))
        self._previous = np.asarray(position, dtype=float)

    @staticmethod
    def distances(position: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
        delta = np.asarray(position, dtype=float) - np.asarray(target, dtype=float)
        return float(np.hypot(delta[0], delta[2])), float(np.linalg.norm(delta))

    def record_stop(self, step: int, position: np.ndarray, target: np.ndarray) -> bool:
        horizontal, spatial = self.distances(position, target)
        within = bool(horizontal <= self.threshold_m)
        on_floor = self.on_target_floor(position)
        success = within and on_floor is not False
        if self.attempts and self.attempts[-1]["step"] == int(step):
            # The episode loop asks before stepping and habitat's terminal STOP
            # follows at the same pose: one stop, not two.
            return bool(self.attempts[-1]["success"])
        self.attempts.append({
            "attempt": len(self.attempts) + 1,
            "step": int(step),
            "position": [float(x) for x in position],
            "distance_horizontal_m": horizontal,
            "distance_3d_m": spatial,
            "floor_dy_m": (None if self.target_floor_y is None
                           else abs(float(position[1]) - float(self.target_floor_y))),
            "on_target_floor": on_floor,
            # What the horizontal-only rule would have said, so a run under the
            # floor check can still be read against the numbers before it.
            "success_horizontal_only": within,
            "success": success,
        })
        return success

    def summary(self, position: np.ndarray, target: Optional[np.ndarray]) -> Dict[str, Any]:
        shortest_m = self.shortest_m
        attempts = list(self.attempts)
        # Success is earned at a STOP and nowhere else: an episode that runs
        # out of steps beside the object has not answered the query.
        success = any(bool(a["success"]) for a in attempts[: self.max_attempts])
        horizontal = spatial = None
        if target is not None:
            horizontal, spatial = self.distances(position, target)
        if success and (shortest_m is None or not math.isfinite(shortest_m)
                        or shortest_m <= 0):
            # LAST RESORT, and it is not a fudge: a successful stop is within
            # `threshold_m` of the object by definition, so the geodesic from
            # the start to that stop is a legitimate shortest-path numerator --
            # it is what the ring sampling was trying to find. Without it a real
            # success reports SPL 0.0 and drags the mean down (1 of 9 successes
            # in outputs/mf5_pass2_final).
            if callable(self.shortest_fallback):
                shortest_m = self.shortest_fallback(attempts)
        spl = 0.0
        if success and shortest_m is not None and math.isfinite(shortest_m) and shortest_m > 0:
            spl = shortest_m / max(shortest_m, self._travelled, 1e-12)
        return {
            "success": int(success),
            "success_horizontal_only": int(any(
                bool(a.get("success_horizontal_only", a["success"]))
                for a in attempts[: self.max_attempts])),
            "spl": float(spl),
            "travelled_m": float(self._travelled),
            "attempts": attempts,
            "attempt_count": len(attempts),
            "final_distance_horizontal_m": horizontal,
            "final_distance_3d_m": spatial,
            "target_floor_y": self.target_floor_y,
            "shortest_path_to_object_m": shortest_m,
            "success_definition": self.definition(),
        }

    def definition(self) -> str:
        text = f"horizontal distance to the object <= {self.threshold_m} m"
        if self.max_floor_dy_m > 0.0:
            text += f" on the object's storey (floor height within {self.max_floor_dy_m} m)"
        return text + f", within {self.max_attempts} attempts"


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
        self.snap_on_agent_island = bool(getattr(cfg.agent, "navmesh_snap_on_agent_island", False))
        self._active_objects: List[Any] = []
        # The released rule, when asked for: a STOP within
        # `object_success_distance_m` of the object, within three attempts.
        self._object_rule = ObjectDistanceRule(
            enabled=bool(getattr(cfg.ycb, "score_by_object_distance", False)),
            threshold_m=float(getattr(cfg.ycb, "object_success_distance_m", 1.0)),
            max_floor_dy_m=float(getattr(cfg.ycb, "object_success_max_floor_dy_m", 0.0)),
        )

    def reset(self):
        self.nav_reasons.clear()
        self.env.reset()
        self._object_rule.reset()
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
        if self._object_rule.enabled:
            self._object_rule.target_floor_y = _target_floor_y(info)
            self._start_position = np.asarray(
                self.env.sim.get_agent_state().position, dtype=float)
            self._object_rule.shortest_fallback = self._shortest_to_stop
            self._object_rule.shortest_m = self._shortest_to_object()
            self._object_rule.travelled(
                np.asarray(self.env.sim.get_agent_state().position, dtype=float)
            )
        return self._to_frame(observations)

    def _shortest_to_object(self) -> Optional[float]:
        """Geodesic from the start to the nearest navigable point within the
        success radius of the object, for SPL under the object-distance rule.

        Sampled on rings around the object at the OBJECT's height, not the
        start's (as the released harness does), because here the object may be
        a storey away: a point at the start's height snaps to the start's floor
        and never reaches the object's."""
        import habitat_sim

        target = self._target_position()
        if target is None:
            return None
        try:
            pathfinder = self.env.sim.pathfinder
            start = np.asarray(self.env.sim.get_agent_state().position, dtype=np.float32)
        except Exception:
            return None
        threshold = float(self._object_rule.threshold_m)
        best = math.inf
        seen = set()
        # Denser than it looks like it needs to be, because the failure mode is
        # silent: when no sample snaps to a pathable point within the radius,
        # `shortest_m` is None and `summary` reports SPL 0.0 for an episode that
        # SUCCEEDED -- measured on 00862/50001, which stopped inside 1.0 m and
        # still scored spl=0.000. Sampling costs milliseconds once per episode.
        for radius in (0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95):
            for sample in range(1 if radius == 0.0 else 72):
                angle = 2.0 * math.pi * sample / max(1, (1 if radius == 0.0 else 72))
                point = np.array(
                    [target[0] + radius * math.cos(angle), target[1],
                     target[2] + radius * math.sin(angle)], dtype=np.float32,
                )
                try:
                    snapped = np.asarray(pathfinder.snap_point(point), dtype=float)
                except Exception:
                    continue
                if not np.isfinite(snapped).all():
                    continue
                if self._object_rule.distances(snapped, target)[0] > threshold:
                    continue
                if self._object_rule.on_target_floor(snapped) is False:
                    # A point under a mezzanine snaps to the storey below; the
                    # shortest path must end where a stop would score.
                    continue
                key = tuple(np.round(snapped, 2))
                if key in seen:
                    continue
                seen.add(key)
                path = habitat_sim.ShortestPath()
                path.requested_start = start
                path.requested_end = np.asarray(snapped, dtype=np.float32)
                if pathfinder.find_path(path):
                    best = min(best, float(path.geodesic_distance))
        return best if math.isfinite(best) else None

    def _shortest_to_stop(self, attempts) -> Optional[float]:
        """Geodesic from the episode start to the STOP that scored.

        Used only when `_shortest_to_object`'s ring sampling found no pathable
        point inside the success radius. That stop is within `threshold_m` of
        the object by construction, so this measures the same quantity the
        sampling was after, from a point known to be both reachable and inside
        the radius.
        """
        import habitat_sim

        start = getattr(self, "_start_position", None)
        if start is None:
            return None
        won = next((a for a in attempts if a.get("success") and a.get("position")), None)
        if won is None:
            return None
        try:
            path = habitat_sim.ShortestPath()
            path.requested_start = np.asarray(start, dtype=np.float32)
            path.requested_end = np.asarray(won["position"], dtype=np.float32)
            if not self.env.sim.pathfinder.find_path(path):
                return None
            d = float(path.geodesic_distance)
        except Exception:
            return None
        return d if math.isfinite(d) and d > 0 else None

    def step(self, action: str):
        # Habitat ends the episode on the terminal STOP without routing it
        # through `attempt_scored`, so the rule has to see it here, before the
        # pose is gone (as sim/dualmap_env.py does).
        if action == "stop" and self._object_rule.enabled:
            self._record_stop()
        frame = super().step(action)
        self._maybe_relocate(frame)
        if self._object_rule.enabled:
            self._object_rule.travelled(
                np.asarray(self.env.sim.get_agent_state().position, dtype=float)
            )
        return frame

    def _target_position(self) -> Optional[np.ndarray]:
        info = (getattr(self.current_episode, "info", None) or {}).get("ycb", {})
        position = info.get("target_position")
        return None if position is None else np.asarray(position, dtype=float)

    def _record_stop(self) -> bool:
        target = self._target_position()
        if target is None:
            return False
        position = np.asarray(self.env.sim.get_agent_state().position, dtype=float)
        return self._object_rule.record_stop(int(self._frame_id), position, target)

    def attempt_scored(self, frame, cfg) -> Optional[bool]:
        """The released rule for spending an attempt, when it is switched on.

        Returning None hands the decision back to `eval/attempts.py`, whose
        fallback is habitat's viewpoint rule; that keeps every run without the
        flag exactly as it was.
        """
        if not self._object_rule.enabled:
            return None
        return self._record_stop()

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
        if self._object_rule.enabled:
            target = self._target_position()
            position = np.asarray(self.env.sim.get_agent_state().position, dtype=float)
            meta["object_distance"] = self._object_rule.summary(position, target)
        return meta

    def metrics(self) -> dict:
        """The object-distance rule's numbers, when it is on, with habitat's
        kept beside them under their own names (as sim/dualmap_env.py does)."""
        base = dict(self.env.get_metrics())
        if not self._object_rule.enabled:
            return base
        block = self.episode_metadata()["object_distance"]
        base["habitat_success"] = float(base.get("success", 0.0))
        base["habitat_spl"] = float(base.get("spl", 0.0))
        base["habitat_distance_to_goal"] = float(base.get("distance_to_goal", -1.0))
        base["success"] = float(block["success"])
        base["spl"] = float(block["spl"])
        base["distance_to_goal"] = float(block["final_distance_horizontal_m"])
        # Carry SPL's inputs through, so a recorded SPL can be audited against
        # the numbers it was computed from rather than taken on trust.
        for key in ("shortest_path_to_object_m", "travelled_m", "attempt_count"):
            base[key] = block.get(key)
        return base

    def benchmark_metadata(self) -> Dict[str, Any]:
        return {
            "mode": "ycb_authored",
            "success_rule": (
                self._object_rule.definition()
                if self._object_rule.enabled else "habitat viewpoint"
            ),
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
