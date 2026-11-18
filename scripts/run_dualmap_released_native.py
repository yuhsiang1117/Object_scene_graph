#!/usr/bin/env python3
"""Re-run DualMap's released dynamic HM3D benchmark without ROS2.

The upstream simulation application uses ROS2 only as transport: Habitat Data
Collector publishes RGB-D/pose and receives DualMap's path.  This runner makes
the same calls directly, while retaining the released collector simulator,
sensor settings, random-start seed, velocity controller, raw dynamic layouts,
prebuilt maps, and DualMap configuration.

The paper's dynamic protocol contains 107 trials: six queried objects in each
of 18 layouts, except for the unreported cracker-box trial in scene 00880's
0128-2 cross-anchor layout.  A query succeeds when the agent stops within one
horizontal metre of any instance of the queried object, within three candidate
attempts.  SPL is additionally computed here; the paper reports SR only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from osg.core.paths import collector_data_root, hm3d_scene_root

# The benchmark itself -- which trials exist, what each query points at, and
# when a stop counts -- is defined once, in the package, because our own system
# is scored by importing the same module.  "Both measured by the same criterion"
# is then a property of the import graph rather than a claim in a document.
from osg.eval.dualmap_release import (
    CONDITIONS,
    HANDLE_TO_QUERY,
    MAX_ATTEMPTS,
    OBJECTS_ROOT,
    PUBLISHED,
    RELEASE_ROOT,
    STATIC_QUERIES,
    SUCCESS_DISTANCE_M,
    TARGETS,
    YCB_QUERIES,
    as_targets,
    distance_to_target,
    file_sha256,
    protocol,
    raw_layout,
    shortest_success_path,
    static_target_positions,
    target_positions,
)

WORKSPACE = Path("/workspace")
RUN_ROOT = WORKSPACE / "outputs/dualmap_released_native"
COMPARISON_ROOT = WORKSPACE / "outputs/dualmap_comparison"
DUALMAP_ROOT = COMPARISON_ROOT / "vendor/DualMap"
COLLECTOR_ROOT = COMPARISON_ROOT / "vendor/habitat-data-collector"
DATA_ROOT = Path(os.environ.get("OSG_DATA_ROOT", collector_data_root()))
HM3D_ROOT = Path(
    os.environ.get("OSG_HM3D_SCENE_ROOT", hm3d_scene_root())
)

COLLECTOR_SEED = 12
SENSOR_WIDTH = 1200
SENSOR_HEIGHT = 680
CAMERA_HEIGHT_M = 1.5
SIM_HZ = 30
OBSERVATION_TICKS = 10
ENDPOINT_GRACE_KEYFRAMES = 6










def git_head(path: Path) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()










def make_collector(scene: str, layout_path: Path):
    from omegaconf import OmegaConf

    sys.path.insert(0, str(COLLECTOR_ROOT))
    from habitat_data_collector.core.simulator_service import SimulatorService
    from habitat_data_collector.utils.config_factory import ConfigFactory
    import habitat_sim

    stem = scene.split("-", 1)[1]
    cfg = OmegaConf.create(
        {
            "default_agent": 0,
            "load_from_config": False,
            "scene_config": str(layout_path),
            # Use the immutable HM3D v0.2 tree directly.  The collector's
            # ``scene_datasets/hm3d`` link may be switched to v0.1 by a data
            # preparation job and must not silently change this benchmark.
            "scene_path": str(HM3D_ROOT / f"val/{scene}/{stem}.basis.glb"),
            "scene_dataset_config": str(
                HM3D_ROOT / "hm3d_annotated_basis.scene_dataset_config.json"
            ),
            "objects_path": str(OBJECTS_ROOT),
            "data_cfg": {
                "seed": COLLECTOR_SEED,
                "rgb": True,
                "depth": True,
                "semantic": True,
                "resolution": {"w": SENSOR_WIDTH, "h": SENSOR_HEIGHT},
                "camera_height": CAMERA_HEIGHT_M,
            },
            "movement_cfg": {
                "move_forward": 0.4,
                "move_backward": 0.4,
                "turn_left": 3,
                "turn_right": 3,
                "look_up": 3,
                "look_down": 3,
            },
            "physics_cfg": {"enable_physics": True},
        }
    )
    simulator = habitat_sim.Simulator(ConfigFactory.create_simulator_config(cfg))
    service = SimulatorService(simulator, cfg)

    # Simulator creation needs rebased scene paths.  Register only the handles
    # named by the released layout: the archived mount contains all 77 YCB
    # configs, whereas the authors' `new_objects` directory contained this
    # experiment's subset.  Calling register_object_templates on all 77 would
    # assign random IDs to the extras and can overwrite a released semantic ID.
    released = json.loads(layout_path.read_text(encoding="utf-8"))
    template_manager = service.get_object_template_manager()
    template_manager.load_configs(str(OBJECTS_ROOT))
    file_handles = list(template_manager.get_file_template_handles())
    handles: Dict[int, str] = {}
    for raw_id, desired in released["id_handle_mapping"].items():
        semantic_id = int(raw_id)
        candidates = [
            handle for handle in file_handles
            if Path(str(handle)).name.split(".", 1)[0] == desired
        ]
        if not candidates:
            raise RuntimeError(f"missing YCB object template {desired}")
        source = sorted(candidates, key=lambda value: (len(str(value)), str(value)))[0]
        attributes = template_manager.get_template_by_handle(source)
        attributes.semantic_id = semantic_id
        template_manager.register_template(attributes, desired)
        handles[semantic_id] = desired
    cfg.load_from_config = True
    objects = service.load_objects_from_config(handles)
    start_state = service.set_random_agent_position()
    service.step_physics(1.0 / SIM_HZ)
    return cfg, service, objects, np.asarray(start_state.position, dtype=float)


def setup_dualmap(scene: str, out: Path, query: str):
    from hydra import compose, initialize_config_dir

    out.mkdir(parents=True, exist_ok=True)
    actions = out / "actions.yaml"
    actions.write_text(
        f"calculate_path: false\ntrigger_find_next: false\n"
        f"get_goal_mode: inquiry\ninquiry_sentence: {query}\n",
        encoding="utf-8",
    )
    preload = RELEASE_ROOT / scene / "global_map/hm3d"
    with initialize_config_dir(config_dir=str(DUALMAP_ROOT / "config"), version_base="1.3"):
        cfg = compose(
            config_name="runner_ros",
            overrides=[
                "use_rerun=false",
                "use_rviz=false",
                "use_end_process=false",
                "use_parallel=true",
                "run_local_mapping_only=false",
                "save_local_map=false",
                "save_global_map=false",
                "preload_global_map=true",
                "preload_layout=true",
                "use_fastsam=false",
                f"scene_id={scene}",
                "dataset_name=self_collected",
                f"output_path={out}",
                f"map_save_path={out / 'runtime_map'}",
                f"preload_path={preload}",
                f"config_file_path={actions}",
                "yolo.given_classes_path=config/class_list/hm3d300_classes_ycb.txt",
            ],
        )
    # An asset swap (scripts/make_dualmap_swap.py) adds the new queries' names to
    # DualMap's detector list; DUALMAP_CLASS_LIST names that file (absolute).
    class_list = os.environ.get("DUALMAP_CLASS_LIST")
    if class_list:
        cfg.yolo.given_classes_path = str(Path(class_list).resolve())
    for group, key in (
        ("yolo", "model_path"),
        ("sam", "model_path"),
        ("fastsam", "model_path"),
        ("yolo", "given_classes_path"),
    ):
        if not Path(str(cfg[group][key])).is_absolute():
            cfg[group][key] = str(DUALMAP_ROOT / str(cfg[group][key]))
    cfg.ros_stream_config_path = str(
        DUALMAP_ROOT / "config/data_config/ros/self_collected.yaml"
    )
    return cfg


def data_input(service, index: int, timestamp: float):
    from scipy.spatial.transform import Rotation
    from utils.types import DataInput
    from habitat_data_collector.utils.coordinate_transform import CoordinateTransform

    observations = service.get_observations()
    color = np.asarray(observations["color_sensor"])[..., :3].copy()
    # Match the collector's 16UC1 ROS publication and DualMap's conversion back
    # to metres, including its millimetre quantisation.
    depth = (np.asarray(observations["depth_sensor"]) * 1000.0).astype(np.uint16)
    depth = depth.astype(np.float32)[..., None] / 1000.0
    sensor = service.get_agent_state().sensor_states["color_sensor"]
    pose_vector = CoordinateTransform.get_agent_pose_in_system(sensor)
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = Rotation.from_quat(pose_vector[3:]).as_matrix()
    pose[:3, 3] = pose_vector[:3]
    intrinsics = np.array(
        [[600.0, 0.0, 600.0], [0.0, 600.0, 340.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return DataInput(
        idx=index,
        time_stamp=timestamp,
        color=color,
        depth=depth,
        color_name=f"{timestamp:.6f}",
        intrinsics=intrinsics,
        pose=pose,
    )


class ReleasedFollower:
    """The released collector follower, driven without a ROS path message."""

    def __init__(self, service) -> None:
        sys.path.insert(0, str(COLLECTOR_ROOT))
        from habitat_data_collector.handlers.navigation_handler import (
            ContinuousPathFollower,
            filter_forward_path,
        )
        from habitat_data_collector.utils.coordinate_transform import CoordinateTransform

        self._class = ContinuousPathFollower
        self._filter = filter_forward_path
        self._transform = CoordinateTransform.transform_path_to_habitat
        self.service = service
        self.path_key: Optional[Tuple[Tuple[float, ...], ...]] = None
        self.follower = None

    @staticmethod
    def key(path: Optional[Sequence[Sequence[float]]]) -> Optional[Tuple[Tuple[float, ...], ...]]:
        if path is None:
            return None
        return tuple(tuple(round(float(value), 5) for value in point) for point in path)

    def update_path(self, path: Optional[Sequence[Sequence[float]]]) -> bool:
        key = self.key(path)
        if key is None or key == self.path_key:
            return False
        self.path_key = key
        state = self.service.get_agent_state()
        converted = [np.asarray(point, dtype=float) for point in self._transform(path)]
        # NavigationHandler deliberately fixes path height to the agent floor.
        converted = [np.array([point[0], state.position[1], point[2]]) for point in converted]
        converted = self._filter(state.position, state.rotation, converted)
        converted = [
            point for i, point in enumerate(converted)
            if i == 0 or np.linalg.norm(point - converted[i - 1]) > 1e-8
        ]
        if len(converted) < 2:
            self.follower = None
        else:
            self.follower = self._class(
                self.service.sim, converted, state, waypoint_threshold=0.1
            )
        return True

    def advance(self, ticks: int) -> Tuple[float, bool]:
        travelled = 0.0
        for _ in range(ticks):
            self.service.step_physics(1.0 / SIM_HZ)
            if self.follower is None or self.follower.progress >= 1.0:
                continue
            state = self.service.get_agent_state()
            before = np.asarray(state.position, dtype=float).copy()
            self.follower.set_state(state)
            position, rotation = self.follower.get_target_state()
            state.position = position
            state.rotation = rotation
            self.service.set_agent_state(state)
            after = np.asarray(position, dtype=float)
            travelled += float(np.linalg.norm(after[[0, 2]] - before[[0, 2]]))
        done = self.follower is None or self.follower.progress >= 1.0
        return travelled, done








def diagnose_local_map(core, targets, query: str) -> Dict[str, Any]:
    """Ask DualMap what it holds and what it would pick, using its own methods.

    Answers two questions the success flag cannot:
      * did the relocated object ever enter the concrete map, and under what label
      * which object does the inquiry matcher actually select

    Nothing here mutates DualMap; `filter_objects_in_global_bbox` and
    `find_best_candidate_with_inquiry` are the calls its own planner makes.
    Local-map coordinates are DualMap's z-up frame, so Habitat (x, z) is (x, -y).
    """
    local = core.local_map_manager
    names = local.visualizer.obj_classes.get_classes_arr()

    def to_habitat(centre):
        return float(centre[0]), float(-centre[1])

    def gap(px, pz):
        best = math.inf
        for centre, half in targets:
            dx = max(abs(px - centre[0]) - half[0], 0.0)
            dz = max(abs(pz - centre[2]) - half[2], 0.0)
            best = min(best, float(math.hypot(dx, dz)))
        return best

    nearest_any = math.inf
    nearest_any_class = None
    nearest_same_class = math.inf
    mapped_objects = 0
    for obj in list(local.local_map):
        try:
            px, pz = to_habitat(obj.bbox.get_center())
        except Exception:
            continue
        mapped_objects += 1
        distance = gap(px, pz)
        label = names[obj.class_id] if obj.class_id < len(names) else str(obj.class_id)
        if distance < nearest_any:
            nearest_any, nearest_any_class = distance, label
        if label == query and distance < nearest_same_class:
            nearest_same_class = distance

    selected_class = None
    selected_distance = None
    selected_score = None
    try:
        if local.global_bbox is not None:
            candidates = local.filter_objects_in_global_bbox(expand_ratio=0.1)
            if candidates:
                chosen, score = local.find_best_candidate_with_inquiry(candidates)
                if chosen is not None:
                    px, pz = to_habitat(chosen.bbox.get_center())
                    selected_distance = gap(px, pz)
                    selected_score = float(score)
                    selected_class = (
                        names[chosen.class_id] if chosen.class_id < len(names) else str(chosen.class_id)
                    )
    except Exception as exc:  # diagnostics must never fail a trial
        selected_class = f"<error: {type(exc).__name__}>"

    return {
        "mapped_objects": mapped_objects,
        "nearest_mapped_object_m": None if not math.isfinite(nearest_any) else nearest_any,
        "nearest_mapped_object_class": nearest_any_class,
        "nearest_same_class_object_m": None if not math.isfinite(nearest_same_class) else nearest_same_class,
        "selected_class": selected_class,
        "selected_distance_to_truth_m": selected_distance,
        "selected_score": selected_score,
    }


def run_trial(trial: Dict[str, Any], out: Path, max_keyframes: int) -> Dict[str, Any]:
    from omegaconf import OmegaConf

    layout_path = Path(trial["layout_path"])
    if trial["condition"] == "static":
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        targets = static_target_positions(trial["scene"], trial["query"])
    else:
        layout = raw_layout(trial["scene"], trial["condition"], trial["layout"])
        targets = as_targets(target_positions(layout, trial["query"]))
    _, service, objects, start = make_collector(trial["scene"], layout_path)
    cfg = setup_dualmap(trial["scene"], out, trial["query"])
    OmegaConf.save(cfg, out / "dualmap_config.yaml")

    sys.path.insert(0, str(DUALMAP_ROOT))
    from dualmap.core import Dualmap
    core = None
    complete = threading.Event()
    trace_path = out / "trace.jsonl"
    shortest = shortest_success_path(service.sim.pathfinder, start, targets)
    follower = ReleasedFollower(service)
    attempts: List[Dict[str, Any]] = []
    travelled = 0.0
    timestamp = 0.0
    keyframes = 0
    terminal = "keyframe_budget"
    success = False
    retry_stage = 0
    last_plan_counter = -1
    endpoint_wait = 0
    false_matches = 0
    visited_uids: set = set()
    plan_diagnostics: List[Dict[str, Any]] = []
    diagnosed_attempt = -1

    def write_actions(*, calculate_path: bool, trigger_find_next: bool) -> None:
        actions = Path(str(cfg.config_file_path))
        actions.write_text(
            OmegaConf.to_yaml(
                OmegaConf.create(
                    {
                        "calculate_path": calculate_path,
                        "trigger_find_next": trigger_find_next,
                        "get_goal_mode": "inquiry",
                        "inquiry_sentence": trial["query"],
                    }
                )
            ),
            encoding="utf-8",
        )

    def wait_for(predicate, description: str, timeout_s: float = 30.0) -> None:
        import time

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        raise RuntimeError(f"DualMap config monitor did not {description}")

    try:
        core = Dualmap(cfg)
        original_process = core.global_map_manager.process_observations

        def tracked_process(observations):
            try:
                return original_process(observations)
            finally:
                complete.set()

        core.global_map_manager.process_observations = tracked_process
        # Use the released actions.yaml monitor rather than mutating navigation
        # flags during detector inference; the monitor is concurrent upstream.
        write_actions(calculate_path=True, trigger_find_next=False)
        wait_for(lambda: core.calculate_path, "accept the initial path request")

        with trace_path.open("w", encoding="utf-8") as trace:
            while keyframes < max_keyframes:
                datum = data_input(service, keyframes + 1, timestamp)
                if not core.check_keyframe(datum.time_stamp, datum.pose):
                    delta, _ = follower.advance(OBSERVATION_TICKS)
                    travelled += delta
                    timestamp += OBSERVATION_TICKS / SIM_HZ
                    continue

                datum.idx = core.get_keyframe_idx()
                complete.clear()
                core.parallel_process(datum)
                if not complete.wait(120.0):
                    raise RuntimeError("DualMap mapping thread did not finish within 120 seconds")
                keyframes += 1

                if retry_stage == 1:
                    wait_for(
                        lambda: core.calculate_path and not core.trigger_find_next,
                        "advance from find-next to global replanning",
                    )
                    retry_stage = 2
                elif retry_stage == 2:
                    retry_stage = 0

                path_changed = follower.update_path(core.action_path)
                if path_changed:
                    endpoint_wait = 0
                if core.path_counter != last_plan_counter:
                    last_plan_counter = core.path_counter
                    # Snapshot the ignore list at plan time.  DualMap appends the
                    # chosen anchor to it here, and clears the whole list later if
                    # a local path is planned, so this is the only point at which
                    # the anchors already tried are all observable.
                    visited_uids.update(core.global_map_manager.ignore_global_obj_list)
                    plan = {
                        "event": "global_plan",
                        "keyframe": keyframes,
                        "attempt": len(attempts) + 1,
                        "path_counter": int(core.path_counter),
                        "candidate": core.global_map_manager.best_candidate_name,
                        "score": float(core.global_map_manager.global_candidate_score),
                        "ignored_candidates": len(core.global_map_manager.ignore_global_obj_list),
                    }
                    trace.write(json.dumps(plan) + "\n")

                state = service.get_agent_state()
                horizontal, spatial = distance_to_target(state.position, targets)
                trace.write(
                    json.dumps(
                        {
                            "event": "keyframe",
                            "keyframe": keyframes,
                            "timestamp": timestamp,
                            "position": np.asarray(state.position, dtype=float).tolist(),
                            "distance_horizontal_m": horizontal,
                            "distance_3d_m": spatial,
                            "path_changed": path_changed,
                            "has_path": core.action_path is not None,
                            "has_local_path": core.curr_local_path is not None,
                            "candidate": core.global_map_manager.best_candidate_name,
                            "local_objects": len(core.local_map_manager.local_map),
                            "travelled_m": travelled,
                        }
                    )
                    + "\n"
                )
                trace.flush()

                if core.curr_local_path is not None and diagnosed_attempt != len(attempts):
                    diagnosed_attempt = len(attempts)
                    try:
                        snapshot = diagnose_local_map(core, targets, trial["query"])
                    except Exception as exc:
                        snapshot = {"error": f"{type(exc).__name__}: {exc}"}
                    snapshot.update({"attempt": len(attempts) + 1, "keyframe": keyframes})
                    plan_diagnostics.append(snapshot)
                    trace.write(json.dumps({"event": "local_plan_diagnostic", **snapshot}) + "\n")
                    trace.flush()

                delta, endpoint = follower.advance(OBSERVATION_TICKS)
                travelled += delta
                timestamp += OBSERVATION_TICKS / SIM_HZ
                if not endpoint or retry_stage:
                    endpoint_wait = 0
                    continue

                state = service.get_agent_state()
                horizontal, spatial = distance_to_target(state.position, targets)
                local_path_finished = core.curr_local_path is not None
                if (
                    horizontal > SUCCESS_DISTANCE_M
                    and not local_path_finished
                    and endpoint_wait < ENDPOINT_GRACE_KEYFRAMES
                ):
                    endpoint_wait += 1
                    continue
                attempt = {
                    "attempt": len(attempts) + 1,
                    "keyframe": keyframes,
                    "position": np.asarray(state.position, dtype=float).tolist(),
                    "distance_horizontal_m": horizontal,
                    "distance_3d_m": spatial,
                    "success": horizontal <= SUCCESS_DISTANCE_M,
                    "candidate": core.global_map_manager.best_candidate_name,
                    "candidate_score": float(core.global_map_manager.global_candidate_score),
                    "local_path_found": core.curr_local_path is not None,
                    "travelled_m": travelled,
                }
                try:
                    attempt["diagnostic"] = diagnose_local_map(core, targets, trial["query"])
                except Exception as exc:
                    attempt["diagnostic"] = {"error": f"{type(exc).__name__}: {exc}"}
                attempts.append(attempt)
                endpoint_wait = 0
                trace.write(json.dumps({"event": "attempt_end", **attempt}) + "\n")
                trace.flush()
                if attempt["success"]:
                    success = True
                    terminal = "success"
                    break
                false_match = core.curr_local_path is not None
                if false_match:
                    false_matches += 1
                if len(attempts) >= MAX_ATTEMPTS:
                    terminal = "false_match" if false_match else "attempts_exhausted"
                    break

                if false_match:
                    # A false match is a completed attempt, not the end of the
                    # trial: the paper counts a query successful if the target is
                    # found within three attempts, and the guide documents
                    # trigger_find_next as the way to continue after a failed
                    # attempt.  DualMap clears ignore_global_obj_list whenever a
                    # local path is planned (core.py), which would make the next
                    # global plan re-select the anchor just rejected, so the
                    # anchors already tried are restored before replanning.
                    # Only DualMap's public state is set; its source is unchanged.
                    core.global_map_manager.ignore_global_obj_list = list(visited_uids)
                    write_actions(calculate_path=True, trigger_find_next=False)
                    wait_for(lambda: core.calculate_path, "accept the retry path request")
                    follower.follower = None
                    endpoint_wait = 0
                    continue

                # This is the direct equivalent of setting trigger_find_next in
                # actions.yaml.  One observation lets the local planner consume
                # the flag; the following observation requests the next global
                # candidate, preserving the upstream ignore list.
                write_actions(calculate_path=False, trigger_find_next=True)
                wait_for(lambda: core.trigger_find_next, "accept the find-next request")
                retry_stage = 1
                follower.follower = None

        final_state = service.get_agent_state()
        horizontal, spatial = distance_to_target(final_state.position, targets)
        spl = 0.0
        if success and math.isfinite(shortest):
            spl = shortest / max(shortest, travelled, 1e-12)
        result = {
            **trial,
            "success": int(success),
            "spl": float(spl),
            "shortest_success_path_m": None if not math.isfinite(shortest) else shortest,
            "travelled_m": travelled,
            "final_distance_horizontal_m": horizontal,
            "final_distance_3d_m": spatial,
            "attempts": attempts,
            "attempt_count": len(attempts),
            "false_match_count": false_matches,
            "plan_diagnostics": plan_diagnostics,
            "keyframes": keyframes,
            "terminal": terminal,
            "start_position": start.tolist(),
            "target_positions": [centre.tolist() for centre, _ in targets],
            "target_half_extents": [half.tolist() for _, half in targets],
            "success_definition": "horizontal distance to any queried instance <= 1.0 m",
            "transport": "direct Python calls; ROS2 serialization bypassed",
            "raw_layout_object_count": len(layout["objects"]),
            "runtime_object_count": len(objects),
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        if core is not None and not core.stop_thread:
            core.stop_threading()
        service.sim.close()


def aggregate(run_root: Path, trials: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    for trial in trials:
        path = run_root / "trials" / trial["trial_id"] / "result.json"
        if path.is_file():
            results.append(json.loads(path.read_text(encoding="utf-8")))

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[(result["scene"], result["condition"])].append(result)

    expected: Dict[Tuple[str, str], int] = defaultdict(int)
    for trial in trials:
        expected[(trial["scene"], trial["condition"])] += 1

    rows = []
    for scene in TARGETS:
        for condition in CONDITIONS:
            values = groups[(scene, condition)]
            published_success, published_trials = PUBLISHED[(scene, condition)]
            rows.append(
                {
                    "scene": scene,
                    "condition": condition,
                    "completed": len(values),
                    "expected": expected[(scene, condition)],
                    "published_trials": published_trials,
                    "successes": sum(int(value["success"]) for value in values),
                    "sr": None if not values else sum(int(value["success"]) for value in values) / len(values),
                    "spl": None if not values else sum(float(value["spl"]) for value in values) / len(values),
                    "published_successes": published_success,
                    "published_sr": published_success / published_trials,
                }
            )
    conditions = []
    for condition in CONDITIONS:
        values = [value for value in results if value["condition"] == condition]
        published_success = sum(
            PUBLISHED[(scene, condition)][0] for scene in TARGETS
        )
        published_trials = sum(
            PUBLISHED[(scene, condition)][1] for scene in TARGETS
        )
        conditions.append(
            {
                "condition": condition,
                "completed": len(values),
                "expected": sum(
                    count for (_, cond), count in expected.items() if cond == condition
                ),
                "successes": sum(int(value["success"]) for value in values),
                "sr": None if not values else sum(int(value["success"]) for value in values) / len(values),
                "spl": None if not values else sum(float(value["spl"]) for value in values) / len(values),
                "published_sr": published_success / published_trials,
                "published_trials": published_trials,
            }
        )

    summary = {
        "seed": COLLECTOR_SEED,
        "completed": len(results),
        "expected": len(trials),
        "successes": sum(int(value["success"]) for value in results),
        "sr": None if not results else sum(int(value["success"]) for value in results) / len(results),
        "spl": None if not results else sum(float(value["spl"]) for value in results) / len(results),
        "paper_reports_spl": False,
        "conditions": conditions,
        "groups": rows,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def write_manifest(run_root: Path, trials: Sequence[Dict[str, Any]]) -> None:
    manifest = {
        "schema_version": 1,
        "description": "DualMap released dynamic HM3D navigation protocol without ROS2 transport",
        "dualmap_commit": git_head(DUALMAP_ROOT),
        "collector_commit": git_head(COLLECTOR_ROOT),
        "release_root": str(RELEASE_ROOT),
        "collector_seed": COLLECTOR_SEED,
        "sensor": {
            "width": SENSOR_WIDTH,
            "height": SENSOR_HEIGHT,
            "camera_height_m": CAMERA_HEIGHT_M,
            "intrinsics": [600.0, 600.0, 600.0, 340.0],
        },
        "success_distance_m": SUCCESS_DISTANCE_M,
        "max_attempts": MAX_ATTEMPTS,
        "endpoint_grace_keyframes": ENDPOINT_GRACE_KEYFRAMES,
        "endpoint_grace_simulated_seconds": (
            ENDPOINT_GRACE_KEYFRAMES * OBSERVATION_TICKS / SIM_HZ
        ),
        "paper_reports_spl": False,
        "trials": list(trials),
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "protocol.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def select_trials(all_trials: Sequence[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    selected = list(all_trials)
    if args.scene:
        selected = [trial for trial in selected if trial["scene"].startswith(args.scene)]
    if args.condition:
        selected = [trial for trial in selected if trial["condition"] == args.condition]
    if args.trial:
        selected = [trial for trial in selected if trial["trial_id"] == args.trial]
    if args.max_trials is not None:
        selected = selected[: args.max_trials]
    return selected


def main() -> None:
    global COLLECTOR_SEED

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUN_ROOT)
    parser.add_argument("--scene", help="full scene name or numeric prefix")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument(
        "--seed", type=int, default=COLLECTOR_SEED,
        help="collector seed; controls the agent start pose (default: %(default)s)",
    )
    parser.add_argument("--trial", help="run one exact protocol trial ID")
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--max-keyframes", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list", action="store_true", help="print selected trials and exit")
    args = parser.parse_args()
    args.output = args.output.resolve()

    COLLECTOR_SEED = args.seed

    os.environ.setdefault("YOLO_CONFIG_DIR", str(COMPARISON_ROOT / "yolo_config"))
    random.seed(COLLECTOR_SEED)
    np.random.seed(COLLECTOR_SEED)
    logging.basicConfig(level=logging.WARNING)
    # Upstream has one malformed warning call in the local planner that emits
    # a full logging traceback every frame.  It does not affect control flow;
    # suppressing that logger keeps long benchmark logs usable.
    logging.getLogger("utils.local_map_manager").setLevel(logging.ERROR)
    trials = protocol()
    selected = select_trials(trials, args)
    if not selected:
        raise SystemExit("no protocol trials matched the selection")
    if args.list:
        print(json.dumps(selected, indent=2))
        return

    write_manifest(args.output, trials)
    os.chdir(DUALMAP_ROOT)
    for index, trial in enumerate(selected, start=1):
        out = args.output / "trials" / trial["trial_id"]
        result_path = out / "result.json"
        if result_path.is_file() and not args.overwrite:
            print(f"SKIP {index}/{len(selected)} {trial['trial_id']}", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"RUN {index}/{len(selected)} {trial['trial_id']}", flush=True)
        try:
            result = run_trial(trial, out, args.max_keyframes)
            print(
                f"DONE {trial['trial_id']} success={result['success']} "
                f"spl={result['spl']:.3f} attempts={result['attempt_count']} "
                f"keyframes={result['keyframes']}",
                flush=True,
            )
        except Exception as exc:
            failure = {**trial, "error": type(exc).__name__, "message": str(exc)}
            (out / "error.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
            logging.exception("trial failed: %s", trial["trial_id"])
        summary = aggregate(args.output, trials)
        print(
            f"PROGRESS {summary['completed']}/{summary['expected']} "
            f"SR={summary['sr']} SPL={summary['spl']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
