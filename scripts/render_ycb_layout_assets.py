#!/usr/bin/env python3
"""Render authored YCB layouts from a shared orthographic camera.

Run inside the Habitat container::

    docker exec docker-nav-1 /opt/conda/envs/habitat/bin/python \
        scripts/render_ycb_layout_assets.py

The result contains the actual YCB meshes from the static, in-anchor, and
cross-anchor JSON layouts.  All layouts share the same camera and crop, so a
paper figure can compare their object poses without illustrative substitutes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAYOUT_ROOT = Path("/habitat-data-collector/outputs/dualmap_multifloor")
HM3D = Path("/habitat-data-collector/data/versioned_data/hm3d-0.2/hm3d")
YCB = Path("/habitat-data-collector/data/versioned_data/ycb/configs")


def template_handle(manager, desired: str) -> str:
    matches = []
    for handle in manager.get_file_template_handles():
        base = Path(str(handle)).name
        stem = base.split(".object_config", 1)[0].split(".", 1)[0]
        if stem == desired or desired in base:
            matches.append(str(handle))
    if not matches:
        raise RuntimeError(f"YCB template {desired!r} was not loaded")
    return min(matches, key=lambda value: (len(value), value))


def place_layout(sim, layout: dict) -> None:
    import habitat_sim
    import magnum as mn

    rigid = sim.get_rigid_object_manager()
    rigid.remove_all_objects()
    templates = sim.get_object_template_manager()
    mapping = {int(k): str(v) for k, v in layout["id_handle_mapping"].items()}
    for item in layout["objects"]:
        sid = int(item["semantic_id"])
        handle = mapping[sid]
        source = template_handle(templates, handle)
        obj = rigid.add_object_by_template_handle(source)
        obj.translation = mn.Vector3(*[float(v) for v in item["translation"]])
        q = [float(v) for v in item["rotation"]]
        obj.rotation = mn.Quaternion(mn.Vector3(q[0], q[1], q[2]), q[3])
        try:
            obj.semantic_id = sid
        except AttributeError:
            pass
        obj.motion_type = habitat_sim.physics.MotionType.STATIC


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="00873-bxsVRursffK")
    parser.add_argument("--layout-root", type=Path, default=DEFAULT_LAYOUT_ROOT)
    parser.add_argument("--layout-index", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs/figures/ycb_layouts")
    parser.add_argument("--resolution", type=int, default=1536)
    parser.add_argument("--span-m", type=float, default=14.0)
    parser.add_argument("--upper-y", type=float, default=1.0)
    parser.add_argument("--lower-y", type=float, default=-0.5)
    args = parser.parse_args()

    import habitat_sim

    scene_dir = args.layout_root / args.scene
    paths = {
        "static": scene_dir / "static_scene_config.json",
        "in_anchor": scene_dir / "dynamic_scene_config/in_anchor" / f"layout_{args.layout_index:02d}.json",
        "cross_anchor": scene_dir / "dynamic_scene_config/cross_anchor" / f"layout_{args.layout_index:02d}.json",
    }
    layouts = {name: json.loads(path.read_text()) for name, path in paths.items()}

    stem = args.scene.split("-", 1)[1]
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(HM3D / "val" / args.scene / f"{stem}.basis.glb")
    backend.scene_dataset_config_file = str(HM3D / "hm3d_annotated_basis.scene_dataset_config.json")
    backend.enable_physics = True
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    sensor.resolution = [args.resolution, args.resolution]
    sensor.position = np.zeros(3, dtype=np.float32)
    sensor.orientation = np.array([-np.pi / 2, 0, 0], dtype=np.float32)
    sensor.ortho_scale = 1.0 / args.span_m
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
    sim.get_object_template_manager().load_configs(str(YCB))
    lo, hi = sim.pathfinder.get_bounds()
    cx, cz = float((lo[0] + hi[0]) / 2), float((lo[2] + hi[2]) / 2)

    frames = {}
    for condition, layout in layouts.items():
        place_layout(sim, layout)
        for floor, height in (("upper", args.upper_y), ("lower", args.lower_y)):
            state = habitat_sim.AgentState()
            state.position = np.array([cx, height, cz], dtype=np.float32)
            state.rotation = np.quaternion(1, 0, 0, 0)
            sim.get_agent(0).set_state(state)
            frames[(condition, floor)] = np.asarray(
                sim.get_sensor_observations()["rgb"][..., :3], dtype=np.uint8
            )
    sim.close()

    union = np.zeros((args.resolution, args.resolution), dtype=bool)
    for frame in frames.values():
        union |= frame.sum(axis=2) > 18
    rows, cols = np.nonzero(union)
    if not rows.size:
        raise RuntimeError("orthographic renders are empty")
    pad = round(args.resolution * .025)
    crop = [max(0, int(cols.min()) - pad), max(0, int(rows.min()) - pad),
            min(args.resolution, int(cols.max()) + pad + 1),
            min(args.resolution, int(rows.max()) + pad + 1)]
    left, top, right, bottom = crop
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for (condition, floor), frame in frames.items():
        alpha = (frame.sum(axis=2) > 18).astype(np.uint8) * 255
        rgba = np.dstack((frame, alpha))[top:bottom, left:right]
        target = args.out_dir / f"{args.scene.split('-', 1)[0]}_{condition}_{floor}.png"
        imageio.imwrite(target, rgba)
        print(f"wrote {target}")
    meta = {
        "scene": args.scene, "layout_root": str(args.layout_root),
        "layout_index": args.layout_index, "resolution": args.resolution,
        "span_m": args.span_m, "crop": crop,
        "camera_center_xz": [cx, cz],
        "floor_camera_y": {"upper": args.upper_y, "lower": args.lower_y},
        "layout_files": {name: str(path) for name, path in paths.items()},
    }
    target = args.out_dir / f"{args.scene.split('-', 1)[0]}_meta.json"
    target.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
