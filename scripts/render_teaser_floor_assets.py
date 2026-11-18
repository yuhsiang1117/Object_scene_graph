#!/usr/bin/env python3
"""Render two aligned orthographic top-down views of a real HM3D scene.

Run in the Habitat Docker environment before composing the teaser::

    docker exec docker-nav-1 /opt/conda/envs/habitat/bin/python \
        scripts/render_teaser_floor_assets.py

The two views use the same camera and crop. Black void outside the scanned
mesh becomes transparent; the image is a scene render, not a logged trajectory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="00873-bxsVRursffK")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs/figures")
    parser.add_argument("--resolution", type=int, default=1536)
    parser.add_argument("--span-m", type=float, default=14.0)
    parser.add_argument("--upper-y", type=float, default=1.0)
    parser.add_argument("--lower-y", type=float, default=-0.5)
    args = parser.parse_args()

    import habitat_sim
    from osg.core.paths import hm3d_scene_root

    scene_root = Path(hm3d_scene_root())
    stem = args.scene.split("-", 1)[1]
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(scene_root / "val" / args.scene / f"{stem}.basis.glb")
    cfg.scene_dataset_config_file = str(
        scene_root / "hm3d_annotated_basis.scene_dataset_config.json"
    )
    cfg.enable_physics = False

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
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
    lo, hi = sim.pathfinder.get_bounds()
    cx, cz = float((lo[0] + hi[0]) / 2), float((lo[2] + hi[2]) / 2)

    frames = {}
    for name, height in (("upper", args.upper_y), ("lower", args.lower_y)):
        state = habitat_sim.AgentState()
        state.position = np.array([cx, height, cz], dtype=np.float32)
        state.rotation = np.quaternion(1, 0, 0, 0)
        sim.get_agent(0).set_state(state)
        frames[name] = np.asarray(
            sim.get_sensor_observations()["rgb"][..., :3], dtype=np.uint8
        )
    # A geometry-checked route through the actual switchback staircase. The
    # navmesh path is a guide for the explanatory overlay, not an episode log.
    def world_from_reference(u: float, v: float, height: float) -> np.ndarray:
        metres_per_px = args.span_m / 800.0
        return np.array([cx + (u - 400.0) * metres_per_px, height,
                         cz + (v - 400.0) * metres_per_px], dtype=np.float32)

    path = habitat_sim.ShortestPath()
    path.requested_start = sim.pathfinder.snap_point(
        world_from_reference(340, 340, 0.0)
    )
    path.requested_end = sim.pathfinder.snap_point(
        world_from_reference(210, 340, -3.0)
    )
    if not sim.pathfinder.find_path(path):
        raise SystemExit("the staircase path between the two floors is missing")
    guide = []
    for point in path.points:
        u = 400.0 + (float(point[0]) - cx) * 800.0 / args.span_m
        v = 400.0 + (float(point[2]) - cz) * 800.0 / args.span_m
        guide.append([round(u, 2), round(v, 2), round(float(point[1]), 3)])
    sim.close()

    masks = {name: image.sum(axis=2) > 18 for name, image in frames.items()}
    union = masks["upper"] | masks["lower"]
    rows, cols = np.nonzero(union)
    if not rows.size:
        raise SystemExit("orthographic render is empty")
    pad = round(args.resolution * 0.025)
    left = max(0, int(cols.min()) - pad)
    right = min(args.resolution, int(cols.max()) + pad + 1)
    top = max(0, int(rows.min()) - pad)
    bottom = min(args.resolution, int(rows.max()) + pad + 1)
    crop = [left, top, right, bottom]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    asset_id = args.scene.split("-", 1)[0]
    for name, frame in frames.items():
        rgba = np.dstack((frame, masks[name].astype(np.uint8) * 255))
        target = args.out_dir / f"hm3d_topdown_{asset_id}_{name}.png"
        imageio.imwrite(target, rgba[top:bottom, left:right])
        print(f"wrote {target}")
    meta = {"scene": args.scene, "resolution": args.resolution,
            "crop": crop, "span_m": args.span_m, "upper_y": args.upper_y,
            "lower_y": args.lower_y, "camera_center_xz": [cx, cz]}
    target = args.out_dir / f"hm3d_topdown_{asset_id}_meta.json"
    target.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {target}")
    target = args.out_dir / f"hm3d_topdown_{asset_id}_stair_path.json"
    target.write_text(json.dumps({"scene": args.scene,
                                  "geodesic_distance_m": round(float(path.geodesic_distance), 3),
                                  "points_uvy": guide}, indent=2) + "\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
