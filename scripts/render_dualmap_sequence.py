#!/usr/bin/env python3
"""Re-render DualMap's recorded mapping trajectory in a swapped static scene.

DualMap's released benchmark ships, per scene, the RGB-D sequence and camera
poses (`rgb/`, `depth/`, `pose.txt`) from which its static global map was
built. When the static scene changes (scripts/make_dualmap_swap.py), that map
has to be rebuilt from the same trajectory over the new scene, so this renders
every recorded pose with habitat-sim at the recorded intrinsics
(`camera_intrinsics.json`) and writes `rgb/<stamp>.png` and `depth/<stamp>.png`
(uint16 millimetres) next to a copied `pose.txt`. `--check N` renders N
recorded poses of the ORIGINAL scene instead and reports the pixel agreement
with the shipped frames, which is how the pose convention was verified.

    python scripts/render_dualmap_sequence.py --scene 00848-ziup5kvtCCR \
        --release $COLLECTOR_DATA/dualmap_swap/HM3D_collect --out .../rendered
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

# The collector mount moved from /datasets/habitat-data-collector to
# /habitat-data-collector; osg.core.paths carries both, so ask it.
from osg.core.paths import collector_data_root  # noqa: E402

_COLLECTOR = collector_data_root()

HM3D = Path(f"{_COLLECTOR}/scene_datasets/hm3d")
OBJECTS = Path(f"{_COLLECTOR}/objects/ycb/configs")
if not OBJECTS.is_dir():
    OBJECTS = Path(f"{_COLLECTOR}/versioned_data/ycb/configs")
if not HM3D.is_dir():
    from osg.core.paths import hm3d_scene_root  # noqa: E402

    HM3D = hm3d_scene_root()


def make_sim(scene: str, width: int, height: int, hfov_deg: float):
    import habitat_sim

    stem = scene.split("-", 1)[1]
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(HM3D / f"val/{scene}/{stem}.basis.glb")
    backend.scene_dataset_config_file = str(HM3D / "hm3d_annotated_basis.scene_dataset_config.json")
    backend.enable_physics = True
    specs = []
    for uuid, kind in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)):
        s = habitat_sim.CameraSensorSpec()
        s.uuid, s.sensor_type = uuid, kind
        s.resolution = [height, width]
        s.hfov = hfov_deg
        s.position = [0.0, 0.0, 0.0]
        specs.append(s)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = specs
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    return sim


def place_objects(sim, layout: dict) -> int:
    import magnum as mn

    mgr = sim.get_object_template_manager()
    rigid = sim.get_rigid_object_manager()
    mapping = {int(k): str(v) for k, v in layout["id_handle_mapping"].items()}
    n = 0
    for obj in layout["objects"]:
        handle = mapping[int(obj["semantic_id"])]
        cfg = OBJECTS / f"{handle}.object_config.json"
        ids = mgr.load_configs(str(cfg))
        tid = ids[0] if ids else mgr.get_template_id_by_handle(str(cfg))
        ro = rigid.add_object_by_template_id(tid)
        ro.translation = mn.Vector3(*[float(x) for x in obj["translation"]])
        q = [float(x) for x in obj["rotation"]]
        ro.rotation = mn.Quaternion(mn.Vector3(q[0], q[1], q[2]), q[3])
        ro.motion_type = habitat_motion_static()
        n += 1
    return n


def habitat_motion_static():
    import habitat_sim
    return habitat_sim.physics.MotionType.STATIC


def read_poses(path: Path):
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 17:
            continue
        out.append((parts[0], np.array([float(v) for v in parts[1:17]]).reshape(4, 4)))
    return out


# World frames: the collector records poses in its ROS frame (z up); habitat is y up.
W_ROS = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)
# Camera frames, as the matrix taking habitat-camera coordinates to the recorded ones.
C_FRAMES = {
    "habitat": np.eye(4),
    "opencv": np.diag([1.0, -1.0, -1.0, 1.0]),               # +z forward, +y down
    "rosbody": np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float),  # x forward, y left, z up
}
CONVENTIONS = [f"{w}/{c}" for w in ("id", "ros") for c in C_FRAMES]


def render_at(sim, T: np.ndarray, convention: str):
    import magnum as mn

    world, cam = convention.split("/")
    M = (W_ROS if world == "ros" else np.eye(4)) @ T @ C_FRAMES[cam]
    agent = sim.get_agent(0)
    node = agent.scene_node
    node.transformation = mn.Matrix4(M.T.tolist()) if False else mn.Matrix4(
        mn.Vector4(*M[:, 0]), mn.Vector4(*M[:, 1]), mn.Vector4(*M[:, 2]), mn.Vector4(*M[:, 3]))
    obs = sim.get_sensor_observations()
    return obs["rgb"][..., :3], obs["depth"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--release", type=Path, required=True, help="the release root holding the layouts to render")
    parser.add_argument("--recorded", type=Path,
                        default=Path(f"{_COLLECTOR}/dualmap/HM3D_collect"),
                        help="the original release with rgb/depth/pose.txt")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--check", type=int, default=0, help="render N original poses and compare with the shipped frames")
    parser.add_argument("--convention", default="ros/opencv", choices=CONVENTIONS)
    parser.add_argument("--every", type=int, default=1)
    args = parser.parse_args()

    rec = args.recorded / args.scene
    intr = json.loads((rec / "camera_intrinsics.json").read_text())
    W, H = int(intr["width"]), int(intr["height"])
    hfov = math.degrees(2.0 * math.atan(W / (2.0 * float(intr["fx"]))))
    poses = read_poses(rec / "pose.txt")
    layout_root = args.recorded if args.check else args.release
    layout = json.loads((layout_root / args.scene / "static_scene_config.json").read_text())
    sim = make_sim(args.scene, W, H, hfov)
    n = place_objects(sim, layout)
    print(f"{args.scene}: {len(poses)} poses, {n} objects placed, {W}x{H} hfov {hfov:.1f}", flush=True)

    if args.check:
        idx = np.linspace(0, len(poses) - 1, args.check).astype(int)
        for conv in CONVENTIONS:
            errs = []
            for i in idx:
                stamp, T = poses[i]
                rgb, depth = render_at(sim, T, conv)
                ref = cv2.imread(str(rec / "rgb" / f"{stamp}.png"))[..., ::-1]
                refd = cv2.imread(str(rec / "depth" / f"{stamp}.png"), -1).astype(np.float32) / 1000.0
                errs.append((float(np.abs(rgb.astype(np.float32) - ref.astype(np.float32)).mean()),
                             float(np.abs(depth - refd)[refd > 0].mean())))
            print(f"  convention {conv}: mean |rgb diff| {np.mean([e[0] for e in errs]):.1f}/255, "
                  f"mean |depth diff| {np.mean([e[1] for e in errs]):.3f} m over {len(idx)} frames", flush=True)
        sim.close()
        return

    out = args.out or (args.release / args.scene)
    if out.resolve() == (args.recorded / args.scene).resolve():
        raise SystemExit("refusing to write into the recorded release")
    for name in ("rgb", "depth"):
        if (out / name).is_symlink():
            (out / name).unlink()  # the swap copy links the originals; replace with real renders
        (out / name).mkdir(parents=True, exist_ok=True)
    for i, (stamp, T) in enumerate(poses):
        if i % args.every:
            continue
        rgb, depth = render_at(sim, T, args.convention)
        cv2.imwrite(str(out / "rgb" / f"{stamp}.png"), rgb[..., ::-1])
        cv2.imwrite(str(out / "depth" / f"{stamp}.png"), np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))
        if i % 500 == 0:
            print(f"  {i}/{len(poses)}", flush=True)
    for name in ("pose.txt", "camera_intrinsics.json"):
        if (out / name).is_symlink():
            (out / name).unlink()
        if not (out / name).exists():
            (out / name).write_text((rec / name).read_text())
    sim.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
