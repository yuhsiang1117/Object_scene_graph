#!/usr/bin/env python3
"""M0 smoke: can habitat-sim open a mounted HM3D scene and render through EGL?

This is the check to run first when a container is new or a driver has moved.
It is deliberately narrower than `pytest tests/integration`: no agent, no
detector, no authored layout -- just a simulator, one scene, one frame. If this
fails, nothing else in the repo can work, and the error you get here is about
EGL or the mount rather than about the pipeline.

    python scripts/smoke_habitat.py                    # first val scene found
    python scripts/smoke_habitat.py --scene 00800-TEEsavR23oF

Exits non-zero on failure, so it is usable as a gate in a shell script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from osg.core.paths import hm3d_scenes_dir  # noqa: E402


def find_scene(scenes_dir: Path, scene: str | None) -> tuple[Path, Path]:
    """(glb, scene_dataset_config) for `scene`, or for the first one found."""
    root = scenes_dir / "hm3d"
    if not root.is_dir():
        raise SystemExit(f"no HM3D corpus under {scenes_dir} -- check the mount")
    pattern = f"*/{scene}/*.basis.glb" if scene else "*/*/*.basis.glb"
    globs = sorted(p for p in root.glob(pattern) if ".semantic." not in p.name)
    if not globs:
        raise SystemExit(f"no scene glb under {root} matching {pattern!r}")
    glb = globs[0]
    # The split-level config is what habitat expects; fall back to the corpus one.
    for candidate in (glb.parent.parent, root):
        configs = sorted(candidate.glob("*.scene_dataset_config.json"))
        if configs:
            return glb, configs[0]
    raise SystemExit(f"no scene_dataset_config.json near {glb}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default=None, help="e.g. 00800-TEEsavR23oF")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    args = ap.parse_args()

    import habitat_sim

    scenes_dir = hm3d_scenes_dir()
    glb, dataset_cfg = find_scene(scenes_dir, args.scene)
    print(f"scenes_dir  {scenes_dir}")
    print(f"scene       {glb.parent.name}")
    print(f"dataset     {dataset_cfg.name}")

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(glb)
    backend.scene_dataset_config_file = str(dataset_cfg)
    backend.enable_physics = False

    specs = []
    for uuid, sensor_type in (
        ("rgb", habitat_sim.SensorType.COLOR),
        ("depth", habitat_sim.SensorType.DEPTH),
    ):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [args.height, args.width]
        spec.position = [0.0, 1.25, 0.0]
        specs.append(spec)

    sim = habitat_sim.Simulator(
        habitat_sim.Configuration(
            backend, [habitat_sim.agent.AgentConfiguration(sensor_specifications=specs)]
        )
    )
    try:
        state = sim.get_agent(0).get_state()
        state.position = sim.pathfinder.get_random_navigable_point()
        sim.get_agent(0).set_state(state)
        obs = sim.get_sensor_observations()
        rgb, depth = obs["rgb"], obs["depth"]
        print(f"rgb         {rgb.shape} {rgb.dtype}")
        print(f"depth       {depth.shape} {depth.dtype}")
        # An all-black frame is what a broken EGL context renders, and it is the
        # failure this script exists to catch -- the Simulator itself constructs
        # happily without a working GPU context.
        if not rgb[..., :3].any():
            print("FAIL: rendered frame is entirely black -- EGL is not rendering")
            return 1
        if not (depth > 0).any():
            print("FAIL: depth is entirely zero")
            return 1
        print(f"navmesh     {sim.pathfinder.is_loaded and 'loaded' or 'MISSING'}")
        print("OK: EGL rendering works")
        return 0
    finally:
        sim.close()


if __name__ == "__main__":
    raise SystemExit(main())
