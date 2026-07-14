"""M0 smoke test: habitat-sim headless EGL init, load an HM3D minival scene,
step random actions, save an RGB + depth frame. This validates the Docker
EGL setup before anything else is debugged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs/smoke")


def main() -> int:
    import habitat_sim

    scene_root = Path("data/scene_datasets/hm3d")
    dataset_cfg = scene_root / "hm3d_annotated_basis.scene_dataset_config.json"
    if not dataset_cfg.exists():
        dataset_cfg = scene_root / "hm3d_basis.scene_dataset_config.json"
    glbs = sorted(scene_root.glob("*val*/*/*.basis.glb"))
    if not glbs:
        print("No HM3D scenes found under data/scene_datasets/hm3d — run scripts/download_data.py first.")
        print("EGL-only check instead...")
        return check_egl_only()

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(glbs[0])
    if dataset_cfg.exists():
        backend.scene_dataset_config_file = str(dataset_cfg)

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [480, 640]
    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [480, 640]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb, depth]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
    rng = np.random.default_rng(0)
    obs = None
    for _ in range(20):
        action = rng.choice(["move_forward", "turn_left", "turn_right"])
        obs = sim.step(str(action))

    OUT.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    imageio.imwrite(OUT / "frame.png", obs["rgb"][..., :3])
    d = obs["depth"]
    print(f"scene: {glbs[0].name}")
    print(f"rgb: {obs['rgb'].shape}, depth min/mean/max = {d.min():.2f}/{d.mean():.2f}/{d.max():.2f}")
    ok = d.max() > 0.5  # sane depth means rendering works
    sim.close()
    print("SMOKE OK" if ok else "SMOKE FAILED: depth all zero")
    return 0 if ok else 1


def check_egl_only() -> int:
    """No data yet: still verify the EGL context can be created."""
    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = "NONE"
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [240, 320]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
    obs = sim.get_sensor_observations()
    sim.close()
    print(f"EGL context OK, rendered {obs['rgb'].shape}")
    print("SMOKE OK (EGL only — download HM3D for the full test)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
