"""Turn the `ros2` config group into bridge command-line flags.

The bridge runs on the system python3.10, which has no hydra and no omegaconf,
so it cannot read the config itself. This runs in the habitat env, composes the
same config a run would, and prints the flags -- which is what makes
`ros2.depth_topic` in a yaml file actually change what the bridge subscribes
to. Without it every field here would be documentation for behaviour that does
not exist.

    python scripts/ros2/bridge_args.py [+experiment=stretch3_map] [ros2.rotate_deg=0]
"""
from __future__ import annotations

import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from osg.core.config import register_configs  # noqa: E402

# Config field -> bridge flag. Only fields the BRIDGE acts on: the pipeline
# side reads the rest (rotate_deg, depth_scale, goal_frame, ...) directly.
FLAGS = {
    "bridge_addr": "--bridge-addr",
    "authkey": "--authkey",
    "rgb_topic": "--rgb-topic",
    "depth_topic": "--depth-topic",
    "camera_info_topic": "--camera-info-topic",
    "sync_slop_s": "--sync-slop-s",
    "map_frame": "--map-frame",
    "camera_frame": "--camera-frame",
    "nav_action": "--nav-action",
    "cmd_vel_topic": "--cmd-vel-topic",
    "odom_frame": "--odom-frame",
    "base_frame": "--base-frame",
    "linear_speed": "--linear-speed",
    "angular_speed": "--angular-speed",
    "move_timeout_s": "--move-timeout-s",
    "head_traj_action": "--head-traj-action",
    "head_tilt_joint": "--head-tilt-joint",
    "floor_topic": "--floor-topic",
}


def main(argv=None) -> int:
    from hydra import compose, initialize_config_dir

    register_configs()
    overrides = list(sys.argv[1:] if argv is None else argv)
    configs = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=configs, version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    out = []
    for field, flag in FLAGS.items():
        value = getattr(cfg.ros2, field)
        # An empty camera_frame means "use the image header's", which the
        # bridge spells the same way; pass it through rather than dropping it.
        out += [flag, str(value)]
    print(" ".join(shlex.quote(token) for token in out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
