"""Two properties the Jetson deployment is built on (docs/THOR.md).

Neither is visible from any single file, and breaking either costs an image
rebuild on a robot to discover:

  the BRIDGE needs numpy and the standard library and nothing else, which is
  why it can be a `ros:humble` container with no CUDA -- matching the Stretch's
  own Ubuntu instead of compiling Humble against JetPack's Python;

  the PIPELINE runs without habitat-sim, which has no aarch64 build at all.

An `import torch` added to `osg/ros2/`, or a module-level `import habitat` on
the robot path, would leave both of those claims false and the docs wrong.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# What the bridge container does NOT install, and must never need to. Naming
# the forbidden set rather than an allowed one keeps this working on every
# interpreter and says what the test is actually for: each of these would turn
# the CUDA-free `ros:humble` container into something that has to carry the ML
# stack, at which point it can no longer match the robot's Ubuntu.
FORBIDDEN = (
    "torch", "torchvision", "cv2", "scipy", "sklearn", "skimage",
    "habitat", "habitat_sim", "magnum", "ultralytics", "matplotlib",
    "hydra", "omegaconf", "imageio", "PIL",
)
ROS_MODULES = (
    "rclpy", "nav2_msgs", "control_msgs", "sensor_msgs", "geometry_msgs",
    "std_msgs", "tf2_ros", "message_filters", "action_msgs",
    "builtin_interfaces", "trajectory_msgs",
)

_BRIDGE_PROBE = f"""
import sys

class _NoRos:
    def find_module(self, name, path=None):
        return self if name.split(".")[0] in {ROS_MODULES!r} else None
    def load_module(self, name):
        raise ImportError(name)

sys.meta_path.insert(0, _NoRos())
sys.path.insert(0, {str(ROOT / "src")!r})
before = set(sys.modules)
import osg.ros2.bridge_node        # noqa: F401
import osg.ros2.fake_robot         # noqa: F401
import osg.ros2.loopback           # noqa: F401
import osg.ros2.frames             # noqa: F401
import osg.ros2.transport          # noqa: F401
tops = {{m.split(".")[0] for m in set(sys.modules) - before}}
print(",".join(sorted(tops)))
"""

_ROBOT_PROBE = f"""
import sys

class _NoHabitat:
    def find_module(self, name, path=None):
        return self if name.split(".")[0] in ("habitat", "habitat_sim", "magnum") else None
    def load_module(self, name):
        raise ImportError(name + " has no aarch64 build")

sys.meta_path.insert(0, _NoHabitat())
sys.path.insert(0, {str(ROOT / "src")!r})

import numpy as np
from osg.core.config import OSGConfig
from osg.pipeline.components import build_agent, build_env, build_run_components
from osg.eval.episode import run_episode
from osg.graph.map_store import save_map

cfg = OSGConfig()
cfg.eval.mode = "ros2"
cfg.agent.navigation = "nav2"
cfg.agent.initial_scan = False
cfg.agent.max_steps = 6
cfg.detector.name = "stub"
cfg.verification.enabled = False
cfg.region_proposal.enabled = False
cfg.ros2.rotate_deg = 0.0
cfg.ros2.step_period_s = 0.0

# A loopback robot on a thread, the same one scripts/ros2 uses.
import threading
from multiprocessing.connection import Listener
from osg.ros2.loopback import LoopbackRobot, parse_args
from osg.ros2.wire import serve_once

listener = Listener(("127.0.0.1", 0), authkey=b"probe")
robot = LoopbackRobot(parse_args(["--width", "32", "--height", "24"]))
def _serve():
    while True:
        conn = listener.accept()
        try:
            while serve_once(conn, robot.handlers()):
                pass
        finally:
            conn.close()
threading.Thread(target=_serve, daemon=True).start()
cfg.ros2.bridge_addr = "%s:%d" % listener.address
cfg.ros2.authkey = "probe"

env = build_env(cfg)
components = build_run_components(cfg, env=env)
env.attach_driver(components["pointnav"])
frame = env.reset()
agent = build_agent(cfg, components, "chair")
outcome = run_episode(cfg, env, agent, env.current_episode, "chair", frame,
                      components["detector"])
import tempfile, pathlib
with tempfile.TemporaryDirectory() as d:
    save_map(pathlib.Path(d) / "m.json", agent, scene="probe")
env.close()
print("STEPS", outcome.steps)
"""


def _run(probe: str) -> str:
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout.strip()


def test_the_bridge_imports_nothing_the_ros_image_would_have_to_carry():
    """`docker/Dockerfile.bridge` installs `python3-numpy` and the ROS message
    packages, and nothing else. One `import torch` in `osg/ros2/` makes that
    image wrong -- and a bridge that needs CUDA cannot be a stock `ros:humble`
    container, which is what lets it match the Stretch's own Ubuntu."""
    pulled = set(_run(_BRIDGE_PROBE).split(",")) - {""}
    assert "numpy" in pulled, "the probe did not actually import the bridge"
    heavy = sorted(pulled.intersection(FORBIDDEN))
    assert not heavy, (
        f"osg.ros2 now imports {heavy} at module scope; "
        "docker/Dockerfile.bridge and docs/THOR.md both assume it does not")


def test_the_robot_path_runs_with_habitat_unavailable():
    """There is no aarch64 build of habitat-sim, so the Thor image has none.
    An `import habitat` anywhere on the robot path would only be discovered on
    the device, after the image was built and shipped."""
    assert _run(_ROBOT_PROBE).endswith("STEPS 6")


def test_costmap_to_occupancy_grid_puts_each_cell_where_frames_py_says():
    """grid[i, j] at pipeline (x, z); OccupancyGrid rows run along ROS y = -z."""
    import numpy as np

    from osg.ros2.bridge_node import grid_to_occupancy

    g = np.full((3, 2), -1, np.int8)      # 3 cells along x, 2 along z
    g[2, 0] = 100                          # x index 2, z index 0 -> pipeline (2.5, 0.5)*res + origin
    data, (ox, oy), (w, h) = grid_to_occupancy(g, (-1.0, -1.0), 0.5)
    assert (w, h) == (3, 2) and ox == -1.0 and oy == -(-1.0 + 2 * 0.5)    # y from -z: [0, 1] -> origin y = 0
    occ = data.reshape(h, w)               # rows: ROS y up; cols: x
    # pipeline z = -1 + 0.5*0.5 = -0.75 -> ROS y = 0.75 -> the TOP row (index 1); x = -1 + 2.5*0.5 = 0.25 -> col 2
    assert occ[1, 2] == 100 and (occ == 100).sum() == 1


def test_compressed_frames_decode_to_what_the_raw_ones_would():
    """jpeg colour and header+png depth, as image_transport publishes them."""
    import struct
    from types import SimpleNamespace

    import numpy as np
    cv2 = pytest.importorskip("cv2")
    from osg.ros2.bridge_node import compressed_to_array, is_compressed_topic

    assert is_compressed_topic("/camera/color/image_raw/compressed")
    assert is_compressed_topic("/camera/aligned_depth_to_color/image_raw/compressedDepth")
    assert not is_compressed_topic("/camera/color/image_raw")

    rgb = np.zeros((8, 12, 3), np.uint8); rgb[..., 0] = 200      # red in RGB
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(rgb[..., ::-1]))
    out, enc = compressed_to_array(SimpleNamespace(format="rgb8; png compressed bgr8", data=buf.tobytes()))
    assert enc == "rgb8" and out.shape == (8, 12, 3) and out[0, 0, 0] == 200 and out[0, 0, 2] == 0

    mm = np.arange(8 * 12, dtype=np.uint16).reshape(8, 12) * 37
    ok, buf = cv2.imencode(".png", mm)
    for fmt in ("16UC1; compressedDepth png", "16UC1; compressedDepth"):   # the Stretch names no codec
        msg = SimpleNamespace(format=fmt, data=struct.pack("<iff", 0, 0.0, 0.0) + buf.tobytes())
        out, enc = compressed_to_array(msg)
        assert enc == "16UC1" and out.dtype == np.uint16 and np.array_equal(out, mm)

    with pytest.raises(ValueError):
        compressed_to_array(SimpleNamespace(format="16UC1; compressedDepth rvl", data=b"\0" * 20))
