"""Is the ROS layer wired up correctly? Ask it, with no robot in the room.

Run through `bash scripts/ros2/check_pipeline.sh`, which starts the fake robot
and the bridge first. This half runs in the habitat env and imports no ROS at
all -- which is itself part of the check: the pipeline must reach the robot
through the socket and nothing else.

The one that matters is the goal round trip. A goal leaves the pipeline in
habitat-world metres, becomes a ROS `map` pose, is driven to by a navigator,
and comes back as a camera pose in habitat-world metres. Each conversion is
unit-tested; only here do they have to be inverses of each other in practice,
and a sign error anywhere in that loop is a robot that drives to the mirror
image of where it meant to go.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osg.core.config import OSGConfig  # noqa: E402
from osg.mapping.costmap import HEIGHT_AXIS, PLANE  # noqa: E402
from osg.planning.nav2_backends import RosNav2Backend  # noqa: E402
from osg.planning.nav2_driver import Nav2Driver  # noqa: E402
from osg.ros2 import frames  # noqa: E402
from osg.sim.ros2_env import Ros2Env  # noqa: E402

TOL_M = 0.35  # the fake navigator's own goal tolerance is 0.2 m


class Checks:
    def __init__(self) -> None:
        self.rows = []

    def check(self, name, ok, why="", info=""):
        """`why` explains a failure; `info` is context worth seeing either way."""
        detail = str(info) if ok else (str(why) or str(info))
        self.rows.append((bool(ok), str(name), detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)
        return bool(ok)

    def report(self) -> int:
        failed = [r for r in self.rows if not r[0]]
        print(f"\n{len(self.rows) - len(failed)}/{len(self.rows)} checks passed")
        for _, name, detail in failed:
            print(f"  FAILED: {name} -- {detail}")
        return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-out", default="outputs/ros2_check/frame.png",
                        help="where to save a converted RGB frame, so the "
                             "camera rotation can be settled by looking at it")
    parser.add_argument("--no-floor-switch", action="store_true",
                        help="skip the floor switch check (the fake robot must "
                             "be started with --floor-switch-after)")
    parser.add_argument("--loopback", action="store_true",
                        help="run against the ROS-free fake robot in this "
                             "process, instead of a bridge and rclpy. Checks "
                             "everything above the bridge with no ROS installed.")
    args = parser.parse_args(argv)

    if args.loopback:
        _start_loopback()

    cfg = OSGConfig()
    cfg.eval.mode = "ros2"
    # The fake publishes upright; the Stretch's head camera does not. Rotation
    # has its own unit tests, and leaving it off here keeps the geometry checks
    # about the ROS conversions.
    cfg.ros2.rotate_deg = 0.0
    cfg.agent.max_steps = 500
    checks = Checks()

    print("\n== the bridge ==")
    env = Ros2Env(cfg)
    info = env.transport.ping()
    checks.check("the bridge answers", info.get("node") == "osg_bridge",
                 why=str(info), info=f"frames seen: {info.get('frames')}")
    checks.check("CameraInfo has arrived", info.get("has_camera_info"),
                 why="no CameraInfo: check ros2.camera_info_topic")

    print("\n== a frame ==")
    frame = env.reset()
    checks.check("rgb is an 8-bit colour image",
                 frame.rgb.ndim == 3 and frame.rgb.shape[2] == 3
                 and frame.rgb.dtype == np.uint8, info=str(frame.rgb.shape))
    checks.check("depth is float32 metres",
                 frame.depth.dtype == np.float32 and frame.depth.shape == frame.rgb.shape[:2],
                 info=f"{frame.depth.dtype} {frame.depth.shape}")
    valid = frame.depth[frame.depth > 0]
    checks.check("depth has plausible readings",
                 valid.size > 0 and 0.1 < float(valid.mean()) < 20.0,
                 info=f"{valid.size} valid, mean "
                      f"{float(valid.mean()) if valid.size else 0:.2f} m")
    K = frame.intrinsics.K()
    checks.check("intrinsics are finite and positive",
                 bool(np.isfinite(K).all()) and frame.intrinsics.fx > 0,
                 info=f"fx={frame.intrinsics.fx:.1f} cx={frame.intrinsics.cx:.1f}")
    checks.check("the pose is a rotation, not a reflection",
                 frames.looks_like_rotation(frame.T_wc),
                 info=f"det={np.linalg.det(frame.T_wc[:3, :3]):.4f}")
    height = float(frame.camera_position[HEIGHT_AXIS])
    checks.check("the camera stands above the floor", 0.2 < height < 2.5,
                 info=f"{height:.2f} m on the height axis")

    saved = _save_rgb(frame.rgb, args.rgb_out)
    print(f"  ...wrote {saved} -- look at it to settle ros2.rotate_deg")

    print("\n== a goal, there and back ==")
    backend = RosNav2Backend(env.transport, cfg.ros2)
    # stop_radius=0 so arrival is the NAVIGATOR's verdict (SUCCEEDED) and not
    # our own proximity test -- otherwise this measures the driver's radius
    # rather than whether the robot got where the pipeline asked.
    driver = Nav2Driver(backend, stop_radius=0.0,
                        goal_resend_m=float(cfg.ros2.goal_resend_m))
    env.attach_driver(driver)
    driver.observe(frame)

    start_xy = frame.camera_position[list(PLANE)].copy()
    goal_xy = start_xy + np.array([1.2, -0.8])
    step = driver.step(goal_xy)
    checks.check("the mover accepted the goal", step.reason == "moving",
                 why=str(step))

    sent = env.transport.last_goal()
    expect_x, expect_y = frames.pipeline_xy_to_ros(goal_xy)
    checks.check(
        "Nav2 received the pipeline's goal, converted",
        sent is not None
        and abs(sent["x"] - expect_x) < 1e-6 and abs(sent["y"] - expect_y) < 1e-6,
        why=f"sent {sent}, expected x={expect_x:.3f} y={expect_y:.3f}",
        info=f"({goal_xy[0]:.2f}, {goal_xy[1]:.2f}) pipeline -> "
             f"({expect_x:.2f}, {expect_y:.2f}) ros")
    checks.check("the goal is in the frame Nav2 plans in",
                 sent is not None and sent["frame_id"] == cfg.ros2.goal_frame,
                 why=f"{sent['frame_id'] if sent else None!r} is not "
                     f"{cfg.ros2.goal_frame!r}")

    arrived, frame = _drive_until_arrived(env, driver, frame, goal_xy, timeout_s=30.0)
    checks.check("the navigator reported arriving", arrived,
                 why="the navigator never reported SUCCEEDED")
    reached = frame.camera_position[list(PLANE)]
    error = float(np.linalg.norm(reached - goal_xy))
    checks.check(
        "the robot ended up where the pipeline asked", error < TOL_M,
        info=f"asked ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}), reached "
             f"({reached[0]:.2f}, {reached[1]:.2f}), off by {error:.2f} m")

    print("\n== the FSM's own actions ==")
    frame = env.step("move_forward")
    before = frame.camera_position[list(PLANE)].copy()
    for _ in range(3):
        frame = env.step("move_forward")
    moved = float(np.linalg.norm(frame.camera_position[list(PLANE)] - before))
    checks.check("move_forward moves the base", moved > 0.05,
                 info=f"{moved:.2f} m in 3 steps")

    heading_before = _heading(frame)
    for _ in range(2):
        frame = env.step("turn_left")
    turned = np.degrees(_wrap(_heading(frame) - heading_before))
    # `turn_left` DECREASES agent_heading (planning/controller.py:82). Getting
    # this backwards mirrors every turn while looking entirely plausible.
    checks.check("turn_left turns the way the controller expects",
                 turned < -5.0, info=f"heading moved {turned:+.1f} degrees")

    print("\n== the operator's floor switch ==")
    if args.no_floor_switch:
        print("  (skipped)")
    else:
        switched, frame = _wait_for_switch(env, timeout_s=20.0)
        checks.check("the /osg/floor switch arrives", switched,
                     why="no switch in 20 s: start the fake robot with "
                         "--floor-switch-after, or publish one by hand")
        if switched:
            lifted = float(frame.camera_position[HEIGHT_AXIS])
            expect = height + float(cfg.floor.virtual_storey_m)
            checks.check(
                "the storey becomes a height the pipeline can separate",
                abs(lifted - expect) < 0.5,
                info=f"{lifted:.2f} m, floor 0 was {height:.2f} m")

    env.close()
    return checks.report()


def _start_loopback() -> None:
    """A fake robot on a thread, answering on the same socket the bridge uses."""
    import threading

    from osg.ros2.loopback import LoopbackRobot, parse_args, serve
    from osg.ros2.wire import parse_addr

    robot_args = parse_args(["--floor-switch-after", "3"])
    robot = LoopbackRobot(robot_args)
    ready = threading.Event()
    threading.Thread(
        target=serve, daemon=True,
        args=(robot, parse_addr(robot_args.bridge_addr),
              robot_args.authkey.encode("utf-8")),
        kwargs={"ready": ready},
    ).start()
    ready.wait(timeout=5.0)
    print("  (loopback: no ROS, no bridge -- everything above the socket)")


def _heading(frame) -> float:
    from osg.planning.controller import agent_heading

    return float(agent_heading(frame.T_wc))


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _drive_until_arrived(env, driver, frame, goal_xy, timeout_s: float):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        driver.observe(frame)
        step = driver.step(goal_xy)
        if step.reason == "arrived":
            return True, frame
        if step.reason == "policy_stop":
            return False, frame
        frame = env.step(step.action)
    return False, frame


def _wait_for_switch(env, timeout_s: float):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        frame = env.step("turn_left")
        if env.floor_key != 0:
            return True, frame
    return False, frame


def _save_rgb(rgb, path: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.imwrite(out, rgb)
    except Exception:  # noqa: BLE001 -- a missing writer must not fail the checks
        out = out.with_suffix(".npy")
        np.save(out, rgb)
    return str(out)


if __name__ == "__main__":
    raise SystemExit(main())
