"""Run the pipeline on a real robot, over ROS 2 / Nav2.

Two runs make an experiment, differing in one field:

    python scripts/run_robot.py +experiment=stretch3_map      # explore, keep the map
    python scripts/run_robot.py +experiment=stretch3_search   # load it, find the thing

That is the protocol the whole dynamic-scene line is built around
(docs/history/DYNAMIC_SCENES.md): run 1 leaves a scene graph and per-storey occupancy;
run 2 starts believing it, and the object has moved. The difference on a robot
is only that the world changes because somebody moved a chair rather than
because a script re-authored a layout.

Separate from `run_eval.py` because a run here is not an evaluation: there is
no episode dataset, no ground truth, nothing to score. What it shares is
everything that matters -- `build_env`, `build_run_components`, `build_agent`
and `run_episode` are the same functions the benchmark uses, so the robot runs
the measured pipeline and not a copy of it.

Needs the bridge up first:

    bash scripts/ros2/bridge.sh                   # a robot, or
    bash scripts/ros2/check_pipeline.sh           # the fake one
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osg.core.config import register_configs  # noqa: E402

register_configs()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Imported late, as run_eval.py does, so `--help` does not load habitat.
    from osg.eval.episode import run_episode
    from osg.core.profiler import Profiler
    from osg.eval.record import (build_episode_record, detector_identity,
                                 episode_tag)
    from osg.eval.visualize import save_topdown
    from osg.graph.map_store import apply_map, load_map, save_map
    from osg.pipeline.components import (build_agent, build_env,
                                         build_run_components,
                                         probe_served_models)

    if str(cfg.eval.mode) != "ros2":
        raise ValueError(
            f"run_robot.py drives a robot, so eval.mode must be 'ros2' "
            f"(got {cfg.eval.mode!r}). Use +experiment=stretch3_map.")
    map_mode = str(cfg.ros2.map_mode)
    if map_mode not in ("map", "search"):
        raise ValueError(f"ros2.map_mode must be map|search, got {map_mode!r}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    # Everything below is written AS IT HAPPENS, not when the run ends. Two
    # runs were lost by their shell closing: no episodes.jsonl, no map, and a
    # stdout log that -- piped through tee -- had been sitting in Python's
    # 8 KB block buffer. So: line-buffered stdout, SIGTERM (kill, docker stop,
    # a closing terminal's hangup) treated as Ctrl-C so the record and the map
    # still get written, and the mapping pass checkpoints its map below.
    import signal
    import sys

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)

    def _stop(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _stop)

    # Same refusal as a benchmark run: a model server that is down is not a
    # degraded run, and on a robot it is a robot driving on a dead detector.
    probe_served_models(cfg)

    env = build_env(cfg)
    # The same kind of refusal as probe_served_models, for the robot's side:
    # a run that would drive found out Nav2 was missing at its first goal --
    # `send_goal: RuntimeError: no navigate_to_pose action server` -- after
    # every model had loaded. Ask the bridge first; it costs nothing.
    info = env.transport.ping()
    if not info.get("nav_server", True):
        raise SystemExit(
            f"no {cfg.ros2.nav_action} action server on the robot. Launch Nav2 there "
            "(stretch_nav2 navigation.launch.py) and check `ros2 action info "
            f"{cfg.ros2.nav_action}` shows a server, then rerun.")
    # A robot on another DDS vendor is the failure that LOOKS like a working
    # robot: topics cross vendors, services and actions do not, so every Nav2
    # goal silently times out and the base never moves (bridge_node.robot_rmw).
    foreign = {v: n for v, n in (info.get("robot_rmw") or {}).items()
               if v != "cyclonedds" and any(name != "osg_bridge" for name in n)}
    if foreign:
        raise SystemExit(
            "the robot's nodes are not on CycloneDDS: "
            + "; ".join(f"{v}: {', '.join(sorted(set(n)))}" for v, n in foreign.items())
            + ". Actions and services do not cross DDS vendors, so Nav2 would never "
            "answer a goal. Relaunch the robot's stack (driver, camera, Nav2, SLAM) with "
            "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp and the CycloneDDS env exported in "
            "EVERY shell, then rerun.")
    # Which map Nav2 plans on. slam_toolbox = the robot maps as it goes (the
    # mapping protocol); map_server = AMCL on a map it was given, which must
    # then be localised BEFORE the run (docs/THOR.md, "Mapping with slam_toolbox").
    map_pubs = info.get("map_publishers")
    if map_pubs is not None:
        print(f"[robot] /map published by: {', '.join(map_pubs) or 'nobody (no map yet?)'}")
    components = build_run_components(cfg, env=env)
    env.attach_driver(components["pointnav"])
    target = env.target_category()
    map_path = Path(cfg.ros2.map_dir) / f"{cfg.ros2.map_tag}.json"

    print(f"[robot] {map_mode} run, target={target!r}, map={map_path}")
    profiler = Profiler()
    # Built BEFORE the first frame, so the floor callback is in place before
    # `reset()` can poll one: a switch landing in that window would set the
    # env's storey (and the height of every pose after it) while the agent was
    # still filing everything under storey 0.
    agent = build_agent(
        cfg, components, target,
        keyframe_dir=str(out_dir / "keyframes" / f"{cfg.ros2.map_tag}")
        if cfg.eval.save_viz else None,
        profiler=profiler,
    )
    # The operator's floor switch, from the topic through to the map and the
    # scene graph (docs/ROS2.md). Needs floor.source=external to take effect.
    env.on_floor_switch = agent.floors.request_floor

    frame = env.reset()
    episode = env.current_episode

    if map_mode == "search":
        if not map_path.exists():
            raise FileNotFoundError(
                f"no map at {map_path}. Run +experiment=stretch3_map first, "
                f"or point ros2.map_tag at one that exists.")
        n_tracks = apply_map(
            agent, load_map(map_path),
            max_log_odds=float(cfg.scene_graph.presence.reload_max_log_odds),
            initial_floor_y=float(frame.camera_position[1]) - float(cfg.agent.camera_height),
            # The storey is declared, not measured: a height comparison here
            # would be between two synthetic numbers (sim/ros2_env.py).
            initial_floor_key=int(env.floor_key),
        )
        print(f"[robot] restored {n_tracks} tracks on floor {env.floor_key}")

    # The simulator's debug video, live: rotated frame, detections, the
    # two-panel view, all on /osg/* for scripts/ros2/osg.rviz. Always on for a
    # robot -- a person is watching -- and it never costs a step (debug_stream.py).
    from osg.eval.debug_stream import DebugStream

    checkpoint_every = int(cfg.ros2.map_checkpoint_steps)

    def _checkpoint(step: int) -> None:
        # The map so far, at the path the search pass will read. A run that dies
        # at step 150 then leaves a step-140 map instead of none; a proper end
        # overwrites it with the final one below.
        if map_mode == "map" and checkpoint_every > 0 and step % checkpoint_every == 0:
            save_map(map_path, agent, scene=str(cfg.ros2.map_tag))
            print(f"[robot] map checkpoint at step {step} -> {map_path}", flush=True)

    debug = DebugStream(env, log_path=out_dir / "stream.jsonl", on_step=_checkpoint)
    if bool(cfg.ros2.wait_for_switch):
        # Resuming after a carry (config/ros2.py): the restored graph is
        # already on RViz through `debug.write`; the operator's switch starts
        # the run, and its first step applies the storey.
        frame = env.wait_for_floor_switch(
            on_frame=lambda f: debug.write(f, agent, target, components["detector"]))
    try:
        outcome = run_episode(cfg, env, agent, episode, target, frame,
                              components["detector"], debug)
    except KeyboardInterrupt as exc:
        print(f"\n[robot] interrupted ({exc or 'Ctrl-C'}) -- stopping the base")
        outcome = None
    finally:
        debug.close()
        env.close()

    driver = components["pointnav"]
    print(f"[robot] {getattr(driver, 'n_goals_sent', 0)} goals posted to Nav2, "
          f"{getattr(driver, 'n_aborts', 0)} refused; "
          f"floor switches: {env.floor_switches or 'none'}")

    if map_mode == "map":
        path = save_map(map_path, agent, scene=str(cfg.ros2.map_tag))
        print(f"[robot] wrote {path}")

    if outcome is None:
        return

    scorer = components["scorer"]
    verifier = components["verifier"]
    rec = build_episode_record(
        cfg=cfg, episode=episode, env=env, agent=agent, outcome=outcome,
        target=target, detector_identity=detector_identity(cfg),
        metrics=env.metrics(), profiler=profiler, scorer=scorer,
        scorer_before=(0, 0, None),
        verifier=verifier, verifier_before=(0, 0),
    )
    with open(out_dir / "episodes.jsonl", "a") as f:
        f.write(json.dumps(_json_safe(rec)) + "\n")
    if cfg.eval.save_viz:
        save_topdown(
            str(out_dir / "viz" / f"{episode_tag(episode)}.png"),
            agent.costmap, outcome.trajectory, scene_graph=agent.scene_graph,
            title=f"{cfg.ros2.map_tag} target={target} steps={outcome.steps}",
        )
    print(f"[robot] {outcome.steps} steps, {outcome.wall_time_s}s -> {out_dir}")


def _json_safe(value):
    """Non-finite floats -> null, so the record is JSON anything can read.

    A robot run has no ground truth, so `success`/`spl`/`distance_to_goal`
    arrive as NaN -- honestly "not measured" in memory, but `json.dumps` writes
    a bare `NaN` token that only Python accepts back.
    """
    import math

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


if __name__ == "__main__":
    main()
