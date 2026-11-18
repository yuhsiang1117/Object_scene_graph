"""Can the agent climb a staircase at all, given one?

Every cross-anchor failure so far has ended `climb_max_dy_x100 = 0`, and each
fix moved the blame somewhere new: furniture instead of stairs, then the wrong
direction, then the carrot firing 400 times with the agent still on the ground
floor. Those were all found INSIDE a 500-step episode that also has to explore,
detect, commit and re-plan, which is a poor instrument.

This removes everything else. It teleports the agent to the foot of a staircase
the SIMULATOR says exists -- not our map, not ASCENT's mask, the navmesh -- and
asks one question: does it get up?

Ground truth without a hand-labelled staircase: ask the pathfinder for a route
between a point on one storey and a point on the next. Whatever it returns
crosses the stairs, because there is no other way between storeys. The segment
where the path's height rises IS the flight.

Three movers are driven up that flight and their height logged:

    navmesh   Habitat's own ShortestPathFollower (PRIVILEGED -- it is given
              simulator geometry). The control: if this cannot climb, nothing
              can, and the problem is the simulator or the agent's step height.
    pointnav  the frozen point-goal policy, aimed straight at the top.
    carrot    the same policy fed the path's own waypoints one at a time --
              a forward point, then another, then another -- which is the
              scheme `_flight_carrot` implements.

    python scripts/probe_climb.py --scene 00800-TEEsavR23oF
"""
from __future__ import annotations

import argparse
import json
from typing import List, Optional

import numpy as np

HEIGHT = 1
PLANE_IDX = [0, 2]


def _storey_points(pathfinder, n: int = 4000, seed: int = 0) -> dict:
    """Navigable points bucketed by height, so we can ask for a route between
    two storeys without knowing where the stairs are."""
    pathfinder.seed(seed)
    buckets: dict = {}
    for _ in range(n):
        p = np.asarray(pathfinder.get_random_navigable_point(), dtype=float)
        if not np.isfinite(p).all():
            continue
        buckets.setdefault(round(float(p[HEIGHT]), 1), []).append(p)
    return {k: v for k, v in buckets.items() if len(v) >= 5}


def _find_flight(pathfinder, lower: List[np.ndarray], upper: List[np.ndarray]):
    """A path between storeys, and the rising part of it."""
    import habitat_sim

    best = None
    for a in lower[:40]:
        for b in upper[:40]:
            path = habitat_sim.ShortestPath()
            path.requested_start = a
            path.requested_end = b
            if not pathfinder.find_path(path):
                continue
            pts = np.asarray(path.points, dtype=float)
            if len(pts) < 3:
                continue
            rise = float(pts[:, HEIGHT].max() - pts[:, HEIGHT].min())
            if best is None or rise > best[0]:
                best = (rise, pts, float(path.geodesic_distance))
    return best


def _drive(env, mover, goal_getter, max_steps: int, label: str,
           frame0=None) -> dict:
    """Step until the mover stops or the budget runs out; log the height."""
    sim = env.env.sim
    y0 = float(sim.get_agent_state().position[HEIGHT])
    ys = [y0]
    frame = frame0
    for i in range(max_steps):
        here = np.asarray(sim.get_agent_state().position, dtype=float)
        goal = goal_getter(here, i)
        if goal is None:
            break
        action = mover(frame, here, goal)
        if action is None or action == "stop":
            break
        frame = env.step(action)
        ys.append(float(sim.get_agent_state().position[HEIGHT]))
    ys = np.asarray(ys)
    peak = int(np.argmax(ys))
    # Where the height went, not just how high: a climb that reaches 1.3 m and
    # returns to the floor is a different defect from one that never leaves it.
    return {
        "mover": label,
        "steps": len(ys) - 1,
        "y_start": round(float(ys[0]), 3),
        "y_end": round(float(ys[-1]), 3),
        "dy_max": round(float(ys.max() - ys[0]), 3),
        "peak_step": peak,
        "fell_back_m": round(float(ys.max() - ys[-1]), 3),
        "steps_above_half_m": int((ys - ys[0] > 0.5).sum()),
        "climbed": bool(ys[-1] - ys[0] > 0.5),
        "y_trace": [round(float(v), 2) for v in ys[::max(1, len(ys) // 40)]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="00800-TEEsavR23oF")
    ap.add_argument("--experiment", default="mf5_osg_on_ascent_map")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir
    from pathlib import Path

    from osg.core.config import register_configs
    register_configs()
    root = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(root), version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            f"+experiment={args.experiment}",
            f"ycb.scenes=[{args.scene}]",
            "eval.save_viz=false", "eval.debug_frames=false",
        ])

    from osg.pipeline.components import build_env
    from osg.planning.pointnav_driver import build_pointnav

    env = build_env(cfg)
    frame0 = env.reset()
    sim = env.env.sim
    pathfinder = sim.pathfinder

    buckets = _storey_points(pathfinder)
    print(f"storeys the navmesh knows: "
          f"{ {k: len(v) for k, v in sorted(buckets.items())} }")
    # The two storeys are the two BIGGEST buckets, not the lowest and the
    # highest: on 00800 the extremes are a 6-point ledge at 4.1 m and a handful
    # of cells at 0.2 m, and pairing those asks for a route between two
    # disconnected scraps. The small buckets in between (0.8, 1.0, 2.2, 2.3 m)
    # are the treads themselves -- navigable points at intermediate heights are
    # what a staircase IS to the navmesh.
    if len(buckets) < 2:
        print("this scene has one navigable storey; nothing to climb")
        return
    by_size = sorted(buckets, key=lambda h: -len(buckets[h]))[:2]
    lo_h, hi_h = min(by_size), max(by_size)
    print(f"storeys taken as the two largest: {lo_h} m "
          f"({len(buckets[lo_h])} pts) and {hi_h} m ({len(buckets[hi_h])} pts)")
    lower, upper = buckets[lo_h], buckets[hi_h]
    found = _find_flight(pathfinder, lower, upper)
    if found is None:
        print("no inter-storey path: the storeys are on disconnected navmesh")
        return
    rise, pts, geo = found
    print(f"ground-truth flight: rise {rise:.2f} m over a {geo:.1f} m route, "
          f"{len(pts)} waypoints")
    ys = pts[:, HEIGHT]
    if int(np.argmax(ys)) < int(np.argmin(ys)):
        pts, ys = pts[::-1], ys[::-1]
    # The FOOT is where the climb begins, not where the route begins: the
    # pathfinder's start is any point on the lower storey, 24 m of corridor
    # away. Walk forward to the first waypoint that actually gains height.
    rises = np.diff(ys)
    climbing = np.flatnonzero(rises > 0.15)
    if not climbing.size:
        print("the route between storeys never rises; nothing to stand at")
        return
    start_i, top_i = int(climbing[0]), int(climbing[-1] + 1)
    foot, top = pts[start_i], pts[top_i]
    print(f"stair foot {np.round(foot, 2)}  ->  top {np.round(top, 2)}   "
          f"({top_i - start_i} waypoints, {ys[top_i] - ys[start_i]:+.2f} m)")

    import habitat_sim
    from habitat_sim.utils.common import quat_from_two_vectors

    def place_at_foot(restart: bool = True):
        """Fresh episode, then teleport. Each mover gets its own episode: three
        300-step drives in one episode exhausts Habitat's step limit and the
        third dies with "Episode over, call reset before calling step"."""
        if restart:
            env.reset()
        state = sim.get_agent_state()
        state.position = foot
        facing = pts[min(start_i + 1, len(pts) - 1)] - foot
        facing[HEIGHT] = 0.0
        if np.linalg.norm(facing) > 1e-6:
            state.rotation = quat_from_two_vectors(
                np.array([0.0, 0.0, -1.0]), facing / np.linalg.norm(facing))
        sim.get_agent(0).set_state(state, reset_sensors=True)

    results = []

    # ---- 1. the privileged control
    place_at_foot()
    results.append(_drive(
        env, lambda _f, _h, g: env.action_to_goal(g[PLANE_IDX], float(g[HEIGHT])),
        lambda _h, _i: top, args.max_steps, "navmesh (privileged)"))

    # ---- 2. pointnav straight at the top
    pointnav = build_pointnav(cfg)
    place_at_foot()
    pointnav.reset()
    # Two opposed turns: back to the starting heading, holding a real frame
    # for the policy's first observe().
    env.step("turn_left")
    frame0 = env.step("turn_right")

    def pn(frame_in, _here, goal):
        """Exactly what `NavAgent._carrot_action` does: stop_radius 0.0, and a
        refusal (None) becomes MOVE_FORWARD rather than ending the climb. A
        0.5 m stop radius ends the run at the first waypoint, which is an
        artifact of the probe and not of the climber."""
        if frame_in is not None:
            pointnav.observe(frame_in)
        step = pointnav.step(np.asarray(goal)[PLANE_IDX], stop_radius=0.0)
        return step.action if step.action is not None else "move_forward"

    results.append(_drive(env, pn, lambda _h, _i: top, args.max_steps,
                          "pointnav -> top", frame0=frame0))

    # ---- 3. pointnav fed the path's own waypoints, one at a time
    place_at_foot()
    pointnav.reset()
    env.step("turn_left")
    frame0 = env.step("turn_right")
    waypoints = [p for p in pts[start_i:top_i + 1]]
    state = {"i": 0}

    def next_waypoint(here, _step):
        """Advance as each is reached. Arriving at a waypoint is progress, not
        the end of the climb -- the thing under test is whether a SEQUENCE of
        near goals gets the agent up, which is the scheme `_flight_carrot`
        implements."""
        while state["i"] < len(waypoints):
            w = waypoints[state["i"]]
            if float(np.linalg.norm(here[PLANE_IDX] - w[PLANE_IDX])) < 0.4:
                state["i"] += 1
                continue
            return w
        return None

    results.append(_drive(env, pn, next_waypoint, args.max_steps,
                          "carrot (path waypoints)", frame0=frame0))

    print()
    for r in results:
        verdict = "CLIMBED" if r["climbed"] else "did not arrive"
        print(f"  {r['mover']:<24} {verdict:<14} dy_max={r['dy_max']:+.2f} m "
              f"at step {r['peak_step']}, fell back {r['fell_back_m']:.2f} m, "
              f"{r['steps_above_half_m']} steps above +0.5 m  "
              f"(y {r['y_start']:.2f} -> {r['y_end']:.2f}, {r['steps']} steps)")
        print(f"      y: {r['y_trace']}")
    payload = {"scene": args.scene, "flight_rise_m": round(rise, 3),
               "foot": [round(float(v), 3) for v in foot],
               "top": [round(float(v), 3) for v in top], "results": results}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")
    env.close()


if __name__ == "__main__":
    main()
