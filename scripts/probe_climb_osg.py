"""Drive OSG's OWN climb loop up a ground-truth staircase.

`probe_climb.py` settled that the simulator can be climbed (the privileged
follower does it in 37 steps) and that a bare waypoint carrot gets 44% up and
slides back. This runs the real thing -- `NavAgent._start_climb` +
`_do_climb`, with the arm's config -- from the foot of the same flight, so
the carrot cascade, the stall test and the budget are all the production code.

Three variants isolate WHICH carrot is at fault:

    depth    `_flight_carrot` has no cells, so `_carrot_action` falls through
             to `_update_carrot`: the depth-ray carrot (ASCENT's, transcribed)
             with the ratchet toward the flight's top. Shadowed in every run
             so far by the flight carrot firing first.
    exact    `_flight_carrot` fed the TRUE treads: the navmesh route's own
             points, with their true heights.
    ramp     the same cells with a linear ramp for heights -- what
             `_ramp_stair_heights` writes when it pastes an ASCENT map.

If `exact` climbs and `ramp` does not, the heights are the problem. If neither
climbs but `depth` does, the flight carrot is. If nothing climbs, the loop is.

    python scripts/probe_climb_osg.py --scene 00800-TEEsavR23oF
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import sys
import numpy as np

HEIGHT, PLANE_IDX = 1, [0, 2]


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "probe_climb", Path(__file__).with_name("probe_climb.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _densify(pts: np.ndarray, step_m: float = 0.15) -> np.ndarray:
    """Waypoints every `step_m` along the polyline, so the carrot's 0.35-1.0 m
    band above the agent always holds a cell."""
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step_m)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.asarray(out)


def _flight(agent, pts3d: np.ndarray, kind: str, heights: str):
    cm = agent.costmap
    for p in pts3d:
        cm.ensure_contains(p[PLANE_IDX], margin_m=1.0)
    rc = np.asarray([cm.world_to_grid(p[PLANE_IDX]) for p in pts3d], dtype=int)
    from navigation.mapping.map_store import ramp_heights   # the paste's own ramp
    lo, hi = float(pts3d[0, HEIGHT]), float(pts3d[-1, HEIGHT])
    if heights == "exact":
        h = pts3d[:, HEIGHT].astype(float)
    elif heights in ("arclen", "arclen0"):
        # Ramp over distance WALKED along the flight. If this climbs and the
        # straight-axis ramp below does not, the flight bends and projection
        # onto its chord runs backwards on the far leg.
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts3d[:, PLANE_IDX], axis=0), axis=1))]
        h = ramp_heights(d / max(d[-1], 1e-6), lo, hi,
                         margin_m=0.0 if heights == "arclen0" else 0.25)
    else:                                            # projection on the chord, as the paste writes
        margin = 0.0 if heights == "ramp0" else 0.25
        axis = pts3d[-1, PLANE_IDX] - pts3d[0, PLANE_IDX]
        t = (pts3d[:, PLANE_IDX] - pts3d[0, PLANE_IDX]) @ (axis / max(np.linalg.norm(axis), 1e-6))
        back = int((np.diff(t) < -0.02).sum())
        print(f"    [ramp] chord projection along the path: {back} of {len(t) - 1} steps run BACKWARDS; "
              f"t range {t.min():.2f}..{t.max():.2f} m over {np.sum(np.linalg.norm(np.diff(pts3d[:, PLANE_IDX], axis=0), axis=1)):.2f} m walked")
        frac = (t - t.min()) / max(t.max() - t.min(), 1e-6)
        h = ramp_heights(frac, lo, hi, margin_m=margin)
    return SimpleNamespace(kind=kind, cells_rc=rc, heights=h, n_cells=len(rc),
                           foot_xy=pts3d[0, PLANE_IDX].astype(float),
                           top_xy=pts3d[-1, PLANE_IDX].astype(float))


def _run_variant(env, cfg, pointnav, foot, top, path3d, variant, max_steps, place, full_fsm=False):
    from osg.agent.nav_agent import NavAgent, State
    from osg.exploration.async_scorer import AsyncScorer
    from osg.exploration.scorer import NullScorer
    from osg.perception.detector import StubDetector

    env.reset()
    place()
    env.step("turn_left")
    frame = env.step("turn_right")
    sim = env.env.sim

    agent = NavAgent(cfg, StubDetector(), AsyncScorer(NullScorer()), None, "bowl",
                     pointnav=pointnav)
    pointnav.reset()
    agent.step_count = 1
    agent.pointnav.observe(frame)
    agent._agent_xy = frame.camera_position[PLANE_IDX].copy()
    agent.floors.observe(frame, agent.step_count)

    kind = "up" if top[HEIGHT] > foot[HEIGHT] else "down"
    if variant == "pasted":
        # THE REAL THING: ASCENT's stored map pasted into this agent's costmap
        # exactly as `load_obstacle_map` does at episode start, `find_flights`
        # run on it, and the flight nearest the true foot climbed. No path
        # cells, no fiction -- the blob, the ramp, the endpoints, as the run
        # sees them.
        from osg.eval.prior_map import paste_snapshots, _scene_map_paths
        from osg.mapping.costmap import Costmap2D
        from osg.mapping.stairs import find_flights
        # Every snapshot for the scene, matched BY HEIGHT -- the same paste
        # `load_obstacle_map` performs. Opening `<scene>.json` directly was
        # union-blind: under `ycb.obstacle_map_union` that file does not exist
        # and the probe raised, which is how it was found.
        here_y = float(foot[HEIGHT]); there_y = float(top[HEIGHT])
        paths = _scene_map_paths(str(cfg.ycb.obstacle_map_in), str(cfg.ycb.scenes[0]))
        if not paths:
            raise SystemExit(f"no obstacle snapshot for {cfg.ycb.scenes[0]} "
                             f"under {cfg.ycb.obstacle_map_in}")
        agent.costmap.ensure_contains(np.asarray(foot[PLANE_IDX], float), margin_m=15.0)
        if agent.costmap.height is None:
            agent.costmap.height = np.full(agent.costmap.grid.shape, np.nan, dtype=np.float32)
        lo, hi = (here_y, there_y) if here_y < there_y else (there_y, here_y)
        other = Costmap2D(resolution=agent.costmap.resolution, size_m=60.0)
        other.height = np.full(other.grid.shape, np.nan, dtype=np.float32)
        maps = [agent.costmap, other] if here_y < there_y else [other, agent.costmap]
        res = paste_snapshots(paths, maps, [lo, hi], strict=False)
        n = sum(int(pair["cells"]) for pair in res["pasted"]
                if maps[int(pair["osg_floor"])] is agent.costmap)
        print(f"    [pasted] {len(paths)} snapshot(s), matched_by="
              f"{sorted({p['matched_by'] for p in res['pasted']})}, "
              f"{len(res['skipped'])} skipped")
        span = max(float(cfg.floor.new_level_m), abs(there_y - here_y)) if bool(
            getattr(cfg.floor, "flight_span_from_levels", False)) else None
        flights = [f for f in find_flights(agent.costmap, here_y,
                                           order_path=bool(getattr(
                                               cfg.agent, "climb_carrot_follow_path", False)),
                                           new_level_m=float(cfg.floor.new_level_m),
                                           wide_span_m=span, wide_mask=getattr(agent.costmap, "stair_mask", None),
                                           min_span_m=float(cfg.floor.flight_min_span_m),
                                           min_cells=int(cfg.floor.flight_min_cells)) if f.kind == kind]
        if not flights:
            print(f"    [pasted] {n} cells pasted, NO {kind} flight found"); flight = None
        else:
            flight = min(flights, key=lambda f: float(np.linalg.norm(f.foot_xy - foot[PLANE_IDX])))
            print(f"    [pasted] {n} cells, {len(flights)} {kind} flights; chosen foot {np.round(flight.foot_xy, 2)} "
                  f"is {np.linalg.norm(flight.foot_xy - foot[PLANE_IDX]):.2f} m from the true foot, "
                  f"{flight.n_cells} cells, span {flight.span_m:.2f} m, heights {flight.heights.min():.2f}..{flight.heights.max():.2f}")
        if flight is None:
            return {"variant": variant, "direction": 0, "steps": 0, "dy_max": 0.0, "y_end": float(foot[HEIGHT]),
                    "fell_back_m": 0.0, "end": {"why": "no flight"}, "carrot_flight": 0, "carrot_top": 0,
                    "forced_forward": 0, "blocked_turn": 0, "actions": {}, "y_trace": [], "carrot_trace": []}
    elif variant == "measured":
        # What the agent's OWN depth camera says the treads are, from the foot:
        # a full turn at the foot, mapping each frame as `_act_inner` does,
        # then the height layer read at the true tread cells. NaN where the
        # camera never saw a cell is filled with the paste's ramp -- which is
        # exactly the substitution the loader would make at runtime.
        m = cfg.mapping
        for _ in range(12):
            frame = env.step("turn_left")
            agent.step_count += 1
            fy = agent.floors.observe(frame, agent.step_count)
            agent.costmap.update(frame, floor_y=fy, obstacle_low=m.obstacle_low_m,
                                 obstacle_high=m.obstacle_high_m, max_range=m.max_range_m,
                                 stride=m.depth_stride)
        agent.pointnav.observe(frame)
        agent._agent_xy = frame.camera_position[PLANE_IDX].copy()
        flight = _flight(agent, path3d, kind, "ramp")
        live = agent.costmap.height
        if live is None:
            print("    [measured] costmap.height is None: the height layer was never allocated")
        else:
            h = live[flight.cells_rc[:, 0], flight.cells_rc[:, 1]].astype(float)
            seen = np.isfinite(h)
            err = np.abs(h[seen] - path3d[seen, HEIGHT]) if seen.any() else np.zeros(0)
            print(f"    [measured] {int(seen.sum())} of {len(h)} tread cells measured from the foot; "
                  f"|measured - true| median {np.median(err) if err.size else float('nan'):.2f} m, "
                  f"max {err.max() if err.size else float('nan'):.2f} m")
            flight.heights = np.where(seen, h, flight.heights)
    elif variant == "depth":
        flight = SimpleNamespace(kind=kind, cells_rc=np.zeros((0, 2), int),
                                 heights=np.zeros(0), n_cells=0,
                                 foot_xy=foot[PLANE_IDX], top_xy=top[PLANE_IDX])
    else:
        flight = _flight(agent, path3d, kind, variant)
    agent.floors.pursuing = True
    agent.floors.pursuit_flight = flight
    agent._goal_xy = np.asarray(top[PLANE_IDX], float)
    agent._goal_floor_y_cache = float(top[HEIGHT])

    ended = {}
    real_end = agent.climb._end_climb
    def _end(ok, why):
        ended["ok"], ended["why"] = bool(ok), str(why)
        real_end(ok, why)
    agent.climb._end_climb = _end

    # What the carrot actually picks, step by step: where the agent stands,
    # how high, where it is sent, and how far that is. The ramps and the true
    # heights differ ONLY in this choice, so this is where the difference is.
    trace = []
    real_carrot = agent.climb._flight_carrot
    def _traced(fr, axy):
        goal = real_carrot(fr, axy)
        standing = float(fr.camera_position[HEIGHT]) - float(cfg.agent.camera_height)
        fl = agent.floors.pursuit_flight
        n_band = 0
        if fl is not None and fl.n_cells:
            ahead = (np.asarray(fl.heights, float) - standing) * (1.0 if agent.climb._climb_direction >= 0 else -1.0)
            n_band = int(((ahead >= 0.35) & (ahead <= 1.0)).sum())
        trace.append((round(standing, 2),
                      None if goal is None else round(float(np.linalg.norm(goal - axy)), 2),
                      n_band))
        return goal
    agent.climb._flight_carrot = _traced

    agent.climb._start_climb(frame)
    ys = [float(sim.get_agent_state().position[HEIGHT])]
    actions = []
    for _ in range(max_steps):
        if agent.state is not State.CLIMB:
            break
        if full_fsm:
            # everything the run does around the climb: floors.observe, the
            # costmap update (or its on-stairs suppression), the keyframe
            # pipeline, the escape filter, the state dispatch
            action = agent.act(frame)
        else:
            agent.step_count += 1
            agent.pointnav.observe(frame)
            agent._agent_xy = frame.camera_position[PLANE_IDX].copy()
            agent.floors.observe(frame, agent.step_count)
            action = agent.climb._do_climb(frame)
        if agent.state is not State.CLIMB or action is None or action == "stop":
            break
        actions.append(action)
        frame = env.step(action)
        ys.append(float(sim.get_agent_state().position[HEIGHT]))
    ys = np.asarray(ys)
    st = agent.stats
    sign = 1.0 if agent.climb._climb_direction >= 0 else -1.0
    prog = (ys - ys[0]) * sign                    # progress IN the climb direction
    return {
        "variant": variant, "direction": int(agent.climb._climb_direction),
        "steps": len(ys) - 1, "dy_max": round(float(sign * prog.max()), 3),
        "y_end": round(float(ys[-1]), 3), "fell_back_m": round(float(prog.max() - prog[-1]), 3),
        "end": ended, "carrot_flight": st.get("climb_flight_carrot", 0),
        "carrot_top": st.get("climb_flight_carrot_top", 0),
        "forced_forward": st.get("climb_forced_forward", 0),
        "blocked_turn": st.get("climb_blocked_turn", 0),
        "actions": {a: actions.count(a) for a in sorted(set(actions))},
        # The sequence, not just its histogram: two different climbs can share
        # a histogram. This is what tests/integration/test_climb_lock.py pins.
        "actions_sha256": hashlib.sha256(",".join(actions).encode("utf-8")).hexdigest(),
        "carrot_trace": trace[:400:2][:24],   # (standing_y, goal_dist_m, cells_in_band), 1 per step
        "y_trace": [round(float(v), 2) for v in ys[::max(1, len(ys) // 30)]],
    }


def build_climb_fixture(cfg, *, storeys=None, descend=False, scene=""):
    """Everything a climb needs before its first action: the environment, the
    ground-truth flight between two CONNECTED storeys, and a `place()` that puts
    the agent at its foot facing up it.

    Extracted verbatim from `main()` so that
    `tests/integration/test_climb_lock.py` drives the same setup this diagnostic
    does rather than a copy of it that will drift. The one deliberate change: a
    scene with no navigable flight now raises instead of returning 2 from a
    function whose return value was discarded.
    """
    probe = _load_probe()
    from osg.pipeline.components import build_env
    from osg.planning.pointnav_driver import build_pointnav
    env = build_env(cfg)
    env.reset()
    sim, pf = env.env.sim, env.env.sim.pathfinder

    buckets = probe._storey_points(pf)
    if storeys:
        want = [float(v) for v in storeys.split(",")]
        lo_h, hi_h = (min(buckets, key=lambda h: abs(h - w)) for w in sorted(want))
        found = probe._find_flight(pf, buckets[lo_h], buckets[hi_h])
    else:
        # The two biggest storeys are not always the two the episodes cross: on
        # 00808 they are the basement and the ground floor, and the basement is
        # 100% off the start island, so no flight between them exists at all.
        # Fall through the size-ordered pairs until one is actually connected.
        order = sorted(buckets, key=lambda h: -len(buckets[h]))
        found = lo_h = hi_h = None
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                a, b = min(order[i], order[j]), max(order[i], order[j])
                got = probe._find_flight(pf, buckets[a], buckets[b])
                if got is not None:
                    found, lo_h, hi_h = got, a, b
                    break
            if found is not None:
                break
    if found is None:
        raise SystemExit(
            f"no navigable flight between any storey pair of {scene or cfg.ycb.scenes[0]}")
    print(f"    storeys {lo_h:+.2f} -> {hi_h:+.2f}")
    rise, pts, _geo = found
    ys = pts[:, HEIGHT]
    if int(np.argmax(ys)) < int(np.argmin(ys)):
        pts, ys = pts[::-1], ys[::-1]
    climbing = np.flatnonzero(np.diff(ys) > 0.15)
    start_i, top_i = int(climbing[0]), int(climbing[-1] + 1)
    foot, top = pts[start_i], pts[top_i]
    path3d = _densify(pts[start_i:top_i + 1])
    if descend:
        # The climb runs the other way: the agent stands at the top and the
        # flight's "foot" -- the mouth on this storey, as `find_flights` names
        # it for a descent -- is the highest tread. `_flight_carrot` then hunts
        # for cells 0.35-1.0 m BELOW the agent, and `_do_climb` pitches the
        # camera down first (ASCENT's phase 2). None of this was covered by
        # the ascending probe, and the first descent on a real staircase
        # (outputs/mf5_pass2_v10 ep1) went 23 cm and stalled.
        foot, top = top, foot
        path3d = path3d[::-1]
        pts, start_i = pts[::-1], len(pts) - 1 - top_i
    print(f"ground-truth flight{' (DESCENDING)' if descend else ''}: start {np.round(foot, 2)} -> end {np.round(top, 2)}  "
          f"({top[HEIGHT] - foot[HEIGHT]:+.2f} m, {len(path3d)} densified points)")

    import habitat_sim  # noqa: F401  (agent state types)
    from habitat_sim.utils.common import quat_from_two_vectors

    def place():
        state = sim.get_agent_state()
        state.position = foot
        facing = path3d[min(3, len(path3d) - 1)] - foot   # a few points along the flight
        facing[HEIGHT] = 0.0
        if np.linalg.norm(facing) > 1e-6:
            state.rotation = quat_from_two_vectors(
                np.array([0.0, 0.0, -1.0]), facing / np.linalg.norm(facing))
        sim.get_agent(0).set_state(state, reset_sensors=True)

    pointnav = build_pointnav(cfg)
    probe_ns = SimpleNamespace(
        env=env, sim=sim, pointnav=pointnav, foot=foot, top=top,
        path3d=path3d, place=place, lo_h=lo_h, hi_h=hi_h)
    return probe_ns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="00800-TEEsavR23oF")
    ap.add_argument("--experiment", default="mf5_osg_on_ascent_map")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--variants", default="depth,exact,ramp")
    ap.add_argument("--maps", default="outputs/maps_mf5_ascent",
                    help="ycb.obstacle_map_in for the `pasted` variant")
    ap.add_argument("--full-fsm", action="store_true",
                    help="drive agent.act() with the state pre-set to CLIMB instead of _do_climb "
                         "directly: the run's climb takes ~200 steps for 1.5 m where this probe takes 50")
    ap.add_argument("--descend", action="store_true",
                    help="start at the top and drive DOWN the same flight")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra Hydra overrides, repeatable -- sweep a climb "
                         "knob without editing the preset it is measured in")
    ap.add_argument("--storeys", metavar="LO,HI",
                    help="storey heights to probe between; without it the two "
                         "biggest CONNECTED storeys are used")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    probe = _load_probe()
    from hydra import compose, initialize_config_dir
    from osg.core.config import register_configs
    register_configs()
    root = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(root), version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            f"+experiment={args.experiment}", f"ycb.scenes=[{args.scene}]",
            f"ycb.obstacle_map_in={args.maps}",
            "eval.save_viz=false", "eval.debug_frames=false", *args.set])

    fx = build_climb_fixture(cfg, storeys=args.storeys, descend=args.descend,
                             scene=args.scene)
    env, pointnav = fx.env, fx.pointnav
    foot, top, path3d, place = fx.foot, fx.top, fx.path3d, fx.place

    results = []
    for variant in args.variants.split(","):
        r = _run_variant(env, cfg, pointnav, foot, top, path3d, variant.strip(),
                         args.max_steps, place, full_fsm=args.full_fsm)
        results.append(r)
        verdict = "ARRIVED" if abs(r["dy_max"]) > 0.5 and r["fell_back_m"] < 0.5 else "did not arrive"
        print(f"\n  {r['variant']:<6} dir={r['direction']:+d}  {verdict:<15} dy_max={r['dy_max']:+.2f} m  "
              f"fell_back={r['fell_back_m']:.2f}  steps={r['steps']}  end={r['end']}")
        print(f"         carrot: flight={r['carrot_flight']} top={r['carrot_top']} "
              f"forced_fwd={r['forced_forward']} blocked_turn={r['blocked_turn']}  actions={r['actions']}")
        print(f"         y: {r['y_trace']}")
        print(f"         carrot (standing, goal_dist, in_band) x{len(r['carrot_trace'])}: {r['carrot_trace']}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    env.close()


if __name__ == "__main__":
    main()
