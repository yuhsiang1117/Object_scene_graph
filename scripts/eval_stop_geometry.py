#!/usr/bin/env python
"""Does an ellipsoid's centre and size change where the agent stops, and so
whether Habitat calls the episode a success?

`ascentnav` cannot answer this in a normal run: its stop never reads the object
layer (it walks to the nearest point of ASCENT's own cloud). So every geometry
is put through ONE stop rule that does read it
(`src/navigation/search/stop_geometry.py`: centre backed off toward the agent
by radius + `approach_standoff_m`) and the resulting point is scored with
Habitat's own success rule -- geodesic distance to the nearest goal viewpoint
at most `success_distance` -- on the navmesh Habitat itself scores on (recut
for the configured agent, see `Scorer`). The navmesh is used for SCORING only,
exactly as Habitat's metric uses it; nothing here is given to an agent.

    # the true boxes, once (needs a GPU simulator: the semantic boxes only
    # exist once the semantic mesh is loaded by a renderer)
    python scripts/eval_stop_geometry.py boxes --out outputs/s84_stop/gt_boxes.json

    # E1: offline, CPU. One sample per (episode, true instance) the baseline
    # committed to or had in view; geometry from each replay's target log
    # (`scripts/replay_object_layer.py ... +replay.target_log=true`)
    python scripts/eval_stop_geometry.py offline outputs/s83_baseline_v0 \
        --replay v0=outputs/s83_baseline_v0/replay_v0 --replay a=outputs/s83_baseline_v0/replay_a \
        --boxes outputs/s84_stop/gt_boxes.json --out outputs/s84_stop

    # E2: the same samples, walked. Habitat and the point-goal policy only --
    # no perception server
    python scripts/eval_stop_geometry.py approach outputs/s84_stop/stop_offline.json \
        --out outputs/s84_stop --geoms gt v0@commit a@commit ascent

What is compared, on the same samples and the same departure pose:

  gt             the dataset's instance centre and its annotated box
  <replay>       the nearest target track within 1.5 m of the instance, as the
                 object layer reported it at the departure step (@commit) and
                 at the end of the episode (@final)
  ascent         ASCENT's rule on its own goal point: 0.6 m short of it
  gt, perturbed  the dose-response -- the true centre moved 0.1 .. 0.5 m (mean
                 over 8 directions) and the true radius scaled 0.5x .. 2x

The dose-response is the point: it says whether a centre error of 0.13 m (A)
or 0.29 m (V0) sits on the flat part of the curve or on its slope, without
depending on how many samples those two variants happen to have.

Self-check, printed first: the baseline's own final position, scored the same
way, must reproduce the `distance_to_goal` Habitat logged for that episode. If
it does not, the scoring is wrong and nothing below it should be read.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from navigation.search.stop_geometry import horizontal_radius, stop_point  # noqa: E402

CAMERA_HEIGHT_M = 0.88            # trace `h` is the camera; the base is this far below
OBJ_BELOW_M, OBJ_ABOVE_M = 0.5, 2.5   # an instance's centre relative to the base, same storey
SAW_RANGE_M, SAW_HALF_DEG = 3.0, 40.0  # the S71 "in view" rule
NEAR_M = 1.5                      # scripts/analyze_trackloc.py: a track this close is the instance's
STANDOFF_M = 0.75                 # agent.approach_standoff_m
ASCENT_STOP_M = 0.6               # `ascent_policy.py:962`: STOP inside this of its goal point
SUCCESS_M = 0.1                   # agent.success_distance
BOX_MATCH_M = 0.25                # a dataset goal and its annotated box: centres this close
SHIFTS_M = (0.1, 0.2, 0.3, 0.5)
RADIUS_SCALES = (0.5, 1.0, 2.0)
DIRECTIONS = 8


def plane(p) -> np.ndarray:
    """Habitat (x, y, z) -> the (x, -z) plane every trace logs in."""
    p = np.asarray(p, float)
    return np.array([p[0], -p[2]])


# ------------------------------------------------------------------ loading

def load_dataset(episodes_root: str) -> dict:
    """{uid: {"goals": [{"pos", "xy", "y", "object_id"}], "viewpoints": (N, 3)}}"""
    out = {}
    for f in sorted(glob.glob(f"{episodes_root}/*.json.gz")):
        d = json.load(gzip.open(f))
        gbc = d.get("goals_by_category", {})
        for i, e in enumerate(d["episodes"]):
            scene = e["scene_id"].split("/")[-1]
            goals = gbc.get(f"{scene}_{e['object_category']}", [])
            vps = [vp["agent_state"]["position"] for g in goals for vp in g.get("view_points", [])]
            if not goals or not vps:
                continue
            out[f"{scene}:{i}"] = {
                "goals": [{"pos": np.asarray(g["position"], float), "xy": plane(g["position"]),
                           "y": float(g["position"][1]), "object_id": g.get("object_id")}
                          for g in goals],
                "viewpoints": np.asarray(vps, float),
            }
    return out


def scene_dir(scenes_dir: str, scene: str) -> Path:
    name = scene.split(".")[0]
    hits = glob.glob(f"{scenes_dir}/hm3d/val/*-{name}")
    if not hits:
        raise SystemExit(f"scene {name} not under {scenes_dir}/hm3d/val (set --scenes-dir or OSG_HM3D_SCENES_DIR)")
    return Path(hits[0])


class Scorer:
    """Habitat's `distance_to_goal` for an arbitrary point, one navmesh per scene.

    Not the `.basis.navmesh` shipped beside the scene: that one is cut for a
    0.1 m-radius, 1.5 m-tall agent, and habitat-lab recomputes it for the
    configured agent (`default_agent_navmesh`) before it scores anything. On
    XB4GS9ShBRE the shipped mesh has 48.3 m^2 navigable against 38.2 m^2, and
    it put one episode's final position 6.17 m from the goal where Habitat
    logged 6.50 m. The same recomputation, in a simulator with no renderer
    (CPU, well under a second per scene), reproduces the logged value exactly.
    """

    def __init__(self, scenes_dir: str) -> None:
        self.scenes_dir = scenes_dir
        self._sim = {}

    def pathfinder(self, scene: str):
        name = scene.split(".")[0]
        if name not in self._sim:
            import habitat_sim
            from habitat.config.default_structured_configs import AgentConfig as LabAgent
            from habitat.config.default_structured_configs import SimulatorConfig as LabSim
            from osg.core.config.agent import AgentConfig
            ours, lab, lab_sim = AgentConfig(), LabAgent(), LabSim()
            cfg = habitat_sim.SimulatorConfiguration()
            cfg.scene_id = str(scene_dir(self.scenes_dir, name) / f"{name}.basis.glb")
            cfg.scene_dataset_config_file = str(
                Path(self.scenes_dir) / "hm3d" / "hm3d_annotated_basis.scene_dataset_config.json")
            cfg.create_renderer = False
            cfg.enable_physics = False
            nav = habitat_sim.nav.NavMeshSettings()      # `HabitatSim.create_sim_config`, same fields
            nav.set_defaults()
            nav.agent_radius = float(ours.agent_radius)
            nav.agent_height = float(ours.camera_height)   # `make_objectnav_config`: agent.height
            nav.agent_max_climb = float(lab.max_climb)
            nav.agent_max_slope = float(lab.max_slope)
            nav.include_static_objects = bool(lab_sim.navmesh_include_static_objects)
            cfg.navmesh_settings = nav
            sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [habitat_sim.agent.AgentConfiguration()]))
            if not sim.pathfinder.is_loaded:
                raise SystemExit(f"no navmesh for {name}")
            self._sim[name] = sim                        # the pathfinder lives as long as its simulator
        return self._sim[name].pathfinder

    def distance(self, scene: str, xy, base_y: float, viewpoints: np.ndarray) -> float:
        """Geodesic metres from the navmesh point under `xy` to the nearest
        viewpoint; inf when the point snaps nowhere or no path exists."""
        import habitat_sim
        pf = self.pathfinder(scene)
        start = np.asarray(pf.snap_point(np.array([xy[0], base_y, -xy[1]], np.float32)), float)
        if not np.all(np.isfinite(start)):
            return float("inf")
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = start.astype(np.float32)
        path.requested_ends = np.asarray(viewpoints, np.float32)
        if not pf.find_path(path):
            return float("inf")
        return float(path.geodesic_distance)


def load_targets(replay_dir: Path, scene: str, episode_id: str) -> list:
    """The replay's per-keyframe target tracks, oldest first; [] if absent."""
    path = replay_dir / f"{scene.split('.')[0]}_ep{episode_id}.targets.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


# ------------------------------------------------------------------ samples

def departures(trace: list, goals: list) -> list:
    """One per true instance the episode engaged: the step the baseline first
    committed to it (`try_nav` with its goal point within NEAR_M of the
    instance), else the first step it was in view. [{"goal", "step", "kind"}]"""
    xy = np.array([t["xy"] for t in trace], float)
    yaw = np.array([float(t["yaw"]) for t in trace])
    base = np.array([float(t["h"]) for t in trace]) - CAMERA_HEIGHT_M
    out = []
    for j, g in enumerate(goals):
        dy = g["y"] - base
        storey = (dy > -OBJ_BELOW_M) & (dy < OBJ_ABOVE_M)
        commit = -1
        for i, t in enumerate(trace):
            gxy = t.get("goal_xy")
            if not (t.get("try_nav") and gxy is not None and storey[i]):
                continue
            d = [float(np.linalg.norm(np.asarray(gxy, float) - h["xy"])) for h in goals]
            if int(np.argmin(d)) == j and d[j] <= NEAR_M:
                commit = i
                break
        if commit >= 0:
            out.append({"goal": j, "step": commit, "kind": "commit"})
            continue
        rel = g["xy"][None, :] - xy
        dist = np.linalg.norm(rel, axis=1)
        ang = (np.arctan2(rel[:, 1], rel[:, 0]) - yaw + np.pi) % (2 * np.pi) - np.pi
        seen = np.flatnonzero((dist < SAW_RANGE_M) & (np.abs(ang) < np.radians(SAW_HALF_DEG)) & storey)
        if len(seen):
            out.append({"goal": j, "step": int(seen[0]), "kind": "sight"})
    return out


def track_near(tracks: list, goal: dict) -> dict | None:
    """The target track nearest the instance, if any is within NEAR_M of it in
    the plane and on its storey."""
    best, best_d = None, NEAR_M
    for tr in tracks:
        c = np.asarray(tr["center"], float)
        if abs(c[1] - goal["y"]) > OBJ_ABOVE_M:
            continue
        d = float(np.linalg.norm(plane(c) - goal["xy"]))
        if d <= best_d:
            best, best_d = tr, d
    return best


def track_at(log: list, goal: dict, step: int) -> tuple:
    """(track at the last keyframe not after `step`, or the first later one
    that has it; "commit" / "late" / None), and the track at the last keyframe."""
    at, when = None, None
    before = [k for k in log if int(k["frame_id"]) <= step]
    if before:
        at = track_near(before[-1]["tracks"], goal)
        when = "commit" if at is not None else None
    if at is None:
        for k in log:
            if int(k["frame_id"]) > step:
                at = track_near(k["tracks"], goal)
                if at is not None:
                    when = "late"
                    break
    final = track_near(log[-1]["tracks"], goal) if log else None
    return at, when, final


def geometry_of(track: dict) -> dict:
    return {"centre": plane(track["center"]), "radius": horizontal_radius(track["axes"]),
            "centre3": np.asarray(track["center"], float)}


def shifted(centre: np.ndarray, shift: float) -> list:
    ang = np.arange(DIRECTIONS) * (2 * np.pi / DIRECTIONS)
    return [centre + shift * np.array([np.cos(a), np.sin(a)]) for a in ang]


# ----------------------------------------------------------------- the test

def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2.0 ** n)


def paired_test(both: list, a: str, b: str) -> dict | None:
    """`both` is [(result under a, result under b)] on the same samples, each
    {"ok", "d_vp"}. Prints one line and returns the numbers."""
    if not both:
        return None
    only_a = sum(x["ok"] and not y["ok"] for x, y in both)
    only_b = sum(y["ok"] and not x["ok"] for x, y in both)
    da = np.array([x["d_vp"] for x, _ in both], float)
    db = np.array([y["d_vp"] for _, y in both], float)
    fin = np.isfinite(da) & np.isfinite(db)
    p_w = None
    if fin.sum() >= 6 and np.any(da[fin] != db[fin]):
        from scipy.stats import wilcoxon
        p_w = float(wilcoxon(da[fin], db[fin]).pvalue)
    out = {"n": len(both), f"{a} ok": sum(bool(x["ok"]) for x, _ in both),
           f"{b} ok": sum(bool(y["ok"]) for _, y in both), f"only {a}": only_a, f"only {b}": only_b,
           "mcnemar_p": mcnemar_exact(only_a, only_b),
           "median_d_diff": float(np.median((db - da)[fin])) if fin.any() else None,
           "wilcoxon_p": p_w}
    diff = f"{out['median_d_diff']:+.3f} m" if out["median_d_diff"] is not None else "n/a"
    print(f"\n  {a} vs {b}, paired on {len(both)} samples: ok {out[f'{a} ok']} vs {out[f'{b} ok']} "
          f"(only {a} {only_a}, only {b} {only_b}, McNemar p = {out['mcnemar_p']:.3f}); "
          f"d to viewpoint {b} - {a} median {diff}"
          f"{f', Wilcoxon p = {p_w:.3f}' if p_w is not None else ''}")
    return out


def run_offline(args) -> None:
    rows = [json.loads(line) for line in (args.run / "episodes.jsonl").open() if line.strip()]
    data = load_dataset(args.episodes_root)
    scorer = Scorer(args.scenes_dir)
    replays = dict(r.split("=", 1) for r in args.replay)
    boxes = json.loads(Path(args.boxes).read_text()) if args.boxes else {}

    # --- self-check: the scoring reproduces Habitat's own number ------------
    diffs, agree, n_chk = [], 0, 0
    for r in rows:
        ds = data.get(f"{r.get('scene')}:{r.get('episode_id')}")
        if ds is None or r.get("final_xy") is None:
            continue
        fx, fz = r["final_xy"]
        d = scorer.distance(r["scene"], (fx, -fz), float(r["final_y"]), ds["viewpoints"])
        diffs.append(abs(d - float(r["distance_to_goal"])))
        n_chk += 1
        agree += int((d <= SUCCESS_M) == (float(r["distance_to_goal"]) <= SUCCESS_M))
    diffs = np.asarray(diffs)
    print(f"self-check, {n_chk} episodes: |offline - Habitat distance_to_goal| median "
          f"{np.median(diffs):.3f} m, max {diffs.max():.3f} m; within-{SUCCESS_M} m verdict agrees "
          f"in {agree} of {n_chk}")
    if diffs.max() > 0.02 or agree < n_chk:
        print("  ** the scoring does not reproduce Habitat: fix that before reading anything below **")

    # --- samples -----------------------------------------------------------
    samples = []
    for r in rows:
        ds = data.get(f"{r.get('scene')}:{r.get('episode_id')}")
        trace = r.get("step_trace") or []
        if ds is None or not trace:
            continue
        logs = {name: load_targets(Path(d), r["scene"], str(r["episode_id"])) for name, d in replays.items()}
        scene_boxes = boxes.get(r["scene"].split(".")[0], [])
        for dep in departures(trace, ds["goals"]):
            g = ds["goals"][dep["goal"]]
            t = trace[dep["step"]]
            agent = np.asarray(t["xy"], float)
            base = float(t["h"]) - CAMERA_HEIGHT_M
            geoms = {}
            box = None
            if scene_boxes:
                cand = min(scene_boxes, key=lambda b: np.linalg.norm(np.asarray(b["center"]) - g["pos"]))
                if np.linalg.norm(np.asarray(cand["center"]) - g["pos"]) <= BOX_MATCH_M:
                    box = cand
            if box is not None:
                geoms["gt"] = {"centre": g["xy"], "radius": horizontal_radius(box["half"])}
            for name, log in logs.items():
                at, when, final = track_at(log, g, int(t.get("n", dep["step"])))
                if at is not None:
                    geoms[f"{name}@commit"] = dict(geometry_of(at), when=when)
                if final is not None:
                    geoms[f"{name}@final"] = geometry_of(final)
            if dep["kind"] == "commit" and t.get("goal_xy") is not None:
                geoms["ascent"] = {"centre": np.asarray(t["goal_xy"], float),
                                   "radius": ASCENT_STOP_M - STANDOFF_M}
            scored = {}
            for name, gm in geoms.items():
                p = stop_point(gm["centre"], gm["radius"], agent, STANDOFF_M)
                d = scorer.distance(r["scene"], p, base, ds["viewpoints"])
                scored[name] = {
                    "d_vp": d, "ok": d <= SUCCESS_M, "radius": float(gm["radius"]),
                    "centre_err": float(np.linalg.norm(gm["centre"] - g["xy"])),
                    "stop_xy": [round(float(v), 3) for v in p], "when": gm.get("when"),
                }
            dose = {}
            if box is not None:
                r_gt = geoms["gt"]["radius"]
                for s in SHIFTS_M:
                    ok = [scorer.distance(r["scene"], stop_point(c, r_gt, agent, STANDOFF_M), base,
                                          ds["viewpoints"]) <= SUCCESS_M for c in shifted(g["xy"], s)]
                    dose[f"shift {s}"] = float(np.mean(ok))
                for k in RADIUS_SCALES:
                    d = scorer.distance(r["scene"], stop_point(g["xy"], k * r_gt, agent, STANDOFF_M),
                                        base, ds["viewpoints"])
                    dose[f"radius x{k}"] = float(d <= SUCCESS_M)
            samples.append({
                "scene": r["scene"], "episode_id": str(r["episode_id"]), "target": r.get("target"),
                "goal": dep["goal"], "step": dep["step"], "kind": dep["kind"],
                "agent_xy": [float(v) for v in agent], "base_y": base, "yaw": float(t["yaw"]),
                "depart_m": float(np.linalg.norm(agent - g["xy"])),
                "episode_success": bool(r["success"]), "geoms": scored, "dose": dose,
            })

    n_commit = sum(s["kind"] == "commit" for s in samples)
    print(f"\n{len(samples)} samples (committed {n_commit}, in view only {len(samples) - n_commit}) "
          f"from {len({(s['scene'], s['episode_id']) for s in samples})} episodes; "
          f"median departure {np.median([s['depart_m'] for s in samples]):.2f} m from the instance")

    names = sorted({n for s in samples for n in s["geoms"]})
    print("\n  geometry        n   stop ok     %   d to viewpoint  centre err  radius   (medians, m)")
    for n in names:
        g = [s["geoms"][n] for s in samples if n in s["geoms"]]
        d = np.array([x["d_vp"] for x in g])
        ok = sum(x["ok"] for x in g)
        late = sum(x.get("when") == "late" for x in g)
        print(f"  {n:13s} {len(g):4d}   {ok:5d}   {100 * ok / len(g):5.1f}   "
              f"{np.median(d[np.isfinite(d)]) if np.isfinite(d).any() else float('nan'):10.2f}   "
              f"{np.median([x['centre_err'] for x in g]):8.2f}   {np.median([x['radius'] for x in g]):5.2f}"
              f"{f'   ({late} formed after the departure step)' if late else ''}")

    def paired(a: str, b: str) -> dict | None:
        return paired_test([(x["geoms"][a], x["geoms"][b]) for x in samples
                            if a in x["geoms"] and b in x["geoms"]], a, b)

    tests = {}
    reps = list(replays)
    for i, a in enumerate(reps):
        for b in reps[i + 1:]:
            for when in ("commit", "final"):
                tests[f"{a}@{when} vs {b}@{when}"] = paired(f"{a}@{when}", f"{b}@{when}")
    for a in reps:
        tests[f"ascent vs {a}@commit"] = paired("ascent", f"{a}@commit")
        tests[f"gt vs {a}@commit"] = paired("gt", f"{a}@commit")

    dosed = [s["dose"] for s in samples if s["dose"]]
    curve = {}
    if dosed:
        gt_ok = np.mean([s["geoms"]["gt"]["ok"] for s in samples if "gt" in s["geoms"]])
        print(f"\n  dose-response on the true geometry ({len(dosed)} samples); unperturbed {100 * gt_ok:.1f}%")
        print("    centre moved   " + "  ".join(f"{s:.1f} m: {100 * np.mean([d[f'shift {s}'] for d in dosed]):5.1f}%"
                                                for s in SHIFTS_M))
        print("    radius scaled  " + "  ".join(f"x{k}: {100 * np.mean([d[f'radius x{k}'] for d in dosed]):5.1f}%"
                                                for k in RADIUS_SCALES))
        curve = {k: float(np.mean([d[k] for d in dosed])) for k in dosed[0]}
        curve["unperturbed"] = float(gt_ok)
    else:
        print("\n  (no --boxes: no true radius, so no `gt` row and no dose-response)")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "stop_offline.json").write_text(json.dumps(
            {"run": str(args.run), "replays": replays, "tests": tests, "dose": curve,
             "self_check": {"n": n_chk, "median_abs_diff": float(np.median(diffs)),
                            "max_abs_diff": float(diffs.max()), "verdict_agree": agree},
             "samples": samples}, indent=1, default=float) + "\n", encoding="utf-8")
        print(f"\nwrote {out / 'stop_offline.json'}")


# ------------------------------------------------------------- true boxes

def run_boxes(args) -> None:
    """Every annotated instance's box, per scene. Needs a renderer: without one
    the semantic mesh is not loaded and every box has zero size."""
    import habitat_sim
    scenes = args.scenes or sorted(Path(f).name.split(".")[0]
                                   for f in glob.glob(f"{args.episodes_root}/*.json.gz"))
    config = Path(args.scenes_dir) / "hm3d" / "hm3d_annotated_basis.scene_dataset_config.json"
    out = {}
    for name in scenes:
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = str(scene_dir(args.scenes_dir, name) / f"{name}.basis.glb")
        cfg.scene_dataset_config_file = str(config)
        cfg.gpu_device_id = 0
        cfg.load_semantic_mesh = True
        # A semantic sensor makes the mesh load, but alone it loads it as one
        # blob and every box is zero (the trap `osg/eval/gt_semantic.py`
        # documents); a colour sensor beside it is what gives the boxes.
        specs = []
        for uuid, kind in (("rgb", habitat_sim.SensorType.COLOR),
                           ("semantic", habitat_sim.SensorType.SEMANTIC)):
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid, spec.sensor_type, spec.resolution = uuid, kind, [8, 8]
            specs.append(spec)
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = specs
        sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
        rows = []
        for obj in sim.semantic_scene.objects:
            if obj is None or obj.category is None:
                continue
            half = 0.5 * np.asarray(obj.aabb.sizes, float)
            if np.all(half > 0) and np.all(np.isfinite(half)):
                rows.append({"id": int(obj.semantic_id), "category": obj.category.name(),
                             "center": [round(float(v), 4) for v in obj.aabb.center],
                             "half": [round(float(v), 4) for v in half]})
        sim.close()
        if not rows:
            raise SystemExit(f"{name}: the semantic scene has no box with a size -- the mesh did not load")
        out[name] = rows
        print(f"{name}: {len(rows)} boxes")
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"wrote {path}")


# ------------------------------------------------- E2: walk the last stretch

ARRIVE_M = 0.125                  # half a forward step: as near a point as the action space resolves


def habitat_rotation(yaw: float) -> list:
    """The agent rotation (x, y, z, w) that faces `yaw` in the (x, -z) plane.
    Habitat's agent faces -z at identity, which is yaw = pi / 2 there."""
    half = 0.5 * (float(yaw) - np.pi / 2.0)
    return [0.0, float(np.sin(half)), 0.0, float(np.cos(half))]


def run_approach(args) -> None:
    """Put the agent back at each sample's departure pose and let it WALK to
    the stop point each geometry gives, then STOP and read Habitat's own
    `success`. Everything before the departure is the baseline's, identical
    across geometries; only the last stretch is re-run, so each sample is its
    own control.

    The mover is the default arm's: the frozen point-goal policy down to its
    stop radius, then `planning/creep.py:SteeredCreep` for what is left (the
    network does not act inside its own radius), until the point is within
    half a step, the creep stalls, or `approach_abandon_steps` are spent. No
    perception server is needed, and nothing simulator-privileged reaches the
    mover: the navmesh is touched once, to snap the departure position the
    trace logged to two decimals back onto it.
    """
    from hydra import compose, initialize_config_dir

    from navigation.geometry import robot_xy_heading
    from osg.core.config import register_configs
    from osg.eval.runner import _episode_uid
    from osg.planning.creep import SteeredCreep
    from osg.planning.pointnav_driver import build_pointnav
    from osg.sim.habitat_env import HabitatObjectNavEnv

    offline = json.loads(Path(args.offline).read_text())
    samples = offline["samples"]
    register_configs()
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
                               version_base="1.3"):
        cfg = compose(config_name="config", overrides=list(args.overrides))
    budget = int(cfg.agent.approach_abandon_steps) or 100
    env = HabitatObjectNavEnv(cfg)
    pointnav = build_pointnav(cfg)
    creep = SteeredCreep(turn_deg=float(cfg.agent.turn_deg))
    episodes = {_episode_uid(e): e for e in env.env.episodes}

    def trial(sample: dict, stop_xy) -> dict:
        env.env.current_episode = episodes[f"{sample['scene']}:{sample['episode_id']}"]
        env.reset()
        sim = env.env.sim
        ax, ay = sample["agent_xy"]
        start = np.asarray(sim.pathfinder.snap_point(
            np.array([ax, sample["base_y"], -ay], np.float32)), float)
        obs = sim.get_observations_at(position=[float(v) for v in start],
                                      rotation=habitat_rotation(sample["yaw"]),
                                      keep_agent_at_new_pose=True)
        if obs is None:
            return {"placed": False}
        frame = env._to_frame(obs)
        xy, yaw = robot_xy_heading(frame)
        turn = (yaw - sample["yaw"] + np.pi) % (2 * np.pi) - np.pi
        if np.linalg.norm(xy - np.array([ax, ay])) > 0.05 or abs(turn) > 0.02:
            raise SystemExit(f"departure pose not reproduced: wanted {(ax, ay, sample['yaw'])}, "
                             f"got {(*xy, yaw)}")
        goal = np.array([stop_xy[0], -stop_xy[1]])           # the mover's plane is (x, z)
        pointnav.reset()
        creep.reset()
        last, steps, terminal, reason = None, 0, False, "budget"
        while steps < budget:
            pointnav.observe(frame)
            rho, theta = pointnav.polar(goal)
            if rho <= ARRIVE_M:
                reason = "arrived"
                break
            action = None
            if not terminal:
                nav = pointnav.step(goal, stop_radius=float(cfg.agent.pointnav_stop_radius))
                terminal, action = nav.action is None, nav.action
            if terminal:
                action, _ = creep.step(goal, frame.camera_position[[0, 2]], rho, theta, last)
                if action is None:
                    reason = "stalled"
                    break
            frame = env.step(action)
            last, steps = action, steps + 1
            if env.episode_over:
                reason = "episode_over"
                break
        if not env.episode_over:
            frame = env.step("stop")
        m = env.metrics()
        end, _ = robot_xy_heading(frame)
        return {"placed": True, "ok": bool(m.get("success", 0.0)),
                "d_vp": float(m.get("distance_to_goal", float("nan"))), "steps": steps,
                "reason": reason, "end_xy": [round(float(v), 3) for v in end],
                "reach_err": float(np.linalg.norm(end - np.asarray(stop_xy, float)))}

    order = sorted(range(len(samples)), key=lambda i: (samples[i]["scene"], samples[i]["episode_id"]))
    names = args.geoms or sorted({n for x in samples for n in x["geoms"]})
    results = []
    for k, i in enumerate(order):
        sample = samples[i]
        row = {"sample": i, "geoms": {}}
        for name in names:
            if name in sample["geoms"]:
                row["geoms"][name] = dict(trial(sample, sample["geoms"][name]["stop_xy"]),
                                          offline_ok=bool(sample["geoms"][name]["ok"]))
        results.append(row)
        if (k + 1) % 10 == 0:
            print(f"  {k + 1} / {len(order)} samples walked", flush=True)
    env.close()

    print(f"\n{len(results)} samples, each walked from its departure pose (at most {budget} steps)")
    print("\n  geometry        n   success     %   d to viewpoint  reached within  steps   offline agrees")
    for name in names:
        g = [r["geoms"][name] for r in results if r["geoms"].get(name, {}).get("placed")]
        if not g:
            continue
        ok = sum(x["ok"] for x in g)
        agree = sum(x["ok"] == x["offline_ok"] for x in g)
        print(f"  {name:13s} {len(g):4d}   {ok:5d}   {100 * ok / len(g):5.1f}   "
              f"{np.median([x['d_vp'] for x in g]):10.2f}   {np.median([x['reach_err'] for x in g]):10.2f} m   "
              f"{np.median([x['steps'] for x in g]):5.0f}   {agree} of {len(g)}")
    tests = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = [(r["geoms"][a], r["geoms"][b]) for r in results
                    if r["geoms"].get(a, {}).get("placed") and r["geoms"].get(b, {}).get("placed")]
            tests[f"{a} vs {b}"] = paired_test(both, a, b)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stop_approach.json").write_text(json.dumps(
        {"offline": str(args.offline), "overrides": list(args.overrides), "budget": budget,
         "arrive_m": ARRIVE_M, "tests": tests, "trials": results}, indent=1, default=float) + "\n",
        encoding="utf-8")
    print(f"\nwrote {out / 'stop_approach.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--episodes-root", default="data/datasets/objectnav/hm3d/v1/val/content")
    common.add_argument("--scenes-dir", default=os.environ.get("OSG_HM3D_SCENES_DIR", ""))
    b = sub.add_parser("boxes", parents=[common])
    b.add_argument("--out", required=True)
    b.add_argument("--scenes", nargs="*", default=[])
    b.set_defaults(fn=run_boxes)
    o = sub.add_parser("offline", parents=[common])
    o.add_argument("run", type=Path)
    o.add_argument("--replay", action="append", default=[], metavar="NAME=DIR")
    o.add_argument("--boxes", default="")
    o.add_argument("--out", default="")
    o.set_defaults(fn=run_offline)
    a = sub.add_parser("approach", parents=[common])
    a.add_argument("offline", help="the stop_offline.json the `offline` step wrote")
    a.add_argument("--out", required=True)
    a.add_argument("--geoms", nargs="*", default=[], help="default: every geometry in the file")
    a.add_argument("--override", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="a Hydra override, as for run_eval.py; repeatable")
    a.set_defaults(fn=run_approach)
    args = ap.parse_args()
    if not args.scenes_dir:
        raise SystemExit("--scenes-dir (or OSG_HM3D_SCENES_DIR) is needed")
    args.fn(args)


if __name__ == "__main__":
    main()
