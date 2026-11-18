#!/usr/bin/env python3
"""How close to each benchmark object can a navmesh agent stand at all?

DualMap's released benchmark scores a stop within 1.0 m of the object. Our agent
moves on Habitat's navmesh, recomputed for its own radius and height
(habitat-lab `default_agent_navmesh=True`), so beside a bed or a desk the
nearest point it can occupy is fixed by geometry before any search or
perception runs. DualMap's released runner is not held to that mesh: its
follower places the agent with `set_agent_state` along a path planned on its
own 0.1 m occupancy grid (scripts/run_dualmap_released_native.py,
`ReleasedFollower.advance`).

This script measures, per dynamic trial, the nearest navmesh point to the
object on the agent's start island -- on the shipped HM3D navmesh and on the
one recomputed for our agent -- and reads every failure against it. A failed
attempt that stopped within a few centimetres of that point is not a search
or an approach failure; it is the rule meeting the floor plan.

Usage:
  python scripts/legacy/analyze_reachability_ceiling.py \\
      --run outputs/osg_released_maps/flat_anchor_v2_island_close \\
      --dualmap outputs/dualmap_official_bench/seed12 \\
      --out outputs/osg_released_maps/REACHABILITY.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from osg.core.config.agent import AgentConfig  # noqa: E402
from osg.core.paths import hm3d_scene_root  # noqa: E402
from analyze_ceiling import bucket, load  # noqa: E402

SCENES = ("00829-QaLdnwvtxbs", "00848-ziup5kvtCCR", "00880-Nfvxx8J5NCo")
RULE_M = 1.0


def load_dualmap(root: Path) -> Dict[str, dict]:
    out = {}
    for path in root.glob("trials/*/result.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        out[str(record["trial_id"])] = record
    return out


def nearest_navigable_m(pf, target, island: Optional[int], rmax: float = 2.0) -> Optional[float]:
    """Horizontal distance from the target to the nearest navmesh point on `island`
    (any island when None), by sampling rings of 0.05 m and snapping each sample."""
    import habitat_sim  # noqa: F401  (pathfinder is passed in)

    best = None
    for rad in np.arange(0.0, rmax + 1e-6, 0.05):
        n = 48 if rad > 0.0 else 1
        for k in range(n):
            ang = 2.0 * math.pi * k / n
            q = np.array([target[0] + rad * math.cos(ang), target[1],
                          target[2] + rad * math.sin(ang)], dtype=np.float32)
            s = pf.snap_point(q, island) if island is not None else pf.snap_point(q)
            if s is None or bool(np.isnan(np.asarray(s)).any()):
                continue
            if abs(float(s[1]) - float(target[1])) > 1.5:
                continue  # another storey
            d = math.hypot(float(s[0]) - target[0], float(s[2]) - target[2])
            if best is None or d < best:
                best = d
        if best is not None and best <= rad + 0.05:
            break  # no farther ring can snap nearer
    return best


def measure(run: Dict[str, dict], dualmap: Dict[str, dict], radius: float, height: float) -> List[dict]:
    import habitat_sim

    root = hm3d_scene_root()
    rows: List[dict] = []
    for scene in SCENES:
        stem = scene.split("-", 1)[1]
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = str(root / f"val/{scene}/{stem}.basis.glb")
        cfg.scene_dataset_config_file = str(root / "hm3d_annotated_basis.scene_dataset_config.json")
        cfg.enable_physics = False
        cfg.create_renderer = False
        agent = habitat_sim.agent.AgentConfiguration()
        agent.radius, agent.height = radius, height
        sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
        shipped = habitat_sim.PathFinder()
        assert shipped.load_nav_mesh(str(root / f"val/{scene}/{stem}.basis.navmesh")), scene
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_radius, settings.agent_height = radius, height
        assert sim.recompute_navmesh(sim.pathfinder, settings), scene
        for tid, record in sorted(run.items()):
            if record["scene"] != scene or tid not in dualmap:
                continue
            block = record["authored_layout"]["dualmap"]
            ref = dualmap[tid]
            start = np.asarray(ref["start_position"], dtype=np.float32)
            targets = ref["target_positions"]
            out = {}
            for name, pf in (("shipped", shipped), ("ours", sim.pathfinder)):
                island = pf.get_island(pf.snap_point(start))
                out[name] = min(nearest_navigable_m(pf, t, island) for t in targets)
            attempts = [float(a["distance_horizontal_m"]) for a in block.get("attempts") or []]
            rows.append({
                "trial_id": tid, "condition": block["condition"], "query": block["query"],
                "success": int(record["success"]), "bucket": bucket(record),
                "shipped_m": out["shipped"], "ours_m": out["ours"],
                "best_attempt_m": min(attempts) if attempts else None,
                "final_m": float(record["distance_to_goal"]),
                "dualmap_final_m": float(ref["final_distance_horizontal_m"]),
                "dualmap_success": int(ref["success"]),
            })
        sim.close()
    return rows


def report(rows: List[dict], radius: float, height: float, run: str, dualmap: str) -> List[str]:
    L: List[str] = []
    say = L.append
    say("# Reachability ceiling: where the 1 m rule meets the floor plan")
    say("")
    say(f"run `{run}`; DualMap `{dualmap}`; navmesh recomputed for radius {radius} m, height {height} m "
        f"(what habitat-lab builds for our agent); `shipped` is the HM3D file.")
    say("")
    say("`navmesh` is the nearest point the agent can occupy on its start island, measured to the object; "
        "`best attempt` is the nearest any of the trial's scored stops came.")
    say("")
    for cond in ("in_anchor", "cross_anchor"):
        sel = [r for r in rows if r["condition"] == cond]
        fails = [r for r in sel if not r["success"]]
        unreach = [r for r in sel if r["ours_m"] > RULE_M]
        reach = [r for r in sel if r["ours_m"] <= RULE_M]
        say(f"## {cond}")
        say("")
        say(f"- trials with **no navmesh point within {RULE_M:.1f} m**: {len(unreach)}/{len(sel)} on our navmesh, "
            f"{sum(1 for r in sel if r['shipped_m'] > RULE_M)}/{len(sel)} on the shipped one")
        say(f"- of the {len(fails)} failures, {sum(1 for r in fails if r['ours_m'] > RULE_M)} are those trials; "
            f"their best attempt minus the navmesh optimum (m): "
            f"{sorted(round(r['best_attempt_m'] - r['ours_m'], 2) for r in fails if r['ours_m'] > RULE_M and r['best_attempt_m'] is not None)}")
        say(f"- DualMap scored {sum(1 for r in unreach if r['dualmap_success'])} of those {len(unreach)} "
            f"(its follower is not held to the navmesh)")
        ours_r = sum(r["success"] for r in reach)
        dm_r = sum(r["dualmap_success"] for r in reach)
        say(f"- **reachable subset** ({len(reach)} trials): ours {ours_r}/{len(reach)} = {100.0 * ours_r / len(reach):.1f}%, "
            f"DualMap {dm_r}/{len(reach)} = {100.0 * dm_r / len(reach):.1f}%; "
            f"full split: ours {sum(r['success'] for r in sel)}/{len(sel)}, DualMap {sum(r['dualmap_success'] for r in sel)}/{len(sel)}")
        say("")
        say("failures by bucket x reachability:")
        say("")
        say("| bucket | reachable | unreachable |")
        say("|---|---:|---:|")
        c = Counter((r["bucket"], r["ours_m"] <= RULE_M) for r in fails)
        for b in sorted({r["bucket"] for r in fails}):
            say(f"| {b} | {c[(b, True)]} | {c[(b, False)]} |")
        say("")
    say("## Every failure")
    say("")
    say("| trial | bucket | navmesh | shipped | best attempt | final | DualMap final |")
    say("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["success"]:
            continue
        ba = "-" if r["best_attempt_m"] is None else f"{r['best_attempt_m']:.2f}"
        flag = " **" if r["ours_m"] > RULE_M else ""
        say(f"| {r['trial_id']} | {r['bucket']} | {r['ours_m']:.2f}{flag} | {r['shipped_m']:.2f} | {ba} | "
            f"{r['final_m']:.2f} | {r['dualmap_final_m']:.2f} |")
    say("")
    say("`**` marks a trial with no navmesh point within 1 m of the object.")
    return L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--dualmap", required=True, help="a DualMap seed directory holding trials/*/result.json")
    ap.add_argument("--radius", type=float, default=AgentConfig().agent_radius)
    ap.add_argument("--height", type=float, default=AgentConfig().camera_height)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    run = load(args.run)
    dualmap = load_dualmap(Path(args.dualmap))
    rows = measure(run, dualmap, args.radius, args.height)
    lines = report(rows, args.radius, args.height, args.run, args.dualmap)
    print("\n".join(lines))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
