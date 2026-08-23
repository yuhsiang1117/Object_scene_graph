#!/usr/bin/env python3
"""Where does the search posterior RANK the surface the object was moved to?

The dynamic benchmark's dominant failure is that the agent never perceives the
object at its new pose, and the reason is not perception: measured over the
authored viewpoints of the dynamic layouts, detector recall is 0.58 and its
correlation with per-target SR is +0.19. The reason is that the search never
goes to the right surface. Measured over 96 episodes of condition C0:

    the new pose is within 0.5 m of a mapped container   96/96
    the surface search ARRIVED at the true surface        0/96

Perfect candidate coverage, zero arrivals. So the question is ranking, and this
script answers it WITHOUT a simulator: the prior map already contains every
container the posterior can propose, and `container_prior` is a pure function of
(target class, container label, top height, footprint area, centre). Rebuilding
the scene graph from the snapshot reproduces exactly the candidate list the
agent would have built on step 1, and the authored layout says which of them is
the right answer.

That turns an 8-hour benchmark into a second, and gives ranking work a number to
move: the rank of the true surface, and the share of episodes where it lands in
the top-N an episode can actually afford to inspect (measured: a median of 5).

    python scripts/rank_search_surfaces.py
    python scripts/rank_search_surfaces.py --maps outputs/maps_hires --top-n 5

Two orderings are reported: the one the agent uses, which weights a surface by
its distance from where the object was last believed to be, and a flat prior that
ignores that distance -- what the agent did after confirming absence until the
proximity model was fixed. The gap between them is what the term is worth.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TARGET_LABELS = {
    "003_cracker_box": "cracker box", "005_tomato_soup_can": "tin can",
    "011_banana": "banana", "019_pitcher_base": "blue plastic pitcher",
    "021_bleach_cleanser": "bleach bottle", "024_bowl": "bowl",
    "025_mug": "mug", "029_plate": "red plate",
}
PLANE = (0, 2)


def _footprint_distance(node, xy: Tuple[float, float]) -> float:
    """Distance from a point to a container's footprint, not to its centre.

    A counter is three metres long; its centre can be two metres from an object
    resting on it. Scoring by centre distance would call the right surface wrong.
    """
    centre = np.asarray(node.center, dtype=float)
    ax = getattr(node, "axes", None)
    ex, ez = (float(ax[PLANE[0]]), float(ax[PLANE[1]])) if ax is not None else (0.0, 0.0)
    dx = max(0.0, abs(centre[PLANE[0]] - xy[0]) - ex)
    dz = max(0.0, abs(centre[PLANE[1]] - xy[1]) - ez)
    return math.hypot(dx, dz)


def load_scene_graph(map_path: Path, sg_cfg=None):
    """Rebuild the agent's step-1 scene graph from a saved map, no simulator.

    The container layer MUST be built with the configured gates, not
    `SceneGraph()`'s bare defaults: `container_min_obs` and `container_min_score`
    exist precisely to keep single-sighting and low-confidence surfaces out of
    the search candidate set, and reconstructing without them measures a
    candidate list the agent never had.
    """
    from osg.core.config import SceneGraphConfig
    from osg.graph.map_store import apply_map, load_map
    from osg.graph.scene_graph import SceneGraph
    from osg.mapping.costmap import Costmap2D
    from osg.mapping.room_seg import VoronoiRoomSegmenter
    from osg.objects.object_layer import ObjectLayer

    blob = load_map(map_path)
    cfg = sg_cfg if sg_cfg is not None else SceneGraphConfig()

    class _Shim:
        """The three attributes apply_map writes through."""

        def __init__(self) -> None:
            self.object_layer = ObjectLayer()
            self.costmap = Costmap2D(resolution=float(blob.get("resolution", 0.05)))
            self.scene_graph = SceneGraph(
                container_top_h_m=tuple(cfg.container_top_h_m),
                container_min_area_m2=float(cfg.container_min_area_m2),
                container_support_tol_m=float(cfg.container_support_tol_m),
                container_min_obs=int(cfg.container_min_obs),
                container_min_score=float(cfg.container_min_score),
                container_merge_m=float(cfg.container_merge_m),
            )
            self._room_labels = None

    shim = _Shim()
    apply_map(shim, blob)
    # apply_map only rebuilds when the snapshot carried room labels; segment
    # here so the container layer exists either way.
    if not shim.scene_graph.containers:
        labels = VoronoiRoomSegmenter().segment(shim.costmap)
        shim.scene_graph.rebuild(labels, shim.costmap, shim.object_layer, floors=None)
    return shim.scene_graph


def rank_of_truth(
    scene_graph,
    target: str,
    truth_xy: Tuple[float, float],
    last_known_xy: Optional[Sequence[float]],
    near_m: float,
) -> Tuple[Optional[int], int, int, Optional[str]]:
    """(rank of the true surface, n candidates, n true surfaces, its label).

    Ranked exactly as `select_candidate` ranks before its top-N cut: by
    `prior * detect_prob`, with detect_prob constant, so this IS the cut order.
    """
    from osg.exploration.search_belief import InspectionLog, build_container_candidates

    cands = build_container_candidates(
        scene_graph, target, InspectionLog(),
        last_known_xy=None if last_known_xy is None else np.asarray(last_known_xy, float),
    )
    ranked = sorted(cands, key=lambda c: -(c.prior * c.detect_prob))
    truth_ids = {
        int(cid) for cid, node in scene_graph.containers.items()
        if _footprint_distance(node, truth_xy) <= near_m
    }
    for i, cand in enumerate(ranked):
        if int(cand.ref_id) in truth_ids:
            return i + 1, len(ranked), len(truth_ids), cand.label
    return None, len(ranked), len(truth_ids), None


def simulate_search(
    scene_graph,
    target: str,
    truth_xy,
    start_xy,
    last_known_xy,
    near_m: float,
    top_n: int,
    budget: int,
) -> Optional[int]:
    """How many inspections until the greedy search reaches the true surface?

    A faithful offline stand-in for the agent's search line: the same candidate
    build, the same top-N cut, the same `prior * d / cost` argmax, the same
    multiplicative belief decay after an unsuccessful look. Path cost is
    Euclidean rather than geodesic, which flatters every ordering equally.

    Returns the 1-based inspection index that lands on the true surface, or None
    if `budget` inspections run out first.
    """
    from osg.exploration.search_belief import InspectionLog, build_container_candidates

    log = InspectionLog()
    here = np.asarray(start_xy, dtype=float)
    truth_ids = {
        int(cid) for cid, node in scene_graph.containers.items()
        if _footprint_distance(node, truth_xy) <= near_m
    }
    if not truth_ids:
        return None
    for step in range(1, budget + 1):
        cands = build_container_candidates(
            scene_graph, target, log,
            last_known_xy=None if last_known_xy is None
            else np.asarray(last_known_xy, dtype=float))
        if not cands:
            return None
        # THE CUT, exactly as select_candidate makes it: by prior alone, before
        # any path is costed. With a flat prior this is close to arbitrary --
        # 64 candidates carried 5 distinct values, so 12 tied desks were cut to
        # 3 in track-id order. The proximity term is what puts real numbers on
        # this axis, which is why it is worth more here than its size suggests.
        ranked = sorted(cands, key=lambda c: -(c.prior * c.detect_prob))[:top_n]
        best, best_util = None, -1.0
        for cand in ranked:
            cost = max(float(np.linalg.norm(cand.goal_xy - here)), 0.5)
            util = cand.prior * cand.detect_prob / cost
            if util > best_util:
                best, best_util = cand, util
        if best is None:
            return None
        if int(best.ref_id) in truth_ids:
            return step
        log.searched(int(best.ref_id), float(best.detect_prob))
        here = np.asarray(best.goal_xy, dtype=float)
    return None


def _summarise(name: str, ranks: List[Optional[int]], totals: List[int], top_n: int) -> None:
    got = [r for r in ranks if r is not None]
    n = len(ranks)
    print(f"\n  {name}")
    if not got:
        print("    the true surface is never a candidate at all")
        return
    got_sorted = sorted(got)
    print(f"    true surface is a candidate      : {len(got)}/{n}")
    print(f"    rank  median / p25 / p75 / worst : "
          f"{got_sorted[len(got_sorted) // 2]} / {got_sorted[len(got_sorted) // 4]} / "
          f"{got_sorted[3 * len(got_sorted) // 4]} / {got_sorted[-1]}")
    print(f"    candidates offered (median)      : {sorted(totals)[len(totals) // 2]}")
    for k in (1, top_n, 10, 20):
        hit = sum(1 for r in got if r <= k)
        print(f"    in the top {k:<3d}                     : {hit}/{n}  ({hit / n:.0%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default="outputs/maps_hires",
                    help="directory of prior maps, one subdirectory per scene")
    ap.add_argument("--layouts", default="outputs/substituted_layouts")
    ap.add_argument("--layout-types", default="in_anchor,cross_anchor")
    ap.add_argument("--layout-indices", default="1,2,3")
    ap.add_argument("--top-n", type=int, default=5,
                    help="surfaces an episode can afford to inspect (measured median: 5)")
    ap.add_argument("--near-m", type=float, default=0.5,
                    help="how close to a container's footprint counts as ON it")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--budget", type=int, default=8,
                    help="inspections an episode can afford before its steps run out")
    ap.add_argument("--episodes", default="outputs/2026082*/episodes.jsonl",
                    help="glob of episode logs, read only for their start poses")
    # Container-layer gates. The true surface is lost here far more often than
    # it is mis-ranked, so these are the knobs worth sweeping, and sweeping them
    # offline costs a second instead of an eight-hour benchmark.
    ap.add_argument("--min-obs", type=int, default=None)
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--merge-m", type=float, default=None)
    ap.add_argument("--top-h-max", type=float, default=None)
    ap.add_argument("--min-area", type=float, default=None)
    args = ap.parse_args()

    from osg.core.config import SceneGraphConfig, register_configs

    register_configs()

    sg_cfg = SceneGraphConfig()
    if args.min_obs is not None:
        sg_cfg.container_min_obs = args.min_obs
    if args.min_score is not None:
        sg_cfg.container_min_score = args.min_score
    if args.merge_m is not None:
        sg_cfg.container_merge_m = args.merge_m
    if args.min_area is not None:
        sg_cfg.container_min_area_m2 = args.min_area
    if args.top_h_max is not None:
        sg_cfg.container_top_h_m = (sg_cfg.container_top_h_m[0], args.top_h_max)
    print(f"container gates: min_obs={sg_cfg.container_min_obs} "
          f"min_score={sg_cfg.container_min_score} merge_m={sg_cfg.container_merge_m} "
          f"top_h={tuple(sg_cfg.container_top_h_m)} min_area={sg_cfg.container_min_area_m2}")

    maps_root = Path(args.maps)
    layout_root = Path(args.layouts)
    types = [t for t in args.layout_types.split(",") if t]
    indices = [int(i) for i in args.layout_indices.split(",") if i]

    rows: List[dict] = []
    for scene_dir in sorted(p for p in maps_root.iterdir() if p.is_dir()):
        scene = scene_dir.name
        map_path = scene_dir / f"{scene}.json"
        if not map_path.is_file():
            found = sorted(scene_dir.glob("*.json"))
            if not found:
                continue
            map_path = found[0]
        graph = load_scene_graph(map_path, sg_cfg)
        n_cont = len(graph.containers)

        static = layout_root / scene / "static_scene_config.json"
        was: Dict[str, Tuple[float, float]] = {}
        if static.is_file():
            blob = json.loads(static.read_text(encoding="utf-8"))
            mapping = {int(k): v for k, v in blob["id_handle_mapping"].items()}
            for obj in blob["objects"]:
                label = TARGET_LABELS.get(mapping[int(obj["semantic_id"])])
                if label:
                    was[label] = (obj["translation"][0], obj["translation"][2])

        starts: Dict[tuple, tuple] = {}
        for run in Path(".").glob(args.episodes) if args.episodes else []:
            for line in run.open(encoding="utf-8"):
                rec = json.loads(line)
                if rec.get("scene") != scene:
                    continue
                al = rec["authored_layout"]
                if al.get("layout_index") is None:
                    continue  # the static pass has no index
                pos = al["start"]["position"]
                starts[(al["layout_type"], int(al["layout_index"]), rec["target"])] = (
                    pos[0], pos[2])

        print(f"\n{scene}: {n_cont} containers in the prior map")
        for ltype in types:
            for idx in indices:
                path = (layout_root / scene / "dynamic_scene_config" / ltype
                        / f"layout_{idx:02d}.json")
                if not path.is_file():
                    continue
                blob = json.loads(path.read_text(encoding="utf-8"))
                mapping = {int(k): v for k, v in blob["id_handle_mapping"].items()}
                for obj in blob["objects"]:
                    label = TARGET_LABELS.get(mapping[int(obj["semantic_id"])])
                    if label is None:
                        continue
                    truth = (obj["translation"][0], obj["translation"][2])
                    r_prox, n_c, n_t, lab = rank_of_truth(
                        graph, label, truth, was.get(label), args.near_m)
                    r_flat, _, _, _ = rank_of_truth(graph, label, truth, None, args.near_m)
                    start = starts.get((ltype, idx, label)) or truth
                    sim_now = simulate_search(
                        graph, label, truth, start, was.get(label), args.near_m,
                        args.top_n, args.budget)
                    sim_fix = simulate_search(
                        graph, label, truth, start, None, args.near_m,
                        args.top_n, args.budget)
                    rows.append({
                        "scene": scene, "layout": f"{ltype}_{idx:02d}", "target": label,
                        "rank_with_proximity": r_prox, "rank_after_absence": r_flat,
                        "n_candidates": n_c, "n_true_surfaces": n_t,
                        "surface_label": lab, "n_containers": n_cont,
                        "inspections_with_proximity": sim_now,
                        "inspections_flat": sim_fix,
                    })

    if not rows:
        raise SystemExit("no (map, layout) pair matched; check --maps and --layouts")

    print("\n" + "=" * 74)
    print(f"RANK OF THE TRUE SURFACE   ({len(rows)} scene-layout-target combinations)")
    print("=" * 74)
    _summarise("with the proximity term -- what the agent ranks by",
               [r["rank_with_proximity"] for r in rows],
               [r["n_candidates"] for r in rows], args.top_n)
    _summarise("flat prior, distance ignored -- the baseline it replaced",
               [r["rank_after_absence"] for r in rows],
               [r["n_candidates"] for r in rows], args.top_n)

    for name, key in (("in_anchor", "in_anchor"), ("cross_anchor", "cross_anchor")):
        sub = [r for r in rows if r["layout"].startswith(key)]
        if not sub:
            continue
        got = [r["rank_with_proximity"] for r in sub if r["rank_with_proximity"]]
        flat = [r["rank_after_absence"] for r in sub if r["rank_after_absence"]]
        print(f"\n  {name}: top-{args.top_n} "
              f"{sum(1 for x in got if x <= args.top_n)}/{len(sub)} with proximity, "
              f"{sum(1 for x in flat if x <= args.top_n)}/{len(sub)} flat")

    print("\n  per target, rank with the proximity term:")
    by: Dict[str, List[int]] = {}
    for r in rows:
        if r["rank_with_proximity"] is not None:
            by.setdefault(r["target"], []).append(r["rank_with_proximity"])
    for target in sorted(by):
        got = sorted(by[target])
        top = sum(1 for x in got if x <= args.top_n)
        print(f"    {target:22} median rank {got[len(got) // 2]:4d}   "
              f"top-{args.top_n}: {top}/{len(got)}")

    print("\n" + "=" * 74)
    print(f"GREEDY SEARCH SIMULATED FROM THE EPISODE START  (budget {args.budget} "
          f"inspections, top-N {args.top_n})")
    print("=" * 74)
    for key, name in (("inspections_with_proximity", "with the proximity term"),
                      ("inspections_flat", "flat prior, distance ignored")):
        got = sorted(r[key] for r in rows if r[key] is not None)
        print(f"\n  {name}")
        print(f"    reaches the true surface within budget : {len(got)}/{len(rows)}"
              f"  ({len(got) / len(rows):.0%})")
        if got:
            print(f"    inspections needed, median / worst    : "
                  f"{got[len(got) // 2]} / {got[-1]}")
            for k in (1, 3, 5):
                print(f"    within {k} inspections                    : "
                      f"{sum(1 for x in got if x <= k)}/{len(rows)}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"\n  wrote {args.json_out}")


if __name__ == "__main__":
    main()
