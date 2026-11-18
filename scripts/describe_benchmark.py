#!/usr/bin/env python3
"""What a YCB benchmark preset IS, before anything is run on it.

The reviews asked for the generation rules, the floor distribution, the
starts and the prior's coverage, none of which the paper states. Everything
except coverage (`scripts/audit_map_coverage.py`) is decided by the episode
manifests, so this reads them -- generating any that are not cached, which
needs the simulator but no model -- and prints:

  * episodes per scene and layout
  * relocation class (from the geometry: static, same_floor_near -- moved under
    the 1 m success radius --, same_floor, upward, downward) x whether the
    start is on the object's storey
  * start-to-goal geodesic distance, overall and by class
  * the storeys each scene's starts are drawn from
  * where the objects were put: destination anchor category and room, from
    the layout's own authoring record

    python scripts/describe_benchmark.py +experiment=mfv2_dataset
    python scripts/describe_benchmark.py +experiment=mfv2_dataset ycb.seed=43
    python scripts/describe_benchmark.py +experiment=mfv2_dataset --seeds 42 43 44 --json b.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

SUCCESS_M = 1.0
STOREY_M = 0.5


def _hyp(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def relocation_class(target: Dict[str, Any], layout_type: str) -> str:
    if layout_type == "static" or target.get("prior_position") is None:
        return "static"
    prior_y, dest_y = target.get("prior_floor_y"), target.get("destination_floor_y")
    if prior_y is None or dest_y is None:
        return "unknown"
    dy = float(dest_y) - float(prior_y)
    if abs(dy) > STOREY_M:
        return "upward" if dy > 0 else "downward"
    moved = _hyp(target["prior_position"], target["position"])
    return "same_floor_near" if moved <= SUCCESS_M else "same_floor"


def anchors(layout_path: Path) -> Dict[int, Dict[str, Any]]:
    try:
        blob = json.loads(layout_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {int(o["semantic_id"]): (o.get("anchor") or {}) for o in blob.get("objects", [])}


def episodes_for(overrides: List[str]) -> List[Dict[str, Any]]:
    from hydra import compose, initialize_config_dir

    from osg.core.config import register_configs
    from osg.sim.ycb_env import prepare_ycb_benchmark

    register_configs()
    root = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(root), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    prepared = prepare_ycb_benchmark(cfg)
    layouts = {(l.scene_name, l.layout_id): l for l in prepared.discovery.layouts}
    rows = []
    for manifest in prepared.manifests:
        meta = manifest.get("layout") or {}
        scene, layout_id = meta.get("scene"), meta.get("layout_id")
        layout = layouts.get((scene, layout_id))
        layout_type = getattr(layout, "layout_type", "?")
        anchor_by_id = anchors(Path(str(layout.layout_path))) if layout is not None else {}
        for item in manifest.get("episodes", []):
            target = item["target"]
            start = item["start"]
            dest_y = target.get("destination_floor_y")
            start_y = float(start["position"][1])
            anchor = anchor_by_id.get(int(target["semantic_id"]), {})
            rows.append({
                "seed": int(cfg.ycb.seed), "scene": str(scene), "layout_type": layout_type,
                "episode_id": item["episode_id"], "label": target["label"],
                "class": relocation_class(target, layout_type),
                "start_on_target_storey": (
                    None if dest_y is None else abs(start_y - float(dest_y)) <= 1.0),
                "start_y": start_y, "destination_floor_y": dest_y,
                "geodesic_m": float(start.get("initial_geodesic_distance") or float("nan")),
                "anchor": anchor.get("category"), "anchor_kind": anchor.get("kind"),
                "room": anchor.get("room"),
            })
    return rows


def _q(values: List[float]) -> str:
    v = sorted(x for x in values if x == x)
    if not v:
        return "-"
    pick = lambda f: v[min(len(v) - 1, int(f * len(v)))]  # noqa: E731
    return f"{pick(0.1):.1f} / {pick(0.5):.1f} / {pick(0.9):.1f}"


def table(title: str, header: List[str], rows: List[List[str]]) -> None:
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


def report(rows: List[Dict[str, Any]]) -> None:
    seeds = sorted({r["seed"] for r in rows})
    print(f"{len(rows)} episodes, seeds {seeds}, "
          f"{len({r['scene'] for r in rows})} scenes")
    by_scene = defaultdict(Counter)
    for r in rows:
        by_scene[r["scene"]][r["layout_type"]] += 1
    types = sorted({r["layout_type"] for r in rows})
    table("Episodes per scene", ["scene"] + types + ["total"],
          [[s] + [c[t] for t in types] + [sum(c.values())] for s, c in sorted(by_scene.items())])

    classes = ["static", "same_floor_near", "same_floor", "upward", "downward", "unknown"]
    grid = []
    for cls in classes:
        part = [r for r in rows if r["class"] == cls]
        if not part:
            continue
        on = sum(1 for r in part if r["start_on_target_storey"])
        grid.append([cls, len(part), on, len(part) - on, _q([r["geodesic_m"] for r in part])])
    on_all = sum(1 for r in rows if r["start_on_target_storey"])
    grid.append(["all", len(rows), on_all, len(rows) - on_all, _q([r["geodesic_m"] for r in rows])])
    table("Relocation class x start storey",
          ["class", "episodes", "start on the object's storey", "start elsewhere",
           "geodesic m (p10 / p50 / p90)"], grid)

    heights = defaultdict(list)
    for r in rows:
        heights[r["scene"]].append(r["start_y"])
    storey_rows = []
    for scene, ys in sorted(heights.items()):
        groups: List[List[float]] = []
        for y in sorted(ys):           # a storey is a run of starts < 1 m apart
            if groups and y - groups[-1][-1] < 1.0:
                groups[-1].append(y)
            else:
                groups.append([y])
        storey_rows.append([scene, ", ".join(
            f"{sum(g) / len(g):.1f} m ({len(g)})" for g in groups)])
    table("Start storeys (mean floor height, starts drawn there)", ["scene", "storeys"],
          storey_rows)

    moved = [r for r in rows if r["class"] not in ("static",)]
    cats = Counter(str(r["anchor"]) for r in moved)
    table("Where moved objects were put (destination anchor category)", ["anchor", "episodes"],
          [[k, v] for k, v in cats.most_common(15)])
    pairs = Counter((r["label"], str(r["anchor_kind"])) for r in moved)
    table("Object -> destination anchor kind", ["object", "anchor kind", "episodes"],
          [[k[0], k[1], v] for k, v in pairs.most_common(20)])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. +experiment=mfv2_dataset")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    rows: List[Dict[str, Any]] = []
    for seed in (args.seeds or [None]):
        extra = [] if seed is None else [f"ycb.seed={seed}"]
        rows += episodes_for(list(args.overrides) + extra)
    report(rows)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
