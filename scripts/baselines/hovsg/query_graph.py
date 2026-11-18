"""Rank a HOV-SG graph's objects against the DualMap benchmark's queries.

HOV-SG answers "object" queries by CLIP: the query's text feature, averaged over its templates
(`get_text_feats_multiple_templates`), against each object's embedding (`Graph.query_object`,
query_method "clip"). Its hierarchical queries ("the X in the Y on floor Z") parse the sentence
with an LLM first; the benchmark's queries name only an object, so that step has nothing to do and
is not run. `query_object` itself reads an unset list when no room is given, so the same
similarity is computed here, over every object of the graph, with HOV-SG's own text features.

Object centres and sizes are the axis-aligned box of each object's point cloud, brought from
HOV-SG's (Habitat's, y up) world back to the release's (z up) -- the inverse of export_sequence.py.

Writes the shared `rankings.json` (scripts/baselines/khronos/query_map.py).

    python query_graph.py --graph <save>/hm3dsem/<scene>/graph --scene <scene> --out <dir>/rankings.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from osg.eval.dualmap_release import STATIC_QUERIES, TARGETS

HOVSG_ROOT = Path(__file__).resolve().parents[3] / "relative_work/baselines/hovsg"


def hab_to_sys(v):
    x, y, z = (float(a) for a in v)
    return [x, -z, y]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, type=Path, help="HOV-SG's <save_path>/graph")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--clip-checkpoint", type=Path, default=HOVSG_ROOT / "checkpoints/laion2b_s32b_b79k.bin")
    args = ap.parse_args()

    from hovsg.graph.graph import Graph
    from hovsg.utils.clip_utils import get_text_feats_multiple_templates

    cfg = OmegaConf.create({"main": {"device": "cuda", "dataset": "hm3dsem"},
                            "models": {"clip": {"type": "ViT-H-14", "checkpoint": str(args.clip_checkpoint)}}})
    graph = Graph(cfg)  # no `pipeline` key: query mode, CLIP only
    graph.load_graph(str(args.graph))
    objects = []
    for obj in graph.objects:
        if obj.embedding is None or obj.pcd.is_empty():
            continue
        box = obj.pcd.get_axis_aligned_bounding_box()
        ext = box.get_extent()
        objects.append({
            "uid": str(obj.object_id),
            "center": hab_to_sys(box.get_center()),
            "dimensions": [float(ext[0]), float(ext[2]), float(ext[1])],
            # As stored: query_object ranks by the plain dot product, not the cosine.
            "feature": np.asarray(obj.embedding, dtype=np.float64).reshape(-1),
        })
    if not objects:
        raise SystemExit(f"{args.graph}: no objects with an embedding")
    queries = list(dict.fromkeys([*STATIC_QUERIES[args.scene], *TARGETS[args.scene]]))
    text = get_text_feats_multiple_templates(queries, graph.clip_model, graph.clip_feat_dim)
    text = np.asarray(text, dtype=np.float64)
    scores = np.stack([o["feature"] for o in objects]) @ text.T
    rankings = {}
    for j, query in enumerate(queries):
        order = np.argsort(-scores[:, j], kind="stable")
        rankings[query] = [[objects[i]["uid"], round(float(scores[i, j]), 6)] for i in order]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "method": "hovsg",
        "scene": args.scene,
        "map": str(args.graph.resolve()),
        "encoder": "HOV-SG open_clip ViT-H-14 laion2b_s32b_b79k, get_text_feats_multiple_templates",
        "floors": len(graph.floors),
        "rooms": len(graph.rooms),
        "objects": [{k: o[k] for k in ("uid", "center", "dimensions")} for o in objects],
        "rankings": rankings,
    }, indent=1) + "\n")
    print(f"{args.scene}: {len(graph.floors)} floors, {len(graph.rooms)} rooms, {len(objects)} objects, "
          f"{len(queries)} queries -> {args.out}")


if __name__ == "__main__":
    main()
