"""Rank a Khronos map's objects against the DualMap benchmark's text queries.

Runs in the Khronos environment (`env.sh`, with the repository's `src` on PYTHONPATH for the
protocol module). Reads the final scene graph Khronos saved on shutdown (`final_map`),
embeds each query with the encoder that produced the objects' features -- semantic_inference's
ClipWrapper, ViT-L/14, the open-set segmenter's -- and ranks every object by cosine similarity.
The query is embedded as it is, as DualMap embeds its own (`Dualmap.convert_inquiry_to_feat`).

The queries are the scene's in `osg.eval.dualmap_release` (static and dynamic), so the ranking
covers every trial of the benchmark and nothing else.

Writes one JSON, the format every static-map baseline hands to `scripts/baselines/static_map_nav.py`:

    {"method", "scene", "map", "encoder",
     "objects":  [{"uid", "center", "dimensions"}, ...],     # the map's world frame (pose.txt's, z up)
     "rankings": {query: [[uid, score], ...]}}               # best first, every object

    python query_map.py --map <out>/khronos --scene 00829-QaLdnwvtxbs --out <out>/rankings.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from osg.eval.dualmap_release import STATIC_QUERIES, TARGETS


def final_map(map_dir: Path) -> Path:
    """The complete final map Khronos saves.

    `dsg.json` is the backend's graph; the objects still inside the active window when the
    sequence ends (its last few seconds) are only in `backend/dsg_with_mesh.sparkdsg`, which
    KhronosPipeline::save writes with them added.
    """
    full = map_dir / "backend/dsg_with_mesh.sparkdsg"
    return full if full.is_file() else map_dir / "dsg.json"


def load_objects(path: Path):
    import spark_dsg as dsg

    graph = dsg.DynamicSceneGraph.load(str(path))
    objects = []
    for node in graph.get_layer(dsg.DsgLayers.OBJECTS).nodes:
        attrs = node.attributes
        # One column per object: Khronos keeps the running mean of its segments' features.
        vec = np.asarray(attrs.semantic_feature, dtype=np.float64).reshape(-1)
        box = attrs.bounding_box
        objects.append({
            "uid": str(node.id.value),
            "center": [float(v) for v in np.asarray(box.world_P_center).reshape(-1)],
            "dimensions": [float(v) for v in np.asarray(box.dimensions).reshape(-1)],
            "feature": vec / max(np.linalg.norm(vec), 1e-12),
        })
    return objects


def encoder(model_name: str):
    import torch
    from semantic_inference.models.wrappers import ClipConfig, ClipWrapper

    config = ClipConfig()
    config.model_name = model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClipWrapper(config).to(device).eval()

    def embed(text: str) -> np.ndarray:
        with torch.no_grad():
            v = model.embed_text(text).float().cpu().numpy().squeeze().astype(np.float64)
        return v / max(np.linalg.norm(v), 1e-12)

    return embed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, type=Path, help="Khronos's output directory")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="ViT-L/14")
    args = ap.parse_args()

    source = final_map(args.map)
    objects = load_objects(source)
    if not objects:
        raise SystemExit(f"{source} has no object nodes")
    queries = list(dict.fromkeys([*STATIC_QUERIES[args.scene], *TARGETS[args.scene]]))
    embed = encoder(args.model)
    features = np.stack([o["feature"] for o in objects])
    rankings = {}
    for query in queries:
        scores = features @ embed(query)
        order = np.argsort(-scores, kind="stable")
        rankings[query] = [[objects[i]["uid"], round(float(scores[i]), 6)] for i in order]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "method": "khronos",
        "scene": args.scene,
        "map": str(source.resolve()),
        "encoder": f"semantic_inference ClipWrapper {args.model}, raw query text",
        "objects": [{k: o[k] for k in ("uid", "center", "dimensions")} for o in objects],
        "rankings": rankings,
    }, indent=1) + "\n")
    print(f"{args.scene}: {len(objects)} objects ({source.name}), {len(queries)} queries -> {args.out}")


if __name__ == "__main__":
    main()
