"""Rank a ConceptGraphs map's objects against the DualMap benchmark's queries.

Runs in the ConceptGraphs environment. Reads the post-processed map `cfslam_pipeline_batch.py`
saves (`pcd_saves/full_pcd_<variant>_<suffix>_post.pkl.gz`) and answers each query as ConceptGraphs'
own interactive query does (`scripts/visualize_cfslam_results.py`): open_clip ViT-H-14
(laion2b_s32b_b79k) text feature of the query, normalised, cosine similarity against each object's
fused `clip_ft`. Object centres and sizes are the axis-aligned box of the object's points, which
are in the release's world frame (export_sequence.py hands ConceptGraphs the release's poses).

Writes the shared `rankings.json` (scripts/baselines/khronos/query_map.py).

    python query_map.py --map <scene dir>/pcd_saves/<...>_post.pkl.gz --scene <scene> --out <dir>/rankings.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np

from osg.eval.dualmap_release import STATIC_QUERIES, TARGETS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, type=Path)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import open_clip
    import torch
    import torch.nn.functional as F

    with gzip.open(args.map, "rb") as fh:
        result = pickle.load(fh)
    objects = []
    for i, obj in enumerate(result["objects"]):
        pts = np.asarray(obj["pcd_np"], dtype=np.float64)
        if len(pts) == 0:
            continue
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        objects.append({"uid": str(i), "center": ((lo + hi) / 2).tolist(), "dimensions": (hi - lo).tolist(),
                        "clip_ft": np.asarray(obj["clip_ft"], dtype=np.float32)})
    if not objects:
        raise SystemExit(f"{args.map}: no objects")

    model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    model = model.to("cuda").eval()
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    queries = list(dict.fromkeys([*STATIC_QUERIES[args.scene], *TARGETS[args.scene]]))
    object_fts = torch.from_numpy(np.stack([o["clip_ft"] for o in objects])).to("cuda")
    rankings = {}
    with torch.no_grad():
        for query in queries:
            ft = model.encode_text(tokenizer([query]).to("cuda"))
            ft = (ft / ft.norm(dim=-1, keepdim=True)).squeeze()
            sims = F.cosine_similarity(ft.unsqueeze(0), object_fts, dim=-1).cpu().numpy().astype(np.float64)
            order = np.argsort(-sims, kind="stable")
            rankings[query] = [[objects[i]["uid"], round(float(sims[i]), 6)] for i in order]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "method": "conceptgraphs",
        "scene": args.scene,
        "map": str(args.map.resolve()),
        "encoder": "open_clip ViT-H-14 laion2b_s32b_b79k, raw query text (visualize_cfslam_results.py)",
        "objects": [{k: o[k] for k in ("uid", "center", "dimensions")} for o in objects],
        "rankings": rankings,
    }, indent=1) + "\n")
    print(f"{args.scene}: {len(objects)} objects, {len(queries)} queries -> {args.out}")


if __name__ == "__main__":
    main()
