"""Clio-batch on a primitive map, ranked against the DualMap benchmark's queries.

Clio clusters a scene graph of object primitives into the objects a list of tasks needs
(information bottleneck, Maggio et al. 2024). This runs its released batch clustering,
`clio_batch.object_cluster.cluster_3d`, unchanged, on primitives mapped by Khronos's active window
(`scripts/baselines/khronos/map_sequence.sh` with Clio's fine-primitive settings,
`config/mapper_dualmap_clio_primitives.yaml`), the front end Clio itself runs on. The tasks are the
scene's queries in `osg.eval.dualmap_release`: Clio is told what will be asked, as its protocol
assumes. Its clustering parameters are the released `large_delta_00001.yaml` (delta 1e-5,
task-similarity threshold 0.23 -- clusters below it for every task are dropped -- top-2 tasks,
cumulative), the values its online pipeline uses too.

Khronos writes its primitives to the OBJECTS layer and Clio reads them from SEGMENTS, so they are
moved there first; nothing else about them changes.

Writes `clio_dsg.json` (Clio's output) and the shared `rankings.json`
(see `scripts/baselines/khronos/query_map.py`): every Clio object ranked by cosine similarity
between its feature and the query, embedded by Clio's own ClipHandler (ViT-L/14).

    python cluster_map.py --map <out>/khronos --scene 00829-QaLdnwvtxbs --out <out>/clio
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import spark_dsg as dsg
import yaml

from osg.eval.dualmap_release import STATIC_QUERIES, TARGETS

CLIO_ROOT = Path(__file__).resolve().parents[3] / "relative_work/baselines/clio"
CLUSTER_CONFIG = CLIO_ROOT / "clio_eval/experiments/configs/cluster/large_delta_00001.yaml"


def primitives_as_segments(src: Path, dst: Path) -> int:
    graph = dsg.DynamicSceneGraph.load(str(src))
    objects = list(graph.get_layer(dsg.DsgLayers.OBJECTS).nodes)
    ids = [node.id.value for node in objects]
    if not graph.has_layer(dsg.DsgLayers.SEGMENTS):
        graph.add_layer(1, 0, dsg.DsgLayers.SEGMENTS)  # spark_dsg's id for SEGMENTS
    for index, node in enumerate(objects):
        graph.add_node(dsg.DsgLayers.SEGMENTS, dsg.NodeSymbol("S", index).value, node.attributes)
    for node_id in ids:
        graph.remove_node(node_id)
    graph.save(str(dst))
    return len(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, type=Path, help="Khronos's output directory (primitives)")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="ViT-L/14")
    ap.add_argument("--cluster-config", type=Path, default=CLUSTER_CONFIG)
    args = ap.parse_args()

    import clio_batch.helpers as helpers
    from clio_batch.object_cluster import cluster_3d

    args.out.mkdir(parents=True, exist_ok=True)
    queries = list(dict.fromkeys([*STATIC_QUERIES[args.scene], *TARGETS[args.scene]]))
    tasks = args.out / "tasks.yaml"
    tasks.write_text(yaml.safe_dump({q: [] for q in queries}, sort_keys=False))
    segments = args.out / "segments_dsg.json"
    # The complete final map, with the objects still in the active window at the end
    # (see scripts/baselines/khronos/query_map.py, final_map).
    source = args.map / "backend/dsg_with_mesh.sparkdsg"
    if not source.is_file():
        source = args.map / "dsg.json"
    n_primitives = primitives_as_segments(source, segments)
    config = yaml.safe_load(args.cluster_config.read_text())

    graph = cluster_3d(str(segments), str(tasks), str(args.out), str(args.cluster_config), args.model,
                       config["sims_thres"], None, 0.0, False)
    graph.save(str(args.out / "clio_dsg.json"))

    objects = []
    for node in graph.get_layer(dsg.DsgLayers.OBJECTS).nodes:
        attrs = node.attributes
        feature = np.asarray(attrs.semantic_feature, dtype=np.float64).reshape(-1)
        box = attrs.bounding_box
        objects.append({
            "uid": str(node.id.value),
            "center": [float(v) for v in np.asarray(box.world_P_center).reshape(-1)],
            "dimensions": [float(v) for v in np.asarray(box.dimensions).reshape(-1)],
            "feature": feature / max(np.linalg.norm(feature), 1e-12),
        })
    if not objects:
        raise SystemExit("Clio kept no objects")
    clip = helpers.ClipHandler(args.model)
    text = clip.get_text_clip_features(queries).astype(np.float64)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    scores = np.stack([o["feature"] for o in objects]) @ text.T
    rankings = {}
    for j, query in enumerate(queries):
        order = np.argsort(-scores[:, j], kind="stable")
        rankings[query] = [[objects[i]["uid"], round(float(scores[i, j]), 6)] for i in order]
    (args.out / "rankings.json").write_text(json.dumps({
        "method": "clio_batch",
        "scene": args.scene,
        "map": str((args.out / "clio_dsg.json").resolve()),
        "encoder": f"Clio ClipHandler {args.model}, raw query text",
        "primitives": n_primitives,
        "objects": [{k: o[k] for k in ("uid", "center", "dimensions")} for o in objects],
        "rankings": rankings,
    }, indent=1) + "\n")
    print(f"{args.scene}: {n_primitives} primitives -> {len(objects)} Clio objects, {len(queries)} queries")


if __name__ == "__main__":
    main()
