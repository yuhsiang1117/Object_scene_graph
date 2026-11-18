#!/usr/bin/env python3
"""A copy of DualMap's released HM3D benchmark with two assets replaced.

The released data is never edited: this writes a sibling tree with the same
scenes, the same layouts, the same object positions, and the asset handle of
the swapped objects changed (`id_handle_mapping`). A swapped object keeps the
old object's (x, z) and its supporting surface -- the new mesh is stood
upright (identity rotation; every YCB config declares `up: [0,0,1]`) with its
bottom where the old object's bottom was, so a 25 cm bottle put in the place
of a 1.6 cm-thick pair of scissors rests on the bed instead of inside it.
Big unchanged files (rgb, depth, data.zip, pose, intrinsics, class tables)
are symlinked; `global_map/` is NOT carried over, because DualMap's prior map
was built from the original static scene and must be rebuilt for the new one
(the README written beside the copy says so).

`swap.json` at the copy's root records the mapping; `osg.eval.dualmap_release`
reads it when `OSG_DUALMAP_RELEASE_ROOT` points at the copy, and renames the
queries in the protocol accordingly.

    python scripts/make_dualmap_swap.py \
        --dst $COLLECTOR_DATA/dualmap_swap/HM3D_collect \
        --swap 037_scissors=006_mustard_bottle 025_mug=002_master_chef_can
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

# The collector mount moved from /datasets/habitat-data-collector to
# /habitat-data-collector; osg.core.paths carries both, so ask it.
from osg.core.paths import collector_data_root  # noqa: E402

_COLLECTOR = collector_data_root()

DEFAULT_SRC = Path(f"{_COLLECTOR}/dualmap/HM3D_collect")
YCB = Path(f"{_COLLECTOR}/versioned_data/ycb")
QUERY_NAME = {
    "006_mustard_bottle": "mustard bottle",
    "002_master_chef_can": "coffee can",
    "072-a_toy_airplane": "toy airplane",
    "021_bleach_cleanser": "bleach bottle",
    "004_sugar_box": "sugar box",
    "010_potted_meat_can": "potted meat can",
}
OLD_QUERY = {"037_scissors": "scissors", "025_mug": "mug"}
LINK = ("rgb", "depth", "data.zip", "pose.txt", "camera_intrinsics.json",
        "class_bbox.json", "class_num.json", "rosbag2_odom")
# YCB meshes are z-up, y-front; habitat stands them with +z -> +y, +y -> -z.
R0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def quat_to_R(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mesh_vertices(handle: str) -> np.ndarray:
    import trimesh

    cfg = json.loads((YCB / "configs" / f"{handle}.object_config.json").read_text())
    path = (YCB / "configs" / cfg["render_asset"]).resolve()
    mesh = trimesh.load(str(path), force="mesh")
    return np.asarray(mesh.vertices, dtype=float) * np.asarray(cfg.get("scale", [1, 1, 1]), dtype=float)


def bottom_y(verts_world: np.ndarray) -> float:
    return float(verts_world[:, 1].min())


def swap_layout(data: dict, swaps: dict, verts: dict) -> tuple[dict, list]:
    mapping = {str(k): str(v) for k, v in data["id_handle_mapping"].items()}
    changed = []
    new_mapping = {k: swaps.get(v, v) for k, v in mapping.items()}
    objects = []
    for obj in data["objects"]:
        sid = str(obj["semantic_id"])
        old = mapping[sid]
        if old in swaps:
            new = swaps[old]
            t = np.asarray(obj["translation"], dtype=float)
            R_old = quat_to_R(obj["rotation"]) @ R0
            old_bottom = t[1] + bottom_y(verts[old] @ R_old.T)
            new_bottom_offset = bottom_y(verts[new] @ R0.T)
            t_new = np.array([t[0], old_bottom - new_bottom_offset, t[2]])
            objects.append({**obj, "translation": [float(v) for v in t_new],
                            "rotation": [0.0, 0.0, 0.0, 1.0]})
            changed.append({"semantic_id": int(sid), "from": old, "to": new,
                            "old_y": round(float(t[1]), 4), "new_y": round(float(t_new[1]), 4),
                            "surface_y": round(float(old_bottom), 4)})
        else:
            objects.append(obj)
    return {**data, "id_handle_mapping": new_mapping, "objects": objects}, changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--swap", nargs="+", required=True, help="OLD_HANDLE=NEW_HANDLE ...")
    parser.add_argument("--force", action="store_true", help="overwrite an existing copy")
    args = parser.parse_args()
    swaps = dict(s.split("=", 1) for s in args.swap)
    for old, new in swaps.items():
        if old not in OLD_QUERY:
            raise SystemExit(f"{old}: only the two benchmark assets this repo has evidence on can be swapped")
        if new not in QUERY_NAME:
            raise SystemExit(f"{new}: add its query name to QUERY_NAME first")
    if args.dst.resolve() == args.src.resolve() or str(args.dst.resolve()).startswith(str(args.src.resolve()) + os.sep):
        raise SystemExit("the destination must not be inside the released data")
    if args.dst.exists():
        if not args.force:
            raise SystemExit(f"{args.dst} exists; pass --force to rebuild it")
        shutil.rmtree(args.dst)
    verts = {h: mesh_vertices(h) for h in set(swaps) | set(swaps.values())}
    manifest = {"source": str(args.src), "handles": swaps,
                "queries": {OLD_QUERY[o]: QUERY_NAME[n] for o, n in swaps.items()},
                "labels": {QUERY_NAME[n]: QUERY_NAME[n] for n in swaps.values()},
                "scenes": {}, "note": "global_map/ is not copied: DualMap's prior map must be rebuilt "
                                       "from the swapped static scene; so must outputs/maps_* for our stack."}
    for scene_dir in sorted(p for p in args.src.iterdir() if p.is_dir()):
        scene = scene_dir.name
        out = args.dst / scene
        out.mkdir(parents=True)
        log = []
        for rel in ["static_scene_config.json"] + [str(p.relative_to(scene_dir)) for p in
                                                   sorted(scene_dir.glob("dynamic_scene_config/*/*.json"))]:
            data = json.loads((scene_dir / rel).read_text(encoding="utf-8"))
            new, changed = swap_layout(data, swaps, verts)
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
            log.append({"layout": rel, "changed": changed})
        for name in LINK:
            if (scene_dir / name).exists():
                os.symlink(scene_dir / name, out / name)
        manifest["scenes"][scene] = log
    for name in ("convert.py", "download_manifest.json"):
        if (args.src / name).exists():
            os.symlink(args.src / name, args.dst / name)
    (args.dst / "swap.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.dst / "README.md").write_text(
        "# DualMap released benchmark, with assets swapped\n\n"
        f"Source: `{args.src}` (unmodified). Swaps: "
        + ", ".join(f"`{o}` -> `{n}` (query {OLD_QUERY[o]!r} -> {QUERY_NAME[n]!r})" for o, n in swaps.items())
        + ".\n\nEvery layout keeps its positions; a swapped object stands upright with its bottom on the "
        "old object's supporting surface (`swap.json` lists old and new heights per layout).\n\n"
        "`global_map/` is deliberately absent: DualMap's prior map must be rebuilt from the swapped "
        "static scene before its benchmark can be rerun here, and our `outputs/maps_*` prior maps "
        "likewise. Point `OSG_DUALMAP_RELEASE_ROOT` at this directory to run our protocol on it.\n",
        encoding="utf-8")
    n = sum(len(l["changed"]) for s in manifest["scenes"].values() for l in s)
    print(f"wrote {args.dst}: {len(manifest['scenes'])} scenes, {n} placements swapped; see swap.json")


if __name__ == "__main__":
    main()
