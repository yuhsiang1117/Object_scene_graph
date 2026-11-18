#!/usr/bin/env python3
"""Rebuild DualMap's static global map for a swapped copy of its benchmark.

DualMap's released `global_map/hm3d` per scene was built from the shipped
RGB-D sequence of the original static scene. After an asset swap
(scripts/make_dualmap_swap.py) that map still holds the old objects, so it is
rebuilt here from the same trajectory re-rendered in the swapped scene
(scripts/render_dualmap_sequence.py), by DualMap's own dataset runner with
its own models and class list plus the swapped-in names. Writes the map into
`<release>/<scene>/global_map/hm3d`, where the released-benchmark harness
preloads it. The released data is never touched: the destination must be
outside it.

    python scripts/legacy/build_dualmap_swap_map.py --scene 00848-ziup5kvtCCR \
        --release $OSG_DATA_ROOT/dualmap_swap/HM3D_collect \
        --class-list configs/dualmap_swap/hm3d300_classes_ycb_swap.txt
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from osg.core.paths import collector_data_root  # noqa: E402

DUALMAP_ROOT = REPO / "outputs/dualmap_comparison/vendor/DualMap"
# The collector mount moved from /datasets/habitat-data-collector to
# /habitat-data-collector; osg.core.paths carries both, so ask it.
ORIGINAL = collector_data_root() / "dualmap/HM3D_collect"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--class-list", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=None, help="DualMap's output_path (logs, detections)")
    parser.add_argument("--use-fastsam", action="store_true")
    args = parser.parse_args()

    release = args.release.resolve()
    if str(release).startswith(str(ORIGINAL.resolve())):
        raise SystemExit("refusing to write a map into the released data")
    scene_dir = release / args.scene
    for name in ("rgb", "depth"):
        p = scene_dir / name
        if p.is_symlink() or not p.exists():
            raise SystemExit(f"{p} must be a real rendered sequence (scripts/render_dualmap_sequence.py)")
    if not (scene_dir / "pose.txt").exists():
        raise SystemExit(f"{scene_dir / 'pose.txt'} is missing")
    n_rgb = len(list((scene_dir / "rgb").glob("*.png")))
    n_pose = sum(1 for l in (scene_dir / "pose.txt").read_text().splitlines() if l.strip())
    print(f"{args.scene}: {n_rgb} rgb frames, {n_pose} poses", flush=True)
    map_dir = scene_dir / "global_map" / "hm3d"
    if map_dir.exists():
        shutil.rmtree(map_dir)
    work = (args.work or (REPO / "outputs/dualmap_swap_support/map_build" / args.scene)).resolve()
    work.mkdir(parents=True, exist_ok=True)
    class_list = args.class_list.resolve()

    sys.path.insert(0, str(DUALMAP_ROOT))
    os.chdir(DUALMAP_ROOT)
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(DUALMAP_ROOT / "config"), version_base=None):
        cfg = compose(
            config_name="runner_dataset",
            overrides=[
                "dataset_name=self_collected",
                "dataset_conf_path=config/data_config/dataset/self_collected.yaml",
                f"dataset_path={release}",
                f"scene_id={args.scene}",
                "use_stride=false",
                "run_local_mapping_only=false",
                "save_local_map=false",
                "save_global_map=true",
                "use_end_process=true",
                "use_rerun=false",
                "use_parallel=true",
                f"use_fastsam={'true' if args.use_fastsam else 'false'}",
                "preload_global_map=false",
                f"output_path={work}",
                f"map_save_path={map_dir}",
                "yolo.given_classes_path=config/class_list/hm3d300_classes_ycb.txt",
            ],
        )
    for group, key in (("yolo", "model_path"), ("sam", "model_path"), ("fastsam", "model_path")):
        if not Path(str(cfg[group][key])).is_absolute():
            cfg[group][key] = str(DUALMAP_ROOT / str(cfg[group][key]))
    cfg.yolo.given_classes_path = str(class_list)
    print(f"building into {map_dir} (work {work}); classes {class_list.name}", flush=True)

    from applications.runner_dataset import main as run

    run(cfg)
    n = len(list(map_dir.glob("*.pkl")))
    print(f"wrote {map_dir}: {n} global objects", flush=True)


if __name__ == "__main__":
    main()
