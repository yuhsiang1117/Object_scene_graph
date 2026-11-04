"""Lay a DualMap release sequence out as a ConceptGraphs "replica" scene: <root>/<scene>/{results, traj.txt}.

ConceptGraphs' ReplicaDataset reads `results/frame*.jpg`, `results/depth*.png` (scaled by the
dataconfig's png_depth_scale) and `traj.txt`, one camera-to-world 4x4 per line, the camera in the
OpenCV convention. The release has all of it already: its `pose.txt` is T_world_cam with the camera
in OpenCV (world z up), and its depth is 16-bit millimetres (dataconfig `dualmap.yaml`,
png_depth_scale 1000, the release's own 1200x680 intrinsics). The images are linked, not copied or
re-encoded -- a PNG under a .jpg name is read by its content -- so ConceptGraphs sees the very
frames DualMap was built from, and its objects come out in the release's world frame.

    python export_sequence.py --scene 00829-QaLdnwvtxbs --out <root>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

RELEASE = Path(os.environ.get("OSG_DUALMAP_RELEASE_ROOT", "/habitat-data-collector/data/dualmap/HM3D_collect"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    src = RELEASE / args.scene
    rgb = sorted((src / "rgb").glob("*.png"))
    depth = sorted((src / "depth").glob("*.png"))
    poses = np.loadtxt(src / "pose.txt")
    if not (len(rgb) == len(depth) == len(poses)):
        raise SystemExit(f"{args.scene}: {len(rgb)} rgb, {len(depth)} depth, {len(poses)} poses")
    dst = args.out / args.scene
    (dst / "results").mkdir(parents=True, exist_ok=True)
    for i, (r, d) in enumerate(zip(rgb, depth)):
        if r.name != d.name:
            raise SystemExit(f"rgb {r.name} and depth {d.name} out of step")
        for name, f in ((f"frame{i:06d}.jpg", r), (f"depth{i:06d}.png", d)):
            link = dst / "results" / name
            if not link.exists():
                link.symlink_to(f)
    with open(dst / "traj.txt", "w") as fh:
        for row in poses:
            fh.write(" ".join(f"{v:.9g}" for v in row[1:]) + "\n")
    (dst / "camera_intrinsics.json").write_text((src / "camera_intrinsics.json").read_text())
    print(f"{args.scene}: {len(rgb)} frames -> {dst}")


if __name__ == "__main__":
    main()
