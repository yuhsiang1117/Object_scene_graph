"""Lay a DualMap release sequence out as HOV-SG's HM3DSem walk: <root>/<split>/<scene>/{rgb,depth,pose}.

HOV-SG's `HM3DSemDataset` pairs the sorted files of the three folders, reads depth in millimetres,
derives the intrinsics from a 90-degree horizontal FOV, and reads each pose as a Habitat camera pose:
world y up, camera in the OpenGL convention, which it turns into OpenCV with diag(1, -1, -1).
The release matches the first three as it is -- 16-bit millimetre depth, and 1200x680 with
fx = 600 is a 90-degree HFOV -- so rgb and depth are linked, not copied. Its `pose.txt` is
T_world_cam with the world z up and the camera in OpenCV, so each pose is brought to Habitat's
world (collector CoordinateTransform: sys = (x, -z, y) of habitat) and to OpenGL:

    T_hab_gl = W_sys->hab . T_sys_cv . diag(1, -1, -1, 1)

HOV-SG's floors and rooms are cut by height along y, which is why its world must be Habitat's.

    python export_sequence.py --scene 00829-QaLdnwvtxbs --out <root>   # writes <root>/val/<scene>
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

RELEASE = Path(os.environ.get("OSG_DUALMAP_RELEASE_ROOT", "/habitat-data-collector/data/dualmap/HM3D_collect"))
# habitat (x, y, z) -> system (x, -z, y), and its inverse
WORLD_HAB_TO_SYS = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
WORLD_SYS_TO_HAB = np.linalg.inv(WORLD_HAB_TO_SYS)
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    src = RELEASE / args.scene
    K = json.loads((src / "camera_intrinsics.json").read_text())
    hfov = 2 * math.degrees(math.atan(K["width"] / (2 * K["fx"])))
    if abs(hfov - 90.0) > 1e-3 or abs(K["cx"] - K["width"] / 2) > 1e-6 or abs(K["cy"] - K["height"] / 2) > 1e-6:
        raise SystemExit(f"{args.scene}: intrinsics {K} are not the 90-degree centred pinhole HOV-SG assumes")
    rgb = sorted((src / "rgb").glob("*.png"))
    depth = sorted((src / "depth").glob("*.png"))
    poses = np.loadtxt(src / "pose.txt")
    if not (len(rgb) == len(depth) == len(poses)):
        raise SystemExit(f"{args.scene}: {len(rgb)} rgb, {len(depth)} depth, {len(poses)} poses")

    dst = args.out / args.split / args.scene
    for sub in ("rgb", "depth", "pose"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    for i, (r, d, row) in enumerate(zip(rgb, depth, poses)):
        if r.name != d.name:
            raise SystemExit(f"rgb {r.name} and depth {d.name} out of step")
        stem = f"{i:06d}"  # HOV-SG pairs by sorted name; a fixed-width index keeps them in step
        for sub, f in (("rgb", r), ("depth", d)):
            link = dst / sub / f"{stem}.png"
            if not link.exists():
                link.symlink_to(f)
        T = WORLD_SYS_TO_HAB @ row[1:].reshape(4, 4) @ CV_TO_GL
        (dst / "pose" / f"{stem}.txt").write_text(" ".join(f"{v:.9g}" for v in T.reshape(-1)) + "\n")
    print(f"{args.scene}: {len(rgb)} frames -> {dst}")


if __name__ == "__main__":
    main()
