"""Run perception + object layer + mapping on a recorded trajectory (M1/M2
verification without the simulator in the loop).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from osg.core.profiler import Profiler
from osg.core.types import CameraIntrinsics, FrameData
from osg.mapping.costmap import PLANE, Costmap2D
from osg.mapping.frontier import FrontierExtractor
from osg.objects.object_layer import ObjectLayer
from osg.perception.keyframe import KeyframeSelector


def load_frames(path: str):
    z = np.load(path)
    n = int(z["n"])
    for i in range(n):
        yield FrameData(
            frame_id=int(z[f"f{i}_frame_id"]),
            rgb=z[f"f{i}_rgb"],
            depth=z[f"f{i}_depth"],
            T_wc=z[f"f{i}_T_wc"],
            intrinsics=CameraIntrinsics(
                fx=float(z[f"f{i}_fx"]), fy=float(z[f"f{i}_fy"]),
                cx=float(z[f"f{i}_cx"]), cy=float(z[f"f{i}_cy"]),
                width=z[f"f{i}_rgb"].shape[1], height=z[f"f{i}_rgb"].shape[0],
            ),
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recording", default="data/recordings/recording.npz")
    p.add_argument("--detector", choices=["yoloe", "none"], default="yoloe")
    p.add_argument("--weights", default="data/weights/yoloe-11s-seg.pt")
    args = p.parse_args()

    detector = None
    if args.detector == "yoloe":
        from osg.perception.detector import YoloeDetector
        from osg.core.config import DEFAULT_VOCABULARY

        detector = YoloeDetector(weights=args.weights, vocabulary=list(DEFAULT_VOCABULARY))

    profiler = Profiler()
    costmap = Costmap2D()
    object_layer = ObjectLayer()
    kf = KeyframeSelector()
    extractor = FrontierExtractor()
    floor_y = None

    n_frames = 0
    for frame in load_frames(args.recording):
        n_frames += 1
        if floor_y is None:
            floor_y = float(frame.camera_position[1] - 0.88)
        with profiler.timeit("costmap"):
            costmap.update(frame, floor_y=floor_y)
        if kf.is_keyframe(frame.T_wc) and detector is not None:
            with profiler.timeit("detector"):
                dets = detector.detect(frame.rgb)
            with profiler.timeit("object_layer"):
                object_layer.update(frame, dets)

    frontiers = extractor.extract(costmap)
    tracks = object_layer.tracks()
    print(f"frames: {n_frames}, coverage cells: {costmap.coverage_cells()}, frontiers: {len(frontiers)}")
    print(f"object tracks: {len(tracks)}")
    for t in sorted(tracks, key=lambda t: -t.n_obs)[:15]:
        c = object_layer.center_of(t)
        print(f"  #{t.id:3d} {t.label:20s} obs={t.n_obs:3d} center=({c[0]:6.2f},{c[1]:6.2f},{c[2]:6.2f})"
              f" linked={sorted(t.linked_ids)}")

    from osg.eval.visualize import save_topdown

    out = Path("outputs/offline")
    save_topdown(str(out / "costmap.png"), costmap, [], title=f"{len(frontiers)} frontiers")
    print(f"viz -> {out / 'costmap.png'}")
    for name, r in profiler.report().items():
        print(f"  {name:14s} mean {r['mean_ms']:7.1f} ms  (n={r['count']})")


if __name__ == "__main__":
    main()
