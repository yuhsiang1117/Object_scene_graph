#!/usr/bin/env python3
"""A labelled set of the detections the agent would act on: the target, and
everything else the detector calls the target.

Why this exists. On the YCB benchmarks most goal commits are wrong -- 31 of 40
on the multi-floor set (outputs/RERUN/mf5), 49 of 73 live commits on DualMap's
static condition -- and every one of them is a detection the agent drove to
and, usually, stopped at. Whether a second opinion (a VLM, an image-text
margin) can tell those apart from the real object is an empirical question,
and the earlier answer ("verification is a dead end", docs/history/
INVESTIGATION.md) was measured on HM3D ObjectNav, where the false positives were
real objects of the right category. Here they are not: a ceiling fixture called
"toy airplane", a book called "cracker box". This builds the data to ask again.

Per static layout, two sources of frames:

  near    the manifest's own viewpoint rule around every authored object
          (`_authored_views`), so the target is in view at 0.8-2 m -- the range
          an arrival is judged from
  random  navigable poses across the scene, random yaw (the FP census)

and for every frame and every YCB label in the layout, the detector exactly as
pass 2 runs it for that query: YOLOE with `target_vocabulary(label, ...)` and
the arm's own thresholds. Each detection of the label is saved as a clean crop
(no overlay) and labelled TRUE when its mask overlaps that object's semantic
mask (IoU > 0.1), with its range from the depth sensor.

    python scripts/build_verifier_set.py +experiment=mfv2_osg \\
        +probe.out=outputs/verifier_set +probe.random=150

Writes `<out>/set.jsonl` (one detection per line), `<out>/crops/`, `<out>/frames/`.
No model servers; YOLOE and the simulator run in-process.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_ycb_detection as probe_tools  # noqa: E402
from osg.agent.stop_gate import crop_with_margin  # noqa: E402

SEMANTIC_ID_OFFSET = probe_tools.SEMANTIC_ID_OFFSET


def _sim_with_depth(layout, cfg):
    """`probe_ycb_detection._sim_with_rgb` plus a depth sensor at the same pose."""
    import habitat_sim
    import magnum as mn

    from osg.sim.ycb_env import inject_layout_objects

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(layout.scene_mesh)
    sim_cfg.scene_dataset_config_file = str(layout.scene_dataset_config)
    sim_cfg.gpu_device_id = 0
    sim_cfg.enable_physics = True
    specs = []
    for uuid, kind in (("rgb", habitat_sim.SensorType.COLOR),
                       ("semantic", habitat_sim.SensorType.SEMANTIC),
                       ("depth", habitat_sim.SensorType.DEPTH)):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = kind
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = [int(cfg.eval.rgb_height), int(cfg.eval.rgb_width)]
        spec.position = [0.0, float(cfg.agent.camera_height), 0.0]
        spec.hfov = mn.Deg(float(cfg.eval.hfov_deg))
        specs.append(spec)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = float(cfg.agent.camera_height)
    agent_cfg.radius = float(cfg.agent.agent_radius)
    agent_cfg.sensor_specifications = specs
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    sim.initialize_agent(0)
    for rigid in inject_layout_objects(sim, layout):
        try:
            rigid.semantic_id = SEMANTIC_ID_OFFSET + int(rigid.semantic_id)
        except (AttributeError, TypeError, ValueError):
            pass
    return sim


def _observe(sim, position, rotation):
    from habitat_sim.utils.common import quat_from_coeffs

    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = quat_from_coeffs(np.asarray(rotation, dtype=np.float32))
    state.sensor_states = {}
    agent.set_state(state, reset_sensors=True)
    obs = sim.get_sensor_observations()
    return (np.asarray(obs["rgb"])[..., :3], np.asarray(obs["semantic"]),
            np.asarray(obs["depth"], dtype=np.float32))


def _save_png(path: Path, rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.ascontiguousarray(rgb[..., ::-1]))


def _range_m(depth: np.ndarray, mask: np.ndarray) -> float | None:
    values = depth[mask & np.isfinite(depth) & (depth > 0)]
    return None if values.size == 0 else float(np.median(values))


def _random_poses(sim, n: int, seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    sim.pathfinder.seed(seed)
    poses = []
    for _ in range(n):
        position = sim.pathfinder.get_random_navigable_point()
        if not np.all(np.isfinite(position)):
            continue
        yaw = float(rng.uniform(-math.pi, math.pi))
        poses.append({"position": [float(x) for x in position],
                      "rotation": [0.0, float(math.sin(yaw / 2)), 0.0, float(math.cos(yaw / 2))]})
    return poses


def detections_for_layout(sim, cfg, layout, detector, out_dir: Path, n_random: int,
                          n_near: int, seed: int) -> List[Dict[str, Any]]:
    from osg.perception.vocabulary import target_vocabulary

    labels = {a.label: SEMANTIC_ID_OFFSET + int(a.semantic_id) for a in layout.objects}
    frames = [("random", p, None) for p in _random_poses(sim, n_random, seed)]
    for authored in layout.objects:
        for view in probe_tools._authored_views(sim, cfg, authored, n_near):
            frames.append(("near", view, authored.label))
    records: List[Dict[str, Any]] = []
    for index, (source, pose, near_label) in enumerate(frames):
        rgb, semantic, depth = _observe(sim, pose["position"], pose["rotation"])
        frame_path = None
        for label, semantic_id in labels.items():
            detector.set_vocabulary(target_vocabulary(label, cfg.detector.vocabulary))
            gt_mask = semantic == semantic_id
            for det_i, det in enumerate(d for d in detector.detect(rgb) if d.label == label):
                if frame_path is None:
                    frame_path = out_dir / "frames" / f"{layout.scene_name}_{index:04d}.png"
                    _save_png(frame_path, rgb)
                stem = (f"{layout.scene_name}_{index:04d}_{label.replace(' ', '_')}_{det_i}")
                crop_path = out_dir / "crops" / f"{stem}.png"
                _save_png(crop_path, crop_with_margin(rgb, det.bbox_xyxy))  # the gate's crop
                iou = probe_tools._iou(det.mask, gt_mask)
                # A false positive ON another benchmark object (the soup can
                # called "coffee can") is a different problem from one on the
                # furniture (a lamp called "toy airplane"): only the second is
                # separable by anything that knows what a lamp looks like.
                other = max((probe_tools._iou(det.mask, semantic == sid)
                             for other_label, sid in labels.items() if other_label != label),
                            default=0.0)
                records.append({
                    "scene": layout.scene_name, "layout_id": layout.layout_id,
                    "source": source, "near_label": near_label, "frame_index": index,
                    "label": label, "score": round(float(det.score), 4),
                    "bbox_xyxy": [round(float(v), 1) for v in det.bbox_xyxy],
                    "bbox_px": round(probe_tools._bbox_px(det), 1),
                    "iou": round(iou, 3), "true": bool(iou > 0.1),
                    "other_ycb_iou": round(float(other), 3),
                    "range_m": _range_m(depth, det.mask),
                    "target_visible_px": int(np.count_nonzero(gt_mask)),
                    "crop": str(crop_path.relative_to(out_dir)),
                    "frame": str(frame_path.relative_to(out_dir)),
                    "position": pose["position"], "rotation": pose["rotation"],
                })
        if (index + 1) % 50 == 0:
            n_true = sum(r["true"] for r in records)
            print(f"  {index + 1}/{len(frames)} frames: {len(records)} detections, "
                  f"{n_true} true", flush=True)
    return records


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from osg.perception.detector import YoloeDetector
    from osg.sim.ycb_layouts import discover_authored_layouts

    probe = cfg.get("probe", {})
    out_dir = Path(str(probe.get("out", "outputs/verifier_set")))
    out_dir.mkdir(parents=True, exist_ok=True)
    discovery = discover_authored_layouts(
        layout_root=Path(str(cfg.ycb.layout_root)),
        data_root=Path(str(cfg.ycb.data_root)),
        scenes=[str(v) for v in cfg.ycb.scenes],
        layout_types=["static"],
        layout_indices=[int(v) for v in cfg.ycb.layout_indices],
        target_labels={str(k): str(v) for k, v in cfg.ycb.target_labels.items()},
        hm3d_root=Path(str(cfg.ycb.hm3d_root)),
    )
    if not discovery.layouts:
        raise SystemExit("no static layout discovered")
    # The arm's own detector settings, so the set holds what pass 2 acts on.
    detector = YoloeDetector(
        weights=str(cfg.detector.weights), conf=float(cfg.detector.conf),
        class_conf=dict(cfg.detector.class_conf or {}), imgsz=int(cfg.detector.imgsz),
        half=bool(cfg.detector.half), device=str(cfg.detector.device),
    )
    out_path = out_dir / "set.jsonl"
    n = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for layout in discovery.layouts:
            print(f"\n===== {layout.scene_name}/{layout.layout_id} =====", flush=True)
            sim = _sim_with_depth(layout, cfg)
            try:
                for rec in detections_for_layout(
                        sim, cfg, layout, detector, out_dir,
                        n_random=int(probe.get("random", 150)),
                        n_near=int(probe.get("near", 6)),
                        seed=int(probe.get("seed", 0))):
                    handle.write(json.dumps(rec) + "\n")
                    n += 1
            finally:
                sim.close()
    print(f"\nwrote {out_path} ({n} detections)")


if __name__ == "__main__":
    main()
