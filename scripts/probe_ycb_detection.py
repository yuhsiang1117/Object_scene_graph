#!/usr/bin/env python3
"""What the detector actually does with the authored YCB targets, measured.

Three questions an episode log cannot answer, each its own mode:

`views` (default) -- at every target's best authored viewpoints, with the
    benchmark's own vocabulary, what does the detector score for the target's
    own label at each `imgsz`? This is the honest "what the agent sees".

`labels` -- for a target the benchmark's label misses, swap ONE candidate name
    into the vocabulary in its place and re-score. Swapping rather than adding
    matters: an open-vocabulary head runs class-competitive NMS, so adding
    "tin can" beside "tomato soup can" takes the detection away from the target
    label rather than helping it, and a probe that adds every synonym at once
    measures the competition instead of the name.

`fp` -- a false-positive census. Random navigable poses across the scene, the
    benchmark's vocabulary, and every detection of a YCB label that does NOT
    overlap that YCB object. This is what fills a map with fifteen "cracker box"
    tracks for a house containing one.

    python scripts/probe_ycb_detection.py +experiment=ycb_authored_nav \\
        ycb.layout_root=outputs/collector_layouts \\
        +probe.mode=views +probe.imgsz=[512,1280] +probe.out=outputs/probe

Ground truth comes from the semantic sensor, with the injected ids offset out of
the scene's range -- see SEMANTIC_ID_OFFSET.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import hydra
import numpy as np
from omegaconf import DictConfig

from osg.core.config import register_configs

register_configs()

# The collector's semantic ids are 26..95 and HM3D annotates this scene's 252
# instances as 0..251, so `semantic == authored.semantic_id` also selects a wall
# or a picture. The benchmark never notices: its manifest simulator carries a
# semantic sensor ALONE and habitat then renders the whole scene as one blob. A
# probe needs RGB, and with a colour sensor attached the scene ids appear and the
# collision is real -- a "plate" mask covering 124506 px of wall and television.
# Offsetting the injected ids restores what the original dualmap authoring did
# with 50001+, without touching the layouts.
SEMANTIC_ID_OFFSET = 50000

# Candidate names for `labels` mode: what else this thing might be called.
PROBE_LABELS: Dict[str, List[str]] = {
    "003_cracker_box": ["cracker box", "cereal box", "cracker carton", "food box",
                        "red cracker box"],
    "005_tomato_soup_can": ["tomato soup can", "soup can", "tin can", "canned food",
                            "red and white can"],
    "019_pitcher_base": ["pitcher", "blue plastic pitcher", "blue pitcher", "jug",
                         "water jug", "water pitcher", "plastic jug", "vase",
                         "kettle", "watering can", "bucket", "carafe"],
    "024_bowl": ["bowl", "red bowl", "dish"],
    "025_mug": ["mug", "coffee mug", "cup"],
    "029_plate": ["plate", "red plate", "dish", "saucer"],
    "037_scissors": ["scissors", "pair of scissors", "shears", "orange scissors",
                     "yellow handled scissors", "kitchen shears", "cutting tool",
                     "craft scissors"],
}


# --------------------------------------------------------------- simulator

def _sim_with_rgb(layout, cfg):
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
                       ("semantic", habitat_sim.SensorType.SEMANTIC)):
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


def _observe(sim, position: Sequence[float], rotation: Sequence[float]):
    from habitat_sim.utils.common import quat_from_coeffs

    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = quat_from_coeffs(np.asarray(rotation, dtype=np.float32))
    state.sensor_states = {}
    agent.set_state(state, reset_sensors=True)
    obs = sim.get_sensor_observations()
    return np.asarray(obs["rgb"])[..., :3], np.asarray(obs["semantic"])


# ------------------------------------------------------------------ helpers

def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.count_nonzero(mask_a | mask_b)
    return 0.0 if union == 0 else float(np.count_nonzero(mask_a & mask_b)) / union


def _bbox_of(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _bbox_px(det) -> float:
    x1, y1, x2, y2 = det.bbox_xyxy
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def _save_frame(path: Path, rgb, gt_mask, dets, title: str) -> None:
    import cv2

    canvas = np.ascontiguousarray(rgb[..., ::-1].copy())
    box = _bbox_of(gt_mask) if gt_mask is not None else None
    if box is not None:
        cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    for det in dets:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 128, 255), 1)
        cv2.putText(canvas, f"{det.label} {det.score:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 255), 1)
    cv2.putText(canvas, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def _crop(path: Path, rgb, gt_mask, pad: int = 24) -> None:
    import cv2

    box = _bbox_of(gt_mask)
    if box is None:
        return
    h, w = rgb.shape[:2]
    patch = rgb[max(0, box[1] - pad):min(h, box[3] + pad),
                max(0, box[0] - pad):min(w, box[2] + pad)][..., ::-1]
    if patch.size == 0:
        return
    scale = max(1, int(240 / max(1, max(patch.shape[:2]))))
    if scale > 1:
        patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.ascontiguousarray(patch))


def _authored_views(sim, cfg, authored, n: int) -> List[Dict[str, Any]]:
    """The manifest's own viewpoint rule: navigable rings, ranked by visible pixels."""
    from osg.sim.ycb_env import _yaw_facing, _finite_point

    pathfinder = sim.pathfinder
    target = np.asarray(authored.translation, dtype=np.float32)
    samples = int(cfg.ycb.viewpoint_angular_samples)
    found: List[Dict[str, Any]] = []
    for radius in [float(v) for v in cfg.ycb.viewpoint_radii_m]:
        for k in range(samples):
            angle = 2.0 * math.pi * k / samples
            requested = np.array([target[0] + radius * math.cos(angle), target[1],
                                  target[2] + radius * math.sin(angle)], dtype=np.float32)
            snapped = _finite_point(pathfinder.snap_point(requested))
            if snapped is None:
                continue
            if float(np.linalg.norm(snapped[[0, 2]] - requested[[0, 2]])) > float(
                cfg.ycb.viewpoint_max_snap_m
            ):
                continue
            if any(float(np.linalg.norm(snapped - np.asarray(item["position"])))
                   < float(cfg.ycb.viewpoint_dedup_m) for item in found):
                continue
            rotation = _yaw_facing(snapped, target)
            _, semantic = _observe(sim, snapped, rotation)
            visible = int(np.count_nonzero(
                semantic == SEMANTIC_ID_OFFSET + int(authored.semantic_id)))
            if visible < int(cfg.ycb.viewpoint_min_visible_pixels):
                continue
            found.append({"position": [float(x) for x in snapped], "rotation": rotation,
                          "visible_pixels": visible, "requested_radius_m": radius})
    found.sort(key=lambda item: (-item["visible_pixels"], item["requested_radius_m"]))
    return found[:n]


# -------------------------------------------------------------------- modes

def _mode_views(sim, cfg, layout, detector, probe, out_dir, handles):
    """Benchmark vocabulary, unchanged. Per target, per imgsz, per viewpoint."""
    imgszs = [int(v) for v in probe.get("imgsz", [512, 1280])]
    n_views = int(probe.get("views", 3))
    detector.set_vocabulary([str(v) for v in cfg.detector.vocabulary])
    records = []
    for authored in layout.objects:
        if handles and authored.handle not in handles:
            continue
        for rank, view in enumerate(_authored_views(sim, cfg, authored, n_views)):
            rgb, semantic = _observe(sim, view["position"], view["rotation"])
            gt_mask = semantic == SEMANTIC_ID_OFFSET + int(authored.semantic_id)
            stem = f"{authored.handle}_v{rank}"
            _crop(out_dir / "crops" / f"{stem}.png", rgb, gt_mask)
            for imgsz in imgszs:
                detector.imgsz = imgsz
                dets = detector.detect(rgb)
                hits = sorted(({"label": d.label, "score": round(float(d.score), 3),
                                "iou": round(_iou(d.mask, gt_mask), 3),
                                "bbox_px": round(_bbox_px(d), 1)}
                               for d in dets if _iou(d.mask, gt_mask) > 0.1),
                              key=lambda h: -h["score"])
                records.append({
                    "mode": "views", "handle": authored.handle, "label": authored.label,
                    "view_rank": rank, "radius_m": view["requested_radius_m"],
                    "gt_pixels": int(np.count_nonzero(gt_mask)),
                    "gt_bbox": _bbox_of(gt_mask), "imgsz": imgsz, "overlapping": hits,
                    "target_score": max([h["score"] for h in hits
                                         if h["label"] == authored.label], default=0.0),
                    "best_label": hits[0]["label"] if hits else None,
                    "best_score": hits[0]["score"] if hits else 0.0,
                })
                if imgsz == imgszs[-1]:
                    _save_frame(out_dir / "frames" / f"{stem}_{imgsz}.png", rgb, gt_mask,
                                dets, f"{authored.handle} imgsz={imgsz}")
                print(json.dumps(records[-1]))
    return records


def _mode_labels(sim, cfg, layout, detector, probe, out_dir, handles):
    """One candidate name at a time, SWAPPED IN for the target's own label."""
    imgszs = [int(v) for v in probe.get("imgsz", [512, 1280])]
    n_views = int(probe.get("views", 3))
    base = [str(v) for v in cfg.detector.vocabulary]
    records = []
    for authored in layout.objects:
        if handles and authored.handle not in handles:
            continue
        views = _authored_views(sim, cfg, authored, n_views)
        frames = [(_observe(sim, v["position"], v["rotation"]), v) for v in views]
        for candidate in PROBE_LABELS.get(authored.handle, [authored.label]):
            vocab = [c for c in base if c != authored.label] + [candidate]
            detector.set_vocabulary(vocab)
            for rank, ((rgb, semantic), view) in enumerate(frames):
                gt_mask = semantic == SEMANTIC_ID_OFFSET + int(authored.semantic_id)
                for imgsz in imgszs:
                    detector.imgsz = imgsz
                    dets = detector.detect(rgb)
                    hits = sorted(({"label": d.label, "score": round(float(d.score), 3),
                                    "iou": round(_iou(d.mask, gt_mask), 3),
                                    "bbox_px": round(_bbox_px(d), 1)}
                                   for d in dets if _iou(d.mask, gt_mask) > 0.1),
                                  key=lambda h: -h["score"])
                    records.append({
                        "mode": "labels", "handle": authored.handle,
                        "candidate": candidate, "view_rank": rank, "imgsz": imgsz,
                        "gt_pixels": int(np.count_nonzero(gt_mask)),
                        "candidate_score": max([h["score"] for h in hits
                                                if h["label"] == candidate], default=0.0),
                        "candidate_iou": max([h["iou"] for h in hits
                                              if h["label"] == candidate], default=0.0),
                        "overlapping": hits,
                    })
                    print(json.dumps(records[-1]))
    return records


def _mode_fp(sim, cfg, layout, detector, probe, out_dir, handles):
    """False-positive census over random navigable poses, benchmark vocabulary."""
    imgsz = int(probe.get("fp_imgsz", cfg.detector.imgsz))
    n = int(probe.get("fp_samples", 200))
    seed = int(probe.get("seed", 0))
    detector.set_vocabulary([str(v) for v in cfg.detector.vocabulary])
    detector.imgsz = imgsz
    ycb_labels = {a.label: SEMANTIC_ID_OFFSET + int(a.semantic_id) for a in layout.objects}
    rng = np.random.default_rng(seed)
    pathfinder = sim.pathfinder
    records = []
    for i in range(n):
        position = pathfinder.get_random_navigable_point(island_index=-1)
        if not np.all(np.isfinite(position)):
            continue
        yaw = float(rng.uniform(-math.pi, math.pi))
        rotation = [0.0, float(math.sin(yaw / 2.0)), 0.0, float(math.cos(yaw / 2.0))]
        rgb, semantic = _observe(sim, position, rotation)
        dets = detector.detect(rgb)
        for det in dets:
            if det.label not in ycb_labels:
                continue
            gt_mask = semantic == ycb_labels[det.label]
            iou = _iou(det.mask, gt_mask)
            records.append({
                "mode": "fp", "sample": i, "label": det.label,
                "score": round(float(det.score), 3), "iou": round(iou, 3),
                "bbox_px": round(_bbox_px(det), 1),
                "true_positive": bool(iou > 0.1),
                "position": [float(x) for x in position],
            })
            if not records[-1]["true_positive"] and probe.get("save_fp", True):
                _save_frame(out_dir / "fp" / f"{det.label.replace(' ', '_')}_{i}.png",
                            rgb, gt_mask, [det],
                            f"FP {det.label} {det.score:.2f}")
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{n} poses, {len(records)} YCB-label detections")
    return records


MODES = {"views": _mode_views, "labels": _mode_labels, "fp": _mode_fp}


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from osg.perception.detector import YoloeDetector
    from osg.sim.ycb_layouts import discover_authored_layouts

    probe = cfg.get("probe", {})
    mode = str(probe.get("mode", "views"))
    if mode not in MODES:
        raise SystemExit(f"probe.mode must be one of {sorted(MODES)}")
    out_dir = Path(str(probe.get("out", "outputs/probe")))
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = [str(v) for v in probe.get("handles", [])]

    discovery = discover_authored_layouts(
        layout_root=Path(str(cfg.ycb.layout_root)),
        data_root=Path(str(cfg.ycb.data_root)),
        scenes=[str(v) for v in cfg.ycb.scenes],
        layout_types=["static"],
        layout_indices=[int(v) for v in cfg.ycb.layout_indices],
        target_labels={str(k): str(v) for k, v in cfg.ycb.target_labels.items()},
    )
    if not discovery.layouts:
        raise SystemExit("no static layout discovered")
    layout = discovery.layouts[0]

    detector = YoloeDetector(
        weights=str(cfg.detector.weights), conf=float(probe.get("conf", 0.05)),
        imgsz=int(cfg.detector.imgsz), half=bool(cfg.detector.half),
        device=str(cfg.detector.device),
    )
    sim = _sim_with_rgb(layout, cfg)
    try:
        records = MODES[mode](sim, cfg, layout, detector, probe, out_dir, handles)
    finally:
        sim.close()

    path = out_dir / f"probe_{mode}.json"
    path.write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(f"\nwrote {path}  ({len(records)} records)")


if __name__ == "__main__":
    main()
