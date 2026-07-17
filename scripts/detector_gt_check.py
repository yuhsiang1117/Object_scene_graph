"""Compare YOLOE's detections against HM3D's ground-truth semantic annotations
to quantify how much of the "hierarchical scene graph is noisy" suspicion
traces back to the detector itself, as opposed to downstream position
estimation.

For each sampled step of a real NavAgent run, this looks up (a) whether the
episode's target category is actually visible in the frame per HM3D's
per-pixel semantic sensor (`.semantic.glb`/`.semantic.txt` annotations,
HM3D-semantics v0.2) and, if so, which pixels belong to a genuine goal
instance (`episode.goals[i].object_id`, confirmed to equal the semantic
sensor's per-pixel id), vs (b) what YOLOE actually reported for that frame.
Classifies each frame into TP / FN / FP-wrong-instance / FP-hallucination
and reports, for true positives, mask IoU and centroid pixel offset (the
offset is what would feed noise into the ellipsoid/3D position estimate
downstream).

Usage:
  python scripts/detector_gt_check.py --categories sofa,bed,plant \
      --episodes-per-category 2 --max-steps 150 --sample-every 3
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from hydra import compose, initialize_config_dir

from osg.core.config import register_configs

register_configs()
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
    cfg = compose(config_name="config", overrides=["eval=hm3d_val_mini"])

from osg.agent.nav_agent import NavAgent  # noqa: E402
from osg.core.types import CameraIntrinsics  # noqa: E402
from osg.eval.runner import _unload_ollama_models, build_detector, build_scorer, build_verifier  # noqa: E402
from osg.mapping.costmap import PLANE  # noqa: E402
from osg.sim.habitat_env import HabitatObjectNavEnv, make_objectnav_config  # noqa: E402

ANNOTATED_SCENE_DATASET = "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"


def _normalize(label: str) -> str:
    return label.lower().replace("_", " ").strip()


class SemanticHabitatEnv(HabitatObjectNavEnv):
    """HabitatObjectNavEnv + a semantic sensor and the annotated HM3D scene
    dataset, so ground-truth per-pixel category ids are available alongside
    the same rgb/depth frames the agent already sees. Kept out of the main
    eval config -- the extra sensor adds render cost this diagnostic accepts
    but the main pipeline shouldn't pay on every run."""

    def __init__(self, cfg) -> None:
        import habitat
        from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig
        from habitat.config.read_write import read_write

        self._hab_cfg = make_objectnav_config(cfg)
        with read_write(self._hab_cfg):
            sim = self._hab_cfg.habitat.simulator
            agent = sim.agents.main_agent
            agent.sim_sensors.update({"semantic_sensor": HabitatSimSemanticSensorConfig()})
            agent.sim_sensors.semantic_sensor.width = cfg.eval.rgb_width
            agent.sim_sensors.semantic_sensor.height = cfg.eval.rgb_height
            agent.sim_sensors.semantic_sensor.hfov = int(cfg.eval.hfov_deg)
            sim.scene_dataset = ANNOTATED_SCENE_DATASET
        self.env = habitat.Env(config=self._hab_cfg)
        self.intrinsics = CameraIntrinsics.from_hfov(
            cfg.eval.hfov_deg, cfg.eval.rgb_width, cfg.eval.rgb_height
        )
        self._frame_id = 0
        self.last_obs = None

    def reset(self):
        obs = self.env.reset()
        self._frame_id = 0
        self.last_obs = obs
        return self._to_frame(obs)

    def step(self, action):
        obs = self.env.step(self.ACTIONS[action])
        self._frame_id += 1
        self.last_obs = obs
        return self._to_frame(obs)


def _majority_category(mask: np.ndarray, sem_ids: np.ndarray, by_semid: dict) -> str:
    ids = sem_ids[mask]
    if ids.size == 0:
        return "empty"
    vals, counts = np.unique(ids, return_counts=True)
    top = int(vals[np.argmax(counts)])
    obj = by_semid.get(top)
    return obj.category.name() if obj is not None else f"id{top}"


def classify_frame(
    sem_ids: np.ndarray, goal_ids: set, cat_ids: set, dets, target: str, iou_hit=0.1,
    by_semid: dict | None = None, dump_dir: Optional[Path] = None, dump_prefix: str = "",
):
    """Returns (verdict, iou_or_None, centroid_offset_px_or_None). If dump_dir
    is given, saves the crop + a majority-ground-truth-category label for any
    FP-hallucination detection (what the detector actually saw instead)."""
    gt_goal_mask = np.isin(sem_ids, list(goal_ids)) if goal_ids else np.zeros_like(sem_ids, dtype=bool)
    gt_cat_mask = np.isin(sem_ids, list(cat_ids)) if cat_ids else np.zeros_like(sem_ids, dtype=bool)
    matches = [d for d in dets if _normalize(d.label) == _normalize(target)]

    if not matches:
        return ("FN" if gt_goal_mask.any() else "TN"), None, None

    best_iou, best_mask, best_det = -1.0, None, None
    for d in matches:
        inter = (d.mask & gt_goal_mask).sum()
        union = (d.mask | gt_goal_mask).sum()
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou, best_mask, best_det = iou, d.mask, d

    if best_iou >= iou_hit:
        gt_ys, gt_xs = np.nonzero(gt_goal_mask)
        det_ys, det_xs = np.nonzero(best_mask)
        offset = float(np.hypot(gt_xs.mean() - det_xs.mean(), gt_ys.mean() - det_ys.mean()))
        return "TP", best_iou, offset
    if any((d.mask & gt_cat_mask).any() for d in matches):
        return "FP-wrong-instance", best_iou, None

    if dump_dir is not None and best_det is not None and best_det.crop is not None and best_det.crop.size > 0:
        actual = _majority_category(best_det.mask, sem_ids, by_semid or {})
        fname = dump_dir / f"{dump_prefix}_actual-{actual.replace(' ', '-')}_score{best_det.score:.2f}.png"
        import cv2

        cv2.imwrite(str(fname), best_det.crop[..., ::-1])  # rgb -> bgr for cv2
    return "FP-hallucination", best_iou, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="sofa,bed,plant")
    ap.add_argument("--episodes-per-category", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--sample-every", type=int, default=3)
    ap.add_argument("--dump-hallucinations", default=None,
                     help="directory to save FP-hallucination crops (labeled with the actual GT category) to")
    args = ap.parse_args()
    wanted_cats = [c.strip() for c in args.categories.split(",")]
    dump_dir = None
    if args.dump_hallucinations:
        dump_dir = Path(args.dump_hallucinations)
        dump_dir.mkdir(parents=True, exist_ok=True)

    _unload_ollama_models(cfg)
    env = SemanticHabitatEnv(cfg)
    detector = build_detector(cfg)
    scorer = build_scorer(cfg)
    verifier = build_verifier(cfg)

    per_cat_counts = defaultdict(int)
    stats = defaultdict(lambda: defaultdict(int))
    ious = defaultdict(list)
    offsets = defaultdict(list)

    n_total = len(env.env.episodes)
    for ep_i in range(n_total):
        frame = env.reset()
        episode = env.current_episode
        target = env.target_category()
        if target not in wanted_cats or per_cat_counts[target] >= args.episodes_per_category:
            continue
        per_cat_counts[target] += 1

        goal_ids = {g.object_id for g in episode.goals}
        sem_scene = env.env.sim.semantic_scene
        by_semid = {o.semantic_id: o for o in sem_scene.objects}
        # HM3D's raw semantic category strings are synonyms of the ObjectNav
        # category, not the literal name (e.g. target "plant" episodes use
        # goal instances labeled "flowerpot"/"flower vase"/"decorative
        # plant") -- derive the broader same-category mask from whatever
        # raw strings the goal instances themselves actually use, instead
        # of assuming episode.object_category matches sem_scene category
        # names directly (true for "chair", false for "plant").
        raw_cat_names = {by_semid[gid].category.name() for gid in goal_ids if gid in by_semid}
        cat_ids = {o.semantic_id for o in sem_scene.objects if o.category.name() in raw_cat_names}

        agent = NavAgent(cfg, detector, scorer, verifier, target, profiler=None)
        print(f"--- ep={episode.episode_id} target={target} "
              f"(goal_ids={sorted(goal_ids)}, {len(cat_ids)} instances of category in scene) ---")

        step = 0
        while not env.episode_over and step < args.max_steps:
            action = agent.act(frame)
            frame = env.step(action)
            step += 1
            if step % args.sample_every != 0:
                continue
            sem_ids = env.last_obs["semantic"][..., 0]
            dets = detector.detect(frame.rgb)
            verdict, iou, offset = classify_frame(
                sem_ids, goal_ids, cat_ids, dets, target,
                by_semid=by_semid, dump_dir=dump_dir, dump_prefix=f"ep{episode.episode_id}_step{step}",
            )
            stats[target][verdict] += 1
            if iou is not None and verdict == "TP":
                ious[target].append(iou)
            if offset is not None:
                offsets[target].append(offset)

    env.close()
    scorer.shutdown()

    print()
    print(f"{'target':11s} {'TP':>4s} {'FN':>4s} {'FP-wrong':>9s} {'FP-halluc':>10s} {'TN':>5s} "
          f"{'recall':>7s} {'mean_IoU':>9s} {'mean_off_px':>11s}")
    for target in wanted_cats:
        s = stats[target]
        tp, fn = s["TP"], s["FN"]
        fp_wrong, fp_halluc, tn = s["FP-wrong-instance"], s["FP-hallucination"], s["TN"]
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        mean_iou = float(np.mean(ious[target])) if ious[target] else float("nan")
        mean_off = float(np.mean(offsets[target])) if offsets[target] else float("nan")
        print(f"{target:11s} {tp:4d} {fn:4d} {fp_wrong:9d} {fp_halluc:10d} {tn:5d} "
              f"{recall:7.3f} {mean_iou:9.3f} {mean_off:11.1f}")


if __name__ == "__main__":
    main()
