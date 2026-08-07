"""Episode loop: builds the agent stack from config, runs HM3D ObjectNav
episodes, writes per-episode jsonl + aggregate summary + timing + trajectory
visualizations.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..agent.nav_agent import NavAgent
from ..core.profiler import Profiler
from ..exploration.async_scorer import AsyncScorer
from ..exploration.llm_scorer import LLMTextScorer
from ..llm.client import ChatClient
from ..mapping.costmap import HEIGHT_AXIS, PLANE
from .floors import episode_floor_fields
from .metrics import aggregate, per_category, per_floor_class
from .visualize import overlay_segmentation, render_costmap_bgr, save_topdown


class _DebugVideo:
    """Per-episode debug video: each frame is [live RGB + YOLOE segmentation
    overlay | top-down costmap] at every step. The detector is re-run here for
    visualization only (it does not feed the object layer), so pipeline
    behaviour / SR is unchanged. Enabled by eval.debug_frames."""

    def __init__(self, cfg, out_dir: Path, tag: str) -> None:
        import cv2

        from ..mapping.costmap import PLANE as _PLANE

        self._cv2 = cv2
        self._plane = list(_PLANE)
        self._cm_w = 480
        self._h = cfg.eval.rgb_height
        self._w = cfg.eval.rgb_width + self._cm_w
        self._traj: list = []
        path = out_dir / "viz" / "debug" / f"{tag}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (self._w, self._h))

    def write(self, frame, agent, target: str, detector) -> None:
        cv2 = self._cv2
        agent_xy = frame.camera_position[self._plane]
        self._traj.append(agent_xy)
        dets = detector.detect(frame.rgb)  # viz-only; does not update object layer
        seg = overlay_segmentation(frame.rgb, dets, target)
        seg = cv2.resize(seg, (self._w - self._cm_w, self._h))
        cm = render_costmap_bgr(
            agent.costmap, agent_xy, self._traj,
            path_xy=getattr(agent, "_current_path", None),
            chosen_frontier=getattr(agent, "_current_frontier", None),
            out_h=self._h,
        )
        cm = cv2.resize(cm, (self._cm_w, self._h))
        panel = cv2.hconcat([seg, cm])
        cv2.putText(panel, f"{target}  step {len(self._traj)}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        self._vw.write(panel)

    def close(self) -> None:
        self._vw.release()


def build_scorer(cfg) -> AsyncScorer:
    # Geometric-only exploration (no LLM): nearest frontier weighted by
    # exploration range (info gain). select_frontier falls back to
    # unscored_prior for every frontier.
    if getattr(cfg.exploration, "scorer", "llm_text") in ("nearest", "geometric", "none"):
        from ..exploration.scorer import NullScorer
        return AsyncScorer(NullScorer())
    # Old-algorithm pipeline: text-LLM frontier ranking over the scene-graph
    # subgraphs (ObjectSceneGraph_old frontiers_ranking).
    client = ChatClient(
        cfg.llm.base_url, cfg.llm.text_model, cfg.llm.api_key,
        cfg.llm.timeout_s, cfg.llm.max_image_px, cfg.llm.send_response_format,
    )
    inner = LLMTextScorer(client, cfg.exploration.subgraph_radius_m,
                          cfg.exploration.max_frontiers_per_call)
    return AsyncScorer(inner)


def build_verifier(cfg):
    """VLM candidate verifier, or None when verification is disabled (the
    old-algorithm terminal: viewpoint pre-position + bbox/depth stop, no VLM).

    The verifier reuses the configured LLM endpoint/key (already NVIDIA NIM in
    the matched setup) and only swaps in the vision model named by
    verification.vlm_model -- the text scorer and the vision verifier share one
    NIM account, differing only by model."""
    if not cfg.verification.enabled:
        return None
    from ..verification.verifier import VLMVerifier

    client = ChatClient(
        cfg.llm.base_url, cfg.verification.vlm_model, cfg.llm.api_key,
        cfg.llm.timeout_s, cfg.llm.max_image_px, cfg.llm.send_response_format,
    )
    return VLMVerifier(
        client,
        accept_confidence=cfg.verification.accept_confidence,
        choice_mode=getattr(cfg.verification, "choice_mode", True),
    )


def build_detector(cfg):
    if cfg.detector.name == "yoloe":
        from ..perception.detector import YoloeDetector

        return YoloeDetector(
            weights=cfg.detector.weights,
            conf=cfg.detector.conf,
            imgsz=cfg.detector.imgsz,
            half=cfg.detector.half,
            device=cfg.detector.device,
        )
    if cfg.detector.name == "stub":
        from ..perception.detector import StubDetector

        return StubDetector()
    raise ValueError(f"unknown detector: {cfg.detector.name}")


def _unload_ollama_models(cfg) -> None:
    """Ask ollama to release VRAM (keep_alive=0) so the one-time YOLOE text
    encoding can run on the GPU; ollama reloads lazily on the next call.

    Must cover every model ollama might be holding: exploration scoring
    (cfg.llm.*) AND the separate, larger verification model
    (cfg.verification.vlm_model) — omitting the latter left a 7B model
    resident from a prior run/benchmark and starved YOLOE's fp32 load of
    VRAM (CUDA OOM observed here on a 6 GB card)."""
    import json as _json
    import urllib.request

    host = str(cfg.llm.base_url).rsplit("/v1", 1)[0]
    models = {cfg.llm.text_model, cfg.llm.vlm_model}
    if cfg.verification.enabled:
        models.add(cfg.verification.vlm_model)
    for model in models:
        try:
            req = urllib.request.Request(
                host + "/api/generate",
                data=_json.dumps({"model": model, "keep_alive": 0}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10).read()
        except Exception:
            pass  # best-effort; ollama may be down in no-LLM ablations


def _target_track_fields(agent) -> dict:
    """Snapshot the committed target track for GT-localization analysis.

    _target_obj_xy is set (in _start_approach) only once a candidate is
    accepted into APPROACH, so it is None for episodes that never committed to
    a target (pure exploration failures) -- recorded as None there."""
    obj_xy = getattr(agent, "_target_obj_xy", None)
    cand_id = getattr(agent, "_candidate_id", None)
    track = agent.object_layer.get(cand_id) if cand_id is not None else None
    best_cam = getattr(track, "best_cam_xy", None) if track is not None else None
    return {
        "target_obj_xy": [float(x) for x in obj_xy] if obj_xy is not None else None,
        "cand_best_cam_xy": [float(x) for x in best_cam] if best_cam is not None else None,
        "cand_best_score": float(track.best_score) if track is not None else None,
        "cand_n_obs": int(track.n_obs) if track is not None else None,
    }


def episode_tag(episode) -> str:
    """Filesystem-safe, UNIQUE per-episode tag: `<scene>_ep<id>`.

    `episode_id` alone is not unique -- HM3D numbers episodes per scene, so a
    run spanning scenes collides. Measured on a 100-episode v1 run: 50 episodes
    yielded only 40 distinct ids, silently overwriting 10 debug videos, top-down
    maps and keyframe directories (last scene wins). Every per-episode artifact
    path must include the scene.
    """
    scene = str(getattr(episode, "scene_id", "")).split("/")[-1].split(".")[0]
    return f"{scene}_ep{episode.episode_id}" if scene else f"ep{episode.episode_id}"


STAIR_LABELS = ("stairs", "staircase", "stair")


def _stair_track_fields(agent) -> dict:
    """Snapshot the mapped `stairs` tracks.

    `"stairs"` is already in DEFAULT_VOCABULARY (core/config.py), so YOLOE has
    always been detecting staircases into the object layer -- but nothing has
    ever consumed or measured them. Stage 4 of the multi-floor plan
    (docs/MULTI_FLOOR.md) assumes these tracks exist and are usable; this logs
    the evidence to confirm or falsify that BEFORE the stair-detection work is
    built on top of it. Cheap: a filter over tracks the layer already holds.
    """
    tracks = [
        t for t in agent.object_layer.tracks(include_blacklisted=True)
        if str(t.label).lower().replace("_", " ") in STAIR_LABELS
    ]
    return {
        "n_stair_tracks": len(tracks),
        "stair_tracks": [
            {
                "center": [round(float(x), 3) for x in agent.object_layer.center_of(t)],
                "n_obs": int(t.n_obs),
                "evidence": round(float(t.evidence), 3),
                "best_score": round(float(t.best_score), 3),
                "best_bbox_px": round(float(t.best_bbox_px), 1),
            }
            for t in tracks[:20]
        ],
    }


def run_eval(cfg) -> dict:
    from ..sim.habitat_env import HabitatObjectNavEnv

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _unload_ollama_models(cfg)
    env = HabitatObjectNavEnv(cfg)
    detector = build_detector(cfg)
    scorer = build_scorer(cfg)
    verifier = build_verifier(cfg)
    if verifier is not None and cfg.eval.debug_frames:
        verifier.debug_dir = str(out_dir / "verify_debug")

    n_total = len(env.env.episodes)
    n_run = n_total if cfg.eval.num_episodes < 0 else min(cfg.eval.num_episodes, n_total)
    wanted = set(cfg.eval.episode_ids) if cfg.eval.episode_ids else None

    results = []
    episodes_file = out_dir / "episodes.jsonl"
    profiler_all = Profiler()

    for ep_i in range(n_run):
        frame = env.reset()
        episode = env.current_episode
        if wanted is not None and str(episode.episode_id) not in wanted:
            continue
        target = env.target_category()
        ep_tag = episode_tag(episode)
        if verifier is not None:
            verifier.debug_tag = ep_tag

        # scorer/verifier are built once and shared across every episode in
        # this run, so their call/error counters are cumulative — snapshot
        # before and report deltas below, or every episode after the first
        # would show the whole run's running total instead of its own.
        llm_calls_before, llm_errors_before = scorer.n_calls, scorer.n_errors
        llm_last_error_before = scorer.last_error
        verify_calls_before = verifier.n_calls if verifier is not None else 0
        verify_errors_before = verifier.n_errors if verifier is not None else 0

        # frontier.id/room.id both restart from 0/1 each episode (fresh
        # FrontierExtractor/RoomSegmenter per NavAgent below), but the
        # scorer's internal caches are keyed by those same small ints and
        # persist across the whole run -- without this, a new episode can
        # silently inherit a stale score/room-label from a previous,
        # unrelated scene the moment an id collides.
        scorer.reset()

        profiler = Profiler()
        agent = NavAgent(
            cfg, detector, scorer, verifier, target,
            keyframe_dir=str(out_dir / "keyframes" / ep_tag)
            if cfg.eval.save_viz else None,
            profiler=profiler,
            nav_fn=env.action_to_goal if cfg.agent.use_habitat_navmesh else None,
            reachable_fn=env.is_reachable if cfg.agent.use_habitat_navmesh else None,
        )
        trajectory = [frame.camera_position[list(PLANE)]]
        # Height is tracked alongside the 2D trajectory (rather than making
        # `trajectory` 3D) so the analyze_*.py tools keep working unchanged,
        # while episodes.jsonl finally records which floor the agent was on.
        # Camera height is subtracted so these are FLOOR heights, directly
        # comparable to episode.start_position and the goal view points.
        cam_h = float(cfg.agent.camera_height)
        trajectory_y = [float(frame.camera_position[HEIGHT_AXIS]) - cam_h]
        dbg = _DebugVideo(cfg, out_dir, ep_tag) if cfg.eval.debug_frames else None
        t0 = time.time()
        steps = 0
        while not env.episode_over:
            action = agent.act(frame)  # updates agent.costmap from `frame`
            if dbg is not None:
                dbg.write(frame, agent, target, detector)
            frame = env.step(action)
            trajectory.append(frame.camera_position[list(PLANE)])
            trajectory_y.append(float(frame.camera_position[HEIGHT_AXIS]) - cam_h)
            steps += 1
        if dbg is not None:
            dbg.close()

        m = env.metrics()
        rec = {
            "episode_id": str(episode.episode_id),
            "scene": str(episode.scene_id).split("/")[-1],
            "target": target,
            "success": float(m.get("success", 0.0)),
            "spl": float(m.get("spl", 0.0)),
            "distance_to_goal": float(m.get("distance_to_goal", -1.0)),
            "steps": steps,
            "wall_time_s": round(time.time() - t0, 1),
            "control_fps": round(profiler.fps("control_loop"), 2),
            "llm_calls": scorer.n_calls - llm_calls_before,
            "llm_errors": scorer.n_errors - llm_errors_before,
            "llm_last_error": scorer.last_error if scorer.last_error != llm_last_error_before else None,
            "agent_stats": agent.stats,
            "state_log": agent.state_log[:40],
            "frontier_select_log": agent.frontier_select_log,
            "giveup_log": agent.giveup_log[:50],
            "approach_bbox_log": agent.approach_bbox_log,
            "approach_stop_reason": agent.approach_stop_reason,
            "approach_diag": agent.approach_diag,
            "final_xy": [float(x) for x in trajectory[-1]],
            "verify_calls": (verifier.n_calls - verify_calls_before) if verifier is not None else 0,
            "verify_errors": (verifier.n_errors - verify_errors_before) if verifier is not None else 0,
            # GT-localization instrumentation: the mapped 3D center (x-z) of the
            # object track the agent committed to APPROACH, plus that track's
            # best-detection camera pose and score. d(target_obj_xy, GT goal)
            # is the scene-graph localization error; d(final_xy, target_obj_xy)
            # is the residual navigation error -- together they split "stopped
            # far from goal" into mislocalized-track vs failed-nav vs false-
            # positive detection (scripts/analyze_localization.py).
            **_target_track_fields(agent),
            # Floor instrumentation: which floor the goal is on relative to the
            # start pose, and whether the agent actually changed level. Without
            # this the single-floor / multi-floor SR split (the dominant
            # remaining loss, docs/MULTI_FLOOR.md) is not reproducible from
            # episodes.jsonl.
            **episode_floor_fields(episode, trajectory_y),
            **_stair_track_fields(agent),
            # Online floor estimate (osg/mapping/floors.py). Compare
            # n_floors_seen against the per-scene navmesh ground truth from
            # scripts/scene_floors.py to validate the estimator before any
            # behaviour is allowed to depend on it.
            "floor_log": agent.floor_log,
            "n_floors_seen": len(agent.floors.levels),
            "floor_y_drift": round(float(agent.floor_y_drift), 4),
            "floor_transitions": len(agent.floors.transitions),
            "portal_log": agent.portal_log,
        }
        results.append(rec)
        with open(episodes_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
        for name, samples in profiler._samples.items():
            for s in samples:
                profiler_all.add(name, s)

        if cfg.eval.save_viz:
            save_topdown(
                str(out_dir / "viz" / f"{ep_tag}.png"),
                agent.costmap,
                trajectory,
                scene_graph=agent.scene_graph,
                title=f"ep {episode.episode_id} target={target} "
                f"success={rec['success']:.0f} spl={rec['spl']:.2f}",
            )
        print(
            f"[{ep_i + 1}/{n_run}] ep={episode.episode_id} target={target} "
            f"success={rec['success']:.0f} spl={rec['spl']:.3f} steps={steps} "
            f"fps={rec['control_fps']}"
        )

    summary = {
        "config": {
            "scorer": cfg.exploration.scorer,
            "verification": cfg.verification.enabled,
            "detector": cfg.detector.name,
            "dataset_version": cfg.eval.dataset_version,
            "split": cfg.eval.split,
            "success_distance": cfg.agent.success_distance,
        },
        "metrics": aggregate(results),
        "per_category": per_category(results),
        "per_floor_class": per_floor_class(results),
        "timing": profiler_all.report(),
        "pipeline_fps": round(profiler_all.fps("control_loop"), 2),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    profiler_all.write_csv(str(out_dir / "timing.csv"))
    scorer.shutdown()
    env.close()
    print(json.dumps(summary["metrics"], indent=2))
    return summary
