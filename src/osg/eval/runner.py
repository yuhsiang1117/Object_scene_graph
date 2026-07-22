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
from ..mapping.costmap import PLANE
from .metrics import aggregate, per_category
from .visualize import save_topdown


def build_scorer(cfg) -> AsyncScorer:
    # Old-algorithm pipeline: text-LLM frontier ranking over the scene-graph
    # subgraphs (ObjectSceneGraph_old frontiers_ranking).
    client = ChatClient(
        cfg.llm.base_url, cfg.llm.text_model, cfg.llm.api_key,
        cfg.llm.timeout_s, cfg.llm.max_image_px, cfg.llm.send_response_format,
    )
    inner = LLMTextScorer(client, cfg.exploration.subgraph_radius_m,
                          cfg.exploration.max_frontiers_per_call)
    return AsyncScorer(inner)


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


def run_eval(cfg) -> dict:
    from ..sim.habitat_env import HabitatObjectNavEnv

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _unload_ollama_models(cfg)
    env = HabitatObjectNavEnv(cfg)
    detector = build_detector(cfg)
    scorer = build_scorer(cfg)
    verifier = None  # old-algorithm terminal: viewpoint pre-positioning + bbox stop, no VLM verify

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

        # scorer/verifier are built once and shared across every episode in
        # this run, so their call/error counters are cumulative — snapshot
        # before and report deltas below, or every episode after the first
        # would show the whole run's running total instead of its own.
        llm_calls_before, llm_errors_before = scorer.n_calls, scorer.n_errors
        llm_last_error_before = scorer.last_error

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
            keyframe_dir=str(out_dir / "keyframes" / f"ep{episode.episode_id}")
            if cfg.eval.save_viz else None,
            profiler=profiler,
        )
        trajectory = [frame.camera_position[list(PLANE)]]
        t0 = time.time()
        steps = 0
        while not env.episode_over:
            action = agent.act(frame)
            frame = env.step(action)
            trajectory.append(frame.camera_position[list(PLANE)])
            steps += 1

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
            "giveup_log": agent.giveup_log[:50],
            "approach_bbox_log": agent.approach_bbox_log,
            "approach_stop_reason": agent.approach_stop_reason,
            "final_xy": [float(x) for x in trajectory[-1]],
        }
        results.append(rec)
        with open(episodes_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
        for name, samples in profiler._samples.items():
            for s in samples:
                profiler_all.add(name, s)

        if cfg.eval.save_viz:
            save_topdown(
                str(out_dir / "viz" / f"ep{episode.episode_id}.png"),
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
