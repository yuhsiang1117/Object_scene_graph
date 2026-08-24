"""The run: build the stack once, drive every episode, summarise.

This file is deliberately thin. Everything it calls is a module named after what
it does -- what the run is MADE of is `pipeline/components.py`, what one episode
DOES is `episode.py`, what gets written down is `record.py` -- so the shape of a
run is legible here in one screen and no mechanism has to be understood to read
it.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..agent.nav_agent import NavAgent
from ..core.profiler import Profiler
from ..pipeline.components import (
    build_detector,
    build_env,
    build_scorer,
    build_verifier,
    unload_ollama_models,
)
from .debug_video import DebugVideo
from .episode import run_episode
from .metrics import aggregate, dynamic_summary, per_category, per_floor_class
from .prior_map import save_map_for_scene
from .record import (
    build_episode_record,
    detector_identity,
    episode_tag,
)
from .visualize import save_topdown


def run_eval(cfg) -> dict:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unload_ollama_models(cfg)
    env = build_env(cfg)
    detector = build_detector(cfg)
    scorer = build_scorer(cfg)
    verifier = build_verifier(cfg)
    identity = detector_identity(cfg)
    if verifier is not None and cfg.eval.debug_frames:
        verifier.debug_dir = str(out_dir / "verify_debug")

    n_total = len(env.env.episodes)
    n_run = n_total if cfg.eval.num_episodes < 0 else min(cfg.eval.num_episodes, n_total)
    wanted = set(cfg.eval.episode_ids) if cfg.eval.episode_ids else None

    results: list = []
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

        # The scorer and verifier are built once and shared, so their counters
        # are cumulative; the record reports deltas against these.
        scorer_before = (scorer.n_calls, scorer.n_errors, scorer.last_error)
        verifier_before = (
            (verifier.n_calls, verifier.n_errors) if verifier is not None else (0, 0)
        )
        # frontier.id/room.id restart from 0/1 each episode (a fresh extractor
        # and segmenter per NavAgent), but the scorer's caches are keyed by
        # those same small ints and persist across the run -- without this, a
        # new episode can inherit a stale score from an unrelated scene the
        # moment an id collides.
        scorer.reset()

        profiler = Profiler()
        agent = NavAgent(
            cfg, detector, scorer, verifier, target,
            keyframe_dir=str(out_dir / "keyframes" / ep_tag) if cfg.eval.save_viz else None,
            profiler=profiler,
            nav_fn=env.action_to_goal if cfg.agent.use_habitat_navmesh else None,
            reachable_fn=env.is_reachable if cfg.agent.use_habitat_navmesh else None,
        )
        debug = DebugVideo(cfg, out_dir, ep_tag) if cfg.eval.debug_frames else None
        outcome = run_episode(cfg, env, agent, episode, target, frame, detector, debug)
        if debug is not None:
            debug.close()

        save_map_for_scene(cfg, agent, episode)
        rec = build_episode_record(
            cfg=cfg, episode=episode, env=env, agent=agent, outcome=outcome,
            target=target, detector_identity=identity, metrics=env.metrics(),
            profiler=profiler, scorer=scorer, scorer_before=scorer_before,
            verifier=verifier, verifier_before=verifier_before,
        )
        results.append(rec)
        with open(episodes_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
        for name, samples in profiler._samples.items():
            for sample in samples:
                profiler_all.add(name, sample)

        if cfg.eval.save_viz:
            save_topdown(
                str(out_dir / "viz" / f"{ep_tag}.png"),
                agent.costmap,
                outcome.trajectory,
                scene_graph=agent.scene_graph,
                title=f"ep {episode.episode_id} target={target} "
                      f"success={rec['success']:.0f} spl={rec['spl']:.2f}",
            )
        print(
            f"[{ep_i + 1}/{n_run}] ep={episode.episode_id} target={target} "
            f"success={rec['success']:.0f} spl={rec['spl']:.3f} "
            f"steps={outcome.steps} fps={rec['control_fps']}"
        )

    summary = _summarise(cfg, env, results, identity, profiler_all)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    profiler_all.write_csv(str(out_dir / "timing.csv"))
    scorer.shutdown()
    env.close()
    print(json.dumps(summary["metrics"], indent=2))
    return summary


def _summarise(cfg, env, results, identity, profiler_all) -> dict:
    """`dynamic` is the block that answers the dynamic-scene questions: how long
    the map took to stop believing a moved object, how often the agent committed
    to a goal it had already disproved, and how often a ghost survived."""
    summary = {
        "config": {
            "eval_mode": str(cfg.eval.mode),
            "scorer": cfg.exploration.scorer,
            "verification": cfg.verification.enabled,
            "detector": identity,
            "dataset_version": cfg.eval.dataset_version,
            "split": cfg.eval.split,
            "success_distance": cfg.agent.success_distance,
        },
        "metrics": aggregate(results),
        "dynamic": dynamic_summary(results),
        "per_category": per_category(results),
        "per_floor_class": per_floor_class(results),
        "timing": profiler_all.report(),
        "pipeline_fps": round(profiler_all.fps("control_loop"), 2),
    }
    benchmark_metadata = getattr(env, "benchmark_metadata", None)
    if callable(benchmark_metadata):
        summary["benchmark"] = benchmark_metadata()
    return summary
