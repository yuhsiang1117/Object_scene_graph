"""How many object nodes does a real run accumulate that were created on a
single sighting and never re-confirmed by a later observation? Every
detection that fails to match an existing same-label track immediately
spawns a new ObjectTrack with no confidence/quality gate (object_layer.py);
this measures how much of that is actually orphaned noise vs. real objects
that just happen to only be seen once (e.g. glimpsed in passing, never
revisited).

For each episode, runs the full NavAgent pipeline and inspects
agent.object_layer at the end: how many tracks exist, how many have
n_obs==1 (never re-associated after creation -- the strictest "orphan"
definition), how many have n_obs < min_obs_for_refine (never accumulated
enough support to even run the Wasserstein refine step), broken down by
label, and whether they were ever promoted to a real APPROACH candidate.

Usage: python scripts/orphan_node_check.py --num-episodes 6
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from hydra import compose, initialize_config_dir

from osg.core.config import register_configs

register_configs()
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
    cfg = compose(config_name="config", overrides=["eval=hm3d_val_mini"])

from osg.agent.nav_agent import NavAgent  # noqa: E402
from osg.eval.runner import _unload_ollama_models, build_detector, build_scorer, build_verifier  # noqa: E402
from osg.sim.habitat_env import HabitatObjectNavEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-episodes", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=500)
    args = ap.parse_args()

    _unload_ollama_models(cfg)
    env = HabitatObjectNavEnv(cfg)
    detector = build_detector(cfg)
    scorer = build_scorer(cfg)
    verifier = build_verifier(cfg)

    totals = Counter()
    label_orphans = Counter()
    label_totals = Counter()
    per_ep_rows = []

    n_total = len(env.env.episodes)
    n_run = min(args.num_episodes, n_total)
    for ep_i in range(n_run):
        frame = env.reset()
        episode = env.current_episode
        target = env.target_category()
        agent = NavAgent(cfg, detector, scorer, verifier, target, profiler=None)

        step = 0
        while not env.episode_over and step < args.max_steps:
            action = agent.act(frame)
            frame = env.step(action)
            step += 1

        tracks = agent.object_layer.tracks(include_blacklisted=True)
        n_total_tracks = len(tracks)
        n_orphan = sum(1 for t in tracks if t.n_obs == 1)
        n_weak = sum(1 for t in tracks if t.n_obs < cfg.scene_graph.min_obs_for_refine)
        n_blacklisted = sum(1 for t in tracks if t.blacklisted)
        n_candidate_ever = sum(
            1 for t in tracks
            if not t.blacklisted and t.n_obs >= cfg.verification.min_obs
            and t.best_score >= cfg.verification.min_score
            and t.best_bbox_px >= cfg.verification.min_bbox_px
        )
        totals["tracks"] += n_total_tracks
        totals["orphan_n_obs1"] += n_orphan
        totals["weak_below_refine"] += n_weak
        totals["blacklisted"] += n_blacklisted
        totals["ever_candidate_quality"] += n_candidate_ever

        for t in tracks:
            label_totals[t.label] += 1
            if t.n_obs == 1:
                label_orphans[t.label] += 1

        per_ep_rows.append((episode.episode_id, target, n_total_tracks, n_orphan, n_weak, n_blacklisted))
        print(f"[{ep_i + 1}/{n_run}] ep={episode.episode_id} target={target} steps={step} "
              f"tracks={n_total_tracks} orphan(n_obs=1)={n_orphan} "
              f"weak(<{cfg.scene_graph.min_obs_for_refine} obs)={n_weak} blacklisted={n_blacklisted}")

    env.close()
    scorer.shutdown()

    print()
    print("=== aggregate ===")
    print(f"total tracks created: {totals['tracks']}")
    print(f"  orphan (n_obs==1, never re-confirmed): {totals['orphan_n_obs1']} "
          f"({100 * totals['orphan_n_obs1'] / max(totals['tracks'], 1):.1f}%)")
    print(f"  weak (< min_obs_for_refine={cfg.scene_graph.min_obs_for_refine}): {totals['weak_below_refine']} "
          f"({100 * totals['weak_below_refine'] / max(totals['tracks'], 1):.1f}%)")
    print(f"  blacklisted (verifier-rejected): {totals['blacklisted']}")
    print(f"  ever reached candidate-quality gate: {totals['ever_candidate_quality']} "
          f"({100 * totals['ever_candidate_quality'] / max(totals['tracks'], 1):.1f}%)")

    print()
    print("=== orphan rate by label (label: orphans/total) ===")
    for label, tot in sorted(label_totals.items(), key=lambda kv: -kv[1]):
        orph = label_orphans.get(label, 0)
        print(f"  {label:20s} {orph:3d}/{tot:3d}  ({100 * orph / tot:.0f}%)")


if __name__ == "__main__":
    main()
