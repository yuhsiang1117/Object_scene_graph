"""Measure how WassersteinRefiner affects object-ellipsoid CENTER accuracy.

For each target-category track, compare its centre at initialization (single-view
back-projection) vs after multi-view refinement, against the HM3D ground-truth
object position (episode.goals[].position, nearest instance). Reports whether
refine moves centres closer to or further from GT, broken down by n_obs, and
saves a scatter (init error vs refined error).

Usage: python scripts/analyze_refine_accuracy.py --num 20
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from osg.core.config import register_configs

register_configs()
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
    cfg = compose(config_name="config", overrides=["eval=hm3d_val_single_floor", "llm=ollama"])

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from osg.eval.runner import build_detector, build_scorer  # noqa: E402
from osg.agent.nav_agent import NavAgent  # noqa: E402
from osg.mapping.costmap import PLANE  # noqa: E402
from osg.sim.habitat_env import HabitatObjectNavEnv  # noqa: E402


def _norm(s):
    return s.lower().replace("_", " ").replace("-", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=300)
    args = ap.parse_args()

    detector = build_detector(cfg); scorer = build_scorer(cfg); env = HabitatObjectNavEnv(cfg)
    recs = []
    for _ in range(min(args.num, len(env.env.episodes))):
        frame = env.reset(); ep = env.current_episode; target = env.target_category()
        scorer.reset()
        agent = NavAgent(cfg, detector, scorer, None, target, profiler=None)

        # capture each refine: track id -> list of (n_obs, before_center, after_center)
        rlog = {}
        orig = agent.object_layer._refiner.refine
        def wrapped(track, _o=orig, _r=rlog, **kw):
            before = track.ellipsoid.center.copy()
            refined = _o(track, **kw)
            _r.setdefault(track.id, []).append(
                (track.n_obs, before, refined.center.copy() if refined is not None else None))
            return refined
        agent.object_layer._refiner.refine = wrapped

        step = 0
        while not env.episode_over and step < args.max_steps:
            frame = env.step(agent.act(frame)); step += 1

        # GT object positions for the target category (habitat world coords)
        gt = []
        for g in ep.goals:
            p = np.asarray(g.position, dtype=float)
            gt.append(p)
        gt = np.array(gt) if gt else np.zeros((0, 3))

        for t in agent.object_layer.tracks():
            if _norm(t.label) != _norm(target):
                continue
            final_c = t.ellipsoid.center.copy()
            log = rlog.get(t.id, [])
            init_c = log[0][1] if log else final_c  # before first refine == init (unchanged pre-refine)
            n_ref = sum(1 for (_, _, a) in log if a is not None)  # accepted refines
            if len(gt) == 0:
                continue
            # errors: full 3D and horizontal (ground plane) to nearest GT instance
            def err(c):
                d3 = np.linalg.norm(gt - c, axis=1).min()
                dh = np.linalg.norm(gt[:, list(PLANE)] - c[list(PLANE)], axis=1).min()
                return float(d3), float(dh)
            e3_i, eh_i = err(init_c)
            e3_f, eh_f = err(final_c)
            recs.append(dict(ep=str(ep.episode_id), target=target, n_obs=t.n_obs,
                             n_refine=n_ref,
                             init_err3=e3_i, refined_err3=e3_f,
                             init_errh=eh_i, refined_errh=eh_f))
        print(f"ep{ep.episode_id} {target}: target tracks={sum(1 for r in recs if r['ep']==str(ep.episode_id))}")

    env.close(); scorer.shutdown()

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/refine_accuracy.json").write_text(json.dumps(recs, indent=1))
    refined = [r for r in recs if r["n_refine"] > 0]
    print(f"\n=== {len(recs)} target tracks ({len(refined)} refined) over episodes ===")
    if refined:
        import statistics as st
        ii = [r["init_errh"] for r in refined]; ff = [r["refined_errh"] for r in refined]
        print(f"horizontal centre error (m): init  mean={st.mean(ii):.2f} median={st.median(ii):.2f}")
        print(f"                             refined mean={st.mean(ff):.2f} median={st.median(ff):.2f}")
        improved = sum(1 for r in refined if r["refined_errh"] < r["init_errh"] - 0.02)
        worse = sum(1 for r in refined if r["refined_errh"] > r["init_errh"] + 0.02)
        same = len(refined) - improved - worse
        print(f"refine effect: improved={improved} worse={worse} ~same={same}")
        for lo, hi in [(3, 3), (4, 5), (6, 9), (10, 999)]:
            g = [r for r in refined if lo <= r["n_obs"] <= hi]
            if g:
                di = st.mean(r["init_errh"] for r in g); dfn = st.mean(r["refined_errh"] for r in g)
                print(f"  n_obs {lo}-{hi if hi<999 else '+'}: n={len(g):>2} init={di:.2f} refined={dfn:.2f} "
                      f"delta={dfn-di:+.2f}m")
        # scatter
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        m = max(max(ii), max(ff), 1.0)
        ax.plot([0, m], [0, m], "k--", lw=1, alpha=0.5)
        ax.scatter(ii, ff, c=[r["n_obs"] for r in refined], cmap="viridis", s=40, edgecolors="k", linewidths=0.4)
        ax.set_xlabel("init centre error (m, horizontal)")
        ax.set_ylabel("refined centre error (m)")
        ax.set_title("refine effect on ellipsoid centre (below line = improved)")
        ax.set_xlim(0, m); ax.set_ylim(0, m); ax.set_aspect("equal")
        fig.savefig("outputs/refine_accuracy.png", dpi=120, bbox_inches="tight")
        print("wrote outputs/refine_accuracy.json + refine_accuracy.png")


if __name__ == "__main__":
    main()
