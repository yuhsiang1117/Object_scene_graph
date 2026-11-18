---
name: run-eval
description: Launch an evaluation of the navigation pipeline correctly — preflight the simulator, the five perception servers and the LLM, smoke-run first, name the output, choose a preset, keep A/Bs reproducible, and avoid comparing privileged and sensor-only movers. Use whenever running scripts/run_eval.py (a smoke test, a preset, an A/B arm, the full split) or when a run refuses to start.
---

# Running an evaluation

Inside the `nav` container. `scripts/run_eval.py` is a Hydra entry point; everything after it
is an override. `docs/USAGE.md` is the long form of this page.

## 1. Preflight — in this order, each one gates the next

```bash
python scripts/smoke_habitat.py           # new container / moved driver: one scene, one EGL frame
bash scripts/serve_perception.sh          # tmux `osg_models`: blip2itm :13182, sam :13183,
                                          #   gdino :13184, ram :13185, dfine :13186 (~90 s to load)
for p in 13182 13183 13184 13185 13186; do curl -s -o /dev/null -w "$p %{http_code}\n" localhost:$p; done
                                          # any code means up (404 on / is fine); 000 means down
python scripts/smoke_ollama.py [llm=<group>] # only if the run uses an LLM; same overrides as the run
```

A run **refuses to start** when a server it needs is down (`PerceptionUnavailable`, raised by
`probe_served_models` before Habitat loads) and raises if one dies mid-run. That is deliberate.
Restart the server; never paper over it with a default score.

Run **one evaluation at a time** against the servers — a second one halves both throughputs.

## 2. Smoke first, then the real run

```bash
python scripts/run_eval.py <overrides> eval.num_episodes=3 output_dir=outputs/_smoke_<name>
python scripts/run_eval.py <overrides> output_dir=outputs/<name>
```

- **Always set `output_dir`.** Without it a run lands in `outputs/<timestamp>/` and is hard to
  find or cite later. Hydra also writes `outputs/<date>/<time>/.hydra/` — the only full record
  of the composed config (`summary.json` keeps just a fingerprint), so do not delete the one
  that belongs to a run you will report.
- The default arm (S71) is 100 episodes, ~3.5 h. Start long runs with `run_in_background` and
  watch progress with `wc -l outputs/<name>/episodes.jsonl`; do not poll with sleeps.
- The full v1 val split (~60 h) goes through the supervisor, which resumes from
  `episodes.jsonl` and restarts dead servers: `bash scripts/run_full_split.sh` (`--status`).

## 3. Choosing what runs

```bash
python scripts/run_eval.py +experiment=<preset>             # configs/experiment/<preset>.yaml
python scripts/run_eval.py +experiment=archive/<preset>     # an exploratory ladder rung
python scripts/run_eval.py detector=yoloe_small llm=ollama verification=off   # swap groups
```

Three orthogonal axes: `agent.navigation` (mover), `agent.policy`, `eval.mode` (benchmark).
The paper's arms are `+experiment=dualmap_osg_unified` (single-floor tables) and
`+experiment=mf5_osg_on_ascent_map` (multi-floor). Read `docs/PAPER.md` Appendix B and
`docs/CROSS_ANCHOR_REPRODUCTION.md` before re-running or quoting a paper number: the
single-floor result depends on which prior maps the run loads.

Useful extras: `eval.behaviour_log=true` (per-step trace, needed by
`compare_ascent_osg.py` / `analyse_behaviour.py`), `eval.debug_frames=true` (mp4 + the exact
VLM input per episode; slow), `eval.episode_ids=[...]` (re-run named episodes).

## 4. Rules that decide whether a result means anything

- **Reproducible A/B → `verification=off`.** The VLM verifier is nondeterministic.
- **Never compare privileged and sensor-only movers.** `agent.navigation=navmesh` and `nav2`
  get simulator geometry (or a given map); their SR/SPL is not comparable to `costmap` or
  `pointnav`.
- **n = 100 moves SR by ±3 points on noise.** A claimed improvement must also move a paired
  trace metric — hand the run to the `diagnose-run` skill rather than reporting SR alone.
- Report what `summary.json` says (`metrics.success_rate`, `metrics.spl`) for the run you
  actually finished, with its `output_dir`.

## When it goes wrong

| symptom | cause |
|---|---|
| `PerceptionUnavailable` before Habitat loads | a server in §1 is down — `tmux attach -t osg_models`, logs in `relative_work/ascent/debug/vlm_logs/` |
| config key error at startup | a typo in an override; every group is a structured dataclass, unknown keys fail fast |
| EGL / scene load error | run `scripts/smoke_habitat.py`; it is the mount or the driver |
| SR 0 on a whole run | check the servers answered (`agent_stats` in `episodes.jsonl`) before suspecting the algorithm |
