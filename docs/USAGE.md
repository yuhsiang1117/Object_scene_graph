# How to run OSG's ObjectNav pipeline

The default configuration is the S71 arm: ASCENT's control flow
(`src/navigation/`) on ASCENT's served perception models, sensor-only, on the
frozen PointNav mover. On the 100-episode HM3D v1 split `scenes20_ep0to4` it
scores **63.0% SR / 0.36 SPL**; native ASCENT on the same episodes scores
65.0% / 0.36 (`docs/history/AB_RESULTS.md`, S71). Everything below runs inside the
`nav` container (see `README.md` → Quick start for building it).

## 1. One-time setup

**See [docs/SETUP.md](SETUP.md)** — the submodule (`--recursive` is mandatory),
the GroundingDINO CUDA extension that must be compiled, ~5 GB of model weights,
and the datasets. In short:

```bash
git submodule update --init --recursive     # ASCENT + its nested submodules (also needed to build)
bash scripts/fetch_ascent_weights.sh
python scripts/download_weights.py --pointnav --rednet
docker exec docker-ollama-1 ollama pull qwen2.5:7b
```

The five ASCENT models run in the `ascent` conda env, which `docker/Dockerfile`
builds alongside `habitat` (`docs/SETUP.md` §5) — BLIP-2's `lavis` and
habitat-sim cannot share an interpreter.

## 2. Start the model servers

```bash
bash scripts/serve_perception.sh          # tmux session `osg_models`, five windows
bash scripts/serve_perception.sh --stop
```

| window | port | role in the pipeline |
|---|---|---|
| `blip2itm` | 13182 | value-map cosine every step; the commit gate latches at ≥ 0.15 |
| `sam` | 13183 | one MobileSAM mask per D-FINE box |
| `gdino` | 13184 | GroundingDINO `"stair ."` ≥ 0.60, ANDed with RedNet for the stair mask |
| `ram` | 13185 | RAM++ scene tags for the LLM prompt, once per step |
| `dfine` | 13186 | D-FINE COCO detector, target detections at conf ≥ 0.8 |

Logs land in `relative_work/ascent/debug/vlm_logs/<window>.log`. The GPU
budget with all five plus habitat is ~12 GB.

Only run **one evaluation at a time** against these servers. They hold
per-image state on the model object and are not reentrant — the submodule is
pinned to upstream, without the lock a local patch once added — so concurrent
requests race and the loser gets HTTP 500. The client retries once, which
covers a transient hit, not a second eval.

The run **refuses to start** if any server it needs is down
(`PerceptionUnavailable`, raised by `probe_served_models` before Habitat
loads), and a server that stops answering mid-run raises the same error
instead of returning a neutral value. That is deliberate: under ASCENT's gate a
silent 0 from BLIP-2 is an agent that never STOPs, and a 0% run would look
like a bad algorithm.

## 3. Run an evaluation

```bash
python scripts/run_eval.py                                   # the default: 100 episodes, ~3.5 h
python scripts/run_eval.py output_dir=outputs/my_run         # name the run
python scripts/run_eval.py eval.behaviour_log=true           # + per-step trace for the comparison script
python scripts/run_eval.py eval.num_episodes=3               # smoke
python scripts/run_eval.py +experiment=final_sensor          # the full v1 val split (2000 episodes)
bash scripts/run_full_split.sh                               # ...supervised: restarts itself until done
```

The full split takes ~60 h and has twice been killed by things unrelated to the
agent (a model server returning HTTP 500 under a concurrent eval; a session
teardown). `scripts/run_full_split.sh` re-derives the unscored episodes from
`episodes.jsonl` before every attempt and relaunches, so a restart can neither
re-run a scored episode nor skip an unscored one; it adopts a run already in
flight, holds a lock so two supervisors cannot race, brings the model servers
back if they died, and gives up after three attempts that score nothing rather
than looping on a real fault. `--status` reports what is left. It is a process
in this container, so it does **not** survive a container restart or a session
teardown — run it again afterwards and it picks up where the record left off.

`+experiment=ascentnav` names the default explicitly and composes to the
identical config (pinned by `tests/unit/golden/experiment_fingerprints.json`).

Outputs, under `output_dir`:

- `episodes.jsonl` — one record per episode: `success`, `spl`, `steps`,
  `distance_to_goal`, `agent_stats` (LLM calls, gate latches, climb attempts,
  give-ups…), `state_log`, `giveup_log`, `approach_stop_reason`, and with
  `eval.behaviour_log=true` the full `step_trace`.
- `summary.json` — the config used, the algorithm field inventory, SR / SPL /
  mean steps.
- `timing.csv` — per-episode wall time.

## 4. Compare against ASCENT

`data/reference/ascent_behaviour_100/` is native ASCENT's recorded
100-episode run on the same split (65.0%). Pair any OSG run against it:

```bash
python scripts/compare_ascent_osg.py data/reference/ascent_behaviour_100 outputs/my_run
```

**The reference trace is not in this checkout.** `data/reference/ascent_behaviour_100/` was
meant to be committed beside the code (commit `11e0b4a` says so), but `data/*` was ignored
and it never was, and the recorder patch that produced it is no longer applied to the
submodule, so it cannot be regenerated. `.gitignore` now admits `data/reference/`: whoever
holds the copy from the S71 machine should put it back there and commit it. Until then,
`scripts/compare_runs.py` still pairs two OSG runs episode by episode.

The report gives paired SR by floor class and category, the episodes each
side wins, and — scored against the dataset's own object positions — the
trace metrics that carried the S71 diagnosis:

| metric | what it tells you | ASCENT | S71 |
|---|---|---|---|
| SAW episodes | the target was within 3 m / ±40° at some step | 90 | 87 |
| never-saw → STOP / timeout | the search was truncated before the target was in frame | 8 / 1 | 9 / 3 |
| climb-mode steps, same-floor episodes | budget lost to staircases on episodes that need none | 5.2% | 5.8% |
| earliest STOP, STOPs before 33 | premature irreversible stops | 23, 3 | 23, 3 |
| P(success \| committed) | conversion once the agent walks at a detection | 0.711 | 0.693 |

A change that improves SR should move one of these; a change that moves SR
without moving any of them is noise at n = 100 (±3 points).

## 5. Variants worth running

| command | what it tests |
|---|---|
| `+experiment=ascentnav_union_stairs` | OSG's looser stair sensing (RedNet union, no passive entry, lip test) in the same control flow |
| `+agent.stair_disable_burns_map=true` | run the reference's dead stair-burn branch (F3) so a failed staircase is not retried (the `+` is needed: no dataclass field) |
| `exploration.llm_multi_floor=true` | the reference's multi-floor prompt, which its own run never reached (F2) |
| `agent.downstair_detector=lip` | OSG's "missing floor" down-stair trigger instead of the reference's mirrored depth |
| `verification.enabled=true` | add OSG's forced-choice VLM verifier on top (needs `NVIDIA_API_KEY`) |
| `eval=scenes20_crossfloor` | only the cross-floor episodes |

Pre-S71 presets (`matched_*`, `ycb_*`, `ascent_sensor*`, `full_v1_navmesh`)
still compose to exactly what they were measured with: each pulls
`configs/legacy_defaults.yaml` first, which restores the old base (YOLOE-11s,
NIM, the VLM verifier, the text-LLM frontier scorer).

## 6. Tests

```bash
pytest tests/unit -q                       # ~1535 tests, ~50 s, no GPU or data
python tests/unit/test_config_snapshot.py  # regenerate the config golden after a deliberate default change
```

The transcription is pinned by `tests/unit/test_ascentnav_{navigate,explore,
planner,perception,stairs,geometry}.py`; each test names the reference line it
was transcribed from.

## 7. Where things are

| | |
|---|---|
| the agent | `src/navigation/agent.py` (dispatch, `_navigate`, `_explore`), `stairs.py`, `planner.py`, `perception.py` |
| the maps | `src/navigation/mapping/` (vendored from ASCENT) |
| served-model clients | `src/osg/perception/ascent_models.py`, `detector.py` (`DFineDetector`), `image_text.py` (`Blip2ItmScorer`) |
| the mover | `src/osg/planning/pointnav_driver.py` (weights bit-identical to ASCENT's) |
| config | `configs/config.yaml` → `agent/s71`, `exploration/s71`, `verification/s71`, `detector/dfine`, `llm/qwen_local`, `scene_graph/place365`, `eval/scenes20_ep0to4` |
| setup | `docs/SETUP.md` — submodule, CUDA extension, weights, datasets |
| results log | `docs/history/AB_RESULTS.md` (S71 is the current record), `src/navigation/README.md` (fidelity notes F1–F14) |
| methodology | `docs/METHODOLOGY.md` — protocol, evidence rules, differences from native ASCENT |
