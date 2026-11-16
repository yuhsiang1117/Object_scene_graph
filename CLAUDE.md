# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` holds the shorter contributor guide (style, commit/PR conventions); this file covers
the architecture and the workflows that need more than one file to understand.

## Environment

Everything runs **inside the `nav` container** — `PYTHONPATH`, `NVIDIA_API_KEY`, the habitat conda
env and the `OSG_*` dataset roots are set by the image/compose env, so no `docker exec` prefix or
`PYTHONPATH=src` is needed once you are in the shell:

```bash
docker compose -f docker/compose.yaml --env-file docker/.env build nav
docker compose -f docker/compose.yaml --env-file docker/.env up -d
docker exec -it docker-nav-1 bash
```

The five ASCENT perception models run in a *second* conda env (`/opt/conda/envs/ascent`,
built by the single `docker/Dockerfile`; containers from before that merge fall back to
`/workspace/.conda-envs/ascent`) because BLIP-2's `lavis` and habitat-sim cannot share an
interpreter. They are served over HTTP on ports 13182–13186 by `bash scripts/serve_perception.sh`
(tmux session `osg_models`; `--stop` to tear down).

For CPU-only work: `pip install -e '.[dev]'`.

## Commands

```bash
pytest tests/unit -q                                   # 1535 tests, ~50 s, no GPU or data
pytest tests/unit/test_floors.py -q                    # one file
pytest tests/unit/test_floors.py::test_name -q         # one test
pytest -m sim                                          # tests/integration; needs Habitat + HM3D
python tests/unit/test_config_snapshot.py              # REGENERATE the config golden (see below)

python scripts/run_eval.py                             # the default arm (S87 = S71 + storey belief/climb check/stair heading memory), 100 episodes, ~3.5 h
python scripts/run_eval.py eval.num_episodes=3         # smoke
python scripts/run_eval.py +experiment=<preset>        # compose a whole preset
python scripts/run_eval.py detector=yoloe_small llm=ollama verification=off   # override groups
python scripts/smoke_habitat.py                        # EGL rendering
python scripts/smoke_ollama.py                         # LLM round-trip
```

A run **refuses to start** if a model server or ollama it needs is down (`PerceptionUnavailable`,
raised by `probe_served_models` before Habitat loads), and a server that goes quiet mid-run raises
rather than returning a neutral value. This is deliberate: a silent 0 from BLIP-2 under ASCENT's
commit gate is an agent that never STOPs, and a 0% run looks like a bad algorithm rather than a
dead server. Do not "fix" this by falling back to a default score.

## Architecture

### Two agent stacks, not one

- `src/osg/` — the project's own pipeline: open-vocab perception → a `floor → room → container →
  object` 3D scene graph → frontier exploration → approach + verify.
- `src/navigation/` — a line-by-line **transcription** of ASCENT's control flow (`Ascent_Policy.act`
  + `Map_Controller`) plus its vendored maps, running in OSG's harness. It is the current default
  arm, and it is an attributed alternative policy, *not* a replacement for OSG's dynamic hierarchy.
  Its rules cite the reference line they came from; `src/navigation/README.md` carries the F1–F14
  fidelity notes. Changing behaviour here means breaking the transcription — do it knowingly, and
  the `test_ascentnav_*.py` tests will tell you.

Three orthogonal axes select what actually runs:

| axis | flag | options |
|---|---|---|
| mover | `agent.navigation` | `costmap` (own A*/Voronoi planner), `pointnav` (frozen depth + point-goal policy), `navmesh` (Habitat `ShortestPathFollower`), `nav2` (publish the goal, let a Nav2 stack drive) |
| policy | `agent.policy` | `nav_agent` (OSG FSM + dynamic world model), `ascent`, `ascentnav` |
| benchmark | `eval.mode` | `objectnav` (HM3D), `ycb_authored` (authored dynamic layouts), `dualmap_protocol` (DualMap's released benchmark and scoring rule), `ros2` (a real robot; see below) |

**`navmesh` is privileged navigation** — it is the only mover given simulator geometry
(`action_to_goal` / `is_reachable`). Its SR/SPL must never be compared against sensor-only methods
(`costmap`, `pointnav`). `agent.use_habitat_navmesh` is a legacy alias for it. **`nav2` is
privileged too**, by both backends: in simulation it wraps that same follower, and on the robot
Nav2 plans on a map the robot was given.

### The real robot

`docs/ROS2.md` is the whole of it. A Hello Robot Stretch 3 runs its own Nav2 stack, so the
pipeline publishes the metric goal it already computes and stops steering; sensor data comes
back the other way. Two facts shape that layer and neither is negotiable: Humble's `rclpy` is
built for Python 3.10 while both conda envs here are 3.9, so the ROS node is a separate
process reached over a socket (`src/osg/ros2/`, importable under both interpreters); and
Nav2's `map` is 2D, so the storey is **declared by the operator** (`/osg/floor`,
`floor.source=external`) rather than estimated from a height that does not exist. Check the
whole path with no robot present: `bash scripts/ros2/check_pipeline.sh`.

The deployment is a **Jetson AGX Thor** beside the robot, and it is a SECOND image pair, not
an override of the x86 one: `docker/compose.thor.yaml` builds `Dockerfile.thor` (L4T, the
pipeline and the models, **no Habitat** — there is no aarch64 habitat-sim, and `eval.mode=ros2`
never needs one) and `Dockerfile.bridge` (`ros:humble-ros-base`, rclpy only, matching the
Stretch's own distro). `docs/THOR.md` has the topology and every knob that can only be
settled on the device.

### What the OSG pipeline is built for

The premise is not an *incomplete* map but a map that is **actively wrong**: pass 1 explores a
static layout and snapshots it, objects are moved, pass 2 starts from that snapshot believing the
target is at A when it is at B. Every mechanism exists to let a map be wrong, notice, and recover —
a presence Bayes filter with a separate "would we have seen it" channel (`objects/presence.py`), an
identity channel (`agent/candidate.py`), absence-as-evidence (`verification/absence.py`), and a
search posterior over mapped surfaces (`exploration/search_belief.py`). `docs/ARCHITECTURE.md`
explains each and why its constants are what they are.

Dataflow: `perception` → `objects` (ellipsoid layer: dual-quadric projection, association,
Wasserstein refine) → `mapping` (per-floor costmaps, floor estimation, frontiers, portals) →
`graph` (hierarchy + priors + LLM serialization) → `exploration` → `planning` → `verification` →
`agent` (FSM) → `sim` / `eval`.

`agent/nav_agent.py` owns the FSM state and nothing else; each line of its control loop delegates
to one module. Keep it that way.

### The eval layering

`eval/runner.py` is deliberately thin: what a run is *made of* is `pipeline/components.py` (the
swappable detector / scorer / verifier / env, each an A/B axis), what one episode *does* is
`eval/episode.py`, what gets written down is `eval/record.py`. Adding a model or a benchmark should
not touch the file that drives the episode loop.

Outputs land in `output_dir` (default `outputs/<timestamp>/`): `summary.json` (SR/SPL + per-module
FPS + config fingerprint), `episodes.jsonl` (rich per-episode diagnostics — `state_log`,
`frontier_select_log`, `approach_diag`, `verify_calls`, and with `eval.behaviour_log=true` a full
`step_trace`), `timing.csv`, `viz/*.png`, and with `eval.debug_frames=true` per-step
`viz/debug/<scene>_ep<ID>.mp4` plus `verify_debug/` (the exact image sent to the VLM). Artifact
names carry the scene because HM3D episode ids repeat across scenes.

### Configuration

Hydra groups with a YAML directory live in `configs/` (`agent`, `detector`, `eval`,
`exploration`, `llm`, `profile`, `scene_graph`, `verification`), and are selected as
`detector=yoloe_small`. `configs/experiment/*.yaml` are composable presets applied with
`+experiment=name`; `configs/experiment/archive/` holds the exploratory ladders no current
document runs, composable as `+experiment=archive/name` and pinned under that key. Every group is backed by a **structured dataclass** under
`src/osg/core/config/` (one module per group) registered with Hydra's ConfigStore, so a typo in a
YAML or CLI override fails fast instead of silently creating a key.

Two base mixins matter:

- `configs/s71_defaults.yaml` — the values the current default arm needs that live outside a group.
- `configs/legacy_defaults.yaml` — the pre-S71 base (YOLOE-11s, NIM, the VLM verifier, the text-LLM
  frontier scorer). Every pre-S71 preset lists it **first** so it still composes to exactly what it
  was measured with.

Nearly every default was chosen by an experiment and carries the measurement in a comment beside
it. Two golden files pin this:

- `tests/unit/golden/config_snapshot.json` — every calibrated constant, flattened. A refactor that
  moves a file must not move a number; when you change a default *on purpose*, run
  `python tests/unit/test_config_snapshot.py` so the change appears in the diff as what it is.
- `tests/unit/golden/experiment_fingerprints.json` — pins what each preset composes to, so a base
  change cannot silently redefine a measured arm.

## Working on this repo

- **Reproducibility:** runs are not reproducible while the VLM verifier is on. Use
  `verification=off` for any A/B meant to prove two configs equivalent.
- **Claiming a result:** the repo's currency is measured behaviour, not plausible reasoning. At
  n=100 SR moves ±3 points on noise, so a change should move one of the paired trace metrics
  (`scripts/compare_ascent_osg.py` prints them) and not just SR. `docs/` holds the reference
  set; `docs/history/` the logs and frozen results — `docs/history/AB_RESULTS.md` is the results
  log, `docs/history/INVESTIGATION.md`, `docs/history/MULTI_FLOOR.md`, `docs/UNIFIED_PIPELINE.md`
  carry why the pipeline is shaped as it is. Read the relevant one before re-litigating a
  decision; `docs/README.md` indexes all of them and marks the superseded ones.
- **The paper:** `docs/PAPER.md` is the ICRA 2027 submission in full. Its Appendix A maps every
  equation to the module implementing it, and Appendix B traces every reported number to the run
  that produced it — including the three rows that no run under `outputs/` currently reproduces.
  The paper calls the system **MD-SG**; the code calls it **OSG** everywhere.
- **Diagnostics before guesses:** `scripts/analyze_*.py` decompose a run's `episodes.jsonl` —
  `analyze_stages.py` (explore vs approach failure), `analyze_floors.py` (SR by floor class,
  floor-estimator audit), `analyze_localization.py` / `analyze_trackloc.py` (stop pose vs GT),
  `analyze_approach.py` (why the terminal approach failed), `analyze_multifloor.py` (Benchmark v2:
same-storey SR by class / start storey, storey decisions, stair climbs, goal commits TRUE / STALE /
FP), `analyze_vlm.py` (every VLM verdict scored against the object's true and prior pose),
`analyze_mfv2.py` (the Benchmark v2 campaign: every arm by seed, each ablation paired against the
final arm with a sign test, failure buckets, wall time). `scripts/release/package.py` packs the
runs into the data release `docs/RELEASE.md` describes.
- **Generated, not source:** `outputs/`, downloads under `data/`, `relative_work/` (the ASCENT
  checkout and its recorded reference runs) and `assets/` (the real-robot recordings) are all
  ignored. Keep secrets and machine-specific dataset paths in `docker/.env`.
- **Skills:** `.claude/skills/` holds the three loops above as procedures — `run-tests` (the
  suites, and when regenerating a golden is right), `run-eval` (preflight, smoke, naming a run,
  what makes an A/B valid) and `diagnose-run` (which analyzer answers which question).
- **Where a script lives says what it is:** `scripts/` is the release surface (entry points,
  analyzers, figure renderers, `ros2/`, `video/`, `campaign/`); `scripts/legacy/` holds the
  one-off probes that produced a number in `docs/history/` and are kept only so that number
  can be re-derived. Do not add to `legacy/`; delete instead.
- The LLM/VLM is queried **asynchronously** — the control loop never blocks on it, which is what
  keeps the pipeline real-time. Preserve that when touching the scorer or verifier.
