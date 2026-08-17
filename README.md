# Object Scene Graph ObjectNav

Open-vocabulary, object-goal navigation on HM3D ObjectNav (Habitat). A
from-scratch Python `osg` package: open-vocab perception → a `floor → room →
object` 3D scene graph → frontier exploration → approach + verify. The pipeline
has two navigation modes and can drive on Habitat's own navmesh (matching the
old ROS stack) for reliable, multi-floor-capable motion.

The current pipeline (see **[docs/INVESTIGATION.md](docs/INVESTIGATION.md)** for
how it got here):

- **Perception — YOLOE** (open-vocab detection **and** segmentation in one
  model; no SAM) → ellipsoid object layer (dual-quadric + Wasserstein refine).
- **Navigation — Habitat navmesh** (`agent.use_habitat_navmesh`): the agent
  drives on Habitat's 3D navmesh (`ShortestPathFollower`), removing the
  from-scratch 2D-costmap failure modes (planner-no-path, stuck). The costmap is
  still built, but only for frontier extraction / the scene graph.
- **Exploration — continuous sweep** (`exploration=sweep`): nearest frontier +
  a momentum bonus that prefers frontiers ahead of the heading, so the agent
  sweeps continuously instead of ping-ponging. **LLM-free** (the LLM frontier
  scorer was found redundant on single-floor).
- **Verification — forced-choice VLM** (`verification=nim`, default): shows the
  VLM the whole frame with the target boxed and makes it pick the category from
  the goal list; rejects detector mislabels (e.g. a stool detected as a chair)
  and unreachable / non-goal instances, then keeps exploring.
- **Terminal — creep** (`agent.terminal_creep`): after the navmesh reports
  arrival, centre the target and walk in until physically blocked, instead of
  stopping at a fixed range. Converts at 78% vs the old depth stop's 55%.
- **Multi-floor** (`floor.*`): per-floor costmaps keyed by an online floor
  estimate, 3D navmesh goals, and portal-based cross-floor exploration timed by
  target-category context. See **[docs/MULTI_FLOOR.md](docs/MULTI_FLOOR.md)**
  (survey + results) and **[docs/MULTI_FLOOR_CN.md](docs/MULTI_FLOOR_CN.md)**
  (中文实现结构说明).

> **Status (2026-08):** best config (`+experiment=full_v1_navmesh`) scores
> **49.8% SR ±2.2 / SPL 0.218** on the **full HM3D v1 val (2000 episodes, 20
> scenes)** — up from ~18% at the start of the SR-gap investigation. Splits:
> single-floor scenes 60.9%, multi-floor scenes 43.8%, cross-floor episodes
> **18.2%** (up from 0.0% before the multi-floor work).
>
> Earlier numbers in these docs quoted a 100-episode subset (5/scene). That
> subset read 48.0% where the truth is 49.8%, with per-split errors up to 8
> points — **do not trust A/Bs run at n=100 here**; with a nondeterministic
> verifier nothing under ~1000 episodes resolves less than about 5 points.
>
> **The dominant remaining loss is committing to the wrong object: 492 of 2000
> episodes (24.6%) end more than 3 m from any goal.** See
> **[docs/ASCENT_GAP.md](docs/ASCENT_GAP.md)** for the full decomposition and
> why this pipeline sits 15.6 points below ASCENT's reported 65.4%.
>
> See **[docs/INVESTIGATION.md](docs/INVESTIGATION.md)** for the SR-gap A/Bs,
> **[docs/MULTI_FLOOR.md](docs/MULTI_FLOOR.md)** for the multi-floor work
> (中文: **[MULTI_FLOOR_CN.md](docs/MULTI_FLOOR_CN.md)**), and
> **[docs/EXPLORATION_COMPARISON.md](docs/EXPLORATION_COMPARISON.md)** for the
> frontier-selection comparison. Note: **runs are not reproducible while the VLM
> verifier is on** — use `verification=off` for any A/B meant to prove two
> configs equivalent.

## Quick start

Build the image, start the containers, and drop into a shell — everything
after that runs **inside the `nav` container** (PYTHONPATH, NVIDIA_API_KEY,
etc. are already set by the image/compose env, so no `docker exec` prefix or
`PYTHONPATH=src` is needed below).

```bash
cp .env.example .env                  # adjust dataset paths if needed
docker compose -f docker/compose.yaml --env-file .env build nav   # nav image (habitat-sim 0.3.1, torch cu121)
docker compose -f docker/compose.yaml --env-file .env up -d       # start ollama + nav containers
docker exec -it object_scene_graph-ollama-1 ollama pull qwen2.5vl:3b
docker exec -it object_scene_graph-nav-1 /entrypoint.sh bash                           # shell inside nav (habitat conda env active)
```

### Smoke tests

```bash
python scripts/smoke_habitat.py       # M0: EGL rendering
python scripts/smoke_ollama.py        # M0: LLM round-trip
pytest tests/unit -q                  # unit tests (no GPU/data needed)
```

### Download the dataset (one-time, requires free HM3D license)

Request access at
https://matterport.com/habitat-matterport-3d-research-dataset for API
credentials, then:

```bash
# scenes — minival (~1 GB) is enough for development
python scripts/download_data.py --username <TOKEN_ID> --password <TOKEN_SECRET> --uids hm3d_minival_v0.2
# later, for full eval: --uids hm3d_val_v0.2
python scripts/download_data.py --episodes-only          # ObjectNav v2 episodes (public)
python scripts/download_weights.py                       # detector + mobileclip weights (offline-safe eval)
```

See `data/README.md` for the full split layout. LLM/VLM defaults to
**NVIDIA NIM** (hosted, `NVIDIA_API_KEY` in `.env`); use `llm=ollama` for a
local model instead.

### Running an eval

```bash
python scripts/run_eval.py eval=hm3d_val_mini            # 3-episode smoke eval
python scripts/run_eval.py eval=hm3d_val                 # full HM3D val (v2), default detector/LLM
python scripts/run_eval.py +experiment=matched_single_floor          # a named experiment preset
python scripts/run_eval.py +experiment=full_v1_navmesh eval.debug_frames=true   # current best config, full debug
```

### Different configs

Override any Hydra group on the CLI, standalone or stacked on a preset:

```bash
python scripts/run_eval.py detector=yoloe_small llm=ollama exploration=sweep verification=off
python scripts/run_eval.py +experiment=matched_single_floor \
    verification=nim eval.num_episodes=35 eval.debug_frames=true
```

Outputs land in `outputs/<timestamp>/`: `summary.json` (SR/SPL + per-module
FPS + config fingerprint), `episodes.jsonl` (rich per-episode diagnostics),
`timing.csv`, `viz/*.png` (top-down maps), and — with `eval.debug_frames=true`
— `viz/debug/<scene>_ep<ID>.mp4` (per-step RGB+segmentation | costmap). Artifact
names carry the scene because HM3D episode ids repeat across scenes.

### Config groups & experiments

Hydra groups under `configs/` — override on the CLI (`group=name`) or compose a
whole preset with `+experiment=name`:

| group | options |
|---|---|
| `detector` | `yoloe` (11l, 640px), `yoloe_small` (11s, 512px) |
| `llm` | `nim` (NVIDIA hosted, default), `ollama` (local) |
| `exploration` | `llm_text` (LLM-scored, default), `nearest` (geometric, no LLM), `sweep` (nearest + momentum, no LLM) |
| `verification` | `nim` (forced-choice VLM, **default**), `nim_terminal` (verify at STOP), `off` |
| `eval` | `hm3d_val` (v2), `hm3d_val_v1` (v1, matched-to-old), `hm3d_val_single_floor`, `hm3d_val_mini` |
| `floor` | multi-floor support; all off by default, enabled by `+experiment=full_v1_navmesh` (see docs/MULTI_FLOOR.md) |

Key agent flags (CLI: `agent.<flag>=...`): `use_habitat_navmesh` (drive on the
navmesh, default off — the `*_navmesh` experiments turn it on),
`exploration.continuity_weight` (momentum), `verification.choice_mode`.

`configs/experiment/` presets: **`full_v1_navmesh`** (current best — navmesh +
sweep + verify, full v1, 5 eps/scene), `matched_navmesh` (single-floor),
`matched_single_floor`, `matched_old`, `matched_verify`,
`matched_terminal_verify`, `single_floor_navgoal`.

### Analysis & debugging

The `scripts/analyze_*.py` tools decompose a run's `episodes.jsonl`:
`analyze_stages.py` (explore vs approach failure), `analyze_floors.py` (SR by
floor class, floor-estimator audit, stair-track rate), `analyze_localization.py` /
`analyze_trackloc.py` (stop-pose / mapped-object vs GT), `analyze_approach.py`
(why the terminal approach failed). Rich per-episode fields include
`state_log`, `frontier_select_log` (every frontier choice: step, agent xy,
chosen frontier, path cost), `approach_diag`, and `verify_calls`.

With `eval.debug_frames=true` a run also writes:
- `viz/debug/<scene>_ep<ID>.mp4` — per-step **RGB + YOLOE segmentation | costmap** (with
  the chosen frontier and planned path drawn), and
- `verify_debug/` (when a verifier is active) — the exact **image sent to the
  VLM** (whole frame + red box) plus `index.jsonl` with the VLM's response and
  accept/reject per call.

See **[docs/INVESTIGATION.md](docs/INVESTIGATION.md)** for the full story.

### 3D scene-graph inspection (re-simulation)

`viz/debug/*.mp4` is a fixed two-panel view (RGB beside a top-down costmap)
and can't be rotated or queried. `scripts/inspect_scene_graph.py` instead
**replays one episode** — it re-runs the agent from scratch outside the eval
loop, so it is not part of `run_eval.py` output — and writes a Rerun `.rrd`
recording: object ellipsoids with true axes/orientation, one costmap plane
per storey at its own height, the 3D trajectory, portals, and goal view
points, all on a scrubbable timeline:

```bash
pip install "rerun-sdk" "numpy<2"          # the pin matters, see below
python scripts/inspect_scene_graph.py +experiment=scene_cvZr5TUy5C5
rerun outputs/inspect/<tag>.rrd            # on your own machine, not the container
```

`--episode-index N` picks which episode in the (filtered) episode list to
replay (default 0), `--max-steps N` stops early, `--floor-every N` controls
how often costmap planes are logged (they dominate file size), and any
Hydra override can be stacked after the experiment, e.g.
`floor.semantic_stairs=true`.

The container is headless, so nothing renders there — copy the `.rrd` out
and open it with the Rerun viewer on your own machine. **The numpy pin is
not optional:** unpinned, pip resolves `rerun-sdk` to a build that requires
numpy>=2, but habitat-sim in this environment is pinned to numpy 1.26.4 and
does not support NumPy 2.0 — installing it breaks the simulator, not just
the viewer.

`--gltf` writes a single self-contained `.glb` instead (ellipsoids as scaled
UV spheres, storeys as textured quads) via `trimesh` — final state only, no
timeline, but viewable in Blender / the VS Code glTF extension / any online
viewer without the Rerun/numpy dependency:

```bash
python scripts/inspect_scene_graph.py +experiment=scene_cvZr5TUy5C5 --gltf
```

Note: the exported object geometry comes from `agent.object_layer`
(`ObjectTrack.ellipsoid`), not from `graph/serialize.to_json` — the scene
graph's `ObjectNodeView` only carries a `center`, no shape. Logging both is
deliberate: where the scene-graph node and the underlying track disagree is
exactly where association or Wasserstein refinement has gone wrong.

## Hardware profiles

| | detector | VLM | fits |
|---|---|---|---|
| default (6 GB, RTX 4050 laptop) | `detector=yoloe_small` (11s, 512px) | `qwen2.5vl:3b` | ~4.5 GB |
| report-quality (>=12 GB) | `detector=yoloe` (11l, 640px) | `llm.text_model=qwen2.5vl:7b llm.vlm_model=qwen2.5vl:7b` | ~8.5 GB |

The LLM/VLM is queried **asynchronously** — the control loop never blocks on
it, which is what keeps the pipeline real-time; decision latency is reported
separately in `timing.csv`.

## Layout

- `src/osg/` — the pipeline: `perception` (YOLOE, keyframes) → `objects`
  (ellipsoid layer: dual-quadric projection, association, Wasserstein
  refinement, linking) → `mapping` (per-floor costmaps, floor estimation,
  frontiers, cross-floor portals, room watershed) →
  `graph` (floor/room/object hierarchy + category priors + LLM serialization) →
  `exploration` (scorers incl. `NullScorer` for geometric, momentum/info-gain
  selector) → `planning` (A*, waypoint controller — used when *not* on the
  navmesh) → `verification` (forced-choice VLM verifier) → `agent` (FSM) →
  `sim` (Habitat env + `ShortestPathFollower` navmesh driving) / `eval`.
- **Navigation** is either the from-scratch costmap planner+controller or, with
  `agent.use_habitat_navmesh`, Habitat's navmesh (`sim/habitat_env.py`:
  `action_to_goal`, `is_reachable`) — the latter is the current best.
- `configs/` — Hydra groups; `configs/experiment/*` are composable presets.
- `scripts/` — eval entry (`run_eval.py`), data/weights download,
  `analyze_*.py` diagnostics, keyframe/video tools, `inspect_scene_graph.py`
  (3D re-simulation of one episode into a Rerun `.rrd` / glTF `.glb`).
- `tests/unit` — synthetic-data tests, no GPU; `tests/integration` — `-m sim`.
