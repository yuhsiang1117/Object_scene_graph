# Object Scene Graph ObjectNav

Open-vocabulary, object-goal navigation on HM3D ObjectNav (Habitat). A
from-scratch Python `osg` package: open-vocab perception → a `floor → room →
container → object` 3D scene graph → frontier exploration → approach + verify.
The pipeline has three navigation modes and three policies, including the
sensor-only ASCENT path and a floor-aware dynamic-scene path.

The current pipeline (see **[docs/INVESTIGATION.md](docs/INVESTIGATION.md)** for
how it got here):

- **Perception — YOLOE** (open-vocab detection **and** segmentation in one
  model; no SAM) → ellipsoid object layer (dual-quadric + Wasserstein refine).
- **Navigation — `agent.navigation`**:
  - `costmap`: the from-scratch A*/Voronoi planner and waypoint controller.
  - `pointnav`: ASCENT's frozen depth + point-goal policy; sensor-only.
  - `navmesh`: Habitat's ground-truth `ShortestPathFollower`. This is
    privileged navigation and its SR/SPL must not be compared with sensor-only
    methods. `agent.use_habitat_navmesh` remains a compatible alias.
- **Policy — `agent.policy`**: `nav_agent` is the OSG state machine and dynamic
  world model; `ascent` keeps OSG maps with ASCENT-style control flow;
  `ascentnav` uses the alternative ASCENT map/control pipeline under
  `src/ascentnav/`.
- **Exploration — continuous sweep** (`exploration=sweep`): nearest frontier +
  a momentum bonus that prefers frontiers ahead of the heading, so the agent
  sweeps continuously instead of ping-ponging. **LLM-free** (the LLM frontier
  scorer was found redundant on single-floor).
- **Verification — forced-choice VLM** (`verification=nim`, default): shows the
  VLM the whole frame with the target boxed and makes it pick the category from
  the goal list; rejects detector mislabels (e.g. a stool detected as a chair)
  and unreachable / non-goal instances, then keeps exploring.

> **Status (2026-09):** the default config is the S71 arm — ASCENT's control
> flow (`src/ascentnav/`) on ASCENT's served perception models, sensor-only —
> at **63.0% SR / 0.36 SPL on `scenes20_ep0to4`** (100 HM3D v1 episodes);
> native ASCENT scores 65.0% / 0.36 on the same episodes, the previous port
> 54.0%. Same-floor 75.6%, cross-floor 18.2%.
>
> **[docs/USAGE.md](docs/USAGE.md)** is the how-to: model servers, running,
> comparing against ASCENT, variants. **[docs/AB_RESULTS.md](docs/AB_RESULTS.md)**
> (S71) has the diagnosis and the paired result; **[docs/INVESTIGATION.md](docs/INVESTIGATION.md)**
> and **[docs/MULTI_FLOOR.md](docs/MULTI_FLOOR.md)** the earlier work. Note from
> that work: **runs are not reproducible while the VLM verifier is on** — the
> default has it off; use `verification=off` on legacy presets for any A/B
> meant to prove two configs equivalent.

## Quick start

Build the image, start the containers, and drop into a shell — everything
after that runs **inside the `nav` container** (PYTHONPATH, NVIDIA_API_KEY,
etc. are already set by the image/compose env, so no `docker exec` prefix or
`PYTHONPATH=src` is needed below).

```bash
cp docker/.env.example docker/.env
# Set the two collector paths and LOCAL_UID=$(id -u), LOCAL_GID=$(id -g).
docker compose -f docker/compose.yaml --env-file docker/.env build nav
docker compose -f docker/compose.yaml --env-file docker/.env up -d
docker exec -it docker-nav-1 bash       # habitat conda env is active through the entrypoint
```

Compose mounts the collector's `data/` and `outputs/dualmap_authoring/`
directories read-write at `/datasets/habitat-data-collector/...`, so `nav` can
use and update the collector data directly. The host-matched UID/GID keeps new
files accessible outside Docker without a second user setup. Generated episode
manifests and experiment artifacts still default to this repository's ignored
`outputs/` directory. The entrypoint repairs ownership only for `outputs/` and
the named detector-weight volume; it does not recursively chown collector data.

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
python scripts/download_weights.py --pointnav            # ASCENT sensor-only mover
python scripts/download_weights.py --rednet              # ASCENT stair segmentation
python scripts/download_weights.py --clip                # ASCENT/value-map image-text model
```

See `data/README.md` for the full split layout. LLM/VLM defaults to
**NVIDIA NIM** (hosted, `NVIDIA_API_KEY` in `.env`); use `llm=ollama` for a
local model instead.

### Running an eval

```bash
bash scripts/serve_perception.sh                         # the five ASCENT model servers (once)
python scripts/run_eval.py                               # the default: S71 on scenes20_ep0to4 (100 eps)
python scripts/run_eval.py eval.num_episodes=3           # smoke
python scripts/run_eval.py +experiment=final_sensor      # the full v1 val split
python scripts/run_eval.py +experiment=matched_single_floor          # a legacy preset (old base, see docs/USAGE.md)
python scripts/compare_ascent_osg.py relative_work/ascent/debug/behaviour_100 outputs/<run>   # paired vs ASCENT
```

The default needs the servers and a local ollama with `qwen2.5:7b`; it refuses
to start if any is down. See [docs/USAGE.md](docs/USAGE.md).

### Authored YCB benchmark

`+experiment=ycb_authored_nav` discovers authored layouts at runtime, creates
and caches target-visible ObjectNav episodes, injects all authored rigid objects
after every Habitat reset, and uses YOLOE-11l. The current `00829` scene is a
fixture, not a hard-coded preset: adding another complete authored scene makes
it available immediately.

```bash
# Download both explicit detector profiles once (inside docker-nav-1).
python scripts/download_weights.py --profile large
python scripts/download_weights.py --profile small

# All complete scenes and their static layouts (default).
python scripts/run_eval.py +experiment=ycb_authored_nav

# One scene, or an arbitrary subset.
python scripts/run_eval.py +experiment=ycb_authored_nav \
  'ycb.scenes=[00829-QaLdnwvtxbs]'
python scripts/run_eval.py +experiment=ycb_authored_nav \
  'ycb.scenes=[00829-QaLdnwvtxbs,00900-FutureScene]'

# A specific dynamic relocation slot.
python scripts/run_eval.py +experiment=ycb_authored_nav \
  'ycb.layout_types=[in_anchor]' 'ycb.layout_indices=[2]'

# Prepare/validate manifests without running the navigation agent.
python scripts/prepare_ycb_episodes.py +experiment=ycb_authored_nav

# Explicit memory fallback; this is never selected silently.
python scripts/run_eval.py +experiment=ycb_authored_nav detector=yoloe_small
```

Valid layout types are `static`, `in_anchor`, and `cross_anchor`. Wildcard
selection skips incomplete scene directories and records the reason in
`summary.json`; explicitly selecting an incomplete scene, layout type, or slot
fails with an actionable error. Cache keys include the scene, layout type/index,
generator settings, seed, and layout SHA-256, so editing an authoring JSON
automatically regenerates its manifest. Each layout starts with a fresh scene
graph; memory is not carried between static and dynamic layouts.

For the combined benchmark, `+experiment=ycb_dynamic_multifloor` emits only
authored relocations whose prior/static floor differs from the destination and
samples every start on the prior floor. It keeps the OSG `nav_agent`, stale-map
presence beliefs and container posterior, but gives every storey its own map.
Build the static maps in a separate pass (the combined preset intentionally
filters its manifests to relocations):

```bash
python scripts/run_eval.py +experiment=ycb_dynamic_multifloor \
  'ycb.cross_floor_relocations_only=false' 'ycb.layout_types=[static]' \
  ycb.map_out=outputs/static_maps
python scripts/run_eval.py +experiment=ycb_dynamic_multifloor \
  ycb.map_in=outputs/static_maps
```

Snapshot schema v2 stores all floors, stairs, connectivity and track floor
keys; v1 single-floor maps still load as floor 0.

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

| group | options (**default** = the S71 arm) |
|---|---|
| `agent` | **`s71`** (ascentnav policy, PointNav mover), `default` (OSG's `nav_agent`) |
| `detector` | **`dfine`** (served, strict), `yoloe` (11l, 640px), `yoloe_small` (11s, 512px) |
| `llm` | **`qwen_local`** (ollama qwen2.5:7b), `nim` (NVIDIA hosted), `ollama` |
| `exploration` | **`s71`** (ASCENT planner + BLIP-2 value map), `llm_text`, `nearest`, `sweep`, `value` |
| `verification` | **`s71`** (off), `nim` (forced-choice VLM), `nim_terminal` (verify at STOP), `off` |
| `scene_graph` | **`place365`**, `default` |
| `eval` | **`scenes20_ep0to4`** (100 v1 eps), `hm3d_val_v1_full`, `hm3d_val` (v2), `hm3d_val_v1`, `hm3d_val_single_floor`, `hm3d_val_mini`, `ycb_authored` |
| `floor` | multi-floor support; all off by default, enabled by `+experiment=full_v1_navmesh` (see docs/MULTI_FLOOR.md) |

Key agent flags (CLI: `agent.<flag>=...`): `navigation`
(`costmap | navmesh | pointnav`), `policy`
(`nav_agent | ascent | ascentnav`), `use_habitat_navmesh` (legacy alias),
`exploration.continuity_weight` (momentum), `verification.choice_mode`.

`configs/experiment/` presets: **`full_v1_navmesh`** (current best — navmesh +
sweep + verify, full v1, 5 eps/scene), `matched_navmesh` (single-floor),
`matched_single_floor`, `matched_old`, `matched_verify`,
`matched_terminal_verify`, `single_floor_navgoal`, `ycb_authored_nav`,
`ycb_dynamic_multifloor`, and the ASCENT A/B presets documented in
[docs/AB_RESULTS.md](docs/AB_RESULTS.md).

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

## Hardware profiles

| | detector | VLM | fits |
|---|---|---|---|
| default (6 GB, RTX 4050 laptop) | `detector=yoloe_small` (11s, 512px) | `qwen2.5vl:3b` | ~4.5 GB |
| report-quality (>=10 GB; verified on RTX 3080) | `detector=yoloe` (11l, 640px) | off or hosted | hardware-dependent |

The global default remains the small profile for general use, while
`+experiment=ycb_authored_nav` deliberately defaults to YOLOE-11l so benchmark
results stay comparable. Select `detector=yoloe_small` explicitly if memory is
tight; the benchmark never falls back silently.

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
- **Navigation** uses one of the three movers above. Only `navmesh` receives
  simulator geometry (`action_to_goal` / `is_reachable`); `costmap` and
  `pointnav` are sensor-only. `src/ascentnav/` is an attributed alternative
  policy, not a replacement for OSG's dynamic hierarchy.
- `configs/` — Hydra groups; `configs/experiment/*` are composable presets.
- `scripts/` — eval entry (`run_eval.py`), data/weights download,
  `analyze_*.py` diagnostics, keyframe/video tools.
- `tests/unit` — synthetic-data tests, no GPU; `tests/integration` — `-m sim`.
