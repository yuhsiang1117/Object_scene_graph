# Object Scene Graph ObjectNav

Open-vocabulary, object-goal navigation on HM3D ObjectNav (Habitat). A
from-scratch Python `osg` package: open-vocab perception → a `building → room →
object` 3D scene graph → frontier exploration → approach + verify. The pipeline
has three navigation modes (`agent.navigation`), one of which is sensor-only.

The current pipeline (see **[docs/AB_RESULTS.md](docs/AB_RESULTS.md)** for every
measured A/B and the decision taken):

- **Perception — YOLOE** (open-vocab detection **and** segmentation in one
  model; no SAM) → ellipsoid object layer (dual-quadric + Wasserstein refine).
- **Navigation — `agent.navigation`**, three movers:
  - `pointnav` — ASCENT's own: a frozen PointNav ResNet reading `(rho, theta)` +
    depth. **Sensor-only**, and therefore the only mode whose numbers are
    comparable to ASCENT's published SR. `+experiment=final_sensor`.
  - `navmesh` — Habitat's `ShortestPathFollower` on the ground-truth navmesh.
    Fewer failure modes, but it is **privileged information**: every number
    taken on it carries that caveat. Spelled `agent.use_habitat_navmesh: true`
    in the older presets, which still works.
  - `costmap` — the from-scratch A*/Voronoi planner + waypoint controller.

  The costmap is built in every mode, since frontier extraction and the scene
  graph need it; only path planning and execution change.
- **Exploration — continuous sweep** (`exploration=sweep`): nearest frontier +
  a momentum bonus that prefers frontiers ahead of the heading, so the agent
  sweeps continuously instead of ping-ponging. **LLM-free** (the LLM frontier
  scorer was found redundant on single-floor).
- **Verification — forced-choice VLM** (`verification=nim`, default): shows the
  VLM the whole frame with the target boxed and makes it pick the category from
  the goal list; rejects detector mislabels (e.g. a stool detected as a chair)
  and unreachable / non-goal instances, then keeps exploring.

> **Status (2026-08):** working toward parity with **ASCENT** (arXiv:2505.23019),
> which scores **63% SR** on HM3D ObjectNav v1 val at `success_distance=0.1`,
> 500 steps, sensor-only.
>
> Historical numbers in this repo (`+experiment=full_v1_navmesh`: 42% on 100
> episodes, single-floor 68.6% / multi-floor 27.7%) were measured under
> **looser** conditions — `success_distance=0.18` and habitat's ground-truth
> navmesh — so they are **not** comparable to that 63%.
> `+experiment=ascent_matched` is the aligned protocol; every new number is
> taken there. **Multi-floor is the dominant remaining loss** — the 2D costmap
> collapses floors. See **[docs/AB_RESULTS.md](docs/AB_RESULTS.md)** for the
> staged plan, every A/B, and the decision taken.

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
— `viz/debug/ep<ID>.mp4` (per-step RGB+segmentation | costmap).

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

Key agent flags (CLI: `agent.<flag>=...`): `navigation`
(`costmap` | `navmesh` | `pointnav`; `use_habitat_navmesh` is the older spelling
of `navmesh` and still works), `exploration.continuity_weight` (momentum),
`verification.choice_mode`.

`navigation=pointnav` needs its weights staged once:
`python scripts/download_weights.py --pointnav`.

`configs/experiment/` presets: **`full_v1_navmesh`** (current best — navmesh +
sweep + verify, full v1, 5 eps/scene), `matched_navmesh` (single-floor),
`matched_single_floor`, `matched_old`, `matched_verify`,
`matched_terminal_verify`, `single_floor_navgoal`.

### Analysis & debugging

The `scripts/analyze_*.py` tools decompose a run's `episodes.jsonl`:
`analyze_stages.py` (explore vs approach failure), `analyze_localization.py` /
`analyze_trackloc.py` (stop-pose / mapped-object vs GT), `analyze_approach.py`
(why the terminal approach failed). Rich per-episode fields include
`state_log`, `frontier_select_log` (every frontier choice: step, agent xy,
chosen frontier, path cost), `approach_diag`, and `verify_calls`.

With `eval.debug_frames=true` a run also writes:
- `viz/debug/ep<ID>.mp4` — per-step **RGB + YOLOE segmentation | costmap** (with
  the chosen frontier and planned path drawn), and
- `verify_debug/` (when a verifier is active) — the exact **image sent to the
  VLM** (whole frame + red box) plus `index.jsonl` with the VLM's response and
  accept/reject per call.

See **[docs/AB_RESULTS.md](docs/AB_RESULTS.md)** for the full story.

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
  refinement, linking) → `mapping` (costmap, frontiers, room watershed) →
  `graph` (building/room/object hierarchy + LLM serialization) →
  `exploration` (scorers incl. `NullScorer` for geometric, momentum/info-gain
  selector) → `planning` (A*, waypoint controller, and the vendored PointNav
  mover under `planning/pointnav/`) → `verification` (forced-choice VLM
  verifier) → `agent` (FSM) → `sim` (Habitat env + `ShortestPathFollower`
  navmesh driving) / `eval`.
- **Navigation** is one of the three movers above. Only `navmesh` touches the
  simulator's geometry (`sim/habitat_env.py`: `action_to_goal`, `is_reachable`);
  `pointnav` and `costmap` receive neither handle. See S8 in
  [docs/AB_RESULTS.md](docs/AB_RESULTS.md).
- `configs/` — Hydra groups; `configs/experiment/*` are composable presets.
- `scripts/` — eval entry (`run_eval.py`), data/weights download,
  `analyze_*.py` diagnostics, keyframe/video tools.
- `tests/unit` — synthetic-data tests, no GPU; `tests/integration` — `-m sim`.
