# Object Scene Graph ObjectNav

Object-goal navigation in Habitat, sensor-only. Two benchmarks live here:

| | task | policy | measured |
|---|---|---|---|
| **HM3D ObjectNav** | find a `chair`/`bed`/`toilet`/`sofa`/`plant`/`tv_monitor` in an unseen house | `ascentnav` — a line-cited transcription of ASCENT's control flow | **63.0% SR / 0.36 SPL** on 100 v1 episodes (native ASCENT: 65.0% / 0.36) |
| **Authored YCB dynamic scenes** | find an object that has been *moved* since the map was built | `nav_agent` — OSG's scene graph, presence beliefs and container posterior | see [docs/DYNAMIC_SCENES.md](docs/DYNAMIC_SCENES.md) |

> **Read this if nothing else.** The default configuration changed in
> September 2026. `python scripts/run_eval.py` with no arguments no longer runs
> OSG's own agent with YOLOE and a hosted VLM verifier — it runs the **S71
> arm**: ASCENT's control flow on ASCENT's five served perception models, on
> the 100-episode HM3D split. It needs those five servers and a local ollama,
> and it refuses to start without them. See [docs/SETUP.md](docs/SETUP.md).

---

## Start here

| you want to… | read |
|---|---|
| install it (submodule, CUDA extension, ~5 GB of weights, datasets) | **[docs/SETUP.md](docs/SETUP.md)** |
| run it, compare against ASCENT, try a variant | **[docs/USAGE.md](docs/USAGE.md)** |
| know what is measured and how claims are established | **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** |
| see every A/B and why it won or lost | **[docs/AB_RESULTS.md](docs/AB_RESULTS.md)** |
| the dynamic-scene benchmark | **[docs/DYNAMIC_SCENES.md](docs/DYNAMIC_SCENES.md)**, [docs/MULTI_FLOOR.md](docs/MULTI_FLOOR.md) |

```bash
git submodule update --init --recursive     # REQUIRED: the build COPYs it
cp docker/.env.example docker/.env
printf 'UID=%s\nGID=%s\n' "$(id -u)" "$(id -g)" >> docker/.env   # or the mount is unwritable
$EDITOR docker/.env                         # point HM3D_SCENES at the scene meshes
docker compose -f docker/compose.yaml --env-file docker/.env build nav   # both conda envs
docker compose -f docker/compose.yaml --env-file docker/.env up -d
docker exec -it docker-nav-1 bash           # everything below runs in here

bash scripts/fetch_ascent_weights.sh        # ~4.6 GB
python scripts/download_weights.py --pointnav --rednet
docker exec docker-ollama-1 ollama pull qwen2.5:7b

bash scripts/serve_perception.sh            # five model servers, ~60 s
python scripts/run_eval.py eval.num_episodes=1      # one live episode
```

One image, **two conda envs**: `habitat` (habitat-sim, OSG, YOLOE — active on
shell entry) and `ascent` (BLIP-2, MobileSAM, GroundingDINO, RAM++, D-FINE).
They cannot be one interpreter — habitat-sim pins numpy < 1.24 and lavis needs
a different transformers — which is why the models are served over HTTP.

**`--recursive` is mandatory, and before the build.** The Dockerfile COPYs
`relative_work/ascent` to compile GroundingDINO's CUDA kernel, so the build
fails without it; and at run time a non-recursive checkout leaves nine empty
directories and servers that import nothing.

---

## The HM3D pipeline (the default)

`src/navigation/` is a transcription of ASCENT's `Ascent_Policy.act` and the
`Map_Controller` it drives, on ASCENT's vendored maps, with every rule cited to
its reference line. It replaced a port that scored 54.0% where the reference
scored 65.0%; a paired trace diagnosis showed the gap was **before the target
was ever in frame**, not in the models — so the fix was control flow, not
tuning. `docs/AB_RESULTS.md` S71 has the full story.

Per step: hole-filled depth → object map (D-FINE at 0.8, one MobileSAM mask per
detection, *every* detection ingested) → obstacle map and the stair state
machine → one BLIP-2 cosine for the value map → dispatch (stairs → pitch →
13-turn opening scan → explore → navigate).

Five models are served over HTTP from the `ascent` env, because BLIP-2's
`lavis` and habitat-sim cannot share an interpreter. Process-per-model is
ASCENT's own architecture, not a workaround.

| port | model | role |
|---|---|---|
| 13182 | BLIP-2 ITM | value map, and the commit gate that latches at ≥ 0.15 |
| 13183 | MobileSAM | one mask per detection box |
| 13184 | GroundingDINO | `"stair ."` ≥ 0.60, ANDed with RedNet |
| 13185 | RAM++ | scene tags for the LLM prompt |
| 13186 | D-FINE | closed-set COCO detector |

> **Run one evaluation at a time.** The servers cache per-image state on the
> model object and are not reentrant; two concurrent evals race and one gets an
> HTTP 500. This killed two multi-hour runs. The client retries once, which
> covers a transient hit, not sustained concurrency.

The pipeline **fails loud**: all five servers are probed before Habitat loads,
and one that stops answering raises `PerceptionUnavailable` instead of
returning a neutral value. That is deliberate — a silently unreachable BLIP-2
scores 0 every step, which under ASCENT's gate is an agent that never stops,
and a 0% run would look like a bad algorithm.

---

## Running

```bash
python scripts/run_eval.py                                # default: 100 episodes, ~2.5 h
python scripts/run_eval.py eval.behaviour_log=true        # + per-step trace
python scripts/run_eval.py output_dir=outputs/my_run
bash scripts/run_full_split.sh                            # full v1 val (2000 eps), supervised
python scripts/compare_ascent_osg.py data/reference/ascent_behaviour_100 outputs/my_run
```

`outputs/<run>/` holds `episodes.jsonl` (per-episode metrics, `agent_stats`
mechanism counters, `state_log`, `giveup_log`, and the full `step_trace` when
`behaviour_log=true`), `summary.json` (config + algorithm inventory) and
`timing.csv`.

The comparison scores both runs against the dataset's own object positions —
geometry neither agent sees — and reports the metrics that actually separate
arms: SAW episodes, never-saw→STOP, climb share on same-floor episodes,
earliest STOP, P(success | committed). **A change that moves SR without moving
one of those is noise at n=100.**

### The dynamic-scene benchmark

`+experiment=ycb_authored_nav` discovers authored layouts at runtime, creates
and caches target-visible episodes, and injects the authored rigid objects after
every Habitat reset. Layout types are `static`, `in_anchor`, `cross_anchor`.

```bash
python scripts/run_eval.py +experiment=ycb_authored_nav
python scripts/run_eval.py +experiment=ycb_authored_nav 'ycb.scenes=[00829-QaLdnwvtxbs]'
python scripts/run_eval.py +experiment=ycb_authored_nav \
  'ycb.layout_types=[in_anchor]' 'ycb.layout_indices=[2]'
python scripts/prepare_ycb_episodes.py +experiment=ycb_authored_nav   # manifests only
```

Wildcard selection skips incomplete scene directories and records why in
`summary.json`; naming an incomplete scene explicitly is an error. Cache keys
cover the scene, layout, generator settings, seed and layout SHA-256, so editing
an authoring JSON regenerates its manifest. Each layout starts with a fresh
scene graph — memory is never carried between static and dynamic layouts.

`+experiment=ycb_dynamic_multifloor` emits only relocations that cross a floor
and starts every episode on the prior floor. Build the static maps first:

```bash
python scripts/run_eval.py +experiment=ycb_dynamic_multifloor \
  'ycb.cross_floor_relocations_only=false' 'ycb.layout_types=[static]' \
  ycb.map_out=outputs/static_maps
python scripts/run_eval.py +experiment=ycb_dynamic_multifloor ycb.map_in=outputs/static_maps
```

---

## Configuration

Hydra groups under `configs/`; override on the CLI (`group=name`) or compose a
preset (`+experiment=name`).

| group | options (**bold** = default) |
|---|---|
| `agent` | **`s71`** (ascentnav policy, PointNav mover), `default` (OSG's `nav_agent`) |
| `detector` | **`dfine`** (served, strict), `yoloe` (11l open-vocab), `yoloe_small`, `yolo_coco` |
| `exploration` | **`s71`** (ASCENT planner + BLIP-2 value map), `value`, `sweep`, `nearest`, `llm_text` |
| `verification` | **`s71`** (off), `nim` (forced-choice VLM), `nim_terminal`, `off` |
| `llm` | **`qwen_local`** (ollama qwen2.5:7b), `nim`, `ollama` |
| `scene_graph` | **`place365`**, `default` |
| `eval` | **`scenes20_ep0to4`** (100 v1 eps), `hm3d_val_v1_full`, `hm3d_val_v1`, `hm3d_val`, `hm3d_val_single_floor`, `hm3d_val_mini`, `ycb_authored` |

Eleven presets remain: `ascentnav` (the default by name),
`ascentnav_union_stairs` (the stair A/B), `final_sensor` (full split), and
seven `ycb_*` dynamic arms. The S8–S70 port-chain presets were removed in the
September 2026 trim; their numbers stand as recorded in `docs/AB_RESULTS.md`
and re-running them means recovering the yaml from git history.

Three navigation modes exist (`agent.navigation`): `pointnav` (ASCENT's frozen
policy, sensor-only, the default), `costmap` (the from-scratch A*/Voronoi
planner), and `navmesh` (Habitat's ground-truth follower — **privileged**, its
SR/SPL must never be quoted against sensor-only methods).

---

## Layout

- `src/navigation/` — **the default HM3D policy.** `agent.py` (dispatch,
  `_navigate`, `_explore`), `stairs.py`, `planner.py`, `perception.py`,
  `depth_filter.py`, `geometry.py`, and `mapping/` (ASCENT's obstacle, value
  and object-cloud maps, vendored).
- `src/osg/` — the rest of the pipeline, and all of the dynamic-scene agent:
  `perception` → `objects` (ellipsoid layer) → `mapping` (per-floor costmaps,
  floors, frontiers, portals) → `graph` (floor/room/container/object) →
  `exploration` → `planning` → `verification` → `agent` → `sim` / `eval`.
- `relative_work/ascent/` — the ASCENT reference, a **submodule** pinned to
  upstream `8f7bbf9`, unpatched. The five model servers live here.
- `data/reference/ascent_behaviour_100/` — ASCENT's recorded 100-episode trace
  (65.0%), the baseline every comparison is scored against.
- `configs/` — Hydra groups; `configs/experiment/*` are composable presets.
- `scripts/` — `run_eval.py`, `run_full_split.sh` (supervised long run),
  `serve_perception.sh`, `compare_ascent_osg.py`, `analyze_*.py` diagnostics.
- `tests/unit` — 949 tests, no GPU or data needed (~20 s). Every
  `test_ascentnav_*` names the reference line it was transcribed from, and
  `tests/unit/golden/` pins every preset's composed config so a default cannot
  move unnoticed.

## Hardware

Developed on one 24 GB GPU: the five servers take ~11 GB and habitat plus
RedNet ~2.5 GB, leaving room for exactly one evaluation. The LLM and VLM are
queried asynchronously — the control loop never blocks on them; decision
latency is reported separately in `timing.csv`.
