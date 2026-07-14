# Object Scene Graph ObjectNav

Real-time object-based hierarchical 3D scene graph (3DSG) for open-vocabulary,
language-guided object navigation, evaluated on HM3D ObjectNav (v2 episodes,
HM3D-semantics v0.2, 6 categories). From-scratch implementation of the RA-L
paper pipeline (`docs/RA_L__2025_.pdf`) with three 2025-era upgrades:

- **A — YOLOE** replaces YOLO-World: open-vocab detection **and** segmentation
  in one real-time model; no SAM stage.
- **B — Multimodal frontier scoring**: the VLM sees keyframe images near each
  frontier in addition to the serialized scene graph text.
- **C — Target verification / last-mile**: approach-viewpoint planning with
  line-of-sight checks + VLM false-positive rejection with blacklisting.

## Quick start

```bash
cp .env.example .env                  # adjust dataset paths if needed
make build                            # nav image (habitat-sim 0.3.1, torch cu121)
make up && make pull-model            # ollama + qwen2.5vl:3b
make smoke                            # M0: EGL rendering + LLM round-trip
make test                             # unit tests (no GPU/data needed)
# one-time: HM3D license + download — see data/README.md
make eval-mini                        # 3-episode end-to-end smoke eval
```

Full eval and ablations:

```bash
docker compose run --rm nav python scripts/run_eval.py eval=hm3d_val
docker compose run --rm nav python scripts/run_eval.py --multirun \
    +ablation=full,no_verify,paper_baseline,no_llm
```

Outputs land in `outputs/<timestamp>/`: `summary.json` (SR/SPL + per-module
FPS + config fingerprint), `episodes.jsonl`, `timing.csv`, `viz/*.png`
(top-down trajectory maps).

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
  `exploration` (scorer ladder + async wrapper + P/d selector) → `planning`
  (A*, waypoint controller) → `verification` (viewpoint + VLM verifier) →
  `agent` (FSM) → `sim`/`eval`.
- `configs/` — Hydra groups; `configs/ablation/*` are one-flag experiment presets.
- `scripts/` — smoke tests, data/weights download, offline pipeline, eval entry.
- `tests/unit` — synthetic-data tests, no GPU; `tests/integration` — `-m sim`.
