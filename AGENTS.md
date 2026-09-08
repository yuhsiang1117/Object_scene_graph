# Repository Guidelines

## Project Structure & Module Organization

The installable Python package lives in `src/osg/`. Keep changes within the existing domain modules: `agent/` coordinates the navigation state machine, `perception/` and `objects/` build object observations, `mapping/` and `graph/` maintain spatial state, and `planning/`, `verification/`, `sim/`, and `eval/` handle their named pipeline stages. Hydra configuration groups live in `configs/`; reusable experiment compositions belong in `configs/experiment/`. Use `scripts/` for evaluation, analysis, data preparation, and visualization entry points. Unit tests are in `tests/unit/`, while Habitat- and dataset-dependent checks are in `tests/integration/`. Treat `outputs/` and downloaded files under `data/` as generated artifacts.

## Build, Test, and Development Commands

Run the supported environment through Docker when using Habitat or GPU models:

```bash
cp docker/.env.example docker/.env
docker compose -f docker/compose.yaml --env-file docker/.env build nav
docker compose -f docker/compose.yaml --env-file docker/.env up -d
```

For lightweight local development, install with `pip install -e '.[dev]'`. Run `pytest tests/unit -q` for the CPU-only suite and `pytest -m sim` only when Habitat and HM3D data are available. Start a small evaluation with `python scripts/run_eval.py eval=hm3d_val_mini`; compose named settings with Hydra syntax such as `+experiment=full_v1_navmesh`.

## Coding Style & Naming Conventions

Follow standard Python conventions: four-space indentation, `snake_case` for modules, functions, and variables, and `PascalCase` for classes. Add type hints to public APIs and keep configuration in typed objects under `src/osg/core/config/` plus matching YAML files. Prefer small, domain-focused modules and explicit Hydra overrides. No formatter or linter is configured, so keep imports organized and match neighboring code.

## Testing Guidelines

Use pytest and name files `test_<behavior>.py` and tests `test_<expected_behavior>`. Add deterministic, synthetic unit coverage for logic changes. Mark environment-heavy tests with `sim` or `gpu`, as declared in `pyproject.toml`. VLM-backed runs are nondeterministic; use `verification=off` when an A/B comparison must be reproducible.

## Commit & Pull Request Guidelines

Recent commits use concise, descriptive subjects, sometimes prefixed by the subsystem (`search: ...`, `nav_reasons: ...`). Keep each commit focused and explain measured behavior when changing navigation or evaluation logic. Pull requests should describe the problem, configuration used, tests run, and relevant SR/SPL or diagnostic changes. Link related issues and include visualizations or debug-frame evidence for behavior that is easiest to review visually.

## Security & Configuration

Keep `NVIDIA_API_KEY`, HM3D credentials, and machine-specific dataset paths in `docker/.env`; never commit secrets. Do not commit generated runs, model weights, or licensed HM3D data.
