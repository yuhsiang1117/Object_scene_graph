---
name: run-tests
description: Run this repo's tests and read their failures correctly — the CPU unit suite, one file or one test, the Habitat-backed `sim` tests, the two config golden files (and when regenerating them is right versus when it hides a bug), and the no-robot ROS 2 pipeline check. Use after changing code under src/, configs/ or scripts/, before committing, or when a golden/snapshot test goes red.
---

# Running the tests

Everything runs inside the `nav` container (`docker exec -it docker-nav-1 bash`); `PYTHONPATH`
and the habitat env are already set there. For CPU-only work outside it: `pip install -e '.[dev]'`.

## 1. The unit suite — always

```bash
pytest tests/unit -q                                  # ~1535 tests, ~50 s, no GPU or data
pytest tests/unit/test_floors.py -q                   # one file
pytest tests/unit/test_floors.py::test_name -q        # one test
pytest tests/unit -q -k presence                      # by keyword
```

Run the whole suite before a commit, not just the file you touched: the golden tests below
compose every preset and will catch a change you did not think was related.

Pick the focused file from what changed:

| changed | run first |
|---|---|
| `src/navigation/` (the ASCENT transcription) | `pytest tests/unit -q -k ascentnav` — each test names the reference line it pins; a failure means the transcription was broken, do it knowingly or not at all |
| `src/osg/core/config/`, `configs/` | `pytest tests/unit/test_config_snapshot.py -q` (see §2) |
| `src/osg/eval/record.py`, an analyzer under `scripts/` | `pytest tests/unit/test_episode_record_schema.py -q` |
| `src/osg/ros2/`, `scripts/ros2/` | `pytest tests/unit -q -k ros2`, then §4 |
| anything else | `pytest tests/unit -q -k <module name>` |

## 2. The two golden files — read the failure before regenerating

`tests/unit/test_config_snapshot.py` pins:

- `tests/unit/golden/config_snapshot.json` — every calibrated default, flattened. Its failure
  prints `defaults changed (was, is)`, `config fields removed`, or `config fields added`.
- `tests/unit/golden/experiment_fingerprints.json` — a sha256 of what every preset under
  `configs/experiment/` (including `archive/<name>`) composes to.

Decide which case you are in:

- **You changed a default on purpose** (or added a field / preset): regenerate, then make sure
  the golden diff shows exactly that change and nothing else.
  ```bash
  python tests/unit/test_config_snapshot.py   # rewrites BOTH goldens
  git diff tests/unit/golden/
  ```
- **You refactored and did not mean to change behaviour**: do NOT regenerate. A moved number
  is the bug the golden exists to catch — fix the code until the test passes unchanged.
- **A fingerprint changed for a preset you did not touch**: a shared base (`config.yaml`,
  `s71_defaults.yaml`, `legacy_defaults.yaml`, a group YAML) silently redefined an arm that was
  measured. Restore the old composition for that preset (pre-S71 presets list
  `configs/legacy_defaults.yaml` first for exactly this reason) unless redefining it is the point.

Moving a preset into `configs/experiment/archive/` shows up as a rename of its key; that is
the only fingerprint diff such a move should produce.

## 3. Simulator tests — only when Habitat and HM3D are mounted

```bash
python scripts/smoke_habitat.py      # first: one scene, one EGL frame; non-zero exit on failure
pytest -m sim                        # tests/integration
```

If `smoke_habitat.py` fails, the problem is EGL or the dataset mount, not the pipeline — fix
that before reading any `sim` failure.

## 4. The ROS 2 layer, with no robot

```bash
bash scripts/ros2/check_pipeline.sh             # fake robot + bridge in tmux, then the checks
bash scripts/ros2/check_pipeline.sh --loopback  # no ROS at all: the fake robot answers on the socket
bash scripts/ros2/check_pipeline.sh --stop
```

It proves frames convert into the pipeline's world, a goal reaches `NavigateToPose` in ROS
coordinates, discrete actions move the base the right way, and the operator's floor switch
arrives. `docs/ROS2.md` explains each check.

## Do not

- Regenerate a golden to make a red test green without reading the diff it produces.
- "Fix" a `PerceptionUnavailable` in a test or a run by falling back to a default score — a
  silent 0 from BLIP-2 is an agent that never STOPs (see CLAUDE.md).
- Report a test run you did not see finish; quote the `N passed` line.
