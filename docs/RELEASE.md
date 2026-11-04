# The revision's data release

One directory (and its `.tar.gz`) holding every run measured for the ICRA 2027 revision, built
by `scripts/release/package.py` from `outputs/`. It is enough to **re-derive every table**
without a GPU or a simulator, and, with the HM3D scenes and this repository, to **re-run any
arm** at the code it ran on. This file is the bundle's README; the build appends an inventory.

```bash
python scripts/release/package.py --out outputs/release_<date> \
    --session-scripts outputs/campaign_scripts/session_2026-10 --tar
```

## Layout

| path | what |
|---|---|
| `MANIFEST.csv` | One row per run: section, arm or method, seed, role, preset, the code it ran on and how that was checked, episodes, successes (on the object's storey), wall hours, path here, path under `outputs/`. |
| `SHA256SUMS` | `sha256sum -c SHA256SUMS` from the bundle root. |
| `REPRODUCE_TABLES.sh` | Every table below, re-derived from this bundle: `REPO=<checkout> bash REPRODUCE_TABLES.sh`. |
| `ENVIRONMENT.txt` | Commit, interpreter, GPU, `pip freeze` of the evaluation env. |
| **`benchmark_v2/`** | The multi-floor moved-object set (`docs/BENCHMARK_V2.md`): 5 scenes, 73 episodes per seed, seeds 42/43/44. |
| `runs/<arm>_s<seed>/` | `scripts/run_mfv2.sh`'s layout: per scene `episodes.jsonl` (every diagnostic of every episode), `summary.json`, `timing.csv`, and `<scene>.log`; `<arm>_s<seed>.out` is the lane's console log. `osg_vet_live_no_twins_s<seed>/` holds one directory per episode: that ablation ran only where its rule fires. |
| `checks/` | `rerun_ee43aba/`: the final arm's seed 42 re-run on the ablations' code (73/73 identical step counts, 0 flips). `profile/`: the speed-up's profiling and bit-identity runs. |
| `inputs/` | What a pass-2 re-run reads: the pass-1 prior maps (`maps_v2_ascent` obstacle maps, `maps_v2_osg` scene graphs), the affinity caches, the authored layouts as JSON (renders left out; each run's `summary.json` records every layout's sha256), the map-coverage audit. |
| `campaign/` | How the campaign was driven: the lane queue logs (which snapshot each job started on), the chain scripts and logs, the done-lists of the one-scene-per-call launchers. |
| `tables/` | Re-derived at build time: `arms.txt` / `arms_by_seed.csv` (every arm x seed), `ablations_final_arm.txt`, `ablations_previous_final_arm.txt`, `wall_time.txt`, `multifloor_final_vs_ascent.md`. |
| `report/` | The 2026-10-25 report as sent: `REPORT.md`, `metrics/`, `charts/`, `img/`, `video/`. |
| **`dualmap_release/`** | DualMap's released benchmark (186 trials, 3 scenes). |
| `runs/` | MD-SG: `mdsg_paper_artifact` (Table I's dynamic rows), `mdsg_rerun_original_maps`, `mdsg_rerun_rebuilt_maps`. DualMap's own runner: `dualmap_rerun_oct`, `dualmap_rerun_nov`. The static-map baselines on the shared navigator: `khronos`, `clio_batch` (+ `clio_batch_cd`, change detection on), `hovsg`, `conceptgraphs`; each `seed<k>/trials/<trial>/result.json`. |
| `rankings/` | Each method's `rankings.json` per scene, the one interface between a map and the navigator (`scripts/baselines/README.md`); `mdsg_original` / `mdsg_rebuilt` are generated from the prior maps. |
| `prior_maps/` | MD-SG's prior maps: `original` (the paper's, as uploaded), `rebuilt` (re-mapped in this repo). |
| `tables/` | `table_I.md` (SR by scene and condition), `retrieval_quality.txt` (top-1 / top-3 within 1 m over the 79 static queries), and `as_reported/` the tables as first written. |
| **`provenance/`** | `code_provenance.csv` (run -> snapshot -> commit), `patches/` (each uncommitted snapshot as a patch on its nearest commit), `code_<commit>.tar.gz` (the tree the bundle was built from), `session_scripts/` (the orchestration scripts as run, with this machine's paths). |

## Re-deriving the numbers

```bash
sha256sum -c SHA256SUMS --quiet
REPO=/path/to/Object_scene_graph bash REPRODUCE_TABLES.sh
```

The Benchmark v2 tables need only `episodes.jsonl`. The DualMap tables also read the release's
layouts (`osg.eval.dualmap_release`, `OSG_DUALMAP_RELEASE_ROOT`) for the targets' positions.

Every ablation is paired against the final arm (`osg_landing_vet_live`, 136/219) episode by
episode, and the sign test is over the discordant pairs. Pairing is sound because a seed's run
repeats: the final arm's seed 42, re-run on other code, reproduced all 73 step counts and every
success (`checks/rerun_ee43aba`). The one nondeterministic component, the absence VLM, is
called 38 times in 219 episodes and changes no outcome (`osg_vet_live_no_vlm` is identical).

## Re-running an arm

1. The code. `MANIFEST.csv`'s `code` column names the snapshot (`outputs/campaign_trees/<name>`)
   each run started on. A hex name is a commit: `git checkout <commit>`. A `wip…` name was an
   uncommitted snapshot: check out the commit in `code_provenance.csv` and
   `git apply provenance/patches/<name>.patch`. When two snapshots are listed the run was
   resumed; `run_mfv2.sh` skips every scene that already has a `summary.json`.
2. The inputs. Copy `inputs/affinity_cache*.json` to `outputs/`, and point the runner at the
   bundle's prior maps:
   ```bash
   OBS=<bundle>/benchmark_v2/inputs/prior_maps/maps_v2_ascent \
   OSG=<bundle>/benchmark_v2/inputs/prior_maps/maps_v2_osg \
   ROOT=outputs/mfv2_rerun bash scripts/run_mfv2.sh osg_landing_vet_live 42
   ```
   Scenes and objects come from HM3D and the collector's layouts as in `docs/SETUP.md`; the
   layouts must hash to the `layout_sha256` each `summary.json` records.
3. Compare: `python scripts/analyze_mfv2.py ablations --root outputs/mfv2_rerun ...`, or diff
   the step counts per episode (`checks/rerun_ee43aba` is that comparison for seed 42).

## What is not in it, and why

| left out | size | how to get it back |
|---|---:|---|
| per-step keyframes and `viz/` of every run | ~240 GB | re-run; the runner writes both |
| the baselines' raw maps (Khronos, Clio, HOV-SG, ConceptGraphs) | 239 GB | `scripts/baselines/README.md`, per method |
| HM3D scenes, YCB objects, model weights | licensed | `docs/SETUP.md` |
| the collector's layout renders | HM3D renders | the collector |
| the git history | | the repository; `--git-bundle` ships it, with the commits' author names and e-mails |

## Secrets

Every run's `summary.json` records the config, the VLM endpoint's API key among it. The
packager redacts on the way in -- API-key-shaped strings, `*api_key` / `*token` / `*secret` /
`*password` fields, every secret value in `docker/.env` and the environment -- then scans every
byte of the finished bundle for those values and refuses to finish on a hit. It prints how many
redactions of each kind it made, never a value.

## Known gaps

- **Unrecorded snapshots.** A run whose start no lane log or launcher recorded says
  `unrecorded` in `MANIFEST.csv`; its preset is still known, and its date bounds the commit.
- **The stale-twin ablation** ran from the worktree on 2026-10-29 before `190ad3f` committed its
  preset; no snapshot of that working tree was kept.
- **MD-SG's Table I rows** are `run_eval.py` outputs (`episodes.jsonl`), not `result.json`
  trials, so `table_I.md` lists DualMap and the static-map baselines only;
  `docs/PAPER.md` Appendix B traces MD-SG's cells to these runs.
- `report/img/` and `report/video/` are renders of HM3D scenes. Check the HM3D terms before
  putting them anywhere public.
