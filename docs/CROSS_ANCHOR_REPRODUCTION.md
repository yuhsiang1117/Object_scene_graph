# Reproducing the cross-anchor headline: it is not in the tree

*2026-09-20. The paper's single-floor cross-anchor result (28/53 = 52.8%, Table I)
does not reproduce from anything the current tree can rebuild. This records what was
tested, what was ruled out, and what is left. **Update 2026-09-27**, at the end: the
original prior maps were recovered and re-run — cross-anchor comes back to 25/53.*

## What reproduces and what does not

Re-run of both benchmarks on the current tree, against the paper:

| | rerun | paper |
|---|---|---|
| Table II multi-floor | **11/25 = 44.0%**, SPL **0.196** | 10/25 = 40.0%, SPL 0.162 |
| Table I static | 51/79 = 64.6% | 64.6% (per scene 69.2/55.6/69.2 vs 65.4/74.1/53.8) |
| Table I in-anchor | 30/54 = 55.6% | 63.0% |
| Table I cross-anchor | **20/53 = 37.7%** | **52.8%** |

The multi-floor arm reproduces. The single-floor dynamic conditions do not: 16
lost against 4 gained on the 107 shared trials, spread across every query.

> **Correction, 2026-10-15:** the multi-floor arm does *not* reproduce. The re-run ran
> layouts re-authored after the paper's run (12 upward relocations where the paper's run had
> none), and 2 of its 11 successes stopped a storey away from the object. On the object's
> storey it is 9/25. `docs/PAPER.md`, Appendix B, *Table II, re-examined*.

## The tree that produced 62/107 is `74e0eab`

`715be7d` (09-13 07:11) is docs-only and its subject records the result —
*"Per-class fused arm on the full 107: 62, +5/-0 against _sensor_v2"*. Its parent
`74e0eab` (09-13 04:00) is therefore the code, and the locked run's shard tag
`r040922` puts its start at 04:09:22, nine minutes later.

## That tree changes nothing

Built 00829's prior map on `74e0eab` itself, same recipe
(`campaign/rebuild_maps.sh`, `LAYOUT_ROOT=outputs/collector_layouts`):

| built by | tracks | label+position signature |
|---|---|---|
| `74e0eab` | 534 | `ef3c22ad1d5c8e14` |
| HEAD | 534 | `ef3c22ad1d5c8e14` |
| what the locked run loaded | **682** | — |

Content-identical, not merely equal in count. Built twice on HEAD: 534 both
times, so the pass is deterministic.

Then 00829's 18 cross-anchor trials on `74e0eab`, using the 534-track map that
tree builds today:

| | result | mean closest approach |
|---|---|---|
| locked run (`74e0eab` + 682 map) | 12/18 | 1.22 m |
| `74e0eab` + 534 map | **10/18** | 1.12 m |
| HEAD + 534 map | **10/18** | 1.12 m |

**`74e0eab` and HEAD agree on 18 of 18 trials.** The agent code is equivalent;
the deficit is the map.

## Ruled out

| candidate | evidence |
|---|---|
| the unified preset / floor group | `_v4_fuse_cls` (the paper's own preset) scores 49/107 today against `_osg_unified`'s 50/107; they agree on 106 of 107 trials. The floor group is worth one episode. |
| `ultralytics` version | `ultralytics==8.3.*` has resolved to 8.3.253 since 2026-01-13; 8.4 came after. Its 09-14 install timestamp is the image rebuild copying files. |
| the YOLOE fp16 dtype fix (`7c2d847`) | 2026-07-15, long before. |
| detector weights | `weights_sha256` identical; the two runs' `config` blocks match bit-for-bit. |
| vocabulary | unchanged, 47 categories. |
| data inputs | `outputs/collector_layouts`, `outputs/recall/recall_model.json`, the HM3D scenes and DualMap's release root are all untouched since 2026-09-09 or earlier. |
| target tracks in the prior | the thinner map is not starving the mechanism: mean target tracks per trial 5.84 -> 5.40, and FEWER trials start with no target track (9 -> 4). |
| detector recall | slightly BETTER today: 0.352 -> 0.381 on in-view keyframes, admitted 0.246 -> 0.261. |

**Retracted.** An earlier reading of this data said the agent now "stops further
away" -- mean closest approach 0.92 m -> 1.13 m across the 107 shared trials,
straddling the 1.0 m success rule. It does not survive subsetting: on the 18
00829 cross-anchor trials the locked run is the one that ends further out
(1.22 m vs 1.12 m). The sign depends on the subset, so there is no last-half-metre
story here.

## What is left

The container. It was rebuilt **2026-09-14 21:02**, after the maps (~09-10) and
after the locked run (09-13 04:09), and the 09-13 commits reworked it
substantially — *"One Dockerfile for everything"*, *"Derive the ascent env from
the one that works, not from a narrative"*, *"Put the base image back to
11.8-devel: it is the part that provably compiled"*. Every other input is
accounted for, and the map build is deterministic given it.

What that buys, measured: 682 -> 534 tracks, which is 86.2 -> 75.0 qualified
support surfaces per trial (-13%). Cross-anchor is the condition that depends
most on the surface inventory -- the mapped pose is empty and the agent must
search *other* surfaces -- and it is the condition that fell furthest
(static ~0, in-anchor -4, cross-anchor -9).

## The one tree not tested here

`mf5arm/` — a second working copy on another machine — reportedly reaches **28/53**
with `bundle code + dualmap_osg_unified` (`mf5arm/outputs/bundle_dualmap_186`), and a
third run, `outputs/unified_dualmap_107`, reaches 27/53. Neither path exists on this
machine, so neither was part of the audit above.

That matters in both directions. It does not contradict the finding — `74e0eab` and
HEAD agree on 18/18 trials **given the same prior map**, and those runs will have used
the 682-track maps — but it does mean the 28/53 code has never been diffed against this
tree. If it differs, it is the only copy. `.gitignore` now excludes `/mf5arm/` so a
`git add -A` cannot sweep a working copy and its outputs into a commit; the rule carries
a warning not to rely on it until that diff has been taken and committed as a branch.

## What would settle it

Rebuild the prior maps under the pre-09-14 image. If they come back at 682 and
cross-anchor returns to ~28/53, the cause is named and the headline is
recoverable. If the image is gone, the options are to re-baseline the paper on
the current environment or to pin an image and treat the locked run as a record
that cannot be re-derived.

Whatever the outcome: `pyproject.toml` pins `ultralytics==8.3.*` and the repo has
no lockfile for the habitat env. A run records its config fingerprint and its
detector's sha256, and neither would have caught this. Recording a `pip freeze`
beside `summary.json` would have.

## Reproducing this investigation

```bash
# the three runs behind the table above
outputs/RERUN/dualmap186            # HEAD, dualmap_osg_unified, 186 trials
outputs/RERUN/control_v4_fuse_cls   # HEAD, _v4_fuse_cls, the 107 dynamic
outputs/RERUN/tree62_xa             # 74e0eab, 00829 cross_anchor, 18 trials
outputs/RERUN/maps_tree62           # 00829's map as 74e0eab builds it today
```

## 2026-09-27 — the original prior maps, re-run

The three pass-1 maps the locked run loaded turned up as an archive (`maps_released/`
with a README: 00829 682 tracks, 00848 800, 00880 618; built 2026-09-10 from
`outputs/collector_layouts`; md5s verified against the README). They were run on the
current tree with both the paper's preset and the unified arm, over all 186 released
trials, verifier on, `MAX_PARALLEL=3`:

| condition | paper (Table I) | rerun 09-20, rebuilt 534-track maps | **rerun 09-27, original maps** |
|---|---|---|---|
| static | 51/79 = 64.6% | 51/79 = 64.6% | 42/79 = 53.2% |
| in-anchor | 34/54 = 63.0% | 30/54 = 55.6% | **35/54 = 64.8%** |
| cross-anchor | 28/53 = 52.8% | 20/53 = 37.7% | **25/53 = 47.2%**, SPL 0.073 |

Cross-anchor per scene, against the archive's README of what these maps scored in the
locked run: 00829 **11**/18 (was 12), 00848 **5**/18 (5), 00880 **9**/17 (11).

`dualmap_protocol_osg_sensor_v4_fuse_cls` and `dualmap_osg_unified` agree on **186 of 186**
trials, so the two presets are one arm here, as `UNIFIED_PIPELINE.md` says.

What this settles: the prior map is the cause of the cross-anchor deficit, as the section
above argued from 00829 alone. With the original maps the two dynamic conditions are back
within noise of the paper (in-anchor above it, cross-anchor three trials under it, on a
53-trial condition where one trial is 1.9 points and the VLM verifier is on). What it does
not settle: static lost nine trials against the same maps' locked run, which the 09-20
rerun on the rebuilt maps had not — so the environment still differs from the locked one
in something the static condition is sensitive to and the dynamic ones are not.

The maps themselves are not in the repository (`outputs/` is ignored); they are the
`prior_map_<scene>.zip` archives, and the run is `outputs/RERUN_UPLOADMAP/{v4_fuse_cls,osg_unified}`
with per-trial results in `*_trials.json` beside them. To repeat it:

```bash
# the three scene directories under one root, then
ARM=sensor_v4_fuse_cls SPLIT=data/splits/dualmap_186.json \
  MAP_ROOT=outputs/maps_released_upload/maps_released MAP_SUBDIR=1 \
  CONDITIONS=cross_anchor OUT=outputs/RERUN_UPLOADMAP/v4_fuse_cls MAX_PARALLEL=3 \
  bash scripts/run_dualmap_arm.sh
```

(`run_dualmap_arm.sh` takes its trial ids from the split, so with the 186-trial split it
runs all three conditions regardless of `CONDITIONS`; use `data/splits/dualmap_cross_anchor.json`
for the 53 alone.)

### The three lost trials, taken apart (2026-09-28)

Against the locked artifact (`outputs/osg_v4_fuse_cls_full`, the 107 dynamic trials),
today's run with the same maps loses 3 cross-anchor trials and gains none:

| trial | locked | today |
|---|---|---|
| 00829 `0128-2` cracker box | success, 445 steps, 0.39 m | fail, 500 steps, **0.99 m** |
| 00880 `0128-2` blue pitcher | success, 417 steps, 0.65 m | fail, 500 steps, 15.5 m |
| 00880 `0129-1` tin can | success, 128 steps, 0.84 m | fail, 500 steps, 8.1 m |

Everything the setup could have missed was tested and ruled out:

- **Config.** The full composed `dualmap_protocol_osg_sensor_v4_fuse_cls` at `74e0eab` and at
  HEAD differ in no shared key; the 96 keys that exist only at HEAD are the robot / climb /
  obstacle-map fields and all default off (the two that touch this arm, `giveup_after_steps`
  15 / `giveup_min_progress_m` 0.2, are the old hard-coded values made configurable).
- **Map.** Same files (md5), same 682 / 618 tracks; both trees restore the stored occupancy.
- **Run-to-run noise and load.** The three trials re-run at HEAD three times (twice at
  `MAX_PARALLEL=3`, once at 1) are **bit-identical** to each other and to the full run —
  same step counts, same final distances. The hosted-LLM calls do not perturb these trials.
- **Code.** The same three trials on a worktree of `74e0eab` (the locked commit) are
  bit-identical to HEAD's. `74e0eab` reproduces the locked run **only in the locked image**.
- **GPU.** Both the locked run and today's ran on the RTX 4090 (PAPER.md §IV-A).

What is left is the image. Between the locked run (09-13 04:09) and the rebuild (09-14 21:02)
the base went from `nvidia/cuda:12.1.1-cudnn8-runtime` to `11.8.0-cudnn8-devel` and
`opencv-python-headless` (unpinned) became `opencv-python==4.10.0.84`; the EGL/GL userland
habitat-sim renders through changed with the base. The fingerprint fits a float-level
perturbation of the perception path, not a logic change: in the pitcher trial the first
mismatch is a track centre that differs in the **fifth decimal** (−13.833261 vs −13.833272)
at step 111, which moves one surface's search prior from 0.9138 to 0.9259 and flips the
surface chosen at step 116; in the cracker-box trial the trajectories are identical for 262
steps and part at the frontier picked next; in the tin-can trial the approach path is
consumed 13 steps sooner (32 vs 45), so the close look, the disbelief and everything after
it land differently.

Across all 107 dynamic trials the perturbation is small and the outcome mostly survives it:
cross-anchor — 31 of 53 trajectories **bit-for-bit identical** to the locked run, 17 diverge
(13 of them only in the last half of the episode), outcomes agree on 50/53; in-anchor — 38
of 54 identical, outcomes agree on 53/54 (one gained). The condition that loses is the one
where a single late surface/frontier choice decides the episode.

So: the number the paper prints (28/53) is what this code, this map and the 09-13 image
produce; the same code and map in the current image produce 25/53. Neither is "the" number
— the honest statement for Table I cross-anchor is 25–28 of 53 (47–53 %) against DualMap's
16/53, with the image pinned as part of the experimental record from here on
(`docker/Dockerfile` at `0cfecec` onward). Static cannot be diffed the same way: the locked
artifact holds only the 107 dynamic trials, and no 79-trial static run from the locked image
exists on this machine.
