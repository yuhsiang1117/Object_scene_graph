# Documentation index

Two kinds of document live here. The **reference** set at this level describes the system as
it is: the paper, the architecture, how to set it up, run it, measure it, and put it on a
robot. Everything under [`history/`](history/) is the record of how it got here — experiment
logs, frozen result tables and superseded design notes, kept because the constants and numbers
of the reference set trace back to them. Being in `history/` is the label: read a file there as
true of its date, not of the current tree. The few that are wrong even as a record of their
date (superseded, never run) also say so at the top.

## Start here

| doc | what it is |
|---|---|
| [PAPER.md](PAPER.md) | The ICRA 2027 submission in full, plus a paper→code map and a number→run provenance table. **The system is MD-SG in the paper and OSG in the code.** |
| [ARCHITECTURE.md](ARCHITECTURE.md) | The OSG pipeline module by module, the four belief mechanisms, why each constant is what it is, and the known limitations. |
| [SETUP.md](SETUP.md) | Clone with submodules, `docker/.env`, weights, the two conda envs. |
| [USAGE.md](USAGE.md) | How to run the pipeline, and what the default arm scores. |

## Measuring it

| doc | what it is |
|---|---|
| [METHODOLOGY.md](METHODOLOGY.md) | How this repo measures ObjectNav, and every difference from native ASCENT. |
| [DYNAMIC_EXPERIMENT_PROTOCOL.md](DYNAMIC_EXPERIMENT_PROTOCOL.md) | The two-pass moved-object protocol: map the static layout, relocate, search from the stale snapshot. |
| [UNIFIED_PIPELINE.md](UNIFIED_PIPELINE.md) | One agent across the three datasets (DualMap's release, the five-scene multi-floor set, the fifteen authored scenes), and the paired run showing the floor group is inert on one storey. |
| [DUALMAP_OFFICIAL_RERUN.md](DUALMAP_OFFICIAL_RERUN.md) | DualMap re-run on its own released benchmark — the source of the 30.2 % cross-anchor figure the paper compares against — and what that benchmark can and cannot resolve. |
| [CROSS_ANCHOR_REPRODUCTION.md](CROSS_ANCHOR_REPRODUCTION.md) | Whether the paper's single-floor cross-anchor headline re-derives, what was ruled out, and the 2026-09-27 re-run on the original prior maps. **Read before quoting Table I.** |
| [ZERO_SHOT_AUDIT.md](ZERO_SHOT_AUDIT.md) | Every piece of hand-authored per-benchmark knowledge in the pipeline. |
| [BENCHMARK_V2.md](BENCHMARK_V2.md) | The multi-floor moved-object set as re-built for the review: generation rules, episode classes and start storeys, the success rule, where objects are moved, and what the prior map holds. |

## Figures and the video

| doc | what it is |
|---|---|
| [FIGURES.md](FIGURES.md) | Every paper figure, its caption, its provenance and the renderer that regenerates it. |
| [VIDEO.md](VIDEO.md) | The ICRA video: storyboard, narration and subtitles, which episode each clip is and why, the real-robot takes, how to rebuild. |

## Running on a real robot

| doc | what it is |
|---|---|
| [ROS2.md](ROS2.md) | The ROS 2 / Nav2 layer for the Stretch 3: why two interpreters, the frame conventions, who owns the base, the manual floor switch, the two-run protocol, and how to check all of it with no robot. |
| [THOR.md](THOR.md) | Where that layer runs: a Jetson AGX Thor beside the robot, two containers, and every knob that can only be settled on the device. |
| [DEMO_FLOW.md](DEMO_FLOW.md) | The two-floor moved-object demo as a checklist of commands: bring-up, the two passes, the carry, the one-line floor switch. |

## History — `history/`

Append-only measurements and the reasoning behind decisions. Large because they are complete.

**The logs, in order**

| doc | |
|---|---|
| [AB_RESULTS.md](history/AB_RESULTS.md) | 236 KB. Every staged change S0–S76 on the ObjectNav / ASCENT-parity line, its measured effect, the decision. |
| [DYNAMIC_SCENES.md](history/DYNAMIC_SCENES.md) | 161 KB. The dynamic-scene design and its experiment log. |
| [AUTHORED_BENCHMARK_LADDER.md](history/AUTHORED_BENCHMARK_LADDER.md) | Conditions A–U on the authored layouts, scored by the 2 m viewpoint rule — the measurements behind many of `ARCHITECTURE.md`'s constants. Moved out of it on 2026-10-15. |
| [DUALMAP_PROTOCOL_EARLY_ARMS.md](history/DUALMAP_PROTOCOL_EARLY_ARMS.md) | Our September arms on DualMap's 186 trials (37.6 % → the tight ring), before region-proposal fusion. Moved out of `DUALMAP_OFFICIAL_RERUN.md` on 2026-10-15. |
| [CROSS_ANCHOR_OBSTACLE_MAP.md](history/CROSS_ANCHOR_OBSTACLE_MAP.md) | Navigating a moved layout over ASCENT's obstacle map — every measurement and every wrong turn. Carries an internal retraction. |
| [CROSS_ANCHOR_STATUS.md](history/CROSS_ANCHOR_STATUS.md) | The summary of the above: where it stands, what was found by layer, what is left. |
| [OVERNIGHT_SR_ITERATION.md](history/OVERNIGHT_SR_ITERATION.md) | Written as it happened, on branch `experiment/sr-70-50`. |
| [SENSOR_V2_107.md](history/SENSOR_V2_107.md) | The `_sensor_v2` arm on the 107 dynamic trials. |

**Design notes** — [MULTI_FLOOR.md](history/MULTI_FLOOR.md) (partly superseded: its
`frontier_cost_free_cell=true` recommendation was reversed with a measurement) ·
[CROSS_FLOOR_CLIMB.md](history/CROSS_FLOOR_CLIMB.md) · [FUSED_PIPELINE.md](history/FUSED_PIPELINE.md) ·
[PROPOSAL_FUSION.md](history/PROPOSAL_FUSION.md) · [ABLATION_FAILURE_MODES.md](history/ABLATION_FAILURE_MODES.md) ·
[APPEARANCE_REID_AND_CANDIDATE_BUDGET_PLAN.md](history/APPEARANCE_REID_AND_CANDIDATE_BUDGET_PLAN.md) ·
[SR_PROPOSAL_CLOSE_LOOK.md](history/SR_PROPOSAL_CLOSE_LOOK.md)

**Frozen results** — one table each, from one run; most of the runs they name are gone from
`outputs/`, so the table is the record — [SENSOR_ONLY_RESULT.md](history/SENSOR_ONLY_RESULT.md) ·
[WHY_THE_SENSOR_ARM_LOSES.md](history/WHY_THE_SENSOR_ARM_LOSES.md) · [CREEP_FIX_RESULTS.md](history/CREEP_FIX_RESULTS.md) ·
[REACHABILITY_CEILING.md](history/REACHABILITY_CEILING.md) · [UNREACHABLE_SET_EXPERIMENT.md](history/UNREACHABLE_SET_EXPERIMENT.md) ·
[RELEASED_MAPS_RESULT.md](history/RELEASED_MAPS_RESULT.md) · [RESIDUE_AFTER_RELEASED_MAPS.md](history/RESIDUE_AFTER_RELEASED_MAPS.md) ·
[FUSION_RESULTS.md](history/FUSION_RESULTS.md) · [FUSION_RESULTS_00808.md](history/FUSION_RESULTS_00808.md) ·
[PROPOSAL_FUSION_CLS_RESULT.md](history/PROPOSAL_FUSION_CLS_RESULT.md) · [PROPOSAL_COMMIT_BAR.md](history/PROPOSAL_COMMIT_BAR.md) ·
[REGION_PROPOSAL_RESULT.md](history/REGION_PROPOSAL_RESULT.md) · [REGION_FALSE_ADMISSION.md](history/REGION_FALSE_ADMISSION.md) ·
[PERCEPTION_RESULT_FINAL.md](history/PERCEPTION_RESULT_FINAL.md) · [PERCEPTION_GAP.md](history/PERCEPTION_GAP.md) ·
[PERCEPTION_INVESTIGATION.md](history/PERCEPTION_INVESTIGATION.md) · [FEATURE_MEMORY_PROBE.md](history/FEATURE_MEMORY_PROBE.md) ·
[FEATURE_MEMORY_ONLINE.md](history/FEATURE_MEMORY_ONLINE.md) · [ASSET_SUBSTITUTION_EVIDENCE.md](history/ASSET_SUBSTITUTION_EVIDENCE.md) ·
[VERIFIER_EVAL.md](history/VERIFIER_EVAL.md) (what tells the target from a false positive: the VLM vs a CLIP margin, r and q at STOP range)

**Superseded** — [INVESTIGATION.md](history/INVESTIGATION.md) (by S71; names a stack that no
longer exists) · [PROPOSAL_FUSION_RESULT.md](history/PROPOSAL_FUSION_RESULT.md) (by
`PROPOSAL_FUSION_CLS_RESULT.md`, 59/107 against 62/107) · [DUALMAP_SWAP.md](history/DUALMAP_SWAP.md)
(paused before any trial ran) · [METHOD_SECTION.zh.md](history/METHOD_SECTION.zh.md) (by `PAPER.md`)

**Chinese-language summaries** — [PROGRESS_zh.md](history/PROGRESS_zh.md) ·
[REPORT_zh.md](history/REPORT_zh.md) · [METHOD_SECTION.zh.md](history/METHOD_SECTION.zh.md)

The scripts these documents ran, where they are no longer part of the release surface, are in
[`scripts/legacy/`](../scripts/legacy/README.md); the exploratory presets they name are in
[`configs/experiment/archive/`](../configs/experiment/archive/README.md) and compose as
`+experiment=archive/<name>`.
