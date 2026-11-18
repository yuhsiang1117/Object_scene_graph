# scripts/legacy

One-off probes, calibrations and campaign drivers that produced a number in a document under
`docs/history/` and were not needed again. They are kept runnable — they compile, and their
imports resolve against the current `src/osg` — because the documents that cite them are the
provenance of numbers the paper still carries, and a reader who wants to re-derive one should
find the tool that made it rather than a description of it.

They are not the release surface. Nothing in `configs/`, `tests/`, `src/` or the top-level
documentation calls them; the entry points that do the work today — `run_eval.py`,
`run_robot.py`, `run_dualmap_arm.sh`, the `analyze_*`, `compare_*`, `report_*` and `render_*`
tools, `ros2/`, `video/`, `campaign/` — stay in `scripts/`.

| script | what it measured | cited by |
|---|---|---|
| `analyze_climb.py` | stair-climb attempts and their outcomes per run | `CROSS_FLOOR_CLIMB.md` |
| `analyze_gt_dump.py` | ground-truth visibility probes against detector hits | `PERCEPTION_INVESTIGATION.md` |
| `analyze_proposal_tracks.py` | region-proposal tracks: admissions, phantoms, commits | `PROPOSAL_FUSION*.md` |
| `analyze_reachability_ceiling.py` | the fraction of targets the mover can reach at all | `REACHABILITY_CEILING.md` |
| `author_semantic_layouts.py` | the first, semantic authored layouts (superseded by the collector import) | `DYNAMIC_SCENES.md` |
| `build_crossfloor_maps.sh` | the pass-1 maps for the early cross-floor A/B | `MULTI_FLOOR.md` |
| `build_dualmap_swap_map.py` | prior maps for the proposed object swap | `DUALMAP_SWAP.md` |
| `calibrate_approach_recheck.py` | the arrival re-check thresholds | `SR_PROPOSAL_CLOSE_LOOK.md` |
| `detector_bakeoff.py` | detector candidates side by side on the same keyframes | `PERCEPTION_*.md` |
| `measure_room_seg.py` | room segmentation statistics | `DYNAMIC_SCENES.md` |
| `measure_stair_accumulation.py` | how stair evidence accumulates over an episode | `CROSS_FLOOR_CLIMB.md` |
| `offline_appearance_proposals.py` | offline appearance-based proposals | `FEATURE_MEMORY_PROBE.md` |
| `offline_appearance_retrieval.py` | the oracle appearance-retrieval gate over a frozen DualMap-protocol run (imported by `offline_appearance_proposals.py`) | `APPEARANCE_REID_AND_CANDIDATE_BUDGET_PLAN.md` |
| `offline_candidate_budget.py` | the candidate budget sweep | `APPEARANCE_REID_AND_CANDIDATE_BUDGET_PLAN.md` |
| `plot_cross_anchor_map.py` | the stored ASCENT obstacle map with the target marked (the auto-writer in `eval/obstacle_map_viz.py` replaced it) | `CROSS_ANCHOR_OBSTACLE_MAP.md` |
| `probe_feature_memory.py` | feature-memory retrieval on recorded keyframes | `FEATURE_MEMORY_*.md` |
| `probe_nearmiss.py` | why the detector missed a target that was plainly in view, re-run over `eval.gt_dump_dir` frames | `PERCEPTION_INVESTIGATION.md`, `ASSET_SUBSTITUTION_EVIDENCE.md` |
| `probe_perception_gap.py` | whether any box is proposed on the perception failures (imported by `probe_region_ranking.py`) | `PERCEPTION_GAP.md` |
| `probe_region_ranking.py` | region-proposal ranking on recorded frames | `REGION_PROPOSAL_RESULT.md` |
| `rank_search_surfaces.py` | where the search posterior ranks the surface the object moved to, without a simulator | `DYNAMIC_SCENES.md`, `FEATURE_MEMORY_PROBE.md` |
| `report_fusion_ab.py`, `run_fusion_ab.sh` | the proposal-fusion A/B | `FUSION_RESULTS*.md` |
| `run_condition.sh` | one condition of the dynamic YCB ladder (its `outputs/maps_hires` priors are gone) | `REPORT_zh.md` |
| `run_mf5_mapping.sh` | the first five-scene mapping pass (superseded by `campaign/rebuild_maps.sh`) | `MULTI_FLOOR.md` |

Run them from the repository root as before, with the new path:
`python scripts/legacy/<name>.py …`. Paths inside them are anchored one level deeper to match
(`Path(__file__).parents[2]` is the repository, Hydra's `config_path` is `../../configs`), and
the scripts that import a sibling (`from probe_perception_gap import …`) have it beside them.
