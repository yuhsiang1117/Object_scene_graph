# Benchmark v2: the multi-floor moved-object set

*2026-10-16. What the episodes are, how they were generated, how they are scored and what
the prior map holds -- the items the review found missing from the paper. Every number is
re-derivable: `scripts/describe_benchmark.py +experiment=mfv2_dataset --seeds 42 43 44`
for the episodes, `scripts/audit_map_coverage.py` (its output: `outputs/mfv2/audit_v2.txt`)
for the prior.*

## Why a v2

The paper's Table II was measured on a set that could not answer the questions asked of
it (`PAPER.md`, Appendix B, "Table II, re-examined"): its two arms ran different layouts;
success was horizontal-only, so two stops a storey away scored; every episode started on the
storey the object had left, so an agent was only ever rewarded for leaving; no prior map held
a target where it had been, so belief revision was never exercised; and the ASCENT arm ran
without its own detector for non-COCO queries and with one attempt to OSG's three. v2 fixes
the episodes (`configs/experiment/mfv2_dataset.yaml`); the arms on top of it are
`mfv2_osg`, `mfv2_ascent`, `mfv2_ascent_yoloe` and the ablations (`scripts/run_mfv2.sh`).

## Scenes, objects, layouts

Five HM3D scenes, each with objects on two storeys: 00800-TEEsavR23oF, 00821-eF36g7L6Z9M,
00862-LT9Jq6dN3Ea, 00873-bxsVRursffK, 00878-XB4GS9ShBRE. Six to eight YCB objects per
scene, from eight classes: cracker box, tin can (tomato soup), blue plastic pitcher, bowl,
banana, toy airplane, coffee can, yellow bottle (mustard).

Each scene has two layouts, both from the data collector's generator
(`/habitat-data-collector`, `replace_placement.py`), never written as coordinates:

* **static** -- the world pass 1 maps.
* **cross_anchor_01** -- every object moved to a *different* anchor, 6-8 per scene onto the
  other storey.

The generator's rules, which every pose satisfies: an object rests on the top surface of an
anchor whose kind *and* room its class affords (food items: kitchen, dining room, living
room); one target per anchor; 0.35 m keep-out between objects; every pose reachable from
the one navmesh island (`verify_dualmap_layouts`, `validate_dualmap_authoring`,
`audit_reachability`: worst approach 0.51-1.01 m). To balance the direction of cross-storey
moves, some static and cross-anchor poses were exchanged and both re-placed through the same
generator (`scripts/flip_cross_anchor.py`, `scripts/restage_near_stairs.py`;
`docs/history/CROSS_ANCHOR_STATUS.md` §4e). Upper-storey anchor supply caps how many
descents two scenes can hold (00873: one affordable anchor; 00878: two).

The multi-floor `in_anchor_01` layouts are not used: they move objects to other anchors and
storeys, which is not what the name says, and `sim/ycb_layouts.py` refuses them.

## Episodes

One episode per (layout, object, seed); seeds 42 / 43 / 44 differ only in where the agent
starts. Per seed, 73 episodes; over the three, 219. The class of an episode is read off its
geometry (`scripts/analyze_multifloor.py`), not off a folder name:

| class | episodes (3 seeds) | start on the object's storey | start elsewhere | start-to-goal geodesic m (p10 / p50 / p90) |
|---|---:|---:|---:|---:|
| static | 111 | 45 | 66 | 5.2 / 10.2 / 23.0 |
| same_floor (moved, same storey) | 33 | 13 | 20 | 4.1 / 9.7 / 17.0 |
| upward (moved one storey up) | 36 | 22 | 14 | 4.3 / 14.4 / 25.2 |
| downward | 39 | 24 | 15 | 3.5 / 6.6 / 21.1 |
| all | 219 | 104 | 115 | 4.4 / 9.7 / 21.1 |

**Starts** (`ycb.start_storey_uniform`): the storey is drawn uniformly from the storeys that
hold at least one of the scene's objects (`_benchmark_storeys`), then a navigable point on
it; so about half the episodes start on the object's storey and the agent has to decide
whether to change storey at all. v1 started every episode on the storey the object had left.

| scene | start storeys (height, starts over 3 seeds) |
|---|---|
| 00800 | 0.2 m (16), 3.2 m (20) |
| 00821 | -3.5 m (22), 0.2 m (23) |
| 00862 | 0.1 m (22), 3.3 m (26) |
| 00873 | -3.0 m (22), 0.0 m (20) |
| 00878 | 0.1 m (22), 2.8 m (26) |

## Where moved objects go

The destination anchors of the 36 relocations (per seed; the layouts are the same for every
seed): a cabinet 17 times (kitchen, sink and bathroom cabinets included), a table 13
(coffee, dining, display and side tables included), a counter, worktop or kitchen island 5,
and a bedroom chest once. Against the affinity prior the agent
searches with (`graph.priors.CONTAINER_AFFINITY` where it lists the class, the LLM's ranking
otherwise; anchor kinds mapped to the container categories), **20 of the 36 destinations
are in the prior's top three for that class, 9 are its first choice, and 33 are ranked at
all**. The single-floor DualMap release is different: its relocations put a tomato-soup can on
a bed seven times (`graph/priors.py`), and there a semantic prior cannot pay; here it can,
which `mfv2_osg_affinity_llm` / `mfv2_osg_affinity_flat` measure.

## Scoring

A STOP succeeds within **1.0 m of the object horizontally and on its storey** -- the stop's
floor height within 1.0 m of the object's (`ycb.object_success_max_floor_dy_m`). **Three
attempts** for every arm (`eval.attempts`), each failed one rearming the agent with its map
kept; **1000 steps**. SR is reported with a Wilson 95% interval, by class and by start
storey, alongside first-attempt SR (which compares a one-attempt run with a three-attempt one
without a rerun) and the horizontal-only rule (`scripts/analyze_multifloor.py`).

## The prior map (pass 1)

`scripts/run_prior_maps.sh`: ASCENT's policy walks every static episode of the scene
(`mf5_mapping_both`) at pass 2's resolution (1280x960, YOLOE-11L at imgsz 1280), and after
each episode looks at its target from three of the layout's own viewpoints
(`ycb.map_tour_views`), so the map the agent inherits has seen every object. Two artifacts,
accumulated over the scene's episodes: ASCENT's obstacle maps (the occupancy pass 2 plans
over) and OSG's scene graph (`outputs/maps_v2_osg`). Audit, `outputs/mfv2/audit_v2.txt`:

| scene | storeys mapped (free fraction) | static targets with a same-label track within 1 m |
|---|---|---:|
| 00800 | 0.81, 0.89 | 6 / 6 |
| 00821 | 0.92, 0.88 | 7 / 8 (cracker box missing) |
| 00862 | 0.91, 0.89, 0.89 | 7 / 8 (cracker box missing) |
| 00873 | 0.87, 0.90 | 6 / 7 (tomato-soup can missing) |
| 00878 | 0.95, 0.90, **0.00** | 7 / 8 (toy airplane missing) |

00878's top storey was never walked in pass 1, so it has no occupancy; its objects are in the
graph from the target tour, and pass 2 explores that storey without a prior occupancy map.

The prior also carries **false positives**: of the 90 graph tracks carrying a benchmark
object's label, 52 are not the object (offline, `stop_gate_vet_prior`'s measurement). A
static episode starting on the other storey often commits first to one of those on its
start storey -- 4 of 17 such OSG episodes succeeded on seed 42, against 11 of 11 starting
on the object's storey -- which is what `mfv2_osg_gate2_vet` addresses.
