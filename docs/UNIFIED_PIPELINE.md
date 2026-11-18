# The unified pipeline: one agent, three datasets

The fusion branch had grown two arms that were never the same agent:

* `dualmap_protocol_osg_sensor_v4_fuse_cls` — the DualMap released benchmark,
  **62/107** (in-anchor 34/54, cross-anchor 28/53). Region-proposal fusion,
  presence, the absence VLM, the search posterior, island snap, the close
  last-metre approach. No floor machinery.
* `ycb_authored_15_fused_v16` — the authored 15-scene and multi-floor dynamic
  benchmarks. The floor stack, per-storey costmaps, stair evidence, the flight
  climb. No region-proposal fusion, and the DualMap approach tuning turned off.

They shared 32 mechanism settings at non-default and disagreed on 54. This
document fuses them into **one agent config** carried by a single mixin, so the
only thing that changes between the three benchmarks is which dataset is loaded.

## The mixin

`configs/experiment/osg_unified_pipeline.yaml` carries every mechanism flag and
nothing about the dataset. It is the union of:

* the **floor / climb group** (32 keys), taken from `_fused_v16`:
  `floor.enabled`, `per_floor_costmap`, `stairs`, `use_prior_stairs`,
  `cross_floor`, `climb_targets: flights_first`, `no_level_on_flight`,
  `climb_enabled`, `climb_relink_flights`, `navmesh_3d_goals`,
  `mapping.multi_floor`, `scene_graph.containers_floor_relative`,
  `floor_mass_rule: mean` + `floor_mass_margin: 1.15`, `floor_llm`,
  `frontier_cost_free_cell`, and the rest;
* the **DualMap perception tuning** (22 keys), taken from `_v4_fuse_cls`:
  the region-proposal fusion (`enabled`, `every_keyframe`, `commit_min_obs 4`,
  `commit_tau 0.28`, `commit_tau_by_class` for bowl/plate/mug), `agent_radius 0.1`,
  `navmesh_snap_on_agent_island`, the close approach, the stale-anchor
  verification group;
* the **shared approach/search tuning** (4 keys) the DualMap and 15-scene bases
  carried but the bare multi-floor base did not: the fine viewpoint rings,
  `ring_radius_extent_aware`, `close_look_before_absence`,
  `search_proximity_len_m: 100`.

Three dataset presets include it:

| preset | eval base | dataset |
|---|---|---|
| `dualmap_osg_unified` | `dualmap_protocol_osg_sensor_v2` | DualMap released, 107 trials |
| `mf5_osg_unified` | `ycb_dynamic_multifloor` | 5-scene multi-floor (00800, 00821, 00862, 00873, 00878), the collector's `outputs/dualmap_multifloor` |
| `ycb15_osg_unified` | `ycb_authored_15` | 15-scene authored, the collector's `outputs/dualmap_authoring` |

The multi-floor table in the paper is a fourth preset, `mf5_osg_on_ascent_map`: the same
dataset with each storey's obstacle map taken from an ASCENT mapping pass
(`docs/DYNAMIC_EXPERIMENT_PROTOCOL.md`). 00862 replaced 00808 in the five scenes; the
preset comment in `mf5_osg_unified.yaml` records why the two arms briefly disagreed.

## The agent is provably identical across the three

Composing the three presets and diffing every key that is not under
`eval.` / `dualmap.` / `ycb.`:

```
dualmap_unified vs mf5_unified  : 0 mechanism differences
dualmap_unified vs ycb15_unified: 0 mechanism differences
```

And `dualmap_osg_unified` differs from the locked `_v4_fuse_cls` in **exactly
the 32 floor-group keys and nothing else** — the region-proposal fusion,
verification, and approach tuning that produce 62/107 are untouched. So the
whole question of whether the fusion preserves the DualMap result reduces to one
thing: is the floor group inert on a single-floor scene?

## Why the floor group is inert on one storey

Every floor-group flag either gates on more than one level existing, or is a
no-op when the single level sits at the agent's own standing height:

* `floor.enabled` + `estimate_only: false` runs the estimator, which finds one
  level on a single-floor scene and never emits a switch.
* `navmesh_3d_goals` snaps a goal at its floor height; on one floor that height
  equals the agent's standing height, which the code notes is "a genuine no-op
  there".
* `containers_floor_relative` subtracts the storey height from a surface's top;
  on floor 0 that height is 0, so the band is unchanged.
* `floor_mass_rule: mean` + `floor_mass_margin` only change how a *second*
  storey is scored against this one; with one storey there is nothing to score.
* `climb_*`, `cross_floor`, `use_prior_stairs`, `no_level_on_flight` all require
  a stair or a second level to fire.

The two that touch single-floor code paths regardless are
`frontier_cost_free_cell` (sets `cost_prefer_free` in the frontier scorer) and
`per_floor_costmap` (wraps the costmap in a stack). The bit-identity smoke below
is what actually decides them.

## Bit-identity smoke (the check that matters)

The floor group is **not** inert by default on a single-floor scene, and the
control run proved it is the floor group and not run-to-run numerics:

| trial (00829 cross_anchor) | `_v4_fuse_cls` in this env vs the locked run | ungated `dualmap_osg_unified` vs the lock | floor_switch_attempts |
|---|---|---|---:|
| bowl | **identical** | differs (stopped at 448 vs 500, 1.04 m vs 2.73 m) | 1 |
| cracker box | **identical** | differs (244 vs 242 steps) | 0 |
| scissors | **identical** | differs (10.99 m vs 9.87 m) | 2 |

`_v4_fuse_cls` reproduces the lock bit-for-bit, so the environment is faithful;
the drift is entirely the floor group, and it tracks `floor_switch_attempts`.
The cause: on a single-floor scene `find_portals` reads a tall bookcase as a
portal, and once the floor's frontiers are far the undirected geometric gate
opens and the agent walks off to it.

### The fix: an undirected switch requires a second known level

`floor.switch_requires_second_level` (default off, on in the mixin) makes an
UNDIRECTED switch a no-op unless the estimator already knows >= 2 levels. This
is safe by construction for the multi-floor benchmark: `apply_map` seeds every
storey of a schema-v2 prior map into the estimator on load, so a genuine
cross-floor scored run has >= 2 levels from step 0 and its switches -- directed
and undirected -- fire exactly as before. A truly single-floor scene has one
level and nowhere to go, so it is gated and reproduces `_v4_fuse_cls`.

Map-building is unaffected: prior maps are built with the baseline
`ycb_authored_15`, deliberately, so the map is never an experimental variable.

### Bit-identity: two of three, and the residue is the container fix

Round 1 -- the two discovery gates (`switch_requires_second_level`) -- made bowl
and cracker box bit-identical to the locked `_v4_fuse_cls`. Scissors still
differed, so it was bisected: seven single-trial arms, each disabling one part of
the floor group.

| arm | scissors distance | containers | vs lock |
|---|---:|---:|---|
| lock (`_v4_fuse_cls`) | 9.8697 | **85** | -- |
| `navmesh_3d_goals=false` | 8.4463 | 87 | differs |
| `exploration.floor_*` all off | 8.4463 | 87 | differs |
| `stair_evidence/down_look` off | 8.4463 | 87 | differs |
| `estimate_only=true` | 8.4463 | 87 | differs |
| `per_floor_costmap=false` | 8.4463 | 87 | differs |
| `mapping.multi_floor=false` | 8.4463 | 87 | differs |
| **`floor.enabled=false`** (whole stack off) | 8.4463 | 87 | **still differs** |

Every arm returns the same number, and disabling the entire floor stack does not
restore the lock -- so the cause was never in the floor group. The container
count names it: 87 against the lock's 85.

It is `scene_graph.containers_floor_relative`, the one flag in the mixin that is
**not** gated by `floor.enabled`. It measures a support surface's height from its
own storey instead of from y=0 -- and 00829's single floor sits at y=0.1357, not
zero. The qualification band is therefore shifted by 13.6 cm, two more surfaces
qualify, the search ranks a different set, and the trajectory diverges.

That is a **correctness fix, not a regression**: measuring a surface from its own
floor is the right rule, and on a scene whose floor is 13.6 cm off the origin the
absolute band was systematically wrong. It is also the fix that made upstairs
containers exist at all (00808 floor 1: 0 -> 62). So the residue is not something
to gate away -- it is the unified pipeline being more correct than the arm it is
being compared against.

**Conclusion.** Bit-identity with `_v4_fuse_cls` is not achievable while keeping
the floor-relative container rule, and should not be the bar. Two of three trials
are identical; the third differs because two extra (correctly qualified) surfaces
enter the search, and it fails in both arms regardless (scissors is 0/9
cross-anchor, the known-undetectable class).

## The full paired run

The test of "keep 62/107" was the paired SR over all 107 dynamic trials. It has been run
twice, both times with the two presets side by side on the same tree:

| date | prior maps | `_v4_fuse_cls` | `dualmap_osg_unified` | trials agreeing |
|---|---|---|---|---|
| 2026-09-20 | rebuilt on the current tree (534 tracks on 00829) | 49/107 | 50/107 | **106 / 107** |
| 2026-09-27 | the originals the locked run loaded (682) | 60/107 | 60/107 | **186 / 186**, static included |

The floor group is worth one episode on the first run and nothing on the second, so the
two presets are one arm on single-floor scenes. Neither run reaches 62/107; that gap is
the prior map and the container image, not this mixin, and
`docs/CROSS_ANCHOR_REPRODUCTION.md` takes it apart (`outputs/RERUN/` and
`outputs/RERUN_UPLOADMAP/` hold both runs).
