# ASCENT vs. our multi-floor implementation

Source: `relative_works/ascent/` (official code for *Stairway to Success*,
arXiv 2505.23019). Compared against `osg` at commit `ff8b661`.

Both systems reach the same top-level answer — **one BEV map per storey, an LLM/
prior-informed decision about when to change floor** — and then differ almost
everywhere below that. The differences are mostly explained by one root choice:
**what defines a "floor".**

---

## 1. The root difference: what is a floor?

| | ASCENT | ours |
|---|---|---|
| Definition | The map you get **after climbing a staircase** | A cluster of **agent standing heights** |
| Index source | `_cur_floor_index`, a counter incremented by a *completed traversal* | `FloorEstimator`, from `camera_y - camera_height` |
| Floors known before climbing | none | any level whose surface has been *seen* |
| Uses agent height at all | **no** | yes, as the primary signal |

ASCENT's floor index only ever moves inside `_process_stair_climb_state`, when
the agent has been on the stair map and then leaves it
(`map_controller.py:299-315`). A new map layer is allocated by `_add_floor_map`
at that moment. If a climb is later judged a misdetection, the layer is deleted
and the index rolled back (`_remove_floor_map`, lines 287-297).

**Consequence: for ASCENT, stair detection *is* floor detection.** There is no
independent notion of "there is another storey up there". That is why the repo
spends ~900 lines on a stair state machine (`_reach_stair`,
`_reach_stair_centroid`, `_climb_stair_flag`, `_stair_dilate_flag`,
`_climb_stair_over`, `_temp_stair_map`, `_passive_up_stair_steps`,
`_passive_down_stair_steps`, plus reverse-climb recovery) — if it fails, nothing
else can work.

Ours inverts the dependency: floors come from the height trace, and stairs are
not needed to *know* a floor exists. This is what let us ship a working
cross-floor system **after our stair detector failed** (see §3).

---

## 2. Per-floor maps — nearly identical

ASCENT (`map_controller.py:85-91`):

```python
self._obstacle_map_list[env]   # one ObstacleMap per floor
self._object_map_list[env]     # one ObjectPointCloudMap per floor
self._value_map_list[env]      # one ValueMap per floor
self._cur_floor_index[env]
self._obstacle_map[env] = self._obstacle_map_list[env][self._cur_floor_index[env]]
```

Ours (`mapping/floor_stack.py`):

```python
FloorStack._layers: Dict[int, FloorLayer]   # costmap + segmenter + room_labels
FloorStack.current_id
NavAgent.costmap -> property -> self._floor_stack.costmap
```

Same idea. Two differences worth noting:

- **ASCENT rebinds `self._obstacle_map[env]` on every floor change**; we expose a
  **property**, so nothing downstream can hold a stale reference. Minor, but it
  means our planner/frontier/room-seg/viewpoint code needed *zero* changes.
- **ASCENT keeps a per-floor `ValueMap`** (BLIP-2 image-text similarity, VLFM
  lineage). We have no value map at all — our frontier ranking is purely
  geometric (nearest + momentum + info-gain). See §5.

Also, ASCENT's floor list is **ordered and contiguous** (index ±1 = the storey
above/below), because it is built by traversal. Our floor ids are
**creation-ordered and stable**, so a basement discovered late gets a fresh id
and nothing is renumbered — we sort by height when order matters.

---

## 3. Stair detection — where we failed and they did not

**ASCENT is semantic-first, and requires two models to agree**
(`obstacle_map.py:519-541`):

```python
if np.any(stair_mask) > 0 and np.sum(seg_mask == STAIR_CLASS_ID) > 20:
    stair_map = (seg_mask == STAIR_CLASS_ID)
    fusion_stair_mask = stair_mask & stair_map      # detector AND segmenter
```

`stair_mask` comes from an open-vocab detector; `seg_mask` from **RedNet**
(a trained indoor semantic segmenter, MPCAT40 class 17). Only their
intersection is projected into the map. Direction is decided by camera pitch:
pitch ≥ 0 → `_up_stair_map`, pitch < 0 → `_down_stair_map`.

Downward stairs get a **second, dedicated mechanism** (`obstacle_map.py:548+`):
depth is *inverted* (`max_depth - depth`) and points below ground level are
extracted, because a descending staircase is invisible to a forward-looking
obstacle band.

**We tried to do this geometrically and it did not work.** Our
`mapping/stairs.py` implements ZONDA's `Δh` criterion — a cell is steppable when
its max height difference to 8-neighbours sits between `min_dh` and
`climb_limit`. Measured: a real staircase (0.28 m treads, 0.17 m risers)
**fragments into 9 disconnected components, 0 regions detected**, because tread
*interiors* are flat and fall below `min_dh`. Only ramps are found.

Reading ASCENT confirms the diagnosis in our notes: **we inverted ZONDA's
criterion.** ZONDA uses `Δh < H_agent` as a *traversability filter* (flat floor
passes) and relies on the semantic label to pick out stairs. ASCENT does the
same — geometry never identifies a staircase in either paper. `floor.stairs`
stays off in our config and the defect is pinned by a strict xfail.

**What we do instead:** `mapping/portals.py` finds patches of the height layer
that sit a full storey away from the current floor — "somewhere another level is
visible" — and heads there; the navmesh walks the actual stairs. This needs no
stair detector, and it sees descending openings for free (we record the height
layer ±3.5 m, both directions), where ASCENT needs its inverted-depth special
case.

**The cost of our approach** is visible in our own numbers: portals require the
agent to physically observe another storey's *surface*, and in **14 of 24
cross-floor episodes `portals_seen == 0`**. ASCENT's semantic detector fires on
a staircase seen from across a room, which is a much easier thing to see. This
is currently our single largest gap, and ASCENT's design is the direct answer to
it.

---

## 4. The floor-switch decision — same skeleton, different substance

**ASCENT** (`llm_planner.py:217`):

```python
if (floor_num[env] > 1
    and num_steps[env] - self.multi_floor_ask_step[env] >= MULTI_FLOOR_ASK_STEP_THRESHOLD   # 60
    and obstacle_map[env]._floor_num_steps >= FLOOR_EXP_STEP_THRESHOLD                       # 100
    and use_multi_floor):
    multi_floor_response = self._llm.chat(multi_floor_prompt)
```

Then the LLM returns a floor number; `> current` → go up (sentinel `-100`),
`< current` → go down (`-200`), `==` → fall through to single-floor frontier
reasoning.

**Ours** (`mapping/portals.py::FloorSwitchPolicy`): min interval 50 steps, late
cutoff at 0.7·budget, plus a category-evidence rule that can bring a switch
forward or hold it back.

| | ASCENT | ours |
|---|---|---|
| Ask interval | 60 steps | 50 steps |
| Min exploration on this floor first | 100 steps on the floor | `min_objects_to_judge = 8` objects mapped |
| Who decides | **LLM**, one call | fixed co-occurrence table, no call |
| Inputs | per-floor room types + object lists, floor priors, room priors | count of target's companion categories on this floor |
| "Stay" rule | LLM sees `fully_explored` per floor | `strong_evidence ≥ 2`, expiring after 120 steps |

ASCENT's prompt is materially richer than our scalar. It feeds, per floor:
`status` (current/other), `fully_explored`, **room types present**, and
**objects present** — then asks the LLM to pick a floor. We compress the same
question into one integer.

### The priors are the biggest single gap

ASCENT ships two statistical tables we do not have:

- **`statistic_priors/hm3d_floor_object_possibility.xlsx`** —
  `P(target on floor k | building has N floors)`, e.g. for a 2-floor building
  one row reads 41% / 59%, for 3 floors 22.2 / 29.6 / 48.1. **We have no
  floor-level prior at all.** A toilet is more likely upstairs in a 2-storey
  house; we do not know that.
- **`statistic_priors/Per_Category_Region_Per_Cat_Votes.csv`** — object→room
  vote counts derived from HM3D annotations across the *full* vocabulary
  (e.g. `air conditioner`: bathroom 38, bedroom 67, kitchen 54, living 27…).

Our `graph/priors.py` is a **hand-written table covering 6 categories**, written
from intuition. ASCENT's is **derived from the dataset** and covers everything.
That CSV is directly reusable — it is plain data, no model, no license issue
beyond the repo's own — and would replace our hand-rolled `CATEGORY_CONTEXT`
with measured co-occurrence.

---

## 5. Frontier ranking — they have a value map, we do not

ASCENT inherits VLFM's **`ValueMap`**: BLIP-2 image-text similarity between the
frontier's view and the target, accumulated into a per-floor 2D value grid, used
to sort frontiers before the LLM ever sees them (`_sort_frontiers_by_value`).

We rank frontiers geometrically: `score / path_cost` with an info-gain term and
a momentum/continuity bonus. Our investigation found LLM *text* scoring over a
serialized scene graph byte-for-byte redundant with geometric nearest — but that
is **not the same thing** as a visual value map, which is what both VLFM and
ASCENT actually use. This remains untested here, and we already have the encoder
in memory (YOLOE loads `mobileclip_blt.ts`).

### Frontier stickiness — they have the fix on our TODO list

`llm_planner.py::_handle_frontier_stick_and_disable` implements exactly the
commitment hysteresis we listed as an open lever:

- `STICKY_FRONTIER_STEP_THRESHOLD = 20` — if the same frontier is selected and
  the distance to it does not change by more than
  `STICKY_FRONTIER_DISTANCE_THRESHOLD = 0.3` m for 20 steps, **disable it**.
- `REPEATED_SELECTION_THRESHOLD = 20` — a frontier selected 20 times
  non-consecutively is disabled outright.

Ours re-decides every 5 steps with a 40% re-selection rate and only a 0.6 m
location blacklist. Their scheme is more principled: it disables on *lack of
progress toward* the frontier, not on proximity to a previously abandoned point.

---

## 6. Things we have that ASCENT does not

- **A floor estimator independent of stairs.** ASCENT cannot represent "there is
  a storey above me" until it has climbed to it. We register a level from the
  height trace or from an observed surface. This is why our system still works
  with stair detection disabled.
- **The horizontal-run test for committing a storey.** We hit a failure ASCENT
  structurally cannot have (a staircase *landing* registering as a floor) and
  solved it with a displacement test — you cannot walk 2.5 m across a 1.2 m
  landing. ASCENT never faces this because only a completed traversal creates a
  floor.
- **Terminal creep.** ASCENT uses a PointNav policy for the last leg; we replaced
  a fixed depth stop with "walk in until physically blocked", which lifted
  terminal conversion 55% → 84%.
- **Reproducibility discipline.** We measured that the hosted VLM verifier makes
  runs non-reproducible and require `verification=off` for equivalence A/Bs.
  Nothing equivalent appears in ASCENT's code or docs.
- **A byte-identical single-floor regression gate** (35/35 episodes).

## 7. Things ASCENT has that we should consider taking

Ranked by expected value against our current bottleneck:

1. **Semantic stair detection.** Directly attacks our largest gap
   (`portals_seen == 0` in 14 of 24 cross-floor episodes). ASCENT needs RedNet
   (a trained MPCAT40 segmenter) plus a detector; we already run YOLOE
   open-vocab and measured `stairs` tracks firing in only 15% of multi-floor
   episodes — so the gap is likely RedNet's per-pixel segmentation, not the
   detector. Worth testing YOLOE's stair *masks* (it does segmentation) fused
   with our height layer, which is the closest cheap analogue.
2. **The inverted-depth downward-stair probe.** A concrete implementation of the
   look-down probe we specified in Stage 4e and never built.
3. **`Per_Category_Region_Per_Cat_Votes.csv`.** Drop-in replacement for our
   hand-written co-occurrence table, dataset-derived and full-vocabulary.
4. **A floor-level prior** (`P(target on floor k | N floors)`). We have nothing
   here and it is exactly the signal our switch gate lacks.
5. **Frontier stickiness / disable-on-no-progress.** Our lever #4, already
   implemented there with sensible constants.
6. **A visual value map.** Larger change; the honest note is that our "LLM is
   redundant" finding does not cover it.

## 8. Numbers, and why they are not directly comparable

ASCENT reports **65.4% SR / 33.5% SPL on HM3D**; we are at **48.0% / 0.234** on
full v1. The gap is real but the setups are not aligned:

- **Different episode sets.** They evaluate HM3D v1 val (2000 episodes, all 20
  scenes); we run 100 episodes at 5/scene. Our own measurements show ±3 episodes
  of verifier noise at n=100, so our number carries a much wider error bar.
- **Different perception stack.** ASCENT runs BLIP-2 + D-FINE + Grounding-DINO +
  Mobile-SAM + RedNet + Qwen2.5-7B. We run one YOLOE model and a VLM verifier,
  targeting a 6 GB laptop profile.
- **Different starting point.** Our 48.0% is measured against our own 40.0%
  baseline on the same episodes; that delta is the meaningful figure here, not
  the absolute against a differently-configured system.

The one number that *is* comparable in spirit: ASCENT reports **33.3% SR on
cross-floor episodes** where VLFM gets 0.4%. We are at **16.7%** (4/24) from a
starting point of 0.0%. Same direction, half the distance, with the gap
concentrated exactly where §3 predicts — finding the staircase in the first
place.
