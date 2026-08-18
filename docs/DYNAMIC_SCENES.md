# Dynamic-scene handling for `osg`: container layer → presence beliefs → posterior search

## Context

We are building an ObjectNav system that should handle **changing scenes** better than
DualMap (arXiv 2506.01950). Analysis of DualMap's released code found its dynamic
handling rests on two things we can beat:

1. Objects leave its map on a **timer**, never because the robot looked and found
   nothing. "I turned away" and "it is gone" are the same state transition.
2. Anchor memory is **append-only** (`utils/object.py:885`) and scored by `max` over an
   undated bag of CLIP vectors, so a moved object is remembered at its old anchor
   forever, and cluttered anchors win every query.

Our answer is a **presence belief per object**, updated by positive *and* negative
evidence, feeding a **posterior search** over where a missing object went. The full
design is in two companion notes (`The Absence Problem`, `Presence-Belief Internals`).

Before any of that, the scene graph needs the layer DualMap's anchors occupy. Today it
is `floor → room → object`; every object hangs off a room centroid, so there is nowhere
to record "the mug is on *that* table" — and both the presence filter and the search
posterior are defined relative to that relation. **Phase 0 adds
`floor → room → container → object`.**

Prior art is real and must be credited, not claimed: the filter is Rosen et al.'s
persistence filter at object level; the search index is classical discrete search
theory; Khronos and TemPest are close relatives. Novelty, if any, is the composition
plus the measurement protocol.

---

## Phase 0 — Container layer (start here)

**Goal:** `floor → room → container → object` in the scene graph. Structure only —
every eval number must be **bit-identical** to today after this lands.

> **Status: done (2026-08-18).** 19 new unit tests; 234 unit + 2 integration tests green.
> 3-episode YCB static regression is identical to the pre-change baseline on every
> navigation metric (SR, SPL, steps, mean distance to goal). Measured cost:
> `scene_graph` rebuild 43.7 ms → 49.5 ms per keyframe (**+5.8 ms**), ≈9% pipeline fps.
> The first implementation cost +37 ms; a height index over surfaces plus a batched
> Mahalanobis test (`containers.ShadowIndex`) removed 6/7 of that. The remaining cost is
> ~3 redundant `shape_matrix()` evaluations per candidate track per rebuild — worth
> caching on `Ellipsoid` when Phase 1 starts calling `world_extent` per track per
> keyframe, not before.

### Design decisions

- **Containers are views over existing tracks, not new entities.** A table is one
  `ObjectTrack` that appears *both* as a `ContainerNode` and as its own
  `ObjectNodeView`. This preserves the module's stated contract: *"the graph never owns
  object state; object nodes are thin views over tracks"* (`src/osg/graph/scene_graph.py:1`).
- **A container is a linked component, not a single track.** `linking.relink()` already
  unions same-label ellipsoids within `link_dist_m` (L-shaped sofas). The container id is
  the minimum track id in the component; its center comes from the existing
  `ObjectLayer.center_of()`, which already averages linked centers.
- **Objects on no container attach directly to the room** with `container_id=None`.
  This is a deliberate divergence from DualMap, which *deletes* a high-mobility object
  with no supporting anchor (`utils/local_map_manager.py:316`) — the reason a mug on the
  floor is unmappable there.
- **Containers do not nest.** A track that is itself a container is never assigned to
  another container; that keeps the relation a forest and avoids cycles.
- **Assignment is recomputed every `rebuild()`**, idempotently, like `relink()`. No
  incremental state to corrupt.

### The geometry helper (do this first — three later phases reuse it)

`Ellipsoid.R` is initialised to the camera rotation and then refined, so it is **not**
world-axis-aligned. Reading `axes[HEIGHT_AXIS]` for "how tall" is wrong. Add to
`src/osg/objects/ellipsoid.py`:

```python
def world_extent(self, axis_unit: np.ndarray) -> float:
    """Half-extent along a world direction: sqrt(dᵀ R diag(a²,b²,c²) Rᵀ d)."""

def ground_footprint(self) -> Tuple[np.ndarray, np.ndarray]:
    """(center_xy, cov_xy) — the ellipsoid projected onto PLANE.
    area = pi * sqrt(det(cov_xy))."""
```

The same `world_extent` is the depth-band extent in Phase 1 and the surface-height test
in Phase 3. Write it once, test it once.

### New module: `src/osg/graph/containers.py`

Keep it free of Habitat/simulator imports so it unit-tests on a machine with no HM3D
licence — the deliberate style of `src/osg/sim/ycb_layouts.py`.

```python
CONTAINER_CATEGORIES = {"table", "desk", "counter", "shelf", "cabinet", "dresser",
                        "nightstand", "bed", "sofa", "stool", "bench", "oven",
                        "washing machine", "refrigerator"}   # all in DEFAULT_VOCABULARY

def is_container(track, *, top_h=(0.2, 1.4), min_area_m2=0.06) -> bool:
    """Category gate, then a real horizontal top: world top height in range and
    ground footprint area above minimum."""

def supports(container_center, container_ell, obj_center, obj_ell, tol_m=0.15) -> bool:
    """Object's bottom within tol of the container's top, and its ground-plane
    center inside the container's footprint."""

def relative_pose(container, obj) -> np.ndarray:
    """p_rel = R_cᵀ (t_o − t_c) — survives anchor migration in Phase 6."""
```

Every category above already exists in `DEFAULT_VOCABULARY` (`src/osg/core/config.py:14`),
so no detector change is needed.

### Files to modify

| File | Change |
|---|---|
| `src/osg/objects/ellipsoid.py` | `world_extent()`, `ground_footprint()` |
| `src/osg/graph/containers.py` | **new** — membership + support rules |
| `src/osg/graph/scene_graph.py` | `ContainerNode`; `RoomNode.container_ids`; `ObjectNodeView.container_id`/`p_rel`; `_rebuild_containers()`; `containers_in_room()`, `objects_on()` |
| `src/osg/graph/serialize.py` | `to_json` gains a `containers` block and `container_id` on objects. **`to_prompt_text` is untouched** — it feeds LLM prompts, and changing it changes behaviour |
| `src/osg/core/config.py` | `SceneGraphConfig`: `container_top_h_m`, `container_min_area_m2`, `container_support_tol_m` |

`SceneGraph.rebuild()` ordering becomes rooms → objects → **containers** → floors;
containers need object centers, and floors already read object floor ids.

### Tests — `tests/unit/test_containers.py`

Reuse the fake `_Track` / `_Layer` pattern from `tests/unit/test_scene_graph_floors.py:18`.

- mug 0.75 m above the floor, over a table's footprint → `container_id` = table.
- same mug shifted 1 m sideways, off the footprint → `container_id is None`, still in the room.
- mug on the floor → `container_id is None` and **still present in `sg.objects`** (the anti-DualMap regression).
- a sliver "table" (tiny footprint) and a table at 1.9 m → rejected by the geometry gate.
- an L-shaped sofa as two linked tracks → **one** container node, id = min track id.
- a table is not assigned into another container (no nesting).
- `world_extent` against a hand-rolled rotated ellipsoid where the answer is analytic.

### Exit criteria

- All unit tests pass.
- `+experiment=ycb_authored_nav ycb.layout_types=[static] verification=off` produces
  **identical** SR / SPL / step counts to a pre-change run on the same seed.
  (`verification=off` is mandatory — the README records that runs are not reproducible
  with the VLM verifier on.)
- `to_json` output contains a populated `containers` block on a real episode.

---
## Phase 1 — C1 presence filter + the minimal C2 scoring hook

**Goal:** every track carries a belief about whether it is *still there*, updated by
positive **and** negative evidence. This is the only phase that adds information the
system does not already have; every later phase spends it.

> **Status: done (2026-08-18), opt-in.** 19 new unit tests; 253 unit + 2 integration
> green. `scene_graph.presence.enabled=false` by default — turning the filter on changes
> which candidate the agent proposes first, so it must be an explicit A/B, not a silent
> default. 3-episode YCB static run: navigation metrics identical to baseline, and the
> mechanism demonstrably fires — ~2.1–2.5k expectations, **~400 negative updates** and
> ~10 objects driven below p=0.1 per episode (`agent_stats.presence_*`).
> Cost: `object_layer` 55.3 ms → 62.0 ms per keyframe (**+6.7 ms**), pipeline 6.4 → 5.7 fps.
> The cost is ~0.2 ms per expected track spread over many small numpy calls, not one
> hotspot; the real fix is a batched multi-track projection, which Phase 3 wants anyway
> for candidate scoring. Do it once, there.
>
> **The recall fit is not yet trustworthy — keep the constant.** Fitted on 6790
> expectations from 3 episodes: Brier 0.1497 against 0.1525 for a constant predictor, i.e.
> the view features buy almost nothing, and the `depth_m` weight comes out **positive**
> (farther ⇒ easier to detect), which is backwards. The cause is structural, not sample
> size: the expectation gate already requires `area ≥ min_det_bbox_px`, so at long range
> only large objects are ever logged and depth acts as a proxy for size. Fixing it needs
> per-class terms and a log that records *gated-out* expectations too, so the fit sees
> the small-and-far cases it is currently blind to. Until then `recall_model_path` stays
> empty and every negative update is the same size — blunt, but unbiased.

### Theory

For track *i* carry a latent binary state `X_i` — "the object is still at its mapped
pose" — as log-odds `ℓ_i`. Three channels touch it, and keeping them separate is the
whole design.

**Channel 1 — measurement.** Two indicators per keyframe: `E_i`, whether the object
*would* be detected if it were still there, and `Z_i`, whether a detection was
associated to it. The binary Bayes update with detector recall `r` and false-alarm
rate `q`:

```
Z=1, E=1 :  Δℓ = log( r / q )            # positive
Z=0, E=1 :  Δℓ = log( (1−r) / (1−q) )    # NEGATIVE — the channel DualMap has no path for
E=0      :  Δℓ = 0                       # unobserved ≠ observed-absent
```

The magnitudes are what make it honest. At `r=0.6, q=0.05` a clean look that finds
nothing gives `Δℓ = log(0.4/0.95) = −0.87`; at `r=0.2` (small, far, oblique) the same
non-detection gives `−0.22`, correctly weak. From `ℓ₀ = 1.5` (P≈0.82) the number of
clean views to drive P below 0.1 is closed form and therefore testable:

```
k ≥ (ℓ₀ − ℓ*) / |Δℓ| = (1.5 − (−2.20)) / 0.87 = 4.3  →  5 good looks
```

**Channel 2 — survival.** `P ← P · 2^(−Δt / t½(class))`. **Inert for our current
benchmark**: a 500-step episode spans no meaningful wall-clock time, so any sane `t½`
leaves P untouched. It exists for the paired-layout and multi-session settings. Do not
tune half-lives before Phase 2 lands.

**Channel 3 — clamping.** `ℓ ∈ [−6, +6]`. No absorbing states: an object at P=0.0025 is
still in the map, still projectable, still resurrectable by one detection. That is both
the cross-anchor "it came back" case and the mitigation for a recall-blind filter
deleting everything the detector blinks on.

### The visibility test

`E_i` is where the win lives and where a naive implementation destroys itself. Expected
only if all of:

| Gate | Test | Start |
|---|---|---|
| Frustum | `ellipsoid.project(K, T_cw)` non-None (its cheirality check is already there) **and** ≥50% of the ellipse bbox inside the image | 0.5 |
| Range | `mean_depth_at(T_cw)` within sensor + detector usable range | 0.4–6.0 m |
| Scale | projected `Ellipse2D.area` ≥ A_min — **reuse `min_det_bbox_px`** | same as the admission gate |
| Occlusion | occluded pixel fraction over the ellipse support ≤ τ_occ | 0.30 |
| Motion | keyframes only — `KeyframeSelector` already rejects fast rotation | — |

The scale gate is not optional: expecting detections at a scale `ObjectLayer.update`
would have filtered out anyway manufactures false negatives on every distant object in
the room. Expectation and admission must use the same threshold or the filter is biased
by construction.

**Why the depth map gives the asymmetry for free.** Sample depth over the projected
ellipse and compare against the expected band from the quadric:

```
extent(d) = sqrt( dᵀ Q d )            # Q = R diag(a²,b²,c²) Rᵀ, i.e. Ellipsoid.world_extent
band      = [ z_c − extent − τ_d , z_c + extent + τ_d ]
```

A pixel reading *inside* the band says something is there; *nearer* says an occluder is
in front; *farther* says you are seeing straight through to the surface behind where the
object used to be. Removal and occlusion are opposite signs of one comparison — the
distinction DualMap's timer cannot make.

### Two places the current pipeline would silently defeat it

1. **The early return.** `ObjectLayer.update` filters detections then does
   `if not dets: return` (`src/osg/objects/object_layer.py:68`) — exactly the frame where
   negative evidence is most valuable (staring at the table, detector produced nothing).
   The presence update must run before it, on frames with zero detections.
2. **The category gate.** `assoc_category_gate=True` means a mug relabelled as a bowl
   produces a spurious negative *and* a new track — the belief would collapse on a
   relabel rather than a removal. Compute `Z` **label-agnostically**: any detection whose
   ellipse overlaps the projection counts as a sighting. Presence is "something is
   there"; identity is C2's problem.

### Files

| File | Change |
|---|---|
| `src/osg/objects/presence.py` | **new** — `PresenceState`, `Expectation`, `RecallModel`, `PresenceFilter` |
| `src/osg/objects/association.py` | `ObjectTrack.presence: PresenceState` |
| `src/osg/objects/object_layer.py` | run the filter every keyframe *before* the early return; label-agnostic `Z`; `candidates()` sorts by `best_score * presence.p` and gains `min_presence` |
| `src/osg/core/config.py` | nested `PresenceConfig` under `SceneGraphConfig` |
| `src/osg/agent/nav_agent.py` | thread the config through |
| `scripts/fit_recall_model.py` | **new** — log expectation features from an eval run, fit `r`, write JSON |

### The recall model

No new data needed. `NavAgent.on_keyframe_detections` already exists as a hook
(`src/osg/agent/nav_agent.py:104`); log `(track_id, expected?, area_px, depth, incidence,
detected?)` for every keyframe of an existing run, then fit

```
r = σ( w₀ + w₁·log A_px + w₂·z + w₃·cos θ )
```

frozen as a small JSON of weights, with a constant-`r` fallback so the filter runs
before any fit exists. `q` stays constant at 0.05 — it only bounds the maximum positive
step and the system is insensitive to it. **Publish the reliability diagram**: a
calibration plot of predicted vs observed detection rate is what makes every negative
update in the system defensible, and no dynamic-mapping paper I know of shows one.

### Tests — `tests/unit/test_presence.py`, all Habitat-free

- Ellipsoid at 2 m, synthetic depth at 5 m → E=1, Z=0, `ℓ` drops by exactly `log((1−r)/(1−q))`.
- Same geometry, depth at 1 m → occluded → E=0 → `ℓ` unchanged. **The test that matters.**
- Behind camera → `project()` None → E=0. Too far / too small → E=0 by range and scale.
- `k` clean negatives cross P<0.1 at the `k` the closed form predicts.
- Clamp: 50 negatives then one detection → P recovers above 0.5.
- Label-agnostic `Z`: a detection with the wrong label over the projection is a sighting, not a miss.
- Zero-detection keyframe still updates beliefs (the early-return regression).

### Exit criteria

- Unit tests pass; full suite green.
- Stale-goal rate drops on static YCB layouts (the agent stops re-proposing objects it
  has walked past and not seen); SR does not regress.
- Per-keyframe cost reported next to the accuracy numbers. DualMap's headline claim is
  efficiency, and "more correct at no extra cost" is a far stronger result than "more
  correct".

---

## Phase 2 — Benchmark protocol (must precede any search-policy number)

**Goal:** make change *observable*, so the mechanism can be measured rather than inferred.

> **Status: done (2026-08-18).** 24 new unit tests; 285 unit + 2 integration green.
> The benchmark is **two passes**, and the change happens **between** them:
>
> ```bash
> # pass 1 -- explore the static layout, keep the map
> python scripts/run_eval.py +experiment=ycb_authored_nav 'ycb.layout_types=[static]' \
>     verification=off scene_graph.presence.enabled=true ycb.map_out=outputs/maps \
>     'eval.episode_ids=[00829-QaLdnwvtxbs__static__50001__s0]'
>
> # pass 2 -- navigate the MOVED world with that map
> python scripts/run_eval.py +experiment=ycb_authored_nav \
>     'ycb.layout_types=[static,cross_anchor]' verification=off \
>     scene_graph.presence.enabled=true ycb.map_in=outputs/maps \
>     'eval.episode_ids=[00829-QaLdnwvtxbs__cross_anchor_01__50001__s0]'
> ```
>
> `graph/map_store.py` persists the map's **evidence** — object tracks with their
> ellipsoids, observations and presence beliefs, the occupancy grid, the room
> segmentation — and rebuilds the scene graph on load, so a snapshot cannot freeze an
> old container rule into a new run. Single storey only; `save_map` refuses a multi-floor
> agent rather than silently dropping a level. The world does **not** change during an
> episode: the pair is known for metadata (what moved, from where), and the firing rule
> is separately gated.
>
> **The result the benchmark exists to produce.** Same agent, same episode, cracker box
> relocated table_175 → table_188:
>
> | | outcome | commits to |
> |---|---|---|
> | fresh map (control) | **success**, 363 steps, SPL 0.066 | the real object, at step 355 |
> | stale map from pass 1 | **failure**, stopped at step 64 | the **ghost** at the old pose, at step 1 |
>
> `ghost_rate 1.0`, `belief_latency.flip_rate 0.0`. The stale map is not merely unhelpful,
> it is actively harmful: it converts a success into a confident failure. That is
> DualMap's dominant failure mode reproduced in our own harness, and it is the baseline
> every later phase has to beat.
>
> **Why the presence filter did not save it, and what that means for Phase 3.** The agent
> commits at **step 1** with p=0.98 and STOPs on arrival, so the episode is over before
> negative evidence can accumulate. Belief latency is therefore unmeasurable on this
> episode — not because the filter is wrong but because nothing consults it before
> committing. Two fixes, both Phase 3's business: refuse to STOP on a target the live
> view does not show (C5's negative confirmation), and rank candidates against the cost
> of reaching them rather than committing to the first plausible one.
>
> **Fixed on the way through: the target/vocabulary collision.** With target `cracker box`
> the detector vocabulary also offered the generic `box`, and YOLOE labelled every
> sighting `box` — the target was mapped and never proposable, which is what starved the
> earlier measurement. `target_vocabulary()` now drops a generic entry that is a
> whole-word part of the target. Pass 1 went from 263 tracks with 0 usable targets to 173
> tracks with the cracker box mapped 0.05 m from its authored pose.
>
> **Still open:** `in_anchor` layouts cannot be synthesised (they need a second pose on the
> *same* surface) and must come from the collector; and the false-disbelief rate on static
> furniture recorded below is unchanged and still worth fixing before Phase 3.

### Two additions

1. **Paired-layout passes (the protocol).** Explore the static layout and store the map
   (`ycb.map_out`), then navigate the moved layout starting from it (`ycb.map_in`). The
   world is fixed for the whole episode; the staleness comes from the snapshot. This is
   the DualMap comparison, and it is what the metrics below are defined against.
2. **Mid-episode relocation (secondary, opt-in).** `ycb.relocate_at_step` re-applies
   poses to the *same* rigid objects during an episode, visibility-gated, so a change can
   be witnessed rather than only inherited. Strictly an extra condition — it is off by
   default and is **not** how the benchmark is run.

### Metrics — `src/osg/eval/metrics.py`

| Metric | Definition | What it proves |
|---|---|---|
| Belief latency | steps between a change becoming *observable* and the belief flipping | C1 works at all; DualMap's value is unbounded |
| Stale-goal rate | fraction of goal commitments to a location already observed empty | attacks the 28.3% false-match bucket |
| Containers inspected | candidate surfaces visited before success | C3's transition model vs greedy ranking |
| Post-failure distance | metres travelled after the first failed container | the SPL half; the attempt-limit bucket |
| Ghost rate | instances still believed present that are physically gone, at episode end | whether memory self-corrects or accumulates |

### Ablation ladder

(a) timeout-only memory + greedy retry — **our DualMap equivalent inside our own
harness**, and the fairest baseline we can publish; (b) + negative evidence; (c) +
instance records and posterior scoring; (d) + transition-model search; (e) + learned
affinities over repeated episodes in one scene. Every rung with `verification=off`.

### In-anchor results, and three blockers found on the way

The collector's real layouts live in a second tree
(`data/dualmap/HM3D_collect/<scene>/dynamic_scene_config/`) in an older schema — no
`authoring` block, no per-object anchor, and a scene path from a machine that no longer
exists. `scripts/import_collector_layouts.py` converts them (3 in_anchor + 3
cross_anchor for 00829-QaLdnwvtxbs) and validates every one through the real loader.
Anchors are derived, and honestly: in this scene the closest pair of static objects is
2.19 m apart, so "one static object, one anchor" is not a guess. In-anchor displacements
are 0.01–1.56 m and cross-anchor 2.18–9.37 m, cleanly separated.

**Blocker 1 — the size gate, now fixed in the experiment config.** YCB targets are an
order of magnitude smaller than HM3D furniture. A bowl reaches 1759 px at its *best*
authored viewpoint against a 1500 px node-creation gate and a 3000 px candidate gate, so
the target was discarded on sight and every episode failed for want of a detection rather
than of navigation. With `min_det_bbox_px: 300` the same episode **succeeds** (SPL 0.32)
and the bowl is mapped 0.03 m from its authored pose. Every earlier YCB number in this
document was measured under the old gate.

**Blocker 2 — four of six targets are not detectable at all.** Probing the best authored
viewpoint of each: bowl 0.90, tomato soup can 0.45, and *nothing* for pitcher, plate,
scissors or cracker box. The pitcher renders as a plain dark tumbler with no handle or
spout (not a lighting problem — flat, default and scene lighting are identical), the
plate is a flat disc on a bed, the scissors are a few hundred pixels, and the cracker
box's best viewpoint clips into geometry. **Only the bowl is a usable target today**, and
that is a data problem, not a mapping one.

**Blocker 3 — in-anchor moves are absorbed by linking, so the belief never collapses.**
This is the real in-anchor finding and it is specific to the condition. Bowl, in_anchor
layout 1, moved 0.80 m on the same table:

| | outcome | what the map did |
|---|---|---|
| stale map | fail, 47 steps | committed at step 1 to the ghost, `ghost_rate 1.0` |
| fresh map (control) | fail, 500 steps | never reached a viewpoint of the new pose |

With the stale map the agent kept the ghost (track 108, old pose) *and* created track 127
at the new pose. `linking.relink` unions them — same label, 0.80 m apart, under
`link_dist_m: 1.0` — so `object_center` reports the component **mean**, 0.42 m from
either bowl, a place where no bowl is. Both tracks report p=0.9975: the ghost is never
disbelieved because the component keeps being re-observed, and `belief_latency.flip_rate`
is 0.0 for a reason that has nothing to do with the filter.

A cross-anchor move is far enough that no linking occurs and a clean second track appears;
an in-anchor move lands inside the linking radius and is averaged into the ghost. So
**C6 is not the lowest-priority item after all for the in-anchor condition** — `relink`
must not union a track the filter is losing faith in with a freshly-observed one, and the
presence filter must see the ghost separately in order to lose faith in it at all.

### Ghosting, fixed — and what it uncovered

The in-anchor ghost survived for two compounding reasons, both mine, both now fixed.

**A saturated belief cannot be argued with.** A sighting is worth `log(r/q) = +2.5` and a
miss only `log((1−r)/(1−q)) = −0.9`, so a symmetric ±6 clamp saturated after three
sightings and then needed **seven** clean misses to unwind. The bowl was stored at
p=0.9975 from five observations. The positive clamp is now +3.0 while disbelief keeps the
−6.0 floor: believing an object's *presence* that hard is unjustified, because the world
changes while you are not looking, whereas an object known to be gone should stay gone.
And a belief restored from a snapshot is capped again at 1.5 (p≈0.82) — the map was built
in another session, so the survival channel of the filter applies, collapsed into one
honest number.

**Linking merged the object with its own past.** `relink` unions same-label tracks within
`link_dist_m`, which is right for two ellipsoid fragments of one sofa and catastrophic for
a bowl that moved 0.80 m: the stale track and the fresh one were merged, so
`object_center` reported their midpoint — 0.42 m from either bowl, a place with no bowl,
which neither observation can ever contradict. Linking now requires co-observation
(`link_max_frame_gap`), because fragments of one object are seen *together* and an object
and its ghost never are. Restored observations are stamped into a previous session so
they cannot be mistaken for current ones. A third fix stops a detection crediting every
track it overlaps: one detection now credits the single best-matching track, so a
neighbour 0.8 m away can no longer keep a ghost alive.

Same episode, bowl moved 0.80 m on the same table:

| | before | after |
|---|---|---|
| tracks at end | 108 and 127 **merged**, both reported at [0.13, 0.86, 1.11] | **separate**: ghost at [-0.30, 0.87, 1.03], bowl at [0.55, 0.86, 1.19] |
| ghost belief | p = 0.9975, never moved | p = **0.654, falling** from 0.82 |
| real bowl | never isolated | mapped **0.05 m** from its true new pose |

**Correction to an earlier reading of this run.** An earlier version of this section said
the agent commits from memory and stops "never having seen the object". That was wrong,
and the episode record says so: `approach_stop_reason: depth` with a `bbox_log` showing
the target closing from 1.5 m to 0.95 m. The agent *does* see the bowl and stops on the
depth criterion. What actually happens on a **correct** preloaded map is a terminal
precision miss — it stops 0.28 m from the nearest authored viewpoint against a 0.18 m
success radius, where the same agent exploring from scratch lands inside it:

| same world, same episode | outcome |
|---|---|
| fresh map | **success**, SPL 0.323, 157 steps |
| perfect preloaded map | **failure**, 49 steps, 0.28 m from the nearest viewpoint |

That is a ~0.10 m problem in where the approach terminates, worth fixing on its own, and
it is not what C5 addresses.

**Targets are now selectable** (`ycb.targets`), because four of six assets cannot be
detected at any authored viewpoint. The layouts are DualMap's original data and are never
edited; this only chooses which episodes to run. Bowl is the usable target today.

### C5 — the VLM as a second sensor, and absence as an observation

**Status: done (2026-08-18).** 8 new unit tests; 307 unit + 2 integration green. No
usable VLM credential in this environment (`NVIDIA_API_KEY` is set but empty), so the
VLM path is unit-tested against a stub client and the detector path is measured end to
end.

Three pieces:

- `VLMVerifier.verify_absence(rgb, region_bbox, categories)` asks which of up to five
  categories are inside a marked region, listing what it sees first so a "no" is grounded
  in a description rather than in agreeing with the question. A failed call returns
  `None` — *no information*, never *absent*; treating a network error as evidence would
  quietly delete objects.
- `PresenceFilter.apply_reading(track, detected, recall, q)` takes a reading from **any**
  sensor with that sensor's own error rates. This is the payoff of writing C1 as a filter:
  fusion needs no fusion code. A VLM at `r=0.85, q=0.02` contributes −1.88 per miss
  against the detector's −0.64, so one trusted look is worth nearly three ordinary ones.
- `NavAgent._absence_at_arrival` makes the three no-sighting terminations
  (`path_consumed`, `deadline`, `retreat`) apply that evidence instead of stopping on
  empty space. A track seen at any point during the approach is left alone — that is a
  geometry problem, not absence.

The thresholds are chosen from the arithmetic rather than picked: from a belief reloaded
at p=0.82, one trusted VLM "no" lands at 0.407 and abandons, the detector's silence alone
needs three failed approaches (0.702 / 0.554 / 0.395), and a belief saturated *in this
episode* survives a single VLM "no" at 0.755. Absence has to be earned, and cheap evidence
earns it more slowly. `detector_absence_recall` is 0.8 from measurement — 6790 logged
expectations give a 0.812 detection rate in the regime the visibility gate admits.

Cross-anchor, bowl moved 6.84 m, stale map, detector only:

| | outcome | what the map did |
|---|---|---|
| absence off | stops at step 36 on empty space | ghost belief had already fallen to 0.25 — **and the stop ignored it** |
| absence on | abandons at step 36, explores to 500 | belief 0.066, ghost blacklisted, `absent_on_arrival:path_consumed` recorded |

Worth being precise about what fixed it: the presence filter's ordinary per-keyframe
negatives had *already* driven the ghost to p=0.25 by the time the agent arrived. The
belief was right and nothing consulted it. The explicit arrival reading pushes it to 0.066
and makes the decision, but the deeper lesson is that a belief nothing reads is not a
belief.

The episode still fails: the agent abandons the ghost and then has 464 steps of undirected
exploration to find a bowl 6.84 m away. Turning "not here" into "then look there" is C3,
and this is the first run where that is the *only* thing left in the way.

Regression: the fresh-map success path is unchanged (SR 1, SPL 0.323, 157 steps), and on a
correct preloaded map the agent still stops on `depth` with no absence check — C5 stays
out of the way when the target is visible.

---

## Phase 3 — C3 search index (+ full C2 posterior)

**Goal:** when the target is not where it was, decide where to look next by *posterior
over locations*, not by next-highest similarity.

### Theory

With belief `b(x)` over candidate locations, per-visit detection probability `d(x)` and
cost `c(x)`, the discrete search problem's classical result is that the optimal *order*
is by the index `b(x)·d(x)/c(x)`, and after an unsuccessful look

```
b(x) ← b(x) · (1 − d(x)) ,  renormalised over X ∪ {nowhere-mapped}
```

Two things follow that belong in any write-up. Greedy is **optimal** under those
assumptions, not a heuristic; what makes it approximate in practice is travel-order
coupling (it is really a profitable-tour problem), and that should be stated rather than
glossed. And **DualMap's ignore list is the degenerate case** `b(a) ← 0` after a single
look, discarded at query end — the multiplicative form keeps a badly-inspected surface
in play in proportion to how badly it was inspected, and because it lives in the
presence beliefs it persists.

### The belief

```
b(a′) ∝ w_sem(c, a′) · w_prox(a, a′) · w_afford(c, a′)      # mapped containers
b(f)  ∝ β · unexplored_mass(f)                             # frontiers
```

- **w_sem** — already exists. `graph/priors.py` holds `CATEGORY_CONTEXT` / `CATEGORY_ROOMS`
  and `floor_target_evidence` is the right computation at the wrong granularity.
  Generalise to `region_target_evidence(scene_graph, region, target)` over floor, room or
  container neighbourhood. A refactor, not new science.
- **w_prox** — `exp(−d_geo(a,a′)/L)`, L≈4 m, using planner geodesics. Never Euclidean, or
  through-wall neighbours get credit they have not earned.
- **w_afford** — does *a′* offer a surface this class can sit on? Straight off the
  Phase-0 container geometry (`top_h`, `area_m2`) plus a small table in `priors.py`:
  `{class: (h_min, h_max, min_area_m2)}`. This term buys cross-anchor performance with
  no LLM call.
- **β·unexplored** — the escape hatch. Must be non-zero or the agent never re-explores
  once its containers are exhausted. Calibrate so that after every container has been
  inspected once, a median frontier outranks a median container.

### Implementation — smaller than it looks

`select_frontier` already computes `argmax score / path_cost`
(`src/osg/exploration/selector.py:88`). That **is** the search index; C3 widens the
candidate set rather than adding a planner.

```python
# src/osg/exploration/search_belief.py  (new)
@dataclass
class SearchCandidate:
    kind: str                 # "container" | "frontier"
    goal_xy: np.ndarray
    prior: float              # b(x)
    detect_prob: float        # d(x), from the SAME recall model as C1
    ref_id: int

def select_candidate(cands, planner, costmap, agent_xy, top_n=6, ...):
    """Same structure as select_frontier: prior-rank, cut to top_n, plan for
    true cost, return argmax prior*detect_prob/path_cost."""
```

`NavAgent._select_new_frontier` (`src/osg/agent/nav_agent.py:648`) becomes
`_select_next_search_target`. Exploration and re-search stop being two subsystems — they
still are in DualMap, whose retry loop cannot decide to go explore instead.

Container scoring becomes `Σ_i P(i is target)·P(present_i)`, never `max`: for an
anchor holding *n* instances with score noise σ, `E[max_i s_i] ≈ μ + σ√(2 ln n)`, so a
max-scored anchor carries a bias term that depends only on how much has ever been on it.

**Failure mode to watch:** if `d(x)` is over-estimated, one glance zeroes a container the
agent barely looked at. Take `d(x)` from the fitted recall model evaluated at the
viewpoint actually planned, not a constant.

### C3 — the search index, implemented and measured

**Status: implemented (2026-08-18), not yet winning episodes.** 12 new unit tests; 322
unit + 2 integration green. Off by default (`exploration.search_posterior=false`).

Mapped surfaces now compete with frontiers under one index, `b*d/c`, which is the index
`select_frontier` already computed — C3 widened the candidate set rather than adding an
objective or a planner. `b(x)` is affordance (binary: a shelf at head height is not a
worse place to look for a bowl, it is not a place) times affinity times proximity to the
object's last believed pose; `d(x)` is 0.8, the measured detection rate; `c(x)` is the
planner's geodesic cost. A visit multiplies belief by `(1 - d)` rather than zeroing it,
so a surface glanced at from four metres stays plausible — the distinction DualMap's
ignore list cannot make, and which it discards at query end regardless.

The provided text model (`nvidia/nemotron-3.5-lightning-30b-a3b`) supplies affinities for
targets `graph/priors.py` has no entry for — every YCB target. One call per unknown class,
cached to `outputs/affinity_cache.json`, so a run is deterministic after the first and the
priors the agent used can be read and diffed afterwards. The static table always wins where
it has an entry: a model must not quietly rewrite a prior someone chose deliberately.
**The model is text-only** (multimodal disabled on this endpoint), so it cannot serve C5's
absence check.

Cross-anchor, bowl moved 6.84 m, stale map, 500 steps:

| | final distance to goal | surfaces inspected |
|---|---|---|
| no posterior | 12.15 m | — |
| posterior, first version | 12.45 m | 7 (none actually inspected) |
| + commit until arrival | 6.88 m | 7 |
| + proximity floor, surfaces preferred | 5.73 m | 13 |
| + unlisted-category weight | **5.28 m** | 10, all plausible |

**No episode succeeds yet, and the honest reading is that this is now a budget and
parameter question rather than a mechanism one.** Three defects the runs exposed, each
fixed and each worth recording:

- Surfaces were marked searched five steps after selection, because the selection guard
  re-runs every five steps and the mark fired whether or not the agent had arrived. It
  visited seven surfaces and inspected none. The agent now stays committed to a surface
  until it arrives or spends its budget, and a surface it never reached earns only a
  quarter of the search credit — spending full belief on it would retire exactly the
  places that were never looked at.
- A pure `exp(-d/L)` proximity prior says an object that moved 7 m is almost impossible,
  when cross-anchor moves here average 5 m. It also punishes distance twice, since the
  cost term already divides by path length. The prior now floors at 0.2: a mixture of
  "moved nearby" and "moved anywhere".
- A category absent from a ranking we *have* was scored 0.5, tying the lowest genuinely
  plausible surface — so a bed and a sofa ranked with a sink as places to look for a bowl,
  and the agent went to both. Absent from a known ranking now ranks below all of it, while
  having no ranking at all still leaves every surface equally plausible.

What is left is a sweep of `search_frontier_weight` and the step budget over the full
episode set, not more single-episode tuning: a 500-step budget to re-find an object 6.84 m
away among 43 candidate surfaces is tight, and one episode cannot separate a good policy
from a lucky one.

### Can a VLM serve as C5's absence sensor? Tested, and no — not on this endpoint

There is no "locate anything" model on the NIM endpoint (102 models; the vision-capable
ones are `meta/llama-3.2-11b-vision-instruct`, `meta/llama-3.2-90b-vision-instruct`,
`nvidia/nemotron-nano-12b-v2-vl`, `nvidia/llama-3.1-nemotron-nano-vl-8b-v1`,
`microsoft/phi-3-vision-128k-instruct`, `nvidia/vila`). The provided text model
`nvidia/nemotron-3.5-lightning-30b-a3b` has multimodal processing disabled, so it cannot
answer C5's question at all — it serves C3's affinity prior instead.

Scored on the real task: the same authored viewpoint with the YCB bowl present, and with
it relocated away.

| model | bowl present | bowl gone | usable |
|---|---|---|---|
| meta/llama-3.2-11b-vision-instruct | present | **present** | no |
| nvidia/nemotron-nano-12b-v2-vl | present | **present** | no |
| nvidia/llama-3.1-nemotron-nano-vl-8b-v1 | present | error | no |
| microsoft/phi-3-vision-128k-instruct | error | error | no |
| meta/llama-3.2-90b-vision-instruct | — | — | times out |

Two things make this less damning than it looks, and both matter. The marked region also
contains a **scene-geometry dish** — HM3D scans come with their own crockery — so "is a
bowl in this region" is genuinely ambiguous and answering "yes" is not pure hallucination.
And that is the real lesson: a VQA question about a *category* cannot decide the presence
of an *instance*, which is what a stale map needs to know. The instrument for that is an
open-vocabulary detector returning boxes at locations — which is YOLOE, already in the
pipeline, and which is why the detector-driven absence path is doing the work.

**A bug this uncovered, and it invalidated the first version of the test above.**
`apply_layout_transforms` was a silent no-op: `inject_layout_objects` marks objects
`MotionType.STATIC`, and a STATIC habitat object **ignores an assignment to
`translation` and then reads back the old pose**, so nothing in the calling code could
tell. The first present/absent pair was byte-identical — zero pixels changed — and all
three VLMs "failed" a test in which nothing had moved. Switching to `KINEMATIC` before
the move changes exactly 1759 pixels, matching the manifest's visible-pixel count for the
bowl. Every mid-episode relocation before this reported six objects moved while the world
stood still. The function now verifies the pose it asked for actually took, and raises if
not; `test_relocation_verifies_the_move_actually_took` pins it. The two-pass benchmark is
unaffected — it injects each layout at reset rather than moving objects.

### The first batch: C3 is not validated, and the dominant failure is elsewhere

Seven episodes (static, in_anchor 1-3, cross_anchor 1-3), bowl, two-pass protocol against
one shared map of the static world, `verification=off`.

| layout | baseline steps / dist | posterior steps / dist | surfaces inspected |
|---|---|---|---|
| cross_anchor_01 | 500 / 12.15 m | 500 / **5.28 m** | 10 |
| cross_anchor_02 | 290 / **3.78 m** | 500 / 7.21 m | 11 |
| cross_anchor_03 | 500 / **6.63 m** | 500 / 7.45 m | 12 |
| in_anchor_01 | 47 / 0.18 m | 47 / 0.18 m | 0 |
| in_anchor_02 | 49 / 0.21 m | 49 / 0.21 m | 0 |
| in_anchor_03 | 49 / 0.19 m | 49 / 0.19 m | 0 |
| static | 49 / 0.28 m | 49 / 0.28 m | 0 |
| **SR / mean distance** | **0.0 / 3.35 m** | **0.0 / 2.97 m** | |

**C3 is not validated by this.** One episode improves a great deal, two get worse, four are
untouched, no episode succeeds either way, and n=7 with zero successes cannot separate a
policy from noise. The mean-distance difference is one episode's worth of movement.

**The dominant failure is not search at all.** Four of seven episodes — every in_anchor
plus the static control — end at **0.18, 0.19, 0.21 and 0.28 m** against a 0.18 m success
radius. The agent finds the real bowl (in_anchor episodes end holding two bowl tracks: the
ghost and the relocated object, 0.6-0.9 m apart, correctly separated since the ghosting
fix) and stops just too far away. Fixing where the approach terminates could plausibly
convert four of seven episodes; nothing in the search posterior can compete with that, and
until it is fixed a search metric measured on this benchmark is reading noise off a
terminal-precision bug.

One earlier change did carry: `abandon_below_p` at 1.0 turned two cross-anchor episodes
from stopping on empty space at ~50 steps into searching for 290 and 500 steps. That is
the C5 rule working, and it is what made the cross-anchor rows above comparable at all.

### The terminal condition: SR 0.0 -> 0.429

The batch above failed four of seven episodes by between 0.01 m and 0.10 m, having found
the object. The cause is geometry, not search. HM3D scores success as the distance from
the final pose to the nearest sampled **goal viewpoint**, and those sit on rings at
0.8 / 1.2 / 1.5 / 2.0 m. The navmesh approach drove at the object and stopped when the
target's depth reached `approach_stop_depth_m = 1.0 m` -- radially half-way between the
two inner rings, about 0.2 m from either, against a 0.18 m radius. The agent was doing
everything right and stopping in the one place that could not score.

`agent.approach_to_viewpoint` drives to a pose from `ViewpointPlanner` instead, which
samples the *same* radii the manifest does, so the agent stands ON a ring and the only
error left is angular -- at 24 samples on the innermost ring, at worst 0.10 m. Two
follow-on fixes were needed, and both came from failures the change produced:

- **Arriving is not looking.** The follower arrives on whatever heading the path ended
  with, so the agent can reach a viewpoint facing away. Without a sweep the absence check
  fired on a *correct* map and abandoned a bowl that was exactly where the map said,
  turning a 49-step stop into 500 steps of searching for something already found. The
  agent now sweeps in place until the target is seen or a full circle is spent.
- **A sweep is one look, not twelve.** Applying a negative reading per sweep frame is
  epistemically tempting and wrong: twelve frames of the same object from the same pose
  share range, lighting and viewing angle, so multiplying their likelihoods turns one
  correlated detector failure into near-certain "absence". Measured: it dropped SR from
  0.429 to 0.286 by abandoning the static control. The sweep gives the detector a chance;
  it does not vote.

| | SR | SPL | in_anchor distances |
|---|---|---|---|
| depth stop (before) | 0.0 | 0.0 | 0.18 / 0.21 / 0.19 m |
| viewpoint terminal | **0.429** | **0.337** | 0.12 / 0.19 / 0.17 m |

Three of seven episodes now succeed: the static control at 0.06 m and two in_anchor
relocations at 0.12 m and 0.17 m, with SPL 0.76, 0.85 and 0.75 -- the agent goes more or
less straight to the object, which is what a map is *for*. in_anchor_02 still misses at
0.19 m, one centimetre out; that is angular sampling and navmesh snap, and closing it
needs a finer ring or a short final alignment step.

The cross-anchor episodes are unchanged at 0-for-3: the object really is 2.8-6.8 m away
and re-finding it is C3's job, which the earlier batch shows is not yet working. But the
benchmark now has a working terminal, so a search improvement can finally show up as
success rather than as a distance that was never going to score.

### The C3 experiment on the fixed terminal: a null result

Seven episodes, search posterior on and off, everything else equal.

| | SR | SPL | surfaces inspected |
|---|---|---|---|
| baseline | 0.429 | 0.337 | — |
| search posterior | 0.429 | 0.337 | **0** |

The two runs are **byte-identical** — same successes, same step counts, and not one surface
inspected in any episode. The posterior never ran, and the reason is upstream of it.

Every episode now ends at 43-56 steps. The agent commits to the remembered object at step
1, drives to a viewpoint, sweeps, and then either sees the object and stops (success) or
fails the absence check's precondition and stops anyway. It never reaches the phase where
"where should I look next" is asked, so the A/B was null by construction. **Two batches
have now been spent measuring a search policy through a pipeline that never invokes it**;
that is the lesson worth keeping.

The precondition is the thing to fix. Absence is only concluded where the presence filter
says a detection was *expected*, and that check runs on whichever frame the sweep ends
on -- after a full circle, the arrival heading again, which need not face the object. The
agent arrives at a ghost, sweeps past it, and concludes nothing (`absence_not_expected`
fires in all three cross-anchor episodes and in the static control).

Gating instead on "was it expected at *any* point during the sweep" fixes that, and costs
something measured: the static control then fails, because the detector misses a bowl that
really is 0.8 m in front of it and the agent abandons an object that was exactly where the
map said. So the choice today is **3/7 with an unmeasurable search policy, or 2/7 with a
measurable one** -- and it exists because the detector's silence at close range is not
reliable enough to carry the decision alone. The instrument that would settle it is a
trustworthy negative check, which is exactly what C5 wanted a vision model for and did not
get on this endpoint. Worth deciding deliberately rather than by default.

A flow graph of the pipeline and the experiment protocol is published as an artifact.

### The VLM as absence sensor: it works, it is cheap, and it is not the bottleneck

**Prompt design was the whole difference.** Measured on 20 real present/absent cases at
the agent's own bounding box:

| how the question is asked | accuracy | present | absent | median latency |
|---|---|---|---|---|
| "which of these categories are present" | 11/20 | 10/10 | 1/10 | 1.3 s |
| forced choice: object / bare / blocked | 12/20 | 6/10 | 6/10 | 1.5 s |
| **forced choice on a zoomed crop** | **17/20** | **9/10** | **8/10** | 2.0 s |

The first form scores 11/20 by answering "yes" to almost everything -- it judges
plausibility, not pixels, and a 100 px object in a 640x480 frame invites exactly that.
Making the model commit to one of three options, on a crop zoomed to the region, turns it
into a usable sensor. `blocked` maps to *no information*, never to absence: an obstructed
view must not delete objects behind doors. Those measured rates (0.9 / 0.2) are now the
sensor's `(r, q)` in the filter, replacing guesses.

**FPS is not the constraint.** The call fires only at the decision point -- an approach
that arrived and never saw its target -- which came to **3 calls across 14 episodes**.
Mean 1.9-4.3 s each, 5.7 s and 12.9 s of VLM time per seven-episode batch, and pipeline
throughput of 4.32 / 3.98 fps against 3.88 fps for the no-VLM baseline. The value-of-
information trigger does its job; a per-keyframe VLM would not have been affordable, and
is not needed.

**But the batch does not improve.** Two runs of the same configuration:

| | SR | SPL | what happened |
|---|---|---|---|
| VLM absence only | 0.286 | 0.229 | the VLM said "bare" on the **static control** -- the 1-in-10 false negative -- and the agent abandoned a bowl that was there |
| VLM absence + search posterior | 0.429 | 0.337 | same call answered correctly; `cross_anchor_03` abandoned and searched for 500 steps |

The two differ only in a flag that cannot affect the VLM call, so the difference is the
model's own nondeterminism. **One stochastic call decides an episode**, and at 90%
accuracy that is a 10% episode-flip rate -- clearly visible in a batch of seven, and a
reason to average over runs rather than read a single one.

**The real blocker moved, and it is category-versus-instance.** In four of seven episodes
the absence check never runs at all, because the agent *does* see a bowl at the ghost
location -- the HM3D dining table carries its own crockery, and the detector is asked
about a category while the benchmark scores a specific YCB instance. The agent stops on
the scene's bowl and fails. That is not something an absence sensor can fix: it is a
sighting, not a silence. The instrument for it is the terminal candidate gate -- ask the
VLM *before stopping* whether the boxed object is the target -- which this run deliberately
disabled (`verification.absence_only=true`) to isolate one variable. Turning it on is the
next experiment.

### Why four of six YCB targets were invisible — and getting three of them back

Three separate causes, only one of which was the asset.

**1. Detector resolution.** Measured at each object's best authored viewpoint, with the
pipeline's own vocabulary and the exact-label match `candidates()` performs:

| target | imgsz 512 | 768 | 960 | 1280 |
|---|---|---|---|---|
| bowl | 0.90 | 0.94 | 0.95 | 0.88 |
| plate | 0.43 | 0.23 | 0.00 | 0.14 |
| tomato soup can | 0.31 | 0.58 | 0.54 | **0.63** |
| cracker box | 0.00 | 0.39 | 0.50 | **0.62** |
| pitcher | 0.00 | 0.00 | 0.00 | 0.00 |
| scissors | 0.00 | 0.00 | 0.00 | 0.00 |
| *cost per frame* | 34 ms | 38 ms | 42 ms | 51 ms |

The default was 512. At that size the cracker box is invisible and the soup can sits
below the 0.35 admission gate; at 1280 both clear it comfortably, for 17 ms per keyframe
against a ~250 ms control loop. This was never an asset problem for these two.

**2. The vocabulary could not name them.** The detector's classes are
`DEFAULT_VOCABULARY` plus *the episode's target*, and no YCB label is in the default list.
A mapping run chasing the bowl therefore had no class for "cracker box" and could not have
mapped it whatever the resolution. The experiment now puts every YCB target in the
vocabulary.

**3. One episode does not see the house.** An episode stops when it succeeds -- the bowl
run ended at 173 steps -- so objects elsewhere were never looked at. Pointing `map_in` and
`map_out` at the same directory accumulates one map across several mapping episodes, which
is what a benchmark with several targets needs anyway.

Together those take the map from one usable target to **three**: bowl (0.06 m error,
score 0.82), tomato soup can (0.01 m, 0.81), cracker box (0.35 m, 0.75). The benchmark
grows from 7 episodes to 21.

**The two that stay out, and why it is not fixable from here.** The compressed and original
meshes are identical -- 1854 verts and 3276 triangles for the pitcher in both, textures
present in both (Basis-compressed in the one Habitat loads, PNG in the original) -- so
nothing is corrupted, and the box and can prove Basis decoding works. The pitcher simply
renders as a plain dark vessel with no handle or spout visible, and scores 0.00 at every
resolution and against every synonym tried (jug, water jug, vase, mug, cup). The scissors
render correctly and are simply too thin to survive detection at any size. Both are
recognition limits of this asset set, not data faults.

**The plate is a genuine trade-off, not a failure.** It is the one object that prefers the
*low* resolution -- 0.43 at 512 against 0.14 at 1280 -- so the setting that recovers the
can and the box loses it, and a full 500-step episode targeting it at 1280 mapped nothing.
Three targets beats two, so 1280 stands, and the plate is excluded with its reason
recorded rather than quietly dropped.

### The 21-episode baseline

Three targets x seven layouts, two-pass protocol against one accumulated map of the static
world, `verification=off`, no search posterior. This is the number future work is measured
against.

| condition | n | SR | SPL | mean distance to goal |
|---|---|---|---|---|
| in_anchor | 9 | **0.444** | 0.367 | 1.45 m |
| static (control) | 3 | 0.333 | 0.252 | 1.54 m |
| cross_anchor | 9 | **0.000** | 0.000 | 4.95 m |
| **overall** | **21** | **0.238** | **0.193** | |

| target | n | SR | SPL |
|---|---|---|---|
| tomato soup can | 7 | 0.429 | 0.350 |
| bowl | 7 | 0.286 | 0.230 |
| cracker box | 7 | 0.000 | 0.000 |

Three things this shows that seven episodes could not.

**The failure is structured, not diffuse.** Every cross-anchor episode fails and nearly
half the in-anchor ones succeed. That is exactly the shape the design predicts: when an
object moves within reach of where the map remembers it, the stale map still gets the
agent close enough to find it, and when it moves across the room the map is worse than
useless. Re-finding it from there is C3's job, and C3 remains the open problem.

**The static control is not a ceiling at 1.0.** It scores 0.333, so a third of the gap in
the dynamic conditions is not about staleness at all -- it is perception and terminal
precision on a correct map. Any dynamic-handling claim has to be read against that, not
against a perfect baseline.

**The cracker box fails everywhere despite being mapped.** It is mapped at 0.35 m error
against 0.06 m for the bowl and 0.01 m for the can, and it never converts. A goal 0.35 m
from the truth is outside the 0.18 m success radius before the agent takes a step, so its
ellipsoid centre -- not its detection -- is what needs work.

### Making cross-anchor usable: the mechanism now runs, and it still fails

Cross-anchor was 0/9 with every episode stopping at 29-58 steps, 440+ unspent. Three
things were in the way, and removing the first two exposed the third as the real one.

**1. The absence gate ended the episode.** The check ran on whichever frame the arrival
sweep finished on -- after a full circle, the arrival heading again, which need not face
the object -- so the agent swept right past a ghost and concluded nothing. Gating instead
on "was it expected at *any* heading of the sweep" turned 7 of 9 episodes from stopping at
~45 steps into abandoning and searching for 500, inspecting 7-9 surfaces each. The
mechanism works now.

**2. The candidate gates were not the blocker, though they looked like one.** One episode
did map the relocated bowl -- 0.04 m from its true pose -- and never proposed it, which
pointed at `min_obs`. Relaxing `min_obs`, `min_evidence` and `min_bbox_px` changed nothing:
in most runs the search never detects the object at its new pose at all, so there is no
track to gate. The single mapped case was a one-off.

**3. The prior does not rank the destination high enough to be reached.** Measured offline
over all nine cross-anchor relocations against the accumulated map's 112 container
surfaces:

| | median rank of the true destination | in the top 8 (what an episode inspects) |
|---|---|---|
| with proximity to the last known pose | 32 of 112 | **0 of 9** |
| without it | 20 of 112 | **3 of 9** |

Proximity encodes "displacements are usually short", and going to the old place and
finding nothing *refutes that premise* -- worse, the surfaces it favours are exactly the
ones just ruled out. Dropping the term once absence is confirmed is now implemented and
is a real improvement in ranking. It did not produce a success: with ~8 surfaces inspected
per 500-step episode and the destination sitting around 20th, the arithmetic does not
close.

**So cross-anchor remains 0/9, and the reason is now quantitative rather than mysterious.**
Closing it needs one of three things, in rough order of expected value:

- **Fewer candidates.** The map holds 112 container surfaces for a six-object scene. That
  is where the search cost goes, and much of it is spurious containers from a permissive
  membership rule. Halving the candidate set is worth more than any prior improvement.
- **Room-level reasoning.** The prior scores surfaces independently; it never asks which
  *room* the object is likely in. A cross-anchor move usually crosses rooms, and
  `graph/priors.py` already has room-level machinery used for floor selection.
- **More steps.** 500 steps is roughly 125 m of travel; inspecting 20 surfaces properly
  needs perhaps twice that. Worth measuring before it is worth optimising.

Enabling the search also costs the other conditions -- overall SR fell 0.238 to 0.143,
because the permissive absence gate false-abandons on correct maps. That trade is real and
should be settled by a better absence sensor rather than by a threshold.

---

## Phase 4 — C4 change log

**Goal:** make transitions estimable instead of asserted, and give the system a memory of
events rather than only of states.

Emit from `PresenceFilter.update()` on threshold crossings, with **hysteresis** — two
thresholds, or the log chatters when a belief sits near the boundary:

```
ℓ < −2.0  and was present  →  "disappear"
ℓ > +1.0  and was absent   →  "reappear"
```

JSONL beside the graph dump in `src/osg/graph/serialize.py`:

```json
{"kf": 412, "kind": "disappear", "track_id": 17, "label": "mug",
 "container_id": 4, "p": 0.04, "n_expected": 6}
```

Transition estimate with a Dirichlet prior centred on the Phase-3 semantic prior:

```
P(a′ | a, c) = ( n(a→a′, c) + α · prior(a′ | a, c) ) / ( n(a, c) + α )
```

α ≈ 5 means "five real observations outweigh the prior". This gives the ablation ladder
its top rung.

**The weak link, stated plainly:** turning a disappearance and an appearance into one
*move* event is re-identification, and greedy label-plus-appearance matching within a
window will make mistakes. The YCB benchmark has ground-truth object ids, so measure
re-ID accuracy as its own number. Under ~80%, keep appear/disappear as separate records
and estimate transitions from container occupancy statistics instead.

---

## Phase 5 — C5 VLM as a second sensor

**Goal:** spend VLM calls where they change a decision, and make a negative trustworthy.

Because C1 is a proper Bayes filter the VLM needs no special handling — it is a second
sensor with its own `(r, q)`, roughly `0.85 / 0.02` against the detector's `0.6 / 0.05`:

```
detector miss : Δℓ = log(0.40/0.95) = −0.87
VLM miss      : Δℓ = log(0.15/0.98) = −1.88     ≈ 2 detector looks
```

Sensor fusion costs nothing extra. Trigger on value of information: call only when the
belief is near a decision boundary (`|ℓ| < 1`) or a commitment is imminent (about to
STOP, about to abandon a container).

```python
# verification/verifier.py
def verify_absence(self, rgb, region_bbox_xyxy, categories: List[str]) -> Dict[str, bool]:
    """Box the container's projected footprint; ask which of `categories` are on
    it. Every listed category not returned is an absence observation."""
```

Keep `categories` to ≤5 — enumeration over a long list is where vision-language models
are least reliable, and a negative you cannot trust is worse than no negative.

**Opportunistic refresh is already free** once C1 runs every keyframe: passing a
container updates its objects with no extra behaviour. Only a *deliberate* turn toward a
high-value container is new, and that costs steps, so it is its own ablation.

**Reproducibility:** the README records that runs are not reproducible with the VLM
verifier on. C1 makes this worse — a VLM answer now perturbs the map itself, not just one
decision. Every C1–C4 comparison runs `verification=off`; C5 gets its own controlled A/B.

---

## Phase 6 — C6 association under motion

**Goal:** let an object that moved be recognised as the same object, and a container that
moved keep its identity.

Lowest priority: our Wasserstein associator (`src/osg/objects/association.py:36`) is
already stronger than DualMap's 2D footprint overlap, and we have no equivalent of its
"larger observation overwrites the anchor's class" bug (`utils/object.py:879`).

Tracking and re-identification are different problems with different gates, and DualMap
runs one gate for both. Per-frame tracking wants a **tight** spatial gate; recovering a
relocated object wants a **loose** gate over appearance and shape with no spatial prior.
One threshold cannot serve both. So make the loose gate **event-triggered**: only after a
disappearance event does that object's descriptor enter a re-ID pool, and only new tracks
are tested against it. That bounds false-merge risk to objects the filter already
believes have moved.

Two concrete pieces:

- **Class posterior instead of a frozen label.** `ObjectTrack.label` is set once at
  creation and never revisited; replace with `label_counts` (a `Counter` plus smoothing)
  and take the MAP label, so a consistently re-detected object can correct itself.
- **Container migration.** When a large container track disappears and a new track of
  similar *shape* (axis-ratio distance, not position) and label appears nearby, migrate
  the container id rather than creating a duplicate — resetting its objects' `p_rel`,
  since a moved table does not carry its mug.

---

## Prior art

Nothing here is novel on its own and the write-up must say so. The presence filter is
Rosen, Mason & Leonard's persistence filter (ICRA 2016) applied at object level; the
temporal-model variant is Krajník's FreMEn; Khronos (Schmid et al. 2024) already builds a
spatio-temporal metric-semantic map with absence detection. The search index is classical
discrete search theory (Koopman; Stone 1975), and semantic-prior object search is
Aydemir et al. (T-RO 2013) and Zeng et al.'s Semantic Linking Maps (ICRA 2020); the
change log's ancestor is Toris & Chernova's TemPest. Data association is Bowman et al.
(ICRA 2017). **These citations are from memory and must be verified before use.**

What is arguably ours: object-level persistence filtering *inside* an abstract/concrete
anchor split so it stays cheap; coupling a collapsed belief directly to a search
posterior rather than a ranked list; and the measurement protocol — observable vs
unobservable change, with belief latency and stale-goal rate. Even the third needs
checking against Khronos and TemPest.

---

## Verification

**Phase 0 (the gate for everything else):**

```bash
# unit — no Habitat, no HM3D licence needed
pytest tests/unit/test_containers.py tests/unit/test_scene_graph_floors.py -q

# regression — must be identical to a pre-change run on the same seed
python scripts/run_eval.py +experiment=ycb_authored_nav \
  'ycb.layout_types=[static]' 'ycb.layout_indices=[1]' verification=off eval.num_episodes=3
```

Compare SR / SPL / step counts against the pre-change baseline; inspect the new
`containers` block in the episode's `to_json` dump and confirm the counts look sane
(a bedroom should yield a bed and a nightstand, not fourteen containers).

**Per phase afterwards:** unit tests first (everything except Phase 2 and the eval runs
is simulator-free by construction), then the YCB run with `verification=off`, then the
metric that phase exists to move — Phase 1 stale-goal rate, Phase 3 containers-inspected
and post-failure distance, Phase 4 re-ID accuracy.

**Standing rule:** no A/B that claims two configs are equivalent may run with the VLM
verifier on.
