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

> **Superseded in part (2026-08-19).** The resolution table below was measured against
> a contaminated ground-truth mask and three of its six rows are wrong; the pitcher is
> recoverable and the choice of `imgsz 1280` it justified was the direct cause of the
> cracker box never converting. See *The cracker box: fifteen tracks for one box*.

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

### Room reasoning, and the measurement that ended the search line

Room-level reasoning was the recommended next lever. It is implemented, and on this scene
it does not help — for reasons worth recording, because they are about the benchmark
rather than the idea.

**The room prior has almost nothing to discriminate with here.** The accumulated map
segments six rooms, and inferring each one's type from its contents gives **three bedrooms
and three bathrooms — no kitchen at all**; this HM3D scene is a hotel suite. A prior that
says "a bowl belongs in a kitchen" cannot choose among three bedrooms. Measured offline
over all nine relocations against 112 candidate surfaces:

| variant | median rank of true destination | in the top 8 |
|---|---|---|
| surface prior only | 20 | 3/9 |
| + room weighted by how many plausible surfaces it holds | 19 | 3/9 |
| + room weighted by inferred type | 20 | 3/9 |

A related check: only **6 of 9** cross-anchor relocations actually cross a room boundary,
so even a perfect room prior would leave a third of them untouched. (An earlier version of
this analysis reported 3/9 because it read room id 0 — which means *unassigned*, not a
room — as a room; using the scene graph's own nearest-room fallback fixes that.)

**Ordering the search by room does help, modestly, and it is implemented.** Simulating both
orders over the real surface layout: a plain global argmax reaches the destination after a
median 45 inspections and 40 m, room-grouped after 37 and 33 m. `search_same_room_bonus`
keeps the agent finishing a room before crossing the house.

**The simulation also produced the number that ends this line of work.** The target is
reached after a median 37-45 inspections — and a real inspection costs the agent about
**fifty steps** (approach, arrival, commitment budget), so a 500-step episode buys seven to
nine. That is a 4-5x gap, so `search_glance_detect_prob` now lets a surface in plain view
count as searched without driving to it, on the theory that the binding budget was
inspections rather than travel.

It was not enough, and then the decisive test: **raising the budget to 2000 steps still
gives 0/3 on cross-anchor**, with every episode running the full budget. So the search is
not merely starved. The reason is detection range: a bowl scores 0.90 from an authored
viewpoint 0.8 m away and nothing at all from the 2-4 m a passing search affords, so
"inspecting" a surface really means approaching it to within a metre and looking — the
fifty steps — for each of 112 candidates. In-anchor works precisely because the object
stays within about a metre of its remembered pose, so the approach to the ghost brings it
into detection range anyway.

**Conclusion for C3 on this benchmark.** Cross-anchor is not a search-policy problem, it
is a detection-range-times-candidate-count problem, and no reordering of the same
expensive inspections closes it. What would: a detector that recognises these objects at
3-4 m (higher resolution helped once already, 512 to 1280, and more may be available), or
far fewer candidate surfaces, or a sensor that can check a surface without standing at it.
Ordering was never the binding constraint, and two batches spent on priors would have been
better spent measuring inspection cost first.

### Matching DualMap's protocol: several attempts per query

Two corrections to how this has been evaluated, both of which made our setup **stricter
than the system we are comparing against** rather than fairer.

**500 steps is the standard budget**, so the earlier 2000-step run should be read as a
diagnostic (it showed the search is not merely starved) and never as a proposed setting.
Everything below is at 500.

**DualMap allows a query several navigation attempts**: when one fails it updates the map
and goes again — that is what its "exceeding navigation attempt limits" failure bucket
counts. Scoring a single attempt, as every batch here did until now, is a harsher rule than
theirs. `eval.attempts` now matches it: when the agent decides to STOP and that decision
would not score, the map keeps everything it has learned — presence beliefs, searched
surfaces, objects mapped along the way — the candidate it stopped on is blacklisted, and it
chooses again, all within the same 500 steps.

Scoring an attempt without ending the episode needs the criterion evaluated directly, since
Habitat only scores STOP and STOP also terminates: `_attempt_succeeded` computes geodesic
distance to the nearest goal view point under the same `success_distance`.

It works mechanically — the cracker box episodes now take three attempts (stopping at
steps 52 and 101 before a third), where before they ended at the first — and it produced
**no successes**: cross-anchor 0/9, static 0/3.

**And it surfaced the thing to fix next.** One episode, `static bowl`, **ended 0.126 m from
the goal — inside the 0.18 m success radius — and never stopped**, running the full 500
steps with `stop_reason: None`. It was standing on the answer. That is not search and not
staleness; it is the terminal decision failing to fire, and it is now the clearest
single-episode win available. The same configuration scored that episode 0.06 m and
successful two commits ago, so something in the recent search changes (the glance
retirement, the same-room bonus, or dropping proximity once moved) has made the agent
wander past a target it had already reached. That regression is worth bisecting before any
further search work.

### Bisecting the regression: it was the blacklist, not the search

The previous entry blamed one of the search changes for an episode that ended 0.126 m from
its goal without stopping. **That was wrong**, and bisecting says so plainly: with the
search posterior turned off entirely the same episode fails identically, and so do variants
with the glance retirement off and the same-room bonus off. Four variants, one outcome.

What every failing variant does share is `absence_abandon: 1` — on a **correct** map the
agent walks to where the bowl really is, does not see it, concludes absence, and
**blacklists the track permanently**. When it later stands 0.088 m from the goal, the only
track that could have been the answer has been struck off, so it cannot stop. It finishes
the episode standing on the answer.

The blacklist was mine, and it contradicts C1's own premise that no state is absorbing --
written into the spec as "never hard-delete: an instance at p≈0.02 stays in the map and can
be resurrected", then violated in the code that acts on absence. Absence now lowers the
belief and nothing else; `min_presence` keeps a disproved track out of the candidate list
until evidence brings it back. The threshold comes from the arithmetic rather than taste:
from a belief reloaded at 0.82, one detector-strength absence reading lands at 0.485 and a
second at 0.166, while one VLM-strength reading lands at 0.359 — so 0.45 makes a single
round of detector silence non-decisive (measured, it misses a bowl 0.8 m in front of it),
two rounds decisive, and one VLM answer decisive on its own. Two sensors with different
error rates should carry different weight; that is the point of having both.

**Result, 21 episodes, three attempts per query, 500 steps:**

| condition | n | SR | SPL |
|---|---|---|---|
| in_anchor | 9 | **0.556** | 0.376 |
| cross_anchor | 9 | 0.000 | 0.000 |
| static (control) | 3 | 0.000 | 0.000 |
| **overall** | **21** | **0.238** | **0.161** |

In-anchor is the best it has been -- 0.444 in the baseline, 0.333 while the blacklist was
in -- and the **cracker box converts for the first time** (SPL 0.873), having been 0/7
throughout. Per target: bowl 0.429, cracker box 0.143, soup can 0.143.

**Two things remain open, and the second is the more troubling.** Cross-anchor is still
0/9, for the detection-range reason established above. And the static control is still
0/3: on a map that is *correct*, the agent still concludes absence at the target's true
location, because the detector genuinely does not see these objects from the viewpoint the
approach ends at. The belief machinery then behaves exactly as designed on evidence that is
wrong. Until the detector can confirm a small object at the range the approach terminates
at, the control condition will keep undercutting every dynamic number measured against it.

### "Detection at terminal range" was not a detection problem

The static control was failing because the agent concluded absence at the target's true
location, and the obvious reading was that the detector cannot see a small object from
where the approach ends. Saving the frame the decision was actually taken on settles it:
**a wall and a painting**, with the bowl's table off frame to the right. The VLM answered
"bare" and was correct about the pixels it was shown.

The cause is the arrival sweep. A viewpoint is a pose the object is visible *from*, but the
navmesh follower arrives on whatever heading the path ended with, and the blind
twelve-turn sweep ends on the heading it began with — so the absence decision was being
taken on whatever happened to be in front. The sweep now turns **toward** the object and
stops as soon as it is facing it, because that is the informative frame; a miss there means
something, a miss pointed at a wall does not.

One three-turn correction restores the control:

| | SR | SPL | static | in_anchor | cross_anchor |
|---|---|---|---|---|---|
| before | 0.238 | 0.161 | 0.000 | 0.556 | 0.000 |
| facing the target | **0.286** | **0.197** | **0.333** | 0.556 | 0.000 |

The static bowl goes from 500 steps and an abandon to **success in 46 steps at SPL 0.757**,
with three face-turns and no abandon at all. Per target, bowl rises to 0.571. This is the
best overall number so far, and more importantly the control condition is no longer
undercutting everything measured against it.

It is worth naming the mistake in the earlier diagnosis: "the detector cannot see it at
this range" was inferred from an outcome, while "the agent is looking at a wall" was
visible in one saved frame the whole time. The absence machinery had been behaving
correctly on evidence that was worthless, and no amount of tuning recall, thresholds or
sensors would have fixed a heading.

Cross-anchor is unchanged at 0/9, and that remains a detection-range-times-candidate-count
problem rather than a heading one.

### Cross-anchor works: the search was never being driven to

Every C3 result in this document up to here measured nothing, and the log entry that
gives it away had been sitting in the data for several rounds:

```
chose desk  prior=3.20 cost=2.3   -> searched 288 arrived=False factor=0.8
chose desk  prior=2.56 cost=2.3   -> searched 288 arrived=False factor=0.64
chose desk  prior=2.05 cost=2.3   -> searched 288 arrived=False factor=0.512
... eight times, one surface, an unchanged 2.3 m path cost
```

An unchanged path cost means the agent never moved. Only `State.GOTO_FRONTIER` follows
`_goal_xy`; the surface selection set the goal and left the state at `EXPLORE`, so the
agent stood still, re-selected the same surface five steps later, and scored it "never
reached" each time. Every "8 surfaces inspected" in the earlier batches was eight
selections that were never acted on. The priors, the room term, the glance retirement, the
proximity question -- all of it was tuning the ranking of a list nobody walked to.

Two fixes together make the search real: drive to the chosen surface (set the state that
follows the goal), and aim at a pose you can **stand in** rather than the middle of the
furniture -- a container's centre is inside the desk, so arrival could never register even
once the agent did move. Alongside them, the candidate set is now filtered to surfaces
that are really there (seen twice, scored above 0.5, duplicates merged): 112 -> 46, which
takes a cross-anchor destination from rank 20 to rank 7.

**21 episodes, three attempts, 500 steps:**

| condition | n | SR | SPL |
|---|---|---|---|
| bowl | 7 | **0.714** | |
| tomato soup can | 7 | 0.286 | |
| cracker box | 7 | 0.143 | |
| in_anchor | 9 | 0.444 | 0.311 |
| **cross_anchor** | 9 | **0.333** | 0.086 |
| static | 3 | 0.333 | 0.252 |
| **overall** | **21** | **0.381** | **0.206** |

**Cross-anchor goes 0.000 -> 0.333**, and the successful episodes look like the design
intends: go to the remembered place, find nothing, abandon it, inspect three or four
plausible surfaces, find the object at its new home 0.05-0.12 m from the goal. Overall SR
0.286 -> 0.381. SPL on those episodes is low (0.086) exactly as it should be -- the agent
walks a long way to re-find something -- and that is the honest signature of re-search
rather than luck.

The three cracker box episodes still never search (`chose=0`): they stop three times
without ever abandoning, which is a separate thread.

The lesson is the same one this document keeps recording: the failing data was
self-diagnosing several rounds before it was read. "Constant path cost, arrived=False,
same surface" says "it never went" and nothing else.

### The cracker box: fifteen tracks for one box

The cracker box had never converted in a cross-anchor episode, and unlike every other
failure in this document it did not look like search or staleness. Six of its seven
episodes ended at *exactly* 194 steps with all three attempts spent, standing 8–11 m from
the goal, on a map where the box is mapped 0.15 m from the truth. The agent was not
failing to find the cracker box. It was finding the wrong one, three times, and running
out of tries.

**The accumulated map holds fifteen tracks labelled "cracker box" for a house containing
one.** The bowl has one; the soup can has one. Under the run's own candidate gates
(`min_obs=1`, `min_score=0.30`, `min_bbox_px=200`, `min_evidence=0.2`, `min_presence=0.45`):

| label | tracks in map | pass the gates | ranked candidates, by distance to the real object |
|---|---|---|---|
| cracker box | 15 | 5 | 4.29 m, 2.29 m, 3.85 m, 2.91 m, 1.55 m |
| tomato soup can | 1 | 1 | **0.01 m** |
| bowl | 1 | 1 | **0.06 m** |

Every candidate the agent can choose for "cracker box" is a false positive. The two tracks
that *are* the cracker box — 0.15 m and 0.35 m from the authored pose, best scores 0.61
and 0.75 — sit at the belief floor, p = 0.0025, and `min_presence` excludes them before the
episode takes a step. Three attempts, three imposters, 194 steps.

Three independent causes, none of them visible from an episode log, plus a fourth found
while checking the repair.

#### 1. `imgsz` is a precision decision, and it was made on recall alone

`scripts/probe_ycb_detection.py mode=fp` puts the agent at 300 random navigable poses,
runs the benchmark's own vocabulary, and asks of every detection carrying a YCB label
whether it overlaps that YCB object:

| imgsz | "cracker box" TP | FP | FP ≥ 0.30 gate | max FP score |
|---|---|---|---|---|
| 512 | 1 | 7 | **0** | 0.18 |
| 768 | 1 | 17 | **2** | 0.39 |
| 960 | 1 | 33 | 5 | 0.46 |
| 1280 | 2 | 125 | **28** | 0.66 |

Every other YCB label is clean at every size — the soup can produces zero false positives
at 1280, the bowl three, none above 0.16. "Cracker box" alone is promiscuous, and it gets
worse with resolution: at 1280 the detector calls 28 things in this house a cracker box
with more confidence than the gate demands, and each of those is a track.

#### 2. The table that chose 1280 was measured against a contaminated ground truth

The earlier probe scored a detection by its overlap with `semantic == authored.semantic_id`.
The collector's semantic ids are 26–95; this HM3D scene annotates 252 instances as 0–251.
So `95` is the cracker box **and** `picture_95`, `35` is the plate **and** `wall_35`. When
the twin was in frame the "ground-truth" mask was the union, the IoU test failed, and a
perfectly good detection was recorded as 0.00. The plate's mask reached 124506 px of a
307200 px frame.

*The benchmark itself was never affected*, and it is worth being precise about why: its
manifest simulator carries a semantic sensor **alone**, and habitat then renders the whole
scene as a single blob, so the ids never collide there. Attach a colour sensor — as any
probe must — and they do. `_viewpoints_for_object` now raises if a target mask covers more
than a fifth of the frame, because a 25 cm object at 0.8 m occupies under 5% and anything
larger is scene geometry wearing the object's id. Phase 4 plans to use these ids as
ground truth for re-ID accuracy; it would have inherited the same bug silently.

Re-measured with the injected ids offset out of the scene's range — recall at the 0.30
admission gate over 20 navigable viewpoints per object, which is a harder and more useful
number than the best single view the old table reported:

| target | imgsz 512 | 768 | 1280 |
|---|---|---|---|
| bowl | 0.90 | 0.90 | 1.00 |
| plate | 0.60 | **0.65** | 0.15 |
| tomato soup can | 0.40 | 0.90 | 1.00 |
| cracker box | 0.20 | 0.70 | 0.90 |
| blue plastic pitcher | 0.60 | 0.60 | — |
| scissors | 0.00 | 0.00 | 0.00 |

The cracker box was never 0.00 at 512, and the plate was never 0.00 at 960. **768 is the
operating point**: it keeps recall usable for every target, brings the plate back as a
usable target rather than a documented casualty, and costs 4 ms per keyframe over the 512
default against a ~250 ms control loop.

#### 3. The true track is disbelieved, because the recall model is a constant

The presence filter sizes every negative update by `P(detected | present, this view)`, and
that has been a constant 0.6 since C1 landed — described in its own docstring as "the
honest default before anything is fitted". The map's own counters say what it should have
been. Detection-when-expected, from `n_expected` and `n_missed` on the tracks that really
are the object:

| track | n_expected | n_missed | empirical recall | log-odds |
|---|---|---|---|---|
| cracker box (0.35 m) | 41 | 37 | **0.10** | −6.0 |
| cracker box (0.15 m) | 31 | 31 | **0.00** | −6.0 |
| bowl (0.06 m) | 40 | 28 | 0.30 | +3.0 |
| soup can (0.01 m) | 3 | 0 | 1.00 | +1.5 |

At r = 0.6 a miss costs log(0.4/0.95) = −0.87; at the measured r = 0.10 it should cost
log(0.9/0.95) = −0.05. Seventeen times too much, thirty-seven times over, and the belief
is on the floor.

The reason the filter is expecting detections it cannot get is a coupling that was
deliberate and correct in isolation. `min_area_px` is bound to the layer's admission gate
— "expecting a detection at a size ObjectLayer would have filtered out anyway
manufactures a false negative on every distant object in the room" — and that gate was
lowered from 1500 px to 300 px so small YCB objects could be mapped at all. Lowering it
also extended the filter's expectation horizon from ~1.8 m to ~2.5 m. Measured recall over
that range, at 768, pooled across the four detectable targets:

| viewpoint radius | 0.8 m | 1.2 m | 1.6 m | 2.0 m | 2.5 m | 3.0 m | 4.0 m |
|---|---|---|---|---|---|---|---|
| recall at the gate | 0.79 | 0.58 | 0.48 | 0.35 | 0.25 | 0.15 | 0.12 |
| median apparent size | 1600 px | 857 px | 515 px | 347 px | 194 px | 127 px | 75 px |

The constant 0.6 is right at about 1.2 m and wrong everywhere else. A change made to admit
small objects propagated, through a coupling written to prevent exactly this class of bias,
into the belief update — which is a nice illustration of why the two thresholds were tied
together in the first place, and why tying them was not enough.

`scripts/fit_recall_model.py` and `PresenceConfig.recall_model_path` have existed since C1
and had never been run. Fitted on 13692 logged expectations from one mapping pass
(detection rate 0.366):

```
bias -2.2015   log_area_px +0.3545   depth_m +0.8424   cos_incidence -3.2344
Brier 0.2123 against 0.2321 for the constant predictor
```

The calibration table is monotone and close over eight bins. The `depth_m` coefficient is
positive, which reads backwards until you remember it is conditioned on apparent area: at
a fixed number of pixels, more distance means a physically larger object, and those are
easier. It is a confound, not physics, and the honest reading of this fit is the
calibration table rather than the signs.

#### 4. A rejection from one episode was being saved into the map forever

Found while checking the rebuilt map: **exactly one track of 587 is saved
`blacklisted`, and it is the pitcher's only correct track** — 0.00 m from the authored
pose, score 0.68, belief 0.82. Every pitcher episode of the next benchmark would have been
unwinnable before it started, and nothing in an episode log would have said why: the track
is in the map, at the right place, with a good score and a healthy belief, and
`candidates()` skips it on the first line.

The blacklist is an episode-scoped device — *this attempt already tried that candidate,
choose another* — set by `_rearm_agent` between attempts and by `_check_candidates` for an
unreachable or VLM-rejected candidate. `save_map` records it and `apply_map` restores it,
so a rejection taken under one episode's evidence becomes a permanent strike-off in every
later session, including sessions where the world has since changed.

This is the same mistake as blacklisting on absence, one layer down, and the earlier fix
named the principle it violates: *no state is absorbing; an instance at p≈0.02 stays in the
map and can be resurrected*. `apply_map` now clears the blacklist on load, alongside the two
things it already refused to carry across intact — a saturated belief and raw frame ids.
Keeping a disproved track out of the candidate list is `min_presence`'s job, because that
is the version a later detection can undo.

#### The map after the fixes

One mapping pass over six targets at 768, otherwise the same two-pass protocol:

| label | tracks (1280) | tracks (768) | pass the gates | best candidate's error |
|---|---|---|---|---|
| cracker box | 15 | **1** | 1 | **0.02 m** |
| tomato soup can | 1 | 1 | 1 | 0.00 m |
| bowl | 1 | 6 | 6 | 0.09 m |
| plate | — | 2 | 2 | 0.03 m |
| blue plastic pitcher | — | 1 | 1 | 0.00 m |
| bleach bottle | — | 1 | 1 | 0.01 m |

The cracker box now has exactly one track, 0.02 m from the authored pose. Every target is
mapped within 3 cm and every one of them is proposable. The map is larger overall (587
tracks against 448) because episodes that no longer stop on an imposter run their full
budget and see more house.

The bowl's several tracks are not a regression of the same kind: the two best are the real
bowl at 0.09 m and 0.10 m, so the ranking is right even though the label is looser than it
was. And the bleach bottle is what shows the third cause mattering on its own: under the
constant recall it was mapped at 0.00 m and still failed the gates, 11 expectations and 8
misses taking it to p = 0.11; under the fitted model the same object reads p = 0.95 and is
proposable. The cracker box's own track tells the same story from the other side — 21
expectations, 5 misses, an empirical recall of 0.76 where the old map recorded 0.10.


#### Results: the cracker box triples, and its failures move to the last metre

42 episodes — six targets × seven layouts — three attempts, 500 steps, the same protocol
as the 21-episode batch: **SR 0.405, SPL 0.219**. That number is not comparable to the
0.381 baseline, because two of its six targets did not exist before. The comparable thing
is the same three targets over the same seven layouts:

| | baseline (21 eps) | after (21 eps) |
|---|---|---|
| static | 0.333 / 0.252 | 0.333 / 0.231 |
| in_anchor | 0.444 / 0.311 | 0.444 / 0.314 |
| cross_anchor | 0.333 / 0.086 | **0.444** / 0.095 |
| **overall** | **0.381** / 0.206 | **0.429** / 0.208 |

| target | SR before | SR after |
|---|---|---|
| cracker box | 0.143 | **0.429** |
| tomato soup can | 0.286 | 0.286 |
| bowl | **0.714** | 0.571 |

**The cracker box result is not really the success rate; it is the shape of the failures.**
Distance to goal at the end of its seven episodes:

```
before   0.13   3.35   4.56   8.25  11.17  11.45  11.46      (six ended at 194 steps,
after    0.03   0.08   0.11   0.16   0.67   0.92   0.95       all three attempts spent)
```

Every cracker box episode now ends within a metre of the target; five of seven used to end
between three and eleven metres away, having spent all three attempts on imposters at 194
steps. The map was the whole disease. What is left is a different and much smaller problem:
**one episode ended 0.03 m from the goal — inside the 0.18 m success radius — and ran the
full 500 steps without stopping**, which is the terminal-decision failure this document has
now recorded three times.

**The bowl paid for it, and the mechanism is the one just fixed, arriving somewhere else.**
Bowl cross-anchor goes 1/3 to 0/3, and the losing episodes end 3.2-6.1 m out with all three
attempts spent — the cracker box's old signature exactly. The cause is the fitted recall
model doing its job indiscriminately: it correctly says "you would not have detected that
anyway" for a small distant track, which keeps the *real* bleach bottle proposable, and
keeps six bowl tracks alive too, including false positives at 2.05 m, 3.32 m and 4.92 m.
Precision was bought at the detector for the cracker box and given back at the filter for
the bowl. A per-track prior on how many instances of a class a house holds, or a
false-positive channel that is not the same channel as absence, is the next thing to try;
the current design has only one way to say "this track is not real", and it is the same
way it says "this object has moved".

Of the 25 remaining failures, 16 end more than 4 m from the goal and 5 end within a metre:
this benchmark is still mostly a search problem, not a terminal-precision one, and the
cracker box is now the exception rather than the rule.

### The pitcher was a naming problem; the scissors are not recoverable

Both had been recorded as "recognition limits of this asset set". One of them was not.

**The pitcher renders correctly.** The mesh is 14.9 × 14.5 × 24.2 cm with the handle in the
geometry, and the texture in the shipped `.glb` is the blue Sterilite pitcher, unaltered —
so "renders as a plain dark vessel with no handle or spout" was a description of a 50 × 97
px crop, not of a broken asset. What was broken was the prompt. Candidate names swapped
*into* the vocabulary in place of "pitcher", one at a time, recall at the 0.30 gate over 20
viewpoints:

| name | imgsz 512 | 768 |
|---|---|---|
| **blue plastic pitcher** | **0.60** | **0.60** |
| blue pitcher | 0.35 | 0.50 |
| water pitcher | 0.15 | 0.25 |
| kettle | 0.10 | 0.15 |
| jug / water jug / plastic jug / watering can | 0.00 | 0.10 |
| carafe / vase / bucket | 0.00 | ≤0.05 |
| pitcher | **0.00** | **0.00** |

Swapped, not added, and the distinction is the finding. An open-vocabulary head runs
class-competitive NMS across its own vocabulary, so putting "tin can" beside "tomato soup
can" hands the detection to the synonym and the target label reads 0.00 — which is how a
probe that adds every candidate at once measures the competition rather than the name. The
earlier survey tried jug, water jug, vase, mug and cup, all single nouns, and concluded the
asset was at fault.

**The scissors are genuinely gone.** Eight names, two resolutions, twenty viewpoints, and
0.00 recall at the gate in every cell; the best single score anywhere is 0.27. The object
is a median 315 px — about a thousandth of the frame — and what the detector *does* say
about it is "lamp" at 0.40. There is nothing here to rescue.

So the slot is refilled instead. `scripts/author_substitute_layout.py` swaps the asset and
keeps everything else: same x and z, same rotation, same anchor, same relocation structure
across static / in_anchor / cross_anchor. Only the height is recomputed, because two meshes
have different origin-to-base offsets and a pose authored for flat-lying scissors would
bury a bottle to its shoulders — the replacement's AABB base is placed on the plane the
scissors rested on. Four candidates, measured at the scissors' own poses:

| replacement | median apparent size | recall @512 | @768 | @1280 |
|---|---|---|---|---|
| **021_bleach_cleanser** ("bleach bottle") | 2068 px | 0.30 | **0.45** | 0.70 |
| 006_mustard_bottle | 1389 px | 0.30 | 0.30 | 0.35 |
| 053_mini_soccer_ball | 1335 px | 0.10 | 0.30 | 0.15 |
| 077_rubiks_cube | 455 px | 0.00 | 0.00 | 0.05 |
| *037_scissors (being replaced)* | *315 px* | *0.00* | *0.00* | *0.00* |

`021_bleach_cleanser` wins at the 768 operating point and is checked for the fault that
started this section: over 300 random poses "bleach bottle" produces one false positive
above the gate and "blue plastic pitcher" none.

**Nothing under `data/` is edited.** The source root is opened read-only and a new root is
written (`outputs/substituted_layouts`), the same separation `import_collector_layouts.py`
already keeps between DualMap's authoring session and ours, and each rewritten layout
records its substitution and its source path in its own `authoring` block.
### Three scenes, a standable goal, and a recoverable rejection

Three things changed together — the benchmark grew from one scene to three, the terminal
approach stopped aiming at cells the agent cannot stand in, and a failed attempt stopped
striking its candidate off. They are reported together because they were measured together;
the per-lever attribution is an ablation that has not been run.

#### The benchmark is three scenes now, and one of them needed a decision

`data/dualmap/HM3D_collect` holds three collected scenes, not one, each with a static
layout and three in_anchor plus three cross_anchor relocations. 00880 imported unchanged.

**00848 places two mugs and gives both semantic id 98.** The schema keys goals, viewpoints
and relocation pairs on that id, so two objects wearing it is not a scene with two mugs —
it is a scene where "the mug" has no referent. The collector's data is already inconsistent
about it: one of the seven layouts omits the second mug, so the object sets do not match
across layouts either. The importer now keeps the first instance of a duplicated id, drops
the rest, and records what it dropped in the layout's `authoring` block. It costs a
distractor that is never a target here and buys a usable scene.

**Targets are chosen per scene by measurement, not assumed.** Detectability is not a
property of the asset alone — the same cracker box mesh reads 0.75 in one scene and 0.00 in
another. Recall at the 0.30 gate, 20 viewpoints per object, imgsz 768:

| target | 00829 | 00848 | 00880 |
|---|---|---|---|
| bowl | 0.90 | — | 1.00 |
| tomato soup can | 0.90 | **0.00** | 0.25 |
| cracker box | 0.75 | 0.40 | **0.00** |
| plate | 0.65 | 0.85 | 0.50 |
| blue plastic pitcher | 0.60 | 0.20 | 0.35 |
| bleach bottle | 0.45 | 0.90 | 0.65 |
| banana | — | 0.25 | — |
| mug | — | **0.00** | — |

The zeroes are excluded, giving 6 + 5 + 5 targets over six dynamic layouts each: **96
episodes**. Static is no longer a scored condition — the static comparison belongs on HM3D's
own ObjectNav split — but the static layout is still what pass 1 maps.

#### "box" and "book" were eating the targets

00848's cracker box scores **0.00 at all twenty viewpoints** while occupying a median 5185
px. It is authored side-on: the nutrition panel reads as a menu, and `book` (0.76) and
`box` (0.72) win the class-competitive NMS that an open-vocabulary head runs over its own
vocabulary. This is the same effect as the pitcher's, seen from the other side — there the
target label was wrong, here two generic labels were right enough to steal it.

Dropping the two classes, recall at the gate:

| target | 00829 | 00848 |
|---|---|---|
| cracker box | 0.60 → **0.75** | 0.00 → **0.40** |
| tomato soup can | 0.77 → **0.90** | 0.00 |
| plate | 0.47 → **0.65** | 0.85 |
| bowl | 0.93 → 0.90 | — |

Nothing measurable is lost, and the container-surface candidate set the search posterior
walks gets smaller too. The general lesson is worth stating plainly: **on an
open-vocabulary detector the vocabulary is a hyperparameter of every class in it, not a
free list.** Adding a name takes probability from its neighbours.

#### The approach goal was a cell the agent cannot stand in

When `ViewpointPlanner.approach_viewpoint()` found nothing — no ring pose that was mapped
FREE with clear line of sight — `_start_approach` fell back to the object's own centre. For
a tabletop object that is an occupied cell inside the furniture, so the follower stalls
against it and the agent ends up **inside** the innermost 0.8 m ring, where HM3D's success
criterion cannot fire however well the object was found. Over the previous 42-episode run:

| final approach goal cell | n | SR | where the agent stalled |
|---|---|---|---|
| `occupied` | 10 | **0.100** | 0.54–1.51 m, never inside 0.18 m |
| `free` | 31 | **0.516** | |

The fallback fired 23 times in 42 episodes. It is not a rare path.

The fix is to relax the viewpoint search rather than abandon it: accept UNKNOWN cells
(unmapped is not unstandable — the navmesh follower finds out for real) and drop the
line-of-sight test (on a 2D costmap the blocking cell is usually the object's own table).
Any pose ON a ring beats any pose off it. Measured on 00829's 36 dynamic episodes, the
relaxed search fired 22 times and returned a pose all 22 times, and **episodes ending on an
occupied goal cell went from 8 to 0**.

#### A failed attempt is not a permanent verdict

`_rearm_agent` blacklisted the candidate an attempt had just failed on. That is the same
absorbing-state mistake as blacklisting on absence and as persisting the blacklist through
`apply_map`, and it bites hardest when the map is RIGHT: five of 42 episodes committed once
to a track 0.00–0.39 m from the true object, failed, struck it off, and then had no way to
stop — three cracker box episodes finished 0.67–0.95 m from the goal with 429 steps unspent.

A failed attempt now applies a VLM-strength negative reading and holds the belief one
detector-step under `min_presence` (p ≈ 0.255), so the next attempt must choose differently
while one later detection (+2.5, to p ≈ 0.807) brings the track straight back. Of 45
multi-attempt episodes, **40 re-committed after a failure**, which was structurally
impossible before.

**And it introduced an oscillation, in 7 of 96 episodes.** A false-positive track that the
detector keeps re-detecting cycles: commit, walk, fail, belief clamped under the bar, detect
again, commit again. One 00848 episode did this 251 times against a single track 7.9 m from
the truth. The clamp is undone by exactly one detection, which is the property that makes it
recoverable and also the property that makes it oscillate. The fix is not to restore the
blacklist but to make the *count* of failed attempts on a track persist — a per-track
penalty that raises that track's bar each time, recoverable but not free. That is not
implemented.

#### Results

96 episodes, three scenes, dynamic conditions only, three attempts, 500 steps:

| | n | SR | SPL |
|---|---|---|---|
| in_anchor | 48 | 0.438 | 0.276 |
| cross_anchor | 48 | 0.292 | 0.129 |
| **pooled** | **96** | **0.365** | **0.203** |

| scene | n | SR | SPL |
|---|---|---|---|
| 00829-QaLdnwvtxbs | 36 | 0.472 | 0.271 |
| 00880-Nfvxx8J5NCo | 30 | 0.367 | 0.246 |
| 00848-ziup5kvtCCR | 30 | 0.233 | 0.080 |

The spread across scenes is wider than any intervention measured in this document, which is
the strongest argument yet that one scene was never enough to conclude anything from.

On 00829, where a comparable prior run exists (same 36 dynamic episodes):

| | before | after |
|---|---|---|
| in_anchor | 0.389 / 0.290 | **0.500** / **0.366** |
| cross_anchor | 0.444 / 0.138 | 0.444 / 0.176 |
| **overall** | **0.417 / 0.214** | **0.472 / 0.271** |

Per target on that scene: cracker box 0.500 → 0.667, soup can 0.333 → 0.500, pitcher
0.167 → 0.333, plate 0.167 → 0.333, bowl 0.500 → 0.500, bleach bottle 0.833 → 0.500. The
bleach bottle is the one regression and at n = 6 it is one episode's worth of noise either
way; it is recorded rather than explained.

**The remaining failure is now one thing.** Pooled over 96 episodes:

| | n |
|---|---|
| committed, succeeded | 35 |
| **only ever committed to wrong tracks** | **50** |
| committed to the right track and still failed | 8 |
| never committed | 3 |

Fifty of the sixty-one failures never see a correct candidate. Not a search problem, not a
terminal problem, not staleness: false-positive tracks outranking the real object, which is
the cracker box's original disease generalised to every target. The recall model made it
worse by design — it correctly excuses a missed detection on a small distant track, and a
false positive is exactly a small distant track. The system has one channel for "this track
is not real" and it is the same channel it uses for "this object has moved", and until those
are separated no amount of search or terminal work will move the number.

One episode of 96 (00880, bleach bottle, cross_anchor) ended on a disconnected navmesh
island with an infinite geodesic to every goal viewpoint. It is a failure either way; SPL and
mean distance are computed over the finite 95.
### What the remaining failures actually are, and the two experiments that answer them

The first classification of the 96-episode run said "50 of 61 failures never commit to a
correct candidate", and that was wrong in a way worth naming: for a *dynamic* episode,
committing to the remembered pose is the correct opening move, not a false positive.
Scoring commits against the object's post-move position counts a perfectly good first
attempt as an error. Splitting on the pre-move position too:

| failure family | n |
|---|---|
| went to the remembered pose and never re-found the object | **28** |
| only ever false positives | 13 |
| the remembered pose, then false positives | 9 |
| reached the right track and still failed | 8 |
| never committed | 3 |

So the dominant failure is the one the system exists to solve, and the false-positive
problem is real but half the size. **All 28 of the first family never detected the object at
its new pose at all** — not once, in any frame, at any point in 500 steps — and 27 of the 28
ran the search. There is no gate to relax and no ranking to improve: the candidate was never
created.

**The arithmetic, measured on this run.** The search selects a median of 14 surfaces per
episode and *arrives* at a median of **5**. It is not thrashing — 401 of 676 consecutive
selections re-select the same surface and only 40 switch away before arriving — so five
inspections per 500 steps is simply what an inspection costs when it means driving to within
a metre of a surface. Against a house, five is not enough, and no reordering of the list
changes that.

That makes detection *range* the variable to move, and there are two ways to try.

#### Cropping does not work, and it is worth knowing why

If a distant surface could be checked by cropping its projected region and upsampling it to
the detector's own `imgsz`, an inspection would cost nothing and the arithmetic would close.
Measured on 00829, same frames scored both ways, recall at the 0.30 gate:

| range | whole frame | crop, upsampled |
|---|---|---|
| 0.8 m | 0.66 | 0.56 |
| 1.6 m | 0.40 | 0.20 |
| 2.5 m | 0.19 | 0.15 |
| 3.0 m | 0.19 | 0.05 |
| ≥ 2.5 m pooled | **0.15** | **0.06** |

It is worse everywhere. Interpolation adds no information — an object 76 px across at 4 m is
76 px of sensor data however large the array holding it — and the crop also throws away the
context the detector uses. This is the same lesson as the earlier `imgsz` sweep, in reverse:
resolution helps only where there are photons behind it.

#### More sensor pixels do work, and the cost is legible

Rendering at 1280×960 instead of 640×480 adds real information. Recall at the gate, 00829,
five targets, against range:

| range | 640×480, imgsz 768 | 1280×960, imgsz 1280 |
|---|---|---|
| 0.8 m | 0.61 | 0.62 |
| 1.2 m | 0.50 | 0.66 |
| 1.6 m | 0.36 | **0.56** |
| 2.0 m | 0.24 | **0.57** |
| 2.5 m | 0.17 | **0.59** |
| 3.0 m | 0.10 | 0.33 |
| ≥ 2.5 m pooled | 0.10 | **0.31** |

The shape matters more than any single cell: from 0.8 m to 2.5 m the curve is **flat**
(0.62, 0.66, 0.56, 0.57, 0.59). A surface at 2.5 m becomes about as checkable as one the
agent is standing at, which is exactly the property five-inspections-per-episode needs, and
it roughly quadruples the area swept per metre travelled.

Two costs, both measured rather than assumed. Detection goes 36.7 → 52.9 ms per keyframe
against a control loop that currently runs at ~300 ms. And precision degrades: over 300
random navigable poses, gate-clearing false positives go 3 → 7 (the cracker box 2 → 6, max
score 0.39 → 0.52) while true positives go 35 → 53. That is a real trade and it must be run
as an A/B, not adopted — the whole cracker-box episode in this document came from choosing a
resolution on recall alone.

#### The false-positive half has an instrument that has never been switched on

`verification.absence_only=true` has been set on every batch here, which disables the
terminal candidate gate: asking the VLM, before committing, whether the boxed object is the
target. It was disabled deliberately to isolate the absence sensor and the note to turn it
on has been outstanding since the seven-episode batch. It addresses 22 of the 61 failures
directly and needs no new code.

The structural version of the same fix is to stop using one channel for two questions. The
presence filter answers "has this object moved", and the system reads the same number as
"is this track real" — so a fitted recall model that correctly excuses a missed detection on
a small distant track also protects every false positive, which is exactly a small distant
track. A per-class instance count ("this house has one cracker box") ranks tracks of a label
against each other rather than against a fixed bar, and is the smallest thing that separates
them.
### The ablation ladder: an identity channel, a VLM gate, and more sensor pixels

Four conditions, 96 episodes each, three scenes, dynamic layouts only, three attempts,
500 steps. Every cell differs from its neighbour by one thing, which is the only way the
three levers proposed after the last run could be told apart.

| | A baseline | B0 identity | B +VLM gate | C0 +hires |
|---|---|---|---|---|
| identity channel | — | yes | yes | yes |
| VLM candidate gate | — | — | **yes** | — |
| sensor | 640×480 / 768 | 640×480 / 768 | 640×480 / 768 | **1280×960 / 1280** |
| **SR / SPL** | 0.365 / 0.203 | 0.385 / 0.210 | **0.260** / 0.142 | **0.458** / 0.223 |
| in_anchor | 0.438 | 0.458 | 0.333 | **0.542** |
| cross_anchor | 0.292 | 0.312 | 0.188 | **0.375** |

B0 reuses A's maps, which are bit-identical across all 1704 tracks, so A → B0 is a genuine
single-variable comparison. C0 needs its own map (a different sensor detects different
things) and its pixel-denominated admission gates are scaled ×4 so the change is *sharper
vision* rather than *a looser gate*.

#### The VLM candidate gate is harmful here, and the reason is measurable

It is the only lever that moves the number down, and it moves it a long way: **0.385 →
0.260**. Eighty-one candidate rejections, and the damage lands exactly where a picture is
hardest to read — pitcher 0.389 → 0.056, tomato soup can 0.333 → 0.000 — while the targets
whose crops are large (bowl, cracker box, plate) are untouched. Episodes that never commit
to anything go 3 → 11: the gate vetoes the only candidate and leaves the agent nothing to
walk to.

The cause is not the model. It is that **the picture does not exist**. `verify()` judges a
restored track from the stored crop of its best detection, and for these targets that crop
is **46–101 px on its longest side**. There are no more pixels to be had; the objects are
small in a 640×480 frame. An 0.85-accuracy model asked a hard question at that size vetoes
a good candidate about as often as it catches a bad one, and a benchmark where the target
is the *only* instance of its class punishes a false veto much harder than a false accept.

Two implementation faults were fixed on the way to this number and both are worth recording
because either would have produced a *fake* null result:

- the crop was not serialised at all, so `verify()` fell through to `_ask(None)`, which
  fails open — the gate silently accepted everything, on precisely the restored tracks that
  produce 62 of 72 false-positive commits;
- with a rejection counted at full weight, one doubt retired a candidate. In the pilot the
  first three rejections were all *correct* candidates and two had converted without the
  gate. A rejection is now one piece of evidence, not a verdict.

So the honest verdict is narrow: the gate is harmful *on stored crops of small objects*.
Asking the same model about a live view at arrival, where the object is metres closer, is a
different experiment and remains open.

#### The identity channel removes the livelock and is worth +0.02

A false positive is an object that really is present, so every look that disproves it as
the target re-detects it as an object and restores the belief. Presence cannot retire it.
With the identity channel:

| | A | B0 |
|---|---|---|
| max goal commits in one episode | **251** | **6** |
| episodes committing more than 10 times | 7 | **0** |
| median search selections / arrivals | 3 / 0 | 9 / 1 |
| SR | 0.365 | 0.385 |

The livelock is gone outright and the search actually runs. The score barely moves, and the
reason is visible in the failure families: the episodes that were livelocked were failing
for the *other* reason as well, so freeing their budget bought attempts at a search that
still could not see the object. A mechanism fixed, a symptom not yet cured — worth keeping
because it is a precondition for anything the freed budget is spent on.

#### More sensor pixels is the lever, and it hits the family it was aimed at

**0.385 → 0.458**, the largest single move measured in this document, and the failure
families say it is the predicted mechanism rather than luck:

| | A | B0 | C0 |
|---|---|---|---|
| succeeded | 35 | 37 | **44** |
| remembered pose only, never re-found it | 28 | 25 | **13** |
| false positives only | 13 | 12 | **5** |
| reached the right track, still failed | 8 | 9 | 13 |

The dominant family halves, 25 → 13. That is exactly what the range sweep predicted: recall
at the 0.30 gate goes 0.36 → 0.56 at 1.6 m, 0.24 → 0.57 at 2.0 m and 0.17 → 0.59 at 2.5 m,
so a surface across the room becomes about as checkable as one underfoot and the five
inspections an episode can afford cover four times the area. False positives also fall by
more than half, because a sharper look resolves what a blurry one guessed at.

The cost shows up where it should: "reached the right track and still failed" *rises*
8 → 13. More episodes now get to the object, so more of them fail at the last metre instead
of never arriving — the bottleneck moving down the pipeline is what progress looks like
here. Per target, the plate goes 0.111 → 0.500 and the cracker box 0.417 → 0.667.

Two things it does not fix. **00880 gets worse** (0.433 → 0.333) while 00829 goes
0.500 → 0.611 and 00848 0.200 → 0.400; the scene spread remains larger than the
intervention, which is the standing argument for more scenes rather than more tuning.
And detection is 36.7 → 52.9 ms per keyframe with the control loop at ~1.6 fps against
~3.3, so this buys accuracy with time and any real-time claim has to be restated at the new
number.

#### Where the failures are now

Of C0's 52 failures: 13 never re-found the object, 17 went to the remembered pose and then
chased a false positive, 13 reached the right track and failed anyway, 5 chased false
positives only, 4 never committed. The distribution is flatter than it has ever been — no
single cause is now more than a third — which is the first time this benchmark has not had
one obvious next thing to fix.

### The search posterior had never once delivered an episode

The failure-family table above says *what* episodes did, not what was missing. Recut by
whether the object was ever perceived at its new pose, C0's 52 failures split cleanly:

| | n | burned the full 500 steps |
|---|---|---|
| never mapped it at the new pose | **40** | 33 |
| mapped it and still failed | 12 | 10 |
| succeeded | 44 | 0 (median 112 steps) |

Seventy-seven percent of failures never perceive the object at all after it moves. The
funnel for the whole run:

```
the new pose is within 0.5 m of a mapped container   96/96   100%
the surface search ran at all                        53/96    55%
the surface search ARRIVED at the true surface        0/96     0%
committed to a track on the real object              57/96    59%
...converted that into a success                     44/57    77%
```

Perfect candidate coverage, zero arrivals. Every one of the 44 successes came from the prior
map (8 committed on step 1) or from frontier exploration finding the object incidentally.
The search line engages only once the easy route has failed — SR 0.814 on the 43 episodes
that never invoked it, **0.170** on the 53 that did.

Three things this rules out. **Perception:** recall at the objects' authored *dynamic*
viewpoints is 0.584 at imgsz 1280 (0.567 at 768), and its correlation with per-target SR is
+0.19 — the bowl has recall 0.99 and SR 0.50, the 00829 plate has recall 0.30 and SR 0.83.
The resolution gain measured earlier was never about recognition quality: by ring radius it
is −0.04 at 0.8 m, +0.13 at 1.2 m, +0.31 at 2.0 m. It buys sightings *in passing*, which is
why it helped a search and would not repeat itself at higher resolution. **Step budget:**
400 → 500 steps bought +0.02; successes are cheap or absent, 30 of 44 inside 150 steps.
**Terminal navigation:** 77% conversion once the right object is committed to.

What remained was ranking, and `scripts/rank_search_surfaces.py` scores it without a
simulator — the prior map holds every container the posterior can propose, `container_prior`
is a pure function of (class, label, top height, area, centre), and the authored layout says
which one is right. An eight-hour benchmark becomes a second, and ranking work gets a number
to move.

#### The floor was clipping the far field into one tie

Two defects, in order of size.

The prior after absence was **category-only**: 65 candidates carrying *five distinct
values*, with 12 tied desks, 10 tied cabinets and 36 things tied at the fallback. Since
`select_candidate` cuts to the top 5 by prior *before* costing any path, the cut chose three
arbitrary desks out of twelve and never planned to the rest — the tie-break was `sorted`'s
stability, i.e. track id.

The prior had a proximity term that would have broken those ties, and it was being discarded
twice over. `_last_known_target_xy` returned None once absence was confirmed, on the
reasoning that walking to the old pose and finding nothing refutes "objects move short
distances". It does not: it refutes one *surface*. in_anchor relocations move a median
0.72 m, so the object is usually a metre away on a neighbouring surface — and the surface
actually ruled out is retired by the `InspectionLog`, which is the right instrument for it.

And where the term did apply it was clipped. `max(exp(-d/L), 0.2)` does not mix "moved
nearby" with "moved anywhere", it *clips*: every candidate past `L·ln(1/floor)` — 6.4 m at
L=4 — receives an identical prior, so the entire far field ties and its order collapses to
track id again. That is precisely the regime a cross-anchor move lives in, and it is why
keeping the term used to measure badly there. Unclipped, the far candidates stay ordered by
distance, which is weak evidence but is evidence.

Scored over 114 (scene, layout, target) combinations, share where the true surface lands in
the top 5 — what one episode can afford to inspect:

| proximity model | overall | in_anchor | cross_anchor | median rank |
|---|---|---|---|---|
| dropped after absence (before) | 12/114 | 5/57 | 7/57 | 27 of 65 |
| kept, `L=4` with a 0.2 floor | 20/114 | 19/57 | 1/57 | 9 |
| **kept, `L=1`, no floor (now)** | **36/114** | **29/57** | **7/57** | **5** |

Simulating the greedy search from each episode's real start pose — same candidate build,
same top-N cut, same `b·d/c` argmax, same multiplicative decay — it reaches the true surface
within an 8-inspection budget in **38/114 against 11/114**, median 1 inspection against 3.
Cross-anchor is not paid for: 7/57 either way.

Two things measured and *rejected* along the way. Breaking the top-N tie by distance to the
agent made it worse (11 → 7): the utility already divides by path cost, so the tie-break
merely narrows the pool to a local cluster. And an annulus — suppressing the radius just
confirmed empty — was much worse (11 → 2), because at a median displacement of 0.72 m the
object is usually *inside* the exclusion.

#### Loosening the container gates is not the answer

The true surface is not even a candidate in 45% of cases, lost at the container layer rather
than mis-ranked: `container_min_score=0.5` rejects desks and counters detected at 0.36–0.43,
the 0.2–1.4 m band rejects a shelf whose top lands at 2.06 m, and `container_merge_m=1.0`
collapses same-label neighbours. But loosening them inflates the candidate list as fast as it
adds coverage, and the top-5 rate barely moves:

| gates | true surface is a candidate | candidates offered | top-5 |
|---|---|---|---|
| `min_obs=2 min_score=0.5 merge=1.0` (current) | 63/114 | 65 | 12/114 |
| `min_obs=2 min_score=0.5 merge=0.0` | 68/114 | 86 | 12/114 |
| `min_obs=1 min_score=0.3 merge=0.0` | 80/114 | 128 | 17/114 |

So the gates stay as they are. This is worth restating as a general shape: with a ranker this
coarse, *coverage is not the constraint* — the candidate set was already at 96/96 by the
looser footprint test — and buying more of it costs exactly what it gains.

### A frontier the agent reached was being called unreachable

`frontier_stub_block` fired 2.87 times per failing episode against 0.39 per success, and
33 of the 40 never-mapped failures had it. It is meant to catch a degenerate stub path — the
planner handing back a goal snapped near the start because the frontier cannot be reached —
but the threshold it used could not tell that apart from an ordinary arrival.

`HybridVoronoiPlanner` navigates the medial axis and stops at the graph node nearest the
goal, within `goal_near_m` = 0.7 m: it stops *near* a goal, not on it. `WaypointController`
then reports arrival within 0.2 m of that endpoint. So a correct arrival leaves the agent up
to **0.9 m** from the frontier goal, and `_frontier_reach_m` was a fixed **0.5 m**. Getting
there was read as never having got there: the frontier was blocked for 100 rounds and
`_last_giveup_pt` was set, which also suppresses the all-frontiers-blocked fallback anywhere
near that point.

The fix is to derive the threshold rather than pick it — `voronoi_goal_near_m` is now a real
config field, read both to build the planner and to set `_frontier_reach_m`, so the two
cannot drift apart. Two tests pin the invariant, one of them by moving the planner's stopping
radius and checking the classification follows.

(One correction to the earlier reading of these logs: `frontier_give_up` is not "exploration
exhausted". It fires when the agent moves less than 0.2 m in 15 steps while pursuing a
frontier — a stalled pursuit, not an exhausted map.)

### One admission gate priced every class the same

Recall at the 0.30 gate is 0.584; at 0.20 it is 0.660. Whether that is worth taking depends
entirely on what arrives with it, so: a false-positive census over 900 random navigable poses
across the three scenes at imgsz 1280, counting every detection carrying a YCB label that
clears the 1200 px node-creation gate.

Globally the move 0.30 → 0.20 is **+8 true positives for +32 false ones**, precision 0.59 →
0.49. Not a trade worth making. Per class it is a different question:

| target | extra false positives | recall 0.30 → 0.20 |
|---|---|---|
| tomato soup can | **0** | 0.26 → 0.34 |
| banana | **0** | 0.67 → 0.75 |
| plate | 2 | 0.50 → 0.56 |
| blue plastic pitcher | 3 | 0.36 → **0.49** |
| bleach bottle | 9 | 0.74 → 0.83 |
| cracker box | **18** | 0.77 → 0.82 |
| bowl | 0 | 0.99 → 0.99 |

The four cheap ones are exactly the four weakest targets in the last full run — pitcher
0.333, soup can 0.333, banana 0.333, plate 0.500 — and between them they cost five false
positives. The two expensive ones are the promiscuous labels, and "cracker box" alone is
eighteen of the thirty-two. So `DetectorConfig.class_conf` now holds per-class thresholds:
inference runs at the lowest gate anyone asks for (a detection ultralytics never returns
cannot be admitted afterwards) and each label is admitted against its own. The mechanism is
symmetric — a *stricter* per-class gate works the same way — and with no overrides it is
exactly the previous behaviour.

The container layer is unaffected: `container_min_score = 0.5` gates surfaces independently,
so a 0.20 detection cannot enlarge the search candidate set and undo the ranking work above.

#### The census was measuring the wrong cost, so this ships OFF

A six-episode pilot ran the same scene with and without the overrides. The tomato soup can —
one of the two the census called *free* — succeeded in 31 steps at the global 0.30 gate and
failed at 500 with its own 0.20 gate. Same single track, 0.97 m from truth, same `n_obs=3`,
same `best_score=0.53`, same 1612 px. What differed was the belief on arrival: **0.457
against 0.095**, the latter already carrying nine missed expectations by step 31, so the
absence check abandoned a correct candidate.

The census counted detections above the node-creation gate at random navigable poses. It
never measured what admitting more of them does to the **presence filter**, and that is where
the cost landed: more admitted detections mean more frames in which a track is expected and
unmatched, and a correct track can be disbelieved before the agent gets to it. A wider census
would not have caught this; only an end-to-end run does.

So `class_conf` ships as a mechanism with an empty default. The machinery is worth having —
it is the only way to price classes separately, and the per-class asymmetry it exposes is
real — but it is one config line away from being switched on once the full 96-episode run has
measured it against the presence filter rather than against a pose sample.

#### What the pilot does and does not say

Six episodes, 00829 in_anchor_01, against the same episodes from C0:

| | SR | search ran | inspections | reached the true surface | stub blocks |
|---|---|---|---|---|---|
| C0 | 4/6 | 3/6 | 27 | **0/6** | 7 |
| levers 1+2, unclipped only | 3/6 | 1/6 | 1 | 0/6 | 1 |
| levers 1+2, scale normalised | 3/6 | 2/6 | 3 | **1/6** | 0 |
| levers 1+2, no per-class gate | **4/6** | — | — | — | — |

The first-ever arrival at the true surface is the result worth having: 0 in 96 C0 episodes,
0 in the unnormalised pilot, 1 in 6 once the scale was fixed. Against that, the plate episode
that C0 won at step 429 is now lost, and the soup can is won instead — one swap each way on
six episodes, which is no evidence of a net change in either direction. The offline scorer is
what argues for these changes; the pilot's job was to catch what it could not see, and it
caught two things.

#### The bug the offline scorer could not have caught

Sharpening proximity from `max(exp(-d/4), 0.2)` to `exp(-d/1)` improved every ordering metric
and switched the search line off. `_select_surface` compares a surface's `prior·d/cost`
against a frontier's `β·score/cost`, so the *magnitude* of an unnormalised score decides
whether the agent searches or explores. The top candidate barely moved (median 0.479 → 0.264
over 19 (scene, target) pairs) but the tail collapsed by orders of magnitude, and once the
few plausible nearby surfaces were retired by the `InspectionLog` nothing could out-score a
frontier again: the pilot went from 27 inspections over six episodes to **one**.

Ranking is scale-invariant and this comparison is not, which is exactly the class of bug an
offline ranking metric is blind to. The fix separates them: candidate priors are normalised
so the best of them equals `search_surface_mass` (0.5, matching where the previously tuned
model sat), with the `InspectionLog` decay applied *after* the anchor — normalising post-decay
would restore the best survivor to full mass every round and the agent would never hand back
to exploration.

#### Not yet measured end to end

The offline ranker predicts the search reaches the true surface 3.5× more often. It does not
predict a success rate: reaching a surface is necessary and not sufficient, the conversion
from a committed correct track is 77%, and the conversion from a *searched* surface has never
been measured at all — the sample having been zero until now. A full rerun is what settles it.

### The overnight campaign: D, E, F

Three conditions of 96 episodes each, one change per rung, against the stored C0. A, B0 and B
were **not** re-run: at ~4 hours a condition that is sixteen hours for rungs whose attribution
(identity channel, VLM gate, sensor resolution) already stands, and it would have crowded out
the iteration the campaign was for. The cost is that D-vs-C0 spans a commit rather than a
config flag, which is what a ladder rung is anyway.

| | C0 | D | E | change from the rung above |
|---|---|---|---|---|
| SR | **0.458** | 0.385 | 0.427 | |
| SPL | 0.220 | 0.196 | 0.221 | |
| in_anchor | 0.542 | 0.479 | 0.500 | |
| cross_anchor | 0.375 | 0.292 | 0.354 | |
| mapped it at the new pose | 52/96 | 44/96 | 46/96 | |
| search ran at all | 53/96 | 37/96 | 38/96 | |
| **search ARRIVED at the true surface** | **0/96** | **5/96** | **6/96** | |
| committed to the real object | 57/96 | 47/96 | 53/96 | |
| frontier selections per episode | 2.15 | **20.76** | 5.30 | |
| surface inspections per episode | 3.34 | 0.88 | 1.24 | |

**D lost, and the half that worked is not the half the score reflects.** The ranking did
exactly what the offline scorer predicted: the first arrivals at a relocation destination this
benchmark has ever produced, four of the five at inspection #1 or #2. What sank it was
frontier exploration, and the cause was a change made for a good reason.

`_frontier_reach_m` was doing two jobs. Deriving it from the planner's true stopping radius
(0.5 → 1.0 m) is right in itself — `HybridVoronoi` stops at a node within `goal_near_m` of a
goal and the controller arrives within its own tolerance of *that*, so a correct arrival could
be 0.9 m out and was being called an unreachable stub. But the same test was the only thing
blocking a frontier when a pursuit ended, and an unblocked frontier is immediately
re-selectable: the FSM drops `GOTO_FRONTIER → EXPLORE`, picks the same frontier, and give-up
cannot break the loop because it counts steps *inside* `GOTO_FRONTIER` and the re-entry resets
its timer. In 00829, `plan_ok` went 1.4 → **39.4** per episode and episodes that ever mapped
the object went 23/36 → 13/36.

`_retire_pursued_frontier` separates the two jobs: the block is unconditional — retiring a
frontier the agent reached costs nothing, it has been explored, which is the point — and the
reach threshold now decides only how long the block lasts and whether this counts as a
give-up point. **E** carries that single change and recovers to 0.427 with SPL back to parity,
three episodes short of C0 and inside the ±0.05 binomial noise at n=96.

**What E leaves is a search that is accurate but inactive.** Against C0 it runs in 38 episodes
rather than 53, for 1.24 inspections rather than 3.34, and arrives at the true surface 6 times
against 0. Ranking has stopped being the constraint. **F** therefore moves on the two axes
that are left — `search_surface_mass` 0.5 → 1.0, because matching the old model's *top* prior
under-funds a search whose tail is no longer propped up by a floor; and
`_face_searched_surface`, because `_mark_surface_searched` multiplies a surface's belief by
(1 − 0.8) on whatever heading the follower stopped at. That is the same mistake the candidate
path already made and fixed, one layer up, and in D the search reached the true surface five
times and converted one.

#### A methodological note worth more than any of the deltas

D is the second time on this benchmark that a change improved every offline metric and cost
real success. Ranking is scale-invariant; `_select_surface`'s comparison against frontier
utility is not, and sharpening the prior switched the search line off (27 inspections over six
episodes → 1). The frontier livelock is the same shape one level up: a threshold that was
*also* silently serving as an interlock, so correcting it removed a guarantee nobody had
written down.

The lesson is not "do not trust offline metrics" — the offline ranker is what found the real
bottleneck and is 4 hours cheaper per iteration. It is that a proxy must be paired with a
six-episode pilot before a four-hour condition is spent on it, and that the pilot should be
read for *mechanism* counters (inspections, selections, arrivals) rather than for SR, which at
n=6 says nothing.

### The benchmark's relocations are not semantic, and that is a limitation to state

Across all 114 relocations, where the destination surface is mapped, the objects land on:

```
bed 26,  desk 19,  table 6,  cabinet 3,  nightstand 3,  bench 2,  stool 2,  sofa 1,  shelf 1
```

**37 of 53 land on a category `CONTAINER_AFFINITY` does not list for that class.** A tomato
soup can is placed on a bed seven times. Ranking the true surface among candidates:

| prior | top-1 | top-5 | median rank |
|---|---|---|---|
| affinity × proximity (shipped) | 19/114 | 36/114 | 5 |
| **proximity alone** | **26/114** | 38/114 | **2** |
| affinity alone | 0/114 | 12/114 | 27 |
| arbitrary order | 2/114 | 14/114 | 25 |

Affinity alone is no better than arbitrary order, and multiplying it in makes proximity worse.
The authoring appears to place relocated objects for reachability rather than for semantic
plausibility, which means **this benchmark cannot reward a semantic search prior** — the
system's headline contribution is being evaluated on data built to be indifferent to it. Any
writeup has to say so, and a weak affinity term here is not evidence that semantic priors fail
in general.

Two separable responses, neither yet shipped. The unlisted-category weight of 0.25 sits
*below* the lowest listed entry — positive evidence against — on the strength of one anecdote
about a bed and a sofa tying with a sink as places to look for a bowl. A six-entry list is not
exhaustive and absence from it is not evidence; raising it to 0.5 is a correctness fix that
would be right on any data. Softening the ranking to a tie-breaker on top of that is
calibrated to *this* benchmark and belongs behind a knob. Measured together offline: median
rank 5 → 2, top-1 19 → 24, and the simulated greedy search finds the surface within one
inspection in 28/114 against 21.

#### The container gates are settled, twice

The true surface is not a candidate at all in 45% of cases, lost at the container layer —
`container_min_score=0.5` rejects desks detected at 0.36–0.43, the 0.2–1.4 m band rejects a
shelf topping out at 2.06 m, `container_merge_m=1.0` collapses same-label neighbours. But
loosening inflates the candidate list as fast as it adds coverage. Before the ranking fix,
top-5 went 36 → 40 → 38 as the gates opened. After it, coverage rises 63 → 80 of 114 and the
metric that matters — reaching the surface within the ~3 inspections an episode actually
affords — goes the wrong way:

| gates | true surface is a candidate | candidates offered | within 3 inspections |
|---|---|---|---|
| `min_obs=2 min_score=0.5` (shipped) | 63/114 | 65 | **35/114** |
| `min_obs=1 min_score=0.4` | 70/114 | 91 | 33/114 |
| `min_obs=1 min_score=0.3` | 76/114 | 102 | 33/114 |

Coverage was never the constraint. The gates stay.

### G: the split, measured rather than argued

Condition G carries the affinity fix and the ground-truth visibility instrument. The
instrument projects the authored target position into every frame, discards it outside the
image or behind the camera, and discards it again when the depth buffer puts something solid
in front. What survives is *the agent was looking at the place where the object is*.

Over 96 episodes:

| | n | SR |
|---|---|---|
| localized — a track within 0.25 m | 37 | **0.89** |
| **looked at it within 3 m and missed** | **35** | 0.23 |
| looked, but only from beyond 3 m | 2 | 0.00 |
| **never looked at the new pose** | **22** | 0.00 |

Median in-view frames 14; median closest approach 1.07 m. So of the 59 episodes that fail to
hold a track on the object, **35 had the agent standing within three metres of it**, a median
of 12 frames at a median 1.16 m, and got nothing usable. Perception outweighs coverage
roughly three to two, and every earlier iteration assumed the reverse.

Splitting those 35 by what the detector did produce:

| | n |
|---|---|
| nothing near the object at all (only stale or false tracks > 2 m) | 15 |
| a track 0.5 – 2.0 m off | 11 |
| a track 0.25 – 0.5 m off (a near miss) | 9 |

And splitting the 22 that never looked: median displacement **6.84 m**, and 17 of 22 are
cross_anchor. Never-looked is the long-move coverage problem, which is what the search line
exists for; it is not the majority of the loss.

So the remaining work divides three ways, and the shares are now known rather than guessed:

* **22 coverage** — the agent never gets eyes on a destination six metres away.
* **15 detection** — it stands at 1.2 m and the detector clears no gate. This is what
  `DetectorConfig.class_conf` was built for, and the per-class census says the two worst
  targets here cost nothing to admit at 0.20.
* **20 localization** — a detection happens and the track lands a quarter to two metres out.

Per target the shares are very uneven, and they point at the same two objects the campaign has
been losing all along:

| target | localized | looked + missed | never looked |
|---|---|---|---|
| tomato soup can | **0** | 9 | 3 |
| plate | 4 | 9 | 5 |
| blue plastic pitcher | 4 | 6 | 8 |
| bowl | 5 | 5 | 2 |
| bleach bottle | 11 | 3 | 4 |
| cracker box | 8 | 2 | 2 |
| banana | 5 | 1 | 0 |

The tomato soup can is localized in **zero of twelve** episodes while being looked at within
three metres in nine of them. Its measured detector recall on these poses is 0.13–0.35, and
lowering its own admission gate to 0.20 costs zero extra false positives in a 900-pose census.

#### The affinity fix did not transfer, and that was predictable

G against F: SR 0.427 against 0.438, arrivals at the true surface identical at 8/96. Offline
the same change took the true surface from median rank 5 to 2 and from 21 to 28 of 114 found
within one inspection. None of it showed up.

That is the third time an offline ranking gain has failed to become an online one, and here
the reason is the benchmark rather than the metric: the fix makes the *semantic* half of the
prior better behaved, and this benchmark's destinations are drawn without regard to semantics.
It is worth keeping anyway — the unlisted-category penalty it removes was wrong on any data,
and the change is cost-free at SR — but it should be re-measured on a dataset whose
relocations are placed the way people place things. Until then the honest statement is that
the semantic half of the search posterior has never been shown to do anything on this
benchmark, in either direction.

#### F, and the campaign's final ledger

| | C0 | D | E | F |
|---|---|---|---|---|
| SR | **0.458** | 0.385 | 0.427 | 0.438 |
| SPL | 0.220 | 0.196 | 0.221 | 0.216 |
| in_anchor | 0.542 | 0.479 | 0.500 | 0.479 |
| cross_anchor | 0.375 | 0.292 | 0.354 | **0.396** |
| search ran at all | 53/96 | 37/96 | 38/96 | 45/96 |
| **arrived at the true surface** | **0/96** | 5/96 | 6/96 | **8/96** |
| frontier selections/episode | 2.15 | 20.76 | 5.30 | 4.12 |
| surface inspections/episode | 3.34 | 0.88 | 1.24 | 1.91 |

Read the mechanism column and the search works: arrivals 0 → 5 → 6 → 8, engagement recovering
toward C0's, cross-anchor at its best of any condition. Read the SR column and three
conditions of work has not beaten a number from before it started, and the spread across
C0/E/F (0.458 / 0.427 / 0.438) is inside the ±0.05 binomial noise at n=96. Both readings are
honest and they are about different things.

Per target, F against C0: banana 0.333 → **0.833**, bleach bottle 0.500 → 0.556, pitcher
0.333 → 0.389, cracker box and bowl unchanged — against plate 0.500 → **0.278** and tomato
soup can 0.333 → **0.083**. Per scene, 00848 0.400 → 0.567 against 00829 0.611 → 0.444. This
is redistribution, not a lift.

### Success is decided by one number, and it is not the search

Of F's 8 arrivals at the surface the object had actually been moved to, **1 converted**. The
geometry is not the problem — a container's centre sits a median 0.51 m from an object resting
on it — and neither is the detector: those episodes carry target-label detections of 3096 to
33995 px at scores up to 0.79, far above the 1200 px node gate. What they carry is a **track
in the wrong place**. The three 00829 plate arrivals detected the plate and built a track
1.05–1.55 m from where it was.

Binning every F episode by how far the *nearest* target-label track ends up from the truth:

| localization error of the best track | n | SR |
|---|---|---|
| ≤ 0.25 m | 37 | **0.892** |
| 0.25 – 0.5 m | 11 | 0.364 |
| 0.5 – 1.0 m | 7 | 0.429 |
| 1.0 – 2.0 m | 6 | 0.167 |
| > 2.0 m | 35 | **0.029** |

That is a cliff, not a gradient. Get a track within a quarter of a metre and the episode is
won nine times in ten; miss by two metres and it is lost 97 times in 100. Everything the
campaign moved — ranking, engagement, frontier retirement, facing — operates on whether the
agent *gets there*, and gets there is worth almost nothing unless the map puts the object
where it is.

Per target the split is stark, and it explains the redistribution above:

| target | median localization error | within 0.25 m | SR |
|---|---|---|---|
| banana | 0.03 m | 6/6 | 0.833 |
| bleach bottle | 0.03 m | 12/18 | 0.556 |
| cracker box | 0.06 m | 9/12 | 0.667 |
| blue plastic pitcher | 0.66 m | 7/18 | 0.389 |
| plate | **1.55 m** | 6/18 | 0.278 |
| bowl | **2.85 m** | 6/12 | 0.500 |
| tomato soup can | **2.94 m** | 2/12 | 0.083 |

**The causal reading of this table was wrong, and the correction matters more than the
table.** Splitting the same episodes by *why* the nearest track is far away:

| | n | SR |
|---|---|---|
| localized (≤ 0.25 m) | 37 | 0.892 |
| only a **stale** track, at the remembered pose | 48 | 0.146 |
| only false positives | 9 | 0.000 |
| re-detected and **mis-placed** | **2** | 1.000 |

Two. The per-target "localization error" above is not a fit error — it is the map still
holding the object where it used to be. The tomato soup can's 2.94 m median is twelve episodes
in which it was never re-detected once, not twelve bad ellipsoids. An upper bound on what a
better fit could recover, counting every episode that is neither localized nor stale, is 11
episodes, and 9 of those are false-positive-only.

So the correlation is real and the diagnosis it suggested is not. `objects/optimization.py` is
**not** the next thing to fix. The bottleneck is re-detection: **57 of 96 episodes never
perceive the object at its new pose at all**, and 48 of them still hold it at the old one.

It also reframes the earlier funnel. "Never mapped it at the new pose" was measured with a
0.5 m threshold and read as a perception-coverage failure. Some of it is: 35 of 96 episodes
have nothing within 2 m. But the band between 0.25 m and 2 m — 24 episodes, SR 0.33 — is the
object being seen and mis-placed, which no amount of better searching will recover.

#### The error is along the ray, and more looking does not fix it

Two measurements that narrow it further. Decomposing the error of the committed candidate
against the camera pose of its best detection, the component **along** the camera-to-object
ray is 0.18 m against 0.07 m **across** it — a depth-and-extent error, not a mask-centroid
error, which is what a nearly-planar depth return on a plate or a shallow bowl would produce.
And the error does not converge with evidence:

| observations on the track | n | median error |
|---|---|---|
| 1–2 | 28 | 0.27 m |
| 3–5 | 25 | 0.52 m |
| 6–15 | 25 | 0.13 m |
| 16+ | 24 | 0.57 m |

No trend — and the radial error is *symmetric*, not a constant offset: over the committed
candidates its median is +0.001 m with 51% of tracks placed beyond the object rather than
short of it. So there is no constant to subtract, and yet averaging over sixteen observations
does not shrink it either. That combination points at per-view errors that are correlated
rather than independent — an agent approaching from one direction sees a plate at much the
same oblique angle every time — or at a fit that re-estimates from the best view instead of
pooling. Which of those it is has not been measured.

Two caveats on this cut. It covers the committed candidates with error under 1.5 m; the
badly-placed plate, bowl and soup-can tracks are a separate population it says nothing about.
And "observe it more" being useless is a statement about the current estimator, not about
observation in principle — observations from *new angles* were never separated from repeat
views of the same one.

#### Why a better search did not buy a better score

The share of episodes whose best target track lands within 0.25 m — the number the cliff says
decides everything — is **42/96 in C0 and 37/96 in F**. Both start from the same prior map, so
the difference is tracks built during the episode, and it runs the wrong way for the condition
with the better search.

That is the campaign's real result. C0's search was inaccurate and its exploration wandered;
the wandering was incidentally producing more and better-separated views of whatever it passed,
and therefore better localization. F goes more directly to the right places and sees them from
fewer angles. Trading coverage for precision in *navigation* traded away precision in
*mapping*, and the second one is what the success criterion actually reads.

This is worth stating plainly because it is easy to mistake for a null result. The search
mechanism demonstrably works now — arrivals 0 → 8, engagement recovered, cross-anchor at its
best — and it did not raise SR because it does not yet run often enough, or accurately enough,
to change how many episodes re-detect the object.

The metric to move is the share of episodes that hold a track within 0.25 m of the target:
**37/96**, worth 0.89 of an episode each against 0.03 for the rest. What is *not* yet known is
the split inside the 57 that fail it — whether the agent never looked at the new location, or
looked and the detection fell under the gate. Those have different fixes and the episode logs
cannot separate them, which is the next thing to build rather than the next thing to tune.

Engagement is a smaller lever than it looks: of the 51 episodes where the search never ran, 34
succeeded without it (median 125 steps) and only **9** burned 400+ steps without succeeding.
And when the search does run it already inspects a median of 4 surfaces. The ceiling on
"search more" is single digits of episodes; the ceiling on "re-detect at all" is 57.
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

**Before trusting any detection claim**, measure it rather than inferring it from an
episode outcome — three rounds of this document did the latter and were wrong each time:

```bash
# what the detector scores at the authored viewpoints, per imgsz
python scripts/probe_ycb_detection.py +experiment=ycb_authored_nav \
  ycb.layout_root=outputs/substituted_layouts \
  +probe.mode=views '+probe.imgsz=[512,768,1280]' +probe.views=20

# whether a different NAME recovers a target the benchmark's label misses
python scripts/probe_ycb_detection.py ... +probe.mode=labels +probe.handles=[019_pitcher_base]

# how often a label fires on something that is not the object -- the number that
# should have chosen imgsz, and did not
python scripts/probe_ycb_detection.py ... +probe.mode=fp +probe.fp_samples=300
```

**Per phase afterwards:** unit tests first (everything except Phase 2 and the eval runs
is simulator-free by construction), then the YCB run with `verification=off`, then the
metric that phase exists to move — Phase 1 stale-goal rate, Phase 3 containers-inspected
and post-failure distance, Phase 4 re-ID accuracy.

**Standing rule:** no A/B that claims two configs are equivalent may run with the VLM
verifier on.
