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

`YCBAuthoredNavEnv.reset()` injects one layout per episode
(`src/osg/sim/ycb_env.py:491`). That reproduces DualMap's protocol — map, change the
world offline, query — which measures only stale-memory recovery. Belief latency cannot
be measured at all, because nothing ever observes the change.

### Two additions

1. **Paired-layout episodes.** Map on layout *i*, evaluate on layout *j* with the same
   map. This is the DualMap comparison, and it needs the map to survive a reset — an
   episode-level `map_from` field in the manifest plus a runner path that keeps the
   `ObjectLayer` across the pair.
2. **Mid-episode relocation.** Re-apply translations/rotations to the *same* rigid
   objects at step *k*. `inject_layout_objects()` already returns the handles, so this
   is `set_translation`/`set_rotation` on existing objects — no sim rebuild. Two
   sub-conditions, and the distinction is the point:
   - **in view** — the agent is looking at the object when it moves. Negative evidence
     should show a large, clean margin over any timeout scheme here.
   - **out of view** — tests whether the search posterior recovers.

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
