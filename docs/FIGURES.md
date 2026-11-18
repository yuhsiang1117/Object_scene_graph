# Figures

Every figure in `docs/PAPER.md`, with its caption, its provenance, and the command that
regenerates it. These were ten separate `docs/*_CAPTION.md` files; the content below is
theirs, unedited.

`docs/figures/` is **gitignored** — 74 MB of PNG/PDF/SVG that the renderers below reproduce.
Run the renderer, do not look for the file in git. One exception worth knowing: the stair
figures are written to `outputs/figures/`, not `docs/figures/`, so they do not survive a
clean of `outputs/`.

| figure | renderer |
|---|---|
| [Teaser](#teaser) | `render_teaser.py`, `render_teaser_floor_assets.py` |
| [Teaser scene](#teaser-scene) | `render_scene_graph_multifloor.py`, `render_teaser_scene.py` |
| [Full pipeline](#full-pipeline) | `render_full_pipeline.py` |
| [Presence belief](#presence-belief) | `render_presence_belief.py` |
| [Presence on a real view](#presence-on-a-real-view) | `render_presence_view.py` |
| [Search and floor decision](#search-and-floor-decision) | `render_search_floor_decision.py` |
| [Floor score](#floor-score) | `render_floor_score_formula.py` |
| [Stair geometry](#stair-geometry) | `render_scene_graph_multifloor.py`, `render_stair_anatomy.py`, `render_stair_evidence.py` |
| [Layout conditions](#layout-conditions) | `render_layout_conditions.py` |
| [Relocation modes](#relocation-modes) | `render_relocation_modes.py`, `render_ycb_layout_assets.py` |

---

## Teaser

*Was `docs/TEASER_CAPTION.md` — Teaser figure.*

**Caption.** A cup previously observed in an upstairs bedroom remains in the
scene graph after being moved to the downstairs kitchen. When the robot finds
the old location empty, it revises the bedroom–table–cup belief. Exploration
selects the staircase frontier, and the robot recovers the cup across floors;
the revised graph links floor, room, counter, and cup. The two tilted,
vertically separated floor backdrops are orthographic top-down renders of HM3D
scene 00873-bxsVRursffK. The staircase route follows a navmesh-checked
two-flight switchback, with an illustrative final approach toward the cup.
Robot pose, cup locations, and belief updates are explanatory overlays, not a
measured episode trajectory. The scene graphs on the right include floor,
room, container, and object layers.

Render the aligned floor assets, then compose the teaser:

```bash
docker exec docker-nav-1 /opt/conda/envs/habitat/bin/python /workspace/scripts/render_teaser_floor_assets.py
python3 scripts/render_teaser.py
```

The PDF and SVG keep labels and routes as vectors; the PNG is a preview. The
figure is a motivation teaser
above the paper's main text, while `F1_architecture` remains the method figure.
For a two-column LaTeX paper, place it near the beginning as a `figure*` with
`\includegraphics[width=\textwidth]{figures/teaser.pdf}` and use the caption above.

---

## Teaser scene

*Was `docs/TEASER_SCENE_CAPTION.md` — Scene-based teaser figure.*

**Caption.** A prior observation places the cup in an upstairs bedroom. When
the robot checks that location after a cross-floor change, it is empty and the
upstairs bedroom–bedside table–cup containment chain becomes outdated.
Exploration recognizes the staircase,
selects a floor-switch frontier, and guides search toward a downstairs kitchen
candidate. The belief is revised from the bedside table's cup to a cup on a kitchen
counter, preserving the floor → room → container → object hierarchy. The backdrop
is a real orthographic HM3D section of `00873-bxsVRursffK`; cup positions,
robot, and arrows are schematic annotations, not a measured episode trajectory.
The red cross marks the empty old site; the dashed green outline identifies the
visible stair flight, and the route runs through it toward the downstairs cup.

Render the section in `docker-nav-1` with
`/opt/conda/envs/habitat/bin/python scripts/render_scene_graph_multifloor.py --scene 00873-bxsVRursffK --maps-root outputs/maps_p1500_osg --side-res 2800 --section-frac 0.35`.
The resulting `outputs/figures/side_00873-bxsVRursffK_2800_0.35.npz` supplies
the image saved as `docs/figures/hm3d_section_00873.png` (black void converted
to transparency). Then run `python3 scripts/render_teaser_scene.py` for the PDF,
SVG, and PNG teaser. Place the PDF near the start of a two-column paper as a
`figure*` with `\includegraphics[width=\textwidth]{figures/teaser_scene.pdf}`.

---

## Full pipeline

*Was `docs/FULL_PIPELINE_CAPTION.md` — Full pipeline figure.*

**Suggested caption.** System-level mapping and navigation pipeline. RGB-D
observations and pose update object tracks and spatial geometry, while visibility
and arrival evidence revise object presence. These outputs update the persistent
3D scene graph, whose building, storey, room, container, and object nodes are
illustrated over two rendered HM3D floor plans. A stair edge joins the two
storey nodes rather than belonging to the containment hierarchy. Given a target
query, the revised graph and map support search-goal selection and multi-floor
navigation. The navigator emits an action; arrival and completed-transition
results return to the persistent graph. The centered building node branches into two storeys, four rooms, and
subsequent container and object layers above the two floor plans. Node counts
and connections are schematic, while the floor-plan images are scene renders.

Render with `python3 scripts/render_full_pipeline.py`. Use the PDF at the top
of the Methodology section:

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/full_pipeline.pdf}
  \caption{System-level mapping and navigation pipeline. RGB-D observations
  update object tracks and spatial geometry; presence revision updates the
  persistent 3D scene graph. The graph and target query guide goal selection
  and multi-floor navigation, which produces an action and returns arrival
  and stair-transition evidence. The floor plans are rendered HM3D scenes;
  raised graph nodes are schematic.}
  \label{fig:full_pipeline}
\end{figure*}
```

---

## Presence belief

*Was `docs/PRESENCE_BELIEF_FIGURE_CAPTION.md` — Presence-belief figure.*

**Caption.** The dashed cup marks an object's stored pose. (a) Valid depth
nearer than the ellipsoid's near boundary indicates occlusion; with no matched
detection, the frame is uninformative and leaves presence unchanged. (b) Depth
beyond the expected object exposes the background; if the view is otherwise
detectable and the detector is silent, the miss lowers presence. (c) A matched
sighting raises presence, including when the expectation test did not pass.
Only the near side gates a miss. (d) A keyframe detector miss, a VLM crop
judgment at arrival, or an expectation-gated detector-scan fallback contribute
likelihood steps to the same bounded log-odds state. The VLM and detector
fallback are **alternative readings for one arrival**, not two updates from
the same event. (e) An illustrative sequence starts at $L_i=1.50$: an occluded
view gives $0$, a clear detector miss gives $-0.87$, a VLM judgment that the cup
is absent gives $-2.08$, and a later matched sighting gives $+2.48$. The dashed arrival branch
shows the detector fallback's $-1.56$ in place of the VLM reading. These values
use the configured default $(r,q)$ and are **not an experiment trace**. Rays
and objects are schematic; the implementation tests a fraction of depth samples
inside the projection. The filter's configured clip is $[-6,+3]$; only its
positive cap appears within the example's plot range.

Render with `python3 scripts/render_presence_belief.py`. Source logic is
`src/osg/objects/presence.py`; outputs are
`docs/figures/presence_belief.{svg,pdf,png}`.

---

## Presence on a real view

*Was `docs/PRESENCE_VIEW_CAPTION.md` — Presence-on-a-real-view figure.*

`docs/figures/presence_view.{png,pdf,svg}`, rendered by
`python scripts/render_presence_view.py` (`--survey` re-scores the poses).

**Caption.** *Presence belief on a view the agent really had. The camera pose
is one of the observations stored on a track of the saved map of HM3D
`00800-TEEsavR23oF`, re-rendered at the agent's own intrinsics; each outline
is that track's ellipsoid put through `Ellipsoid.project` — the same
dual-conic projection used to associate detections and to decide whether a
track should have been visible — and the number beside it is the track's
presence score $p=\sigma(L)$ at the end of the run. Colour runs from
disproved (red) to believed (green). The kitchen island's two tracks sit at
the $L=-6$ clamp after 14 misses in 16 and 21 misses in 25 exposed views; the
cabinet track on the right is at the $+3$ ceiling. A track whose mapped
geometry disagrees with the rendered depth by more than its own size is a
mapping error rather than a belief, and is not drawn.*

**What the numbers mean here.** This is a static mapping pass, so a low $p$ is
not an object that moved: it is a track the detector kept failing to confirm
from views where the stored ellipsoid was exposed — a mislocated or spurious
track being retired by the same arithmetic that retires a moved object.
That is the honest reading of this figure, and it is worth one clause in the
text; the moved-object case is what §C's schematic panel shows.

**Provenance.** Pose, intrinsics, ellipsoids, $L$, and the miss counters all
come from `outputs/maps_try_steps/00800-TEEsavR23oF.json`. The selection rule
is in the script: a track is drawn when its projected ellipse is between 11
and 110 px, its centre 0.9–6 m away and inside the frame, its evidence at
least 1.0, and its depth agrees with the rendered depth to within its own
extent; of those, up to six are kept, widest spread of belief first, none
within 95 px of another. The pose index is pinned in the script (`POSE`), so
a re-render reproduces the same figure. Only the word labels are annotation —
and they are the detector's own labels, mislabels included.

---

## Search and floor decision

*Was `docs/SEARCH_FLOOR_DECISION_CAPTION.md` — Multi-floor navigation figure.*

**Suggested caption.** Evidence-guided floor switching. Negative inspections
reduce support for local search candidates. A switch request requires a margin
in the mean residual search score and must satisfy timing constraints; a
credible, untested target on the current floor defers switching. The accepted
request first seeks a geometric stair flight. The dashed connector denotes a
candidate crossing between the two views, while green segments illustrate
approach and continued search. Only a completed transition creates a confirmed
StairEdge. Floor images and the stair guide come from HM3D scene 00873; the
policy state is schematic, not a recorded episode.

The figure shows the flight case. Detected staircase tracks and geometric
portals are fallback cues described in the text; a portal alone does not certify
reachability. No formula or numeric score is repeated in this figure.

Render with `python3 scripts/render_search_floor_decision.py`.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/search_floor_decision.pdf}
  \caption{Evidence-guided floor switching. Negative inspections reduce local
  support, and a score margin can trigger a switch subject to target-presence
  and timing constraints. The accepted request prioritizes a geometric stair
  flight. The dashed connector illustrates a candidate crossing; only a
  completed transition adds a confirmed StairEdge, after which search resumes
  on the destination floor. The policy state is schematic over rendered HM3D
  floor plans.}
  \label{fig:floor_switch}
\end{figure*}
```

---

## Floor score

*Was `docs/FLOOR_SCORE_FIGURE_CAPTION.md` — Floor-score figure (the §E formula, computed).*

`docs/figures/floor_score_formula.{png,pdf,svg}`, rendered by
`python scripts/render_floor_score_formula.py`. Companion to
`search_floor_decision.png`, same scene and same visual language.

**Caption.** *The floor score and the rule that acts on it, computed by the
pipeline's own code on the saved two-storey map of HM3D
`00873-bxsVRursffK` for the query "tin can", with the target's own mapped
track as the last believed position. **Left:** every eligible support surface
on each floor, sorted by $\bar b(x)\lambda(x)d(x)$; the dashed line is the
floor's score $M_\varphi$. Because $M_\varphi$ is a mean and not a sum, the
floor with 57 surfaces does not outscore the floor with 24 merely by being
larger. **Right:** the same two scores as the agent inspects the best
remaining surface on the floor it is standing on, each unsuccessful look
applying $\lambda(x)\leftarrow\lambda(x)(1-d(x))$. The scores cross after
eight inspections and the other floor clears the $1.15\times$ margin after
ten, at which point the switch request is accepted — the agent leaves because
what it has already searched is spent, not because it saw anything new.
**Bottom:** the order the decision is taken in.*

**Numbers in the figure** (printed by the script, so a re-render cannot drift
from the text): $|\mathcal S| = 57$ and $24$ surfaces, $M_\varphi = 0.104$ and
$0.049$ at the start, scores cross at 8 inspections, margin cleared at 10.

**What it must not be read as saying.** $M_\varphi$ is a mean over mapped
surfaces whose proximity term is a ground-plane distance, so it carries very
little information about *which storey* — measured on 00808's prior map, the
two storeys' means agree to within half a percent (`exploration/strategy.py`,
the comment above `floor_mass_rule`). That is why the rule is a mean plus a
margin rather than an argmax: the score is allowed to say "no opinion", and
the storey is then decided by the mechanisms that do carry floor information —
the stale anchor hold, and a storey disproved by a failed arrival.

**Provenance.** Bars, scores and the crossing are computed with
`build_container_candidates` and `InspectionLog` — the same functions
`ExplorationStrategy._select_surface` calls — at the shipped constants
(`search_detect_prob` 0.8, `search_surface_mass` 1.0,
`search_proximity_len_m` 1.0, `floor_mass_rule` mean, `floor_mass_margin`
1.15). The inspection *sequence* is the real decay rule applied to the best
remaining surface on the current floor; it is not replayed from one episode,
and the caption should not claim an episode outcome.

---

## Stair geometry

*Was `docs/STAIR_FIGURE_CAPTION.md` — Stair geometry figure.*

**Caption.** A real orthographic section of HM3D scene `00800-TEEsavR23oF`
shows the staircase and both landings (a). On the saved height layer, the
largest difference to neighbouring cells separates flat floor, riser-height
steps within the agent's 0.20 m climb limit, and wall-sized jumps (b). A local
steppable test needs a connected rise before its cells are trusted: the
per-cell test alone fired in 28/35 single-floor episodes. The saved stair mask
marks where the agent may step (d); as a traversability layer it is broader
than the detector's local riser band, and it gives neither a direction nor the
staircase mouth. A flight is recovered independently from contiguous cells
whose heights lie between the two storeys; its lowest and highest cells give a
foot and top (e). A nearby portal only indicates that the upper storey is
visible. After a completed upper-to-lower transition in this episode, the
graph writes a directed `StairEdge` between storey nodes with entry and exit
locations (c), preserving the containment hierarchy. The geometric flight is
the first climb target; a detected `stairs` track corroborates it, and a
portal is weaker evidence (f). Across the cited 100-episode run, such tracks
fired in only 15% of multi-floor episodes. The photo pointers
are explanatory; the map markers and edge values come from the saved map.

Render with `python3 scripts/render_stair_evidence.py`. It reads the real HM3D
section cached by `scripts/render_scene_graph_multifloor.py` at 2800 px and
section fraction 0.50, plus `outputs/maps_try_steps/00800-TEEsavR23oF.json`
and `.npz`. Output is `outputs/figures/F_stair_geometry.pdf` with PNG and SVG
variants.

---

## Stair anatomy figure (the two camera views)

**Caption.** *Height cues on one staircase. Colored dots mark the stair flight,
the set of connected intermediate-height cells recovered from the height
representation. The two circles denote its lowest and highest cells, the
candidate endpoints. **(a)** The lowest cell serves as the entry position
$\mathbf{x}_{\mathrm{entry}}$; the stair mask marks the steppable height change
at each riser, and the flat tread interiors are supplied by the flight.
**(b)** During descent, the same flight reduces to a sliver of tread at the edge
of the frame. The entry position therefore lies at the flight's highest cell.
A stair edge is added only after the traversal completes and certifies
traversability.*

Render with `python scripts/render_stair_anatomy.py`. The two views are
Habitat renders cached under `outputs/figures/`; the flight cells, their
heights and the lowest/highest markers are read from
`outputs/maps_try_steps/00800-TEEsavR23oF.{json,npz}` and projected into the
camera with a depth test, so nothing coloured is hand-placed — only the word
labels are annotations. The pose of (b) and the X come from the climb trace of
`00800-TEEsavR23oF__cross_anchor_01__50008__s0`
(`outputs/mf5_pass2_p1500/`), a different run in the same scene; say so if the
text could be read as one run. Output is
`outputs/figures/F_stair_anatomy.pdf` with PNG and SVG variants.

---

## Layout conditions

*Was `docs/LAYOUT_CONDITIONS_CAPTION.md` — Layout-conditions figure.*

`docs/figures/layout_conditions.{png,pdf,svg}`, rendered by
`python scripts/render_layout_conditions.py`.

**Caption.** *The four layout conditions, on the authored layouts of HM3D
`00873-bxsVRursffK`. A ring marks where the static layout puts an object —
the pose pass 1 maps and pass 2 begins by believing — and a disc where the
relocated layout puts it. **(a) static:** pass 2 sees the layout pass 1
mapped; this is the control that separates a stale map from a hard scene.
**(b) in-anchor:** the object stays on the same piece of furniture and moves
a few tens of centimetres, so the mapped pose is wrong but an arrival at it
still sees the object. **(c) cross-anchor:** the object moves to a different
piece of furniture on the same storey; the mapped pose is simply empty, and
nothing in the prior map points at the new one. **(d) multi-floor:** the
relocation crosses a storey, so the search has to give up the storey the map
points at before any surface on the right one can be scored. Distances and
anchors are read from the layout files; the panel examples are chosen by rule
(see below), not by hand.*

**Selection rule** (in `classify` / `pick`): an object is *in-anchor* when it
keeps the same anchor instance and moves less than 0.6 m, *multi-floor* when
its nearest storey changes, and *cross-anchor* otherwise. Panels (b) and (c)
show the largest qualifying move on the storey that has both, so the two
panels are comparable; (d) shows the smallest cross-storey move that fits
both rendered floors. The script prints what it picked.

**Provenance.** Object positions, anchor names and storey heights come from
`static_scene_config.json` and `dynamic_scene_config/{in_anchor,cross_anchor}/
layout_*.json` under `$OSG_YCB_MULTI_FLOOR_ROOT`; the floor plans are the
orthographic renders already used by the teaser and the floor-decision figure
(`hm3d_topdown_00873_*`).

⚠ **The layout set is being regenerated.** At the time of writing, every
scene's `static_scene_config.json` and `cross_anchor/layout_01.json` had been
rewritten (2026-09-16 02:53) while the `in_anchor` layouts still dated from
the previous evening. Against the new static layout, some objects in the
`in_anchor` files now read as cross-anchor or even cross-floor moves — which
is why this figure classifies every object itself instead of trusting the
directory name. Re-run the script after the authoring job finishes, and check
the printed picks before quoting the numbers in the paper.

---

## Relocation modes

*Was `docs/RELOCATION_MODES_CAPTION.md` — Target-relocation patterns.*

Figure files: [PNG](figures/relocation_modes.png), [PDF](figures/relocation_modes.pdf), [SVG](figures/relocation_modes.svg). The source views in [`figures/ycb_layouts/`](figures/ycb_layouts/) are direct Habitat-Sim renders of the authored JSON layouts.

Regenerate the source renders and composed figure with:

```bash
docker exec docker-nav-1 /opt/conda/envs/habitat/bin/python \
  /workspace/scripts/render_ycb_layout_assets.py
python3 scripts/render_relocation_modes.py
```

Suggested LaTeX caption:

```latex
\caption{Authored target-relocation patterns rendered from the YCB layouts. After a static mapping pass, a target is moved (a) within its original support surface (in-anchor), (b) to another surface on the same floor (cross-anchor), or (c) to another floor (a cross-floor subset of cross-anchor). The search pass loads the prior map and searches the changed scene. Blue and green rings mark the target in the prior and relocated layouts, respectively.}
```

The backgrounds and target meshes are direct orthographic renders of scene `00873-bxsVRursffK`, layout index 1. The rings, arrows, and text are explanatory overlays. Cross-floor is a subset of the cross-anchor condition in the authored benchmark; it is shown separately to make the storey change legible.
