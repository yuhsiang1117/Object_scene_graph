# MD-SG — Multi-floor Open-vocabulary Online 3D Scene Graph with Dynamic Handling Navigation

*The ICRA 2027 submission, transcribed in full. Sections I–V and the references are the
paper as submitted; nothing there is edited, corrected or updated. The three appendices are
**not** part of the paper — they map its symbols onto the modules that implement them, trace
each reported number back to the run that produced it, and record where the paper and this
repository disagree.*

*The system is called **MD-SG** in the paper and **OSG** everywhere in the code. See
Appendix C.*

---

## Abstract

Object-goal navigation in changing indoor environments requires robots to recover targets
relocated within or across floors. Previously observed object locations can become outdated,
requiring agents to revise their scene memory during search. We introduce MD-SG, an online,
open-vocabulary 3D scene graph navigation pipeline built around two complementary mechanisms:
(1) Presence-Belief Revision, which uses visibility-gated visual-geometric evidence to
reassess remembered object locations, and (2) Belief-Guided Search and Unified Selection,
which combines scene-graph priors with inspection evidence to balance candidate inspection
and frontier exploration. Integrated within a floor-aware scene graph, these mechanisms allow
the agent to reject outdated locations and redirect search within or across floors. On
single-floor cross-anchor relocation tasks in HM3D, MD-SG achieves a success rate of 52.8%,
compared with 30.2% for DualMap re-evaluated under the same protocol. We further introduce a
cross-floor relocation benchmark comprising 25 episodes across five scenes, on which MD-SG
achieves a success rate of 40.0%. A real-world demonstration additionally illustrates
recovery after object relocation. These results highlight the value of revisable scene
memory.

---

## I. Introduction

Object-goal navigation (ObjectNav) requires an embodied agent to navigate an unknown indoor
environment and locate an object specified by its semantic category or natural-language
description. However, most ObjectNav systems assume that objects remain stationary, whereas
humans and robots frequently relocate objects within or across floors in real environments.
For example, a cup previously observed in an upstairs bedroom may later be moved to a
downstairs kitchen. Humans can handle such changes by remembering the scene structure and
revising their knowledge of object locations. Three-dimensional scene graphs (3DSGs) provide
objects and their attributes as nodes and their relationships as edges. However, most
3DSG-based ObjectNav systems cannot revise outdated object locations, particularly when
objects move across floors. An effective agent must recognize that its prior knowledge is no
longer valid, search for the relocated object, and update its scene representation
accordingly.

Existing ObjectNav methods generally rely on semantic maps or 3DSGs. Semantic-map methods
such as VLFM [1] estimate the semantic relevance of exploration frontiers and project these
scores onto a 2D value map to guide navigation. ASCENT [2] extends this approach with
floor-level abstractions and staircase transitions, enabling online operation across floors.
While effective for exploration, semantic maps provide limited support for representing and
reasoning about relationships among individual objects. In contrast, 3DSGs explicitly capture
object-level semantics and spatial structure. Hydra [3] enables real-time scene graph
construction, but its closed-set semantics restrict navigation toward arbitrary targets.
HOV-SG [4] supports open-vocabulary, hierarchical representations of multi-story
environments, while ConceptGraphs [5] introduces open-vocabulary object graphs for perception
and planning. Nevertheless, these methods primarily model static object configurations and
are not designed to revise outdated object locations during navigation. Existing approaches
therefore address open-vocabulary mapping, online navigation, and dynamic handling
separately, and do not jointly support these capabilities.

Dynamic multi-floor ObjectNav requires the agent to reason about both outdated object
locations and possible floor transitions. When a target is absent from its expected location,
the agent must invalidate that location, maintain alternative hypotheses, and determine
whether to continue searching locally or move to another floor, as illustrated in Fig. 1.
DualMap [6] takes an important step toward this problem by combining open-vocabulary
perception, online semantic mapping, and object-level updates. It maintains a global
anchor-based abstract map for candidate selection and constructs a local concrete map during
navigation. However, its representation is designed around single-floor environments and does
not model floor connectivity. It therefore cannot maintain and compare relocation hypotheses
across floors or determine when invalidating local candidates should trigger a floor
transition. This limitation highlights a broader gap in dynamic ObjectNav rather than an
issue specific to DualMap [6].

To address this gap, we introduce MD-SG, an online, open-vocabulary 3DSG-based navigation
pipeline for dynamic, multi-floor environments. Our central insight is that a map in a
dynamic environment must not only preserve prior knowledge but also recognize when that
knowledge has become incorrect. MD-SG therefore represents object locations as revisable
beliefs within a floor-aware 3DSG rather than as fixed entries. Visibility-gated observations
update these beliefs and invalidate outdated locations, while the floor-aware graph allows
target hypotheses to be maintained and compared across floors. When candidates on the current
floor are invalidated, MD-SG can redirect navigation toward another floor. By jointly
supporting belief revision and cross-floor reasoning, MD-SG allows robots to preserve useful
knowledge from prior exploration while adapting to environmental changes, an essential
capability for persistent operation in homes, offices, and other shared spaces. The full
pipeline is at Fig. 2.

Our contributions are summarized as follows:

- We identify and formulate dynamic multi-floor ObjectNav, an overlooked setting in which
  objects may be relocated within or across floors and previously mapped locations can no
  longer be treated as fixed.
- We introduce MD-SG, an online, open-vocabulary 3DSG navigation pipeline that revises
  object-location beliefs and reasons over target hypotheses across floors.
- We establish, to our knowledge, the first ObjectNav benchmark specifically designed for
  dynamic multi-floor environments, featuring controlled same-floor and cross-floor object
  relocation. Experiments demonstrate improved target reacquisition over DualMap [6].

> **Fig. 1** — Dynamic cross-floor object search with MD-SG. Given the query "Find the cup,"
> the robot (1) revisits the remembered upstairs location, (2) verifies that the target is
> absent, (3) switches floors, and (4) recovers the cup in the downstairs kitchen. The
> floor-aware 3DSG revises the outdated object-location belief using new observations to guide
> subsequent search. The overlaid route and cup positions are schematic.

---

## II. Related Work

### A. Semantic Mapping and Multi-floor Exploration

Semantic mapping augments geometric maps with semantic cues that guide goal-directed
exploration within and across floors. VLFM [1] projects image–language similarity onto a
two-dimensional value map to rank exploration frontiers, and ASCENT [2] extends this frontier
reasoning to multi-floor buildings with hierarchical spatial abstraction and multi-floor
topology modeling. Both systems target the exploration of unfamiliar space. Recovering an
object from an outdated map poses a different challenge: previously observed locations can
mislead navigation, and the robot must reassess its memory as new evidence arrives. MD-SG
couples memory revision with multi-floor search, and evidence on one floor redirects
exploration across floors.

### B. Online and Dynamic 3D Scene Graphs

3D scene graphs organize spatial entities and their relationships into structured environment
memory, and recent work builds them online and maintains them under scene changes. Hydra [3]
demonstrates real-time hierarchical scene graph construction with closed-set semantics, while
ConceptGraphs [5] and HOV-SG [4] bring open-vocabulary semantics to object graphs and to
floor–room–object hierarchies for multi-story buildings. Clio [7] constructs open-set scene
graphs in real time and retains objects according to task relevance. These systems establish
the value of structured semantic memory. Target recovery from outdated object locations,
however, remains outside their scope. An early feature-based approach [8] models the
persistence of mapped landmarks in semi-static environments. Khronos [9] unifies short-term
motion and long-term scene changes within a spatio-temporal metric-semantic SLAM framework,
with a focus on temporally consistent reconstruction rather than goal-directed recovery.
DualMap [6] connects online open-vocabulary mapping with navigation through a global abstract
map and a local concrete map, and supports candidate reselection when the target is absent.
Its planar abstraction nevertheless does not represent floor transitions or coordinate
relocation hypotheses across floors. MD-SG combines the two lines through an online,
open-vocabulary 3D scene graph that performs evidence-driven belief revision and floor-aware
candidate selection. These mechanisms allow the robot to reconsider both where to search and
which floor to visit.

---

## III. Methodology

### A. Overview and Problem Formulation

We consider object search in a multi-floor environment whose object configuration may change
after initial exploration. Given a target query and a scene graph constructed during previous
operation, the robot must locate the target while revising object locations that are no
longer supported by current observations. At time $t$, the system receives an RGB image
$I_t$, a depth image $D_t$, and a camera pose $T_t$, and maintains a persistent scene graph
$G_t = (V_t, E_t)$. Its objective is to recover the target through observation-driven search
within and across floors.

MD-SG integrates Presence-Belief Revision with Belief-Guided Search and Unified Selection
within a floor-aware 3D scene graph. The graph stores object geometry, support relations, and
floor connectivity. Presence-Belief Revision reassesses remembered object locations using
current observations. When direct target candidates are rejected, Belief-Guided Search and
Unified Selection prioritizes support-surface inspections and frontier exploration using
scene priors, inspection evidence, and travel cost. A multi-floor policy uses the remaining
support for candidate locations to coordinate floor transitions. Observations gathered during
navigation update the same scene memory, closing the loop between belief revision and
subsequent search.

> **Fig. 2** — Overview of our mapping and navigation pipeline. The upper branch integrates
> object mapping, spatial mapping, and presence revision into a persistent 3D scene graph with
> building, storey, room, container, and object levels. Stair edges connect storey nodes. The
> lower branch uses the target query and updated graph to select a search goal and guide
> multi-floor navigation. Arrival and transition evidence feeds back into the mapping process.
> The graph overlaid on the rendered floor plans is schematic.

### B. Hierarchical 3D Scene Graph

The scene graph records object geometry, support relations, and floor connectivity. We use
RTAB-Map [10] as the geometric mapping backend in both simulation and real-world experiments.

#### 1) Perception Frontend

Pose-selected keyframes are processed by an open-vocabulary detection channel and a
class-agnostic proposal channel. We use YOLOE [11] to segment prompted indoor categories and
the target, and FastSAM [12] with MobileCLIP [13] to provide additional query-relevant
regions. The fused masks are back-projected using depth to form object observations. Tracks
enter the graph after repeated supporting observations, with channel-specific admission
criteria accounting for differences between detector confidence and image–language
similarity.

#### 2) Object Representation

Following VOOM [14], each object track $i$ is represented by a dual-quadric ellipsoid $Q_i$,
initialized from one observation and refined across views. Its projected ellipse,
parameterized by $(\mu_i, \Sigma_i)$, supports Wasserstein-based association and the
visibility assessment in Section III-C.

We adapt this representation to target-driven navigation with two constraints. Associations
require category agreement to preserve the semantics used for target retrieval. In addition,
navigation may provide a narrow range of viewing angles, which leaves geometric refinement
poorly constrained. We reject refinements that displace the ellipsoid center by more than
$\alpha_{\max} = 0.5\ \mathrm{m}$ from its preceding estimate.

#### 3) Scene Hierarchy

Containment relations organize the graph into four levels: floor, room, container, and
object. Object nodes reference their underlying tracks, sharing geometry and presence state.
Floors are identified primarily from the agent's standing height. Peaks in the ground-height
representation may additionally identify visible but unvisited floors. Within each floor,
doorway-scale morphological erosion separates connected free space into room regions.

Container nodes represent support surfaces that may hold a target. Support-surface
eligibility for a mapped object $o$ is defined as follows:

$$
o \in \mathcal{S} \iff \ell_o \in \mathcal{L}_{\mathrm{sup}}
\ \wedge\ h_{\min} \le h_o \le h_{\max}
\ \wedge\ A_o \ge A_{\min},
\tag{1}
$$

where $\mathcal{S}$ is the set of eligible support surfaces, $\ell_o$ the object category,
$\mathcal{L}_{\mathrm{sup}}$ the support-category set, $h_o$ the top height bounded by
$h_{\min}$ and $h_{\max}$, and $A_o$ the ground-plane footprint area bounded below by
$A_{\min}$. Joint semantic and geometric filtering reduces spurious search locations.
Same-category surface fragments are merged using footprint overlap relative to their spatial
extent. The graph, track beliefs, and floor connectivity are retained across missions.

#### 4) Cross-floor Connectivity

Stair connections augment the containment hierarchy with floor-level edges. Each confirmed
connection is stored as follows:

$$
e_{\mathrm{stair}} = (\varphi_{\mathrm{from}},\ \varphi_{\mathrm{to}},\
x_{\mathrm{entry}},\ x_{\mathrm{exit}}),
\tag{2}
$$

where $e_{\mathrm{stair}}$ denotes a stair edge, $\varphi_{\mathrm{from}}$ and
$\varphi_{\mathrm{to}}$ the source and destination floors, and $x_{\mathrm{entry}}$ and
$x_{\mathrm{exit}}$ the entry and exit positions. An edge is added only after a completed
traversal, which distinguishes confirmed connectivity from candidate transitions.

Before traversal, the height representation provides two complementary cues: stair masks
capture steppable height changes, while stair flights group connected intermediate-height
cells to estimate candidate endpoints, including flat tread regions. Detected staircase
tracks provide additional supporting evidence. Geometric portals indicate potentially visible
floors but do not establish traversability. Fig. 3 illustrates these cues.

> **Fig. 3** — Height cues on one staircase. Colored dots mark the *stair flight*, the set of
> connected intermediate-height cells recovered from the height representation. The two circles
> denote its lowest and highest cells, the candidate endpoints. **(a)** The lowest cell serves
> as the entry position $x_{\mathrm{entry}}$, the *stair mask* marks the steppable height change
> at each riser, and the flat tread interiors are supplied by the flight. **(b)** During
> descent, the same flight reduces to a sliver of tread at the edge of the frame. The entry
> position therefore lies at the flight's highest cell. A stair edge (Eq. 2) is added only after
> the traversal completes and certifies traversability.

### C. Presence-Belief Revision

Presence-Belief Revision estimates whether an object remains at its mapped location. A missed
detection alone cannot establish absence, since the object may be occluded or outside a
reliable viewing region. We therefore gate negative evidence by observation eligibility,
allowing informative misses to reduce belief while leaving uninformative observations
unchanged.

#### 1) Visibility-gated Evidence

Let $L_i$ denote the bounded log-odds score of track $i$, with presence score
$p_i = \sigma(L_i)$ and $\sigma$ the sigmoid function. Let $Z_i = 1$ indicate a detection
assigned to the track's projected extent. Each detection contributes to at most one track.
The eligibility indicator $E_i = 1$ denotes an observation from which the object could have
been detected at its mapped location. Given recall $r_i$ and false-alarm rate $q$, evidence
is accumulated as follows:

$$
\Delta L_i =
\begin{cases}
\log(r_i / q), & Z_i = 1,\\[2pt]
\log\!\big((1 - r_i)/(1 - q)\big), & Z_i = 0,\ E_i = 1,\\[2pt]
0, & Z_i = 0,\ E_i = 0,
\end{cases}
\tag{3}
$$

where $\Delta L_i$ denotes the evidence increment for track $i$, $Z_i$ the detection
indicator, $E_i$ the eligibility indicator, $r_i$ the view-dependent recall, and $q$ the
false-alarm rate. Detections thus support presence, whereas misses reduce belief only when
the observation is informative. Recall depends on apparent object size, range, and viewing
angle.

Eligibility requires a sufficiently large, readable projection within a reliable detection
range. Depth further distinguishes occlusion from an exposed location. Along the central
viewing ray, the near boundary is $z_{\mathrm{near}} = z_c - e_i - \delta$, where $z_c$ is
center depth, $e_i$ is the extent toward the camera, and $\delta$ is a tolerance. If a
sufficient fraction of valid projected depths lies closer than this boundary, the expected
object is occluded and $E_i = 0$. Background depths beyond the expected object do not exclude
the observation, which allows a miss at an exposed location to reduce belief. Fig. 4(a–c)
illustrates these cases.

#### 2) Bounded Updates and Arrival Verification

The accumulated evidence is clipped asymmetrically as follows:

$$
L_i \leftarrow \operatorname{clip}(L_i + \Delta L_i,\ L_{\min},\ L_{\max}),
\tag{4}
$$

where $L_i$ is the log-odds score of track $i$, $\Delta L_i$ the evidence increment, and
$L_{\min}$ and $L_{\max}$ the lower and upper bounds, with $L_{\min} < 0 < L_{\max}$ and
$|L_{\min}| > L_{\max}$. The positive ceiling limits confidence accumulated from historical
sightings, which allows subsequent misses to revise an outdated location. The lower bound
suppresses rejected locations while permitting recovery through renewed detections. We use
$p_i$ as a bounded decision score, without assuming a calibrated posterior.

Arrival verification uses the same update with observer-specific reliability parameters. A
readable view that confirms target absence supplies negative evidence through a
vision-language observer or an eligible detector scan. An unreadable view supplies no absence
evidence. A failed VLM call is not treated as a negative observation. An eligible detector
scan may instead provide the arrival reading. These alternative arrival observers complement
keyframe evidence, as illustrated in Fig. 4(d–e).

> **Fig. 4** — Presence belief update with MD-SG. Visibility and multi-sensor presence
> revision. (a–c) Nearer depth blocks a miss, exposed background renders it informative, and a
> matched sighting raises belief. (d) Keyframe detector misses and one arrival observer feed the
> same bounded update. (e) An illustrative trajectory uses configured sensor reliabilities. The
> VLM and detector fallback are alternatives.

### D. Belief-Guided Search and Unified Selection

Search proceeds in two stages. The robot first considers mapped target tracks whose presence
scores exceed a threshold. Eligible tracks are ranked by presence, with accumulated track
evidence breaking ties when scores reach their bounds. A separate rejection count handles
repeated arrivals at physically present but query-incompatible objects. When no eligible
direct target candidate remains, the robot considers mapped support surfaces and reachable
frontiers. Surface candidates encode plausible target locations, while frontiers provide
opportunities to discover unmapped locations. A shared utility balances their search value
against geodesic travel cost.

#### 1) Search Candidates

Let $\mathcal{F}$ denote reachable frontiers on the current floor and $\mathcal{S}$ its
eligible support surfaces. Frontier relevance $P_j$ combines an exploration prior,
information gain, and heading continuity. In the evaluated configuration, these terms are
geometric and independent of the target category. The support-surface score $b(x)$ is defined
as follows:

$$
b(x) = a(x)\,\kappa(\ell^\star, \ell_x)^{\gamma}
\exp\!\left(-\frac{\lVert c_x - c_{\mathrm{last}} \rVert}{L}\right),
\tag{5}
$$

where $a(x)$ indicates geometric compatibility with the target, $\kappa$ measures affinity
between target category $\ell^\star$ and surface category $\ell_x$, and $\gamma < 1$ softens
that affinity. The distance term favors proximity of the surface location $c_x$ to the
target's last mapped position $c_{\mathrm{last}}$, with scale $L$. Robot travel is accounted
for separately. The score $b(x)$ expresses prior search preference rather than a calibrated
probability. We obtain $\tilde{b}(x)$ by peak normalization at a fixed reference scale and
then apply inspection discounts. Keeping this scale fixed preserves the reduction in search
priority caused by unsuccessful inspections.

#### 2) Unified Utility and Inspection Memory

Let $c(g)$ be the geodesic travel cost to candidate $g$, $d(x)$ the estimated probability of
detecting a target present on surface $x$ during inspection, and $\rho(x)$ a persistent
inspection discount. Candidate utility is defined as follows:

$$
U(g) =
\begin{cases}
\beta\,P_j \,/\, c(f_j), & g = f_j \in \mathcal{F},\\[3pt]
\tilde{b}(x)\,\rho(x)\,d(x) \,/\, c(x), & g = x \in \mathcal{S},
\end{cases}
\tag{6}
$$

where $U(g)$ denotes the utility of candidate $g$, $c(g)$ the geodesic travel cost, $P_j$ the
relevance of frontier $f_j \in \mathcal{F}$, $\beta$ the coefficient that balances frontier
exploration against surface inspection, and $\tilde{b}(x)$, $\rho(x)$, and $d(x)$ the
normalized surface score, the inspection discount, and the detection probability of surface
$x \in \mathcal{S}$. The preferred candidate is selected as
$g^\star = \arg\max_{g \in \mathcal{F} \cup \mathcal{S}} U(g)$. Candidates are shortlisted by
relevance before geodesic costs are evaluated.

After an unsuccessful inspection, the surface discount is updated as follows:

$$
\rho(x) \leftarrow \rho(x)\,\big(1 - d(x)\big),
\tag{7}
$$

where $\rho(x)$ denotes the persistent inspection discount of surface $x$ and $d(x)$ its
detection probability during inspection. The discount records negative inspection evidence
across search attempts. An unsuccessful inspection with low detection probability only weakly
reduces the surface's priority, whereas a reliable negative inspection produces a larger
reduction. As inspected surfaces lose priority, frontier exploration receives higher priority
relative to surface inspection under the same utility.

### E. Multi-Floor Navigation Policy

The multi-floor policy answers two questions: on which storey the target is, and how to get
there.

**Which storey.** A score per storey averaged over its support surfaces cannot answer the
first question. The affinity priors say which furniture holds an object, not which storey, and
on our prior maps the per-storey means agree to within half a percent (Appendix C). The policy
therefore maintains a belief over the $K$ storeys the agent knows, built from the two kinds of
evidence that do carry storey information, the target's tracks and the search coverage:

$$
P_\varphi \propto \Big(w\,a_\varphi + \frac{1-w}{K}\Big)\big(1 - d\,c_\varphi\big),
\qquad
a_\varphi = 1 - \prod_{t \in \mathcal{T}_\varphi} \big(1 - p_t\big),
\tag{8}
$$

where $\mathcal{T}_\varphi$ are the target-labelled tracks on storey $\varphi$, prior-map
anchors and live detections alike, with presence beliefs $p_t$ from Section III-C, each
discounted by the track's identity rejections; $c_\varphi$ is the fraction of the storey's
mapped floor the agent has passed within 2 m of during the episode; $d = 0.85$ is the
probability that passing near the object means seeing it; and $w = 0.8$. A remembered location
the agent has inspected and found empty loses its presence belief and stops anchoring its
storey, and a storey searched without success decays through $c_\varphi$. A track's storey is
read off its height rather than off the layer it was mapped on, and levels closer than one
storey height are merged, because a pass-1 map stores a stair landing as a storey. The agent
requests storey $\varphi'$ once $P_{\varphi'} \ge \kappa\,P_{\varphi_{\mathrm{cur}}}$ with
$\kappa = 1.8$, and while it knows a second storey it never leaves on geometry alone.

**Where to cross.** Route selection for a requested switch rests on a single principle: a
floor transition is only as reliable as the evidence that certifies it. First come stair flights
whose treads are recovered from the height representation, ranked by how much of each the
detector labels as stairs; a descending flight is pursued to its lip, extrapolated up the
flight's own slope, because from a distance the upper treads are hidden behind the edge. Next
comes a staircase the robot has already traversed. The scene graph records a stair edge (Eq. 2)
only for transitions the robot completed, at the point where the new storey was confirmed, so
the mouth on the current storey is where a past traversal arrived on it from the target's side.
Geometric portals, which reveal a visible floor rather than a traversable path, come last. A
pursuit that fails is remembered at its place, and a remembered mouth is given up after two
failures beside it.

**Climbing.** The stairs are walked along a route planned over a two-level height map, which
keeps for every cell the lowest surface and the highest one below the camera. The route is a
shortest path over 0.1 m blocks whose steps rise at most one riser (0.25 m), ending on flat
floor at the requested storey's height, and it is re-planned as the view of the treads changes.
When no route is in view from the start floor, the agent first walks to the mouth and looks
round there: a descending flight is visible only from near its lip. On a descent the camera sees a tread only along lines steeper than the flight, so an unseen band
just ahead is bridged from its seen neighbours and surfaces overhead are ignored. The climb ends
when the agent stands within 0.25 m of the requested storey's height, and is abandoned when it
stops gaining height.

---

## IV. Experimental Results

### A. Experimental Settings

**Datasets.** We evaluate MD-SG on the dynamic HM3D dataset [15]. For single-floor dynamic
navigation, we utilize the environments constructed by DualMap [6], which employs custom
tools built upon the Habitat Simulator [16]. Object changes are categorized into two types:
in-anchor relocation, where an object is moved within the same anchor (e.g., shifting a cup
on a table), and cross-anchor relocation, where an object is transferred to a different
anchor (e.g., from a table to a shelf). We specifically focus on cross-anchor relocations, as
the spatial displacement of in-anchor relocations is too minor to rigorously validate our
dynamic handling pipeline. For multi-floor dynamic navigation, we construct a new benchmark
with the same toolkit employed by DualMap [6] on five two-storey HM3D scenes (00800, 00821,
00862, 00873, 00878), each holding six to eight YCB objects from eight classes. Every scene
has a static layout and a cross-anchor layout in which every object is moved to a different
anchor, most of them onto the other storey. Poses are produced by the toolkit's generator
rather than written by hand: an object rests on the top surface of an anchor whose kind and
room its class affords, one object per anchor, 0.35 m apart, every pose reachable from the
scene's single navigable island. One episode is defined per (layout, object, start), and three
seeds that differ only in the start pose give 73 episodes each, 219 in total: 111 static, 33
moved on the same storey, 36 moved one storey up and 39 one storey down. The start storey is
drawn uniformly from the storeys that hold objects, so 104 episodes start on the object's
storey and 115 elsewhere, and the agent must decide whether to change storey at all rather
than being rewarded only for leaving. Before each episode the agent inherits a prior map built
by its own exploration of the static layout (pass 1), in which the agent also views each object
from the layout's viewpoints, so the prior holds the object where it was. That map also carries
the detector's errors: of its 90 tracks bearing a benchmark object's label, 52 are not that
object. Appendix D gives the generator rules, the per-scene distributions, the destinations
against the affinity prior and the coverage audit of the prior map. Fig. 5 visualizes one
episode of each relocation mode together with the generation protocol.

**Metrics.** To evaluate the performance of object navigation tasks, we report Success Rate
(SR). Following the evaluation criteria of DualMap [6], an episode is considered successful
if the agent terminates within 1.0 meter of the target object. In dynamic scenes, success
additionally requires locating the target within three attempts. On the multi-floor benchmark
a stop must also be on the object's storey (the stop's floor within 1.0 m of the object's
height), every method gets the same three attempts, and we additionally report Success
weighted by Path Length (SPL), which scales success by the ratio of the shortest-path length
to the actual path length, first-attempt SR, and SR split by relocation class and by start
storey, each with a Wilson 95% interval over the 73 episodes of a seed. Single-floor episodes
use a 500-step budget, and multi-floor episodes use 1,000 steps to account for the longer
paths that cross-floor search requires.

**Compared Methods.** For our baseline comparisons, we include ConceptGraphs [5] and HOV-SG
[4]. Since our experimental setup strictly aligns with DualMap [6], we directly report the
performance metrics for these two methods as evaluated in their work. As noted in DualMap
[6], both baselines were explicitly extended in the Habitat Simulator to support navigation
using identical planning algorithms, ensuring a fair and consistent comparison. For DualMap
[6], we utilize its official open-source implementation. The re-evaluation over three seeds
follows the exact success criteria defined in their paper. It reproduces the published
in-anchor result exactly (64.8%) and the static result within 3.4 points (67.1% vs. 70.5%),
while the cross-anchor result falls to 30.2% against the published 60.3%. Table I accordingly
reports both the originally published metrics and the re-evaluated results under the unified
experimental setup.

**Implementation Details.** Our experiments are conducted in the Habitat simulator [16],
running on a workstation equipped with an NVIDIA RTX 4090 GPU and an AMD Ryzen 9 7950X3D CPU.
We adopt ASCENT [2] as our base navigation pipeline. Following their configuration, we
utilize D-FINE [17] and Grounding-DINO [18] for object detection, and Mobile-SAM [19] for
object segmentation. For comprehensive scene understanding and frontier reasoning, we employ
Places365 [20] for scene classification, RAM [21] for object tagging, BLIP-2 [22] for
semantic similarity, and Qwen2.5-7B-Instruct [23] for handling language queries. Furthermore,
to construct and maintain our dynamic 3D scene graph, we specifically employ YOLOE-11L-seg
[11] and FastSAM-s [12] paired with MobileCLIP-S2 [13] for open-vocabulary object detection
and segmentation.

> **Fig. 5** — Relocation modes in the evaluation benchmarks. Each panel contrasts the prior
> layout (blue circle: mapped target pose) with the relocated layout (green circle: target after
> the change): (a) in-anchor relocation moves the target within the same anchor on the same
> floor, (b) cross-anchor relocation transfers it to a different anchor on the same floor, and
> (c) cross-floor relocation moves it to an anchor on a different floor. Episodes follow a
> three-step protocol in which the agent first explores and maps each scene, selected YCB objects
> are then relocated, and the agent searches with the prior map.

**TABLE I** — Success Rate Comparison Across Different Scene Types and Methods on HM3D

| Scene Type | Method | 00829 | 00848 | 00880 | Trials | Avg. SR |
|---|---|---|---|---|---|---|
| Static | ConceptGraphs | 69.2% | 53.8% | 61.5% | 78 | 61.5% |
| Static | HOV-SG | 53.8% | 46.2% | 57.7% | 78 | 52.6% |
| Static | DualMap | 73.1% | 69.2% | 69.2% | 78 | 70.5% |
| Static | DualMap&nbsp;† | 80.8% | 63.0% | 57.7% | 78 | 67.1% |
| Static | **Ours** | 65.4% | 74.1% | 53.8% | 78 | **64.6%** |
| Dynamic (In-anchor) | DualMap | 66.7% | 66.7% | 61.1% | 54 | 64.8% |
| Dynamic (In-anchor) | DualMap&nbsp;† | 77.8% | 55.6% | 61.1% | 54 | 64.8% |
| Dynamic (In-anchor) | **Ours** | 88.9% | 33.3% | 66.7% | 54 | **63.0%** |
| Dynamic (Cross-anchor) | DualMap | 55.6% | 61.1% | 64.7% | 53 | 60.3% |
| Dynamic (Cross-anchor) | DualMap&nbsp;† | 33.3% | 33.3% | 23.5% | 53 | 30.2% |
| Dynamic (Cross-anchor) | **Ours** | 66.7% | 27.8% | 64.7% | 53 | **52.8%** |

† Indicates rerun results.

> **REVISION (2026-10-17).** The table below is the submitted v1 result (25 episodes); its
> problems are listed in Appendix B, "Table II, re-examined". It is replaced by the Benchmark v2
> results (73 episodes x 3 seeds, Appendix D) once the final arm's seeds 43 and 44 finish.

**TABLE II** — Success Rate Comparison Across Different Multi-floor Scene Cross-anchor Changes

| Scene Type | Method | Trials | Avg. SR | Avg. SPL |
|---|---|---|---|---|
| Multi-floor Cross-anchor | ASCENT | 25 | 0% | 0 |
| Multi-floor Cross-anchor | **Ours** | 25 | **40.0%** | **0.162** |

Evaluation on HM3D v0.2 with custom objects placed with multi-floor changes on the anchors.
The used scenes are 00800, 00821, 00862, 00873, 00878. With step limit set to 1000 for
dynamic handling.

### B. Simulation Experiment Results

Table I compares MD-SG with the semantic-map and scene-graph baselines across static,
in-anchor, and cross-anchor conditions, and MD-SG's relative standing improves as the
condition demands more relocation handling. In the static setting, which exercises mapping
quality alone, MD-SG outperforms ConceptGraphs [5] and HOV-SG [4] and performs on par with
re-evaluated DualMap [6] (64.6% vs. 67.1%). In-anchor displacements leave remembered
locations approximately valid, and MD-SG remains comparable to DualMap (63.0% vs. 64.8%).
Cross-anchor relocation invalidates the remembered location entirely, the condition where
presence revision and belief-guided reselection matter most, and MD-SG reaches 52.8% against
30.2% for DualMap re-evaluated under the same protocol, the largest margin.

Table II extends the evaluation to cross-floor relocation, where MD-SG recovers relocated
targets that the base pipeline cannot. Here, a switch requires the leading floor to exceed
the current one by a 15% margin on mean surface score, and is permitted only once per 50
steps, between steps 50 and 700 of the 1000-step budget. Recovery in this setting demands
exactly the two capabilities our pipeline introduces: revising the presence belief of a
remembered location and comparing the remaining hypotheses across floors before redirecting
the search. MD-SG reaches a success rate of 40.0% with an SPL of 0.162. ASCENT, the base
pipeline without these mechanisms, retains no prior-layout memory to revise, and fails to
recover any relocated target within the same budget.

### C. Ablation Study

The ablation targets the three components that MD-SG introduces for dynamic handling:
presence-belief revision, belief-guided surface scoring, and VLM-based absence verification.
Each is removed in turn, and the resulting pipeline is evaluated in the cross-anchor
condition, where these mechanisms are exercised most. Table III locates the gain in the
reject-and-redirect loop that the two belief mechanisms form, rather than in any single
component. Without presence-belief revision, the agent continues to act on stale locations.
All seven lost episodes commit to a prior-map track and the stale belief never decays, the
success rate falls from 52.8% to 41.5%. Without belief-guided surface scoring, the stale
hypothesis is correctly rejected but rejection frees the search without directing it, and the
agent falls back to frontier exploration. The success rate falls further to 35.8%, the
largest individual drop.

**TABLE III** — Contribution of Individual Mechanisms on the single-floor cross-anchor dataset

| Options | SR |
|---|---|
| w/o Presence-Belief Revision | 41.5% |
| w/o Belief-Guided Surface Scoring | 35.8% |
| w/o VLM-Based Absence Verification | 52.8% |
| **Full System** | **52.8%** |

The two contributions are not strictly additive, as belief-guided surface scoring becomes
most useful after an outdated hypothesis has been rejected. Moreover, belief-guided surface
scoring primarily improves candidate ordering rather than explicitly directing the robot
toward the highest-scored surfaces. In contrast, disabling VLM-based absence verification
leaves the success rate unchanged. Both configurations reach 52.8%. Further inspection shows
that the verification module is invoked only five times across 53 episodes, which indicates
that it is rarely activated under the current pipeline rather than inherently ineffective.
Even the weakest ablated variant remains above re-evaluated DualMap in the same condition
(35.8% vs. 30.2%), which separates two layers of gain: the floor-aware pipeline itself
already improves over the planar baseline, and the belief mechanisms contribute the rest.
These results identify presence-belief revision and belief-guided surface scoring as the
primary mechanisms behind MD-SG's improvement under dynamic object relocation.

### D. Real-World Deployment

To validate that our pipeline can handle dynamic environments and efficiently locate
relocated targets, we deployed MD-SG on a wheeled robot equipped with an RGB-D camera and
NVIDIA AGX Thor for model inference. As illustrated in Fig. 6, during the initial exploration
phase, the robot observed a blue bottle on Table A and an empty Table B. Subsequently, upon
receiving the language query target 'blue bottle,' the robot first navigated to Table A to
verify the target's presence. Upon detecting its absence, the MD-SG pipeline inferred a high
probability that the bottle had been moved to Table B. Consequently, the robot navigated to
Table B and successfully located the bottle.

> **Fig. 6** — Real-world navigation in dynamic environments with MD-SG. During initial
> exploration, the robot (1) observes a blue bottle on table A and (2) maps an empty table B. The
> robot (3) then receives the target query "blue bottle" after (4) the bottle has been
> transferred from table A to table B. The robot (5) revisits table A, detects that the bottle is
> absent, and shifts its presence belief toward table B. The robot (6) navigates to table B and
> retrieves the relocated target.

---

## V. Conclusion

We presented MD-SG, an online, open-vocabulary 3D scene graph navigation pipeline for dynamic
multi-floor ObjectNav, and showed that representing remembered object locations as revisable
beliefs, rather than fixed map entries, enables an agent to recover relocated targets within
and across floors. By combining presence belief updates, belief-guided candidate selection,
and floor-aware navigation, MD-SG connects the rejection of outdated locations with the
search that follows. The experimental results demonstrated improved single-floor cross-anchor
recovery over re-evaluated DualMap, and the cross-floor benchmark and real-world
demonstration extended the evaluation to multi-floor relocation and physical deployment. The
results position revisable, floor-aware scene memory as a first-class design axis for object
navigation in changing multi-floor environments. The proposed benchmark provides a direct way
to measure this capability and opens a concrete avenue for future research on dynamic
multi-floor object navigation.

---

## References

1. N. Yokoyama, S. Ha, D. Batra, J. Wang, and B. Bucher, "VLFM: Vision-language frontier maps
   for zero-shot semantic navigation," in *International Conference on Robotics and Automation
   (ICRA)*, 2024.
2. Z. Gong, R. Li, T. Hu, R. Qiu, L. Kong, L. Zhang, G. Zhao, Y. Ding, and J. Liang, "Stairway
   to success: An online floor-aware zero-shot object-goal navigation framework via LLM-driven
   coarse-to-fine exploration," *IEEE Robotics and Automation Letters*, 2026.
3. N. Hughes, Y. Chang, and L. Carlone, "Hydra: A real-time spatial perception system for 3D
   scene graph construction and optimization," in *Proceedings of Robotics: Science and
   Systems*, New York City, NY, USA, June 2022.
4. A. Werby, C. Huang, M. Büchner, A. Valada, and W. Burgard, "Hierarchical open-vocabulary 3D
   scene graphs for language-grounded robot navigation," in *Proceedings of Robotics: Science
   and Systems*, Delft, Netherlands, July 2024.
5. Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen, A. Agarwal, C. Rivera,
   W. Paul, K. Ellis, R. Chellappa, et al., "ConceptGraphs: Open-vocabulary 3D scene graphs for
   perception and planning," in *2024 IEEE International Conference on Robotics and Automation
   (ICRA)*. IEEE, 2024, pp. 5021–5028.
6. J. Jiang, Y. Zhu, Z. Wu, and J. Song, "DualMap: Online open-vocabulary semantic mapping for
   natural language navigation in dynamic changing scenes," *IEEE Robotics and Automation
   Letters*, vol. 10, no. 12, pp. 12612–12619, 2025.
7. D. Maggio, Y. Chang, N. Hughes, M. Trang, D. Griffith, C. Dougherty, E. Cristofalo,
   L. Schmid, and L. Carlone, "Clio: Real-time task-driven open-set 3D scene graphs," *IEEE
   Robotics and Automation Letters*, 2024.
8. D. M. Rosen, J. Mason, and J. J. Leonard, "Towards lifelong feature-based mapping in
   semi-static environments," in *2016 IEEE International Conference on Robotics and Automation
   (ICRA)*, 2016, pp. 1063–1070.
9. L. Schmid, M. Abate, Y. Chang, and L. Carlone, "Khronos: A unified approach for
   spatio-temporal metric-semantic SLAM in dynamic environments," in *Proc. of Robotics: Science
   and Systems (RSS)*, Delft, Netherlands, July 2024.
10. M. Labbé and F. Michaud, "RTAB-Map as an open-source lidar and visual simultaneous
    localization and mapping library for large-scale and long-term online operation," *Journal
    of Field Robotics*, vol. 36, no. 2, pp. 416–446, 2019.
11. A. Wang, L. Liu, H. Chen, Z. Lin, J. Han, and G. Ding, "YOLOE: Real-time seeing anything,"
    2025. <https://arxiv.org/abs/2503.07465>
12. X. Zhao, W. Ding, Y. An, Y. Du, T. Yu, M. Li, M. Tang, and J. Wang, "Fast segment
    anything," 2023.
13. P. K. A. Vasu, H. Pouransari, F. Faghri, R. Vemulapalli, and O. Tuzel, "MobileCLIP: Fast
    image-text models through multi-modal reinforced training," in *Proceedings of the IEEE/CVF
    Conference on Computer Vision and Pattern Recognition (CVPR)*, June 2024.
14. Y. Wang, C. Jiang, and X. Chen, "VOOM: Robust visual object odometry and mapping using
    hierarchical landmarks," in *Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA)*,
    2024.
15. S. K. Ramakrishnan, A. Gokaslan, E. Wijmans, O. Maksymets, A. Clegg, J. M. Turner,
    E. Undersander, W. Galuba, A. Westbury, A. X. Chang, et al., "Habitat-Matterport 3D dataset
    (HM3D): 1000 large-scale 3D environments for embodied AI," in *Thirty-fifth Conference on
    Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*, 2021.
16. M. Savva, A. Kadian, O. Maksymets, Y. Zhao, E. Wijmans, B. Jain, J. Straub, J. Liu,
    V. Koltun, J. Malik, D. Parikh, and D. Batra, "Habitat: A platform for embodied AI
    research," in *Proceedings of the IEEE/CVF International Conference on Computer Vision
    (ICCV)*, 2019.
17. Y. Peng, H. Li, P. Wu, Y. Zhang, X. Sun, and F. Wu, "D-FINE: Redefine regression task in
    DETRs as fine-grained distribution refinement," 2024.
18. S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, C. Li, J. Yang, H. Su, J. Zhu, et al.,
    "Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection,"
    *arXiv preprint arXiv:2303.05499*, 2023.
19. C. Zhang, D. Han, Y. Qiao, J. U. Kim, S.-H. Bae, S. Lee, and C. S. Hong, "Faster segment
    anything: Towards lightweight SAM for mobile applications," *arXiv preprint
    arXiv:2306.14289*, 2023.
20. B. Zhou, A. Lapedriza, A. Khosla, A. Oliva, and A. Torralba, "Places: A 10 million image
    database for scene recognition," *IEEE Transactions on Pattern Analysis and Machine
    Intelligence*, 2017.
21. Y. Zhang, X. Huang, J. Ma, Z. Li, Z. Luo, Y. Xie, Y. Qin, T. Luo, Y. Li, S. Liu, Y. Guo,
    and L. Zhang, "Recognize anything: A strong image tagging model," 2023.
    <https://arxiv.org/abs/2306.03514>
22. J. Li, D. Li, S. Savarese, and S. Hoi, "BLIP-2: Bootstrapping language-image pre-training
    with frozen image encoders and large language models," 2023.
    <https://arxiv.org/abs/2301.12597>
23. A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei,
    et al., "Qwen2.5 technical report," *arXiv preprint arXiv:2412.15115*, 2024.

---
---

# Appendices

*Not part of the submission. Written 2026-09-17 so the paper and the repository can be read
against each other.*

## Appendix A — paper → code

| paper | implementation |
|---|---|
| §III-B.1 perception front end (YOLOE + FastSAM/MobileCLIP) | `perception/detector.py`, `perception/region_proposer.py`, `perception/feature_encoder.py` |
| §III-B.2 dual-quadric ellipsoid; Wasserstein association | `objects/ellipsoid.py`, `objects/optimization.py`, `objects/association.py` |
| §III-B.2 category agreement on association | `objects/object_layer.py:52` (`assoc_category_gate`) |
| §III-B.2 $\alpha_{\max} = 0.5\ \mathrm{m}$ refinement reject | `scene_graph.refine_max_center_move_m` = 0.5 |
| **Eq. 1** support-surface eligibility | `graph/priors.py:229` `affords()` over the per-category `AFFORDANCE` table (`:138`); applied in `graph/containers.py` with `min_area_m2` |
| **Eq. 2** stair edge $(\varphi_{\mathrm{from}}, \varphi_{\mathrm{to}}, x_{\mathrm{entry}}, x_{\mathrm{exit}})$ | `mapping/stairs.py`; persisted across missions by `graph/map_store.py` |
| §III-B.3 floor / room / container / object | `graph/scene_graph.py`; `mapping/floors.py` (standing height), `mapping/floor_stack.py` (per-storey grids), `mapping/room_seg.py` (doorway-scale erosion) |
| §III-B.4 stair masks, flights, portals | `mapping/stairs.py` `apply_stair_mask` / `find_flights` / `mouth_xy`; `mapping/portals.py` |
| **Eq. 3** visibility-gated evidence ($Z_i$, $E_i$, $r_i$, $q$) | `objects/presence.py:190` `expectation()` decides $E_i$; `:318` `update()` applies the three channels |
| §III-C.1 $z_{\mathrm{near}} = z_c - e_i - \delta$, near-side-only gate | `objects/presence.py:245–257` |
| §III-C.1 recall from size, range, incidence | `objects/presence.py:66` `RecallModel` |
| **Eq. 4** asymmetric clip, $\lvert L_{\min}\rvert > L_{\max}$ | `objects/presence.py:360`; `l_clamp` = 6.0, `l_clamp_pos` = 3.0 |
| §III-C.2 per-observer $(r, q)$ at arrival | `objects/presence.py:380` `apply_reading()`; `verification/absence.py` |
| §III-D direct-target stage, rejection count | `agent/candidate.py` |
| **Eq. 5** $b(x)$ | `exploration/search_belief.py:96` `container_prior()`; $\kappa$ and $\gamma$ are `AFFINITY_POWER` / `CONTAINER_AFFINITY` in `graph/priors.py` |
| **Eq. 6** unified utility $U(g)$ | `exploration/search_belief.py:157` `select_candidate()`, driven from `exploration/strategy.py` |
| **Eq. 7** $\rho(x) \leftarrow \rho(x)(1 - d(x))$ | `exploration/search_belief.py:72` `InspectionLog.searched()` |
| **Eq. 8** storey belief $P_\varphi$ ($w$, $d$, $c_\varphi$, $a_\varphi$) | `exploration/storey_belief.py` `StoreyBelief`; `exploration.storey_belief`, `storey_anchor_weight` = 0.8, `storey_detect_prob` = 0.85, `storey_cover_radius_m` = 2.0, `storey_rejection_discount` = 0.3 |
| §III-E request at $\kappa = 1.8$; no geometric switch while a second storey is known | `exploration.storey_switch_ratio` = 1.8; `exploration/strategy.py` (`undirected_switch_skipped`) |
| §III-E landings merged, tracks filed by height | `StoreyBelief.storeys` / `storey_of_height`; `eval/prior_map.py` `drop_prior_landings` (`ycb.drop_prior_landings`) |
| §III-E flights > traversed staircase > portals; flights ranked by the detector's stair mask; descent to its lip | `agent/floor_policy.py` `try_switch()`, `_rank_flights`; `floor.climb_targets` = `flights_first`, `flights_prefer_stair_mask`; `mapping/stairs.py` `mouth_xy` / `descent_mouth_xy` (`floor.descent_goal` = `extrapolated`) |
| §III-E traversed staircase: the arrival on this storey | `agent/floor_policy.py` `_remembered_arrival`; `floor.prior_stair_mouth` = `arrival`, `prefer_prior_stairs` (ahead of portals), `prior_stair_max_failures` = 2. `prior_stairs_before_flights` (ahead of flights) is off: on the 3 seed-42 ascents it changed, no new success and 00821 static 50001 lost |
| §III-E route climb over the two-level height map | `mapping/stair_route.py` `plan_stair_route`; `agent/climb.py`; `agent.climb_route`, `climb_route_gap_fill_m` = 0.8, `climb_route_approach_steps` = 60 |
| §III-E walk to the mouth when no route is in view | `agent/climb.py` `_walk_to_mouth`; `plan_stair_route(explore=True, toward_xy=...)`; `agent.climb_route_mouth_first` |
| §III-E (submitted) $M_\varphi$ surface mean, margin, cadence, window | `exploration/strategy.py` `_select_surface`, `exploration.floor_mass_rule` = `"mean"`, `floor_mass_margin` = 1.15; `floor.switch_min_interval` = 50, `no_switch_before` = 50, `no_switch_after_frac` = 0.7 in `mapping/portals.py` `may_switch()` |

## Appendix B — number → run provenance

The arm is `+experiment=dualmap_osg_unified` for the single-floor tables and
`+experiment=mf5_osg_on_ascent_map` for the multi-floor one. SR is the mean of the `success`
field over `episodes.jsonl` (`eval/metrics.py:13` `aggregate`), which is DualMap's released
rule — a stop within 1.0 m of the object, within three attempts — not Habitat's 0.18 m
viewpoint rule.

| paper row | expected | run directory | status |
|---|---|---|---|
| Table I, Ours, in-anchor 63.0% | 34 / 54 | `outputs/osg_v4_fuse_cls_full` | the artifact holds it. Re-runs: **35/54** on the original prior maps (`outputs/RERUN_UPLOADMAP`, 2026-09-27), 30/54 on rebuilt maps (`outputs/RERUN/dualmap186`, 09-20) |
| Table I, Ours, cross-anchor 52.8% | 28 / 53 | `outputs/osg_v4_fuse_cls_full` | the artifact holds it. Re-runs: **25/53** on the original maps, 20/53 on rebuilt maps; the 28→25 residue is the container image (`CROSS_ANCHOR_REPRODUCTION.md`) |
| Table II, ASCENT 0% / 0 | 0 / 25 | `outputs/ascent_crossanchor` | **reproduces** |
| Table I, Ours, static 64.6% | 51 / 79 | `outputs/RERUN/dualmap186` | **51/79 on rebuilt maps** (09-20); 42/79 on the original maps (09-27). The locked run held only the 107 dynamic trials, so no static run from the paper's image exists |
| Table II, Ours 40.0% / 0.162 | 10 / 25 | `outputs/RERUN/mf5` | ~~reproduces, one episode above: 11/25 = 44.0%, SPL 0.196~~ -- **does not**: the re-run is on re-authored layouts and two of its 11 successes stopped a storey away; see *Table II, re-examined* below. The earlier `outputs/mf5_pass2_final` holds 9/25 = 36.0%, SPL 0.138 |
| Table III, w/o presence revision 41.5% | 22 / 53 | — | **not on this machine**: no 53-episode run exists |
| Table III, w/o surface scoring 35.8% | 19 / 53 | — | **not on this machine** |
| Table III, w/o VLM verification 52.8% | 28 / 53 | — | **not on this machine** as a separate run |

> **The single-floor dynamic rows no longer re-derive from rebuilt maps.** (With the original prior maps recovered on 2026-09-27: in-anchor 35/54 = 64.8%, cross-anchor 25/53 = 47.2%, static 42/79 = 53.2% — `docs/CROSS_ANCHOR_REPRODUCTION.md`.) Re-run 2026-09-20: static
> 51/79 = 64.6%, in-anchor 30/54 = 55.6%, cross-anchor 20/53 = 37.7%. The cause is
> not the code — the tree that produced 62/107 (`74e0eab`) agrees with HEAD on 18 of
> 18 trials when given the same prior map — but the prior map, which rebuilds
> deterministically at 534 tracks against the 682 the locked run loaded.
> [`CROSS_ANCHOR_REPRODUCTION.md`](CROSS_ANCHOR_REPRODUCTION.md) has the full audit.
> Table II: 11/25 = 44.0%, SPL 0.196 -- but see *Table II, re-examined* below.

The three ablations are re-runnable as one override each on the full arm:

```bash
python scripts/run_eval.py +experiment=dualmap_osg_unified \
    scene_graph.presence.enabled=false          # w/o Presence-Belief Revision
python scripts/run_eval.py +experiment=dualmap_osg_unified \
    exploration.search_posterior=false          # w/o Belief-Guided Surface Scoring
python scripts/run_eval.py +experiment=dualmap_osg_unified \
    verification=off                            # w/o VLM-Based Absence Verification
```

Two smaller disagreements, recorded rather than resolved:

- Table I gives the static condition **78** trials (26 per scene, which is what the
  per-scene percentages divide into). `docs/DUALMAP_OFFICIAL_RERUN.md` gives **79**.
- *Resolved:* `configs/experiment/mf5_osg_unified.yaml` used to list scene **00808** and
  **26** relocations; it now names the paper's five scenes (00800, 00821, **00862**, 00873,
  00878) and 25 episodes, matching `mf5_ascent_crossanchor.yaml`.

### Table II, re-examined (2026-10-15)

Four problems with what Table II measures, found while answering the reviews. Each is
checked against the runs on disk; `scripts/analyze_multifloor.py` reproduces the numbers.

| | finding | evidence |
|---|---|---|
| 1 | **A stop a storey away scored.** The success rule is horizontal-only (DualMap's single-storey rule). | `outputs/RERUN/mf5`: 00862 and 00873's cracker boxes "succeeded" 2.33 / 2.11 m from the object in 3D, on the wrong storey. On the object's storey the re-run is **9/25 = 36.0%** [Wilson 95%: 20, 55], not 11/25. Fixed: `ycb.object_success_max_floor_dy_m` (default 1.0). |
| 2 | **The two arms ran different episodes.** The layout files were re-authored at 2026-09-16 02:39. | `outputs/mf5_pass2_final` (OSG, 09-16 01:22) ran the earlier layouts, on which all 25 relocations are **downward**. `outputs/ascent_crossanchor` (ASCENT, 09-16 08:09) and `outputs/RERUN/mf5` ran the current ones: **12 upward, 13 downward**, other layout hashes. So "Table II reproduces" compares two benchmarks. On the current layouts OSG scores 1/12 upward and 8/13 downward. |
| 3 | **The prior map never held the targets.** | 0 of 25 episodes have a same-label track within 1 m of the target's static pose in `outputs/maps_p1500_osg`; 24 have no track of the label at all. Pass 1 kept one episode's graph (best-of), whose world model was prompted with only its own target. So Table II never exercised presence-belief revision, and Fig. 1's "revisit, find absent, switch storey" did not happen in the benchmark. Fixed: pass 1 accumulates (`scripts/run_prior_maps.sh`, `ycb.map_in`), and `audit_map_coverage.py` gates on every static target being in the graph. |
| 4 | **ASCENT could not name most targets.** | `mf5_ascent_crossanchor` never set `detector.gdino_target_url`, so D-FINE (closed-set COCO) looked for a cracker box, toy airplane, tin can. The target was in view in 3 of 25 episodes and never detected. ASCENT's own non-COCO branch (GroundingDINO by name) is `configs/experiment/mfv2_ascent.yaml`. |

Benchmark v2 (`configs/experiment/mfv2_dataset.yaml`) is the replacement: static and
cross-anchor layouts with every relocation (same-storey ones included), starts on a uniformly
drawn storey, success on the object's storey, seeds 42/43/44.

## Appendix C — naming, and one note on the submitted Eq. 8

**MD-SG is OSG.** The paper's system name does not appear in the code, which says `osg`
throughout — the package, the config namespace, and every preset. Renaming would rotate all
87 preset fingerprints and 505 pinned configuration keys in
`tests/unit/golden/` for no measured gain, so the code keeps its name and this file records
the mapping.

| paper | code |
|---|---|
| MD-SG | the `osg` package; the `nav_agent` policy on the `dualmap_osg_unified` / `mf5_osg_on_ascent_map` arms |
| Presence-Belief Revision | `objects/presence.py`, wired by `pipeline/beliefs.py` |
| Belief-Guided Search and Unified Selection | `exploration/search_belief.py` + `exploration/strategy.py` |
| Multi-Floor Navigation Policy | `agent/floor_policy.py` + `mapping/portals.py` |
| ASCENT, the base pipeline | `src/navigation/`, a line-by-line transcription (see its README for the F1–F14 fidelity notes) |

**Why the submitted Eq. 8 took the mean, and why the revision replaced it.** The code comment beside it records the measurement that chose
it. On 00808's prior map, over five targets, floor 0 held 78 surfaces summing to 52.5 and
floor 1 held 33 summing to 22.3 — means of 0.673 and 0.676, agreeing to within half a
percent. The container priors carry essentially no information about *which storey*. A sum
turns that tie into a 2.4× preference for the larger floor, so an agent upstairs asks to go
down every round whatever it is looking for. The mean makes the score scale-free and the 15%
margin then requires another storey to be decisively better, which together let the posterior
return "no opinion" — on this evidence the honest answer — and leave the storey choice to the
mechanisms that do carry floor information.

That abstention is why the revision replaces the surface mean with the storey belief of Eq. 8.
With the mean as the only storey signal, the decision fell to the mechanisms around it: a switch
after two failed attempts, which on Benchmark v2 seed 42 were false-positive STOPs in 7 of the 11
episodes that left the object's own storey; and a geometric "nothing near is left here" gate that
took the nearest flight in either direction, so 9 of the 28 episodes that had to go down pursued
only flights going up. The belief reads what does carry storey information: where the target's
tracks are, how far each is still believed, and how much of each storey has been searched.

## Appendix D — the multi-floor benchmark (Benchmark v2)

What the reviewers found undocumented: how the episodes were generated, how storeys and starts
are distributed, where moved objects go, how a run is scored and what the prior map holds.
`docs/BENCHMARK_V2.md` is the working version; every number below is re-derivable with
`scripts/describe_benchmark.py +experiment=mfv2_dataset --seeds 42 43 44` (episodes) and
`scripts/audit_map_coverage.py` (prior map, `outputs/mfv2/audit_v2.txt`).

**Scenes and objects.** Five HM3D v0.2 scenes with objects on two storeys — 00800, 00821,
00862, 00873, 00878 — and six to eight YCB objects per scene from eight classes: cracker box,
tin can (tomato soup), blue plastic pitcher, bowl, banana, toy airplane, coffee can, yellow
bottle (mustard).

**Layouts and the generator.** Each scene has two layouts, both produced by the data
collector's generator (`replace_placement.py`) and never written as coordinates: *static*, the
world pass 1 maps, and *cross_anchor_01*, in which every object is moved to a different anchor.
Every pose satisfies the generator's rules: the object rests on the top surface of an anchor
whose kind and room its class affords (food items: kitchen, dining room, living room); one
object per anchor; a 0.35 m keep-out between objects; every pose reachable from the scene's
one navigable island (worst approach 0.51–1.01 m). To balance the direction of cross-storey
moves, some static and cross-anchor poses were exchanged and both re-placed through the same
generator; upper-storey anchor supply caps the descents two scenes can hold (00873 has one
affordable anchor upstairs, 00878 two). The multi-floor `in_anchor_01` layouts are not used:
they move objects across anchors and storeys, which is not what the name says.

**Episodes.** One episode per (layout, object, seed); the seeds 42, 43 and 44 differ only in
the start pose — 73 episodes per seed, 219 in all. An episode's class is read off its geometry,
not off the layout name:

| class | episodes (3 seeds) | start on the object's storey | start elsewhere | start-to-goal geodesic, m (p10 / p50 / p90) |
|---|---:|---:|---:|---:|
| static | 111 | 45 | 66 | 5.2 / 10.2 / 23.0 |
| moved, same storey | 33 | 13 | 20 | 4.1 / 9.7 / 17.0 |
| moved one storey up | 36 | 22 | 14 | 4.3 / 14.4 / 25.2 |
| moved one storey down | 39 | 24 | 15 | 3.5 / 6.6 / 21.1 |
| all | 219 | 104 | 115 | 4.4 / 9.7 / 21.1 |

**Starts.** The start storey is drawn uniformly from the storeys holding at least one of the
scene's objects, then a navigable point on it, so about half the episodes start on the object's
storey and the agent must decide whether to change storey at all (the submitted Table II
started every episode on the storey the object had left):

| scene | start storeys: height (starts over 3 seeds) |
|---|---|
| 00800 | 0.2 m (16), 3.2 m (20) |
| 00821 | −3.5 m (22), 0.2 m (23) |
| 00862 | 0.1 m (22), 3.3 m (26) |
| 00873 | −3.0 m (22), 0.0 m (20) |
| 00878 | 0.1 m (22), 2.8 m (26) |

**Where moved objects go.** The 36 relocations (the layouts are the same for every seed) end on
a cabinet 17 times (kitchen, sink and bathroom cabinets included), a table 13 times (coffee,
dining, display and side tables), a counter, worktop or kitchen island 5 times and a bedroom
chest once. Against the affinity prior the agent searches with, 20 of the 36 destinations are in
the prior's top three for that class, 9 are its first choice and 33 are ranked at all. This
answers the reviewer's point on non-semantic destinations: the single-floor DualMap release
places a tomato-soup can on a bed seven times, where no semantic prior can pay; here it can,
and the affinity ablations (LLM-generated prior, flat prior) measure how much.

**Scoring.** A STOP succeeds within 1.0 m of the object horizontally and on its storey (the
stop's floor height within 1.0 m of the object's). Every arm gets three attempts, a failed one
re-arming the agent with its map kept, within 1,000 steps. We report SR with a Wilson 95%
interval, by class and by start storey, alongside first-attempt SR and the horizontal-only rule
of the submitted version.

**The prior map (pass 1).** ASCENT's policy walks every static episode of the scene at pass 2's
resolution (1280×960, YOLOE-11L at 1280), and after each episode the world model is shown its
target from three of the layout's own viewpoints, so the map the agent inherits has seen every
object. Two artifacts accumulate over the scene's episodes: the obstacle maps pass 2 plans over,
and the scene graph. Audit:

| scene | storeys mapped (free fraction) | static targets with a same-label track within 1 m |
|---|---|---:|
| 00800 | 0.81, 0.89 | 6 / 6 |
| 00821 | 0.92, 0.88 | 7 / 8 (cracker box missing) |
| 00862 | 0.91, 0.89, 0.89 | 7 / 8 (cracker box missing) |
| 00873 | 0.87, 0.90 | 6 / 7 (tomato-soup can missing) |
| 00878 | 0.95, 0.90, **0.00** | 7 / 8 (toy airplane missing) |

00878's top storey was never walked in pass 1, so it has no occupancy; its objects are in the
graph from the target tour, and pass 2 explores that storey without a prior occupancy map. The
graph also records where pass 1 changed storey (its stair connectivity) — the point at which
the floor estimator committed the new storey, i.e. an arrival on the storey reached — and, in
00800 and 00821, a stair landing stored as a storey of its own, which the arms with the storey
belief remove before the episode.

**False positives in the prior.** Of the 90 graph tracks carrying a benchmark object's label,
52 are not the object. A static episode starting on the other storey often commits first to one
of those on its start storey: on seed 42 (OSG baseline) 4 of 17 such episodes succeeded, against
11 of 11 starting on the object's storey. The prior is therefore wrong in both directions the
method is built for — objects that moved, and objects that were never there.

**Detector vocabulary.** Every arm's detector is given the episode's target. The OSG arm's
fixed vocabulary, which decides what else is mapped, names five of the eight classes (cracker
box, tin can, blue plastic pitcher, bowl, banana) but not toy airplane, coffee can or yellow
bottle; those are mapped only as a target. The ASCENT arm runs its own detector (D-FINE over a
fixed vocabulary of 41 classes, Grounding-DINO for a target outside COCO), and the ASCENT-YOLOE
arm swaps in OSG's detector and vocabulary so that perception is aligned. Neither detector is
zero-shot on this benchmark: the OSG vocabulary and its lowered score thresholds for the
pitcher, the tin can and the banana (0.2 against 0.3) were chosen on these objects
(`docs/ZERO_SHOT_AUDIT.md`), and ASCENT-YOLOE is given the same advantage.
