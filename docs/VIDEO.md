# The ICRA 2027 video

Three minutes, 20 MB, one story told in six parts: **the problem → our system → its three
modules → experiments and comparison → a real robot → conclusion.** The cut leads with
simulator footage — the agent moving through HM3D in first person with its detections on
screen — uses the paper's figures only as short inserts, and closes on the real Stretch 3 with
the phone camera and the Thor's RViz side by side. An outline card follows the title and a
chip in the top-right corner names the current section throughout; every spoken line is also
a subtitle. `scripts/video/shotlist.json` is the storyboard as data and
`scripts/video/narration.json` the words; the table below is printed from both
(`python scripts/video/build_video.py --print-storyboard`), so the doc and the cut cannot
drift apart.

```bash
/tmp/.conda/envs/tts/bin/python scripts/video/make_narration.py   # voice.wav + cues.json + icra2027.srt + transcript.md
python scripts/video/make_assets.py            # every shot -> outputs/video/segments/*.mp4 (~12 min); subtitles from cues.json
python scripts/video/build_video.py --narration outputs/video/narration/voice.wav   # -> icra2027_rough.mp4, ≤ 180 s / 20 MB
python scripts/video/make_assets.py --frame hero 12.0          # one PNG at t=12 s of a shot, to look at it
```

The order matters: the subtitles are drawn from the timings of the synthesised audio, so the
narration is made first, and `build_video.py` refuses to assemble a cut whose `cues.json` is
older than the words or the shot list.

Everything under `outputs/video/` is generated. The real-robot recordings live in
`assets/real_experiment/` (gitignored, 1.1 GB): `human_camera/` holds the two setup photos and
the three phone takes, `rviz/` the three screen captures of the Thor's operator windows, one
pair per step of `docs/DEMO_FLOW.md`.

## Storyboard

| t | section | shot | beat | on screen | narration |
|---|---|---|---|---|---|
| 0:00 (+4 s) |  | `title` | Title over the hero clip's detection frame | MD-SG — title, venue | MD-SG: a scene graph that lets a map be wrong, and recover. |
| 0:04 (+5 s) |  | `outline` | The outline: six parts | 1 problem · 2 system · 3 modules · 4 experiments · 5 real robot · 6 conclusion | Problem, system, three modules, experiments, real robot, conclusion. |
| 0:09 (+10 s) | 1 · The problem | `problem_intro` | Problem statement on the paper's Fig. 1: a cup seen in an upstairs bedroom is moved to the downstairs kitchen | teaser.png, the two floors and the route (illustration) | Object navigation assumes objects stay put. In real homes, they move. A cup seen upstairs turns up downstairs: wrong place, and another floor. |
| 0:19 (+10 s) | 1 · The problem | `problem_clip` | Problem, shown: the agent walks to the remembered table, the verifier rejects, the map was wrong | first-person RGB + detections, costmap inset, verifier image inset | The map says the cracker box is here. Someone moved it. The verifier finds the table bare. The map is wrong, not just incomplete. |
| 0:29 (+9 s) | 1 · The problem | `problem_still` | The three-step protocol, on the real floor plates | layout_conditions panel (d): cracker box, table → kitchen island, 2.71 m of it vertical | Our benchmark: map, move the objects, then search from the old map. Moved across floors, the agent must leave the storey the map points at. |
| 0:38 (+10 s) | 2 · Our system | `system` | Our system: one persistent scene graph + three mechanisms | full_pipeline.png + three numbered module boxes | One persistent scene graph of floors, rooms, containers and objects. Three mechanisms: the graph, presence revision, and belief-guided search. |
| 0:48 (+9 s) | 3 · Module 1/3 · Scene graph | `perception_clip` | Perception to scene graph, on footage | detections + the scene-graph tree sliding in | Module one: detections become ellipsoid tracks under floor, room and container. A stair edge joins two storeys only after a real climb. |
| 0:57 (+12 s) | 3 · Module 2/3 · Presence belief | `presence_clip` | Presence-belief revision, on footage | belief bar dropping as the agent inspects an empty surface | The map holds a cracker box here at belief 0.77. Up close the surface is in clear view and empty: every miss counts. Belief falls to 0.44; the search moves on. |
| 1:09 (+5 s) | 3 · Module 2/3 · Presence belief | `eq3_card` | Eq. 3, the three-row log-odds table | positive / eligible miss / ineligible miss | A detection raises belief, a visible miss lowers it, occlusion changes nothing. |
| 1:14 (+10 s) | 3 · Module 3/3 · Belief-guided search | `floor_clip` | Belief-guided search and the floor decision, on footage | floor-mass readout from search_log_events step 13 | Every surface carries a search posterior; a floor's score is their mean. The floor below now outscores the mapped floor: take the stairs. |
| 1:24 (+5 s) | 3 · Module 3/3 · Belief-guided search | `floor_still` | Floor score, redrawn: the two floors' scores as inspections fail (real numbers, 00873 / 'tin can') | curve chart: cross after 8 looks, switch accepted after 10 | Failed looks spend belief; after ten, the other floor wins the switch. |
| 1:29 (+19 s) | 4 · Experiments | `hero` | Hero: ours vs ASCENT on the identical episode | side by side, shared step clock, ours freezes on the detection | Same episode, same prior map. The bottle was mapped upstairs; it is now downstairs. MD-SG takes the stairs it already mapped and finds the bottle in 207 steps. ASCENT, with the same map, wanders to the step limit and never does. |
| 1:48 (+5 s) | 4 · Experiments | `montage_1` | Montage 1/4: two flights down |  | Two more episodes: this cracker box was moved two floors down. |
| 1:53 (+4 s) | 4 · Experiments | `montage_2` | Montage 2/4: one flight up |  | This bottle was moved one floor up. Both found. |
| 1:57 (+5 s) | 4 · Experiments | `numbers_1` | Table II | ASCENT 0 % / 0.000 vs ours 40.0 % / 0.162 (n = 25) | 25 cross-floor episodes: ASCENT 0 %, MD-SG 40 %. |
| 2:02 (+5 s) | 4 · Experiments | `numbers_2` | Table I cross-anchor + Table III trend | DualMap† 30.2 % vs ours 52.8 %; ablation 41.5 / 35.8 / 52.8 | Against DualMap, on one floor: 30 % → 53 %. |
| 2:07 (+3 s) | 5 · Real robot | `setup_wide` | Real robot, the setup (wide) | setup_2.jpg | Now, a real Stretch 3, in a two-storey building. |
| 2:10 (+5 s) | 5 · Real robot | `setup_close` | Real robot, who does what | setup_1.jpg with three callouts | The Thor runs perception and the graph; the Stretch keeps its own navigation. |
| 2:15 (+10 s) | 5 · Real robot | `robot_step0` | Pass 1 on the real robot, one take at 6x: maps floor 0, finds the ball | phone | RViz, robot camera inset, speed badge | Pass one: the robot explores the ground floor, and every detection goes into the graph. There is the ball, on the sofa. |
| 2:25 (+17 s) | 5 · Real robot | `robot_step1` | Pass 2, one take at 8x: old spot empty, floor 0 exhausted, WANTS FLOOR 1, waits at the lift | phone | RViz | The ball goes upstairs. From the stale map, the robot drives straight to the sofa: gone. It searches the rest of the floor, then asks for floor one. It cannot climb, so it waits at the lift. The operator rides up with it and declares the floor. |
| 2:42 (+10 s) | 5 · Real robot | `robot_step2` | Floor 1, one take at 6x: fresh grid, the corridor, the lab, the ball | phone | RViz | Floor one: a fresh grid, while the graph still remembers floor zero. And there is the ball, in the lab. |
| 2:52 (+8 s) | 6 · Conclusion | `close` | Closing card over the robot at the ball | takeaway + paper / code line | To conclude: a map should hold beliefs, not facts; a search should know when a floor is spent; and both run on a real robot today. |

Total: 180 s of 180 s.

Footage tally: ≈ 66 s simulator, 40 s real robot (three continuous takes, sped up, badge on screen), ≈ 74 s cards and figure inserts (title, outline, the Fig. 1 problem statement, benchmark plate, system, Eq. 3, floor score, two result cards, conclusion). 180 s exactly.

## Narration — the transcript

Synthesised with Kokoro-82M (voice `af_heart`, speed 1.1) by `scripts/video/make_narration.py`
from `scripts/video/narration.json`: 36 cues, 428 words, 135 s of speech in the 180 s cut (Kokoro's ~0.3 s of padding per cue is trimmed so voice and subtitle start together).
Kokoro speaks these lines at ~2.5 words/s, so every cue is written to the seconds its shot
leaves free; the script refuses to build a track in which one cue runs into the next, or one
whose subtitle needs a third line. A cue may carry a `display` string when the subtitle
should read differently from what is said (digits on screen, words in the voice). Outputs:
`outputs/video/narration/voice.wav`, `cues.json` (timings measured from the audio — what the
renderer draws the subtitles from), `transcript.md`, and `outputs/video/icra2027.srt` for a
player that wants soft subtitles.

Kokoro needs Python ≥ 3.10 with pip, which neither conda env here has:
`conda create -n tts python=3.11` (conda's `envs_dirs` puts it at `/tmp/.conda/envs/tts`),
then `pip install kokoro soundfile pillow`. The model `hexgrad/Kokoro-82M` is cached under
`~/.cache/huggingface/hub/`. To use a human voice instead, read the table against the cut and
hand the WAV to the same `--narration` flag — the subtitles will still follow `cues.json`, so
keep the cue starts or re-time them.

| time | shot | narration |
|---|---|---|
| 0:00.3–0:04.0 | `title` | MD-SG: a scene graph that lets a map be wrong, and recover. |
| 0:04.3–0:08.3 | `outline` | Problem, system, three modules, experiments, real robot, conclusion. |
| 0:09.3–0:13.4 | `problem_intro` | Object navigation assumes objects stay put. In real homes, they move. |
| 0:13.8–0:17.9 | `problem_intro` | A cup seen upstairs turns up downstairs: wrong place, and another floor. |
| 0:19.3–0:21.9 | `problem_clip` | The map says the cracker box is here. Someone moved it. |
| 0:24.0–0:28.1 | `problem_clip` | The verifier finds the table bare. The map is wrong, not just incomplete. |
| 0:29.2–0:32.6 | `problem_still` | Our benchmark: map, move the objects, then search from the old map. |
| 0:33.2–0:36.8 | `problem_still` | Moved across floors, the agent must leave the storey the map points at. |
| 0:38.3–0:42.4 | `system` | One persistent scene graph of floors, rooms, containers and objects. |
| 0:43.4–0:47.3 | `system` | Three mechanisms: the graph, presence revision, and belief-guided search. |
| 0:48.4–0:52.9 | `perception_clip` | Module one: detections become ellipsoid tracks under floor, room and container. |
| 0:54.0–0:56.9 | `perception_clip` | A stair edge joins two storeys only after a real climb. |
| 0:57.3–1:00.3 | `presence_clip` | The map holds a cracker box here at belief 0.77. |
| 1:01.4–1:04.9 | `presence_clip` | Up close the surface is in clear view and empty: every miss counts. |
| 1:05.8–1:08.6 | `presence_clip` | Belief falls to 0.44; the search moves on. |
| 1:09.1–1:13.6 | `eq3_card` | A detection raises belief, a visible miss lowers it, occlusion changes nothing. |
| 1:14.5–1:18.2 | `floor_clip` | Every surface carries a search posterior; a floor's score is their mean. |
| 1:19.4–1:22.5 | `floor_clip` | The floor below now outscores the mapped floor: take the stairs. |
| 1:24.2–1:28.0 | `floor_still` | Failed looks spend belief; after ten, the other floor wins the switch. |
| 1:29.4–1:34.3 | `hero` | Same episode, same prior map. The bottle was mapped upstairs; it is now downstairs. |
| 1:35.8–1:41.5 | `hero` | MD-SG takes the stairs it already mapped and finds the bottle in 207 steps. |
| 1:43.5–1:47.0 | `hero` | ASCENT, with the same map, wanders to the step limit and never does. |
| 1:48.3–1:51.8 | `montage_1` | Two more episodes: this cracker box was moved two floors down. |
| 1:53.3–1:55.8 | `montage_2` | This bottle was moved one floor up. Both found. |
| 1:57.3–2:01.8 | `numbers_1` | 25 cross-floor episodes: ASCENT 0 %, MD-SG 40 %. |
| 2:02.2–2:05.6 | `numbers_2` | Against DualMap, on one floor: 30 % → 53 %. |
| 2:07.2–2:09.9 | `setup_wide` | Now, a real Stretch 3, in a two-storey building. |
| 2:10.2–2:14.2 | `setup_close` | The Thor runs perception and the graph; the Stretch keeps its own navigation. |
| 2:15.4–2:20.4 | `robot_step0` | Pass one: the robot explores the ground floor, and every detection goes into the graph. |
| 2:22.6–2:24.0 | `robot_step0` | There is the ball, on the sofa. |
| 2:25.3–2:30.2 | `robot_step1` | The ball goes upstairs. From the stale map, the robot drives straight to the sofa: gone. |
| 2:31.3–2:34.1 | `robot_step1` | It searches the rest of the floor, then asks for floor one. |
| 2:35.6–2:40.6 | `robot_step1` | It cannot climb, so it waits at the lift. The operator rides up with it and declares the floor. |
| 2:42.3–2:45.7 | `robot_step2` | Floor one: a fresh grid, while the graph still remembers floor zero. |
| 2:50.4–2:51.9 | `robot_step2` | And there is the ball, in the lab. |
| 2:52.3–2:59.5 | `close` | To conclude: a map should hold beliefs, not facts; a search should know when a floor is spent; and both run on a real robot today. |

## The real-robot experiment, and how it is cut

Recorded 2026-09-23 on the Stretch 3 following `docs/DEMO_FLOW.md`; target `sports ball`.
Two cameras ran at once: a phone on a tripod (`human_camera/stepN_phone.mp4`, 1280×720
HEVC) and a screen capture of the Thor (`rviz/stepN.mp4`, 1920×1080) showing the two operator
windows of `docs/THOR.md` — the `/osg/detections` camera panel and the `osg_map.rviz` map with
the scene graph (objects coloured by presence belief), the costmap, Nav2's plan and the goal.

| step | what happened | phone | rviz |
|---|---|---|---|
| 0 | pass 1 (`stretch3_map`): explores floor 0, finds the ball on the sofa | 102 s | 103 s |
| 1 | pass 2 (`stretch3_search`) from the stale map: drives to the sofa, absent, searches floor 0, prints `>>> WANTS FLOOR 1`, drives to the lift and waits; the operator rides up with it, loads the floor-1 map, publishes `/osg/floor 1` | 160 s | 258 s |
| 2 | floor 1: fresh grid, explores the corridor and the lab, finds the ball on a desk | 121 s | 135 s |

Each step is **one continuous take**, not a cut: the `dual` shot names a window of the phone
take (`phone_t`) and the RViz capture time it lines up with (`rviz_t0`), and both are played
at the same `speed` — shown on screen as a `▶ 6× speed` badge — so the two views stay in step
and nothing is elided. The RViz layout key (`A`/`B`/`full`, or a list of `[t_rel, key]` when
the capture changes window mid-take) tells the renderer where the windows sit; the camera
panel is cropped out as the portrait "robot camera" inset over the phone view. The pairing was
set by eye from contact sheets — the phone and the screen recorder were not clock-synced, so
treat the alignment as ±2 s.

| shot | phone | rviz | speed | what to look for |
|---|---|---|---|---|
| `robot_step0` | step0 30–96 s | step0 from 30 s | 6× | exploring, then `sports ball 0.93` on the sofa in the robot-camera inset |
| `robot_step1` | step1 4–150 s | step1 from 12 s | 8× | the sofa, `absent on arrival`; the corridor; the status line `>>> WANTS FLOOR 1: carry the robot, then ros2 topic pub …`; parked at the lift while the operator's RViz loads the floor-1 map |
| `robot_step2` | step2 42–114 s | step2 from 42 s | 6× | floor 1's fresh costmap with the graph still carrying floor 0; the corridor, the lab, the ball on the desk, `DONE` |

Nothing is scored on the robot: `Ros2Env.metrics()` returns NaN by design, and the narration
says "demonstration", never a rate. The storey is declared by the operator (`/osg/floor`),
exactly as `docs/ROS2.md` says it must be; the video shows that and does not pretend the
robot took the stairs.

## Where the footage comes from

Every simulator clip is a pass-2 `cross_anchor_01` episode from `outputs/RERUN/mf5` (the
2026-09-18 reproduction of Table II, 11/25 = 44.0 %, the run
`docs/CROSS_ANCHOR_REPRODUCTION.md` says reproduces) or its ASCENT twin from
`outputs/ascent_crossanchor` (0/25). The debug videos are 8 fps diagnostics written by
`src/osg/eval/debug_video.py`: the left panel is live RGB with YOLOE masks and boxes (target
class in red), the right panel the costmap. `make_assets.py` centre-crops the RGB to 16:9,
crops the costmap to its non-grey bounding box and places it as an inset, and burns in the
step, floor and FSM state from that episode's `episodes.jsonl`. Speed-ups are frame skips;
no frame is synthesised.

| shot | episode | steps | why this one |
|---|---|---|---|
| `problem_clip` | `00873-bxsVRursffK …50001` cracker box | 1–54 | commits to the prior-map position at step 3 (p 0.94), the VLM verifier rejects the arrival at step 40 (`verify_debug/…call001_crackerbox_REJ.jpg`, answer "bare"), the search resumes at 54 |
| `presence_clip` | same | 156–196 | approach a mapped track (p 0.77) → close look at 171 → `absent_on_arrival` at 190, p 0.44 |
| `perception_clip` | `00821-eF36g7L6Z9M …50001` cracker box | 226–282 | the richest detection coverage of the set, ending on the recovery (282 steps, SPL 0.69) |
| `floor_clip` | `00878-XB4GS9ShBRE …50006` yellow bottle | 40–112 | the descent; the floor mass at step 13 was 4.55 (below) vs 0.38 (mapped floor) |
| `hero` | same, both policies | 1–207 / 1–489 | the identical episode: ours succeeds (SPL 0.84, floor 3 → 2 at step 158), ASCENT fails 2.4 m away |
| `montage_1` | `00821 …50001` | 190–282 | floor 1 → 0 at 223 |
| `montage_2` | `00862-LT9Jq6dN3Ea …50006` yellow bottle | 160–292 | the one *upward* recovery, floor 1 → 2 at 188 |
| `montage_3` | `00878 …50001` cracker box | 1–132 | the shortest recovery, climb at 15–84 |
| `montage_4` | `00800-TEEsavR23oF …50001` cracker box | 690–789 | the third climb attempt succeeds |

Figures: `docs/figures/teaser.png` (the paper's Fig. 1, left 70 %, as the problem statement — its route and cup positions are the caption's "explanatory overlays", and the narration treats it as an illustration), `docs/figures/layout_conditions.png` (panel d), `docs/figures/full_pipeline.png`
and `outputs/figures/scene_graph_multifloor_00873_hi.png` (left half) — regenerable per
`docs/FIGURES.md`; `docs/figures/` is gitignored. The floor-score chart (`floor_still`) is
**redrawn for the screen** by `make_assets.py` (`card_floor_score`) from the same computation
as the paper figure: it imports `scripts/render_floor_score_formula.py` and runs its `sweep`
on the saved 00873 map, caching the curve in `outputs/video/assets/floor_score_curve.json`
(57 / 24 surfaces, M = 0.104 / 0.049, margin cleared at 10 — the numbers the paper caption
prints); on screen it is reduced to the two curves and the switch point, no legend, no
formula. The scene-graph inset at 0:36 is **drawn** (`draw_scene_graph`), a five-column
tree with the node labels of the paper's 00873 figure, because the figure's own tree was
too thin to read at inset size; it is an illustration of the hierarchy, not a rendering of
that episode's graph. On every one of our simulator clips the costmap inset **follows the
agent** (`map_zoom: 200` panel px around the red agent dot, clamped to the explored cells,
scaled into a 320-px inset) instead of showing the whole tall grid, which was unreadable.

**Subtitles and the section chip.** Every frame is finished by `Renderer.finish_frame`: it
draws the section chip (top-right, from the shot's `section` field) and the subtitle for the
frame's absolute time (`subtitle_at`, from `cues.json`: a cue shows from its audio start until
0.4 s after its audio ends, wrapped to at most two 26-px lines of 1100 px). The old per-clip
caption bands are gone — the narration says what they said — but the short event pills
("absent on arrival", ">>> WANTS FLOOR 1", "ball found") stay, because they mark a moment,
not a sentence. The `outline` card lists the six parts; the `system` card puts the pipeline
figure over three numbered module boxes that the chip then counts through (Module 1/3 … 3/3);
the `close` card carries three takeaways. Setup photos: `assets/real_experiment/human_camera/setup_2.jpg`
(the room) and `setup_1.jpg` (the Thor, the robot, the cable, with the three callouts).

## What the footage does not support — say nothing beyond this

- **`00873 …50001` is not a recovery.** It is scored as a success (SPL 0.795) by the 2-D
  distance rule while the box sits 2 m below the agent, which never left the upper storey.
  The cut uses its first 54 steps and its steps 156–196 for the two belief beats and never
  shows its ending.
- **Table I's dynamic rows and all of Table III do not reproduce on a rerun**
  (`docs/CROSS_ANCHOR_REPRODUCTION.md`: in-anchor 55.6 %, cross-anchor 37.7 % on 2026-09-20,
  traced to the prior map, not the code). `numbers_2` shows the paper's numbers exactly as
  printed, for six seconds, and claims nothing more.
- **The multi-floor rerun gives 44.0 %; the card says 40.0 %**, the paper's number.
- The floor-mass readout in `floor_clip` is from step 13 of that episode; the burn-in says so.
  The belief values in `problem_clip` and `presence_clip` are the logged `goal_commit` and
  `presence_events` values; where no value was logged (after the verifier's rejection) the bar
  is hidden rather than invented.
- ASCENT's panels are ASCENT's own obstacle and value maps; the composite labels them.
- `floor_still` is computed by the real code, on 00873 with the query "tin can" — a different
  scene from the clip it follows; the chart is redrawn, the numbers are not.
- The paper's teaser overlays are schematic; the cut shows it only as the problem statement, with no episode claim.
- Nothing in the robot section is measured, and the phone/RViz pairing is by eye (±2 s); the
  floor switch is the operator's, not the robot's.

## Encoding

1280×720, 24 fps, H.264 high profile, two-pass at the bitrate that lands the whole file near
18.5 MB (≈ 780 kbps video for 175 s), mono AAC at 64 kbps — the voice track costs ≈ 1.1 MB. `build_video.py` refuses to report
success if the result exceeds 20 MB or 180 s.

## Versions

Saved cuts live beside the working file so a change can be compared against what it replaced:

| file | date | what it is |
|---|---|---|
| `outputs/video/icra2027_rough_v1_2026-09-23.mp4` | 2026-09-23 | first cut with the real-robot section (three sped-up takes) |
| `outputs/video/icra2027_rough_v2_silent_2026-09-23.mp4` | 2026-09-23 | v2, approved picture: floor-score chart redrawn (two curves + the switch point), scene-graph inset drawn, every "ours" costmap inset zoomed to follow the agent; no audio |
| `outputs/video/icra2027_rough_v3_2026-09-23.mp4` | 2026-09-23 | v3: v2 picture + Kokoro voiceover + the text-fit pass |
| `outputs/video/icra2027_rough.mp4` | — | the working cut, v4: six-part outline (outline card, section chip), a problem statement on the paper's Fig. 1 drawn from the introduction, system card, three-line conclusion, subtitles on every line drawn from the narration's timings, narration rewritten to say what each shot shows and why |
