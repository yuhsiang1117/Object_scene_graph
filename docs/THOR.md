# Deploying on a Jetson AGX Thor, beside a Stretch 3

`docs/ROS2.md` describes the layer; this describes where it runs. Three
participants, and the split between them is the whole design:

```
   ┌─ Jetson AGX Thor ──────────────────────────────┐        ┌─ Stretch 3 ────────┐
   │                                                │        │                    │
   │  osg-thor     the pipeline, the scene graph,   │        │  Nav2              │
   │  (L4T, CUDA)  the five perception models,      │        │  stretch_driver    │
   │               ollama. No ROS. No Habitat.      │        │  RealSense driver  │
   │                    ▲                           │        │  TF                │
   │                    │ socket 127.0.0.1:18765    │        │                    │
   │                    ▼                           │  DDS   │                    │
   │  osg-bridge   rclpy, and nothing else.         │◀──────▶│                    │
   │  (ros:humble) numpy + stdlib + ROS messages.   │        │                    │
   └────────────────────────────────────────────────┘        └────────────────────┘
```

**The robot keeps its own low-level control.** This stack never starts a navigator, a
controller or a driver. It publishes a goal pose to the Stretch's existing `NavigateToPose`
and reads back the camera and TF — see *Who owns the base* in `docs/ROS2.md` for the one tick
where that hands over.

## Why two containers on the Thor

Because the bridge genuinely needs nothing. With ROS stubbed out, importing
`osg.ros2.bridge_node` pulls numpy and the standard library and stops — no torch, no OpenCV,
no scipy. `src/osg/ros2/` was written that way because it has to import under two
interpreters, and on this platform that pays three times over:

- **The bridge matches the robot.** A Stretch 3 runs ROS 2 Humble on Ubuntu 22.04.
  `ros:humble-ros-base` *is* Ubuntu 22.04 with Humble's own `rclpy`, so both ends of every
  topic and action are the same distro. No cross-distro DDS, and no Humble compiled from
  source against a Python it was never tested against — which is what a single image on
  JetPack's Ubuntu would have forced.
- **The bridge needs no CUDA.** Nothing in it links against the host's L4T libraries, so a
  jammy userspace on JetPack's Ubuntu is not a compatibility question at all.
- **The Thor image needs no ROS**, and so is free to be whatever JetPack wants: system
  Python, NVIDIA's torch wheels, nothing bent around `rclpy`'s Python version.

They reach each other over the loopback because both take `network_mode: host` — the bridge
needs the host's network for DDS discovery (multicast, which the default bridge network
cannot carry) and the Thor container needs the same loopback to reach the bridge's socket.

## What is NOT in the Thor image

**Habitat.** There is no aarch64 build of habitat-sim — the `aihabitat` conda channel is
linux-64 only — and the robot does not need one. This is verified rather than assumed: with
`habitat`, `habitat_sim` and `magnum` blocked at import, the whole robot path (`Ros2Env` →
`NavAgent` → `run_episode` → `save_map`) runs to completion. `eval.mode=ros2` is the only
mode this image can run; every benchmark mode stays on the x86 image, which is where the
measured numbers come from anyway.

## Build and run

Needs the submodule, same as x86 — the perception servers are built from it:

```bash
git submodule update --init --recursive
cp docker/.env.example docker/.env      # set UID/GID, and the Jetson block

docker compose -f docker/compose.thor.yaml --env-file docker/.env build
docker compose -f docker/compose.thor.yaml --env-file docker/.env up -d
docker exec -it osg-thor bash
```

### Stage the weights, once

Nothing is baked in, and `data/weights` is the named volume `weights-cache`, which *shadows* the
host directory — so the YOLOE half has to be staged from **inside** the container:

```bash
docker exec -it osg-thor bash
python scripts/download_weights.py --profile large        # yoloe-11l-seg.pt + mobileclip_blt.ts
mkdir -p outputs/dualmap_comparison/vendor/DualMap/model && \
  (cd outputs/dualmap_comparison/vendor/DualMap/model && python -c "from ultralytics import FastSAM; FastSAM('FastSAM-s.pt')")
bash scripts/fetch_ascent_weights.sh                     # RAM++ (2.9 GB), GroundingDINO, D-FINE
```

and the LLM, from the host: `docker exec osg-ollama ollama pull qwen2.5:7b` (the ranker's model)
plus `qwen2.5vl:3b` (unused by the run; `smoke_ollama.py` checks both roles). BLIP-2 has no file
— lavis fetches it on the server's first start, into `HF_HOME` inside that same volume.

The FastSAM line exists because the Stretch presets inherit `region_proposal.enabled: true`,
which loads FastSAM plus a MobileCLIP-S2 encoder through `open_clip` — both packages are in this
image for that reason, and the MobileCLIP weights resolve from `HF_HOME` on first use.

### The two runs

Then, inside `osg-thor`, from `/workspace` (both passes must share a cwd: `ros2.map_dir` is
relative):

```bash
bash scripts/serve_perception.sh              # the five model servers
for p in 13182/blip2itm 13185/ram; do curl -s -o /dev/null -w "$p %{http_code}\n" http://localhost:$p; done
python scripts/smoke_ollama.py
python scripts/run_robot.py +experiment=stretch3_map    ros2.map_tag=lab ros2.target=cup
# ... move the cup ...
python scripts/run_robot.py +experiment=stretch3_search ros2.map_tag=lab ros2.target=cup
```

**Check the two endpoints by hand.** The pre-flight refusal CLAUDE.md describes
(`probe_served_models`) only fires when `detector.strict` is true, and `detector=yoloe` leaves it
false — so on these presets it is a no-op, and a dead BLIP-2 (`value_strict: true`) surfaces as
`PerceptionUnavailable` **after the robot is moving**. Only `:13182` and `:13185` are on this path;
the other three windows may die for want of weights, harmlessly.

**Set `ros2.target` on the mapping pass too.** `stretch3_map.yaml` does not set it, so left alone
the mapping pass hunts the default `chair`. Under the moved-object protocol the two passes want the
same target: pass 1 finds it at A and the snapshot carries that belief, pass 2 starts from it.

`serve_perception.sh` needs no changes here: it already selects its interpreter through
`ASCENT_PYTHON`, which this image points at the models venv.

### Measured on this robot (2026-09-21)

What the Stretch actually publishes, read through the bridge with the robot standing still.
Every one of these is either now recorded in config or confirms a value that already was:

| what | measured | where it lives |
|---|---|---|
| RGB / aligned depth | **1280x720**, `rgb8` / `16UC1` (mm), `camera_color_optical_frame`, ~10 Hz each | `ros2.depth_scale 0.001` confirmed; `eval.rgb_width/height` are *not* read on the robot path (only the simulators and `debug_video` use them), intrinsics come from `CameraInfo` |
| intrinsics | fx = fy = 911.2 at 1280x720 -> HFOV 70.1° landscape, **43.1° once rotated** | `eval.hfov_deg 42.5` (the frontier FOV wedge, `nav_agent.py`) — close enough |
| camera height | `map -> camera_color_optical_frame` z = **1.297 m** | `agent.camera_height 1.3` confirmed |
| **rotation** | the published frame is the room on its side; **`+90` (the default) is upside down, `-90` is upright** | `configs/ros2/stretch3.yaml` now sets `rotate_deg: -90.0`; the three frames are `outputs/ros2_check/real_frame_rot*.png` |
| Nav2 | `/navigate_to_pose` served by `/bt_navigator`; `/map` from `/rtabmap` on the first setup, now **slam_toolbox** (below) — `run_robot.py` prints who publishes it at preflight | `ros2.nav_action`, `goal_frame map` confirmed |
| head tilt | `/stretch_controller/follow_joint_trajectory` served by `/stretch_driver` | `ros2.head_traj_action` confirmed |
| clocks | robot vs Thor skew **1 s** | fine; a first-frame `extrapolation into the past` TF warning at bridge start-up is just the buffer filling |
| detection floors | YOLOE scored the ball **0.28 / 0.22** on the first run's scan keyframes (motion-blurred, base still decelerating) — under the simulator-tuned `detector.conf 0.30` and `verification.min_score 0.35`, so the only `ball` track was the region proposer's and the gate never let it through | `stretch3_map.yaml` now sets `detector.conf 0.2`, `verification.min_score 0.25`; the RViz target label shows the gate numbers live (`s= box= n= ev=`, `PROPOSAL-ONLY n_prop= sim=`) |

Recording `rotate_deg` changed a pinned default, so both goldens were regenerated
(`python tests/unit/test_config_snapshot.py`). All 82 experiment fingerprints moved, not just
the two Stretch ones: `configs/config.yaml` puts the `ros2` group in *every* composition, so its
value is part of every hash, though no simulator arm reads it.

### Measuring latency

The review asked for real-world latency, and no Thor number exists yet: the simulator's
(`outputs/mfv2/osg_s42`, RTX 4090 shared with two other evals and the model servers) is a
*contended* 1.74 Hz, median step 463 ms, region proposals 33% and the object layer 25% of it.
Every robot run writes the same profiler output as a simulator run, so the measurement is a
normal search run on a quiet device and one script:

```bash
# nothing else on the Thor's GPU (no simulator, no second pipeline)
python scripts/run_robot.py +experiment=stretch3_search ros2.map_tag=lab ros2.target=cup
python scripts/analyze_latency.py outputs/<that run>          # the table
# attribution run: charge queued CUDA work to the stage that queued it (slower; do not
# quote its rate, quote its shares)
python scripts/run_robot.py +experiment=stretch3_search ros2.map_tag=lab ros2.target=cup \
    eval.profile_sync=true
```

`analyze_latency.py` prints per stage the calls per control step, mean / p50 / p90 / max ms
and the share of control-loop time, then the control rate (mean, p50, p90), then the
**synchronous** model calls separately. Report those as they are: the arrival-time absence
reading asks the VLM inside the step (27 s mean on NIM in simulation, 2 calls in 12
episodes), which is a stall, not a rate. The frontier ranker and floor LLM are asynchronous
and never block a step; their latency is the reply time, in `episodes.jsonl` (`llm_calls`).

### Watching it: two windows

The bridge is the only ROS process on the Thor, so it is the only place a viewer can run. Both
windows come from a second build target of the same image (`Dockerfile.bridge`, target `rviz` —
the bridge proper stays small) and one on-demand compose service, started with `run` rather
than `up` because it is a window:

```bash
docker compose -f docker/compose.thor.yaml --env-file docker/.env run --rm rviz               # both windows
docker compose -f docker/compose.thor.yaml --env-file docker/.env run --rm rviz --view map    # one of them
docker compose -f docker/compose.thor.yaml --env-file docker/.env run --rm rviz --view camera
```

They open on the Thor's own screen, side by side, and each shows one thing:

| window | what | how |
|---|---|---|
| **camera** | the rotated frame the pipeline consumes, with the detector's boxes, masks and labels on it, the target in red — the simulator's debug video, live | `scripts/ros2/image_window.py` on `/osg/detections`, a 560×1000 window scaled to the portrait frame (set `OSG_CAMERA_TOPIC=/osg/rgb` for the plain frame; `--view camera-rqt` for `rqt_image_view`, which opens as a small strip) |
| **map** | the map the robot is building (`/map`, latched), the pipeline's own costmap for the storey it is on (`/osg/costmap`, the grid the frontiers are chosen from, drawn over it in costmap colours), its scan and pose, Nav2's plan, the goal the bridge last handed to Nav2 (`/osg/goal`, red arrow), and the scene graph as markers: objects coloured by presence belief, the target's gate numbers as its label, the chosen frontier, and the status line with the storey the agent WANTS | `rviz2` with `scripts/ros2/osg_map.rviz`, top-down |

Closing the map window ends the container and the camera window with it. Verified on the Thor's screen with a recorded frame and a test map fed through the bridge (2026-09-22). The old
everything-in-one view (`scripts/ros2/osg.rviz`: costmaps, plans, three image panels, raw
camera streams) is still there as `--view full`.

Everything on `/osg/*` is the pipeline's rather than the robot's, relayed by the bridge because
the pipeline has no rclpy:

| topic | what |
|---|---|
| `/osg/goal` | the pose the bridge last handed to Nav2. An action goal is not a topic, so the bridge republishes it (transient-local: a window opened mid-run shows the current goal at once). Watch it against `/plan` to see what Nav2 made of the request |
| `/osg/rgb` | the **rotated** frame the pipeline actually consumes — the raw camera topics are the room on its side |
| `/osg/detections` | that frame with the detector's boxes, masks and labels — drawn by the same `overlay_segmentation` the simulator's debug video uses |
| `/osg/debug` | the simulator video's two-panel picture, [detections \| top-down costmap with trajectory, path and chosen frontier] — only in the `full` view now |
| `/osg/scene_graph` | the scene graph as markers, in `map`; storeys stack in z by `floor.virtual_storey_m` |
| `/osg/costmap` | the pipeline's costmap for the current storey as a latched `OccupancyGrid` (unknown / free / obstacle), lifted by the same storey offset |

The images come from `src/osg/eval/debug_stream.py`, which `run_robot.py` always attaches: it
is `DebugVideo` with the bridge in place of an mp4, re-running the detector for visualisation
only, exactly as the simulator does, so the object layer and the run are unchanged. It is best
effort by construction — a bridge that cannot take the images is logged once and the run goes
on. The bridge confines it to `/osg/*`.

The same stream also appends one JSON line per step to **`outputs/<run>/stream.jsonl`**, flushed
as it is written: step, state, storey, the wanted storey, agent and goal positions, the target
tracks' gate numbers exactly as the RViz label shows them, and — whenever they gain an entry —
the agent's own diagnostic logs from the episode record (`state_log`, `frontier_select_log`,
`candidate_reject_log`, `goal_commit_log`, `giveup_log`, the approach logs, `presence_events`,
`floor_log`, and `approach_diag` when it changes). So the question the record answers afterwards
("why that frontier, why not the ball") is answerable during the run: `tail -f` it, or
`grep candidate_reject stream.jsonl`. Nothing about a run waits for it to end any more:

- stdout is line-buffered, so `... | tee run.log` shows lines as they happen;
- the mapping pass rewrites its map every `ros2.map_checkpoint_steps` steps (20) at the path the
  search pass reads, so a run that dies at step 150 leaves the step-140 map;
- `SIGTERM`/`SIGHUP` (`kill`, `docker stop`, a closed terminal) are handled like Ctrl‑C: the Nav2
  goal is cancelled, the final map is written, and the run ends cleanly.

Only `episodes.jsonl` still needs the run to reach its STOP — an interrupted run has no outcome
to score — and `stream.jsonl` holds everything it would have said about the path there.

Three things the setup relies on, all overridable in `docker/.env`: the Thor's GNOME session is
X11 on **`:1`** (`ls /tmp/.X11-unix`), it belongs to uid 1000 — the uid the container runs as —
so its GDM cookie (`/run/user/1000/gdm/Xauthority`) authorises the window without `xhost`, and
rendering is software GL (llvmpipe), which is plenty for a debug view (19 fps with the full
view). If the window never appears, `DISPLAY` or the cookie path is the first thing to check;
the launcher says which.

Four ways this has actually failed, each looking like "RViz shows nothing":

- **`DISPLAY` pointing at the machine you ssh'd in from.** `:0` is your laptop; the Thor has no
  `:0`. The container exits at once and `run --rm` deletes it, so nothing is left to read. The
  launcher now names this. Keep `DISPLAY=:1` in `docker/.env`.
- **The Thor's screen locked itself** — GNOME locks after 5 idle minutes, and a lock screen is
  "nothing" on the monitor and over RDP alike. For a robot Thor, turn it off:
  `gsettings set org.gnome.desktop.session idle-delay 0` and
  `gsettings set org.gnome.desktop.screensaver lock-enabled false`.
- **The config's window layout.** A hand-written `QMainWindow State` docked every image panel
  as a collapsed strip with the 3D view crushed into a corner; with *no* geometry block rviz2
  sat on its splash screen indefinitely. `osg.rviz` now carries size and position only.
- **rviz2 rejecting the file** (`Could not load display config: Invalid argument`, and an empty
  view). Bisected: a fuller `Panels`/`Tools`/`Views` block did it, as did naming two image
  displays `RGB (as published)` and `Depth (aligned)` together. The file is built on the
  minimal skeleton that a Grid-only test proved clean; extend it from a saved session
  (File > Save Config), not by hand.

The robot's own RViz, X-forwarded onto the Thor's screen, is on the same DDS domain and can
show the `/osg/*` topics too — Add → By topic — which is a fine fallback.

On x86, where ROS sits beside the pipeline in one container, `bash scripts/ros2/bridge.sh --rviz`
opens the same view next to the bridge (needs `ros-humble-rviz2` in that image).

### Mapping with slam_toolbox

The robot localises itself; the pipeline only reads `map -> camera` from TF and hands Nav2 goals
in `map`. So *which* localiser runs is the robot's business — but it decides whether a run can
work at all. Two runs on 2026-09-22 were driven on AMCL against a map the robot had been given,
and AMCL had not converged: the pose in `map` jumped **6 m in one 2.5 s step** (step 47 of
`outputs/20260921_212343/stream.jsonl`), so every object placed before it — the ball twice —
was somewhere else, one of them outside the far wall, and every frontier goal of the run was
cancelled unreached. `run_robot.py` now prints `POSE JUMP` when that happens, but the cure is
upstream: **map as you go with slam_toolbox**, which starts from wherever the robot stands and
never has a wrong prior to converge from.

**Who runs it: the robot, not the Thor.** slam_toolbox is a node in the robot's own Nav2 bring-up; the Thor subscribes to its `/map` and reads `map -> camera` from TF, and nothing on the Thor starts, configures or stops it. Whether it is running is visible in three places: `run_robot.py`'s preflight line `/map published by: slam_toolbox`, the map window's `Map (slam_toolbox)` display, and on the robot `ros2 node list | grep slam_toolbox`.

The robot's own stack already has it. `stretch_nav2`'s `navigation.launch.py` takes
`use_slam:=True`, which swaps AMCL + `map_server` for slam_toolbox inside the same Nav2
bring-up. One thing to add: with the stock `nav2_params.yaml` that starts slam_toolbox on its
*default* parameters, i.e. the unfiltered `/scan`, while the Stretch's lidar launch also runs a
laser filter that removes the robot's own mast and publishes `/scan_filtered` — which is what
the costmaps read. `docker/stretch/nav2_params_slam.yaml` is `stretch_nav2`'s Nav2 params plus
its own `slam_toolbox` block (`mapper_params_online_async.yaml`, `scan_topic: /scan_filtered`),
which is the condition under which `nav2_bringup`'s `slam_launch.py` passes the file through.
Copy it to the robot and launch, from `stretch_main`'s deploy compose (`nav` service) or any
shell there with the CycloneDDS env exported:

```bash
ros2 launch stretch_nav2 navigation.launch.py use_slam:=True \
     params_file:=/path/on/robot/nav2_params_slam.yaml use_rviz:=false
```

`use_rviz:=false` because the robot's RViz was rendering onto the Thor's screen; the map window
above replaces it. Then on the Thor, `run_robot.py`'s preflight line should read
`/map published by: slam_toolbox`. Nothing in the pipeline's config changes.

What SLAM changes for the protocol — the `map` frame now starts at the robot's start pose, and
exists only while slam_toolbox runs:

- **Do not relaunch the robot's stack between the mapping pass and the search pass.** The OSG
  map (`outputs/robot_maps/<tag>.json`) is in the `map` frame of the pass that wrote it; a new
  slam_toolbox is a new origin, and the restored objects would be wherever the robot happened to
  start. If a relaunch is unavoidable, start the search pass from the same spot and heading (tape
  on the floor), or save the session at the end of the mapping pass
  (`ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: /path/lab}"`)
  and bring slam_toolbox back in `localization` mode on it.
- **Two storeys.** A 2D SLAM cannot hold two floors in one map: carried to the other storey it
  would draw the new floor over the old. Relaunch it there (a new `map`, origin at the
  put-down spot) and publish `/osg/floor` as before; the pipeline keeps the storeys apart as
  separate layers, so their `map` origins need not agree with each other — only, per storey,
  between the two passes (previous point).
- Loop closures move the map by centimetres, not metres; the `POSE JUMP` line (threshold 1 m)
  stays quiet for them.

### Compressed camera transport

The bridge subscribes to the D435i as **jpeg colour + png depth** — `/camera/color/image_raw/compressed`
and `/camera/aligned_depth_to_color/image_raw/compressedDepth` — instead of the raw 1280×720 frames,
about a tenth of the bytes over the wired link. The choice is the topic *name*: a topic ending in
`/compressed` or `/compressedDepth` is subscribed as `CompressedImage` and decoded in the bridge
(`bridge_node.compressed_to_array`; jpeg/png colour, and compressed_depth_image_transport's
12-byte header + png for 16UC1 depth, RVL refused loudly). `configs/ros2/stretch3.yaml` carries
the two names; put the `.../image_raw` names back for raw. The bridge logs which transport it is
on at start-up (`camera: … (compressed)`), and the fake robot follows the same rule, so
`check_ros2_pipeline.py` exercises the decoder. On the robot the compressed topics exist when
image_transport's plugins are installed (`ros2 topic list | grep compressed`); the depth
encoder's `png_level` on the driver is worth setting to 1, since level 9 is CPU the robot does
not have. `CameraInfo` and TF are unchanged.

### Gates on the robot

For the real-world runs the Stretch presets admit **everything the detector names** into the
scene graph and let **one sighting** be a candidate: `scene_graph.min_det_score 0`,
`min_det_bbox_px 0`, `target_bypasses_gates true`, `verification.min_obs 1`, `min_evidence 0.2`,
`min_bbox_px 600` (`configs/experiment/stretch3_map.yaml`, inherited by `stretch3_search`). The
runs of 2026-09-22 showed why: the ball entered the graph from single sightings no gate would
act on, and its clearest sighting never entered at all because admission is
`scene_graph.min_det_score`, not the two floors lowered the day before. These are preset values,
not defaults — the simulator arms keep theirs. The cost is phantom tracks from one bad
detection; the height-plausibility rule and presence still apply, and the map window shows
every track with its gate numbers.

### Two storeys: the operator's protocol

Nothing climbs (`stretch3_map` turns the stair machinery off), so a second storey is reached
the way a Stretch reaches one in real life: someone carries it, or rides the lift with it.
The pipeline needs to be **told**, because Nav2's `map` is 2D and reports both storeys at the
same height — `/osg/floor` is the height sensor (`docs/ROS2.md`, *Two floors, one switch*).
The whole two-floor experiment is still the same two commands; the storey switch is a third
command, `scripts/ros2/switch_floor.sh`, run from the host (it re-enters the bridge container
itself) or from any ROS shell on the same domain:

```bash
bash scripts/ros2/switch_floor.sh 1 lab_upstairs     # after the carry: that map, localised, /osg/floor 1
bash scripts/ros2/switch_floor.sh 1 --check-only     # just: is the robot inside its current map?
bash scripts/ros2/switch_floor.sh 1 --via none       # a virtual storey on the same physical floor
```

It does, in order, the three things that used to be typed one at a time — and the order is
what matters: (1) the robot's navigation map becomes the named one, `<map_name>.yaml` through
Nav2's `map_server` (`/map_server/load_map`) or `<map_name>.db` through RTAB-Map
(`/rtabmap/load_database`, then localization mode), whichever the robot runs (`--via` forces
one; the name resolves under `--maps-dir`, a directory **on the robot**, default
`/home/hello-robot/ament_ws`);
(2) it waits until `map->base_link` lies on a free cell **inside** that map — measured
2026-09-22: a goal posted while the robot sat outside the grid was accepted by Nav2 and held
for 68 s without a wheel turning, because the planner had no start cell (`--initial-pose "X Y
YAW_DEG"` seeds AMCL when the localiser needs a hint after a carry); (3) only then does it
publish `/osg/floor N`, and it confirms the message was heard on the domain. The raw form is
still `ros2 topic pub --once /osg/floor std_msgs/Int32 "{data: 1}"` from the bridge container.

**Mapping pass, both floors, one run.** Start on floor 0 with the target placed on floor 1:

```bash
python scripts/run_robot.py +experiment=stretch3_map ros2.map_tag=lab ros2.target=cup
```

Let it explore floor 0 until it has nothing left there. That state is visible: no new frontier
or goal appears in RViz and the status line stops changing, because when a storey is exhausted
`select` returns nothing and the agent **idles rather than ending the run** (`nav_agent.py`,
`if choice is None: return`). The timing is therefore not delicate — carry the robot up
whenever floor 0 looks done, then publish `{data: 1}`. The switch lifts the pose by
`floor.virtual_storey_m` (3 m), so floor 1 gets its own grid, frontiers and room labels instead
of being pasted over floor 0. It explores floor 1, finds the cup, STOPs, and `save_map` writes a
schema-v2 snapshot with **both storeys** and the cup on floor 1. Do not run the mapping pass
twice: a second `map_mode=map` run does not load the first and would overwrite `lab.json`.

**Search pass.** Move the cup (to floor 0, say). The snapshot is restored with
`initial_floor_key = <the declared storey>`, and `reset()` deliberately **drains** any switch
published before the run started — so start the run on the floor you are on, and publish the
switch only *after* `[robot] restored N tracks on floor 0` has printed if that floor is not 0:

```bash
python scripts/run_robot.py +experiment=stretch3_search ros2.map_tag=lab ros2.target=cup
```

The restored belief says *cup, floor 1*. The search posterior aggregates its mass per storey
(`floor_mass_rule: mean`, with `floor_mass_margin`), and when another storey is decisively
better the strategy records a wish — `exploration.requested_floor` — and hands it to
`FloorPolicy`, which on the Stretch has no way to act on it, so the agent keeps searching the
storey it is on. That wish is what the operator acts on. It is surfaced two ways: the RViz
status line turns into **`>>> WANTS FLOOR 1: carry the robot, then ros2 topic pub …`**, and
`run_robot.py` prints the same line once. Carry it, publish the switch, and the search continues
on floor 1 with floor 1's restored map — where absence-as-evidence does its work when the cup
is not where the map said.

**Search pass from a one-storey map.** The mapping pass need not cover both floors. When run 1
mapped floor 0 only and the target is then carried upstairs, run 2 restores a map with no
storey 1 in it, and neither request above can name one: the posterior scores storeys that
have surfaces, and the disproved rule asks for the nearest *known* level. So on the Stretch
presets (`agent.request_new_storey_when_exhausted`, on in `stretch3_map`) a storey the agent
has **finished** — no frontier left on it, or `agent.request_new_storey_after_steps` spent on
it (0 is no budget), or disproved by failed attempts with nowhere known to go — makes it ask
for the lowest storey key nobody has stood on: the same **`>>> WANTS FLOOR 1`** cue, in RViz
and once on the console. Carry it up, give RTAB-Map a fresh session, publish `{data: 1}`, and
it explores floor 1 from a clean grid and a scene graph with nothing filed on that storey. The
same rule on floor 1 would ask for floor 2 — the pipeline does not know how many storeys the
building has; the operator does, and `agent.max_steps` (400) is the budget across both.

**Where it waits.** Give the staircase's position at launch, in the robot's `map` frame:

```bash
python scripts/run_robot.py +experiment=stretch3_search ros2.map_tag=lab ros2.target=cup ros2.stairs_xy=[3.2,-1.5]
```

With it, the request drives the robot to the stairs (Nav2, like any goal) and holds the base
there — `wait`, not the exploring turn — until `/osg/floor` arrives; the RViz status line
says `driving to` / `waiting at the stairs (x, y)`. Without it, it waits where the request
found it. Read the coordinates off RViz (`Publish Point`) or `tf2_echo map base_link` with
the robot parked at the foot of the stairs.

Two things on the robot side that this repo cannot fix for you: RTAB-Map's 2D map will also
overprint the two storeys unless it is given a fresh session per floor (Nav2 then plans on a
map with both floors' walls); and the raw camera streams in RViz stay the room on its side —
use the `/osg/*` ones.

### Before the robot is in the room

The checks in `docs/ROS2.md` work on the Thor unchanged. Start with the one that needs
neither ROS nor a robot:

```bash
bash scripts/ros2/check_pipeline.sh --loopback     # everything above the bridge
```

Then, with the bridge container up and the fake robot running in it, the ROS path:

```bash
docker exec -it osg-bridge bash -c 'source /opt/ros/humble/setup.bash && ros2 topic list'
docker exec -it osg-bridge bash scripts/ros2/fake_robot.sh --floor-switch-after 6
docker exec -it osg-thor   python scripts/check_ros2_pipeline.py
```

## Set these for your unit

The image is parameterised because none of it can be checked from here. `docker/.env`:

| build arg | what it is | how to find it |
|---|---|---|
| `BASE_IMAGE` | the container image matching the device's JetPack | `cat /etc/nv_tegra_release` on the Thor; a mismatch between the container's CUDA userspace and the host driver is the failure that looks like a broken GPU. **Not an `l4t-*` tag on JetPack 7** — see below |
| `CUDA_ARCH` | `TORCH_CUDA_ARCH_LIST` for the compiled kernels | `python3 -c "import torch; print(torch.cuda.get_device_capability())"` once torch is in. Thor is Blackwell-generation and reports `11.0`; Orin is `8.7`. Wrong value → GroundingDINO silently falls back to CPU, which is a day rather than a run |
| `TORCH_INDEX_URL` | NVIDIA's Jetson wheel index, for a base image that has no torch | **Empty on JetPack 7**: the NGC base carries torch. Only a JetPack-6 `l4t-jetpack` base needs one, and the `jp7` index this file used to name has been withdrawn (`410 Gone`) |
| `WITH_BLIP2` | whether to install salesforce-lavis | `1`, and verified to work here — see below. `0` if it ever stops: it feeds only the value map and ASCENT's commit gate, and `exploration.value_model=clip` replaces it |
| `ROS_DOMAIN_ID` | must match the robot's | `echo $ROS_DOMAIN_ID` on the Stretch |
| `RMW_IMPLEMENTATION` | must match the robot's | the Stretch runs CycloneDDS; mismatched vendors discover each other unreliably and then fail in ways that look like network trouble |
| `CYCLONEDDS_IFACE` | **this** machine's wired NIC, not the robot's | `ip -brief addr` on the Thor — `enP2p1s0`. The shared DDS config pins to it; a name that does not exist, or a NIC that is merely down, stops Cyclone with `does not match an available interface` |
| `CYCLONEDDS_CONFIG` | the wired-link DDS config | `docker/cyclonedds-eth.xml`, a copy of `stretch_main`'s — that repo owns the settings, and both ends must pin identically or they never discover each other, so keep the copies in sync. See *CycloneDDS on the link to the robot* in `docs/ROS2.md` |

The robot-side items — topic names, the portrait camera's rotation, the camera height, the
head-tilt joint — are all in the `ros2` config group and listed in `docs/ROS2.md`. `bridge.sh`
composes that group and passes it to the node, so settling one is a yaml edit on either
machine.

### Why the base image is not an `l4t-*` one

The Jetson-specific container line stopped at JetPack 6. `nvcr.io/nvidia/l4t-jetpack`
publishes nothing past `r36.4.0` and `l4t-cuda` nothing past `12.6`, so an `r38`/`r39` tag —
the L4T release a Thor actually reports — does not exist, and naming one fails the build at
`FROM` with `no such manifest`, before a single instruction runs.

JetPack 7 does not need that line: the ordinary arm64 NGC images run on the device.
`nvcr.io/nvidia/pytorch:25.10-py3` is Ubuntu 24.04, CUDA 13.0 and torch 2.9 built with
`sm_110` among its architectures — which is what this GPU reports — and it ships `nvcc`, so
GroundingDINO's fused attention kernel is still compiled in-image. Two consequences follow,
and both are the reverse of what the x86 image does:

- **No Jetson wheel index.** Torch is in the base. `TORCH_INDEX_URL` stays empty and exists
  only for a JetPack-6 base.
- **`numpy>=2`, not `numpy<2`.** The pin is not about numpy, it is about matching the ABI the
  torch in use was built against. The x86 wheel is a 1.x build; this one is a 2.x build, and
  a 1.x numpy under it is what produces `_ARRAY_API not found`.

Ubuntu 24.04 also ships a `ubuntu` user at uid 1000, so the image takes that id over when
`UID=1000`; without that the container starts and immediately fails with `unable to find user
user`.

### BLIP-2 on aarch64

`WITH_BLIP2=1` works here, but not the way it does on x86. `salesforce-lavis` depends on
`decord`, which has **no aarch64 distribution at all** — no wheel, and no version that builds
— so a plain install ends in `ResolutionImpossible` naming it. The image installs `decord2`
instead, a maintained fork that publishes manylinux aarch64 wheels and installs its module
under the name `decord`, which is the name lavis imports; lavis itself then goes in with
`--no-deps` and a dependency list supplied by hand.

This matters because `configs/exploration/s71.yaml` — the default arm — sets
`value_model: blip2itm`. Had BLIP-2 simply been dropped, the robot would have been running a
different value model from every measured result. `exploration.value_model=clip` remains the
fallback if it ever breaks.

### The versions are pinned to the x86 reference, and checked

Left to float on this platform, the models venv resolves to `transformers` 5.x, `timm` 1.x and
OpenCV 5.0 — none of which the recorded ASCENT environment has ever run. They are pinned to
its versions (`transformers==4.37.0`, `timm==0.4.12`, OpenCV 4.x) in
`docker/thor-model-requirements.txt` and `/etc/pip/constraint.txt`, and the build's last
perception step asserts what it actually got, so a silent upgrade fails the build rather than a
model server on the robot.

## If something is wrong

| symptom | first thing to check |
|---|---|
| `BridgeUnavailable: no ROS bridge at 127.0.0.1:18765` | the `osg-bridge` container is up, and **both** containers are on `network_mode: host` |
| `ros2 topic list` in the bridge shows nothing from the robot | `ROS_DOMAIN_ID` and `RMW_IMPLEMENTATION` match the Stretch's, and `ROS_LOCALHOST_ONLY=0`. Then check both ends pin the same NIC — one side left on Cyclone's `autodetermine` can sit on Wi-Fi while the other is on Ethernet, and they never meet |
| the bridge exits at once with `does not match an available interface` | `CYCLONEDDS_IFACE` names a NIC that is absent **or down** — on the Thor, that usually means the cable to the robot is not plugged in |
| the bridge exits with `can't open configuration file` | `CYCLONEDDS_CONFIG` points nowhere, so docker made a directory at the mount point; or the XML is malformed — a comment containing a double hyphen will do it |
| images stutter or vanish, with no error | `net.core.rmem_max` on the HOST (`network_mode: host` means the host's value applies) — `docker/.env.example` has the one-liner |
| `no camera frame within …s` | the topic names in the `ros2` group, and that TF actually carries `map` → the image header's frame |
| `torch.cuda.is_available()` is False at run time | `BASE_IMAGE` does not match the device's JetPack |
| the build stops on `no such manifest` before any step runs | `BASE_IMAGE` names an `l4t-*` tag that was never published — see *Why the base image is not an `l4t-*` one* |
| GroundingDINO is slow | `CUDA_ARCH` is wrong, so the kernel compiled for another architecture and it fell back to CPU |
| the run refuses to start on a model server | `bash scripts/serve_perception.sh`, and `WITH_BLIP2=0` builds need `exploration.value_model=clip` |
| `PerceptionUnavailable ... 13182/blip2itm` **mid-run**, robot already moving | the pre-flight probe is a no-op under `detector=yoloe`; curl `:13182` and `:13185` before every run (see *The two runs*) |
| `ModuleNotFoundError: No module named 'open_clip'` in `build_run_components` | an image built before `open_clip_torch` + `ml-mobileclip` were added to `Dockerfile.thor`; rebuild |
| the mapping pass STOPs almost at once | it found the default target `chair`; pass `ros2.target=` on the mapping pass too |
| `run_robot.py` refuses with `no navigate_to_pose action server on the robot` | Nav2 is not running on the Stretch (a reboot loses it). `ros2 action info /navigate_to_pose` must show a **server**, not just our client; relaunch `stretch_nav2 navigation.launch.py` there |
| `run_robot.py` refuses with `the robot's nodes are not on CycloneDDS: fastdds: …` | the robot's stack came up on ROS's default Fast-DDS (a reboot does this unless `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` is exported in every launching shell). Topics still flow, so it looks alive, but actions/services never cross vendors: every Nav2 goal times out and the base never moves. Relaunch the whole robot stack with the CycloneDDS env |
| the initial spin works, the scene graph builds, goals appear in RViz, but the base never drives and `/plan` stays empty; the run ends with many `frontier_give_up` | the same vendor mismatch, seen from the run. Confirm with `ros2 topic info -v /tf`: a GID starting `01.0f` is Fast-DDS, `01.10` is CycloneDDS |
| RViz opens with no displays; its log says `Could not load display config: Invalid argument` | rviz2 rejected `scripts/ros2/osg.rviz`. Bisect by loading prefixes of the display list (`rviz2 -d`); the file's header records the one naming clash already found |
| the build stops on `models venv drifted: ...` | a dependency pulled one of the pinned versions up; pin it in `docker/thor-model-requirements.txt` rather than relaxing the check |
| `pip install salesforce-lavis` fails with `ResolutionImpossible` naming `decord` | expected on aarch64 — the image installs `decord2`; see *BLIP-2 on aarch64* |

## Files

| file | |
|---|---|
| `docker/Dockerfile.thor` | the pipeline and the models, on the arm64 NGC base |
| `docker/Dockerfile.bridge` | `ros:humble-ros-base` plus the message packages |
| `docker/compose.thor.yaml` | both containers, plus ollama, all on host networking |
| `docker/thor-model-requirements.txt` | the servers' aarch64 dependencies, and why they are not the x86 pins |
| `docker/entrypoint.thor.sh` | no conda wrapper: the pipeline env is the system python |
