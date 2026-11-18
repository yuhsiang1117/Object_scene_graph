# The robot: ROS 2 Humble, Nav2, and a Hello Robot Stretch 3

Everything before this document runs in Habitat. This is how the same pipeline drives a real
robot — what changes, what deliberately does not, and how to check the whole path with no
robot in the room.

The short version: **the pipeline keeps doing all of the thinking and stops steering.** The
Stretch already runs Nav2, which has the robot's footprint, its local costmap, its recovery
behaviours and a controller tuned for that base. Nothing in this repo plans better than that.
So the frontier/approach goal the pipeline has always computed is published to
`NavigateToPose` instead of being turned into `move_forward`, and sensor data flows back the
other way.

```
  Stretch 3  ──DDS──▶  bridge_node.py          ──socket──▶  Ros2Env + Nav2Driver
  camera, TF           (system python3.10,                  (habitat env, python3.9)
  Nav2, cmd_vel         rclpy, no pipeline)                  NavAgent, scene graph
             ◀──────  NavigateToPose  ◀─────────────────────  the metric goal
```

On the real deployment those two halves are two containers on a Jetson AGX Thor, and the
same split is what lets the bridge match the robot's Ubuntu exactly while the pipeline
follows JetPack's. See [THOR.md](THOR.md).

## Why there are two processes

`rclpy` in Humble is built against Ubuntu 22.04's Python 3.10. Both conda envs in this image
are Python 3.9, because habitat-sim 0.3.1 pins them there — the Dockerfile has said so beside
the habitat env since long before this layer existed. They cannot share an interpreter.

So they share a socket. `src/osg/ros2/bridge_node.py` runs on `/usr/bin/python3` beside the
ROS graph and does nothing but relay; everything else in `src/osg/ros2/` is written to import
under *both* interpreters (numpy and the standard library only) and is unit-tested in the
normal CPU suite. This is the same shape as the five perception models, which are separate
processes for the same kind of reason.

`multiprocessing.connection`, not HTTP or ZeroMQ: it is in the standard library of both
interpreters, so neither environment grows a dependency. Arrays are encoded explicitly
(`{dtype, shape, bytes}`) rather than pickled, because the two sides are different numpy
builds.

## Frames

Three conversions hide behind the word "frame". All three live in `src/osg/ros2/frames.py`,
which imports no ROS and is tested against projected points rather than against itself —
every mistake here produces a mirrored or side-lying map that renders entirely convincingly
and fails 400 steps later.

| | ROS | pipeline |
|---|---|---|
| world | `map`: z up, x forward, y left | habitat: y up, ground plane `(x, z)` |
| camera | `*_optical_frame`: z fwd, x right, y down | identical (OpenCV) |
| goal | `(x, y)` in `map` | `(x, z)` in world metres |

World: `hab = (ros_x, ros_z, -ros_y)`. This is the repo's existing `W_ROS`
(`scripts/render_dualmap_sequence.py`), reused so a map built on the robot and one built from
a collector recording are in the same frame; a test pins them together. The camera needs no
conversion at all, which is why `T_wc = T_HAB_ROS @ T_map_cam` is the whole of it. Goals go
back as `ros = (hab_x, -hab_z)`.

The head camera is mounted **portrait**, so the published image is the room on its side while
every consumer assumes upright — the detector was trained on upright photographs, the costmap
bands by height. `ros2.rotate_deg` (default 90) rotates the image, the intrinsics and the
pose together.

## Who owns the base

In Habitat every tick is one discrete action and the question never arises. Here the FSM still
issues its own actions in several states — the opening 360° scan, the escape window, a
`look_down` near stairs, the close look — while Nav2 may be driving toward a goal. Two things
steering one base is two things fighting over the wheels.

`Ros2Env.step` resolves it every tick:

| the driver posted a goal | a goal is outstanding | what happens |
|---|---|---|
| yes | yes | Nav2 is driving. The `move_forward` the driver returned is a **placeholder** and is not executed; the tick is a `step_period_s` pause, then a fresh frame. |
| no | yes | The FSM is steering. Cancel the goal first, then meter the action on `cmd_vel`. |
| either | no | Meter the action. Nothing to cancel. |

`stop` always cancels and ends the run.

The FSM's own actions are metered against odometry (`map → base_link` via TF), not timed: an
open-loop timed move drifts with battery and carpet, and `agent.forward_m` is a distance the
costmap and the frontier reach radius both assume.

## Two floors, one switch

Every floor decision in the pipeline reads `camera_position[1]` — the agent's own height above
the floor. **On the robot that signal does not exist.** Nav2's `map` is 2D and reports both
storeys at the same z.

So the operator declares the storey:

```bash
ros2 topic pub --once /osg/floor std_msgs/Int32 '{data: 1}'   # upstairs
ros2 topic pub --once /osg/floor std_msgs/Int32 '{data: 0}'   # back down
```

`floor.source=external` makes that the only source; the estimator is never consulted. The
switch propagates:

```
/osg/floor ─▶ bridge ─▶ Transport.pop_floor_switch ─▶ Ros2Env.step
   ─▶ FloorPolicy.request_floor  (queued: the costmap must not change
                                  underneath a control loop that read it)
   ─▶ FloorPolicy.observe, next step: the storey's own grid, planner and room
      labels; new objects filed under `floor_key`; `rebuild_floor` on that
      storey; frontiers and the search posterior on that storey.
```

The switch also **lifts the pose** by `key * floor.virtual_storey_m` (3 m). The switch is the
height sensor. Without it, both storeys occupy one costmap band, one `floor_of_height`
answer, and one position in the LLM prompt's height-ordered floor list — so the second floor's
map would be pasted on top of the first. With it, every height consumer downstream sees the
geometry it was calibrated on, and run 1's snapshot is consistent with run 2's.

The same path is exercisable in Habitat with no ROS at all:

```bash
python scripts/run_eval.py +experiment=floor_schedule_habitat eval=dev50_mf \
       eval.num_episodes=1 eval.behaviour_log=true
# episodes.jsonl: floor_log keys 0 -> 1 -> 0 at steps 60/160,
#                 agent_stats.floor_switch_external == 2
```

**Nothing climbs.** `stretch3_map` turns off `climb_enabled`, `floor.stairs` and
`floor.cross_floor`: the stair machinery assumes the agent can walk up a staircase it
detected, the Stretch cannot, and a portal pursuit with no way to complete it just burns the
step budget. Floors change because an operator says so and by no other route.

## The two runs

```bash
# run 1 — explore, and keep what was learned
python scripts/run_robot.py +experiment=stretch3_map ros2.map_tag=lab

# ... move something ...

# run 2 — start out confidently wrong, and notice
python scripts/run_robot.py +experiment=stretch3_search ros2.map_tag=lab ros2.target=cup
```

The two presets are one field apart on purpose: the arm that searches must be the arm that
mapped. Run 1 writes `outputs/robot_maps/lab.json` (+ `.npz`) — the scene graph's evidence and
per-storey occupancy, schema v2, exactly what the YCB two-pass protocol writes. Run 2 restores
it with `apply_map(..., initial_floor_key=<the declared storey>)`, capped at
`reload_max_log_odds` so a stale belief cannot saturate, and then the presence filter, the
identity channel and absence-as-evidence do what they were built for
(`docs/ARCHITECTURE.md`).

Nothing is scored. `Ros2Env.metrics()` returns NaN rather than a 0.0 success rate that would
sit in `summary.json` looking measured; there is no ground truth in a real room.

## Checking it without a robot

There are two fake robots, and the difference between them is the point.

```bash
# everything ABOVE the bridge, with no ROS installed at all
bash scripts/ros2/check_pipeline.sh --loopback

# the real thing: rclpy, DDS, real topics, a real NavigateToPose server
docker compose -f docker/compose.yaml -f docker/compose.ros2.yaml --env-file docker/.env up -d
docker exec -it docker-nav-1 bash
bash scripts/ros2/check_pipeline.sh          # fake robot + bridge in tmux, then the checks
bash scripts/ros2/check_pipeline.sh --stop
```

`--loopback` (`src/osg/ros2/loopback.py`) serves the same wire protocol directly, sharing the
room and the driving model with `fake_robot.py` and differing only in how they reach the
pipeline. So a check that passes on the loopback and fails against the bridge has localised
the fault to the ROS half — which is the whole reason for keeping both. The loopback also
needs no image rebuild, so it works in a container built before this layer existed.

`src/osg/ros2/fake_robot.py` publishes a **real box room** rendered by ray casting — not a
constant depth image, which would map to no free space and produce no frontier — with a
doorway gap that reads as unexplored. It drives when driven, so its `NavigateToPose` server
actually moves the published TF.

The check that matters is the goal round trip: a goal leaves the pipeline in habitat-world
metres, becomes a ROS `map` pose, is driven to, and comes back as a camera pose in
habitat-world metres. Each conversion is unit-tested; only here do they have to be inverses of
each other in practice. It also saves the converted RGB to `outputs/ros2_check/frame.png` —
look at it to settle `ros2.rotate_deg`.

Then the whole pipeline — both runs, the map written and read back — against either fake,
with no model servers:

```bash
python -m osg.ros2.loopback --floor-switch-after 8 &     # or the bridge + fake_robot
python scripts/run_robot.py +experiment=stretch3_map    detector.name=stub \
       exploration=nearest region_proposal.enabled=false agent.max_steps=40 ros2.map_tag=fake
python scripts/run_robot.py +experiment=stretch3_search detector.name=stub \
       exploration=nearest region_proposal.enabled=false agent.max_steps=25 ros2.map_tag=fake
```

Measured on the loopback: 40 steps, 3 goals posted, a floor switch at step 18, and a schema-v2
snapshot with **two storeys** — floor 0 with ~25k known cells and floor 1 with ~2.7k, in
separate grids, which is the virtual storey offset doing its job. With `--abort-goals` every
goal is refused and the FSM retires one frontier per refusal (6 posted, 6 refused, 6 retired),
which is the `policy_stop` path end to end.

And the mover itself, in Habitat, where there is ground truth:

```bash
python scripts/run_eval.py +experiment=nav2_habitat eval.num_episodes=3
```

`SimNav2Backend` gives habitat's navmesh follower Nav2's semantics (accepted / active /
SUCCEEDED / ABORTED). **Privileged navigation, exactly as `navmesh` is** — the follower plans
on simulator geometry — so SR/SPL from it is never comparable with a sensor-only arm. Its
value is attribution: if the pipeline works there and not on the robot, the fault is in the
bridge.

## Verify on the robot

Every item below is a config field, so settling one is a yaml edit — `bridge.sh` composes the
`ros2` group and passes it to the bridge (`scripts/ros2/bridge_args.py`), because the bridge's
own interpreter has no Hydra:

```bash
bash scripts/ros2/bridge.sh +experiment=stretch3_map        # the preset's values
bash scripts/ros2/bridge.sh ros2.depth_topic=/camera/depth/image_rect_raw
bash scripts/ros2/bridge.sh -- --rgb-topic /one_off         # after --, straight to the node
```

This list is `src/osg/core/config/ros2.py`.

| what to check | knob |
|---|---|
| Image/depth/info topic names; whether depth is really aligned to colour; raw or compressed (a `/compressed` / `/compressedDepth` name selects the transport) | `ros2.rgb_topic`, `depth_topic`, `camera_info_topic` |
| Portrait camera: which rotation, or whether `stretch_core` already rotates (then 0) | `ros2.rotate_deg` — look at `outputs/ros2_check/frame.png` |
| The optical frame name in TF | `ros2.camera_frame` (empty = the image header's) |
| Depth encoding and scale (16UC1 mm vs 32FC1 m) | `ros2.depth_scale` |
| The base takes velocity here, and the driver is in navigation mode (`ros2 service call /switch_to_navigation_mode std_srvs/srv/Trigger`) | `ros2.cmd_vel_topic` |
| Head tilt action and joint name | `ros2.head_traj_action`, `head_tilt_joint`, `look_step_deg` |
| Nav2's action name, goal frame, and how long it flails on an unreachable goal before aborting | `ros2.nav_action`, `goal_frame`, `nav_timeout_s` |
| DDS: CycloneDDS, host networking, a matching domain | `docker/compose.ros2.yaml`, `ROS_DOMAIN_ID`; `ros2 topic list` from the bridge shell |
| DDS pinned to the **wired** NIC at both ends, with the peer list and buffers sized for camera frames | `CYCLONEDDS_CONFIG`, `CYCLONEDDS_URI`, `CYCLONEDDS_IFACE` in `docker/.env` — see below |
| Floor-to-optical-frame height, and the FOV after rotation | `agent.camera_height`, `eval.hfov_deg`, `eval.rgb_width/height` |

## CycloneDDS on the link to the robot

The DDS tuning is **not this repo's**. `stretch_main` drives the same Stretch and owns it;
`docker/cyclonedds-eth.xml` here is a copy of `stretch_main/docker/cyclonedds-eth.xml`, and
both composes mount it at `/etc/cyclonedds-eth.xml` with `CYCLONEDDS_URI` pointing there.

**Keep the two in sync.** The config pins DDS to the wired NIC, and it only works if **both
ends are pinned the same way** — a machine left on Cyclone's `autodetermine` can choose its
Wi-Fi NIC, and then the two never discover each other at all. Drift between the copies
reproduces exactly that, and silently. Everything below the copy's header block is
stretch_main's file verbatim, so the check is a plain diff:

```bash
diff docker/cyclonedds-eth.xml ../stretch_main/docker/cyclonedds-eth.xml   # header only
```

What the file buys:

- **The wired NIC, chosen explicitly.** `autodetermine` picks one interface by Cyclone's own
  ranking, and with Wi-Fi and Ethernet both up that choice is not stable. A 1280x720 `rgb8`
  frame is 2.76 MB; at 30 Hz that is ~663 Mbit/s, which Wi-Fi will not carry.
- **A unicast peer list**, as a backstop where a direct cable or a dumb switch does not carry
  multicast reliably.
- **`MaxAutoParticipantIndex=60`.** Unicast discovery makes each participant claim an index,
  and the default cap is ten per host per domain. Past the tenth, nodes die at startup with
  `Failed to find a free participant index for domain 0` — which reads like a Cyclone/robot
  incompatibility and is only this cap.
- **Buffers sized for image traffic** (`SocketReceiveBufferSize` 10 MB, `WhcHigh` 4 MB). This
  needs `net.core.rmem_max` raised on the **host**, because `network_mode: host` means the
  host's value applies; `docker/.env.example` has the one-liner.

Three variables in `docker/.env` drive it, and `CYCLONEDDS_IFACE` is the only one that is
per-machine — **this** machine's wired NIC, not the robot's (`ip -brief addr`). The Thor's is
`enP2p1s0`.

Neither failure here is silent. A NIC name that does not exist, or one that is merely **down**
(no cable), gives `enP2p1s0: does not match an available interface` followed by
`rmw_create_node: failed to create domain`. A missing config file gives `can't open
configuration file` — docker will have created a *directory* at the mount point because
`CYCLONEDDS_CONFIG` pointed nowhere.

One editing trap, if you ever add a comment to the file: **an XML comment may not contain a
double hyphen**, so a rule line of dashes makes the whole file unparseable and Cyclone refuses
to start.

**Both ends must also be the same vendor.** `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` has to be
exported in *every* shell that launches part of the robot's stack — the driver, the camera,
Nav2, SLAM — or the robot comes up on ROS's default Fast-DDS. That mismatch is the one that
looks like a working robot: DDS topics interoperate across vendors, so the map, TF, the images
and `cmd_vel` all flow, but services and actions do **not**, so every `NavigateToPose` goal goes
unanswered, `Nav2Driver` times it out as refused after `nav_timeout_s`, and the base never
moves while frontiers are retired one by one. The bridge now reads each `/tf` publisher's
vendor from its GID and `run_robot.py` refuses to start against a Fast-DDS robot; by hand,
`ros2 topic info -v /tf` shows GIDs starting `01.0f` (Fast-DDS) or `01.10` (CycloneDDS).

## Where the code is

| file | |
|---|---|
| `src/osg/ros2/wire.py` | the socket protocol and the array codec |
| `src/osg/ros2/frames.py` | every ROS↔pipeline conversion, ROS-free and unit-tested |
| `src/osg/ros2/transport.py` | the pipeline's end; the seam the tests fake |
| `src/osg/ros2/bridge_node.py` | the rclpy node (system python3.10 only) |
| `src/osg/ros2/fake_robot.py` | the synthetic robot, over ROS |
| `src/osg/eval/debug_stream.py` | the simulator's debug view and the scene graph, pushed through the bridge to RViz (`/osg/*`) |
| `scripts/ros2/rviz.sh`, `scripts/ros2/image_window.py`, `scripts/ros2/osg_map.rviz`, `scripts/ros2/osg.rviz` | the operator's two windows (camera: `image_window.py` on `/osg/detections`; map: rviz2 on `osg_map.rviz`) and the old everything-in-one view; on the Thor, the `rviz` compose service |
| `docker/stretch/nav2_params_slam.yaml` | the robot's Nav2 params with its slam_toolbox block, for `navigation.launch.py use_slam:=True` (docs/THOR.md, *Mapping with slam_toolbox*) |
| `src/osg/ros2/loopback.py` | the same robot with the ROS taken out |
| `src/osg/planning/nav2_driver.py` | the mover: `goal_xy -> NavStep` |
| `src/osg/planning/nav2_backends.py` | the robot's navigator, and habitat's |
| `src/osg/sim/ros2_env.py` | the env contract, base ownership, the storey lift |
| `src/osg/agent/floor_policy.py` | `request_floor` and the external source |
| `scripts/run_robot.py` | the two runs |
| `scripts/check_ros2_pipeline.py` | the checks |
