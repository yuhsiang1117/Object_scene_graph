# The two-floor demo, step by step

The moved-object demo on the Stretch 3, as commands. Why each step is what it is lives in
`docs/THOR.md` and `docs/ROS2.md`; this page is only the order to type things in.

The story: pass 1 maps floor 0 and finds the ball there. The ball is moved "upstairs". Pass 2
starts from the stale map, finds the ball gone, searches floor 0, gives up, drives to the
stairs and waits. A person carries the robot up and tells the pipeline; it explores floor 1
and finds the ball.

## 0. Bring everything up (once per session)

On the Stretch: its driver, RTAB-Map in localization mode on a pre-built map that covers
**every** area you will use (including where the robot gets put down), and Nav2.

On the Thor, a host shell in the repo:

```bash
docker compose -f docker/compose.thor.yaml --env-file docker/.env up -d      # osg-thor, osg-ollama, osg-bridge
docker exec -it osg-thor bash                                                 # -> /workspace
bash scripts/serve_perception.sh                                              # the model servers (tmux osg_models)
for p in 13182/blip2itm 13185/ram; do curl -s -o /dev/null -w "$p %{http_code}\n" http://localhost:$p; done
python scripts/smoke_ollama.py
```

Both curls must print `404` — that is "alive". A run does not refuse to start on a dead
BLIP-2; it dies mid-episode instead, so check before every run.

RViz on the Thor's screen, optional:

```bash
docker compose -f docker/compose.thor.yaml --env-file docker/.env run --rm rviz
```

Once, to learn where the stairs are: park the robot at their foot, then

```bash
docker exec osg-bridge bash -c 'source /opt/ros/humble/setup.bash && ros2 run tf2_ros tf2_echo map base_link' | grep -m1 Translation
```

The first two numbers are `X Y` for step 3.

## 1. Pass 1 — map floor 0, ball on floor 0

Inside `osg-thor`, at `/workspace`:

```bash
python scripts/run_robot.py +experiment=stretch3_map ros2.map_tag=lab ros2.target='sports ball'
```

It explores, finds the ball, stops, and writes `outputs/robot_maps/lab.{json,npz}`. Do not run
this again with the same tag — it overwrites the map.

## 2. Move the ball "upstairs"

Put the robot back at the pass-1 start spot and heading (tape on the floor).

## 3. Pass 2 — search from the stale map

```bash
python scripts/run_robot.py +experiment=stretch3_search ros2.map_tag=lab ros2.target='sports ball' \
    ros2.stairs_xy=[X,Y]
```

Optional: `agent.request_new_storey_after_steps=150` to cap how long it searches floor 0.

What you will see: `[robot] restored N tracks on floor 0` → it drives to where the ball was →
finds it gone → searches floor 0 → when nothing is left (or the budget is spent) it prints
`[robot] the agent wants floor 1 ...`, drives to `(X, Y)` and stands still. RViz shows
`>>> WANTS FLOOR 1 ... [waiting at the stairs (X, Y)]`. **The run stays alive** — leave it.

## 4. Carry the robot up, then one command

From a host shell:

```bash
bash scripts/ros2/switch_floor.sh 1 --via none          # same physical floor: keep the current map
bash scripts/ros2/switch_floor.sh 1 lab_upstairs        # a real floor: load ament_ws/lab_upstairs.yaml first
```

It waits until the robot is localised inside the map, then publishes `/osg/floor 1` and
confirms it was heard. If it answers `OUTSIDE the grid`, point the camera at a mapped view
(or add `--initial-pose "X Y YAW_DEG"` for AMCL) and run it again. `--check-only` only asks
where the robot is.

## 5. Watch it finish

Pass 2 continues on floor 1 with a fresh grid and nothing filed on that storey, finds the
ball, stops. Everything is in `outputs/<timestamp>/`: `stream.jsonl` (step, state, floor,
what it wants, its goal), `episodes.jsonl`, `viz/`.

## If the search run died after the carry

The robot is upstairs, its navigation map is right, and the pipeline process is gone. Start
the search pass again with a hold, and publish the switch when RViz shows the restored graph:

```bash
# inside osg-thor
python scripts/run_robot.py +experiment=stretch3_search ros2.map_tag=floor0 ros2.target='sports ball' \
    ros2.wait_for_switch=true
# host shell, once "[robot] holding on floor 0" has printed
bash scripts/ros2/switch_floor.sh 1 --via none
```

The run restores the floor-0 map (stale ball and all), holds with the base still, applies
the switch on its first step — the storey lift on RViz — and explores floor 1.

## Starting the upstairs search from a place you choose

`ros2.waypoint_xy=[X,Y] ros2.waypoint_yaw_deg=D ros2.waypoint_floor=1` makes the first
exploration goal on floor 1 that pose in the robot's `map` frame (a quaternion `(z, w)` is
`D = 2 * atan2(z, w)` in degrees), and the search continues from there. This is a human
choosing where the search starts — say so wherever the run is reported.

## Do not

- Restart `osg-bridge` while a run is live — the run dies.
- Call `/rtabmap/reset` — it throws away the pre-built map.
- Forget `agent.max_steps` (400) is the budget across **both** floors; raise it for a big lab.
