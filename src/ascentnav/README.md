# `ascentnav` — ASCENT's navigation pipeline, reimplemented

## Why this package exists

Porting ASCENT's mechanisms into `osg`'s agent one at a time plateaued: the
sensor-only arm has sat at 42% since S30, and S31 (carrot climb), S33 (RedNet
stairs), S34 (re-selection cadence) and S35 (obstacle band) each measured null or
worse. S37 replaced the *control flow* but kept OSG's maps.

This package replaces the maps too. ASCENT's `ObstacleMap`, `ValueMap` and
`ObjectPointCloudMap` are vendored from the reference implementation and driven
by ASCENT's control flow, because those maps differ from OSG's in kind rather
than in parameters:

| | OSG costmap | ASCENT ObstacleMap |
|---|---|---|
| obstacle band | `[0.15, 1.5]` m | `[0.61, 0.88]` m |
| free space | per-frame depth raycast | fog-of-war reveal over a navigable map |
| navigable | `~inflated(radius)`, computed on demand | maintained, square agent-radius kernel |
| frontiers | contour of the explored region | `detect_frontier_waypoints` on explored ∧ navigable |
| stairs | a side hit-grid | first-class up/down maps, excluded from dilation |

## Layout

```
ascentnav/
  agent.py       AscentNavAgent: ASCENT's act() dispatch, OSG runner interface
  geometry.py    OSG world frame -> ASCENT's episodic frame (unit-tested)
  constants.py   verbatim from ascent/constants.py
  mapping/       obstacle_map, value_map, object_point_cloud_map (from ascent/)
  vendor/        the frontier_exploration and vlfm helpers those maps import
```

Run it with `+experiment=ascentnav`, which sets `agent.policy: ascentnav`.

## Substitutions, and why each is defensible

Three ASCENT components cannot run in this container. Each is replaced by the
OSG equivalent rather than dropped, and each was already a documented deviation:

* **Detector** — ASCENT runs D-FINE + GroundingDINO + MobileSAM. `transformers`
  is not installed and the weights are HF-format. OSG's YOLOE is used.
  S15 measured the detector as worth **zero** on this split (YOLOE-11s@512
  against 11l@640: 0 at the loose gate, −1 at the tight one), so of the three
  this is the least likely to matter.
* **Value-map scorer** — ASCENT uses BLIP-2 ITM; `lavis` is not installed. OSG's
  CLIP scorer supplies the cosine through the same `ValueMap.update_map` call.
* **LLM** — ASCENT runs Qwen2.5-7B locally; the hosted client is used instead.

`open3d` is also absent. It appears exactly twice, both times for
`cluster_dbscan`, and is replaced by `sklearn.cluster.DBSCAN` — the same
algorithm with the same noise convention — rather than taking a 400 MB
dependency for one call.

So this is ASCENT's **navigation** — maps, frontiers, control flow, mover — with
OSG's perception plumbing. Given the gap being chased is navigation (ASCENT 70%
sensor-only against OSG's 63% *with* a ground-truth navmesh), that is the part
worth reproducing faithfully.

## The object scene graph

Built from the same object cloud and **read-only with respect to every
navigation decision**, which is the premise this package was requested under.

## Two things the smoke caught

* **Frontier stop radius.** Handing the mover `stop_radius=0.9` made it report
  "arrived" for any frontier within 0.9 m, and each of those became a blind
  forced-forward: 189 of 500 steps. ASCENT passes `stop=False` on every frontier
  call, which makes its `rho < stop_radius` branch inert
  (`ascent_policy.py:869-872`) — there is no such thing as arriving at a
  frontier. With `stop_radius=0.0` the count went to **0** and the network
  chooses every action.
* **Camera frame.** VLFM's `get_point_cloud` returns `(z, -x, -y)`, i.e. X
  forward / Y left / Z up — not the OpenCV convention an OSG habit assumes.
  `tests/unit/test_ascentnav_geometry.py` derives it from the real function
  rather than asserting one, and pins the transform end to end including the
  looking-down sign check.
