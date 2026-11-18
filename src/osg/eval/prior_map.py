"""The two-pass protocol: navigate from a map the world has since invalidated.

This is what makes the benchmark a DYNAMIC-scene benchmark rather than an
incomplete-map one. Pass 1 explores the static layout and writes one snapshot
per scene; pass 2 runs the moved layout starting from that snapshot, so the map
the agent navigates with is genuinely stale -- it confidently says the object is
at A, and the object is at B.

The staleness IS the experiment. An agent that rebuilds from scratch every
episode is never wrong about anything, and measures nothing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .record import authored_episode_metadata, episode_tag, safe_tag


def authored_scene(episode) -> str:
    return str(authored_episode_metadata(episode).get("scene", "scene"))


def _map_path(root: str, scene: str) -> Path:
    return Path(str(root)) / f"{safe_tag(scene)}.json"


def _episode_map_path(root: str, scene: str, episode) -> Path:
    return Path(str(root)) / f"{safe_tag(scene)}__{safe_tag(episode_tag(episode))}.json"


def _scene_map_paths(root: str, scene: str) -> list:
    """Every snapshot for this scene: the best-of file and any per-episode
    ones, in a stable order so a union is reproducible."""
    base = Path(str(root))
    found = [p for p in sorted(base.glob(f"{safe_tag(scene)}__*.json"))]
    single = _map_path(root, scene)
    if single.exists():
        found.insert(0, single)
    return found


def storeys_from_snapshots(paths, gap_m: float = 0.9) -> list:
    """The scene's storeys, clustered out of the obstacle snapshots.

    Every mapped floor contributes the height its mapping agent stood at,
    weighted by how much of it was explored, and floors closer together than
    `gap_m` are ONE storey. That last part is the whole point: ASCENT allocates
    a floor per staircase it notices, so one storey arrives as several -- on
    00808 the upper one shows up at 2.86, 3.07, 3.11 and 3.26 across eight
    snapshots, and a landing shows up at 0.86. Averaging the cluster by
    explored area puts the storey where the agent actually walked, rather than
    at whichever fragment happened to be read first.
    """
    from navigation.mapping.map_store import floor_summaries, load_obstacle_maps

    seen = []
    for path in paths:
        for floor in floor_summaries(load_obstacle_maps(path)):
            if floor["explored_cells"] <= 0 or floor.get("floor_y") is None:
                continue
            seen.append((float(floor["floor_y"]), int(floor["explored_cells"])))
    if not seen:
        return []
    seen.sort()
    clusters, current = [], [seen[0]]
    for height, cells in seen[1:]:
        if height - current[-1][0] <= float(gap_m):
            current.append((height, cells))
        else:
            clusters.append(current)
            current = [(height, cells)]
    clusters.append(current)
    out = []
    for cluster in clusters:
        weight = sum(c for _, c in cluster) or 1
        out.append(sum(h * c for h, c in cluster) / weight)
    return out


def _seed_storeys(cfg, stack, paths) -> tuple:
    """Reconcile the scene-graph prior's storeys with the snapshots'.

    Returns `(added, uncorroborated)`. A storey the snapshots show and the
    stack lacks is ADDED; a stack layer no storey backs is reported, and the
    caller leaves it out of the paste. Layers the snapshots DO back are never
    moved: their rooms, containers and tracks were built at that height.

    Both halves are needed. Measured on 00808, where no single mapping episode
    visited both storeys: the scene graph kept one that started mid-staircase,
    so its storeys are 0.06 and 1.03 when the real ones are 0.06 and ~3.0.
    Adding 3.0 is not enough -- with 1.03 still in the set, the lower storey's
    stair ramp is written from 0.06 to 1.03, the flight tops out a metre up,
    and the climb dies there: 7 climbs across 3 episodes, none gaining more
    than 0.40 m.
    """
    if not bool(getattr(cfg.ycb, "seed_storeys_from_obstacle_map", False)):
        return [], []
    tol = float(getattr(cfg.ycb, "storey_seed_tol_m", 1.0) or 1.0)
    storeys = storeys_from_snapshots(paths)
    if not storeys:
        return [], []
    added = []
    for height in storeys:
        known = [float(getattr(layer, "floor_y", 0.0)) for layer in stack._layers.values()]
        if any(abs(height - other) <= tol for other in known):
            continue
        key = (max(stack._layers) + 1) if stack._layers else 0
        stack.set_height(key, height)          # `layer()` creates it
        added.append(round(height, 3))
    loose = [
        int(key) for key, layer in stack._layers.items()
        if all(abs(float(getattr(layer, "floor_y", 0.0)) - h) > tol for h in storeys)
    ]
    return added, loose


def _match_floors(stored, keys, heights, tol_m: float = 1.0, strict: bool = False):
    """Which OSG storey each of a snapshot's floors belongs to.

    BY HEIGHT when the snapshot records one per floor -- `floor_y`, the mean
    height the mapping agent stood at on that floor. That is the only way a
    snapshot which mapped ONE storey can be placed on a two-storey stack, and
    four of the six mapping episodes on 00800 are exactly that.

    Otherwise BY ORDER, which is all an older snapshot allows: the floor list
    runs bottom storey first and the stack is sorted by height. Order matching
    a snapshot with a different number of mapped storeys would put a whole
    floor on the wrong one, so for a union that case yields nothing and is
    reported instead.

    Returns `(pairs, how)` where `pairs` is [(floor, index into keys)].
    """
    if stored and all(f.get("floor_y") is not None for f in stored):
        pairs = []
        for floor in stored:
            gaps = [abs(float(floor["floor_y"]) - h) for h in heights]
            i = min(range(len(gaps)), key=gaps.__getitem__)
            if gaps[i] <= tol_m:
                pairs.append((floor, i))
        return pairs, "height"
    if strict and len(stored) != len(keys):
        return [], "order-mismatch"
    return [(f, i) for i, f in enumerate(stored[:len(keys)])], "order"


def save_obstacle_map_for_scene(cfg, agent, episode) -> Optional[str]:
    """Pass 1: keep the ASCENT obstacle-map stack this episode built.

    Only an agent that HAS one -- `navigation.AscentNavAgent` owns `_floors`
    and the episode anchor its pixels are measured from. Any other policy
    leaves this alone rather than writing an empty snapshot, because an empty
    snapshot loads silently and a missing one does not.
    """
    root = str(getattr(cfg.ycb, "obstacle_map_out", "") or "")
    if not root or not getattr(agent, "_floors", None):
        return None
    from navigation.mapping.map_store import save_obstacle_maps

    authored = authored_episode_metadata(episode)
    scene = authored_scene(episode)
    if bool(getattr(cfg.ycb, "obstacle_map_union", False)):
        # Every episode keeps its own snapshot; pass 2 unions them. No
        # best-of comparison, because none of them is being discarded.
        path = _episode_map_path(root, scene, episode)
        save_obstacle_maps(path, agent, scene=scene,
                           layout_id=str(authored.get("layout_id", "")),
                           episode_id=str(episode_tag(episode)))
        return str(path)
    path = _map_path(root, scene)
    # Keep the pass that mapped the most: the protocol runs several episodes
    # per scene and the map worth keeping is the one that saw the most of it.
    #
    # Ranked on CELLS, with storeys only as a tie-break, and both counted over
    # storeys the agent actually WALKED -- `_new_floor` allocates one the
    # moment a staircase is merely detected, so empty storeys are evidence of
    # nothing. Ranking on storeys FIRST repeats that same mistake one layer up:
    # measured on 00821, an episode that ran 215 steps over a 3.4 x 1.2 m box
    # while touching four storeys beat episodes that covered 10 x 28 m, and the
    # scene's stored map came out at 2975 cells against 30-36k for its
    # neighbours. A storey stepped onto is not a storey mapped.
    if path.exists():
        try:
            from navigation.mapping.map_store import floor_summaries, load_obstacle_maps

            previous = [f for f in floor_summaries(load_obstacle_maps(path))
                        if f["explored_cells"] > 0]
            mine = [f["obstacle"] for f in agent._floors
                    if int(f["obstacle"].explored_area.sum()) > 0]
            old = (sum(f["explored_cells"] for f in previous), len(previous))
            new = (sum(int(om.explored_area.sum()) for om in mine), len(mine))
            if old >= new:
                return str(path)
        except Exception:
            # A corrupt prior must not stop a fresh valid snapshot replacing it.
            pass
    save_obstacle_maps(path, agent, scene=scene,
                       layout_id=str(authored.get("layout_id", "")),
                       episode_id=str(episode_tag(episode)))
    return str(path)


def paste_snapshots(paths, costmaps, heights, *, overwrite: bool = False,
                    strict: Optional[bool] = None) -> dict:
    """Paste every snapshot in `paths` onto `costmaps[i]`, the storey at `heights[i]`.

    The loop `load_obstacle_map` runs, with the destination costmaps passed in
    rather than reached out of a `FloorStack`, so an OFFLINE caller -- the
    coverage audit, the map figure, the frame check -- pastes exactly the cells
    a run will and cannot drift from it. The paste was transposed once
    (`navigation/mapping/map_store._to_geometric_layout`) and every geometric
    reading taken before that was found was 8-9 m from the truth; a second
    implementation is how that returns.

    EMPTY STOREYS ARE SKIPPED before matching, and that is not a tidy-up:
    `_new_floor` allocates a storey the moment a staircase is DETECTED, so a
    real pass ends with placeholders above and below the storeys it walked.

    `strict` defaults to "a union of several snapshots is strict" -- with one
    snapshot an order match is the shipped behaviour, with several an order
    mismatch discards that snapshot rather than putting a floor on the wrong
    storey.
    """
    from navigation.mapping.map_store import (
        apply_to_costmap,
        floor_summaries,
        load_obstacle_maps,
    )

    paths = list(paths)
    if strict is None:
        strict = len(paths) > 1
    written, pairs, used, skipped, snapshots = 0, [], [], [], []
    first_blob, stored_first, all_first = None, [], []
    for path in paths:
        blob = load_obstacle_maps(path)
        all_floors = floor_summaries(blob)
        stored = [f for f in all_floors if f["explored_cells"] > 0]
        if first_blob is None:
            first_blob, stored_first, all_first = blob, stored, all_floors
        matched, how = _match_floors(stored, range(len(costmaps)), heights,
                                     strict=strict)
        snapshots.append({
            "path": str(path), "file": Path(path).name,
            "layout_id": str(blob.get("layout_id", "")),
            "episode_id": str(blob.get("episode_id", "")),
            "floors": all_floors, "matched_by": how,
        })
        if not matched:
            # A snapshot that cannot be placed is skipped, never guessed at:
            # floors matched BY ORDER need the snapshot to have mapped the same
            # NUMBER of storeys, and pasting one that did not would put a whole
            # floor on the wrong storey.
            skipped.append({"path": Path(path).name, "mapped_floors": len(stored),
                            "why": how})
            continue
        for floor, i in matched:
            # The storey a flight from here would arrive at: the one above, or
            # for the topmost the one below, since its staircase descends.
            nxt = heights[i + 1] if i + 1 < len(heights) else (
                heights[i - 1] if i > 0 else None)
            n = apply_to_costmap(blob, floor["index"], costmaps[i],
                                 overwrite=overwrite,
                                 floor_y=heights[i], next_floor_y=nxt)
            written += n
            pairs.append({"ascent_floor": floor["index"], "osg_floor": i,
                          "cells": int(n), "from": Path(path).name,
                          "matched_by": how})
        used.append(Path(path).name)
    return {
        "pasted": pairs, "skipped": skipped, "used": used,
        "cells_written": int(written), "snapshots": snapshots,
        "first_blob": first_blob, "stored_first": stored_first,
        "all_first": all_first,
    }


def paste_scene(root: str, scene: str, storeys, *, union: bool = True,
                size_m: float = 60.0, overwrite: bool = False,
                track_height: bool = True, strict: Optional[bool] = None) -> dict:
    """Every snapshot for `scene` under `root`, on one fresh costmap per storey.

    The offline entry point. Resolution is READ FROM THE SNAPSHOT rather than
    assumed: `apply_to_costmap` raises `ObstacleStoreError` on a mismatch, and
    a map directory written at another `mapping.resolution` should say so
    rather than be mis-pasted.
    """
    from navigation.mapping.map_store import floor_summaries, load_obstacle_maps
    from ..mapping.costmap import Costmap2D

    paths = _scene_map_paths(root, scene) if union else []
    if not paths:
        path = _map_path(root, scene)
        if not path.exists():
            return {"paths": [], "costmaps": [], "pasted": [], "skipped": [],
                    "used": [], "cells_written": 0, "snapshots": [],
                    "first_blob": None, "stored_first": [], "all_first": []}
        paths = [path]

    resolution = None
    for path in paths:
        for floor in floor_summaries(load_obstacle_maps(path)):
            if floor["explored_cells"] > 0:
                resolution = float(floor["resolution"])
                break
        if resolution is not None:
            break
    if resolution is None:
        return {"paths": [str(p) for p in paths], "costmaps": [], "pasted": [],
                "skipped": [], "used": [], "cells_written": 0, "snapshots": [],
                "first_blob": None, "stored_first": [], "all_first": []}

    heights = [float(h) for h in storeys]
    costmaps = []
    for _ in heights:
        costmap = Costmap2D(resolution=resolution, size_m=size_m)
        if track_height:
            # `_ramp_stair_heights` writes the tread ramp into this plane and
            # is a no-op without it, which costs the climber its waypoints.
            costmap.height = np.full(costmap.grid.shape, np.nan, dtype=np.float32)
        costmaps.append(costmap)
    result = paste_snapshots(paths, costmaps, heights, overwrite=overwrite,
                             strict=strict)
    result["paths"] = [str(p) for p in paths]
    result["costmaps"] = costmaps
    result["resolution"] = resolution
    return result


def load_obstacle_map(cfg, agent, scene: str) -> Optional[dict]:
    """Pass 2: plan over the occupancy ASCENT's navigation built.

    Floors are matched BY ORDER -- the snapshot's list runs bottom storey
    first (`agent.py` inserts at 0 on a down-stair), and the OSG stack is
    sorted by height -- because an `ObstacleMap` records no world height to
    match on. With a multi-storey `map_in` already loaded the two orders agree;
    on a fresh single-floor agent only the lowest MAPPED storey is restored,
    which is reported here rather than guessed at.

    EMPTY STOREYS ARE SKIPPED, and that is not a tidy-up. `_new_floor`
    allocates a storey the moment a staircase is DETECTED, so a real pass
    ends with placeholders above and below the storeys it actually walked --
    the first 00800 snapshot had four floors of which two held nothing, and
    floor 0 was one of them. Zipping those against the OSG stack hands a fresh
    agent an empty map and reports success, which is the one failure this
    whole path must not have.
    """
    root = str(getattr(cfg.ycb, "obstacle_map_in", "") or "")
    if not root:
        return None

    union = bool(getattr(cfg.ycb, "obstacle_map_union", False))
    paths = _scene_map_paths(root, scene) if union else []
    if not paths:
        path = _map_path(root, scene)
        if not path.exists():
            if str(getattr(cfg.ycb, "obstacle_map_out", "") or "") == root:
                return None  # pass 1 accumulating into its own output
            raise FileNotFoundError(f"no obstacle snapshot for {scene} at {path}")
        paths = [path]

    stack = agent._floor_stack
    seeded, loose = _seed_storeys(cfg, stack, paths)
    keys = sorted(
        (k for k in stack._layers if k not in set(loose)),
        key=lambda k: float(getattr(stack._layers[k], "floor_y", 0.0)),
    )
    # The storey heights, so the pasted treads can be given a height ramp --
    # without it `find_flights` sees no staircase and the waypoint climber
    # never fires (navigation/mapping/map_store.py:_ramp_stair_heights).
    heights = [float(getattr(stack._layers[k], "floor_y", 0.0)) for k in keys]

    result = paste_snapshots(
        paths, [stack._layers[k].costmap for k in keys], heights,
        overwrite=bool(getattr(cfg.ycb, "obstacle_map_overwrite", False)),
    )
    # `paste_snapshots` indexes storeys positionally, because an offline caller
    # has no FloorStack; the record names the stack's own keys.
    pairs = [dict(pair, osg_floor=int(keys[pair["osg_floor"]]))
             for pair in result["pasted"]]
    stored_first = result["stored_first"]
    landings = (drop_prior_landings(cfg, agent)
                if bool(getattr(cfg.ycb, "drop_prior_landings", False)) else [])

    return {
        "path": str(paths[0]),
        "from_layout": str((result["first_blob"] or {}).get("layout_id", "")),
        "stored_floors": len(result["all_first"]),
        "mapped_floors": len(stored_first),
        "traj_on_map": [f.get("traj_on_map") for f in stored_first],
        "matched_floors": pairs,
        "cells_written": int(result["cells_written"]),
        "unmatched_floors": max(0, len(stored_first) - len(keys)),
        "matched_by": ({p.get("matched_by") for p in pairs} and
                       sorted({str(p.get("matched_by")) for p in pairs})),
        "storeys_seeded": seeded,
        "storeys_uncorroborated": loose,
        "landings_dropped": landings,
        "union_snapshots": result["used"] if union else [],
        "union_skipped": result["skipped"],
    }


def drop_prior_landings(cfg, agent) -> list:
    """Take the prior's LANDINGS out of the storeys the agent can arrive on.

    A pass-1 graph stores a landing as a storey: 00800's holds 3.16, 1.74 and
    0.64 m for real floors at 3.16 and 0.16, 00821's holds -2.40 beside -3.47.
    Every restored layer is a level to the floor estimator, so a descent
    "arrives" when it passes 1.74 -- measured with the navmesh walking the
    stairs (outputs/mfv2/diag_climb/oracle_00800): the estimator committed the
    landing at 1.53, the climb ended there, and the agent climbed back up.

    Landings are found as the storey belief finds them (greedy by mapped area,
    a storey at least `floor.new_level_m` from every other), which is also why
    this runs AFTER the obstacle paste: the area is the pasted occupancy. Not
    by "no snapshot corroborates it" -- 00878's top storey is real and was
    never walked in pass 1. A landing's tracks are re-filed to the storey their
    height says; its level and layer go. The layer the agent starts on is
    never dropped. Returns the dropped heights.
    """
    from ..exploration.storey_belief import StoreyBelief

    stack = agent._floor_stack
    layers = {int(k): (float(layer.floor_y), layer.costmap) for k, layer in stack._layers.items()}
    belief = StoreyBelief(cfg.exploration, float(cfg.floor.new_level_m), stats={})
    storeys, rep, _ = belief.storeys(layers)
    landings = [k for k in layers if rep[k] != k and k != int(stack.current_id)]
    if not landings:
        return []
    estimator = getattr(getattr(agent, "floors", None), "estimator", None)
    layer_obj = getattr(agent, "object_layer", None)
    if layer_obj is not None:
        for track in layer_obj.tracks(include_blacklisted=True, include_proposals=True):
            if int(getattr(track, "floor_key", -1)) in landings:
                y = float(layer_obj.center_of(track)[1])
                track.floor_key = int(StoreyBelief.storey_of_height(y, storeys))
    dropped = []
    for k in landings:
        dropped.append(round(float(layers[k][0]), 2))
        # The stair edges still name the landing; their direction is its height
        # (`FloorPolicy._remembered_arrival`).
        if isinstance(getattr(stack, "dropped_levels", None), dict):
            stack.dropped_levels[int(k)] = float(layers[k][0])
        stack._layers.pop(k, None)
        if estimator is not None:
            estimator._levels.pop(k, None)
            getattr(estimator, "_samples", {}).pop(k, None)
    return dropped


def tour_target_views(cfg, env, agent, episode) -> int:
    """Pass 1: end the episode with a look at its target. Returns views used.

    The dynamic benchmark's premise is that the robot SAW the object in its
    earlier session -- Fig. 1's "during initial exploration the robot observes
    the bottle on table A" -- and pass 2 is about revising that memory. An
    exploration walk does not guarantee it: on 00800, of six 1500-step ASCENT
    episodes each prompting its own target, two never had the target in view at
    all (pitcher, cracker box). So after the episode is over and scored, the
    world model is shown the target from the layout's own goal view points (the
    manifest's, 0.8-2 m, ranked by visible pixels), up to `ycb.map_tour_views`.

    Only the object layer is fed. The occupancy is the walk's own, and each
    view's detections are filed under the storey of the view point's floor
    height, not the estimator's current storey, which a teleport would leave
    on the wrong floor. A storey the walk never reached is registered
    (`FloorPolicy.register_storey`): on 00878 the first mapping episode
    started in the basement, knew no other storey, and the tour had nowhere to
    file a single view.
    """
    n = int(getattr(cfg.ycb, "map_tour_views", 0) or 0)
    if n <= 0 or not str(cfg.ycb.map_out or ""):
        return 0
    world = getattr(agent, "world", None)
    if world is None or not hasattr(world, "object_layer"):
        return 0
    from habitat_sim.utils.common import quat_from_coeffs

    info = (getattr(episode, "info", None) or {}).get("ycb", {})
    sim = env.env.sim
    used = 0
    for view in (info.get("viewpoints") or [])[:n]:
        key = world.floors.register_storey(float(view["position"][1]))
        agent_obj = sim.get_agent(0)
        state = agent_obj.get_state()
        state.position = np.asarray(view["position"], dtype=np.float32)
        state.rotation = quat_from_coeffs(np.asarray(view["rotation"], dtype=np.float32))
        state.sensor_states = {}
        agent_obj.set_state(state, reset_sensors=True)
        env._frame_id += 1
        frame = env._to_frame(sim.get_sensor_observations())
        dets = world.detector.detect(frame.rgb)
        world.object_layer.update(frame, dets, floor_key=int(key))
        world.keyframes.add(frame)
        used += 1
    world.stats["tour_views"] = world.stats.get("tour_views", 0) + used
    return used


def save_map_for_scene(cfg, agent, episode) -> None:
    """Pass 1: keep the map this episode built, keyed by scene.

    Both artifacts come from one pass when the driving policy carries an OSG
    world model: `ascentnav` owns the `ObstacleMap` stack and its `world` owns
    the scene graph (`agent/world_model.py`), so the scene is walked once.
    """
    save_obstacle_map_for_scene(cfg, agent, episode)
    root = str(cfg.ycb.map_out or "")
    if not root:
        return
    # The snapshot is written from whatever holds the object layer: NavAgent
    # itself, or the world model a policy without one was given.
    agent = getattr(agent, "world", None) or agent
    if not hasattr(agent, "_floor_stack"):
        return
    # Small same-label fragments folded into their object before the map is
    # kept (scene_graph.ellipsoid.consolidate_max_obs; a no-op when 0).
    agent.object_layer.consolidate()
    from ..graph.map_store import load_map, save_map

    authored = authored_episode_metadata(episode)
    path = _map_path(root, authored_scene(episode))
    if path.exists():
        try:
            previous = load_map(path)
            old_grids = previous.get("_grids", {})
            old_known = sum(
                int((grid != -1).sum()) for name, grid in old_grids.items()
                if name.endswith("grid")
            )
            old_score = (
                len(previous.get("floors") or [0]), old_known,
                len(previous.get("tracks") or []),
            )
            new_score = (
                len(agent._floor_stack._layers),
                sum(int((floor.costmap.grid != -1).sum())
                    for floor in agent._floor_stack._layers.values()),
                len(list(agent.object_layer.tracks(include_blacklisted=True))),
            )
            if old_score > new_score:
                return
        except Exception:
            # A corrupt prior must not prevent a fresh valid snapshot replacing it.
            pass
    save_map(
        path,
        agent,
        scene=authored_scene(episode),
        layout_id=str(authored.get("layout_id", "")),
    )


def load_prior_map(
    cfg, agent, scene: str, *, initial_floor_y: Optional[float] = None
) -> Optional[dict]:
    """Pass 2: start from the map pass 1 built, not from an empty one."""
    root = str(cfg.ycb.map_in or "")
    if not root:
        return None
    from ..graph.map_store import apply_map, load_map

    path = _map_path(root, scene)
    if not path.exists() and str(cfg.ycb.map_out or "") == root:
        # Pass 1 accumulating into its own output: the first episode of a
        # scene has nothing to start from yet, and that is not an error.
        return None
    blob = load_map(path)
    restore_occupancy = bool(getattr(cfg.ycb, "map_in_occupancy", True))
    # Into whatever holds the object layer, as `save_map_for_scene` writes from:
    # NavAgent itself, or the world model riding beside a policy that has none.
    # Without this a pass 1 driven by `ascentnav` could not accumulate -- each
    # episode's graph only ever named its own target, the best-of rule kept one
    # of them, and 0 of the 25 multi-floor episodes had the target in the prior
    # (outputs/maps_p1500_osg).
    agent = getattr(agent, "world", None) or agent
    n = apply_map(
        agent, blob,
        max_log_odds=float(cfg.scene_graph.presence.reload_max_log_odds),
        initial_floor_y=initial_floor_y,
        restore_occupancy=restore_occupancy,
    )
    return {
        "path": str(path),
        "from_layout": str(blob.get("layout_id", "")),
        "tracks": int(n),
        "occupancy_restored": restore_occupancy,
        "schema_version": int(blob.get("schema_version", 1)),
        "floors": len(blob.get("floors") or [0]),
    }
