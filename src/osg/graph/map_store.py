"""Persist a built map so a later episode can start from it.

The dynamic-scene benchmark is two passes over the same scene: explore the
STATIC layout and keep the map, then move the objects and navigate the moved
world *with that map*. The staleness is the experiment -- an agent that rebuilds
from scratch is never wrong about anything, and measures nothing
(docs/DYNAMIC_SCENES.md, Phase 2).

What is stored is the map's EVIDENCE, not its conclusions: object tracks with
their ellipsoids and presence beliefs, the occupancy grid, and the room
segmentation. The scene graph itself -- rooms, containers, object views -- is
derived, so it is rebuilt on load rather than serialised; that way a change to
the container rule cannot be silently frozen into an old snapshot.

Schema v2 stores every floor independently, including stable floor keys,
height order, stair evidence and connectivity. Schema v1 remains readable as
the single floor with key 0.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..objects.association import Observation, ObjectTrack
from ..objects.ellipsoid import Ellipsoid
from ..objects.presence import PresenceState

SCHEMA_VERSION = 2


class MapStoreError(RuntimeError):
    """A snapshot cannot be written or read."""


# Longest side of a stored identity crop. 128 px is well above what the VLM
# needs to name a boxed object and keeps a 600-track snapshot in the low MB.
CROP_MAX_PX = 128


def _encode_crop(crop) -> Optional[str]:
    """PNG + base64, or None. Never fail a snapshot over a thumbnail."""
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    try:
        import base64

        import cv2
        import numpy as np

        img = np.asarray(crop)
        if img.ndim != 3 or img.shape[2] < 3:
            return None
        longest = max(img.shape[:2])
        if longest > CROP_MAX_PX:
            scale = CROP_MAX_PX / float(longest)
            img = cv2.resize(img, (max(1, int(img.shape[1] * scale)),
                                   max(1, int(img.shape[0] * scale))),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", img[..., ::-1])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
    except Exception:  # noqa: BLE001 - a thumbnail is never worth a failed save
        return None


def _decode_crop(blob) -> Optional["np.ndarray"]:
    if not blob:
        return None
    try:
        import base64

        import cv2
        import numpy as np

        raw = np.frombuffer(base64.b64decode(blob), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        return None if img is None else img[..., ::-1].copy()
    except Exception:  # noqa: BLE001
        return None


def _f(values) -> List[float]:
    return [float(v) for v in np.asarray(values, dtype=float).ravel()]


def _track_record(track: ObjectTrack) -> Dict[str, Any]:
    ell = track.ellipsoid
    return {
        "id": int(track.id),
        "label": str(track.label),
        "floor_key": int(getattr(track, "floor_key", 0)),
        "center": _f(ell.center),
        "axes": _f(ell.axes),
        "R": _f(ell.R),
        "best_score": float(track.best_score),
        "best_bbox_px": float(track.best_bbox_px),
        "best_cam_xy": None if track.best_cam_xy is None else _f(track.best_cam_xy),
        "first_cam_xy": None if track.first_cam_xy is None else _f(track.first_cam_xy),
        "blacklisted": bool(track.blacklisted),
        # The candidate verifier's only evidence about a track's IDENTITY is a
        # picture of it, and a track restored from a snapshot had neither the
        # frame nor the crop -- so `verify()` fell through to `_ask(None)`, which
        # fails open, and the VLM gate was a silent no-op on exactly the tracks
        # that cause most false-positive commits (62 of 72 in the 96-episode
        # run). The full frame is far too large to store per track; the crop is
        # what `verify_crop` wants anyway, and capped at CROP_MAX_PX it costs a
        # few KB.
        "best_crop_png": _encode_crop(track.best_crop),
        "evidence": float(track.evidence),
        "refined_at_obs": int(track.refined_at_obs),
        "linked_ids": sorted(int(i) for i in track.linked_ids),
        "presence": {
            "log_odds": float(track.presence.log_odds),
            "last_seen_kf": int(track.presence.last_seen_kf),
            "last_absent_kf": track.presence.last_absent_kf,
            "n_expected": int(track.presence.n_expected),
            "n_missed": int(track.presence.n_missed),
        },
        # Observations are the track's evidence: n_obs gates candidate
        # selection and the Wasserstein refiner reads them, so a snapshot that
        # dropped them would load tracks the pipeline then ignores.
        "observations": [
            {
                "frame_id": int(o.frame_id),
                "mu": _f(o.mu),
                "cov": _f(o.cov),
                "K": _f(o.K),
                "T_cw": _f(o.T_cw),
                "mean_depth": float(o.mean_depth),
            }
            for o in track.observations
        ],
    }


def _track_from_record(rec: Dict[str, Any]) -> ObjectTrack:
    track = ObjectTrack(
        id=int(rec["id"]),
        label=str(rec["label"]),
        ellipsoid=Ellipsoid(
            center=np.asarray(rec["center"], dtype=float),
            axes=np.asarray(rec["axes"], dtype=float),
            R=np.asarray(rec["R"], dtype=float).reshape(3, 3),
        ),
    )
    track.best_score = float(rec.get("best_score", 0.0))
    track.floor_key = int(rec.get("floor_key", 0))
    track.best_bbox_px = float(rec.get("best_bbox_px", 0.0))
    if rec.get("best_cam_xy") is not None:
        track.best_cam_xy = np.asarray(rec["best_cam_xy"], dtype=float)
    if rec.get("first_cam_xy") is not None:
        track.first_cam_xy = np.asarray(rec["first_cam_xy"], dtype=float)
    track.blacklisted = bool(rec.get("blacklisted", False))
    track.best_crop = _decode_crop(rec.get("best_crop_png"))
    track.evidence = float(rec.get("evidence", 0.0))
    track.refined_at_obs = int(rec.get("refined_at_obs", 0))
    track.linked_ids = {int(i) for i in rec.get("linked_ids", [])}
    p = rec.get("presence", {})
    track.presence = PresenceState(
        log_odds=float(p.get("log_odds", 1.5)),
        last_seen_kf=int(p.get("last_seen_kf", -1)),
        last_absent_kf=p.get("last_absent_kf"),
        n_expected=int(p.get("n_expected", 0)),
        n_missed=int(p.get("n_missed", 0)),
    )
    track.observations = [
        Observation(
            frame_id=int(o["frame_id"]),
            mu=np.asarray(o["mu"], dtype=float),
            cov=np.asarray(o["cov"], dtype=float).reshape(2, 2),
            K=np.asarray(o["K"], dtype=float).reshape(3, 3),
            T_cw=np.asarray(o["T_cw"], dtype=float).reshape(4, 4),
            mean_depth=float(o["mean_depth"]),
        )
        for o in rec.get("observations", [])
    ]
    return track


def save_map(path: Path, agent, *, scene: str = "", layout_id: str = "") -> Path:
    """Write the agent's map. `path` is the JSON; grids go beside it as .npz."""
    stack = agent._floor_stack
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grids = {}
    floors = []
    for key, layer in sorted(stack._layers.items()):
        if not hasattr(layer, "costmap"):
            raise MapStoreError(f"floor {key} has no costmap")
        costmap = layer.costmap
        prefix = f"floor_{int(key)}_"
        grids[prefix + "grid"] = costmap.grid
        grids[prefix + "origin"] = costmap.origin
        for name in ("height", "stair_mask"):
            value = getattr(costmap, name, None)
            if value is not None:
                grids[prefix + name] = value
        for name in ("room_labels", "up_stair_hits", "down_stair_hits", "disabled_stair"):
            value = getattr(layer, name, None)
            if value is not None:
                grids[prefix + name] = value
        value_map = getattr(layer, "value_map", None)
        if value_map is not None:
            # ValueMap2D owns arrays in the same grid frame as this floor. They
            # are optional, but when present they are evidence just like room
            # and stair arrays and must not disappear on a static-map reload.
            for name in ("value", "conf"):
                value = getattr(value_map, name, None)
                if value is not None:
                    grids[prefix + name] = value
        floors.append({
            "key": int(key),
            "height_y": float(getattr(layer, "floor_y", 0.0)),
            "resolution": float(costmap.resolution),
            "prefix": prefix,
            "first_step": int(getattr(layer, "first_step", 0)),
            "entry_xy": None if getattr(layer, "entry_xy", None) is None else _f(layer.entry_xy),
            "explored": bool(getattr(layer, "explored", False)),
            "visits": int(getattr(layer, "visits", 1)),
            "value_map": value_map is not None,
            "value_n_updates": int(getattr(value_map, "n_updates", 0))
            if value_map is not None else 0,
        })
    npz_path = path.with_suffix(".npz")
    np.savez_compressed(npz_path, **grids)

    tracks = list(agent.object_layer.tracks(include_blacklisted=True))
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "scene": scene,
                "layout_id": layout_id,
                "resolution": float(floors[0]["resolution"] if floors else agent.costmap.resolution),
                "grids": npz_path.name,
                "floors": floors,
                "current_floor_key": int(getattr(stack, "current_id", 0)),
                "connectivity": [
                    {
                        "from_floor": int(edge.from_floor),
                        "to_floor": int(edge.to_floor),
                        "entry_xy": None if edge.entry_xy is None else _f(edge.entry_xy),
                        "exit_xy": None if edge.exit_xy is None else _f(edge.exit_xy),
                        "step": int(edge.step),
                        "n_traversals": int(edge.n_traversals),
                    }
                    for edge in getattr(stack, "stair_edges", [])
                ],
                "next_track_id": int(agent.object_layer._next_id),
                "tracks": [_track_record(t) for t in tracks],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return path


def load_map(path: Path) -> Dict[str, Any]:
    path = Path(path)
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MapStoreError(f"no map snapshot at {path}") from exc
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise MapStoreError(f"{path}: corrupt snapshot metadata") from exc
    version = int(blob.get("schema_version", -1))
    if version not in (1, SCHEMA_VERSION):
        raise MapStoreError(
            f"{path}: unsupported snapshot schema {blob.get('schema_version')}"
        )
    grids_name = blob.get("grids")
    grids = {}
    if grids_name:
        try:
            with np.load(path.parent / grids_name) as data:
                grids = {k: data[k] for k in data.files}
        except (OSError, ValueError) as exc:
            raise MapStoreError(f"{path}: cannot read grid archive {grids_name}") from exc
    if version == SCHEMA_VERSION:
        floors = blob.get("floors")
        if not isinstance(floors, list) or not floors:
            raise MapStoreError(f"{path}: schema v2 snapshot contains no floors")
        try:
            prefixes = [str(floor["prefix"]) for floor in floors]
            keys = [int(floor["key"]) for floor in floors]
        except (KeyError, TypeError, ValueError) as exc:
            raise MapStoreError(f"{path}: corrupt floor metadata") from exc
        if len(keys) != len(set(keys)):
            raise MapStoreError(f"{path}: duplicate stable floor keys")
        for prefix in prefixes:
            if prefix + "grid" not in grids or prefix + "origin" not in grids:
                raise MapStoreError(f"{path}: missing arrays for floor prefix {prefix}")
    elif "grid" not in grids or "origin" not in grids:
        raise MapStoreError(f"{path}: schema v1 snapshot has no occupancy grid")
    blob["_grids"] = grids
    return blob


PRIOR_SESSION_OFFSET = 1_000_000


def apply_map(
    agent,
    blob: Dict[str, Any],
    *,
    max_log_odds: float = 1.5,
    initial_floor_y: Optional[float] = None,
) -> int:
    """Load a snapshot into a freshly constructed agent. Returns track count.

    Called after NavAgent.__init__ (which resets), before the first act().

    Three things are deliberately NOT carried across intact:

    `max_log_odds` caps how sure a restored belief may be. The map was built in
    another session; the world had every opportunity to change in between, and a
    belief that saturated at p=0.998 then would need seven clean misses to
    unwind now -- so the agent commits to a stale goal on step 1 and the episode
    is over before the evidence arrives. Capping is the survival channel of the
    presence filter collapsed into one honest number: time passed, so believe
    less. Disbelief is left alone; an object already known to be gone has not
    become more likely by sitting in a file.

    Observation frame ids are shifted into negative territory so nothing can
    mistake a previous session's frames for this one's -- in particular
    `relink`, which must not merge a fresh track with a track last seen before
    the world changed.

    The blacklist is cleared. It is an EPISODE-scoped device -- "this attempt
    already tried that candidate, choose another" -- and persisting it turns a
    rejection taken under one episode's evidence into a permanent, unrecoverable
    strike-off in every later session. Measured: one track in a 587-track map was
    saved blacklisted, and it was the pitcher's only correct track (0.00 m from
    the authored pose, score 0.68, belief 0.82), so every pitcher episode in the
    next benchmark was unwinnable before it started. This is the same mistake as
    blacklisting on absence, one layer down: no state may be absorbing, and
    `min_presence` over a belief is the recoverable way to keep a disproved track
    out of the candidate list.
    """
    layer = agent.object_layer
    layer._tracks = {}
    for rec in blob.get("tracks", []):
        track = _track_from_record(rec)
        track.blacklisted = False
        track.identity_rejections = 0  # episode-scoped, for the same reason
        track.presence.log_odds = min(float(track.presence.log_odds), float(max_log_odds))
        for obs in track.observations:
            obs.frame_id = int(obs.frame_id) - PRIOR_SESSION_OFFSET
        layer._tracks[track.id] = track
    layer._next_id = int(blob.get("next_track_id", max(layer._tracks, default=-1) + 1))

    grids = blob.get("_grids", {})
    version = int(blob.get("schema_version", 1))

    def restore_grid(costmap, prefix: str, room_owner) -> None:
        snap_res = float(blob.get("resolution", costmap.resolution))
        if version >= 2:
            floor_meta = next(f for f in blob["floors"] if f["prefix"] == prefix)
            snap_res = float(floor_meta["resolution"])
        if abs(snap_res - float(costmap.resolution)) > 1e-9:
            raise MapStoreError(
                f"snapshot resolution {snap_res} != this costmap's {costmap.resolution}; "
                "both passes must share mapping.resolution_m"
            )
        def get(name):
            return grids.get(prefix + name)
        grid = get("grid")
        if grid is not None:
            costmap.grid = np.array(grid, dtype=costmap.grid.dtype)
        origin = get("origin")
        if origin is not None:
            costmap.origin = np.asarray(origin, dtype=float)
        if costmap.height is not None:
            height = get("height")
            costmap.height = (
                np.array(height, dtype=np.float32) if height is not None
                else np.full(costmap.grid.shape, np.nan, dtype=np.float32)
            )
        stair_mask = get("stair_mask")
        if stair_mask is not None:
            costmap.stair_mask = np.array(stair_mask, dtype=bool)
        for name, dtype in (
            ("room_labels", np.int32), ("up_stair_hits", np.int16),
            ("down_stair_hits", np.int16), ("disabled_stair", bool),
        ):
            value = get(name)
            if value is not None:
                setattr(room_owner, name, np.array(value, dtype=dtype))
        # Restore semantic value evidence when the freshly constructed agent
        # has a value map. If a caller supplied a minimal agent without one,
        # create the lightweight map from the snapshot arrays (no model is
        # loaded here); this keeps v2 snapshots lossless without requiring a
        # CLIP dependency for ordinary map consumers.
        value = get("value")
        conf = get("conf")
        if value is not None or conf is not None:
            value_map = getattr(room_owner, "value_map", None)
            if value_map is None and value is not None and conf is not None:
                try:
                    from ..mapping.value_map import ValueMap2D

                    value_map = ValueMap2D(costmap)
                    room_owner.value_map = value_map
                except Exception:  # pragma: no cover - optional dependency path
                    value_map = None
            if value_map is not None:
                if value is not None:
                    value_map.value = np.asarray(value, dtype=np.float32).copy()
                if conf is not None:
                    value_map.conf = np.asarray(conf, dtype=np.float32).copy()
                if version >= 2:
                    meta = next(
                        f for f in blob["floors"] if f["prefix"] == prefix
                    )
                    value_map.n_updates = int(meta.get("value_n_updates", 0))

    stack = agent._floor_stack
    if version == 1:
        costmap = agent.costmap
        # v1 used unprefixed array names and is defined as stable floor 0.
        restore_grid(costmap, "", stack.current)
        if hasattr(stack, "current_id"):
            stack.current_id = 0
        if hasattr(stack.current, "floor_y") and initial_floor_y is not None:
            stack.current.floor_y = float(initial_floor_y)
    else:
        floor_meta = list(blob.get("floors") or [])
        if not floor_meta:
            raise MapStoreError("schema v2 snapshot contains no floors")
        if hasattr(stack, "layer") and callable(stack.layer):
            stack._layers = {}
            for meta in floor_meta:
                key = int(meta["key"])
                floor_layer = stack.layer(key)
                floor_layer.floor_y = float(meta["height_y"])
                floor_layer.first_step = int(meta.get("first_step", 0))
                floor_layer.entry_xy = (
                    None if meta.get("entry_xy") is None
                    else np.asarray(meta["entry_xy"], dtype=float)
                )
                floor_layer.explored = bool(meta.get("explored", False))
                floor_layer.visits = int(meta.get("visits", 1))
                restore_grid(floor_layer.costmap, str(meta["prefix"]), floor_layer)
        else:
            if len(floor_meta) != 1 or int(floor_meta[0]["key"]) != 0:
                raise MapStoreError("target agent cannot restore multiple floors")
            restore_grid(agent.costmap, str(floor_meta[0]["prefix"]), stack.current)
        # Select by the episode's first observed floor height, never by the
        # floor that happened to be active when the snapshot was written.
        chosen = min(
            floor_meta,
            key=lambda f: abs(float(f["height_y"]) - float(
                initial_floor_y if initial_floor_y is not None else 0.0
            )),
        ) if initial_floor_y is not None else min(floor_meta, key=lambda f: float(f["height_y"]))
        if hasattr(stack, "current_id"):
            stack.current_id = int(chosen["key"])
        if hasattr(stack, "stair_edges"):
            from ..mapping.floor_stack import StairEdge
            stack.stair_edges = [
                StairEdge(
                    from_floor=int(edge["from_floor"]), to_floor=int(edge["to_floor"]),
                    entry_xy=None if edge.get("entry_xy") is None else np.asarray(edge["entry_xy"], float),
                    exit_xy=None if edge.get("exit_xy") is None else np.asarray(edge["exit_xy"], float),
                    step=int(edge.get("step", 0)), n_traversals=int(edge.get("n_traversals", 1)),
                )
                for edge in blob.get("connectivity", [])
            ]

    # FloorPolicy owns both the persistent map stack and the online height
    # estimator. Restore them as one state: otherwise the first live frame
    # bootstraps estimator floor 0 and silently detaches a restored stable key
    # such as 4 or 9 from the storey it names.
    floor_policy = getattr(agent, "floors", None)
    estimator = getattr(floor_policy, "estimator", None)
    if estimator is not None:
        if version == 1:
            height = float(initial_floor_y if initial_floor_y is not None else 0.0)
            heights = {0: height}
        else:
            heights = {
                int(meta["key"]): float(meta["height_y"])
                for meta in floor_meta
            }
        estimator._levels = dict(heights)
        estimator._samples = {key: [height] for key, height in heights.items()}
        estimator._next_id = max(heights, default=-1) + 1
        estimator.current = int(getattr(stack, "current_id", 0))
        floor_policy._floor_y = float(heights[estimator.current])

    costmap = agent.costmap

    # Derived structure is rebuilt, never restored: a snapshot must not freeze
    # yesterday's container rule into today's run.
    if version >= 2 and hasattr(stack, "items"):
        for key, floor_layer in stack.items():
            if floor_layer.room_labels is not None:
                agent.scene_graph.rebuild_floor(
                    floor_layer.room_labels, floor_layer.costmap, layer,
                    floor_key=key, floor_height=floor_layer.floor_y,
                )
    elif agent._room_labels is not None:
        agent.scene_graph.rebuild(agent._room_labels, costmap, layer, floors=None)
    return len(layer._tracks)
