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

Single storey only. Snapshotting a FloorStack means several costmaps, several
room segmentations and a floor estimator whose ids must survive; that is real
work and the YCB benchmark scenes are one floor, so `save_map` refuses rather
than writing a snapshot that would quietly lose a storey.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..objects.association import Observation, ObjectTrack
from ..objects.ellipsoid import Ellipsoid
from ..objects.presence import PresenceState

SCHEMA_VERSION = 1


class MapStoreError(RuntimeError):
    """A snapshot cannot be written or read."""


def _f(values) -> List[float]:
    return [float(v) for v in np.asarray(values, dtype=float).ravel()]


def _track_record(track: ObjectTrack) -> Dict[str, Any]:
    ell = track.ellipsoid
    return {
        "id": int(track.id),
        "label": str(track.label),
        "center": _f(ell.center),
        "axes": _f(ell.axes),
        "R": _f(ell.R),
        "best_score": float(track.best_score),
        "best_bbox_px": float(track.best_bbox_px),
        "best_cam_xy": None if track.best_cam_xy is None else _f(track.best_cam_xy),
        "first_cam_xy": None if track.first_cam_xy is None else _f(track.first_cam_xy),
        "blacklisted": bool(track.blacklisted),
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
    track.best_bbox_px = float(rec.get("best_bbox_px", 0.0))
    if rec.get("best_cam_xy") is not None:
        track.best_cam_xy = np.asarray(rec["best_cam_xy"], dtype=float)
    if rec.get("first_cam_xy") is not None:
        track.first_cam_xy = np.asarray(rec["first_cam_xy"], dtype=float)
    track.blacklisted = bool(rec.get("blacklisted", False))
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
    if len(stack._layers) > 1:
        raise MapStoreError(
            "map snapshots are single-storey; this agent mapped "
            f"{len(stack._layers)} floors and saving would silently drop all but one"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    costmap = agent.costmap
    grids = {"grid": costmap.grid, "origin": costmap.origin}
    if costmap.height is not None:
        grids["height"] = costmap.height
    if costmap.stair_mask is not None:
        grids["stair_mask"] = costmap.stair_mask
    room_labels = agent._room_labels
    if room_labels is not None:
        grids["room_labels"] = room_labels
    npz_path = path.with_suffix(".npz")
    np.savez_compressed(npz_path, **grids)

    tracks = list(agent.object_layer.tracks(include_blacklisted=True))
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "scene": scene,
                "layout_id": layout_id,
                "resolution": float(costmap.resolution),
                "grids": npz_path.name,
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
    if int(blob.get("schema_version", -1)) != SCHEMA_VERSION:
        raise MapStoreError(
            f"{path}: snapshot schema {blob.get('schema_version')} != {SCHEMA_VERSION}"
        )
    grids_name = blob.get("grids")
    grids = {}
    if grids_name:
        with np.load(path.parent / grids_name) as data:
            grids = {k: data[k] for k in data.files}
    blob["_grids"] = grids
    return blob


PRIOR_SESSION_OFFSET = 1_000_000


def apply_map(agent, blob: Dict[str, Any], *, max_log_odds: float = 1.5) -> int:
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
        track.presence.log_odds = min(float(track.presence.log_odds), float(max_log_odds))
        for obs in track.observations:
            obs.frame_id = int(obs.frame_id) - PRIOR_SESSION_OFFSET
        layer._tracks[track.id] = track
    layer._next_id = int(blob.get("next_track_id", max(layer._tracks, default=-1) + 1))

    grids = blob.get("_grids", {})
    costmap = agent.costmap
    snap_res = float(blob.get("resolution", costmap.resolution))
    if abs(snap_res - float(costmap.resolution)) > 1e-9:
        raise MapStoreError(
            f"snapshot resolution {snap_res} != this costmap's {costmap.resolution}; "
            "both passes must share mapping.resolution_m"
        )
    if "grid" in grids:
        # The costmap GROWS as the agent explores, so a snapshot is routinely a
        # different shape from the fresh 20 m grid it is loaded into. Adopt the
        # stored extent wholesale -- with the origin below it describes the same
        # world, and refusing here would reject every real mapping run.
        costmap.grid = np.array(grids["grid"], dtype=costmap.grid.dtype)
    if "origin" in grids:
        costmap.origin = np.asarray(grids["origin"], dtype=float)
    if costmap.height is not None:
        costmap.height = (
            np.array(grids["height"], dtype=np.float32)
            if "height" in grids
            else np.full(costmap.grid.shape, np.nan, dtype=np.float32)
        )
    if "stair_mask" in grids:
        costmap.stair_mask = np.array(grids["stair_mask"])
    if "room_labels" in grids:
        agent._room_labels = grids["room_labels"]

    # Derived structure is rebuilt, never restored: a snapshot must not freeze
    # yesterday's container rule into today's run.
    if agent._room_labels is not None:
        agent.scene_graph.rebuild(agent._room_labels, costmap, layer, floors=None)
    return len(layer._tracks)
