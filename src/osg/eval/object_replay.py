"""Replay a recording (eval/object_recorder.py) through an object layer.

`replay_episode` rebuilds the layer from a config exactly as the agents do
(`pipeline.objects.build_object_layer`), starts it from a prior, and feeds it
the recorded keyframes. With the config the recording was made under, the
result must be the recorded `final.json` to the bit (`tracks_equal`); that is
the harness's acceptance test, and every variant is measured against a replay
of the unchanged layer rather than against the online run.

Pass 1 chains episodes: each starts from the snapshot an earlier one wrote,
and `save_map_for_scene` only replaces a snapshot that its best-of score
prefers. `chain_scene` recovers which episode each prior came from by matching
the recorded priors against the recorded finals, so a variant's episodes can
be chained the same way through ITS OWN outputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .object_recorder import _tracks_blob, iter_keyframes


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _comparable(blob: dict) -> List[dict]:
    """Track records in id order, without the crop (PNG bytes are not part of
    the geometry, and the encoder is not guaranteed stable across builds)."""
    out = []
    for rec in sorted(blob.get("tracks", []), key=lambda r: int(r["id"])):
        rec = dict(rec)
        rec.pop("best_crop_png", None)
        out.append(rec)
    return out


def tracks_equal(a: dict, b: dict) -> bool:
    return _comparable(a) == _comparable(b)


def first_difference(a: dict, b: dict) -> Optional[str]:
    ra, rb = _comparable(a), _comparable(b)
    if len(ra) != len(rb):
        return f"track count {len(ra)} vs {len(rb)}"
    for x, y in zip(ra, rb):
        for key in sorted(set(x) | set(y)):
            if x.get(key) != y.get(key):
                return f"track {x.get('id')} field {key}: {str(x.get(key))[:120]} vs {str(y.get(key))[:120]}"
    return None


def load_prior_into(layer, blob: dict, *, restore: bool, max_log_odds: float = 1.5) -> None:
    """Start `layer` from `blob`.

    restore=False: `blob` is a recorded prior.json, captured AFTER apply_map
    ran, so its tracks go in verbatim. restore=True: `blob` is an episode's
    final state, and goes through `restore_tracks` as a snapshot load does."""
    from ..graph.map_store import _track_from_record, restore_tracks

    if restore:
        restore_tracks(layer, {"tracks": blob.get("tracks", []),
                               "next_track_id": blob.get("state", {}).get("next_id")},
                       max_log_odds=max_log_odds)
        return
    layer._tracks = {}
    for rec in blob.get("tracks", []):
        track = _track_from_record(rec)
        track.from_prior = bool(rec.get("from_prior", False))
        layer._tracks[track.id] = track
    layer._next_id = int(blob.get("state", {}).get("next_id", len(layer._tracks)))


def ref_layer_factory(cfg):
    """`build_object_layer`, with every observation tagged by the `ref` its
    detection carries (`obs.det_ref`): how a replay ties a track back to the
    keyframe and mask each of its observations came from."""
    from ..objects.object_layer import ObjectLayer
    from ..pipeline.objects import build_object_layer

    class RefLayer(ObjectLayer):
        @staticmethod
        def _make_observation(det, frame, K, T_cw):
            obs = ObjectLayer._make_observation(det, frame, K, T_cw)
            if obs is not None:
                obs.det_ref = getattr(det, "ref", None)
            return obs

    layer = build_object_layer(cfg)
    layer.__class__ = RefLayer
    return layer


def start_layer(
    cfg,
    episode_dir: Path,
    prior: Optional[dict] = None,
    *,
    restore: bool = False,
    layer_factory: Optional[Callable] = None,
):
    """A layer in the state the recorded episode started in, before its first
    keyframe. `prior` defaults to the episode's own recorded prior.json."""
    from ..pipeline.objects import build_object_layer

    episode_dir = Path(episode_dir)
    recorded_prior = read_json(episode_dir / "prior.json")
    layer = (layer_factory or build_object_layer)(cfg)
    state = recorded_prior.get("state", {})
    layer.set_target(state.get("target_label", ""))
    layer.keep_cloud_labels = set(state.get("keep_cloud_labels", []))
    layer._disabled_pts = [(np.asarray(p[:2], dtype=float), str(p[2]))
                           for p in state.get("disabled_pts", [])]
    if prior is None:
        load_prior_into(layer, recorded_prior, restore=False)
        layer._next_id = int(state.get("next_id", layer._next_id))
    else:
        load_prior_into(layer, prior, restore=restore,
                        max_log_odds=float(cfg.scene_graph.presence.reload_max_log_odds))
    layer._next_prop_id = int(state.get("next_prop_id", layer._next_prop_id))
    return layer


def replay_episode(
    cfg,
    episode_dir: Path,
    prior: Optional[dict] = None,
    *,
    restore: bool = False,
    layer_factory: Optional[Callable] = None,
    on_keyframe: Optional[Callable] = None,
    on_start: Optional[Callable] = None,
):
    """Feed one recorded episode through a fresh layer; returns the layer.

    `prior` defaults to the episode's own recorded prior.json (restore=False).
    `layer_factory(cfg)` lets a caller substitute an instrumented subclass."""
    layer = start_layer(cfg, episode_dir, prior, restore=restore, layer_factory=layer_factory)
    if on_start is not None:
        on_start(layer)
    for frame, dets, floor_key in iter_keyframes(episode_dir):
        layer.update(frame, dets, floor_key=floor_key)
        if on_keyframe is not None:
            on_keyframe(layer, frame, dets, floor_key)
    return layer


def final_blob(layer) -> dict:
    return _tracks_blob(layer)


# ------------------------------------------------------------------ chaining


def scene_episodes(root: Path, scene: str) -> List[Path]:
    """This scene's recorded episodes in the order they ran."""
    eps = []
    for d in sorted(Path(root).iterdir()):
        if not (d / "prior.json").exists():
            continue
        meta = read_json(d / "prior.json").get("meta", {})
        if meta.get("scene") == scene:
            eps.append((float(meta.get("wall_start", 0.0)), d.name, d))
    return [d for _, _, d in sorted(eps)]


def chain_scene(cfg, episodes: List[Path]) -> Dict[str, Optional[str]]:
    """{episode name: name of the episode whose final state its prior is, or
    None for an empty prior}. Raises if a non-empty prior matches no earlier
    final -- the recording then does not describe a chain this can replay."""
    from ..pipeline.objects import build_object_layer

    parents: Dict[str, Optional[str]] = {}
    finals: List[tuple] = []
    for ep in episodes:
        prior = read_json(ep / "prior.json")
        if not prior.get("tracks"):
            parents[ep.name] = None
        else:
            match = None
            for name, fin in reversed(finals):
                probe = build_object_layer(cfg)
                load_prior_into(probe, fin, restore=True,
                                max_log_odds=float(cfg.scene_graph.presence.reload_max_log_odds))
                if tracks_equal(_tracks_blob(probe), prior):
                    match = name
                    break
            if match is None:
                raise ValueError(f"{ep.name}: prior matches no earlier episode's final state")
            parents[ep.name] = match
        if (ep / "final.json").exists():
            finals.append((ep.name, read_json(ep / "final.json")))
    return parents
