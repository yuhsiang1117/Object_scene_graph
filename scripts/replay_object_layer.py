#!/usr/bin/env python3
"""Replay a recorded run's object-layer input under any config.

    # V0: the recording's own config -- must reproduce every final.json exactly
    python scripts/replay_object_layer.py +experiment=mf5_mapping_both \\
        +replay.root=outputs/objrec_v2 +replay.out=outputs/objreplay/V0 +replay.check=true

    # a variant: any scene_graph override
    python scripts/replay_object_layer.py +experiment=mf5_mapping_both \\
        +replay.root=outputs/objrec_v2 +replay.out=outputs/objreplay/V1 \\
        scene_graph.ellipsoid.init=points

Episodes are replayed scene by scene in the order they ran, each started from
the REPLAYED final state of the episode its recorded prior came from
(eval/object_replay.chain_scene), so a variant's prior map is its own.
`+replay.stride=n` keeps every n-th keyframe of a scene's episodes in order.

Writes `<out>/<episode>.json`: the layer's final tracks (map_store's record),
each with `det_refs` -- the (keyframe, detection) of every observation, which
is how scripts/analyze_ellipsoids.py ties a track to ground truth -- plus the
funnel and per-keyframe update time. `+replay.snapshots=<dir>` additionally
writes, per scene, a copy of the recorded run's pass-1 snapshot
(`+replay.snapshot_src`) with its tracks replaced by the replay's last state.

`+replay.target_log=true` additionally writes `<out>/<episode>.targets.jsonl`:
one line per keyframe that leaves the layer holding a track of the episode's
target, with each such track's reported centre and size AT THAT KEYFRAME.
The final state says where a track ended up; a stop point is decided by where
it was when the agent committed (scripts/eval_stop_geometry.py).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig

from osg.core.config import register_configs

register_configs()


def _target_rows(layer, want: str) -> list:
    """The target's tracks as the map reports them now (objects/linking.py)."""
    from osg.core.labels import normalize_label
    from osg.objects.linking import reported_geometry

    rows = []
    for t in layer._tracks.values():
        if normalize_label(t.label) != want or t.blacklisted:
            continue
        g = reported_geometry(t)
        rows.append({
            "id": int(t.id),
            "center": [round(float(v), 4) for v in layer.center_of(t)],
            "geom_center": [round(float(v), 4) for v in g.center],
            "axes": [round(float(v), 4) for v in g.axes],
            "fused": getattr(t, "extent", None) is not None,
            "n_obs": int(t.n_obs),
            "p": round(float(t.presence.p), 4),
            "evidence": round(float(t.evidence), 3),
            "best_score": round(float(t.best_score), 3),
        })
    return rows


def _snapshot_source(snapshot: dict, eps) -> str:
    from osg.eval.object_replay import _comparable, read_json

    def strip(blob):
        recs = [r for r in _comparable(blob) if not r.get("proposal_only")]
        for r in recs:
            r.pop("proposal_only", None)
            r.pop("from_prior", None)
        return recs

    want = strip(snapshot)
    for ep in reversed(eps):
        if strip(read_json(ep / "final.json")) == want:
            return ep.name
    raise SystemExit("the pass-1 snapshot matches no recorded episode's final state")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from osg.core.labels import normalize_label
    from osg.eval import object_recorder as rec_mod
    from osg.eval.object_replay import (
        chain_scene, final_blob, first_difference, read_json, ref_layer_factory, scene_episodes,
    )

    rp = cfg.get("replay", {})
    root = Path(str(rp.get("root")))
    out = Path(str(rp.get("out")))
    out.mkdir(parents=True, exist_ok=True)
    check = bool(rp.get("check", False))
    target_log = bool(rp.get("target_log", False))
    want_scenes = [str(s) for s in rp.get("scenes", [])]

    scenes = sorted({read_json(d / "prior.json").get("meta", {}).get("scene")
                     for d in root.iterdir() if (d / "prior.json").exists()})
    if want_scenes:
        scenes = [s for s in scenes if s in want_scenes]

    # Tag each detection with its (keyframe index, detection index) as it is
    # loaded, so RefLayer can carry it onto the observation.
    orig_iter = rec_mod.iter_keyframes
    # Keep every n-th keyframe of a scene's episodes taken in order, the frames
    # baselines/conceptgraphs/export_recording.py exports at the same stride.
    stride = int(rp.get("stride", 1))
    seen = [0]

    def tagged_iter(episode_dir):
        name = Path(episode_dir).name
        for k, (frame, dets, fk) in enumerate(orig_iter(episode_dir)):
            seen[0] += 1
            if (seen[0] - 1) % stride:
                continue
            for i, det in enumerate(dets):
                det.ref = (name, k, i)
            yield frame, dets, fk

    import osg.eval.object_replay as replay_mod
    replay_mod.iter_keyframes = tagged_iter

    failures = 0
    for scene in scenes:
        eps = scene_episodes(root, scene)
        seen[0] = 0
        parents = chain_scene(cfg, eps)
        finals = {}
        all_refs = {}
        for ep in eps:
            parent = parents[ep.name]
            meta = read_json(ep / "prior.json").get("meta", {})
            want = normalize_label(str(meta.get("target", "")))
            times = []
            target_lines = []
            t0 = time.perf_counter()
            last = [t0]

            def on_kf(layer, frame, dets, fk):
                now = time.perf_counter()
                times.append(now - last[0])   # load + update of this keyframe
                if target_log:
                    rows = _target_rows(layer, want)
                    if rows:
                        target_lines.append(json.dumps(
                            {"kf": len(times) - 1, "frame_id": int(frame.frame_id), "tracks": rows}))
                    now = time.perf_counter()     # the log is not the layer's time
                last[0] = now

            def on_start(layer, parent=parent):
                # A restored track's observations are the parent episode's,
                # in the same order: give them back their references.
                if parent is None:
                    return
                for t in layer._tracks.values():
                    refs = all_refs[parent].get(str(t.id), [])
                    for o, r in zip(t.observations, refs):
                        o.det_ref = None if r is None else tuple(r)

            layer = replay_mod.replay_episode(
                cfg, ep,
                prior=None if parent is None else finals[parent],
                restore=parent is not None,
                layer_factory=ref_layer_factory,
                on_keyframe=on_kf,
                on_start=on_start,
            )
            if target_log:
                (out / f"{ep.name}.targets.jsonl").write_text("".join(x + "\n" for x in target_lines))
            recorded_state = final_blob(layer)   # what final.json holds: before the save
            blob = recorded_state
            # Pass 1 saved this episode's map: consolidate as
            # prior_map.save_map_for_scene does (a no-op unless enabled), and
            # the next episode starts from the consolidated state.
            if meta.get("map_out") and layer.consolidate():
                blob = final_blob(layer)
            finals[ep.name] = blob
            refs = {str(t.id): [None if getattr(o, "det_ref", None) is None else list(o.det_ref)
                                for o in t.observations] for t in layer._tracks.values()}
            all_refs[ep.name] = refs
            record = dict(blob, meta=meta, parent=parent, det_refs=refs,
                          kf_seconds=[round(x, 5) for x in times],
                          wall_s=round(time.perf_counter() - t0, 2))
            (out / f"{ep.name}.json").write_text(json.dumps(record))
            msg = (f"{scene} {ep.name}: {len(layer._tracks)} tracks, {len(times)} kf, "
                   f"{record['wall_s']} s")
            if check:
                diff = first_difference(recorded_state, read_json(ep / "final.json"))
                msg += "  IDENTICAL" if diff is None else f"  DIFFERS: {diff}"
                failures += diff is not None
            print(msg, flush=True)
        snaps = str(rp.get("snapshots", "") or "")
        if snaps and eps:
            src = Path(str(rp.get("snapshot_src"))) / f"{scene}.json"
            blob = json.loads(src.read_text())
            # The snapshot on disk is whichever episode the best-of rule kept
            # last: the one whose recorded final state it holds.
            source = _snapshot_source(blob, eps)
            final = finals[source]
            blob["tracks"] = [t for t in final["tracks"] if not t.get("proposal_only")]
            for t in blob["tracks"]:
                t.pop("proposal_only", None)
                t.pop("from_prior", None)
            blob["next_track_id"] = int(final["state"]["next_id"])
            Path(snaps).mkdir(parents=True, exist_ok=True)
            (Path(snaps) / f"{scene}.json").write_text(json.dumps(blob))
            print(f"{scene}: snapshot with {len(blob['tracks'])} tracks -> {snaps}", flush=True)
    if check and failures:
        raise SystemExit(f"{failures} episode(s) did not replay identically")


if __name__ == "__main__":
    main()
