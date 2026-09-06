"""One line of `episodes.jsonl`, and the identity of the run that produced it.

Nine analysis scripts read this record by name. None of them would fail loudly
on a dropped field -- they would report zero, and the conclusion drawn from that
zero would be wrong -- so the key set is pinned by
tests/unit/test_episode_record_schema.py.

The record deliberately carries far more than SR. The headline number is the
least informative thing a run produces; what decides the next experiment is the
funnel, and every field here exists because some question could not be answered
without it: `gt_kf_*` splits never-looked from looked-and-missed,
`search_log_events` says whether the search posterior proposed the surface the
object was moved to, `target_tracks` says why a track at the right place never
became a goal, `approach_diag` says why the agent stopped short of one that did.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..core.labels import normalize_label


def authored_episode_metadata(episode) -> dict:
    info = getattr(episode, "info", None) or {}
    if not isinstance(info, dict):
        return {}
    authored = info.get("ycb", {})
    return dict(authored) if isinstance(authored, dict) else {}


def safe_tag(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-_")


def episode_tag(episode) -> str:
    """Filesystem-safe, UNIQUE per-episode tag: `<scene>_ep<id>`.

    `episode_id` alone is not unique -- HM3D numbers episodes per scene, so a
    run spanning scenes collides. Measured on a 100-episode v1 run: 50 episodes
    yielded only 40 distinct ids, silently overwriting 10 debug videos, top-down
    maps and keyframe directories (last scene wins). Every per-episode artifact
    path must include the scene.
    """
    authored = authored_episode_metadata(episode)
    if authored:
        scene = safe_tag(authored.get("scene", "scene"))
        layout = safe_tag(authored.get("layout_id", "layout"))
        return f"{scene}_{layout}_ep{safe_tag(episode.episode_id)}"
    scene = str(getattr(episode, "scene_id", "")).split("/")[-1].split(".")[0]
    return f"{scene}_ep{episode.episode_id}" if scene else f"ep{episode.episode_id}"


def detector_identity(cfg) -> dict:
    weights = Path(str(cfg.detector.weights))
    digest = None
    if weights.is_file():
        sha256 = hashlib.sha256()
        with weights.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                sha256.update(chunk)
        digest = sha256.hexdigest()
    return {
        "name": str(cfg.detector.name),
        "weights": str(cfg.detector.weights),
        "weights_sha256": digest,
        "imgsz": int(cfg.detector.imgsz),
        "conf": float(cfg.detector.conf),
        "half": bool(cfg.detector.half),
        "device": str(cfg.detector.device),
    }


STAIR_LABELS = ("stairs", "staircase", "stair")


def stair_track_fields(agent) -> dict:
    """Snapshot the mapped `stairs` tracks.

    `"stairs"` is already in DEFAULT_VOCABULARY (core/config.py), so YOLOE has
    always been detecting staircases into the object layer -- but nothing has
    ever consumed or measured them. Stage 4 of the multi-floor plan
    (docs/MULTI_FLOOR.md) assumes these tracks exist and are usable; this logs
    the evidence to confirm or falsify that BEFORE the stair-detection work is
    built on top of it. Cheap: a filter over tracks the layer already holds.
    """
    tracks = [
        t for t in agent.object_layer.tracks(include_blacklisted=True)
        if normalize_label(t.label) in STAIR_LABELS
    ]
    return {
        "n_stair_tracks": len(tracks),
        "stair_tracks": [
            {
                "center": [round(float(x), 3) for x in agent.object_layer.center_of(t)],
                "n_obs": int(t.n_obs),
                "evidence": round(float(t.evidence), 3),
                "best_score": round(float(t.best_score), 3),
                "best_bbox_px": round(float(t.best_bbox_px), 1),
            }
            for t in tracks[:20]
        ],
    }


def target_track_fields(agent) -> dict:
    """Snapshot the committed target track for GT-localization analysis.

    _target_obj_xy is set (in _start_approach) only once a candidate is
    accepted into APPROACH, so it is None for episodes that never committed to
    a target (pure exploration failures) -- recorded as None there."""
    obj_xy = getattr(agent, "_target_obj_xy", None)
    cand_id = getattr(agent, "_candidate_id", None)
    track = agent.object_layer.get(cand_id) if cand_id is not None else None
    best_cam = getattr(track, "best_cam_xy", None) if track is not None else None
    return {
        "target_obj_xy": [float(x) for x in obj_xy] if obj_xy is not None else None,
        "cand_best_cam_xy": [float(x) for x in best_cam] if best_cam is not None else None,
        "cand_best_score": float(track.best_score) if track is not None else None,
        "cand_n_obs": int(track.n_obs) if track is not None else None,
    }


def build_episode_record(
    *, cfg, episode, env, agent, outcome, target, detector_identity,
    metrics, profiler, scorer, scorer_before, verifier, verifier_before,
) -> dict:
    """One line of episodes.jsonl.

    `*_before` are snapshots taken at the top of the episode: the scorer and
    verifier are built once and shared across the whole run, so their counters
    are cumulative -- without the deltas, every episode after the first would
    report the run's running total instead of its own.
    """
    from .floors import episode_floor_fields

    authored = authored_episode_metadata(episode)
    # The env knows things the episode record cannot: whether a relocation
    # fired, when, and whether the agent was looking at the time.
    if hasattr(env, "episode_metadata"):
        live = env.episode_metadata()
        if isinstance(live, dict):
            authored = {**authored, **live}
    reloc = authored.get("relocation")
    if isinstance(reloc, dict) and reloc.get("step") is None and outcome.map_note is not None:
        # Two-pass protocol: the objects moved between the mapping run and this
        # one, so the map is stale from step 0. Recording it this way lets
        # belief latency read "how long until the map noticed" with no special
        # case -- the clock simply starts at the episode start.
        reloc["step"] = 0
        reloc["offline"] = True

    want = normalize_label(target)
    return {
        "episode_id": str(episode.episode_id),
        "scene": authored.get("scene", str(episode.scene_id).split("/")[-1]),
        "target": target,
        "detector": detector_identity,
        "authored_layout": authored or None,
        "success": float(metrics.get("success", 0.0)),
        "spl": float(metrics.get("spl", 0.0)),
        "distance_to_goal": float(metrics.get("distance_to_goal", -1.0)),
        "steps": outcome.steps,
        "wall_time_s": outcome.wall_time_s,
        "control_fps": round(profiler.fps("control_loop"), 2),
        "llm_calls": scorer.n_calls - scorer_before[0],
        "llm_errors": scorer.n_errors - scorer_before[1],
        "llm_last_error": scorer.last_error if scorer.last_error != scorer_before[2] else None,
        "agent_stats": {
            # Why the navmesh follower stopped, from the env -- the agent cannot
            # see the difference and has been guessing at it.
            **dict(getattr(env, "nav_reasons", {}) or {}),
            **agent.stats,
            **agent.exploration.survival_report(),
            **agent.object_layer.funnel,
        },
        # Phase 2 dynamic-scene evidence: when beliefs flipped, what the agent
        # believed when it committed to a goal, and what it still believed about
        # the target at the end.
        "prior_map": outcome.map_note,
        "attempts_used": outcome.attempts_used,
        "attempt_log": outcome.attempt_log,
        "presence_events": agent.presence_events,
        "search_log_events": agent.search_log_events,
        "goal_commit_log": agent.goal_commit_log,
        "target_tracks": [
            {
                "track_id": int(t.id),
                "label": str(t.label),
                "center": [float(v) for v in agent.object_layer.center_of(t)],
                "p": round(float(t.presence.p), 4),
                # Why a track was or was not proposable: the candidate gates
                # read exactly these, and without them a track that sits in the
                # map at the right place but never becomes a goal is
                # undiagnosable from the record.
                "n_obs": int(t.n_obs),
                "best_score": round(float(t.best_score), 3),
                "best_bbox_px": round(float(t.best_bbox_px), 1),
                "evidence": round(float(t.evidence), 3),
            }
            for t in agent.object_layer.tracks()
            if normalize_label(t.label) == want
        ],
        "state_log": agent.state_log[:40],
        "frontier_select_log": agent.frontier_select_log,
        "giveup_log": agent.giveup_log[:50],
        "approach_bbox_log": agent.approach_bbox_log,
        "approach_stop_reason": agent.approach_stop_reason,
        "approach_diag": agent.approach_diag,
        "approach_retarget_log": agent.approach_retarget_log,
        "final_xy": [float(x) for x in outcome.trajectory[-1]],
        "verify_calls": (verifier.n_calls - verifier_before[0]) if verifier is not None else 0,
        "verify_errors": (verifier.n_errors - verifier_before[1]) if verifier is not None else 0,
        # GT-localization instrumentation: the mapped 3D center (x-z) of the
        # track the agent committed to APPROACH, plus that track's best-detection
        # camera pose and score. d(target_obj_xy, GT goal) is the scene-graph
        # localization error; d(final_xy, target_obj_xy) is the residual
        # navigation error -- together they split "stopped far from goal" into
        # mislocalized-track vs failed-nav vs false-positive detection
        # (scripts/analyze_localization.py).
        **target_track_fields(agent),
        # Which floor the goal is on relative to the start pose, and whether the
        # agent actually changed level (docs/MULTI_FLOOR.md).
        **episode_floor_fields(episode, outcome.trajectory_y),
        **stair_track_fields(agent),
        # Ground-truth visibility (runner-side only; the agent never sees it).
        # Splits "never perceived the object at its new pose" into never-looked
        # and looked-but-missed, which the rest of the record cannot do and
        # which two iterations had to guess at.
        **outcome.gt_view.fields(),
        # Online floor estimate (mapping/floors.py). Compare n_floors_seen
        # against the per-scene navmesh ground truth from scripts/scene_floors.py
        # before letting behaviour depend on the estimator.
        "floor_log": agent.floor_log,
        "n_floors_seen": len(agent.floors.levels),
        "floor_y_drift": round(float(agent.floor_y_drift), 4),
        "floor_transitions": len(agent.floors.transitions),
        "portal_log": agent.portal_log,
    }
