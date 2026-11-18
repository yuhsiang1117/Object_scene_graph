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

    `"stairs"` is already in DEFAULT_VOCABULARY (core/config/detector.py), so YOLOE has
    always been detecting staircases into the object layer -- but nothing has
    ever consumed or measured them. Stage 4 of the multi-floor plan
    (docs/history/MULTI_FLOOR.md) assumes these tracks exist and are usable; this logs
    the evidence to confirm or falsify that BEFORE the stair-detection work is
    built on top of it. Cheap: a filter over tracks the layer already holds.
    """
    object_layer = getattr(agent, "object_layer", None)
    if object_layer is None:
        return {"n_stair_tracks": 0, "stair_tracks": []}
    tracks = [
        t for t in object_layer.tracks(include_blacklisted=True)
        if normalize_label(t.label) in STAIR_LABELS
    ]
    return {
        "n_stair_tracks": len(tracks),
        "stair_tracks": [
            {
                "center": [round(float(x), 3) for x in object_layer.center_of(t)],
                "n_obs": int(t.n_obs),
                "evidence": round(float(t.evidence), 3),
                "best_score": round(float(t.best_score), 3),
                "best_bbox_px": round(float(t.best_bbox_px), 1),
            }
            for t in tracks[:20]
        ],
    }


def container_table(agent) -> dict:
    sg = getattr(agent, "scene_graph", None)
    nodes = getattr(sg, "containers", None) or {}
    out = {}
    for cid, node in nodes.items():
        try:
            out[str(int(cid))] = {
                "label": str(node.label),
                "center": [round(float(v), 3) for v in node.center],
                "floor": int(getattr(node, "floor_id", getattr(node, "floor", 0))),
            }
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def target_track_fields(agent) -> dict:
    """Snapshot the committed target track for GT-localization analysis.

    _target_obj_xy is set (in _start_approach) only once a candidate is
    accepted into APPROACH, so it is None for episodes that never committed to
    a target (pure exploration failures) -- recorded as None there."""
    object_layer = getattr(agent, "object_layer", None)
    if object_layer is None:
        return {
            "target_obj_xy": None, "cand_best_cam_xy": None,
            "cand_best_score": None, "cand_n_obs": None,
        }
    obj_xy = getattr(agent, "_target_obj_xy", None)
    cand_id = getattr(agent, "_candidate_id", None)
    track = object_layer.get(cand_id) if cand_id is not None else None
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
    from .floors import episode_floor_fields, runtime_floor_fields

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
    exploration = getattr(agent, "exploration", None)
    object_layer = getattr(agent, "object_layer", None)
    tracks = (
        list(object_layer.tracks(include_proposals=True)) if object_layer is not None else []
    )
    survival = (
        exploration.survival_report()
        if exploration is not None and hasattr(exploration, "survival_report") else {}
    )
    funnel = dict(getattr(object_layer, "funnel", {}) or {})
    return {
        "episode_id": str(episode.episode_id),
        "scene": authored.get("scene", str(episode.scene_id).split("/")[-1]),
        "target": target,
        "detector": detector_identity,
        "authored_layout": authored or None,
        "success": float(metrics.get("success", 0.0)),
        # When a benchmark scores by its own rule (dualmap_protocol, or
        # ycb.score_by_object_distance), habitat's viewpoint numbers ride
        # along under their own names; None on runs scored by habitat alone.
        "habitat_success": (
            None if metrics.get("habitat_success") is None else float(metrics["habitat_success"])
        ),
        "habitat_spl": None if metrics.get("habitat_spl") is None else float(metrics["habitat_spl"]),
        "spl": float(metrics.get("spl", 0.0)),
        # The object-rule numbers SPL is built from. Without them a recorded
        # SPL cannot be checked from episodes.jsonl at all -- every episode of
        # outputs/mf5_pass2_final carried shortest_path_to_object_m=None because
        # nothing copied it here, which is how an SPL of 0.0 on a SUCCESSFUL
        # episode went unnoticed.
        "shortest_path_to_object_m": metrics.get("shortest_path_to_object_m"),
        "travelled_m": metrics.get("travelled_m"),
        "attempt_count": metrics.get("attempt_count"),
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
            **survival,
            **funnel,
        },
        # Phase 2 dynamic-scene evidence: when beliefs flipped, what the agent
        # believed when it committed to a goal, and what it still believed about
        # the target at the end.
        # Per-step trace, only when debug frames are on: ~20 numbers a step,
        # which would bloat every normal run's episodes.jsonl. It is what the
        # S49/S50 detector-recall and framing measurements were built from.
        # Where the climb's steps went (agent/climb.py, ClimbPolicy._traced_climb).
        **({"climb_trace": agent.climb_trace}
           if (cfg.eval.debug_frames or getattr(cfg.eval, "behaviour_log", False))
           and getattr(agent, "climb_trace", None) else {}),
        **({"step_trace": agent.step_trace}
           if (cfg.eval.debug_frames or getattr(cfg.eval, "behaviour_log", False))
           and hasattr(agent, "step_trace") else {}),
        **({"behaviour": agent.behaviour.summary()}
           if hasattr(agent, "behaviour") else {}),
        "prior_map": outcome.map_note,
        # Which ASCENT-built storeys this episode planned over, and how many
        # cells each contributed. Recorded because "the obstacle map was
        # reused" is otherwise indistinguishable from a snapshot that loaded
        # zero cells onto the wrong floor.
        "prior_obstacle_map": outcome.obstacle_note,
        "attempts_used": outcome.attempts_used,
        "attempt_log": outcome.attempt_log,
        "presence_events": getattr(agent, "presence_events", []),
        "search_log_events": getattr(agent, "search_log_events", []),
        # Each change in the storey belief: per storey its belief, the anchor
        # mass and the coverage behind it, and the storey asked for.
        **({"storey_belief_log": agent.storey_belief_log}
           if getattr(agent, "storey_belief_log", None) else {}),
        # Every VLM call this episode -- role, step, track, centre, verdict --
        # so what the VLM decided can be scored against ground truth.
        **({"vlm_call_log": list(verifier.call_log[verifier_before[2]:])}
           if verifier is not None and len(verifier_before) > 2
           and getattr(verifier, "call_log", None) else {}),
        # The close-look ledger and the closest range each surface was ever
        # glanced from (agent/close_look.py): what the search looked at, from
        # where, and whether the detector answered.
        "close_look_log": getattr(agent, "close_look_log", []),
        "glance_ranges": getattr(agent, "glance_ranges", {}),
        # Every container id the logs above name, resolved: the agent's id is
        # the smallest track id of a linked component and may belong to a track
        # mapped live, so no prior map can resolve it offline. Snapshot at the
        # end of the episode; ids are stable for its length.
        "containers": container_table(agent),
        "goal_commit_log": getattr(agent, "goal_commit_log", []),
        # One entry per appearance commit: the surface, the track it chose,
        # its cosine and how many tracks it beat.
        "feature_pick_log": getattr(agent, "feature_pick_log", []),
        "target_tracks": [
            {
                "track_id": int(t.id),
                "label": str(t.label),
                "center": [float(v) for v in object_layer.center_of(t)],
                "p": round(float(t.presence.p), 4),
                # Why a track was or was not proposable: the candidate gates
                # read exactly these, and without them a track that sits in the
                # map at the right place but never becomes a goal is
                # undiagnosable from the record.
                "n_obs": int(t.n_obs),
                "best_score": round(float(t.best_score), 3),
                "best_bbox_px": round(float(t.best_bbox_px), 1),
                "evidence": round(float(t.evidence), 3),
                "floor_key": int(getattr(t, "floor_key", 0)),
                # Provenance: how many of the observations the proposal stage
                # contributed, and the running-mean feature's cosine to the
                # query. A track with n_proposal_obs == n_obs was never named.
                "n_proposal_obs": int(getattr(t, "n_proposal_obs", 0)),
                "proposal_sim": round(float(getattr(t, "proposal_sim", -1.0)), 4),
            }
            for t in tracks
            if normalize_label(t.label) == want
        ],
        "state_log": getattr(agent, "state_log", [])[:40],
        "frontier_select_log": getattr(agent, "frontier_select_log", []),
        "giveup_log": getattr(agent, "giveup_log", [])[:50],
        "approach_bbox_log": getattr(agent, "approach_bbox_log", []),
        "approach_stop_reason": getattr(agent, "approach_stop_reason", None),
        "approach_diag": getattr(agent, "approach_diag", None),
        "approach_retarget_log": getattr(agent, "approach_retarget_log", []),
        "candidate_reject_log": getattr(agent, "candidate_reject_log", []),
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
        # agent actually changed level (docs/history/MULTI_FLOOR.md).
        **episode_floor_fields(episode, outcome.trajectory_y),
        **runtime_floor_fields(episode, outcome.trajectory_y, agent, authored),
        **stair_track_fields(agent),
        # Ground-truth visibility (runner-side only; the agent never sees it).
        # Splits "never perceived the object at its new pose" into never-looked
        # and looked-but-missed, which the rest of the record cannot do and
        # which two iterations had to guess at.
        **outcome.gt_view.fields(),
        # Online floor estimate (mapping/floors.py). Compare n_floors_seen
        # against the per-scene navmesh ground truth from scripts/scene_floors.py
        # before letting behaviour depend on the estimator.
        "floor_log": getattr(agent, "floor_log", []),
        "n_floors_seen": len(getattr(getattr(agent, "floors", None), "levels", {}))
        or len(getattr(agent, "_floors", []) or []),
        "floor_y_drift": round(float(getattr(agent, "floor_y_drift", 0.0)), 4),
        "floor_transitions": len(
            getattr(getattr(agent, "floors", None), "transitions", [])
        ),
        "portal_log": getattr(agent, "portal_log", []),
        "floor_llm_log": getattr(agent, "floor_llm_log", []),
    }
