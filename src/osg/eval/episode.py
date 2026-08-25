"""One episode, start to STOP.

The loop itself is short. What surrounds it is the dynamic-scene protocol:

  the map is loaded before the first step, from a snapshot a previous pass
  built, so the agent starts out confidently wrong rather than merely ignorant;

  a STOP that does not score is not the end of the episode -- the map keeps
  everything it learned and the agent goes again, which is the protocol DualMap
  is scored under and the loop the search posterior was built for;

  and ground truth is observed alongside, on the runner's side of the wall, so
  a failure can afterwards be attributed to never having looked rather than to
  having looked and missed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..mapping.costmap import HEIGHT_AXIS, PLANE
from .attempts import attempt_succeeded, rearm_after_failed_attempt
from .instruments import GroundTruthVisibility
from .prior_map import load_prior_map
from .record import authored_episode_metadata


@dataclass
class EpisodeOutcome:
    """Everything the record needs that only the loop can know."""

    steps: int = 0
    wall_time_s: float = 0.0
    trajectory: list = field(default_factory=list)
    trajectory_y: list = field(default_factory=list)
    attempts_used: int = 1
    attempt_log: list = field(default_factory=list)
    map_note: Optional[dict] = None
    gt_view: Any = None


def run_episode(cfg, env, agent, episode, target, frame, detector, debug=None) -> EpisodeOutcome:
    """Drive one episode to termination and report what happened."""
    outcome = EpisodeOutcome()
    outcome.map_note = load_prior_map(cfg, agent, str(
        authored_episode_metadata(episode).get("scene", "scene")))

    # Height is tracked alongside the 2D trajectory (rather than making
    # `trajectory` 3D) so the analyze_*.py tools keep working unchanged, while
    # episodes.jsonl still records which floor the agent was on. Camera height
    # is subtracted so these are FLOOR heights, directly comparable to
    # episode.start_position and the goal view points.
    cam_h = float(cfg.agent.camera_height)
    outcome.trajectory = [frame.camera_position[list(PLANE)]]
    outcome.trajectory_y = [float(frame.camera_position[HEIGHT_AXIS]) - cam_h]

    attempts_allowed = max(1, int(cfg.eval.attempts))
    outcome.gt_view = GroundTruthVisibility(
        authored_episode_metadata(episode).get("target_position"),
        min_det_score=cfg.scene_graph.min_det_score,
        min_det_bbox_px=cfg.scene_graph.min_det_bbox_px,
    )
    # Read-only: the agent hands over what it saw, and is given nothing.
    agent.on_keyframe_detections = (
        lambda f, dets: outcome.gt_view.observe_keyframe(f, dets, target))

    t0 = time.time()
    while not env.episode_over:
        outcome.gt_view.observe(frame)
        action = agent.act(frame)  # updates agent.costmap from `frame`
        if action == "stop" and outcome.attempts_used < attempts_allowed:
            scored = attempt_succeeded(env, frame, cfg)
            outcome.attempt_log.append(
                {"attempt": outcome.attempts_used, "step": outcome.steps,
                 "success": bool(scored)}
            )
            if not scored:
                # Not here after all. Keep the map, drop the thing it stopped
                # on, and let it choose again.
                outcome.attempts_used += 1
                rearm_after_failed_attempt(agent, cfg)
                continue
        if debug is not None:
            debug.write(frame, agent, target, detector)
        frame = env.step(action)
        outcome.trajectory.append(frame.camera_position[list(PLANE)])
        outcome.trajectory_y.append(float(frame.camera_position[HEIGHT_AXIS]) - cam_h)
        outcome.steps += 1
    outcome.wall_time_s = round(time.time() - t0, 1)
    return outcome
