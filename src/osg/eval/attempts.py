"""Several navigation attempts per query, matching DualMap's protocol.

Habitat scores success only when STOP is passed to the task, and that also
terminates the episode -- so a multi-attempt protocol has to evaluate the same
criterion itself, which `attempt_succeeded` does.

Scoring a single attempt is a STRICTER protocol than the system being compared
against, so this exists to match theirs rather than to flatter ours. When an
attempt does not score, the map keeps everything it learned -- beliefs, searched
surfaces, objects mapped along the way -- because that carry-over is the whole
point of retrying, and the agent goes again.

What a failed attempt is WORTH is a protocol question and lives here; putting
the agent back into EXPLORE is the agent's own `rearm`.
"""
from __future__ import annotations

import math

import numpy as np


def attempt_succeeded(env, frame, cfg) -> bool:
    """Would STOPping here score? Asked without ending the episode.

    Habitat scores success only when STOP is passed to the task, and that also
    terminates the episode -- so a multi-attempt protocol has to evaluate the
    same criterion itself: geodesic distance from the agent to the nearest goal
    view point, under the same success_distance.
    """
    try:
        episode = env.current_episode
        sim = env.env.sim
        pos = sim.get_agent_state().position
        best = float("inf")
        for goal in getattr(episode, "goals", []) or []:
            for vp in getattr(goal, "view_points", []) or []:
                import habitat_sim

                path = habitat_sim.ShortestPath()
                path.requested_start = np.asarray(pos, dtype=np.float32)
                path.requested_end = np.asarray(vp.agent_state.position, dtype=np.float32)
                if sim.pathfinder.find_path(path):
                    best = min(best, float(path.geodesic_distance))
        return best <= float(cfg.agent.success_distance)
    except Exception:  # never let scoring bookkeeping end a run
        return False


def rearm_after_failed_attempt(agent, cfg) -> None:
    """Give the agent another attempt without giving it a new map.

    Everything learned survives -- presence beliefs, searched surfaces, objects
    mapped along the way -- because that carry-over is the whole point of
    retrying. Only the navigation state is reset.

    The candidate just rejected has its belief driven below `min_presence`
    rather than being blacklisted. Blacklisting is permanent and C1's premise is
    that no state is absorbing; this is the same mistake the absence path and
    the map loader each had to have removed, and it bites hardest exactly when
    the map is RIGHT. Measured over 42 episodes: five episodes committed once to
    a track 0.00-0.39 m from the true object, failed the attempt, struck the
    track off, and then had no way to stop -- three cracker box episodes finished
    0.67-0.95 m from the goal with 429 steps unspent, and a soup can episode
    ended 10.95 m away with 475 unspent.

    One VLM-strength negative reading takes a belief reloaded at 0.82 to 0.36,
    under the 0.45 bar. A track first detected in THIS episode sits at the +3.0
    positive clamp, where the same reading lands at 0.75 and the next attempt
    would simply repeat it, so the belief is additionally held just under the
    bar -- that much is the attempt protocol's requirement rather than an
    inference, and it is written as a clamp so it reads as one. What matters is
    that it stays a belief: one later detection is worth +2.5 and puts the track
    back above the bar, which is the whole difference from a blacklist.
    """
    track = (
        agent.object_layer.get(agent._candidate_id)
        if agent._candidate_id is not None else None
    )
    presence = getattr(agent.object_layer, "presence_filter", None)
    if track is not None and presence is not None:
        vc = cfg.verification
        presence.apply_reading(
            track, False,
            float(vc.vlm_recall),
            float(vc.vlm_q),
        )
        bar = float(cfg.scene_graph.presence.min_presence)
        bar = min(max(bar, 1e-3), 1.0 - 1e-3)
        # One detector-strength step below the bar: far enough that this attempt
        # is over, near enough that one sighting undoes it.
        under_the_bar = math.log(bar / (1.0 - bar)) - 0.87
        track.presence.log_odds = min(float(track.presence.log_odds), under_the_bar)
    elif track is not None:
        # No presence filter running (the C1-off ablation): without a belief to
        # lower there is nothing else that stops the next attempt repeating this
        # candidate, so the blacklist stays as the fallback.
        agent.object_layer.blacklist(track.id)
    if track is not None:
        # An attempt that ended without scoring is also evidence about IDENTITY,
        # and that is the half the belief cannot hold: a false positive is an
        # object that is really there, so the next keyframe re-detects it and
        # restores the belief the clamp just lowered.
        track.identity_rejections += 1
    agent.rearm(cfg.agent.max_steps)
