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

from ..mapping.costmap import PLANE


def attempt_succeeded(env, frame, cfg) -> bool:
    """Would STOPping here score? Asked without ending the episode.

    Habitat scores success only when STOP is passed to the task, and that also
    terminates the episode -- so a multi-attempt protocol has to evaluate the
    same criterion itself: geodesic distance from the agent to the nearest goal
    view point, under the same success_distance.

    A benchmark that defines success differently supplies `attempt_scored` and
    is asked instead. That is not a convenience: when the point of a run is to
    compare against another system, the other system's rule has to be the one
    the agent's attempts are spent against, or the two runs are not answering
    the same question.
    """
    scored = getattr(env, "attempt_scored", None)
    if callable(scored):
        try:
            verdict = scored(frame, cfg)
        except Exception:  # never let scoring bookkeeping end a run
            return False
        # None means "this env has the hook but its rule is switched off":
        # the decision falls through to habitat's, below.
        if verdict is not None:
            return bool(verdict)
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
    # Not every policy has OSG's object layer. The ASCENT transcription has no
    # `object_layer` and no `_candidate_id`, and reaching for them raises
    # AttributeError mid-episode -- which is how the ASCENT arm on DualMap's
    # protocol died before writing a single episode. There is simply no belief
    # to demote in that case; the re-arm below still applies.
    layer = getattr(agent, "object_layer", None)
    candidate = getattr(agent, "_candidate_id", None)
    track = layer.get(candidate) if (layer is not None and candidate is not None) else None
    presence = getattr(layer, "presence_filter", None) if layer is not None else None
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
    elif track is not None and layer is not None:
        # No presence filter running (the C1-off ablation): without a belief to
        # lower there is nothing else that stops the next attempt repeating this
        # candidate, so the blacklist stays as the fallback.
        layer.blacklist(track.id)
    if track is not None and bool(getattr(cfg.verification, "failed_attempt_disables_place", False)):
        # The place was wrong, whichever fragment of it the commit named: every
        # track of the label within the layer's false-positive radius of the
        # stop's centre is retired, and new detections there are refused.
        try:
            layer = agent.object_layer
            centre = np.asarray(layer.center_of(track), dtype=float)[list(PLANE)]
            radius = float(getattr(layer, "fp_disable_radius_m", 0.5))
            n = 0
            for other in list(layer.tracks(include_blacklisted=True, include_proposals=True)):
                if other.label.lower().replace(" ", "_") != agent.target.lower().replace(" ", "_"):
                    continue
                d = float(np.linalg.norm(
                    np.asarray(layer.center_of(other), dtype=float)[list(PLANE)] - centre))
                if d <= radius and layer.disable_target(other.id):
                    n += 1
            layer.disable_place(centre, agent.target)
            agent.stats["failed_attempt_place_disabled"] = (
                agent.stats.get("failed_attempt_place_disabled", 0) + n
            )
        except Exception:
            pass
    if track is not None:
        # An attempt that ended without scoring is also evidence about IDENTITY,
        # and that is the half the belief cannot hold: a false positive is an
        # object that is really there, so the next keyframe re-detects it and
        # restores the belief the clamp just lowered.
        track.identity_rejections += 1
        track.failed_attempts = int(getattr(track, "failed_attempts", 0)) + 1
    # `rearm` is OSG's FSM reset. A policy that does not have one simply keeps
    # navigating with everything it has learned, which is what this function is
    # for -- the ASCENT transcription has no `rearm` and raised AttributeError
    # here, killing the arm before it wrote a single episode.
    rearm = getattr(agent, "rearm", None)
    if callable(rearm):
        rearm(cfg.agent.max_steps)
