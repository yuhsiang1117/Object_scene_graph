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

from .record import authored_episode_metadata, safe_tag


def authored_scene(episode) -> str:
    return str(authored_episode_metadata(episode).get("scene", "scene"))


def _map_path(root: str, scene: str) -> Path:
    return Path(str(root)) / f"{safe_tag(scene)}.json"


def save_map_for_scene(cfg, agent, episode) -> None:
    """Pass 1: keep the map this episode built, keyed by scene."""
    root = str(cfg.ycb.map_out or "")
    if not root:
        return
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
    blob = load_map(path)
    n = apply_map(
        agent, blob,
        max_log_odds=float(cfg.scene_graph.presence.reload_max_log_odds),
        initial_floor_y=initial_floor_y,
    )
    return {
        "path": str(path),
        "from_layout": str(blob.get("layout_id", "")),
        "tracks": int(n),
        "schema_version": int(blob.get("schema_version", 1)),
        "floors": len(blob.get("floors") or [0]),
    }
