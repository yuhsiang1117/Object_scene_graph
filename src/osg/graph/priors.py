"""Category context priors: does this floor look like it holds the target?

The floor-switch gate was purely geometric -- it fired when the current storey
ran out of near frontiers, which by construction cannot happen until most of it
is explored. Measured on 25 episodes: transitions at a mean step of ~200, with
5 of 11 arriving upstairs with under 100 steps left to search. The agent was
deciding *late* because it had no reason to suspect the target was elsewhere.

The cheap signal it was ignoring is co-occurrence. A toilet lives with a sink,
a bathtub, a shower; a bed with pillows and a nightstand. Once a floor has been
looked at enough to have mapped a reasonable number of objects, seeing NONE of
the target's usual companions is real evidence the target is not there -- and
that judgement is available long before the floor is exhausted.

Deliberately not an LLM call. `docs/INVESTIGATION.md` records that LLM frontier
scoring produced byte-identical trajectories to the geometric heuristic across
all 35 episodes while adding a network dependency; a fixed co-occurrence table
over the 6 ObjectNav goal categories costs nothing and is inspectable. The
scene graph has carried a per-object `floor_id` since Stage 6, so the counts are
already there.
"""
from __future__ import annotations

from typing import Iterable, Optional, Set

# Categories that reliably share a room with each ObjectNav goal. Drawn from the
# detector vocabulary (core/config.py) so every entry is something we can
# actually see. `plant` is deliberately empty: plants appear in every room type,
# so their absence says nothing and a prior would only add noise.
CATEGORY_CONTEXT = {
    "toilet": {"sink", "bathtub", "shower", "towel", "mirror"},
    "bed": {"pillow", "cushion", "nightstand", "wardrobe", "dresser", "lamp"},
    "sofa": {"cushion", "tv_monitor", "table", "lamp", "fireplace"},
    "tv_monitor": {"sofa", "cushion", "cabinet", "table", "fireplace"},
    "chair": {"table", "desk", "cabinet", "shelf"},
    "plant": set(),
}

# Rooms whose label alone settles it, when the LLM room labeller has run.
CATEGORY_ROOMS = {
    "toilet": {"bathroom", "toilet", "restroom", "washroom"},
    "bed": {"bedroom"},
    "sofa": {"living room", "lounge", "family room"},
    "tv_monitor": {"living room", "lounge", "family room", "office"},
    "chair": {"dining room", "kitchen", "office", "living room"},
}


def _norm(label: str) -> str:
    return str(label).lower().replace("_", " ").strip()


def context_categories(target: str) -> Set[str]:
    return {_norm(c) for c in CATEGORY_CONTEXT.get(_norm(target).replace(" ", "_"), set())}


def floor_target_evidence(
    scene_graph, floor_id: int, target: str
) -> tuple:
    """(evidence, n_objects) for one storey.

    `evidence` counts how many DISTINCT context categories for the target were
    mapped on this floor, plus a large bonus if the target category itself is
    there. `n_objects` is how much has been mapped at all -- the caller needs it
    to know whether zero evidence means "not here" or merely "not looked yet".
    """
    tgt = _norm(target)
    ctx = context_categories(target)
    if not ctx:
        # No usable prior for this category (see `plant`): report "unknown" by
        # returning evidence None so the caller falls back to geometry alone.
        objs = [o for o in scene_graph.objects if o.floor_id == floor_id]
        return None, len(objs)

    seen: Set[str] = set()
    n = 0
    has_target = False
    for obj in scene_graph.objects:
        if obj.floor_id != floor_id:
            continue
        n += 1
        label = _norm(obj.label)
        if label == tgt:
            has_target = True
        if label in ctx:
            seen.add(label)

    # A room the LLM has already named as the target's home counts as strong
    # evidence even before its contents are mapped.
    rooms = CATEGORY_ROOMS.get(tgt.replace(" ", "_"), set())
    if rooms:
        for room in getattr(scene_graph, "rooms", {}).values():
            if room.floor_id == floor_id and room.label and _norm(room.label) in rooms:
                seen.add("__room__")
                break

    evidence = len(seen) + (10 if has_target else 0)
    return evidence, n
