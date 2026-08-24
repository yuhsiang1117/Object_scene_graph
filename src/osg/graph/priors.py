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

from typing import Set
from ..core.labels import normalize_label

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


def context_categories(target: str) -> Set[str]:
    return {normalize_label(c) for c in CATEGORY_CONTEXT.get(normalize_label(target).replace(" ", "_"), set())}


def floor_target_evidence(
    scene_graph, floor_id: int, target: str
) -> tuple:
    """(evidence, n_objects) for one storey.

    `evidence` counts how many DISTINCT context categories for the target were
    mapped on this floor, plus a large bonus if the target category itself is
    there. `n_objects` is how much has been mapped at all -- the caller needs it
    to know whether zero evidence means "not here" or merely "not looked yet".
    """
    tgt = normalize_label(target)
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
        label = normalize_label(obj.label)
        if label == tgt:
            has_target = True
        if label in ctx:
            seen.add(label)

    # A room the LLM has already named as the target's home counts as strong
    # evidence even before its contents are mapped.
    rooms = CATEGORY_ROOMS.get(tgt.replace(" ", "_"), set())
    if rooms:
        for room in getattr(scene_graph, "rooms", {}).values():
            if room.floor_id == floor_id and room.label and normalize_label(room.label) in rooms:
                seen.add("__room__")
                break

    evidence = len(seen) + (10 if has_target else 0)
    return evidence, n


# ---------------------------------------------------------- search posterior
# Where does an object of this class REST? CATEGORY_CONTEXT above answers "what
# else is in the room", which is the right question for choosing a storey and
# the wrong one for choosing a surface. These two tables answer "which surface",
# and they are what turns "not here" into "then look there" (Phase 3).

# Container categories a class is plausibly found on, best first. Deliberately
# small and inspectable; anything absent falls back to LLM affinity when it is
# enabled, and to a flat prior when it is not.
CONTAINER_AFFINITY = {
    "bowl": ["table", "counter", "desk", "cabinet", "shelf", "sink"],
    "mug": ["table", "counter", "desk", "shelf", "cabinet", "sink"],
    "red plate": ["table", "counter", "cabinet", "shelf", "sink"],
    "cup": ["table", "counter", "desk", "shelf"],
    "bottle": ["table", "counter", "desk", "shelf", "refrigerator"],
    "cracker box": ["counter", "table", "shelf", "cabinet", "desk"],
    "tin can": ["counter", "shelf", "cabinet", "table"],
    "pitcher": ["counter", "table", "shelf", "cabinet"],
    "scissors": ["desk", "table", "cabinet", "shelf", "counter"],
    "banana": ["counter", "table", "bowl", "shelf"],
    "book": ["desk", "table", "shelf", "nightstand", "bed"],
    "pillow": ["bed", "sofa", "bench"],
}

# What a class needs from a surface: (min top height, max top height, min area).
# Read straight off the container geometry the scene graph already computes, so
# a mug is not proposed on the floor and a bowl is not proposed on a shelf at
# head height. Generous bands -- this is a prior, not a constraint.
AFFORDANCE = {
    "bowl": (0.4, 1.3, 0.06),
    "mug": (0.4, 1.3, 0.04),
    "red plate": (0.4, 1.3, 0.06),
    "cracker box": (0.3, 1.4, 0.06),
    "tin can": (0.3, 1.4, 0.04),
    "pitcher": (0.4, 1.3, 0.06),
    "scissors": (0.4, 1.4, 0.04),
    "banana": (0.4, 1.3, 0.04),
    "book": (0.2, 1.6, 0.04),
    "pillow": (0.2, 0.9, 0.15),
}
DEFAULT_AFFORDANCE = (0.15, 1.6, 0.03)


# Weight of an unlisted category, and how hard the ranking bites.
#
# CONTAINER_AFFINITY is six entries long per class and was never meant to be
# exhaustive, but an unlisted category used to score 0.25 -- BELOW the lowest
# listed entry, i.e. positive evidence against. Measured on the benchmark's own
# relocations, that is backwards. Of 53 moves whose destination surface is
# mapped, the objects land on:
#
#     bed 26, desk 19, table 6, cabinet 3, nightstand 3, bench 2, stool 2, ...
#
# and 37 of the 53 land on a category the table does not list for that class --
# a tomato soup can is put on a bed seven times. Ranking the 114 relocations by
# the true surface's position in the candidate list:
#
#     affinity x proximity (as shipped)      top-1 19   top-5 36   median  5
#     proximity alone, no affinity           top-1 26   top-5 38   median  2
#     affinity alone, no proximity           top-1  0   top-5 12   median 27
#     neither (arbitrary order)              top-1  2   top-5 14   median 25
#
# Affinity alone is no better than arbitrary order, and multiplying it in makes
# proximity worse. So: absence from the list is not evidence (0.5, the same
# value used when there is no ranking at all), and the ranking that remains is
# softened to a tie-breaker rather than a veto.
#
# The first of those two is a correctness fix and would be right on any data.
# The second is calibrated against THIS benchmark, whose relocations appear to
# be placed for reachability rather than for semantic plausibility -- see
# docs/DYNAMIC_SCENES.md. On a benchmark that moved objects the way people do,
# a stronger affinity term would be worth more, and AFFINITY_POWER is the knob.
UNLISTED_AFFINITY = 0.5
AFFINITY_POWER = 0.5


def affinity_scores(target: str, source=None) -> dict:
    """{container category: weight in (0, 1]}, best first, or {} if unknown.

    `source` is an optional callable (an LLM affinity provider) consulted only
    when the static table has no entry -- the table stays authoritative so a
    model cannot quietly rewrite a prior someone chose deliberately.
    """
    key = normalize_label(target)
    ranked = CONTAINER_AFFINITY.get(key)
    if ranked is None and source is not None:
        ranked = source(key)
    if not ranked:
        return {}
    n = len(ranked)
    return {normalize_label(c): 1.0 - 0.5 * i / max(n - 1, 1) for i, c in enumerate(ranked)}


def affords(target: str, top_h: float, area_m2: float) -> float:
    """Can a surface at this height and size hold this class? 1.0 or 0.0.

    Binary on purpose: a shelf at 1.9 m is not a slightly worse place to look
    for a bowl, it is not a place to look for a bowl.
    """
    h_min, h_max, a_min = AFFORDANCE.get(normalize_label(target), DEFAULT_AFFORDANCE)
    return 1.0 if (h_min <= top_h <= h_max and area_m2 >= a_min) else 0.0
