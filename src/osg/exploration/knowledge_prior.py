"""Statistical priors: does this direction look like where the target lives?

Two priors, both lifted from ASCENT and subset to the HM3D goal categories by
scripts/make_priors.py:

    room prior    P(room type | goal category)   -- toilets are in bathrooms
    floor prior   P(storey | goal, n storeys)    -- beds are upstairs

The room prior is used WITHOUT an LLM. ASCENT asks a language model to name the
room a frontier sits in and puts that in a prompt; the same information is
already implicit in the objects mapped nearby, and the knowledge graph is a
joint distribution over (object, room). So a frontier's nearby objects give a
room-type distribution, and its affinity with the goal is how much that overlaps
the goal's own room distribution:

    affinity(f) = sum_r  P(r | objects near f) * P(r | goal)

A frontier surrounded by a shower and a towel scores high for "toilet" and low
for "bed", with no model call. That also sidesteps the reason the LLM scorer was
found redundant here: it never actually influenced a selection, because its
results arrived asynchronously and were keyed on frontier ids that are
reassigned every extraction.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_DIR = Path("data/priors")


def _norm(label: str) -> str:
    return str(label).lower().replace("_", " ").strip()


class KnowledgeGraph:
    """P(room type | object category), for goals and for observed objects."""

    def __init__(self, room_prior: Dict[str, Dict[str, float]]) -> None:
        self.room_prior = {_norm(k): v for k, v in room_prior.items()}
        self.rooms: List[str] = sorted(
            {r for dist in self.room_prior.values() for r in dist}
        )

    @classmethod
    def load(cls, path: Optional[str] = None) -> "KnowledgeGraph":
        p = Path(path) if path else DEFAULT_DIR / "hm3d_room_prior.json"
        return cls(json.loads(Path(p).read_text()))

    def room_probabilities(self, category: str) -> Dict[str, float]:
        """Normalised room distribution for a category, {} if unknown."""
        dist = self.room_prior.get(_norm(category))
        if not dist:
            return {}
        total = sum(dist.values())
        return {r: v / total for r, v in dist.items()} if total > 0 else {}

    def affinity(self, goal: str, nearby_labels: Iterable[str]) -> Optional[float]:
        """Overlap between the goal's rooms and the rooms implied by `nearby_labels`.

        None when nothing useful is known -- the caller must not read that as
        "unpromising", only as "no evidence", or unexplored regions with no
        mapped objects would be permanently deprioritised.
        """
        goal_dist = self.room_probabilities(goal)
        if not goal_dist:
            return None

        acc: Dict[str, float] = {}
        n = 0
        for label in nearby_labels:
            dist = self.room_probabilities(label)
            if not dist:
                continue  # the graph is subset to the goal categories
            n += 1
            for room, p in dist.items():
                acc[room] = acc.get(room, 0.0) + p
        if n == 0:
            return None
        return sum(goal_dist.get(r, 0.0) * (v / n) for r, v in acc.items())


class FloorPrior:
    """P(storey | goal category, number of storeys in the building)."""

    def __init__(self, table: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        self.table = {_norm(k): v for k, v in table.items()}

    @classmethod
    def load(cls, path: Optional[str] = None) -> "FloorPrior":
        p = Path(path) if path else DEFAULT_DIR / "hm3d_floor_prior.json"
        return cls(json.loads(Path(p).read_text()))

    def probabilities(self, category: str, total_floors: int) -> Dict[int, float]:
        """{floor_order (0 = ground): probability}, {} if unknown.

        Clamps to the largest building size the table covers, so a 6-storey
        scene still gets the shape of the distribution rather than nothing.
        """
        per_total = self.table.get(_norm(category))
        if not per_total:
            return {}
        sizes = sorted(int(k) for k in per_total)
        if not sizes:
            return {}
        key = str(min(max(total_floors, sizes[0]), sizes[-1]))
        dist = per_total.get(key)
        if not dist:
            return {}
        total = sum(dist.values())
        if total <= 0:
            return {}
        # Table floors are 1-based; FloorStack order is 0-based.
        return {int(f) - 1: v / total for f, v in dist.items()}
