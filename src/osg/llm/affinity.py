"""Where does a class of object get put down? Asked once, cached forever.

`graph.priors.CONTAINER_AFFINITY` is a hand-written table over the categories
this project has cared about so far. The YCB benchmark introduces targets it has
never seen -- and a prior of "no idea" makes the search posterior fall back to a
flat weight over every surface in the house, which is exactly the undirected
wandering Phase 3 exists to replace.

One text call per unknown target answers it. The result is cached to disk, so a
run is deterministic after the first, an A/B is not paying for the model, and
the priors the agent used are inspectable afterwards -- which matters, because
docs/INVESTIGATION.md records an earlier LLM scorer that produced byte-identical
trajectories to a geometric heuristic while adding a network dependency. A prior
you can read and diff is the difference between using a model and trusting one.

The static table always wins where it has an entry: a model must not quietly
rewrite a prior someone chose deliberately.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

AFFINITY_SYSTEM = (
    "You know where household objects are normally kept. Answer with JSON only."
)

AFFINITY_USER = """Which of these surfaces would a {target} most likely be resting on in a home?
Surfaces: {surfaces}.
Order them from most to least likely and drop any that make no sense.
Respond as JSON: {{"surfaces": ["<most likely>", "..."]}}"""


class AffinityProvider:
    """Callable: `provider("bowl") -> ["table", "counter", ...]` or None."""

    def __init__(
        self,
        client=None,
        surfaces: Sequence[str] = (),
        cache_path: Optional[str] = None,
    ) -> None:
        self.client = client
        self.surfaces = [str(s) for s in surfaces]
        self.cache_path = Path(cache_path) if cache_path else None
        self.n_calls = 0
        self.n_errors = 0
        self._cache: Dict[str, List[str]] = {}
        if self.cache_path and self.cache_path.is_file():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._cache = {}

    def __call__(self, target: str) -> Optional[List[str]]:
        key = str(target).lower().strip()
        if key in self._cache:
            return self._cache[key]
        if self.client is None or not self.surfaces:
            return None
        self.n_calls += 1
        try:
            reply = self.client.chat(
                AFFINITY_SYSTEM,
                AFFINITY_USER.format(target=key, surfaces=", ".join(self.surfaces)),
                json_response=True,
            )
        except Exception as exc:  # a missing prior is not worth failing a run over
            self.n_errors += 1
            logger.warning("affinity call failed for %r: %s", key, exc)
            return None
        ranked = reply.get("surfaces") if isinstance(reply, dict) else None
        if not isinstance(ranked, list) or not ranked:
            self.n_errors += 1
            return None
        # Keep only surfaces we actually asked about: a model that invents
        # "kitchen island" when the map has no such category is answering a
        # different question than the one the search posterior can act on.
        allowed = {s.lower(): s for s in self.surfaces}
        ranked = [allowed[str(r).lower()] for r in ranked if str(r).lower() in allowed]
        if not ranked:
            self.n_errors += 1
            return None
        self._cache[key] = ranked
        self._save()
        return ranked

    def _save(self) -> None:
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, indent=1, sort_keys=True),
                                       encoding="utf-8")
        except OSError as exc:
            logger.warning("could not write affinity cache: %s", exc)
