"""ASCENT's frontier LLM: a forced choice among three described areas.

Ported from `ascent/llm_planner.py` (`_prepare_single_floor_prompt` :409-487,
`llm_analyze_single_floor` :273-305). Three differences from `LLMTextScorer`,
all of them the reason to try it:

* **Forced choice, not scoring.** `LLMTextScorer` asks for a probability per
  frontier for up to eight frontiers (`prompts.FRONTIER_SCORE_USER`). This asks
  which *one* of three is best. Rating many options invites a flat answer that
  discriminates nothing, and switching to a forced choice is exactly what made
  the VLM verifier work (34.3% -> 42.9%).
* **Synchronous.** The async scorer's results are keyed by `Frontier.id`, which
  is reassigned on every extraction, so they almost never reached a decision.
  One blocking call every `ranker_every_steps` is cheaper than a signal that
  does not arrive.
* **Priors inside the prompt.** The knowledge graph is currently a multiplier
  applied outside the model. ASCENT states the room-to-goal probabilities as
  text and lets the model weigh them against what was actually observed.

**Known deviation from ASCENT, recorded because it is not a small one.** ASCENT
describes each area with RAM tags and a Place365 room classification
(`map_controller.py:805-830`). Neither model is a dependency here, so areas are
described from OSG's own perception: YOLOE track labels near the frontier and
the room label from the Voronoi room segmenter. The prompt *shape* is ASCENT's;
the content comes from different models.

Every failure path returns index 0 -- the caller's own top-ranked frontier --
matching `llm_planner.py:296-300`. A network error must not end an episode.
"""
from __future__ import annotations

import os

import logging
import re
from typing import List, Optional, Sequence

import numpy as np

from ..graph.scene_graph import SceneGraph
from ..llm import prompts
from ..llm.client import ChatClient
from ..mapping.frontier import Frontier

log = logging.getLogger(__name__)


def _format_priors(kg, target: str) -> str:
    """Room-to-goal probabilities as ASCENT prints them: `"Bathroom": 90.0%`."""
    probs = kg.room_probabilities(target) if kg is not None else {}
    if not probs:
        return ' ' * 8 + '"Unknown": 0.0%'
    # Descending by probability, ties broken by name -- llm_planner.py:434-437.
    ordered = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    return ',\n'.join(
        f'{" " * 8}"{room.replace("_", " ").capitalize()}": {p * 100.0:.1f}%'
        for room, p in ordered
    )


def describe_area(
    f: Frontier,
    sg: Optional[SceneGraph],
    radius_m: float,
    max_objects: int = 8,
    semantics=None,
    mode: str = "frame",
) -> str:
    """`a bedroom containing objects: bed, nightstand` (llm_planner.py:445).

    With `semantics` (a FrontierSemantics), the room and objects come from the
    frame in which this frontier was first seen -- ASCENT's actual source
    (map_controller.py:800-830 -> llm_planner.py:418-419). Without it they come
    from a spatial query against the accumulated scene graph, which describes
    the mapped surroundings of the point rather than the view through the
    opening.

    `mode` splits the two halves so they can be measured apart:
    `"frame"` takes both from the frame, `"frame_objects"` takes only the object
    list and leaves the room to the graph. The split exists because S27 changed
    both at once and regressed, and a 10-episode probe then found the graph has
    no room label at all in 66% of cases -- so most of what `"frame"` does is
    replace an explicit "unknown room" with a confident guess, which is a
    different intervention from swapping the object tagger.

    Falls back to the graph when the frontier has no bound frame, so enabling
    the frame source can only ever replace a description, never delete one.
    """
    frame_seen = semantics.describe(f) if semantics is not None else None
    if frame_seen is not None and mode == "frame":
        room, objects = frame_seen
        objs = ", ".join(sorted(objects)) if objects else "no visible objects"
        return f'a {room.replace("_", " ")} containing objects: {objs}'
    room, objects = "unknown room", []
    if sg is not None:
        node = sg.room_of_point(f.centroid_xy)
        if node is not None and node.label:
            room = str(node.label)
        # Sort by distance so the closest objects survive the cap, then present
        # them in a stable order -- the model should not see the list reshuffle
        # between rounds for the same place.
        near = sg.objects_near(f.centroid_xy, radius_m)
        near = sorted(
            near, key=lambda o: float(np.linalg.norm(o.center[[0, 2]] - f.centroid_xy))
        )
        seen = set()
        for o in near:
            if o.label not in seen:
                seen.add(o.label)
                objects.append(o.label)
            if len(objects) >= max_objects:
                break
    if frame_seen is not None and mode == "frame_objects":
        objects = frame_seen[1]  # room stays whatever the graph knows, if
        # anything -- this arm asks only whether the object channel helps.
    objs = ", ".join(sorted(objects)) if objects else "no visible objects"
    return f'a {room.replace("_", " ")} containing objects: {objs}'


class AscentFrontierRanker:
    def __init__(
        self,
        client: ChatClient,
        kg=None,
        topk: int = 3,
        subgraph_radius_m: float = 3.0,
    ) -> None:
        self.client = client
        self.kg = kg
        self.topk = topk
        self.subgraph_radius_m = subgraph_radius_m
        self.calls = 0
        self.overrides = 0  # times the model disagreed with the value ranking
        self.abstains = 0   # replies with no index in them ("none")
        self.desc_frame = 0
        self.desc_differs = 0

    def reset(self) -> None:
        """Zero the counters at the start of an episode.

        The runner builds one ranker and shares it across every episode, so
        without this the per-episode stats are a running total and reading them
        back as per-episode numbers overstates usage by ~20x (measured: 67
        calls/episode reported against 3.5 actual).
        """
        self.calls = 0
        self.overrides = 0
        self.abstains = 0
        # How many frontier descriptions came from a frame, and how many of
        # those actually read differently from the scene-graph version.
        self.desc_frame = 0
        self.desc_differs = 0

    def pick(
        self,
        frontiers: Sequence[Frontier],
        target: str,
        sg: Optional[SceneGraph] = None,
        semantics=None,
        mode: str = "frame",
    ) -> int:
        """Index into `frontiers` (already ordered best-first by the caller).

        Returns 0 on every failure, so the caller falls back to its own ranking.
        """
        if len(frontiers) < 2:
            return 0  # nothing to choose between (llm_planner.py:79-81)

        cand = list(frontiers)[: self.topk]
        descs = []
        for f in cand:
            d = describe_area(f, sg, self.subgraph_radius_m,
                              semantics=semantics, mode=mode)
            if semantics is not None:
                # Binding a frontier to a frame is not the same as changing what
                # the LLM reads. If Places365 and the scene graph agree, the
                # prompt is identical and the A/B is measuring nothing. Count the
                # ones that actually differ, so an inert run is visible as a
                # number rather than as an unexplained null result.
                self.desc_frame += 1
                graph_d = describe_area(f, sg, self.subgraph_radius_m)
                if d != graph_d:
                    self.desc_differs += 1
                if os.environ.get("OSG_DEBUG_DESC"):
                    print(f"[desc] frame: {d}\n[desc] graph: {graph_d}", flush=True)
            descs.append(d)
        areas = ',\n'.join(
            f'{" " * 8}"Area {i + 1}": "{d}"' for i, d in enumerate(descs)
        )
        user = prompts.ASCENT_RANK_USER.format(
            example=prompts.ASCENT_RANK_EXAMPLE,
            target=target,
            priors=_format_priors(self.kg, target),
            areas=areas,
        )
        try:
            self.calls += 1
            resp = self.client.chat(prompts.ASCENT_RANK_SYSTEM, user)
        except Exception as e:  # noqa: BLE001 - any failure means "keep the ranking"
            log.warning("frontier ranker failed, keeping the value ranking: %s", e)
            return 0
        # The model does not always answer with a bare number. "none" is a
        # legitimate reply -- it means no area looks better than the value
        # ranking already says -- and `int("none")` turned that into an
        # exception that disabled the ranker for the call and logged it as a
        # failure. Take the first integer in whatever came back.
        raw = str(resp.get("Index", "")).strip()
        m = re.search(r"-?\d+", raw)
        if m is None:
            self.abstains += 1
            log.debug("frontier ranker abstained (%r), keeping the value ranking", raw)
            return 0
        idx = int(m.group()) - 1  # 1-based
        if not (0 <= idx < len(cand)):
            log.warning("frontier ranker returned out-of-range index %d", idx + 1)
            return 0
        if idx != 0:
            self.overrides += 1
        return idx
