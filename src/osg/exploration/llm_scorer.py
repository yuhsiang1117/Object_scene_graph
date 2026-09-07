"""Text-only LLM frontier scoring (paper baseline): room labels are assigned
by the LLM and cached per room id; one call scores all frontiers using the
serialized scene graph plus per-frontier local subgraph descriptors.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..graph.scene_graph import SceneGraph
from ..graph.serialize import to_prompt_text
from ..llm import prompts
from ..llm.client import ChatClient
from ..mapping.frontier import Frontier
from ..perception.keyframe import KeyframeStore
from .scorer import FrontierScorer


class LLMTextScorer(FrontierScorer):
    def __init__(
        self,
        client: ChatClient,
        subgraph_radius_m: float = 3.0,
        max_frontiers_per_call: int = 8,
    ) -> None:
        self.client = client
        self.subgraph_radius_m = subgraph_radius_m
        self.max_frontiers = max_frontiers_per_call
        self._room_label_cache: Dict[int, str] = {}

    def reset(self) -> None:
        # room.id restarts from 1 each episode (fresh RoomSegmenter per
        # NavAgent) -- an unreset cache would label a new scene's room 1
        # with whatever a previous, unrelated scene's room 1 was called.
        self._room_label_cache.clear()

    # ------------------------------------------------------------- room labels

    def label_rooms(self, sg: SceneGraph) -> None:
        for room in sg.unlabeled_rooms():
            if room.id in self._room_label_cache:
                room.label = self._room_label_cache[room.id]
                continue
            objs = sg.objects_in_room(room.id)
            names = sorted({o.label for o in objs})
            if not names:
                continue
            try:
                resp = self.client.chat(
                    prompts.ROOM_LABEL_SYSTEM,
                    prompts.ROOM_LABEL_USER.format(
                        objects=", ".join(names), room_types=", ".join(prompts.ROOM_TYPES)
                    ),
                )
                label = str(resp.get("room_type", "")).strip().lower()
                if label:
                    room.label = label
                    self._room_label_cache[room.id] = label
            except Exception:
                continue  # labeling is best-effort; retry next cycle

    # ----------------------------------------------------------------- scoring

    def _frontier_text(self, frontiers: List[Frontier], sg: SceneGraph) -> str:
        lines = []
        for f in frontiers:
            nearby = sg.objects_near(f.centroid_xy, self.subgraph_radius_m)
            room = sg.room_of_point(f.centroid_xy)
            desc = ", ".join(sorted({o.label for o in nearby})) or "nothing mapped nearby"
            room_txt = f" in/near room {room.id} ({room.label})" if room and room.label else ""
            lines.append(f"- frontier {f.id}{room_txt}: near {desc}")
        return "\n".join(lines)

    def _select_for_call(self, frontiers: List[Frontier]) -> List[Frontier]:
        return sorted(frontiers, key=lambda f: -f.size)[: self.max_frontiers]

    def score(self, frontiers, sg, target, keyframes: Optional[KeyframeStore] = None):
        if not frontiers:
            return {}
        self.label_rooms(sg)
        subset = self._select_for_call(frontiers)
        user = prompts.FRONTIER_SCORE_USER.format(
            target=target,
            scene_text=to_prompt_text(sg),
            frontier_text=self._frontier_text(subset, sg),
            image_note="",
        )
        resp = self.client.chat(prompts.FRONTIER_SCORE_SYSTEM, user)
        return self._parse_scores(resp, subset)

    @staticmethod
    def _parse_scores(resp: dict, subset: List[Frontier]) -> Dict[int, float]:
        raw = resp.get("scores", resp)
        out: Dict[int, float] = {}
        valid = {f.id for f in subset}
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    fid = int(str(k).strip().lstrip("frontier_ "))
                    p = float(v)
                except (ValueError, TypeError):
                    continue
                if fid in valid:
                    out[fid] = float(np.clip(p, 0.0, 1.0))
        return out
