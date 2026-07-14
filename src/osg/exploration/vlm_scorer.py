"""Improvement B: multimodal frontier scoring. In addition to the serialized
scene graph text, the VLM receives representative keyframe images near each
candidate frontier — visual evidence text-only scene graphs irreversibly
discard (cf. MSGNav 2025).
"""
from __future__ import annotations

from typing import Optional

from ..graph.serialize import to_prompt_text
from ..llm import prompts
from ..perception.keyframe import KeyframeStore
from .llm_scorer import LLMTextScorer


class VLMScorer(LLMTextScorer):
    def __init__(
        self,
        client,
        subgraph_radius_m: float = 3.0,
        max_frontiers_per_call: int = 4,
        images_per_frontier: int = 2,
    ) -> None:
        super().__init__(client, subgraph_radius_m, max_frontiers_per_call)
        self.images_per_frontier = images_per_frontier

    def score(self, frontiers, sg, target, keyframes: Optional[KeyframeStore] = None):
        if not frontiers:
            return {}
        self.label_rooms(sg)
        subset = self._select_for_call(frontiers)

        images = []
        image_lines = []
        if keyframes is not None:
            for f in subset:
                refs = keyframes.nearest_facing(f.centroid_xy, k=self.images_per_frontier)
                for ref in refs:
                    img = keyframes.load_image(ref)
                    if img is not None:
                        images.append(img)
                        image_lines.append(f"image {len(images)}: view near frontier {f.id}")

        image_note = ""
        if images:
            image_note = prompts.FRONTIER_IMAGE_NOTE + "; ".join(image_lines) + "\n\n"

        user = prompts.FRONTIER_SCORE_USER.format(
            target=target,
            scene_text=to_prompt_text(sg),
            frontier_text=self._frontier_text(subset, sg),
            image_note=image_note,
        )
        resp = self.client.chat(prompts.FRONTIER_SCORE_SYSTEM, user, images=images or None)
        return self._parse_scores(resp, subset)
