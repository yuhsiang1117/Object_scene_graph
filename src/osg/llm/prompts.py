"""Prompt templates for room labeling, frontier scoring and target
verification. All prompts demand a strict JSON response.
"""
from __future__ import annotations

ROOM_TYPES = [
    "bedroom", "living room", "kitchen", "bathroom", "dining room",
    "hallway", "office", "closet", "laundry room", "garage", "other",
]

ROOM_LABEL_SYSTEM = (
    "You classify indoor rooms from the objects observed in them. "
    "Answer with JSON only."
)

ROOM_LABEL_USER = """Objects observed in this room: {objects}
Choose the most likely room type from: {room_types}
Respond as JSON: {{"room_type": "<type>"}}"""


FRONTIER_SCORE_SYSTEM = (
    "You are guiding a robot searching an unseen house for a target object. "
    "Given the rooms and objects mapped so far, rate how promising each "
    "unexplored frontier is for finding the target. Use typical house layouts "
    "and object co-occurrence. Answer with JSON only."
)

FRONTIER_SCORE_USER = """Target object: {target}

Scene mapped so far:
{scene_text}

Unexplored frontiers (with nearby mapped objects):
{frontier_text}

{image_note}Rate each frontier with a probability in [0, 1] that exploring it leads
toward the target. Respond as JSON: {{"scores": {{"<frontier_id>": <prob>, ...}}}}"""

FRONTIER_IMAGE_NOTE = (
    "The attached images show views near the frontiers, in the order listed. "
)


VERIFY_SYSTEM = (
    "You help a robot double-check its object detector. Reject only clearly "
    "mislabeled objects; partial views or unusual angles of the right "
    "category count as correct. Answer with JSON only."
)

# Describe-then-decide: making the model describe the image first grounds
# the decision (direct yes/no flipped on borderline crops in prompt-lab
# tests; the description-anchored variant was correct on all references).
VERIFY_USER = """First describe what you see, then decide: does the image show a {target},
even partially or occluded?
Respond as JSON: {{"description": "<short>", "is_target": true/false, "confidence": <0-1>}}"""
