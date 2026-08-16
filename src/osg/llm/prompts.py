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
    "You help a robot double-check its object detector. You are shown the full "
    "camera image with ONE candidate object outlined by a red bounding box. "
    "Judge only the object inside the red box, using the rest of the scene as "
    "context. Reject only clearly mislabeled objects; partial views or unusual "
    "angles of the right category count as correct. Answer with JSON only."
)

# Describe-then-decide: making the model describe the boxed object first grounds
# the decision (direct yes/no flipped on borderline cases in prompt-lab tests;
# the description-anchored variant was correct on all references). The full
# image + red box gives the VLM scene context a bare crop loses.
VERIFY_USER = """The image has one object outlined by a red bounding box. First describe the
object inside the red box, then decide: is the object inside the red box a {target},
even partially or occluded?
Respond as JSON: {{"description": "<short>", "is_target": true/false, "confidence": <0-1>}}"""


# Forced-choice variant: rather than confirm one label (which the VLM tends to
# agree with), show the full category list and make it commit to the single best
# match -- a detector mislabel (table called a chair) then gets named correctly
# and rejected. The candidate categories are the HM3D ObjectNav goal set; "none"
# lets the VLM reject an object that matches nothing in the list.
OBJECTNAV_CATEGORIES = ["chair", "bed", "sofa", "toilet", "plant", "tv monitor"]

VERIFY_CHOICE_SYSTEM = (
    "You help a robot identify an object. You are shown the full camera image "
    "with ONE object outlined by a red bounding box. From the given list of "
    "categories, choose the SINGLE category that best matches the object inside "
    "the red box; if it clearly matches none of them, answer \"none\". Judge only "
    "the boxed object, using the rest of the scene as context. Answer with JSON only."
)

VERIFY_CHOICE_USER = """Categories: {categories}.
First describe the object inside the red bounding box, then choose the single
best-matching category from the list above (or "none" if it fits none).
Respond as JSON: {{"description": "<short>", "category": "<one category or none>", "confidence": <0-1>}}"""


# ---------------------------------------------------------------------------
# ASCENT coarse-to-fine reasoning (arXiv 2505.23019, ascent/llm_planner.py).
#
# Ported close to the original, including the one-shot example and the
# `{"Index": ..., "Reason": ...}` reply shape -- the example is doing real work
# for small instruct models, which otherwise return prose or a bare integer.
# Two deviations, both forced by what our pipeline can observe:
#
#   * ASCENT's area room type comes from Places365 scene classification of the
#     frame the frontier was seen in. We have no scene classifier, so the room
#     is our LLM-cached room label when one exists and "unknown room" otherwise,
#     and the object list carries the discriminative load.
#   * ASCENT dedupes candidate areas by SSIM over the frontier crops. We have no
#     per-frontier crop in the text path; frontier extraction already merges
#     adjacent cells into one component, which covers the same duplicate case.
# ---------------------------------------------------------------------------

AREA_CHOICE_SYSTEM = (
    "You select the optimal area for a robot to explore next, based on prior "
    "probabilistic data and environmental context. Answer with JSON only."
)

AREA_CHOICE_USER = """You need to select the optimal area based on prior probabilistic data and environmental context.
You need to answer the question in the following JSON format:
Example Input:
{{
    "Goal": "toilet",
    "Prior Probabilities between Room Type and Goal Object": [
        "Bathroom": 90.0%,
        "Bedroom": 10.0%,
    ],
    "Area Descriptions": [
        "Area 1": "a bathroom containing objects: shower, towel",
        "Area 2": "a bedroom containing objects: bed, nightstand",
        "Area 3": "a garage containing objects: car",
    ]
}}
Example Response:
{{"Index": "1", "Reason": "Shower and towel in Bathroom indicate toilet location, with high probability (90.0%)."}}
Now answer question:
Input:
{{
    "Goal": "{goal}",
    "Prior Probabilities between Room Type and Goal Object": [
{room_priors}
    ],
    "Area Descriptions": [
{areas}
    ]
}}"""

FLOOR_CHOICE_SYSTEM = (
    "You select the optimal floor for a robot to search next, based on prior "
    "probabilistic data and environmental context. Answer with JSON only."
)

FLOOR_CHOICE_USER = """You need to select the optimal floor based on prior probabilistic data and environmental context.
You need to answer the question in the following JSON format:
Example Input:
{{
    "Goal": "bed",
    "Prior Probabilities between Floor and Goal Object": [
        "Floor 1": 10.0%,
        "Floor 2": 10.0%,
        "Floor 3": 80.0%,
    ],
    "Prior Probabilities between Room Type and Goal Object": [
        "Bedroom": 80.0%,
        "Living room": 15.0%,
        "Bathroom": 5.0%,
    ],
    "Floor Descriptions": [
        "Floor 1": "Current floor. There are room types: hall, living room, containing objects: tv, sofa",
        "Floor 2": "Other floor. There are room types: bathroom containing objects: shower, towel. You do not need to explore this floor again",
        "Floor 3": "Other floor. There are room types: unknown rooms containing objects: unknown objects",
    ]
}}
Example Response:
{{"Index": "3", "Reason": "The bedroom is most likely to be on the Floor 3, and the room types and object types on the Floor 1 and Floor 2 are not directly related to the target object bed, especially it do not need to explore Floor 2 again."}}
Now answer question:
Input:
{{
    "Goal": "{goal}",
    "Prior Probabilities between Floor and Goal Object": [
{floor_priors}
    ],
    "Prior Probabilities between Room Type and Goal Object": [
{room_priors}
    ],
    "Floor Descriptions": [
{floors}
    ]
}}"""
