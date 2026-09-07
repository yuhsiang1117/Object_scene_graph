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


# --------------------------------------------------------- ASCENT-style ranker
#
# Ported from ascent/llm_planner.py::_prepare_single_floor_prompt (:409-487) and
# the system prompt in model_api/qwen25_out.py:71.
#
# Two things differ from FRONTIER_SCORE_* above and both are the point of the
# port. The task is a forced choice among three, not a 0-1 rating of up to
# eight: rating many options invites a flat, undiscriminating answer, and the
# same switch is what made the VLM verifier work (34.3% -> 42.9%). And the
# room-to-goal priors are stated *in the prompt* rather than applied as a
# multiplier outside it, so the model can weigh prior against observation
# instead of having the product handed to it.

ASCENT_RANK_SYSTEM = (
    "You are an AI assistant with advanced spatial reasoning capabilities. "
    "Your task is to choose the optimal option to find the target object."
)

# ASCENT indents its JSON by hand; reproduced so the shape the model sees is the
# same one it was shown in the example.
_I1, _I2 = " " * 4, " " * 8

ASCENT_RANK_EXAMPLE = f"""Example Input:
{{
{_I1}"Goal": "toilet",
{_I1}"Prior Probabilities between Room Type and Goal Object": [
{_I2}"Bathroom": 90.0%,
{_I2}"Bedroom": 10.0%,
{_I1}],
{_I1}"Area Descriptions": [
{_I2}"Area 1": "a bathroom containing objects: shower, towel",
{_I2}"Area 2": "a bedroom containing objects: bed, nightstand",
{_I2}"Area 3": "a garage containing objects: car",
{_I1}]
}}
Example Response:
{{"Index": "1", "Reason": "Shower and towel in Bathroom indicate toilet location, with high probability (90.0%)."}}"""

ASCENT_RANK_USER = """You need to select the optimal area based on prior probabilistic data and environmental context.
You need to answer the question in the following JSON format:
{example}
Now answer question:
Input:
{{
    "Goal": "{target}",
    "Prior Probabilities between Room Type and Goal Object": [
{priors}
    ],
    "Area Descriptions": [
{areas}
    ]
}}"""


# ------------------------------------------------------- floor decision (coarse)
#
# Ported from ascent/llm_planner.py::_prepare_multiple_floor_prompt (:491-590).
# The shape is ASCENT's; the floor descriptions are built from OSG's
# floor->room->object graph rather than from the two flat string sets ASCENT
# aggregates per storey, which is the one substantive difference.
#
# The "You do not need to explore this floor again" clause matters more than it
# looks: it is what lets the model rule a storey OUT, rather than only rank
# them. ASCENT appends it per floor at :538-541.

FLOOR_DECISION_SYSTEM = ASCENT_RANK_SYSTEM

FLOOR_DECISION_EXAMPLE = f"""Example Input:
{{
{_I1}"Goal": "bed",
{_I1}"Prior Probabilities between Floor and Goal Object": [
{_I2}"Floor 1": 10.0%,
{_I2}"Floor 2": 10.0%,
{_I2}"Floor 3": 80.0%,
{_I1}],
{_I1}"Prior Probabilities between Room Type and Goal Object": [
{_I2}"Bedroom": 80.0%,
{_I2}"Living room": 15.0%,
{_I2}"Bathroom": 5.0%,
{_I1}],
{_I1}"Floor Descriptions": [
{_I2}"Floor 1": "Current floor. There are room types: hall, living room, containing objects: tv, sofa",
{_I2}"Floor 2": "Other floor. There are room types: bathroom, containing objects: shower, towel. You do not need to explore this floor again",
{_I2}"Floor 3": "Other floor. There are room types: unknown rooms, containing objects: unknown objects",
{_I1}]
}}
Example Response:
{{"Index": "3", "Reason": "The bedroom is most likely to be on Floor 3, and the room types on Floor 1 and Floor 2 are not related to the target object bed, especially as Floor 2 does not need exploring again."}}"""

FLOOR_DECISION_USER = """You need to select the optimal floor based on prior probabilistic data and environmental context.
You need to answer the question in the following JSON format:
{example}
Now answer question:
Input:
{{
    "Goal": "{target}",
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
