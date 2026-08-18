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


# Absence: the other half of verification. Confirming what IS there is only
# useful once; asking what is NOT there is what lets a map correct itself. The
# question is deliberately scoped to a marked REGION rather than the whole
# frame -- "is there a mug anywhere in this room" is unanswerable, "is there a
# mug on this table" is not -- and the model is asked to list what it does see
# first, so a "no" is grounded in a description rather than produced by a model
# agreeing with the question's framing.
# Forced choice, not a yes/no list. Measured on 20 real present/absent cases at
# the agent's own bounding box: asking "which of these categories are present"
# scored 11/20 because the model answered "yes" to almost everything -- it was
# judging plausibility, not visibility. Making it commit to one of three options,
# on a crop zoomed to the region, scored 17/20 (9/10 when the object was there,
# 8/10 when it had been moved away). "blocked" is a real answer and maps to NO
# INFORMATION, never to absence.
ABSENCE_CHOICE_SYSTEM = (
    "You check whether a specific object is still in a place a robot remembers "
    "it. You see a camera image with one region outlined in red. Describe what "
    "is inside that red region, then choose the single option that best "
    "matches. Answer with JSON only."
)

ABSENCE_CHOICE_USER = """Look ONLY inside the red box. First describe what you see there, then choose one:
"{target}" if a {target} is visibly there,
"bare" if you see only an empty surface, floor or furniture with no {target},
"blocked" if the view is obstructed.
Respond as JSON: {{"seen": "<short description>", "choice": "<{target}|bare|blocked>"}}"""


ABSENCE_SYSTEM = (
    "You help a robot check whether objects are still where it remembers them. "
    "You are shown a camera image with ONE region outlined by a red box. List "
    "what you actually see inside that red region, then say which of the asked "
    "categories are present there. Be strict: only say an object is present if "
    "you can actually see it inside the red region. Answer with JSON only."
)

ABSENCE_USER = """Look only inside the red box. First list what you see there, then for each of
these categories say whether it is present inside the red box: {categories}.
Respond as JSON: {{"visible": "<short list of what you see>", "present": [<categories that ARE there>]}}"""
