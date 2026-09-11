"""Per-step scene tagging and the stair-detector half, as ASCENT does them.

`Map_Controller._update_current_step_scene_info` (`map_controller.py:800-831`)
runs on every step the object map is updated: RAM++ tags the frame, Places365
names the room, and both are written into the OBJECT map keyed by the floor's
step counter -- `each_step_objects[_floor_num_steps]`,
`each_step_rooms[_floor_num_steps]` -- plus the running `this_floor_objects` /
`this_floor_rooms` sets. That is the content the LLM planner's prompts are
built from (`llm_planner.py:433-446`), so the key has to be the same counter
`extract_frontiers_with_image` records against a frontier.

The stair half (`map_controller.py:782-789`) is GroundingDINO's `stair` boxes
at logit >= 0.60, each segmented with MobileSAM, unioned into one mask; the
obstacle map ANDs it with RedNet's stair class.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def tag_scene(rgb: np.ndarray, floor_num_steps: int, object_map, ram, room_classifier,
              img_b64: Optional[str] = None, stats: Optional[dict] = None) -> None:
    """`_update_current_step_scene_info`, for one environment."""
    object_map.each_step_objects[floor_num_steps] = []
    object_map.each_step_rooms[floor_num_steps] = []

    tags: List[str] = ram.tags(rgb, img_b64) if ram is not None else []
    object_map.each_step_objects[floor_num_steps] = tags
    for t in tags:
        object_map.this_floor_objects.add(t)
    if stats is not None and ram is not None:
        stats["ram_calls"] = stats.get("ram_calls", 0) + 1

    room = None
    if room_classifier is not None:
        # `Place365RoomClassifier.classify` is `extract_room_categories` over
        # the top-5: the first class with a mapping into REFERENCE_ROOMS, else
        # the top-1 class (`ascent/utils.py:209-229`).
        room = room_classifier.classify(rgb)
    if room:
        object_map.each_step_rooms[floor_num_steps] = room
        object_map.this_floor_rooms.add(room)
