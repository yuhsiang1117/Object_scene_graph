from __future__ import annotations

import numpy as np

from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN, Costmap2D
from osg.mapping.room_seg import VoronoiRoomSegmenter


def _two_room_map() -> Costmap2D:
    """Two 3x3 m rooms joined by a narrow door."""
    cm = Costmap2D(resolution=0.1, size_m=12.0)
    cm.grid[:, :] = UNKNOWN
    cm.grid[30:60, 30:60] = FREE   # room A
    cm.grid[30:60, 62:92] = FREE   # room B
    cm.grid[30:60, 60:62] = OCCUPIED  # dividing wall...
    cm.grid[43:47, 60:62] = FREE      # ...with a 0.4 m door
    return cm


def test_two_rooms_detected():
    cm = _two_room_map()
    seg = VoronoiRoomSegmenter(min_room_radius_m=0.5, door_width_m=1.0, min_room_cells=50)
    labels = seg.segment(cm)
    room_ids = set(np.unique(labels)) - {0}
    assert len(room_ids) == 2
    a = labels[45, 45]
    b = labels[45, 75]
    assert a != 0 and b != 0 and a != b


def test_room_ids_stable_across_reruns():
    cm = _two_room_map()
    seg = VoronoiRoomSegmenter(min_room_radius_m=0.5, door_width_m=1.0, min_room_cells=50)
    l1 = seg.segment(cm)
    # Map grows a bit (more of room B revealed)
    cm.grid[30:60, 92:96] = FREE
    l2 = seg.segment(cm)
    assert l2[45, 45] == l1[45, 45]
    assert l2[45, 75] == l1[45, 75]


def test_open_space_not_oversegmented():
    """One big open area should not split into many rooms."""
    cm = Costmap2D(resolution=0.1, size_m=12.0)
    cm.grid[:, :] = UNKNOWN
    cm.grid[20:80, 20:80] = FREE
    seg = VoronoiRoomSegmenter(min_room_radius_m=0.5, door_width_m=1.0, min_room_cells=50)
    labels = seg.segment(cm)
    room_ids = set(np.unique(labels)) - {0}
    assert len(room_ids) == 1
