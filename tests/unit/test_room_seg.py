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


def test_erosion_is_the_lever_on_room_count():
    """Why the default moved from 12 to 6.

    A partially explored costmap's free space is a narrow region carved along
    the trajectory. Eroding it by 0.6 m (12 passes) destroys every core but the
    widest, and the regrow step then assigns the whole map to that one core --
    measured on 12 real costmaps as median 2 rooms, with both 500-step episodes
    collapsing to fewer than 3. Less erosion keeps more cores.
    """
    cm = Costmap2D(resolution=0.05, size_m=12.0)
    cm.grid[:, :] = UNKNOWN
    # Two 2x2 m rooms joined by a 0.5 m doorway -- narrow enough that heavy
    # erosion wipes out one core entirely.
    r0, c0 = cm.world_to_grid(np.array([-2.5, -1.0]))
    r1, c1 = cm.world_to_grid(np.array([-0.5, 1.0]))
    cm.grid[r0:r1, c0:c1] = FREE
    r2, c2 = cm.world_to_grid(np.array([0.5, -1.0]))
    r3, c3 = cm.world_to_grid(np.array([2.5, 1.0]))
    cm.grid[r2:r3, c2:c3] = FREE
    dr0, dc0 = cm.world_to_grid(np.array([-0.5, -0.25]))
    dr1, dc1 = cm.world_to_grid(np.array([0.5, 0.25]))
    cm.grid[dr0:dr1, dc0:dc1] = FREE

    def n_rooms(erode):
        labels = VoronoiRoomSegmenter(erode_iters=erode).segment(cm)
        return len(set(np.unique(labels).tolist()) - {0})

    assert n_rooms(6) >= 2, "moderate erosion should keep both rooms"
    assert n_rooms(6) >= n_rooms(20), "heavier erosion must not find MORE rooms"


def test_default_erosion_matches_the_config():
    """The swept value has to be what the agent actually constructs; the two
    knobs already in SceneGraphConfig (room_min_radius_m, room_door_width_m) are
    accepted by the constructor and then ignored, so a default that only lives
    in the segmenter is easy to leave behind."""
    from osg.core.config import SceneGraphConfig

    assert SceneGraphConfig().room_erode_iters == 6
