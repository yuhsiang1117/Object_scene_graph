"""Persist the ASCENT obstacle-map stack so a later pass can start from it.

`osg/graph/map_store.py` stores the OSG side of a pass -- object tracks, their
presence beliefs, the room segmentation, the per-storey occupancy the OSG
planner drives. This stores the other half: the maps `Map_Controller` builds
(`navigation/mapping/obstacle_map.py`), one `ObstacleMap` per storey, exactly
as the transcription leaves them at the end of the episode.

The two are written side by side, keyed by scene, because the dynamic-scene
protocol needs both and they are built in different frames. What makes them
reusable together is the **anchor**: ASCENT's maps live in an episodic frame
whose origin is the pose the episode started at and whose axes are rotated to
its facing (`geometry.EpisodeAnchor`, `ascent_policy.py:233-237`), while the
OSG snapshot is in world coordinates. So the anchor is stored with the grids,
and `apply_to_costmap` maps every cell back through it. The rotation is not a
detail to be approximated away: see that function.

What is stored is EVIDENCE, not conclusions. The obstacle, explored and stair
masks are what the depth camera actually witnessed; the navigable maps are a
dilation of the obstacle mask by the agent radius and are recomputed on load
rather than trusted, so a change to `agent_radius` cannot be silently frozen
into an old snapshot.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN

SCHEMA_VERSION = 1


class ObstacleStoreError(RuntimeError):
    """A snapshot cannot be written or read."""


# The boolean evidence layers, saved verbatim. `_navigable_map` and
# `_strict_navigable_map` are deliberately NOT here: they are a pure function of
# `_map` and the agent radius, and recomputing them on load keeps a snapshot
# honest when the radius changes.
_MASKS = (
    "_map",
    "explored_area",
    "_up_stair_map",
    "_down_stair_map",
    "_disabled_stair_map",
    "stair_boundary",
    "stair_boundary_goal",
)

# Point/segment geometry the stair controller and the planner read back.
_POINTS = (
    "_up_stair_start",
    "_up_stair_end",
    "_down_stair_start",
    "_down_stair_end",
    "_up_stair_frontiers",
    "_down_stair_frontiers",
    "_up_stair_frontiers_px",
    "_down_stair_frontiers_px",
    "frontiers",
    "_frontiers_px",
    "_disabled_frontiers_px",
)

# Flags that say what the floor has been shown to contain, and how long the
# agent spent on it. `_floor_num_steps` gates the reference's first-step
# detector skip (A11), so a reloaded floor must not claim to be on step 0.
_FLAGS = (
    "_has_up_stair",
    "_has_down_stair",
    "_explored_up_stair",
    "_explored_down_stair",
    "_this_floor_explored",
    "_done_initializing",
    "_look_for_downstair_flag",
    "_reinitialize_flag",
    "_tight_search_thresh",
    "_disable_end",
)


def _self_consistency(om) -> Dict[str, Any]:
    """How much of this floor's own trajectory lies on its own EXPLORED area.

    A DIAGNOSTIC, NOT A VALIDITY TEST, and the distinction cost a long
    investigation. It reads as though a low score meant a broken map, and it
    does not: `explored_area` is what the agent SAW, not where it walked.
    `obstacle_map.py:374` erases every cell within an agent-radius dilation of
    an obstacle on every step, and `reveal_fog_of_war` propagates only through
    navigable cells, so a stairwell interior is never marked at all. An agent
    in a corridor or on stairs is therefore standing on cells its own map
    deliberately excludes.

    Measured on 00800, one episode each: bowl 0.54/0.77, banana 0.00/0.18,
    pitcher 0.11 -- and pitcher never climbed a single step, so this does not
    even track stair time. Every one of those maps is correctly framed. Gating
    reuse on this number would reject all of them.

    What it is good for is comparing coverage between passes of the same scene,
    and for noticing that a pass spent most of its budget somewhere its map
    cannot represent.
    """
    poses = list(getattr(om, "_camera_positions", []) or [])
    if not poses:
        return {"traj_poses": 0, "traj_on_map": None}
    px = om._xy_to_px(np.asarray(poses, dtype=float).reshape(-1, 2))
    size = int(om.size)
    cover = np.asarray(om.explored_area, dtype=bool)
    for name in ("_up_stair_map", "_down_stair_map"):
        mask = getattr(om, name, None)
        if mask is not None:
            cover = cover | np.asarray(mask, dtype=bool)
    inside = ((px[:, 0] >= 0) & (px[:, 0] < size)
              & (px[:, 1] >= 0) & (px[:, 1] < size))
    hits = int(cover[px[inside, 0], px[inside, 1]].sum())
    return {"traj_poses": int(len(poses)),
            "traj_on_map": round(hits / float(len(poses)), 4)}


def _occupancy(om) -> np.ndarray:
    """The floor's occupancy in OSG's Costmap2D coding, still in ASCENT pixels.

    -1 unknown, 0 free, 100 occupied. The coding is OSG's because that is what
    consumes it; the FRAME is still ASCENT's episodic one, and turning those
    pixels into world cells is `apply_to_costmap`'s job -- it needs the anchor,
    and the anchor is a rotation, so there is no correct way to do it with an
    origin alone.
    """
    return np.where(om._map.astype(bool), OCCUPIED,
                    np.where(om.explored_area, FREE, UNKNOWN)).astype(np.int8)


def save_obstacle_maps(
    path: Path, agent, *, scene: str = "", layout_id: str = "",
    episode_id: str = "",
) -> Path:
    """Write every floor's `ObstacleMap`. `path` is the JSON; grids go beside it.

    Called with the ascentnav agent itself: it owns `_floors`, the list
    `Map_Controller` grows as stairs are discovered (`agent.py:_new_floor`),
    and `anchor`, the episode-start pose every pixel is measured from.
    """
    floors = list(getattr(agent, "_floors", []) or [])
    if not floors:
        raise ObstacleStoreError("agent has no floors to save")
    anchor = getattr(agent, "anchor", None)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    records: List[Dict[str, Any]] = []
    for index, floor in enumerate(floors):
        om = floor.get("obstacle")
        if om is None:
            raise ObstacleStoreError(f"floor {index} has no obstacle map")
        prefix = f"floor_{index}_"
        for name in _MASKS:
            value = getattr(om, name, None)
            if value is not None:
                # Packed: a 1600x1600 bool mask is 2.56 MB raw and seven of
                # them per floor, which is most of a snapshot's size.
                arrays[prefix + name] = np.packbits(np.asarray(value, dtype=bool))
        for name in _POINTS:
            value = getattr(om, name, None)
            if value is not None and np.size(value):
                arrays[prefix + name] = np.asarray(value, dtype=float)
        arrays[prefix + "occupancy"] = _occupancy(om)
        poses = list(getattr(om, "_camera_positions", []) or [])
        if poses:
            arrays[prefix + "traj_ep"] = np.asarray(poses, dtype=float).reshape(-1, 2)

        value_map = floor.get("value")
        if value_map is not None:
            # `ValueMap._map` is its CONFIDENCE channel (BaseMap's array), not
            # an occupancy mask -- namespaced so it cannot collide with the
            # obstacle map's `_map` under the same floor prefix.
            for name in ("_value_map", "_map"):
                value = getattr(value_map, name, None)
                if value is not None:
                    arrays[prefix + "value" + name] = np.asarray(value, dtype=np.float32)

        records.append({
            "index": index,
            "prefix": prefix,
            "size": int(om.size),
            "pixels_per_meter": float(om.pixels_per_meter),
            "resolution": 1.0 / float(om.pixels_per_meter),
            "episode_pixel_origin": [int(v) for v in om._episode_pixel_origin],
            "agent_radius": float(om.agent_radius),
            "min_height": float(om._min_height),
            "max_height": float(om._max_height),
            "floor_num_steps": int(getattr(om, "_floor_num_steps", 0)),
            # WHICH storey this floor is, in world metres -- the mean height
            # the agent stood at while mapping it, off the stairs
            # (`navigation/agent.py:_new_floor`). Without it a snapshot's
            # floors can only be matched to a stack BY ORDER, which needs the
            # snapshot to have mapped the same NUMBER of storeys; four of the
            # six mapping episodes on 00800 mapped one storey each and could
            # not be used at all. None on a floor that was never walked, and on
            # any snapshot written before this field existed.
            "floor_y": (
                None if not int(floor.get("standing_y_n", 0) or 0)
                else float(floor["standing_y_sum"]) / int(floor["standing_y_n"])
            ),
            "explored_cells": int(np.count_nonzero(om.explored_area)),
            "obstacle_cells": int(np.count_nonzero(om._map)),
            **_self_consistency(om),
            "flags": {name: bool(getattr(om, name, False)) for name in _FLAGS},
            "mask_shape": [int(om.size), int(om.size)],
            "value_map": value_map is not None,
        })

    npz_path = path.with_suffix(".npz")
    np.savez_compressed(npz_path, **arrays)
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "scene": scene,
                "layout_id": layout_id,
                # WHICH episode's map this is. The protocol runs several per
                # scene and keeps the one that explored most, so without this
                # a checker cannot tell a wrong frame from a pose another
                # episode walked and this one never mapped.
                "episode_id": episode_id,
                "grids": npz_path.name,
                "floor_index": int(getattr(agent, "_floor_idx", 0)),
                "anchor": None if anchor is None else {
                    "xy": [float(v) for v in anchor.xy],
                    "heading": float(anchor.heading),
                },
                "floors": records,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return path


def load_obstacle_maps(path: Path) -> Dict[str, Any]:
    """Read a snapshot back, masks unpacked to their stored shape."""
    path = Path(path)
    if not path.exists():
        raise ObstacleStoreError(f"no obstacle snapshot at {path}")
    blob = json.loads(path.read_text(encoding="utf-8"))
    npz_path = path.with_name(str(blob.get("grids") or path.with_suffix(".npz").name))
    if not npz_path.exists():
        raise ObstacleStoreError(f"snapshot {path} names missing grids {npz_path}")
    with np.load(npz_path) as data:
        arrays = {key: data[key] for key in data.files}
    for record in blob.get("floors", []):
        shape = tuple(int(v) for v in record["mask_shape"])
        count = int(np.prod(shape))
        for name in _MASKS:
            key = record["prefix"] + name
            if key in arrays:
                arrays[key] = np.unpackbits(
                    arrays[key], count=count
                ).astype(bool).reshape(shape)
    blob["_arrays"] = arrays
    _to_geometric_layout(blob)
    return blob


def floor_summaries(blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What each stored storey contains, bottom storey first.

    That IS the order `_floors` keeps: `agent.py` appends on a discovered
    up-stair and inserts at 0 on a down-stair, so index 0 is always the lowest
    storey the pass ever stood on.
    """
    out = []
    for record in blob.get("floors", []):
        flags = record.get("flags") or {}
        out.append({
            "index": int(record["index"]),
            "resolution": float(record["resolution"]),
            "floor_y": (None if record.get("floor_y") is None
                        else float(record["floor_y"])),
            "traj_poses": int(record.get("traj_poses", 0)),
            "traj_on_map": record.get("traj_on_map"),
            "explored_cells": int(record.get("explored_cells", 0)),
            "obstacle_cells": int(record.get("obstacle_cells", 0)),
            "has_up_stair": bool(flags.get("_has_up_stair", False)),
            "has_down_stair": bool(flags.get("_has_down_stair", False)),
            "explored": bool(flags.get("_this_floor_explored", False)),
            "steps": int(record.get("floor_num_steps", 0)),
        })
    return out


def _anchor_from(blob: Dict[str, Any]):
    meta = blob.get("anchor")
    if not meta:
        return None
    from ..geometry import EpisodeAnchor

    return EpisodeAnchor(np.asarray(meta["xy"], dtype=float),
                         float(meta["heading"]))



def _grid_keys(blob: Dict[str, Any]):
    """Every stored 2-D grid: the packed masks and the derived occupancy."""
    arrays = blob.get("_arrays") or {}
    for key, value in arrays.items():
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == value.shape[1]:
            yield key


def _to_geometric_layout(blob: Dict[str, Any]) -> None:
    """Transpose every grid so [row, col] means what `_px_to_world` assumes.

    vlfm's `BaseMap._xy_to_px` returns GEOMETRIC pixels, (row, col) =
    (size - y*ppm - origin, x*ppm + origin), and that is the convention every
    point it records is in -- `robot_px`, the stair endpoints, the frontiers.
    But its grids are written `_map[px[:, 1], px[:, 0]]`: indexed [col, row].
    So on disk the ARRAYS are the transpose of the POINTS, and reading both
    the same way puts the map 8-9 m from where the agent walked.

    Measured on 00800 against the navmesh (the one frame that is unambiguous):
    lower-storey navigable points landed FREE 0.04 / OCCUPIED 0.04 / UNKNOWN
    0.92 on the paste as stored, and FREE 0.49 / OCCUPIED 0.00 / UNKNOWN 0.51
    transposed; the stair treads hit the pasted stair mask 0.00 as stored and
    0.36 transposed; ASCENT's own trajectory touched the stored stair mask 0
    times and the transposed one 146. The earlier trajectory "validation"
    looked the map up through vlfm's own indexing and so could not see this.

    Applied ONCE, here, so every consumer -- the paste, the figure, the
    summaries -- sees geometric grids, and undone in `apply_obstacle_maps`,
    which hands them back to an `ObstacleMap` that indexes them vlfm's way.
    """
    if blob.get("_layout") == "geometric":
        return
    arrays = blob["_arrays"]
    for key in list(_grid_keys(blob)):
        arrays[key] = np.ascontiguousarray(arrays[key].T)
    blob["_layout"] = "geometric"


def _to_vlfm_layout(blob: Dict[str, Any]) -> None:
    if blob.get("_layout") != "geometric":
        return
    arrays = blob["_arrays"]
    for key in list(_grid_keys(blob)):
        arrays[key] = np.ascontiguousarray(arrays[key].T)
    blob["_layout"] = "vlfm"

def apply_to_costmap(
    blob: Dict[str, Any],
    index: int,
    costmap,
    *,
    overwrite: bool = False,
    margin_m: float = 2.0,
    floor_y: Optional[float] = None,
    next_floor_y: Optional[float] = None,
) -> int:
    """Paste a stored ASCENT floor into an OSG `Costmap2D`. Returns cells written.

    This is the whole point of the snapshot: the occupancy ASCENT's navigation
    built becomes the occupancy OSG's planner, frontier extractor and room
    segmenter read, so pass 2 searches a map it did not build.

    IT IS A RESAMPLE, NOT A PASTE. Three things separate the two lattices:
    `BaseMap._xy_to_px` swaps the axes and flips the row, OSG's PLANE is
    habitat's (x, z) where ASCENT's frame is (x, -z), and ASCENT's map is
    rotated to the episode's start facing (`geometry.EpisodeAnchor`,
    `ascent_policy.py:233-237`). Copying with a shifted origin -- which is all
    `_CostmapView` does, and it is only ever drawn, never navigated -- puts the
    walls 10 m out at heading 0 and 22 m out at heading pi/2 (measured). So
    every destination cell is mapped back through all three (nearest
    neighbour, exact at equal resolution).

    `overwrite=False` writes only where the destination is still UNKNOWN, so a
    live map keeps what it has seen for itself and gains what it has not.

    STAIRS COME ACROSS TOO, into `costmap.stair_mask`. ASCENT keeps staircases
    in `_up_stair_map` / `_down_stair_map` and out of `explored_area`, so the
    occupancy alone has a hole exactly where a cross-floor run needs it -- the
    00800 map holds 933 up-stair and 933 down-stair cells, and the poses that
    fell outside it were overwhelmingly climb states (278 of 296 on one
    episode). `Costmap2D` means the same thing by the field: cells confirmed
    traversable, written FREE and exempted from the OCCUPIED stamp
    (`mapping/costmap.py:36-40, 186-191`), which is how `mapping/stairs.py`
    writes it.
    """
    arrays = blob.get("_arrays")
    if arrays is None:
        raise ObstacleStoreError("blob was not produced by load_obstacle_maps")
    record = next((f for f in blob.get("floors", []) if int(f["index"]) == int(index)), None)
    if record is None:
        raise ObstacleStoreError(f"snapshot has no floor {index}")
    source = arrays.get(record["prefix"] + "occupancy")
    if source is None:
        raise ObstacleStoreError(f"floor {index} has no stored occupancy")

    resolution = float(record["resolution"])
    if abs(resolution - float(costmap.resolution)) > 1e-9:
        raise ObstacleStoreError(
            f"obstacle snapshot is {resolution} m/cell and this costmap is "
            f"{costmap.resolution}; resample before loading"
        )
    anchor = _anchor_from(blob)
    size = int(record["size"])
    epo = np.asarray(record["episode_pixel_origin"], dtype=float)
    ppm = float(record["pixels_per_meter"])

    # Everything this floor has evidence for. The stair masks are part of it
    # and lie OUTSIDE `explored_area` by construction, so a bounding box taken
    # from the occupancy alone clips the staircases off -- which is the one
    # region a cross-floor pass cannot do without.
    stair_masks = [
        arrays[record["prefix"] + name]
        for name in ("_up_stair_map", "_down_stair_map")
        if record["prefix"] + name in arrays
    ]
    covered = source != UNKNOWN
    for mask in stair_masks:
        covered = covered | mask

    # Grow the destination to cover what was mapped, before writing into it.
    known = np.argwhere(covered)
    if known.size == 0:
        return 0
    corners_px = np.array([
        [known[:, 0].min(), known[:, 1].min()], [known[:, 0].min(), known[:, 1].max()],
        [known[:, 0].max(), known[:, 1].min()], [known[:, 0].max(), known[:, 1].max()],
    ], dtype=float)
    for row, col in corners_px:
        costmap.ensure_contains(
            _px_to_world(row, col, size, epo, ppm, anchor), margin_m=margin_m
        )

    # Only the window the source can actually reach. The destination grows by
    # doubling and a late-episode costmap is mostly empty, so meshing all of it
    # would allocate tens of megabytes to look up cells no storey covers.
    dest_corners = np.array([
        costmap.world_to_grid(_px_to_world(row, col, size, epo, ppm, anchor))
        for row, col in corners_px
    ])
    r0 = max(0, int(dest_corners[:, 0].min()) - 1)
    r1 = min(costmap.grid.shape[0], int(dest_corners[:, 0].max()) + 2)
    c0 = max(0, int(dest_corners[:, 1].min()) - 1)
    c1 = min(costmap.grid.shape[1], int(dest_corners[:, 1].max()) + 2)
    if r0 >= r1 or c0 >= c1:
        return 0

    # Inverse map: every destination cell centre -> the source pixel it came
    # from. Forward-mapping the source would leave holes wherever the rotation
    # spreads neighbouring pixels apart.
    rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    world_x = costmap.origin[0] + (rr + 0.5) * resolution
    world_z = costmap.origin[1] + (cc + 0.5) * resolution
    # OSG's PLANE is (x, z) of habitat's y-up world; ASCENT's CCW frame is
    # (x, -z) -- the same flip `geometry.robot_xy_heading` applies.
    ascent_xy = np.stack([world_x, -world_z], axis=-1)
    if anchor is not None:
        # `EpisodeAnchor.to_episodic`, one matrix multiply for the whole
        # window instead of a Python call per cell.
        flat = ascent_xy.reshape(-1, 2) - anchor.xy
        ep = flat @ anchor._r_world_to_ep.T
        ascent_xy = ep.reshape(rr.shape[0], rr.shape[1], 2)
    src_col = np.rint(ascent_xy[..., 0] * ppm + epo[1]).astype(int)
    src_row = np.rint(size - (ascent_xy[..., 1] * ppm + epo[0])).astype(int)
    inside = (
        (src_row >= 0) & (src_row < size) & (src_col >= 0) & (src_col < size)
    )
    values = np.full(rr.shape, UNKNOWN, dtype=np.int8)
    values[inside] = source[src_row[inside], src_col[inside]]

    window = costmap.grid[r0:r1, c0:c1]
    writable = values != UNKNOWN
    if not overwrite:
        writable &= window == UNKNOWN
    window[writable] = values[writable]

    # The stair evidence, sampled through the same inverse map. Written FREE as
    # well as into the mask: `Costmap2D` treats a stair cell as traversable,
    # and a staircase left UNKNOWN is a hole the planner will not cross.
    stairs = np.zeros(rr.shape, dtype=bool)
    for mask in stair_masks:
        stairs |= mask[src_row.clip(0, size - 1), src_col.clip(0, size - 1)] & inside
    if stairs.any():
        if costmap.stair_mask is None:
            costmap.stair_mask = np.zeros(costmap.grid.shape, dtype=bool)
        costmap.stair_mask[r0:r1, c0:c1] |= stairs
        window[stairs] = FREE
        ends = _stair_ends_from_trajectory(blob, record, stairs, r0, c0, costmap, anchor)
        if ends is None:
            ends = _stair_base_rc(
                arrays, record, costmap, size, epo, ppm, anchor, r0, c0,
                floor_y, next_floor_y,
            )
        _ramp_stair_heights(costmap, stairs, r0, c0, floor_y, next_floor_y, near_rc=ends)

    return int(np.count_nonzero(writable))


def _stair_ends_from_trajectory(blob, record, stairs, r0, c0, costmap, anchor,
                                near_m: float = 1.0, min_poses: int = 8):
    """(this storey's end, far end) of the flight, in window coordinates, from
    where ASCENT's own agent WALKED.

    The stored endpoints are unreliable for this: on 00800, `_up_stair_start`
    -> `_up_stair_end` runs from the TOP of the real staircase to the bottom,
    and a ramp oriented by them correlates -0.91 with the true tread heights.
    Free-floor adjacency fails too, because ASCENT's lower-storey map bleeds
    up the stairs onto the upper landing (both ends touch "this storey's"
    floor, 0.61 vs 0.47). What cannot be wrong is the order in which the agent
    walked the treads: the storey whose time-ordered trajectory runs along
    the flight entered it at ITS OWN end and left toward the other storey.
    Measured: floor 2's track has 143 poses on the flight, entry t=0.33,
    exit t=2.12, and that names the far end correctly against the navmesh.

    One physical staircase serves both storeys, so the walker's orientation
    settles the other storey's ramp as well, mirrored. Floor indices order
    storeys bottom-up, which is how "the other storey" is placed.
    """
    if anchor is None:
        return None
    this_index = int(record["index"])
    rc = np.argwhere(stairs).astype(float)
    if rc.shape[0] < 2:
        return None
    centred = rc - rc.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    best = None
    for other in blob.get("floors", []):
        key = other["prefix"] + "traj_ep"
        traj = blob["_arrays"].get(key)
        if traj is None or np.size(traj) < 2 * min_poses:
            continue
        ep = np.asarray(traj, dtype=float).reshape(-1, 2)
        world = np.array([anchor.to_world(p) for p in ep])
        world[:, 1] *= -1.0                                  # ASCENT (x, -z) -> OSG PLANE
        grid = np.array([costmap.world_to_grid(w) for w in world], dtype=float)
        win = grid - np.array([r0, c0], dtype=float)
        # poses on or beside the flight, in time order
        d = np.min(np.linalg.norm(win[:, None, :] - rc[None, :, :], axis=2), axis=1)
        on = win[d <= near_m / float(costmap.resolution)]
        if on.shape[0] < min_poses:
            continue
        if best is None or on.shape[0] > best[1].shape[0]:
            best = (int(other["index"]), on)
    if best is None:
        return None
    walker, on = best
    k = max(2, on.shape[0] // 4)
    entry, exit_ = on[:k].mean(axis=0), on[-k:].mean(axis=0)
    # collapse both onto the flight's own axis so a wandering track cannot
    # put an "end" beside the treads
    t_entry, t_exit = float((entry - rc.mean(axis=0)) @ axis), float((exit_ - rc.mean(axis=0)) @ axis)
    if abs(t_exit - t_entry) < 2.0:                          # walked across, not along
        return None
    lo_end = rc[int(np.argmin(centred @ axis))]
    hi_end = rc[int(np.argmax(centred @ axis))]
    walker_near, walker_far = (lo_end, hi_end) if t_entry < t_exit else (hi_end, lo_end)
    if walker == this_index:
        return walker_near, walker_far
    return walker_far, walker_near


def _stair_base_rc(arrays, record, costmap, size, epo, ppm, anchor,
                   r0, c0, floor_y, next_floor_y):
    """The flight's two ends, this-storey first, in window coordinates.

    ASCENT records the two ends of each staircase, and they are the one piece
    of direction the tread cloud itself does not carry. `_up_stair_start` is
    where an ascent begins (this storey) and `_up_stair_end` where it arrives;
    a descent names them the other way round, and the two storeys' records
    agree -- measured on 00800, floor 1's `_up_stair_start` and floor 2's
    `_down_stair_end` are the SAME pixel, as are the other two.

    The endpoints are `robot_px`, i.e. `_xy_to_px` output: GEOMETRIC (row,
    col), the same frame `_px_to_world` takes. They were once read as (col,
    row) because that made them sit on the stair MASK -- but the mask was the
    thing that was transposed (`_to_geometric_layout`); read as (row, col)
    they land on the navmesh staircase and on ASCENT's own climb.
    """
    prefix = record["prefix"]
    ascending = (
        floor_y is not None and next_floor_y is not None
        and float(next_floor_y) > float(floor_y)
    )
    kind = "_up_stair" if ascending else "_down_stair"
    out = []
    for suffix in ("_start", "_end"):
        point = arrays.get(prefix + kind + suffix)
        if point is None or np.size(point) < 2:
            return None
        px = np.asarray(point, dtype=float).ravel()
        world = _px_to_world(px[0], px[1], size, epo, ppm, anchor)  # geometric (row, col)
        rc = costmap.world_to_grid(world)
        out.append(np.array([float(rc[0]) - r0, float(rc[1]) - c0], dtype=float))
    return out[0], out[1]


def _ramp_stair_heights(costmap, stairs, r0, c0, floor_y, next_floor_y,
                        *, near_rc=None) -> None:
    """Give the pasted treads a HEIGHT, so OSG can see a staircase at all.

    This is the seam that carries ASCENT's multi-floor navigation into OSG's.
    `mapping/stairs.find_flights` -- which feeds `FloorPolicy.pursuit_flight`
    and therefore `NavAgent._flight_carrot`, the waypoint climber -- reads
    `costmap.height` and returns [] the moment it is None. It looks for cells
    whose height sits strictly BETWEEN two storeys. ASCENT's `ObstacleMap` is
    2D and stores no heights, so pasting occupancy and a stair mask alone
    leaves those cells NaN, no flight is ever found, `pursuit_flight` stays
    None and the carrot silently never fires -- measured: zero
    `climb_flight_carrot` counters across a whole cross-anchor run.

    So the treads are given a linear ramp from this storey to the next, along
    each cell's projection onto the flight's own principal axis. The heights
    are SYNTHETIC and say so: ASCENT knows where the staircase is, not how
    tall each tread is. What they have to be is monotonic and spanning, which
    is all `find_flights` asks of them (`min_span_m`, and strictly between the
    two storeys).
    """
    if floor_y is None or next_floor_y is None or costmap.height is None:
        return
    lo, hi = float(floor_y), float(next_floor_y)
    if abs(hi - lo) < 1e-6:
        return
    rc = np.argwhere(stairs)
    if rc.shape[0] < 2:
        return
    pts = rc.astype(float)
    centred = pts - pts.mean(axis=0)
    axis = None
    if near_rc is not None:
        # ASCENT's own two stair endpoints, low end first. This is the whole
        # direction signal and there is no other: measured on 00800 the
        # `_up_stair_map` is a 3.2 x 2.0 m BLOB over the stairwell, not a long
        # thin flight, so its first singular vector is not the climb direction
        # at all -- the recorded start projects to t=-4.5 in a range of
        # [-30.5, 36.2], the middle of the run rather than an end.
        base, top = np.asarray(near_rc[0], float), np.asarray(near_rc[1], float)
        delta = top - base
        norm = float(np.linalg.norm(delta))
        if norm > 1.0:            # ends a cell apart say nothing about direction
            axis = delta / norm
    if axis is None:
        # No endpoints stored: fall back to the tread cloud's principal axis,
        # which gets the SPAN right and the sign only by luck. `find_flights`
        # asks for monotonic and spanning, so a flight is still found; whether
        # its foot is the real foot is then unknown.
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        axis = vt[0]
    t = centred @ axis
    span = float(t.max() - t.min())
    if span <= 0:
        return
    frac = (t - t.min()) / span
    keep = np.ones(len(frac), dtype=bool)
    if RAMP_CLIP_TO_ENDPOINTS and near_rc is not None and axis is not None:
        # The ramp spans between ASCENT'S OWN RECORDED STAIR ENDS, not between
        # the extremes of the mask. The mask is a BLOB over the stairwell and
        # takes in the landing at the top: ramping across all of it hands the
        # flat floor a synthetic height and the waypoint climber then walks
        # onto floor that the map says is a tread.
        #
        # Measured on 00821's descent (the pasted union): over the 615 flight
        # cells that snap to the navmesh, the ramp spanned 3.17 m where the
        # true surface under them spans 1.34 m -- stretched 2.4x -- the
        # correlation with true height was only +0.70, and the flight's lowest
        # and highest cells came out 0.27 m apart in the plane for a 3.18 m
        # flight. The climb gained 0.00 m in 52 steps, turning 30 times against
        # 21 forwards.
        #
        # Cells beyond the endpoints keep whatever height they had (NaN for an
        # unseen cell), because inventing a height for them is the defect.
        base = (np.asarray(near_rc[0], float) - pts.mean(axis=0)) @ axis
        top = (np.asarray(near_rc[1], float) - pts.mean(axis=0)) @ axis
        lo_t, hi_t = (base, top) if base <= top else (top, base)
        if hi_t - lo_t > 1e-6:
            frac = (t - lo_t) / (hi_t - lo_t)
            keep = (frac >= -RAMP_ENDPOINT_SLACK) & (frac <= 1.0 + RAMP_ENDPOINT_SLACK)
            frac = np.clip(frac, 0.0, 1.0)
    if not keep.any():
        return
    rows = rc[keep, 0] + r0
    cols = rc[keep, 1] + c0
    if RAMP_FIRST_WRITER_WINS:
        # SYNTHETIC RAMPS DO NOT COMPOSE. Under `ycb.obstacle_map_union` a
        # scene is pasted from one snapshot per mapping episode, and each one
        # stamps its own ramp over the same costmap from its own recorded stair
        # ends -- measured on 00821, NINE ramp writes with nine different
        # endpoint pairs, several running in opposite directions. Whatever
        # wrote last won, and the result was an incoherent height field: over
        # the 615 flight cells that snap to the navmesh the correlation with
        # true height was +0.70, the ramp spanned 3.17 m where the true surface
        # spans 1.34, and the flight's lowest and highest cells came out 0.27 m
        # apart for a 3.18 m flight. The waypoint climber followed it onto flat
        # floor and gained 0.00 m.
        #
        # So a cell keeps the first ramp written over it. That is one snapshot's
        # coherent run rather than a blend of several, which is what
        # `find_flights` needs -- it asks for monotonic and spanning, and a
        # blend is neither. Occupancy still unions normally; only the height
        # plane is first-writer-wins.
        fresh = ~np.isfinite(costmap.height[rows, cols])
        if not fresh.any():
            return
        rows, cols, frac = rows[fresh], cols[fresh], frac[keep][fresh]
        costmap.height[rows, cols] = ramp_heights(frac, lo, hi).astype(np.float32)
        return
    costmap.height[rows, cols] = ramp_heights(frac[keep], lo, hi).astype(np.float32)
    # Mark them invented, so the agent's own depth replaces them on sight
    # rather than losing a running minimum to a ramp that reads too low.
    synthetic = getattr(costmap, "height_synthetic", None)
    if synthetic is not None:
        synthetic[rows, cols] = True


# ZERO. The ramp must start AT this storey's height, because `_flight_carrot`
# aims at the nearest cell 0.35-1.0 m above where the agent STANDS, and any
# lift at the foot end pulls that cell toward the agent's own feet. Measured
# with scripts/probe_climb_osg.py on 00800's true flight, same cells each
# time, only the heights swapped:
#
#     true heights                       +1.87 m in 40 steps   climbed
#     ramp, margin 0.12 * span (0.36 m)   0.00 m in 200        goal 0.02-0.18 m away
#     ramp, margin 0.25 m                 0.47 m in 200        goal 0.02-0.18 m away
#     ramp, margin 0                     +1.87 m in 48 steps   climbed, goal 0.2-0.8 m
#
# `find_flights` excludes cells within 0.2 m of the floor from the flight's
# component, which trims the bottom two treads off the Flight and moves its
# foot 0.4 m up the run; that is harmless, and it does not need the ramp
# lifted to happen.
RAMP_MARGIN_M = 0.0

# Ramp only BETWEEN the recorded stair ends, not across the whole mask.
# See `_ramp_stair_heights`. The slack is a fraction of the run, so a tread
# just outside the recorded ends is still given a height and the flight is not
# trimmed to nothing on a scene whose endpoints sit slightly inside the run.
# BOTH OFF, and both kept as MEASURED NEGATIVES. The pasted ramp on 00821's
# descent correlates only +0.70 with the true surface under its own cells and
# spans 3.17 m where that surface spans 1.34 -- stretched 2.4x -- and two
# attempts to improve it did not:
#
#   clip the ramp to ASCENT's recorded stair ends, instead of the whole mask
#     -> 1027 cells become 1026, correlation +0.701 -> +0.702. The recorded
#        ends already span the mask, so there is nothing outside them to trim.
#
#   first writer wins, so a union of snapshots stops blending nine ramps
#     -> correlation +0.70 -> +0.10, WORSE. Nine synthetic ramps averaged over
#        a cell are closer to the truth than any one of them alone; taking the
#        first is taking one episode's guess instead of their consensus.
#
# What is left is the real constraint: ASCENT's map is 2D, the heights are
# invented, and no rearrangement of an invented field makes it a staircase.
RAMP_CLIP_TO_ENDPOINTS = False
RAMP_FIRST_WRITER_WINS = False
RAMP_ENDPOINT_SLACK = 0.15


def ramp_heights(frac: np.ndarray, lo: float, hi: float,
                 margin_m: float = RAMP_MARGIN_M) -> np.ndarray:
    """Tread heights along a flight, `frac` in [0, 1] from this storey's end.

    Linear from `lo + margin` to `hi - margin`, the margin taken TOWARD the
    other storey so it is a fixed distance whichever way the flight goes.
    The default margin is zero; see RAMP_MARGIN_M for why it must be.
    """
    s = 1.0 if float(hi) >= float(lo) else -1.0
    start, end = float(lo) + s * margin_m, float(hi) - s * margin_m
    return start + (end - start) * np.asarray(frac, dtype=float)


def _px_to_world(row, col, size, epo, ppm, anchor) -> np.ndarray:
    """One ASCENT pixel -> the OSG world (x, z) of its centre."""
    x_ep = (float(col) - epo[1]) / ppm
    y_ep = (size - float(row) - epo[0]) / ppm
    xy = np.array([x_ep, y_ep], dtype=float)
    if anchor is not None:
        xy = anchor.to_world(xy)
    return np.array([xy[0], -xy[1]], dtype=float)


def apply_obstacle_maps(agent, blob: Dict[str, Any]) -> int:
    """Restore a saved stack onto a fresh ascentnav agent. Returns floors loaded.

    The navigable maps are recomputed from the restored obstacle mask by the
    agent's own kernels rather than read from the file, so the reloaded map
    obeys the radius THIS run is configured with.
    """
    _to_vlfm_layout(blob)
    try:
        return _apply_obstacle_maps_vlfm(agent, blob)
    finally:
        _to_geometric_layout(blob)


def _apply_obstacle_maps_vlfm(agent, blob: Dict[str, Any]) -> int:
    """`apply_obstacle_maps` with the grids already in vlfm's [col, row]."""
    arrays = blob.get("_arrays")
    if arrays is None:
        raise ObstacleStoreError("blob was not produced by load_obstacle_maps")
    records = list(blob.get("floors", []))
    if not records:
        return 0

    # Grow the stack to the stored height before filling it: `_new_floor` is
    # the only thing that knows how to build a floor for THIS config.
    while len(agent._floors) < len(records):
        agent._floors.append(agent._new_floor())

    for record in records:
        om = agent._floors[int(record["index"])]["obstacle"]
        prefix = record["prefix"]
        for name in _MASKS:
            key = prefix + name
            current = getattr(om, name, None)
            if key in arrays and current is not None:
                stored = arrays[key]
                if stored.shape == np.asarray(current).shape:
                    setattr(om, name, stored.copy())
        for name in _POINTS:
            key = prefix + name
            if key in arrays:
                setattr(om, name, arrays[key].copy())
        for name, value in (record.get("flags") or {}).items():
            if hasattr(om, name):
                setattr(om, name, bool(value))
        om._floor_num_steps = int(record.get("floor_num_steps", 0))
        _recompute_navigable(om)

    agent._floor_idx = min(int(blob.get("floor_index", 0)), len(agent._floors) - 1)
    planner = getattr(agent, "planner", None)
    if planner is not None and hasattr(planner, "floor_num"):
        planner.floor_num = len(agent._floors)
    return len(records)


def _recompute_navigable(om) -> None:
    """`ObstacleMap.update_map`'s dilation step, on a restored obstacle mask."""
    import cv2

    obstacles = om._map.astype(np.uint8)
    om._navigable_map = (
        1 - cv2.dilate(obstacles, om._navigable_kernel, iterations=1)
    ).astype(bool)
    om._strict_navigable_map = (
        1 - cv2.dilate(obstacles, om._strict_navigable_kernel, iterations=1)
    ).astype(bool)
