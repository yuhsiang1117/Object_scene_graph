"""DualMap's released HM3D benchmark: its trials, its targets, its success rule.

This module is the single definition of the benchmark both systems are scored
on.  `scripts/run_dualmap_released_native.py` drives the released DualMap
through it and `osg.sim.dualmap_env` drives ours, so "measured by the same
criterion" is a fact about the import graph rather than a claim in a document.

Three things live here and nowhere else.

  **The trial list.**  186 queries: 79 static (Appendix Table VIII), 54
  in-anchor and 53 cross-anchor (the released layout JSONs, minus the
  cracker-box trial the appendix does not report for `00880` `0128-2`).

  **What a query points at.**  YCB additions are placed by the layout files and
  are small enough that the released translation IS the object, so they carry a
  zero extent.  Every other static query is an HM3DSem class whose instances the
  release enumerates in `class_bbox.json` as full-size boxes.

  **The success rule.**  DualMap defines SR as stopping within one metre of the
  queried object, and for the dynamic splits within three attempts.  Measured to
  the box, not the centroid: the centre of a 2.33 m bed is more than a metre
  from any navigable floor, so a centroid rule would be unsatisfiable by
  construction for exactly the furniture classes the static split is made of.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.paths import collector_data_root


DATA_ROOT = Path(os.environ.get("OSG_DATA_ROOT", collector_data_root()))
# A sibling copy of the release with assets swapped (scripts/make_dualmap_swap.py)
# is selected by pointing this at the copy; the copy's `swap.json` renames the
# queries below. The released data itself is never edited.
RELEASE_ROOT = Path(os.environ.get("OSG_DUALMAP_RELEASE_ROOT", str(DATA_ROOT / "dualmap/HM3D_collect")))
OBJECTS_ROOT = DATA_ROOT / "objects/ycb/configs"

SUCCESS_DISTANCE_M = 1.0
MAX_ATTEMPTS = 3

SCENES = ("00829-QaLdnwvtxbs", "00848-ziup5kvtCCR", "00880-Nfvxx8J5NCo")
CONDITIONS = ("static", "in_anchor", "cross_anchor")

TARGETS = {
    "00829-QaLdnwvtxbs": (
        "soup can", "bowl", "plate", "scissors", "cracker box", "pitcher"
    ),
    "00848-ziup5kvtCCR": (
        "scissors", "plate", "pitcher", "mug", "banana", "cracker box"
    ),
    "00880-Nfvxx8J5NCo": (
        "soup can", "bowl", "pitcher", "plate", "cracker box", "scissors"
    ),
}

HANDLE_TO_QUERY = {
    "003_cracker_box": "cracker box",
    "005_tomato_soup_can": "soup can",
    "011_banana": "banana",
    "019_pitcher_base": "pitcher",
    "024_bowl": "bowl",
    "025_mug": "mug",
    "029_plate": "plate",
    "037_scissors": "scissors",
}

# Appendix Table VIII, "Queried Objects in HM3D Test".  YCB additions are
# underlined there; every other name resolves against the scene's
# class_bbox.json, which the release ships "for evaluation".
STATIC_QUERIES = {
    "00829-QaLdnwvtxbs": (
        "chair", "picture", "towel", "table", "ottoman", "tap", "sofa", "bin",
        "tv", "cabinet", "magazine", "washbasin counter", "bed", "telephone",
        "clothes", "bathtub", "bag", "tv stand", "decoration", "toilet",
        "cracker box", "soup can", "pitcher", "bowl", "plate", "scissors",
    ),
    "00848-ziup5kvtCCR": (
        "pillow", "tap", "stool", "toilet paper", "towel", "magazine", "vase",
        "bed", "armchair", "bowl of fruit", "bathroom counter",
        "christmas tree", "bench", "kettle", "coffee maker", "microwave",
        "cooker", "refrigerator", "kitchen island", "bathtub", "cracker box",
        "soup can", "pitcher", "mug", "plate", "scissors", "banana",
    ),
    "00880-Nfvxx8J5NCo": (
        "shelf", "painting", "table", "cabinet", "curtain", "container",
        "mirror", "hat", "tv", "washing machine", "laptop", "clothes", "couch",
        "microwave", "sink", "trashcan", "dishwasher", "ironing board",
        "printer", "desk", "cracker box", "soup can", "pitcher", "bowl",
        "plate", "scissors",
    ),
}



def _load_swap(root: Path) -> Dict[str, Any]:
    path = root / "swap.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


SWAP = _load_swap(RELEASE_ROOT)
if SWAP:
    # Rename the swapped queries everywhere the protocol names them, and teach
    # the handle table the new assets. Trial ids change with the query name;
    # `load_reference` maps them back to the released ids for start poses.
    _rename = dict(SWAP.get("queries", {}))
    for _old, _new in SWAP.get("handles", {}).items():
        HANDLE_TO_QUERY[_new] = _rename.get(HANDLE_TO_QUERY.pop(_old, _old), _new)
    TARGETS = {scene: tuple(_rename.get(q, q) for q in qs) for scene, qs in TARGETS.items()}
    STATIC_QUERIES = {scene: tuple(_rename.get(q, q) for q in qs) for scene, qs in STATIC_QUERIES.items()}

YCB_QUERIES = frozenset(HANDLE_TO_QUERY.values())


def swapped_trial_id(released_id: str) -> str:
    """The id a released trial has under the active swap (identity without one)."""
    if not SWAP:
        return released_id
    scene, condition, layout, query = released_id.split("__", 3)
    fwd = {k.replace(" ", "_"): v.replace(" ", "_") for k, v in SWAP.get("queries", {}).items()}
    return f"{scene}__{condition}__{layout}__{fwd.get(query, query)}"


PUBLISHED = {
    ("00829-QaLdnwvtxbs", "in_anchor"): (12, 18),
    ("00848-ziup5kvtCCR", "in_anchor"): (12, 18),
    ("00880-Nfvxx8J5NCo", "in_anchor"): (11, 18),
    ("00829-QaLdnwvtxbs", "cross_anchor"): (10, 18),
    ("00848-ziup5kvtCCR", "cross_anchor"): (12, 18),
    ("00880-Nfvxx8J5NCo", "cross_anchor"): (10, 17),
    # Table II-Static reports 73.1% / 69.2% / 69.2% over 78 trials.  Those
    # percentages are 19/26 and 18/26; Table VIII however lists 27 queries for
    # 00848, so the released query lists total 79.  The paper does not say
    # which 00848 query was dropped, so all 79 are run and both are reported.
    ("00829-QaLdnwvtxbs", "static"): (19, 26),
    ("00848-ziup5kvtCCR", "static"): (18, 26),
    ("00880-Nfvxx8J5NCo", "static"): (18, 26),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_layout(scene: str, condition: str, filename: str) -> Dict[str, Any]:
    path = RELEASE_ROOT / scene / "dynamic_scene_config" / condition / filename
    return json.loads(path.read_text(encoding="utf-8"))


def protocol() -> List[Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    for scene in TARGETS:
        layout_path = RELEASE_ROOT / scene / "static_scene_config.json"
        digest = file_sha256(layout_path)
        for query in STATIC_QUERIES[scene]:
            trials.append(
                {
                    "trial_id": f"{scene}__static__static__{query.replace(' ', '_')}",
                    "scene": scene,
                    "condition": "static",
                    "layout": layout_path.name,
                    "layout_path": str(layout_path),
                    "layout_sha256": digest,
                    "query": query,
                }
            )
    for scene, targets in TARGETS.items():
        for condition in ("in_anchor", "cross_anchor"):
            directory = RELEASE_ROOT / scene / "dynamic_scene_config" / condition
            for layout_path in sorted(directory.glob("*.json")):
                for query in targets:
                    if (
                        scene == "00880-Nfvxx8J5NCo"
                        and condition == "cross_anchor"
                        and layout_path.name == "0128-2.json"
                        and query == "cracker box"
                    ):
                        continue
                    trial_id = f"{scene}__{condition}__{layout_path.stem}__{query.replace(' ', '_')}"
                    trials.append(
                        {
                            "trial_id": trial_id,
                            "scene": scene,
                            "condition": condition,
                            "layout": layout_path.name,
                            "layout_path": str(layout_path),
                            "layout_sha256": file_sha256(layout_path),
                            "query": query,
                        }
                    )
    dynamic = [trial for trial in trials if trial["condition"] != "static"]
    static = [trial for trial in trials if trial["condition"] == "static"]
    if len(dynamic) != 107:
        raise RuntimeError(f"released dynamic protocol should contain 107 trials, got {len(dynamic)}")
    if len(static) != 79:
        raise RuntimeError(f"released static protocol should contain 79 trials, got {len(static)}")
    return trials


def as_targets(
    centers: Sequence[np.ndarray], sizes: Optional[Sequence[np.ndarray]] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Pair each target centre with its half-extent.

    The YCB additions are small enough that the released translation is the
    object, so they carry a zero extent and behave exactly as a point target.
    HM3DSem instances are furniture: `class_bbox.json` gives a full-size box,
    and "within one metre of the queried object" has to be measured from that
    box, since the centre of a 2.33 m bed is more than a metre from any
    navigable floor and would be unreachable by construction.
    """
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for index, centre in enumerate(centers):
        half = (
            np.zeros(3, dtype=float)
            if sizes is None
            else np.asarray(sizes[index], dtype=float) / 2.0
        )
        out.append((np.asarray(centre, dtype=float), half))
    return out


def distance_to_target(
    position: Sequence[float], targets: Sequence[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[float, float]:
    position = np.asarray(position, dtype=float)
    horizontal = math.inf
    spatial = math.inf
    for centre, half in targets:
        gap = np.maximum(np.abs(position - centre) - half, 0.0)
        horizontal = min(horizontal, float(np.linalg.norm(gap[[0, 2]])))
        spatial = min(spatial, float(np.linalg.norm(gap)))
    return horizontal, spatial


def horizontal_distance_xz(
    goal_xz: Sequence[float], targets: Sequence[Tuple[np.ndarray, np.ndarray]]
) -> float:
    """Box-relative horizontal distance from an (x, z) point to the nearest instance.

    The two systems record their chosen goal in the ground plane and nothing else:
    DualMap as the last waypoint of ``path/local_path/N.json`` (its own z-up frame,
    so ``(x, z) = (path_x, -path_y)``), ours as ``target_obj_xy`` on the committed
    track.  Neither carries a trustworthy height, so the comparison is horizontal.
    ``distance_to_target`` already ignores the vertical component of its horizontal
    output, so the y passed here is arbitrary.
    """
    x, z = float(goal_xz[0]), float(goal_xz[1])
    return distance_to_target((x, 0.0, z), targets)[0]


def position_correct(
    goal_xz: Optional[Sequence[float]],
    targets: Sequence[Tuple[np.ndarray, np.ndarray]],
    tolerance: float = SUCCESS_DISTANCE_M,
) -> bool:
    """Did the system actually localise the queried object?

    ``SUCCESS_DISTANCE_M`` asks where the *agent* stopped, which credits an agent
    that stops near the object without ever having identified it -- DualMap scores
    that way at a text-matched furniture anchor, and any system can score that way
    by luck.  This asks the stricter question of where the system *aimed*: the goal
    it committed to must itself lie within ``tolerance`` of a legitimate instance.

    A goal of ``None`` -- no local path planned, no track committed -- is not
    position-correct, which is the point: it is the case the raw rule over-credits.
    """
    if goal_xz is None:
        return False
    return horizontal_distance_xz(goal_xz, targets) <= tolerance


def static_target_positions(scene: str, query: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Ground-truth positions for a static-split query.

    The eight YCB additions are placed by static_scene_config.json; every other
    query is an HM3DSem class whose instances the release enumerates in
    class_bbox.json.  Both are expressed in the same Habitat world frame.
    """
    if query in YCB_QUERIES:
        layout = json.loads(
            (RELEASE_ROOT / scene / "static_scene_config.json").read_text(encoding="utf-8")
        )
        return as_targets(target_positions(layout, query))
    boxes = json.loads(
        (RELEASE_ROOT / scene / "class_bbox.json").read_text(encoding="utf-8")
    )
    matched = [key for key in boxes if key.lower() == query.lower()]
    if not matched:
        raise RuntimeError(f"query {query!r} is absent from {scene} class_bbox.json")
    entries = [entry for key in matched for entry in boxes[key]]
    return as_targets(
        [entry["center"] for entry in entries], [entry["sizes"] for entry in entries]
    )


def target_positions(layout: Dict[str, Any], query: str) -> List[np.ndarray]:
    mapping = {int(key): str(value) for key, value in layout["id_handle_mapping"].items()}
    positions = [
        np.asarray(obj["translation"], dtype=float)
        for obj in layout["objects"]
        if HANDLE_TO_QUERY.get(mapping[int(obj["semantic_id"])]) == query
    ]
    if not positions:
        raise RuntimeError(f"query {query!r} has no object in released layout")
    return positions


def trial_targets(trial: Dict[str, Any]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Every instance the query may legitimately be answered by."""
    if trial["condition"] == "static":
        return static_target_positions(trial["scene"], trial["query"])
    layout = raw_layout(trial["scene"], trial["condition"], trial["layout"])
    return as_targets(target_positions(layout, trial["query"]))


def shortest_success_path(
    pathfinder, start: np.ndarray, targets: Sequence[Tuple[np.ndarray, np.ndarray]]
) -> float:
    import habitat_sim

    candidates: List[np.ndarray] = []
    for centre, half in targets:
        # Ring the box, not just the centre: for furniture the reachable band
        # lies outside the footprint.
        base = float(max(half[0], half[2]))
        for offset in (0.0, 0.25, 0.5, 0.75, 0.95):
            radius = base + offset
            samples = 1 if radius == 0.0 else 72
            for sample in range(samples):
                angle = 2.0 * math.pi * sample / samples
                point = np.array(
                    [
                        centre[0] + radius * math.cos(angle),
                        start[1],
                        centre[2] + radius * math.sin(angle),
                    ],
                    dtype=np.float32,
                )
                snapped = np.asarray(pathfinder.snap_point(point), dtype=float)
                if not np.isfinite(snapped).all():
                    continue
                if distance_to_target(snapped, targets)[0] <= SUCCESS_DISTANCE_M:
                    candidates.append(snapped)
    best = math.inf
    for goal in candidates:
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(start, dtype=np.float32)
        path.requested_end = np.asarray(goal, dtype=np.float32)
        if pathfinder.find_path(path):
            best = min(best, float(path.geodesic_distance))
    return best
