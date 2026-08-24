"""Container (anchor) membership and the support relation.

The scene graph was `floor -> room -> object`: every object hung off a room
centroid, so there was nowhere to record that the mug is on *that* table. Both
halves of the dynamic-scene work need exactly that edge -- the presence filter
asks "should I be seeing this object at its mapped pose", and the search
posterior asks "if it left this surface, which surface did it go to"
(docs/DYNAMIC_SCENES.md). This module supplies the two predicates the
`container` layer is built from.

Membership is a category gate AND a geometry gate. DualMap decides the same
question with CLIP similarity to a word list alone (utils/object_detector.py:
is_low_mobility), which promotes any detection carrying an anchor-ish label --
including a sliver false positive -- to a permanent map anchor. Requiring a
real horizontal top of real area costs one ellipsoid query and removes that
failure mode.

Deliberately free of simulator imports: like sim/ycb_layouts.py, this must stay
unit-testable on a machine with no Habitat and no HM3D licence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ..core.labels import normalize_label
from ..mapping.costmap import HEIGHT_AXIS, PLANE

# Every category here is already in core.config.DEFAULT_VOCABULARY, so the
# detector needs no change. The test is "does a person put things down on it",
# which is why `bed` and `sofa` are in and `chair` is out (a chair's seat is a
# support surface in principle, but in HM3D scans chairs are small, numerous and
# badly segmented, and each one admitted costs every later phase a candidate).
CONTAINER_CATEGORIES = frozenset(
    {
        "table",
        "desk",
        "counter",
        "shelf",
        "cabinet",
        "dresser",
        "nightstand",
        "bed",
        "sofa",
        "stool",
        "bench",
        "oven",
        "washing machine",
        "refrigerator",
    }
)

# Defaults mirrored by SceneGraphConfig; kept here so the predicates are usable
# (and testable) without a Hydra config in hand.
DEFAULT_TOP_H_M: Tuple[float, float] = (0.2, 1.4)
DEFAULT_MIN_AREA_M2: float = 0.06
DEFAULT_SUPPORT_TOL_M: float = 0.15


def _height_unit() -> np.ndarray:
    e = np.zeros(3)
    e[HEIGHT_AXIS] = 1.0
    return e



def is_container_label(label: str) -> bool:
    return normalize_label(label) in CONTAINER_CATEGORIES


def top_height(center: np.ndarray, ellipsoid) -> float:
    """World height of the object's upper silhouette -- its support surface."""
    return float(center[HEIGHT_AXIS] + ellipsoid.world_extent(_height_unit()))


def bottom_height(center: np.ndarray, ellipsoid) -> float:
    return float(center[HEIGHT_AXIS] - ellipsoid.world_extent(_height_unit()))


def footprint(center: np.ndarray, ellipsoid) -> Tuple[np.ndarray, np.ndarray]:
    """(center_xy, cov_xy) of the ground-plane shadow, centred on `center`.

    `center` is passed separately because a linked component's centre is the
    mean of its members (ObjectLayer.center_of), not any one ellipsoid's.
    """
    _, cov = ellipsoid.ground_footprint(PLANE)
    return np.asarray(center, dtype=float)[list(PLANE)], cov


def footprint_query(center: np.ndarray, ellipsoid) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(center_xy, INVERSE cov_xy), the form the inside-test wants.

    Inverted once per container per rebuild rather than solving a 2x2 system per
    (object, container) pair -- with ~100 surfaces and ~130 loose objects that
    inner loop runs often enough to dominate the whole scene-graph rebuild.
    None when the shadow is degenerate (a zero-thickness ellipsoid), which is
    simply a surface nothing can rest on.
    """
    mu, cov = footprint(center, ellipsoid)
    try:
        return mu, np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return None


def footprint_area(ellipsoid) -> float:
    _, cov = ellipsoid.ground_footprint(PLANE)
    return float(np.pi * np.sqrt(max(np.linalg.det(cov), 0.0)))


def point_in_footprint(center_xy: np.ndarray, inv_cov_xy: np.ndarray, p_xy: np.ndarray) -> bool:
    """Mahalanobis test against the shadow ellipse: inside iff d^2 <= 1."""
    d = np.asarray(p_xy, dtype=float) - np.asarray(center_xy, dtype=float)
    return float(d @ inv_cov_xy @ d) <= 1.0


def qualifies(
    label: str,
    top_h: float,
    area_m2: float,
    *,
    top_h_m: Tuple[float, float] = DEFAULT_TOP_H_M,
    min_area_m2: float = DEFAULT_MIN_AREA_M2,
) -> bool:
    """The membership rule itself, over already-aggregated geometry.

    Split out from `is_container` because a linked component's top is the max
    over its members and its area is the sum -- the scene graph aggregates
    first and asks second, and both callers must apply the identical rule.
    """
    if not is_container_label(label):
        return False
    if not (top_h_m[0] <= top_h <= top_h_m[1]):
        return False
    return area_m2 >= min_area_m2


def is_container(
    label: str,
    center: np.ndarray,
    ellipsoid,
    *,
    top_h_m: Tuple[float, float] = DEFAULT_TOP_H_M,
    min_area_m2: float = DEFAULT_MIN_AREA_M2,
) -> bool:
    """Single-track convenience form: category gate, then a real horizontal top
    of real area.

    The geometry gate is what a pure word list cannot do: a mis-segmented
    sliver labelled "table", or a "shelf" whose top lands at 1.9 m where
    nothing gets put down, both fail here.
    """
    return qualifies(
        label,
        top_height(center, ellipsoid),
        footprint_area(ellipsoid),
        top_h_m=top_h_m,
        min_area_m2=min_area_m2,
    )


def supports(
    surface_h: float,
    footprints: Sequence[Tuple[np.ndarray, np.ndarray]],
    obj_bottom_h: float,
    obj_xy: np.ndarray,
    *,
    tol_m: float = DEFAULT_SUPPORT_TOL_M,
) -> bool:
    """Is the object resting on this surface?

    Two conditions: its underside sits within `tol_m` of the surface height
    (above it, or slightly below to absorb ellipsoid fit error), and its
    ground-plane centre falls inside one of the container's member footprints.
    `footprints` is a sequence because a linked component -- an L-shaped sofa
    split across two ellipsoids -- is one container with two shadows, each given
    as (centre_xy, inverse cov_xy) from `footprint_query`.

    The object is described by its already-computed underside height and
    ground-plane centre rather than its ellipsoid: the caller tests one object
    against many surfaces, and re-deriving the same height from the shape matrix
    per surface is the difference between a millisecond and a hundred.
    """
    if not (surface_h - tol_m <= obj_bottom_h <= surface_h + tol_m):
        return False
    p_xy = np.asarray(obj_xy, dtype=float)
    return any(point_in_footprint(c, inv, p_xy) for c, inv in footprints)


@dataclass(frozen=True)
class ShadowIndex:
    """Every container shadow in the map, laid out for one batched query.

    `supports` is the readable statement of the rule and stays the reference;
    this is the same rule evaluated for all surfaces at once, because the scene
    graph asks it once per (object, surface) pair on every keyframe and the
    per-call numpy overhead -- not the arithmetic -- was the dominant cost of
    the whole rebuild. test_containers.py::test_batched_support_matches_the_
    scalar_rule pins the two together on random inputs.
    """

    cid: np.ndarray  # (S,) owning container id per shadow
    top: np.ndarray  # (S,) surface height
    mu: np.ndarray  # (S, 2) shadow centre
    inv: np.ndarray  # (S, 2, 2) inverse shadow covariance

    @staticmethod
    def build(entries: Sequence[Tuple[int, float, np.ndarray, np.ndarray]]) -> "ShadowIndex":
        if not entries:
            empty = np.empty(0)
            return ShadowIndex(empty, empty, np.empty((0, 2)), np.empty((0, 2, 2)))
        return ShadowIndex(
            cid=np.array([e[0] for e in entries], dtype=int),
            top=np.array([e[1] for e in entries], dtype=float),
            mu=np.stack([np.asarray(e[2], dtype=float) for e in entries]),
            inv=np.stack([np.asarray(e[3], dtype=float) for e in entries]),
        )

    def supporting(
        self, obj_bottom_h: float, obj_xy: np.ndarray, *, tol_m: float = DEFAULT_SUPPORT_TOL_M
    ) -> np.ndarray:
        """Container ids whose surface this object rests on (usually 0 or 1)."""
        if self.cid.size == 0:
            return np.empty(0, dtype=int)
        near = np.abs(self.top - float(obj_bottom_h)) <= tol_m
        if not near.any():
            return np.empty(0, dtype=int)
        d = np.asarray(obj_xy, dtype=float)[None, :] - self.mu[near]
        m2 = np.einsum("ij,ijk,ik->i", d, self.inv[near], d)
        return np.unique(self.cid[near][m2 <= 1.0])


def relative_pose(
    container_center: np.ndarray, container_R: np.ndarray, obj_center: np.ndarray
) -> np.ndarray:
    """p_rel = R_c^T (t_o - t_c).

    Anchor-relative, so a container that physically moves can migrate its id and
    carry its instances (Phase 6), and so "the mug is at the left end of the
    counter" survives a re-observation that shifts the counter's centre.
    """
    return np.asarray(container_R, dtype=float).T @ (
        np.asarray(obj_center, dtype=float) - np.asarray(container_center, dtype=float)
    )
