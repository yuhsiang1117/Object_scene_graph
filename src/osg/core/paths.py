"""Canonical data roots used by the navigation and authoring pipelines.

HM3D is downloaded beside the collector checkout in two versioned trees.  The
``scene_datasets/hm3d`` link is convenient for Habitat, but it is mutable and
has historically pointed at the wrong release.  Runtime code therefore uses
the v0.2 tree directly.  ``OSG_DATA_ROOT`` and the other environment variables
are explicit overrides for this container's mounted layout.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


HM3D_VERSION = "0.2"
# The collector checkout is bind-mounted whole at /habitat-data-collector
# (docker/compose.yaml); before that merge only its `data` and
# `outputs/dualmap_authoring` were mounted, under /datasets. Both are tried, in
# that order, so a container built either way resolves without an env override.
_DEFAULT_DATA_ROOTS = (
    Path("/habitat-data-collector/data"),
    Path("/datasets/habitat-data-collector/data"),
)
_DEFAULT_AUTHORING_ROOTS = (
    Path("/habitat-data-collector/outputs/dualmap_authoring"),
    Path("/datasets/habitat-data-collector/outputs/dualmap_authoring"),
)
_DEFAULT_MULTI_FLOOR_ROOTS = (
    Path("/habitat-data-collector/outputs/dualmap_multifloor"),
    Path("/datasets/habitat-data-collector/outputs/dualmap_multifloor"),
)


def _configured_path(names: Iterable[str]) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return None


def collector_data_root() -> Path:
    """Return the mounted collector ``data`` directory."""
    configured = _configured_path(("OSG_DATA_ROOT", "HABITAT_DATA_ROOT"))
    if configured is not None:
        return configured
    for candidate in _DEFAULT_DATA_ROOTS:
        if candidate.is_dir():
            return candidate
    return _DEFAULT_DATA_ROOTS[0]


def hm3d_scenes_dir() -> Path:
    """Return the parent consumed by Habitat's ``scenes_dir`` setting.

    Scene IDs begin with ``hm3d/val/...``; consequently this function returns
    ``.../versioned_data/hm3d-0.2`` rather than the child ``.../hm3d``.
    """
    configured = _configured_path(("OSG_HM3D_SCENES_DIR", "HM3D_SCENES_DIR"))
    if configured is not None:
        return configured
    return collector_data_root() / "versioned_data" / f"hm3d-{HM3D_VERSION}"


def hm3d_scene_root() -> Path:
    """Return the canonical HM3D v0.2 ``hm3d`` tree."""
    return hm3d_scenes_dir() / "hm3d"


def ycb_multi_floor_root() -> Path:
    """Return the multi-storey layout mount.

    A SEPARATE dataset from the 15-scene authoring root, not a subset of it:
    `scripts/make_multifloor_dataset.py` re-plans the cross-anchor layout of
    five multi-storey scenes so that EVERY object changes storey. The
    `mf5_*` presets read it, and until now they hard-coded the /datasets
    spelling, which stopped existing when the mount moved.
    """
    configured = _configured_path(("OSG_YCB_MULTI_FLOOR_ROOT", "YCB_MULTI_FLOOR_ROOT"))
    if configured is not None:
        return configured
    for candidate in _DEFAULT_MULTI_FLOOR_ROOTS:
        if candidate.is_dir():
            return candidate
    return _DEFAULT_MULTI_FLOOR_ROOTS[0]


def ycb_authoring_root() -> Path:
    """Return the authored-layout mount, independent of released DualMap data."""
    configured = _configured_path(("OSG_YCB_AUTHORING_ROOT", "YCB_AUTHORING_ROOT"))
    if configured is not None:
        return configured
    for candidate in _DEFAULT_AUTHORING_ROOTS:
        if candidate.is_dir():
            return candidate
    return _DEFAULT_AUTHORING_ROOTS[0]
