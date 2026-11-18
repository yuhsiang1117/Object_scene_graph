"""Integration coverage for the mounted authored-YCB staging dataset.

The authored corpus is regenerated from time to time, so nothing here may pin
which scenes it contains: five of the fifteen scenes this file was written
against are gone and five others have appeared. What is worth asserting is the
relationship between the corpus and the loader -- discovery returns exactly the
scenes that carry a complete layout -- and that staging one of them is
deterministic and places the objects where the layout says.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.sim

pytest.importorskip("habitat")

from osg.core.paths import collector_data_root, ycb_authoring_root

DATA_ROOT = collector_data_root()
LAYOUT_ROOT = ycb_authoring_root()


def _mounted() -> bool:
    return DATA_ROOT.is_dir() and LAYOUT_ROOT.is_dir()


def _authored_scenes() -> list[str]:
    """Scene directories on the mount that carry a static layout."""
    return sorted(
        path.name
        for path in LAYOUT_ROOT.iterdir()
        if (path / "static_scene_config.json").is_file()
    )


def _smoke_scene() -> str:
    """The original deterministic scene when it is mounted, else any authored
    one -- what this test checks is the loader, not the scene."""
    scenes = _authored_scenes()
    if not scenes:
        pytest.skip(f"no authored scenes under {LAYOUT_ROOT}")
    return "00829-QaLdnwvtxbs" if "00829-QaLdnwvtxbs" in scenes else scenes[0]


def _config(scene: str):
    from hydra import compose, initialize_config_dir

    from osg.core.config import register_configs

    register_configs()
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[
                "+experiment=ycb_authored_nav",
                f"ycb.scenes=[{scene}]",
                "eval.save_viz=false",
                "eval.debug_frames=false",
            ],
        )


def test_current_wildcard_discovery_selects_complete_scene():
    if not _mounted():
        pytest.skip("collector data is not mounted")
    from osg.core.config import YCB_TARGET_LABELS
    from osg.sim.ycb_layouts import discover_authored_layouts

    found = discover_authored_layouts(
        data_root=DATA_ROOT,
        layout_root=LAYOUT_ROOT,
        scenes=["*"],
        layout_types=["static"],
        layout_indices=[1, 2, 3],
        target_labels=YCB_TARGET_LABELS,
        hm3d_root=DATA_ROOT / "versioned_data/hm3d-0.2/hm3d",
    )
    # The invariant is loader-versus-mount, not a list of scene names: a
    # re-authored corpus must move this assertion, never break it.
    expected = set(_authored_scenes())
    assert expected, f"no authored scenes under {LAYOUT_ROOT}"
    assert {layout.scene_name for layout in found.layouts} == expected
    assert len(found.layouts) == len(expected)


@pytest.mark.timeout(600)
def test_authored_episodes_are_deterministic_and_injected():
    if not _mounted():
        pytest.skip("collector data is not mounted")
    from osg.sim.ycb_env import YCBAuthoredNavEnv, prepare_ycb_benchmark

    cfg = _config(_smoke_scene())
    prepared = prepare_ycb_benchmark(cfg, force=True)
    regenerated = prepare_ycb_benchmark(cfg, force=True)
    assert regenerated.manifests == prepared.manifests
    episodes = [item for manifest in prepared.manifests for item in manifest["episodes"]]
    assert episodes, "staging produced no episodes"
    assert all(item["start"]["initial_geodesic_distance"] >= 3.0 for item in episodes)
    assert all(item["viewpoints"] for item in episodes)
    assert all(
        viewpoint["visible_pixels"] >= cfg.ycb.viewpoint_min_visible_pixels
        for item in episodes
        for viewpoint in item["viewpoints"]
    )

    env = YCBAuthoredNavEnv(cfg)
    env.reset()
    authored = env.episode_metadata()
    layout = env._layout_by_key[(authored["scene"], authored["layout_id"])]
    expected = {obj.semantic_id: np.asarray(obj.translation) for obj in layout.objects}
    object_manager = env.env.sim.get_rigid_object_manager()
    assert object_manager.get_num_objects() == len(layout.objects)
    objects = env._active_objects
    actual = {
        int(obj.semantic_id): np.asarray(obj.translation, dtype=float)
        for obj in objects
    }
    assert actual.keys() == expected.keys()
    for semantic_id, translation in expected.items():
        np.testing.assert_allclose(actual[semantic_id], translation, atol=1e-5)

    env.step("stop")
    metrics = env.metrics()
    assert {"success", "spl", "distance_to_goal"} <= metrics.keys()
    assert all(
        np.isfinite(float(metrics[name]))
        for name in ("success", "spl", "distance_to_goal")
    )
    env.close()
