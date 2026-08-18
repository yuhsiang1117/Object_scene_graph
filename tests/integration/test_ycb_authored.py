"""Integration coverage for the mounted authored-YCB staging dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.sim

pytest.importorskip("habitat")

DATA_ROOT = Path("/datasets/habitat-data-collector/data")
LAYOUT_ROOT = Path("/datasets/habitat-data-collector/outputs/dualmap_authoring")


def _mounted() -> bool:
    return DATA_ROOT.is_dir() and LAYOUT_ROOT.is_dir()


def _config():
    from hydra import compose, initialize_config_dir

    from osg.core.config import register_configs

    register_configs()
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[
                "+experiment=ycb_authored_nav",
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
    )
    assert [(layout.scene_name, len(layout.objects)) for layout in found.layouts] == [
        ("00829-QaLdnwvtxbs", 6)
    ]
    assert any(item["scene"] == "00800-TEEsavR23oF" for item in found.skipped)


@pytest.mark.timeout(600)
def test_six_deterministic_episodes_and_reset_injection():
    if not _mounted():
        pytest.skip("collector data is not mounted")
    from osg.sim.ycb_env import YCBAuthoredNavEnv, prepare_ycb_benchmark

    cfg = _config()
    prepared = prepare_ycb_benchmark(cfg, force=True)
    regenerated = prepare_ycb_benchmark(cfg, force=True)
    assert regenerated.manifests == prepared.manifests
    episodes = [item for manifest in prepared.manifests for item in manifest["episodes"]]
    assert len(episodes) == 6
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
    assert object_manager.get_num_objects() == 6
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
