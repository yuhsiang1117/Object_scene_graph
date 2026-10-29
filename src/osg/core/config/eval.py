"""`eval` group: which benchmark runs, over which episodes, with what recorded.

`mode` picks the benchmark: `objectnav` is the standard HM3D dataset,
`ycb_authored` is the dynamic-scene benchmark built from authored layouts.
`attempts` is the protocol knob -- DualMap allows a query several navigation
attempts and scoring one is a stricter protocol than the system being compared
against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


from ..paths import hm3d_scenes_dir


@dataclass
class EvalConfig:
    # Navigation attempts per query. DualMap allows several: a failed attempt
    # updates the map and the agent goes again. Scoring one attempt is a
    # STRICTER protocol than the system being compared against, so this exists
    # to match theirs rather than to flatter ours. 1 keeps the old behaviour.
    attempts: int = 1
    # `objectnav` loads the standard HM3D episode dataset. `ycb_authored`
    # discovers scene-layout JSON files written by habitat-data-collector and
    # builds equivalent ObjectNav episodes from the placed YCB objects.
    mode: str = "objectnav"
    split: str = "val"
    dataset_version: str = "v2"  # HM3D-semantics v0.2, 6 categories
    episodes_path: str = "data/datasets/objectnav/hm3d/v2/{split}/{split}.json.gz"
    # Explicit HM3D v0.2 parent.  Do not use the mutable
    # ``data/scene_datasets/hm3d`` symlink: v1/v2 episode IDs are resolved
    # against ``<scenes_dir>/hm3d/...`` by Habitat.
    scenes_dir: str = field(default_factory=lambda: str(hm3d_scenes_dir()))
    # The habitat-lab benchmark config the env is built from; it fixes the
    # dataset type and the goal-category set.
    benchmark_config: str = "benchmark/nav/objectnav/objectnav_hm3d.yaml"
    num_episodes: int = -1  # -1 = all
    # >0 forces habitat to move to a new scene after this many episodes, so a
    # fixed-size subset spans the split instead of draining one scene first.
    # -1 = habitat default (group by scene, ~10000-step budget per scene).
    max_scene_repeat_episodes: int = -1
    shuffle_episodes: bool = False
    max_scene_repeat_steps: int = 50_000
    allow_sliding: bool = False
    episode_ids: Optional[List[str]] = None
    # Restrict the eval to specific scene ids (None/["*"] = all). Used by the
    # single-floor preset since the 2D scene graph cannot represent stairs.
    content_scenes: Optional[List[str]] = None
    save_viz: bool = True
    # Per-step debug video: for each episode write viz/debug/ep<ID>.mp4 whose
    # frames are [live RGB + YOLOE segmentation overlay | top-down costmap] at
    # every step. The detector is re-run per step FOR VISUALIZATION ONLY (it
    # does not feed the object layer -- keyframe detection is unchanged), so SR
    # is unaffected; it roughly doubles detector load, hence off by default.
    debug_frames: bool = False
    # H.264 quality for those videos, re-encoded on close. cv2 writes MPEG-4
    # Part 2, which is ~4x larger for no benefit here: a 500-step episode is
    # 36 MB raw and 9.5 MB at CRF 30, and 30 MB is where file sharing stops.
    # Lower is better quality and bigger; 0 disables the re-encode.
    debug_video_crf: int = 30
    # Draw the prior obstacle map with each episode's target on it, once per
    # scene, into `viz/obstacle_map_<scene>.png`. Only runs that read
    # `ycb.obstacle_map_in` have anything to draw, so this is a no-op
    # elsewhere. On by default: the cross-anchor failures are geometric, and
    # the first of these figures answered three questions the numbers had not
    # (docs/history/CROSS_ANCHOR_OBSTACLE_MAP.md).
    obstacle_map_png: bool = True
    # Per-step behaviour log in episodes.jsonl, WITHOUT the debug videos.
    # `debug_frames` writes an mp4 per episode, which is minutes of encoding and
    # tens of MB on a 100-episode run; the analysis in
    # scripts/analyse_behaviour.py only ever wanted the trace.
    behaviour_log: bool = False
    # Synchronise CUDA at every profiler section, so queued GPU work is charged
    # to the stage that queued it. Costs real time; for attributing a run's
    # wall clock, not for running one (core/profiler.py).
    profile_sync: bool = False
    # Ground-truth keyframe dump: for every keyframe where the instrument says
    # the target was in view and unoccluded, write the RGB with the projected
    # object marked and the RAW detections drawn beside it. This is the picture
    # `gt_kf_in_view=14, gt_kf_detected=0` refuses to give you -- whether the
    # object was a legible object at that range or four grey pixels behind a
    # chair. Off by default; it writes one JPEG per in-view keyframe.
    gt_dump_dir: str = ""
    # Recogniser dump: every crop the STOP gate's recogniser judges -- prior
    # vetting, live vetting, the STOP itself -- written with its verdict, its
    # margin and the track's geometry, plus the full frame for live and STOP
    # crops so any other crop treatment can be re-cut offline
    # (eval/vet_dump.py). Read-only; a run with it is the run without it. A
    # relative path is under `output_dir`. Off by default.
    vet_dump_dir: str = ""
    rgb_width: int = 640
    rgb_height: int = 480
    hfov_deg: float = 79.0
    depth_min_m: float = 0.5
    depth_max_m: float = 5.0

