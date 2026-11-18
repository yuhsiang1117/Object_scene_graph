"""Class-agnostic region proposal: find the object the detector will not name."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RegionProposalConfig:
    """Off by default; every field is inert while `enabled` is False.

    Measured justification for each number is in
    `src/osg/perception/region_proposer.py` and `docs/history/PERCEPTION_GAP.md`.
    """

    enabled: bool = False
    weights: str = "outputs/dualmap_comparison/vendor/DualMap/model/FastSAM-s.pt"
    backbone: str = "mobileclip_s2"     # beat ViT-B/32 at every setting probed
    checkpoint: str = ""                # mobileclip resolves from the HF cache
    device: str = "cuda"
    imgsz: int = 1024
    conf: float = 0.25
    iou: float = 0.7
    # Admission. 0.24 is the knee of the precision curve measured over 166
    # frames: 40 correct against 7 wrong, 85% precision, 24% recall. At 0.22
    # precision falls to 64%, at 0.28 recall falls to 9 frames.
    tau: float = 0.24
    # A detection's score is what every downstream gate is calibrated against,
    # so an admitted region is handed the detector's own "good enough" value
    # rather than its cosine, which would read as almost-certainly-false.
    admit_score: float = 0.50
    # Padding the box by a quarter of its longer side, with the natural phrasing
    # in region_proposer.NATURAL, moved overall rank-1 from 52% to 70%.
    pad_frac: float = 0.25
    # The size band a proposal must fall in to be a candidate for the object
    # rather than the furniture it rests on.
    min_area_px: float = 200.0
    max_area_frac: float = 0.02
    max_regions: int = 64
    # Run the stage only when the label path found nothing for the target on
    # this keyframe, and at most this many times per episode. FastSAM is 13 ms
    # a frame but encoding ~60 crops is not, and an unbounded stage would admit
    # on 15% of frames at 85% precision, which is a lot of wrong tracks.
    # The gate, and the thing that decides whether this stage is a fallback or
    # a competitor. `only_when_unnamed` is evaluated PER FRAME, and on a trial
    # where the detector works the target is absent from most individual
    # frames -- so the stage fired on nearly all of them, admitting ~13.5
    # regions an episode and competing with a detector that was about to
    # succeed. Measured on the full 107: the perception subset went 0 -> 6 and
    # everything else went 57 -> 41, with 13 of the 18 lost trials exhausting
    # all three attempts.
    #
    # `require_never_named` makes the condition EPISODE-level instead: once the
    # label path has named the target even once, this object is one the
    # detector can see and the stage stays off for the rest of the episode.
    # `unnamed_keyframes` is the grace period before it activates at all.
    only_when_unnamed: bool = True
    require_never_named: bool = True
    unnamed_keyframes: int = 20
    # Let the absence sensor see the proposals too.
    #
    # `_best_target_detection` re-runs the RAW detector and filters by label,
    # and it is what the close look asks before concluding a committed track is
    # absent. That is the same detector that could not name the object -- which
    # is exactly why the trial is in this bucket -- so an absence verdict
    # reached that way is not evidence. Measured: in_anchor__0117__banana
    # walked to the banana, looked from 1.5 m, was told "not detected",
    # abandoned the track (p 0.95 -> 0.433) and committed elsewhere, while the
    # proposal stage had admitted 16 regions in the same episode.
    use_for_absence: bool = True
    max_per_episode: int = 40
    # The fusion that replaces the gates above. `every_keyframe` runs the stage
    # on every keyframe the detector did not name the target on -- the same
    # regime DualMap's class-agnostic segmentation runs in -- and the episode
    # gates (`require_never_named`, `unnamed_keyframes`) no longer apply; only
    # `max_per_episode` still does. What keeps the phantoms out is no longer
    # WHEN the stage runs but how its output is judged: a proposal observation
    # carries its region's feature, the track keeps a running mean over views,
    # and a track only proposals have seen may become a candidate only with
    # `commit_min_obs` observations and a mean-feature cosine of at least
    # `commit_tau` against the query phrase, and then only behind every track
    # the detector has named. `commits: false` is the observation arm: the
    # stage runs and its tracks are recorded, but none may be committed to --
    # the run that sets the two numbers above from true-vs-phantom tracks
    # rather than from per-frame precision.
    every_keyframe: bool = False
    commits: bool = True
    commit_min_obs: int = 1
    commit_tau: float = -1.0
    # A per-class bar over `commit_tau`, keyed by the normalised target label
    # ("bowl", "red_plate", "mug"). Calibrated OFF-RUN, on the offline
    # false-admission probe (docs/history/REGION_FALSE_ADMISSION.md): the classes whose
    # text vector admits on more than 15% of object-free frames -- bowl 42%,
    # plate 26%, mug 16% -- are the classes whose phantoms cleared 0.28 on the
    # full 107 (13 of 15), while every true track of theirs that scored sat
    # at or above 0.30. Empty means one bar for every class.
    commit_tau_by_class: dict = field(default_factory=dict)
