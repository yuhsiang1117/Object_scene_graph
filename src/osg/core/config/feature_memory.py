"""Matching an object by what it looks like, the way DualMap does.

Measured before it was written (`docs/history/FEATURE_MEMORY_PROBE.md`). Two findings
shape every default here.

The first is that a threshold does not work and an argmax does. Asked for an
absolute cosine that admits the mug without admitting the other fifty regions in
its frame, the answer was no threshold exists. DualMap never asks: on reaching an
anchor it takes `sorted_candidates[0]` over the objects near it and goes. So this
config has no admission bar. It has a radius, a minimum observation count, and a
budget.

The second is that our detector does propose the object, under the wrong name.
On the 160 dumped mug frames a box small enough to BE the mug covers it on 39,
labelled `towel` 27 times -- our equivalent of the `speaker` detection DualMap's
mug result rests on. Ranking every detection in a frame by cosine to "a photo of
a mug" picks that box on 27.8% of frames in the 2.5-4 m band. Per frame that is
weak; DualMap wins because it ranks 3D objects accumulated over several views
and scoped to one piece of furniture, which is what `local_pick_on_arrival` does.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeatureMemoryConfig:
    # Everything below is inert while this is False, and the label path is then
    # byte-identical -- the benchmark's runs are exactly reproducible and every
    # scored result on it predates this file.
    enabled: bool = False

    # MobileCLIP-S2 is DualMap's own and clearly the better of the two measured:
    # named targets reached the top five of a 689-track map 11 times in 14
    # against ViT-B/32's 6. It loads through open_clip from the HuggingFace
    # cache; `checkpoint` is read only by the `vitb32` backbone, whose weights
    # are the file already staged for the value map.
    backbone: str = "mobileclip_s2"
    checkpoint: str = "data/clip/ViT-B-32.pt"
    device: str = "auto"  # "auto" follows detector.device
    half: bool = True
    prompt_template: str = "a photo of a {target}"

    # The local stage. On arriving at a searched surface without a candidate,
    # rank the live tracks near it by cosine against the query text and commit
    # to the best one. No threshold: the pick always names its best guess, and
    # the identity channel, the absence sensor and the attempt protocol judge it
    # exactly as they judge a candidate the label path proposed.
    local_pick_on_arrival: bool = True
    # DualMap scopes its inquiry with `filter_objects_in_global_bbox`, the
    # anchor's own bounding box grown 10%. Our containers carry a centre and a
    # footprint area rather than a box, so the scope is a radius about the
    # centre; 2.0 m covers a large desk or bed from its middle.
    local_radius_m: float = 2.0
    # DualMap requires `observed_num > 2`. A feature averaged over three views is
    # a far quieter signal than one 2 m crop, and this is the cheapest half of
    # why its per-object match beats a per-frame one.
    local_min_obs: int = 3
    # How many times an episode may commit on appearance alone. Unbounded, a
    # wrong pick that survives its absence reading can be re-picked at every
    # surface; two leaves room for a second opinion without spending the budget.
    max_local_picks: int = 2

    # The pick may only consider tracks small enough to BE the thing asked for.
    #
    # Without it the argmax is furniture: over 24 mug and scissors trials the
    # first version picked a pillow 13 times, a lamp 5, a bench 4 -- 37 of 44
    # picks were furniture-sized -- because a bed's crop is a perfectly good
    # match for "a photo of a mug" next to a 40-pixel smudge of the real one.
    # DualMap never faces this: its detector proposes the small object (its mug
    # arrives labelled `speaker`), so its local map holds tight crops and the
    # argmax runs over things that are already object-shaped. Ours mostly
    # proposes the furniture the object rests on, so the bound is applied here,
    # from mapped geometry alone -- no ground truth, no per-target tuning.
    #
    # 0.6 m is deliberately loose: every movable query in this benchmark is
    # under 0.3 m across, so this rejects beds and counters without pretending
    # to know which object is being sought.
    local_max_extent_m: float = 0.6

    # Fuse the appearance score with the presence belief before the argmax.
    #
    # This is the one place we have something DualMap does not. Its memory of a
    # failed candidate is an ignore list -- belief set to zero, discarded when
    # the query ends -- so its inquiry cannot express "I have walked past that
    # thing four times and never seen it again". Ours can: presence.p is a
    # graded, recoverable belief that something is still where the map says, and
    # it is exactly the channel an appearance match lacks. A track that looks
    # like the query but has been quietly decaying should lose to one that looks
    # slightly less like it and is still believed.
    #
    # The cosine is not a probability and must not be multiplied as one -- CLIP
    # text-image scores sit in a narrow band, so the informative quantity is the
    # gap to the best candidate. The appearance side is therefore
    # exp(-beta * (best - sim)), floored, and presence multiplies that.
    fuse_presence: bool = False
    # A 0.05 cosine gap halves the appearance weight. The true-versus-runner-up
    # gaps measured over the prior maps were 0.01-0.10, so this makes the two
    # channels comparable rather than letting either dominate outright.
    presence_beta: float = 14.0
    presence_floor: float = 0.1

    # One batched encode per keyframe over the admitted detections, capped so a
    # crowded frame cannot stall the control loop.
    max_live_crops_per_keyframe: int = 32
