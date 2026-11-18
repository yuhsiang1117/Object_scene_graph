# Vendored RedNet (MPCAT40 RGB-D semantic segmentation)

`rednet_model.py` is copied from ASCENT's `RedNet/RedNet_model.py`, which is in
turn the RedNet used by SemExp / habitat semantic-mapping work. The only edit is
the removal of `import RedNet.utils`, which that file never used (it uses
`torch.utils.model_zoo` and `torch.utils.checkpoint`).

## Why it is here

This is the up-stair signal. OSG's own S14a stage pre-registered the rule:

> if UP-stair YOLOE recall rises materially with pitch, the 24% is a viewpoint
> problem and a look_up probe fixes it (S14b, cheap). If it does not, the
> detector is the problem and only a dedicated segmentation model will move it
> (S14c, RedNet, ~200 MB).

S32 measured it -- 10.4% level, 7.2% at +30, 3.2% at -30. Pitch does not help,
so the rule points at S14c, and ASCENT is running exactly this model
(`ascent_policy.py:151`, `pretrained_weights/rednet_semmap_mp3d_40.pth`).

## Weights

`data/weights/rednet_semmap_mp3d_40.pth`, staged by

    python scripts/download_weights.py --rednet

Checkpoint layout: `{"epoch", "iter", "model_state", ...}` with a `module.`
prefix from DataParallel, 928 tensors, `final_deconv_custom` emitting 40
channels -- one per MPCAT40 class.

## Class ids

`RedNetResizeWrapper.forward` returns `argmax + 1`, so classes are 1..40 and
anything scoring under 0.8 is forced to 1. Stairs are **17** (MPCAT40 index 16,
plus the offset) -- ASCENT's `constants.py:214 STAIR_CLASS_ID = 17`.

## Inputs

The wrapper wants RGB as `B x H x W x 3` in 0..255 and depth as `B x H x W x 1`
**normalised to [0, 1]**, not metres: it applies `mean=0.213, std=0.285` and
takes `depth < 1.0` as the valid mask. OSG's `FrameData.depth` is in metres, so
`stair_seg.py` normalises against the same `[depth_min_m, depth_max_m]` range
the PointNav mover uses.
