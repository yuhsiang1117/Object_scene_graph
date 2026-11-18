"""ConceptGraphs' own `generate_gsa_results.py`, run in two passes that fit a shared GPU.

The class-agnostic ConceptGraphs (`--class_set none`: SAM "segment all", then CLIP ViT-H-14
on each mask's padded crop) holds SAM ViT-H, CLIP ViT-H and GroundingDINO on the GPU at once,
about 10 GB. This runs the same code in two passes so that only one large model is resident:

  --pass masks   the script's own `main`, with GroundingDINO's constructor stubbed (`main`
                 builds it before looking at `--class_set` and never calls it under `none`)
                 and CLIP replaced by a stand-in that returns nothing; every frame's SAM
                 masks, boxes and confidences are written exactly as the script writes them;
  --pass clip    for every written frame, the script's own `compute_clip_features` with the
                 script's own CLIP model, on the same image and detections, and the three
                 outputs (`image_crops`, `image_feats`, `text_feats`) put in place.

Nothing is computed differently from one `generate_gsa_results.py` run. ConceptGraphs'
hard-coded TkAgg backend is swapped for Agg (no display), as baselines/hovsg/run_headless.py
does for HOV-SG.

Run in the ConceptGraphs env, from its `conceptgraph/` directory, with GSA_PATH set:

    python /workspace/scripts/baselines/conceptgraphs/run_detections.py --pass masks \\
        --dataset_root <export root> --dataset_config <root>/dataconfig.yaml \\
        --scene_id <scene> --class_set none --stride 1
    python /workspace/scripts/baselines/conceptgraphs/run_detections.py --pass clip \\
        --dataset_root <export root> --scene_id <scene>
"""
import gzip
import pickle
import runpy
import sys
from pathlib import Path


class _NoGroundingDINO:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        raise RuntimeError("GroundingDINO is not loaded: run_detections.py is for --class_set none")


class _NoCLIP:
    """Stands in for the CLIP model, preprocess and tokenizer in the masks pass; `main` only
    moves it to the device, and its one user (compute_clip_features) is stubbed too."""

    def to(self, *a, **k):
        return self


def _headless():
    import matplotlib

    use = matplotlib.use
    matplotlib.use = lambda backend, *a, **k: use("Agg" if str(backend).lower() == "tkagg" else backend, *a, **k)


def _pop(flag: str) -> str:
    i = sys.argv.index(flag)
    value = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return value


def pass_masks() -> None:
    if "--class_set" not in sys.argv or sys.argv[sys.argv.index("--class_set") + 1] != "none":
        raise SystemExit("run_detections.py only runs --class_set none")
    _headless()
    import groundingdino.util.inference as gdi
    import open_clip

    import conceptgraph.utils.model_utils as mu

    gdi.Model = _NoGroundingDINO
    open_clip.create_model_and_transforms = lambda *a, **k: (_NoCLIP(), None, None)
    open_clip.get_tokenizer = lambda *a, **k: _NoCLIP()
    mu.compute_clip_features = lambda image, detections, *a, **k: ([], [], [])
    script = Path.cwd() / "scripts" / "generate_gsa_results.py"
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")


def pass_clip() -> None:
    import argparse

    import cv2
    import numpy as np
    import open_clip
    import supervision as sv
    import torch
    from tqdm import tqdm

    from conceptgraph.utils.model_utils import compute_clip_features

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--scene_id", required=True)
    ap.add_argument("--detections", default="gsa_detections_none")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    # The same model, weights and tokenizer as generate_gsa_results.main.
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    clip_model = clip_model.to(args.device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    scene = args.dataset_root / args.scene_id
    files = sorted((scene / args.detections).glob("*.pkl.gz"))
    done = 0
    for f in tqdm(files):
        with gzip.open(f, "rb") as fh:
            res = pickle.load(fh)
        if len(res["image_feats"]) == len(res["xyxy"]) and len(res["xyxy"]):
            continue  # already has its features
        if not len(res["xyxy"]):
            # A frame without detections (the recording has them; SAM's "segment all" never
            # does): compute_clip_features cannot concatenate nothing, so the empty matrices.
            dim = clip_model.visual.output_dim
            res["image_crops"], res["image_feats"], res["text_feats"] = [], np.zeros((0, dim), np.float32), \
                np.zeros((0, dim), np.float32)
            with gzip.open(f, "wb") as fh:
                pickle.dump(res, fh)
            continue
        image = cv2.imread(str(scene / "results" / (f.name[:-len(".pkl.gz")] + ".jpg")))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        det = sv.Detections(xyxy=res["xyxy"], confidence=res["confidence"],
                            class_id=res["class_id"], mask=res["mask"])
        res["image_crops"], res["image_feats"], res["text_feats"] = compute_clip_features(
            image_rgb, det, clip_model, clip_preprocess, clip_tokenizer, res["classes"], args.device)
        with gzip.open(f, "wb") as fh:
            pickle.dump(res, fh)
        done += 1
    print(f"{args.scene_id}: CLIP features for {done} of {len(files)} frames")


def main() -> None:
    which = _pop("--pass") if "--pass" in sys.argv else ""
    if which == "masks":
        pass_masks()
    elif which == "clip":
        pass_clip()
    else:
        raise SystemExit("--pass masks|clip")


if __name__ == "__main__":
    main()
