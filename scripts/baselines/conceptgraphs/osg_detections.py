"""The recording's own detections, written as ConceptGraphs detections for the exported frames.

For "same detections, ConceptGraphs' mapper": every frame export_recording.py exported gets a
`gsa_detections_osg/frame%06d.pkl.gz` holding exactly the detections the object layer received
on that keyframe (eval/object_recorder.py: label, score, box, full-resolution mask), in the
format generate_gsa_results.py writes, plus `gsa_classes_osg.json`, the scene's label set. The
CLIP features are added afterwards by `run_detections.py --pass clip --detections
gsa_detections_osg`, with ConceptGraphs' own model and crop code.

    python scripts/baselines/conceptgraphs/osg_detections.py --rec outputs/objrec_v2 \\
        --cg outputs/baselines/cg_rec --scene 00800-TEEsavR23oF
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True, type=Path)
    ap.add_argument("--cg", required=True, type=Path)
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()

    scene = args.cg / args.scene
    index = json.loads((scene / "frames.json").read_text())["frames"]
    kfs = {}
    paths = []
    for ep, k in index:
        if ep not in kfs:
            kfs[ep] = sorted((args.rec / ep).glob("kf_*.npz"))
        paths.append(kfs[ep][k])
    classes = sorted({lab for p in paths for lab in json.loads(str(np.load(p)["labels"]))})
    out = scene / "gsa_detections_osg"
    out.mkdir(exist_ok=True)
    n = 0
    for i, p in enumerate(paths):
        z = np.load(p)
        labels = json.loads(str(z["labels"]))
        mh, mw = (int(v) for v in z["mask_shape"])
        masks = np.stack([np.unpackbits(z["masks"][j], count=mh * mw).reshape(mh, mw).astype(bool)
                          for j in range(len(labels))]) if labels else np.zeros((0, mh, mw), bool)
        res = {
            "xyxy": np.asarray(z["bboxes"], np.float32).reshape(-1, 4),
            "confidence": np.asarray(z["scores"], np.float32).reshape(-1),
            "class_id": np.array([classes.index(lab) for lab in labels], dtype=int),
            "mask": masks,
            "classes": classes,
            "image_crops": [],
            "image_feats": [],
            "text_feats": [],
        }
        with gzip.open(out / f"frame{i:06d}.pkl.gz", "wb") as fh:
            pickle.dump(res, fh)
        n += len(labels)
    (scene / "gsa_classes_osg.json").write_text(json.dumps(classes))
    print(f"{args.scene}: {len(paths)} frames, {n} detections, {len(classes)} labels")


if __name__ == "__main__":
    main()
