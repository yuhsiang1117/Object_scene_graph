"""P1h follow-up: does more descriptive prompt engineering for the class
text (still fed through YOLOE's existing set_classes/get_text_pe path, no
model changes) improve sofa/armchair discrimination, now that a plain
vocabulary addition ("armchair") was shown NOT to help (docs/
DESIGN_AND_ROADMAP.md P1h)?

Runs several class-phrase variants against the same saved crop images:
the 25 known sofa-hallucination crops (mostly a real armchair, per P1h's
majority-GT-category labeling) plus real true-positive sofa crops (dumped
by detector_gt_check.py's --dump-hallucinations run, TP-labeled files),
to check both directions: does the new phrasing stop calling the armchair
"sofa", AND does it still correctly call the real sofa "sofa"?

Usage:
  python scripts/prompt_experiment.py --crops-dir <dir with *.png files
      named ..._actual-<cat>_... or ..._TP-<target>_...>
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import cv2
import numpy as np

from osg.perception.detector import YoloeDetector

# variant name -> {canonical_key: prompt phrase actually fed to the detector}
VARIANTS = {
    "baseline": {"sofa": "sofa", "armchair": "armchair"},
    "plus_word": {"sofa": "sofa couch", "armchair": "armchair recliner chair"},
    "seat_count": {"sofa": "multi-seat sofa couch", "armchair": "single-seat armchair"},
    "long_phrase": {
        "sofa": "a large sofa couch for multiple people to sit on",
        "armchair": "a small armchair for only one person to sit on",
    },
    "armchair_only": {"sofa": "sofa", "armchair": "recliner lounge armchair with one seat"},
}


def load_labeled_crops(crops_dir: str):
    """Returns list of (path, true_label) where true_label is 'armchair' or
    'sofa' (skips crops whose GT/verdict label is neither, e.g. piano/wall
    hallucinations -- not useful for a binary sofa-vs-armchair test)."""
    out = []
    for path in sorted(glob.glob(os.path.join(crops_dir, "*.png"))):
        name = os.path.basename(path)
        m_actual = re.search(r"actual-(.+?)_score", name)
        m_tp = re.search(r"_TP-(.+?)_iou", name)
        if m_actual and m_actual.group(1) == "armchair":
            out.append((path, "armchair"))
        elif m_tp and m_tp.group(1) == "sofa":
            out.append((path, "sofa"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-dir", required=True)
    ap.add_argument("--conf", type=float, default=0.1)
    args = ap.parse_args()

    samples = load_labeled_crops(args.crops_dir)
    n_armchair = sum(1 for _, l in samples if l == "armchair")
    n_sofa = sum(1 for _, l in samples if l == "sofa")
    print(f"loaded {len(samples)} labeled crops ({n_armchair} armchair-GT, {n_sofa} sofa-GT)")
    if not samples:
        print("no usable crops found -- check --crops-dir and filename patterns")
        return

    detector = YoloeDetector(weights="data/weights/yoloe-11s-seg.pt", conf=args.conf)

    print()
    print(f"{'variant':14s} {'armchair->armchair':19s} {'armchair->sofa':15s} "
          f"{'armchair->neither':18s} {'sofa->sofa':11s} {'sofa->armchair':15s} {'sofa->neither':14s}")
    for variant_name, phrase_map in VARIANTS.items():
        rev = {v: k for k, v in phrase_map.items()}
        detector.set_vocabulary(list(phrase_map.values()))

        counts = defaultdict(int)
        for path, true_label in samples:
            img = cv2.imread(path)
            rgb = img[..., ::-1]
            dets = detector.detect(rgb)
            if not dets:
                counts[(true_label, "neither")] += 1
                continue
            best = max(dets, key=lambda d: d.score)
            pred_canonical = rev.get(best.label.strip().lower(), "neither")
            # normalize() lowercases + strips underscores; match loosely since
            # get_text_pe/model.names may reformat the class phrase slightly
            if pred_canonical == "neither":
                for phrase, canon in rev.items():
                    if phrase in best.label.strip().lower() or best.label.strip().lower() in phrase:
                        pred_canonical = canon
                        break
            counts[(true_label, pred_canonical)] += 1

        print(f"{variant_name:14s} "
              f"{counts[('armchair', 'armchair')]:19d} {counts[('armchair', 'sofa')]:15d} "
              f"{counts[('armchair', 'neither')]:18d} {counts[('sofa', 'sofa')]:11d} "
              f"{counts[('sofa', 'armchair')]:15d} {counts[('sofa', 'neither')]:14d}")


if __name__ == "__main__":
    main()
