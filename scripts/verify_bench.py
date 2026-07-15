"""Offline benchmark for TargetVerifier against a hand-labeled set of real
crops pulled from verify_debug/ dumps (tests/fixtures/verify_bench).

Lets a prompt/model/threshold change be checked in seconds against known
ground truth instead of a 20-60 minute eval-mini run. See
docs/DESIGN_AND_ROADMAP.md P1c and tests/fixtures/verify_bench/labels.json
for how the set was built and why each case is interesting (mislabeled
detections, small/occluded crops, borderline confidence, etc).

Usage:
    python scripts/verify_bench.py                      # default vlm_model
    python scripts/verify_bench.py --model qwen2.5vl:3b  # compare models
    python scripts/verify_bench.py --accept-confidence 0.4
"""
from __future__ import annotations

import argparse
import json
import os
import types
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from osg.llm.client import ChatClient
from osg.verification.verifier import TargetVerifier

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "verify_bench"


def load_cases() -> list[dict]:
    data = json.loads((FIXTURES / "labels.json").read_text())
    return data["cases"]


def run(model: str, accept_confidence: float) -> list[dict]:
    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/v1"
    client = ChatClient(base_url, model, timeout_s=180.0, max_image_px=256)
    verifier = TargetVerifier(client, accept_confidence)  # debug_dir=None: no re-dumping

    results = []
    for case in load_cases():
        img = np.asarray(imageio.imread(FIXTURES / "images" / case["image"]))[..., :3]
        track = types.SimpleNamespace(best_crop=img)
        predicted = verifier.verify(track, case["target"])
        results.append({**case, "predicted": predicted})
    return results


def report(results: list[dict]) -> float:
    scored = [r for r in results if not r.get("ambiguous")]
    tp = sum(1 for r in scored if r["ground_truth"] and r["predicted"])
    fp = sum(1 for r in scored if not r["ground_truth"] and r["predicted"])
    fn = sum(1 for r in scored if r["ground_truth"] and not r["predicted"])
    tn = sum(1 for r in scored if not r["ground_truth"] and not r["predicted"])
    n = len(scored)
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")

    print(f"{'image':32s} {'target':12s} {'truth':6s} {'pred':6s} {'':3s} note")
    for r in results:
        truth = "ambig" if r.get("ambiguous") else str(r["ground_truth"])
        mark = "" if r.get("ambiguous") or r["ground_truth"] == r["predicted"] else "  <-- WRONG"
        print(f"{r['image']:32s} {r['target']:12s} {truth:6s} {str(r['predicted']):6s} {mark:3s} {r['note'][:60]}")

    print(f"\nscored (non-ambiguous): {n}  |  ambiguous (excluded): {len(results) - n}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"accuracy={accuracy:.3f}  precision={precision:.3f}  recall={recall:.3f}")
    return accuracy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5vl:7b")
    p.add_argument("--accept-confidence", type=float, default=0.5)
    p.add_argument("--min-accuracy", type=float, default=None,
                    help="exit 1 if accuracy falls below this (for CI-style gating)")
    args = p.parse_args()

    results = run(args.model, args.accept_confidence)
    accuracy = report(results)

    if args.min_accuracy is not None and accuracy < args.min_accuracy:
        raise SystemExit(f"accuracy {accuracy:.3f} below required {args.min_accuracy:.3f}")


if __name__ == "__main__":
    main()
