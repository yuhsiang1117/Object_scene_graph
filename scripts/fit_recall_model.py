#!/usr/bin/env python3
"""Fit P(detected | present, view) from a logged eval run.

Every negative update the presence filter makes is sized by the detector's
recall in that view (objects/presence.py). A constant is unbiased but blunt: a
mug filling the frame at 1 m and a mug 40 px wide at 5 m must not produce the
same evidence when neither is detected. This fits the view-dependent form.

    # 1. log expectation features from an ordinary eval run
    python scripts/run_eval.py +experiment=ycb_authored_nav verification=off \
        scene_graph.presence.enabled=true \
        scene_graph.presence.log_path=outputs/recall/log.jsonl

    # 2. fit, and read the calibration table it prints
    python scripts/fit_recall_model.py outputs/recall/log.jsonl \
        -o outputs/recall/recall_model.json

    # 3. use it
    python scripts/run_eval.py ... \
        scene_graph.presence.recall_model_path=outputs/recall/recall_model.json

The label is `detected`: was a detection associated to this track on a keyframe
where the geometry said one was expected. That makes the fit self-supervised --
no annotation, no extra runs.

Newton/IRLS logistic regression in numpy; no new dependency for a 4-parameter
fit. Ridge term keeps it finite when a feature is degenerate (e.g. every logged
row at the same depth).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

FEATURES = ("bias", "log_area_px", "depth_m", "cos_incidence")


def load_rows(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y, labels = [], [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            X.append(
                [
                    1.0,
                    math.log(max(float(row["area_px"]), 1.0)),
                    float(row["depth_m"]),
                    float(row["cos_incidence"]),
                ]
            )
            y.append(float(row["detected"]))
            labels.append(str(row.get("label", "")))
    if not X:
        raise SystemExit(f"no rows in {path} -- was scene_graph.presence.log_path set?")
    return np.asarray(X), np.asarray(y), labels


def fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3, iters: int = 50) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        grad = X.T @ (y - p) - ridge * w
        s = np.clip(p * (1.0 - p), 1e-6, None)
        H = -(X.T * s) @ X - ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, grad)
        w_new = w - step
        if np.max(np.abs(w_new - w)) < 1e-9:
            w = w_new
            break
        w = w_new
    return w


def calibration(X: np.ndarray, y: np.ndarray, w: np.ndarray, bins: int = 8) -> str:
    """Reliability table: predicted vs observed. This is what makes the negative
    updates defensible -- if the two columns diverge, the filter is mis-sizing
    its evidence and no amount of tuning downstream will fix it."""
    p = 1.0 / (1.0 + np.exp(-X @ w))
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    lines = [f"  {'predicted':>10}  {'observed':>9}  {'n':>6}"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi)
        if m.sum() < 5:
            continue
        lines.append(f"  {p[m].mean():10.3f}  {y[m].mean():9.3f}  {int(m.sum()):6d}")
    brier = float(np.mean((p - y) ** 2))
    base = float(np.mean((y.mean() - y) ** 2))
    lines.append(f"  Brier {brier:.4f} vs {base:.4f} for a constant predictor")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, help="JSONL written by PresenceFilter.log_path")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--min-rows", type=int, default=200,
                    help="refuse to fit on too little evidence")
    args = ap.parse_args()

    X, y, labels = load_rows(args.log)
    print(f"{len(y)} expectations, detection rate {y.mean():.3f}")
    if len(y) < args.min_rows:
        raise SystemExit(
            f"only {len(y)} rows (< {args.min_rows}); a constant recall is more honest "
            "than a fit on this much data"
        )
    if y.min() == y.max():
        raise SystemExit(
            "every row has the same outcome; nothing to fit -- keep the constant"
        )

    w = fit_logistic(X, y)
    print("weights:")
    for name, value in zip(FEATURES, w):
        print(f"  {name:>14} {value:+.4f}")
    print("calibration:")
    print(calibration(X, y, w))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"features": list(FEATURES), "weights": [float(v) for v in w],
             "n_rows": int(len(y)), "detection_rate": float(y.mean())},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
