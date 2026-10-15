#!/usr/bin/env python3
"""Can anything tell the target from what the detector mistakes for it?

Reads the labelled detections `scripts/build_verifier_set.py` writes and scores
each candidate verifier on them. A verifier's job at a STOP is binary, so each
is reported as the two numbers the presence filter needs (Eq. 3 of the paper,
`objects/presence.py`): r = P(accept | the target) and q = P(accept | a false
positive), split by range -- arrivals are judged from 0.5-1.5 m, glances from
further -- and by the false positive's kind (on another benchmark object, or on
the furniture). Continuous scores also get a rank AUC.

  score       the detector's own confidence (what the agent uses now)
  clip_bg     MobileCLIP-S2 margin: cos(crop, "a photo of a <query>") minus the
              best of the detector's background vocabulary. Target-agnostic.
  clip_all    the same, with every other benchmark query among the confusers.
              Knows the query set, as DualMap's class list does.
  vlm[m]         the pipeline's VLM verifier (`verification/verifier.py`) on model
                 m, yes/no on the full frame with the candidate boxed
  vlm_choice[m]  the same, forced choice between the query and the background
                 vocabulary (+ "none")

    python scripts/eval_verifiers.py outputs/verifier_set
    python scripts/eval_verifiers.py outputs/verifier_set --vlm-n 150 \
        --vlm-model meta/llama-3.2-11b-vision-instruct \
        --vlm-model nvidia/nemotron-3.5-super-vl-120b-a12b

VLM rows call a hosted model (cfg `verification.vlm_model` on NIM) and are
capped by `--vlm-n` per class; `--no-vlm` skips them.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

RANGES = ((0.0, 1.5, "<=1.5 m"), (1.5, 3.0, "1.5-3 m"), (3.0, 1e9, ">3 m"))


def load(root: Path) -> List[dict]:
    rows = [json.loads(l) for l in (root / "set.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if r.get("range_m") is not None]


def rank_auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """P(score of a random target > score of a random false positive)."""
    if not pos or not neg:
        return None
    neg_sorted = np.sort(np.asarray(neg, dtype=float))
    total = 0.0
    for p in pos:
        lo = np.searchsorted(neg_sorted, p, side="left")
        hi = np.searchsorted(neg_sorted, p, side="right")
        total += lo + 0.5 * (hi - lo)
    return float(total / (len(pos) * len(neg_sorted)))


def fp_kind(row: dict) -> str:
    return "on_ycb" if float(row.get("other_ycb_iou", 0.0)) > 0.1 else "on_scene"


def read_rgb(path: Path) -> np.ndarray:
    import cv2

    return cv2.imread(str(path))[..., ::-1].copy()


# ------------------------------------------------------------------ verifiers

def clip_margins(rows: List[dict], root: Path, background: List[str],
                 queries: List[str]) -> None:
    from osg.perception.feature_encoder import FeatureEncoder

    encoder = FeatureEncoder(backbone="mobileclip_s2", device="cuda")
    from osg.agent.stop_gate import background_for

    feats = encoder.encode_images([read_rgb(root / r["crop"]) for r in rows])
    prompt = "a photo of a {}"
    for row, f in zip(rows, feats):
        query = row["label"]
        # The gate's own confusers (`stop_gate.background_for`), so this margin
        # is the number the online gate thresholds.
        bg = encoder.encode_text([prompt.format(c) for c in background_for(query, background)])
        t = encoder.encode_text([prompt.format(query)])[0]
        s_t = float(f @ t)
        s_bg = float(np.max(bg @ f))
        others = [q for q in queries if q != query]
        s_all = s_bg
        if others:
            oth = encoder.encode_text([prompt.format(q) for q in others])
            s_all = max(s_bg, float(np.max(oth @ f)))
        row["clip_bg"] = s_t - s_bg
        row["clip_all"] = s_t - s_all


def vlm_answers(rows: List[dict], root: Path, cfg, background: List[str],
                model: str) -> List[str]:
    """Ask `model` about every row; returns the two row keys it wrote."""
    from osg.llm.client import ChatClient
    from osg.verification.verifier import VLMVerifier

    short = model.split("/")[-1]
    client = ChatClient(
        (cfg.verification.base_url or cfg.llm.base_url), model,
        (getattr(cfg.verification, "api_key", "") or cfg.llm.api_key),
        cfg.llm.timeout_s, cfg.llm.max_image_px, cfg.llm.send_response_format,
    )
    yesno = VLMVerifier(client, accept_confidence=cfg.verification.accept_confidence,
                        choice_mode=False)
    choice = VLMVerifier(client, accept_confidence=cfg.verification.accept_confidence,
                         choice_mode=True,
                         categories=[c.replace("_", " ") for c in background] + ["none"])
    from osg.agent.stop_gate import background_for

    for i, row in enumerate(rows):
        frame = read_rgb(root / row["frame"])
        box = np.asarray(row["bbox_xyxy"], dtype=float)
        choice.categories = background_for(row["label"], background) + ["none"]
        for name, verifier in ((f"vlm[{short}]", yesno), (f"vlm_choice[{short}]", choice)):
            errors = verifier.n_errors
            t0 = time.time()
            accepted = verifier.verify_bbox(frame, box, row["label"])
            row[name + "_s"] = round(time.time() - t0, 2)
            # The verifier fails OPEN on an API error; an error is not an answer.
            row[name] = None if verifier.n_errors > errors else bool(accepted)
        if (i + 1) % 20 == 0:
            print(f"  {short} {i + 1}/{len(rows)}", flush=True)
    return [f"vlm[{short}]", f"vlm_choice[{short}]"]


# -------------------------------------------------------------------- report

def binary_rates(rows: List[dict], key: str) -> Dict[str, Optional[float]]:
    pos = [r[key] for r in rows if r["true"] and r.get(key) is not None]
    neg = [r[key] for r in rows if not r["true"] and r.get(key) is not None]
    return {
        "r": (sum(pos) / len(pos)) if pos else None, "n_true": len(pos),
        "q": (sum(neg) / len(neg)) if neg else None, "n_fp": len(neg),
    }


def fmt(x: Optional[float], digits: int = 2) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def report(rows: List[dict], vlm_keys: Sequence[str] = ()) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    print(f"\n{len(rows)} detections: {sum(r['true'] for r in rows)} on the target, "
          f"{sum(not r['true'] for r in rows)} false positives "
          f"({sum(1 for r in rows if not r['true'] and fp_kind(r) == 'on_ycb')} on another "
          f"benchmark object)\n")
    print("| verifier | range | AUC | r = P(accept|target) | q = P(accept|FP) | "
          "q, FP on scene | q, FP on another YCB object | n target / n FP |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for key, threshold in ([("score", None), ("clip_bg", 0.0), ("clip_all", 0.0)]
                           + [(k, None) for k in vlm_keys]):
        if not any(r.get(key) is not None for r in rows):
            continue
        for lo, hi, name in RANGES:
            part = [r for r in rows if lo <= r["range_m"] < hi and r.get(key) is not None]
            if not part:
                continue
            if isinstance(part[0][key], bool):
                acc = {id(r): r[key] for r in part}
                auc = None
            else:
                pos = [r[key] for r in part if r["true"]]
                neg = [r[key] for r in part if not r["true"]]
                auc = rank_auc(pos, neg)
                # The detector already thresholded what reached the set, so its
                # own operating point accepts everything; a margin accepts > 0.
                acc = {id(r): (True if threshold is None else r[key] > threshold) for r in part}
            tagged = [dict(r, _acc=acc[id(r)]) for r in part]
            rates = binary_rates(tagged, "_acc")
            scene = binary_rates([r for r in tagged if r["true"] or fp_kind(r) == "on_scene"], "_acc")
            ycb = binary_rates([r for r in tagged if r["true"] or fp_kind(r) == "on_ycb"], "_acc")
            print(f"| {key} | {name} | {fmt(auc)} | {fmt(rates['r'])} | {fmt(rates['q'])} | "
                  f"{fmt(scene['q'])} | {fmt(ycb['q'])} | {rates['n_true']} / {rates['n_fp']} |")
            out[f"{key}/{name}"] = {"auc": auc, **rates, "q_scene": scene["q"], "q_ycb": ycb["q"]}
    print()
    for key in vlm_keys:
        latency = [r[key + "_s"] for r in rows if r.get(key + "_s") is not None]
        errors = sum(1 for r in rows if key + "_s" in r and r.get(key) is None)
        if latency:
            print(f"{key}: median {np.median(latency):.2f} s, p90 "
                  f"{np.percentile(latency, 90):.2f} s per call, {len(latency)} calls, "
                  f"{errors} errors")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set_dir", type=Path)
    ap.add_argument("--experiment", default="mfv2_osg",
                    help="the preset whose VLM / LLM settings to use")
    ap.add_argument("--max-range", type=float, default=1e9,
                    help="drop detections further than this before anything else")
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--vlm-model", action="append", default=[],
                    help="repeatable; default the preset's verification.vlm_model")
    ap.add_argument("--vlm-n", type=int, default=150,
                    help="VLM calls are capped at this many targets AND this many FPs")
    ap.add_argument("--vlm-max-range", type=float, default=3.0,
                    help="ask the VLM only about detections this close (the STOP regime)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, help="write the per-detection rows here")
    args = ap.parse_args(argv)

    from hydra import compose, initialize_config_dir

    from osg.core.config import register_configs

    register_configs()
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"+experiment={args.experiment}"])
    background = [str(v) for v in cfg.detector.vocabulary]
    rows = [r for r in load(args.set_dir) if r["range_m"] <= args.max_range]
    queries = sorted({r["label"] for r in rows})
    if not args.no_clip:
        clip_margins(rows, args.set_dir, background, queries)
    vlm_keys: List[str] = []
    if not args.no_vlm:
        near = [r for r in rows if r["range_m"] <= args.vlm_max_range]
        rng = random.Random(args.seed)
        pos = [r for r in near if r["true"]]
        neg = [r for r in near if not r["true"]]
        rng.shuffle(pos)
        rng.shuffle(neg)
        if not os.environ.get("NVIDIA_API_KEY") and "nvidia" in str(cfg.verification.base_url):
            print("NVIDIA_API_KEY is not set; skipping the VLM rows", file=sys.stderr)
        else:
            for model in (args.vlm_model or [str(cfg.verification.vlm_model)]):
                vlm_keys += vlm_answers(pos[: args.vlm_n] + neg[: args.vlm_n],
                                        args.set_dir, cfg, background, model)
    out = report(rows, vlm_keys)
    if args.json:
        args.json.write_text(json.dumps({"summary": out, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
