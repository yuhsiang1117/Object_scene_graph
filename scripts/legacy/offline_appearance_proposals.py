#!/usr/bin/env python3
"""Class-agnostic proposal replay for stale-object appearance matching.

FastSAM generates regions from frozen RGB without a query or ground truth.
Runner-only projected target pixels are consulted afterward solely to measure
whether a proposal covered the target and where the best such proposal ranked.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from offline_appearance_retrieval import (
    ImageEncoder,
    YCB_EXTENT_M,
    dump_frames,
    load_records,
    reference_crops,
    summarize,
)


def proposal_crops(
    model: Any,
    image: Image.Image,
    *,
    imgsz: int,
    confidence: float,
    iou: float,
    device: str,
    min_area_px: float,
    max_area_fraction: float,
    padding_fraction: float,
) -> Tuple[List[Image.Image], List[np.ndarray], List[List[float]]]:
    result = model.predict(
        np.asarray(image),
        imgsz=imgsz,
        conf=confidence,
        iou=iou,
        retina_masks=True,
        verbose=False,
        device=device,
    )[0]
    if result.boxes is None or result.masks is None:
        return [], [], []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    masks = result.masks.data.detach().cpu().numpy().astype(bool)
    width, height = image.size
    max_area = max_area_fraction * width * height
    crops: List[Image.Image] = []
    kept_masks: List[np.ndarray] = []
    kept_boxes: List[List[float]] = []
    for box, mask in zip(boxes, masks):
        x1, y1, x2, y2 = [float(v) for v in box]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < min_area_px or area > max_area:
            continue
        pad = padding_fraction * max(x2 - x1, y2 - y1)
        crop_box = (
            int(np.floor(x1 - pad)), int(np.floor(y1 - pad)),
            int(np.ceil(x2 + pad)), int(np.ceil(y2 + pad)),
        )
        crops.append(image.crop(crop_box).convert("RGB"))
        kept_masks.append(mask)
        kept_boxes.append([x1, y1, x2, y2])
    return crops, kept_masks, kept_boxes


def is_target_proposal(mask: np.ndarray, box: Sequence[float], u: int, v: int) -> bool:
    if not (box[0] - 4 <= u <= box[2] + 4 and box[1] - 4 <= v <= box[3] + 4):
        return False
    height, width = mask.shape
    return 0 <= u < width and 0 <= v < height and bool(mask[v, u])


def metric_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    proposed = [r for r in rows if r["proposal_recall"]]
    retrieval_rows = [
        {
            **r,
            "positive_score": r["best_target_score"],
            "negative_scores": r["distractor_scores"],
            "rank": r["rank"],
        }
        for r in proposed
    ]
    summary = summarize(retrieval_rows)
    summary.update({
        "all_frames": len(rows),
        "raw_center_cover_frames": sum(bool(r["raw_center_cover"]) for r in rows),
        "raw_center_cover": (
            sum(bool(r["raw_center_cover"]) for r in rows) / len(rows) if rows else 0.0
        ),
        "proposal_recall_frames": len(proposed),
        "proposal_recall": len(proposed) / len(rows) if rows else 0.0,
        "end_to_end_top1": sum(r["rank"] == 1 for r in proposed) / len(rows) if rows else 0.0,
        "end_to_end_top5": sum(r["rank"] <= 5 for r in proposed) / len(rows) if rows else 0.0,
        "mean_proposals": float(np.mean([r["proposal_count"] for r in rows])) if rows else 0.0,
    })
    return summary


def grouped(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: metric_rows(items) for name, items in sorted(groups.items())}


def pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def add_table(lines: List[str], values: Dict[str, Dict[str, Any]]) -> None:
    lines += [
        "| Slice | Frames | Raw centre cover | Plausible proposal recall | Conditional Top-1 | Conditional Top-5 | End-to-end Top-1 | End-to-end Top-5 | Median rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in values.items():
        lines.append(
            f"| {name} | {row.get('all_frames', 0)} | {pct(row.get('raw_center_cover'))} | "
            f"{pct(row.get('proposal_recall'))} | "
            f"{pct(row.get('top1'))} | {pct(row.get('top5'))} | "
            f"{pct(row.get('end_to_end_top1'))} | {pct(row.get('end_to_end_top5'))} | "
            f"{row.get('median_rank', 'n/a')} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/osg_dualmap_tightring"))
    parser.add_argument("--maps", type=Path, default=Path("outputs/maps_v5"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/clip/ViT-B-32.pt"))
    parser.add_argument(
        "--proposal-model", type=Path,
        default=Path("outputs/dualmap_comparison/vendor/DualMap/model/FastSAM-s.pt"),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames-per-episode", type=int, default=8)
    parser.add_argument("--reference-radius-m", type=float, default=0.75)
    parser.add_argument("--reference-limit", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--proposal-confidence", type=float, default=0.25)
    parser.add_argument("--proposal-iou", type=float, default=0.7)
    parser.add_argument("--min-area-px", type=float, default=144.0)
    parser.add_argument("--max-area-fraction", type=float, default=0.35)
    parser.add_argument("--padding-fraction", type=float, default=0.10)
    parser.add_argument(
        "--target-max-area-ratio", type=float, default=16.0,
        help="evaluation-only: largest proposal box area relative to the projected square of the known YCB max extent",
    )
    args = parser.parse_args()
    out = args.out or args.run / "offline_improvements" / "APPEARANCE_PROPOSALS.md"

    if not args.checkpoint.is_file():
        raise SystemExit(f"image checkpoint not found: {args.checkpoint}")
    if not args.proposal_model.is_file():
        raise SystemExit(f"proposal model not found: {args.proposal_model}")
    records = load_records(args.run)
    if not records:
        raise SystemExit(f"no dynamic records under {args.run}")

    from ultralytics import FastSAM

    proposal_model = FastSAM(str(args.proposal_model))
    encoder = ImageEncoder(args.checkpoint, args.device)
    ref_cache: Dict[Tuple[str, str, str], Tuple[np.ndarray, List[Dict[str, Any]]]] = {}
    for record in records.values():
        block = record["authored_layout"]["dualmap"]
        key = (str(record["scene"]), str(block["query"]), str(record["target"]))
        if key in ref_cache:
            continue
        refs, metadata = reference_crops(
            args.maps, key[0], key[1], key[2], args.reference_radius_m, args.reference_limit
        )
        ref_cache[key] = (encoder.encode(refs), metadata)

    rows: List[Dict[str, Any]] = []
    controls: List[Dict[str, Any]] = []
    for trial_id, record in sorted(records.items()):
        block = record["authored_layout"]["dualmap"]
        scene, query, label = str(record["scene"]), str(block["query"]), str(record["target"])
        ref_features, ref_metadata = ref_cache[(scene, query, label)]
        if not len(ref_features):
            continue
        opportunity = (
            not bool(block.get("success"))
            and int(record.get("gt_kf_in_view") or 0) > 0
            and int(record.get("gt_kf_detected") or 0) == 0
        )
        for frame in dump_frames(args.run / "gt_dump", trial_id, args.frames_per_episode):
            image = Image.open(frame["path"]).convert("RGB")
            crops, masks, boxes = proposal_crops(
                proposal_model, image,
                imgsz=args.imgsz,
                confidence=args.proposal_confidence,
                iou=args.proposal_iou,
                device=args.device,
                min_area_px=args.min_area_px,
                max_area_fraction=args.max_area_fraction,
                padding_fraction=args.padding_fraction,
            )
            if crops:
                features = encoder.encode(crops)
                proposal_scores = np.max(features @ ref_features.T, axis=1).astype(float)
            else:
                features = np.empty((0, 512), dtype=np.float32)
                proposal_scores = np.empty((0,), dtype=float)
            raw_target_indices = [
                i for i, (mask, box) in enumerate(zip(masks, boxes))
                if is_target_proposal(mask, box, frame["u"], frame["v"])
            ]
            width = image.size[0]
            focal = width / 2.0  # the frozen run used a 90-degree horizontal FOV
            projected_side = (
                focal * YCB_EXTENT_M[query] / max(float(frame["range_m"]), 0.1)
            )
            largest_target_area = args.target_max_area_ratio * projected_side ** 2
            target_indices = [
                i for i in raw_target_indices
                if (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                <= largest_target_area
            ]
            distractor_indices = [i for i in range(len(crops)) if i not in set(target_indices)]
            best_target = (
                float(max(proposal_scores[i] for i in target_indices)) if target_indices else None
            )
            rank = (
                1 + sum(float(proposal_scores[i]) >= best_target for i in distractor_indices)
                if best_target is not None else None
            )
            row = {
                "trial_id": trial_id,
                "condition": block["condition"],
                "query": query,
                "scene": scene,
                "seen_never_named": opportunity,
                "frame": frame["path"].name,
                "proposal_count": len(crops),
                "raw_center_cover_count": len(raw_target_indices),
                "raw_center_cover": bool(raw_target_indices),
                "target_proposal_count": len(target_indices),
                "proposal_recall": bool(target_indices),
                "best_target_score": best_target,
                "distractor_scores": [float(proposal_scores[i]) for i in distractor_indices],
                "rank": rank,
                "projected_extent_side_px": projected_side,
                "largest_plausible_target_area_px": largest_target_area,
                "target_proposal_boxes": [boxes[i] for i in target_indices],
                "reference_tracks": ref_metadata,
            }
            rows.append(row)

            for wrong_key, (wrong_features, _) in ref_cache.items():
                if (
                    wrong_key[0] != scene or wrong_key[1] == query
                    or not len(wrong_features) or not len(features)
                ):
                    continue
                scores = np.max(features @ wrong_features.T, axis=1).astype(float)
                positive = float(max(scores[i] for i in target_indices)) if target_indices else None
                negative = [float(scores[i]) for i in distractor_indices]
                controls.append({
                    **row,
                    "reference_query": wrong_key[1],
                    "best_target_score": positive,
                    "distractor_scores": negative,
                    "rank": (
                        1 + sum(score >= positive for score in negative)
                        if positive is not None else None
                    ),
                })

    overall = metric_rows(rows)
    seen_not_named = metric_rows([r for r in rows if r["seen_never_named"]])
    control = metric_rows(controls)
    result = {
        "input_run": str(args.run),
        "encoder": f"OpenAI CLIP ViT-B/32 ({args.checkpoint})",
        "proposal_model": str(args.proposal_model),
        "ground_truth_usage": "evaluation only: whether a generated proposal mask covers projected target centre",
        "parameters": {
            "frames_per_episode": args.frames_per_episode,
            "imgsz": args.imgsz,
            "proposal_confidence": args.proposal_confidence,
            "proposal_iou": args.proposal_iou,
            "min_area_px": args.min_area_px,
            "max_area_fraction": args.max_area_fraction,
            "padding_fraction": args.padding_fraction,
            "target_max_area_ratio": args.target_max_area_ratio,
        },
        "metrics": {
            "overall": overall,
            "by_condition": grouped(rows, "condition"),
            "by_target": grouped(rows, "query"),
            "seen_never_named": seen_not_named,
            "wrong_reference_control": control,
        },
        "frames": rows,
    }

    lines = [
        "# Offline class-agnostic appearance proposal replay",
        "",
        f"Frozen input: `{args.run}`. FastSAM-s generates regions from RGB alone; CLIP "
        "ranks them against the stale static-map crop. Ground truth is used only after "
        "proposal generation to measure whether a mask covers the projected target centre "
        "and whether its box is plausible for the known YCB extent and recorded range. "
        "This is **not an SR result**.",
        "",
    ]
    add_table(lines, {
        "overall": overall,
        **result["metrics"]["by_condition"],
        "seen-never-named": seen_not_named,
    })
    lines += [
        "",
        "## Identity control",
        "",
        "| Reference | Plausible proposal recall | Conditional Top-1 | Conditional Top-5 | End-to-end Top-1 | End-to-end Top-5 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| correct target | {pct(overall.get('proposal_recall'))} | {pct(overall.get('top1'))} | "
        f"{pct(overall.get('top5'))} | {pct(overall.get('end_to_end_top1'))} | "
        f"{pct(overall.get('end_to_end_top5'))} |",
        f"| wrong target, same scene | {pct(control.get('proposal_recall'))} | "
        f"{pct(control.get('top1'))} | {pct(control.get('top5'))} | "
        f"{pct(control.get('end_to_end_top1'))} | {pct(control.get('end_to_end_top5'))} |",
        "",
        "## By target",
        "",
    ]
    add_table(lines, result["metrics"]["by_target"])
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Proposal recall is only a centre-pixel and size proxy because the dump does not "
        "contain the true target mask. The report shows raw centre coverage separately, "
        "then rejects boxes over 16 times the square of the projected maximum YCB extent "
        "to reduce supporting-surface leakage. A retained region may still be a surface. "
        "An online pilot must therefore require "
        "temporal/geometric confirmation and inspect saved top-ranked crops before enabling "
        "automatic candidate admission.",
        "",
        f"Machine-readable details: [{out.with_suffix('.json').name}]({out.with_suffix('.json').name}).",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
