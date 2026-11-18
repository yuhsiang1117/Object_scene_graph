#!/usr/bin/env python3
"""Offline appearance-retrieval gate over a frozen DualMap-protocol run.

This deliberately uses runner-only projected target centres to construct an
oracle positive crop.  It tests whether a static-map appearance exemplar can
recognise the same YCB object in a later RGB frame; it does not test a deployable
proposal generator and must not be reported as SR.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from osg.eval import dualmap_release as release


UV_RE = re.compile(r"_u(?P<u>\d+)_v(?P<v>\d+)\.png$")

# Maximum YCB mesh extent, measured from the staged render meshes.  It is used
# only to choose an oracle diagnostic crop size, never by the agent.
YCB_EXTENT_M = {
    "cracker box": 0.213421,
    "soup can": 0.101895,
    "banana": 0.178395,
    "pitcher": 0.242288,
    "bowl": 0.161225,
    "mug": 0.116941,
    "plate": 0.260670,
    "scissors": 0.201556,
}


def load_records(root: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.glob("*/episodes.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            block = (record.get("authored_layout") or {}).get("dualmap") or {}
            if block.get("condition") in {"in_anchor", "cross_anchor"}:
                records[str(block["trial_id"])] = record
    return records


def evenly_sample(items: Sequence[Any], limit: int) -> List[Any]:
    if len(items) <= limit:
        return list(items)
    indices = np.linspace(0, len(items) - 1, limit, dtype=int)
    return [items[int(i)] for i in indices]


def dump_frames(dump: Path, trial_id: str, limit: int) -> List[Dict[str, Any]]:
    matches = [p for p in dump.glob("*/index.tsv") if p.parent.name.endswith("_ep" + trial_id)]
    if len(matches) != 1:
        return []
    index = matches[0]
    rows = list(csv.DictReader(index.open(encoding="utf-8"), delimiter="\t"))
    selected = evenly_sample(rows, limit)
    out: List[Dict[str, Any]] = []
    for row in selected:
        kf = int(row["kf"])
        images = list((index.parent / "raw").glob(f"kf{kf:04d}_*.png"))
        if len(images) != 1:
            continue
        found = UV_RE.search(images[0].name)
        if found is None:
            continue
        out.append({
            **row,
            "path": images[0],
            "u": int(found.group("u")),
            "v": int(found.group("v")),
        })
    return out


def reference_crops(
    maps: Path,
    scene: str,
    query: str,
    agent_label: str,
    radius_m: float,
    limit: int,
) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
    path = maps / scene / f"{scene}.json"
    if not path.is_file():
        return [], []
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    targets = release.static_target_positions(scene, query)
    candidates = []
    for track in snapshot.get("tracks") or []:
        if str(track.get("label", "")).strip().lower() != agent_label.strip().lower():
            continue
        encoded = track.get("best_crop_png")
        center = track.get("center")
        if not encoded or not center or len(center) < 3:
            continue
        error = release.horizontal_distance_xz((center[0], center[2]), targets)
        if error <= radius_m:
            candidates.append((error, -float(track.get("best_score") or 0.0), track, encoded))
    candidates.sort(key=lambda x: (x[0], x[1]))
    images: List[Image.Image] = []
    metadata: List[Dict[str, Any]] = []
    for error, _, track, encoded in candidates[:limit]:
        try:
            image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        except Exception:
            continue
        images.append(image)
        metadata.append({
            "track_id": int(track["id"]),
            "map_error_m": float(error),
            "best_score": float(track.get("best_score") or 0.0),
            "crop_size": list(image.size),
        })
    return images, metadata


def square_crop(image: Image.Image, cx: float, cy: float, side: float) -> Image.Image:
    side_i = max(8, int(round(side)))
    half = side_i // 2
    left = int(round(cx)) - half
    top = int(round(cy)) - half
    # PIL pads out-of-frame portions with black.  Keeping the requested square
    # avoids silently changing scale at image boundaries.
    return image.crop((left, top, left + side_i, top + side_i)).convert("RGB")


def region_crops(
    image: Image.Image,
    u: int,
    v: int,
    range_m: float,
    query: str,
    hfov_deg: float,
    crop_padding: float,
) -> Tuple[List[Image.Image], List[Image.Image]]:
    width, height = image.size
    focal = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    extent = YCB_EXTENT_M[query]
    base = float(np.clip(crop_padding * focal * extent / max(range_m, 0.1), 48.0, 384.0))
    scales = (0.72, 1.0, 1.38)
    positives = [square_crop(image, u, v, base * scale) for scale in scales]

    negatives: List[Image.Image] = []
    # Three deterministic grids approximate multi-scale region proposals.  A
    # region that could contain the oracle centre is excluded from negatives.
    for scale, cols, rows in ((0.72, 5, 4), (1.0, 5, 4), (1.38, 4, 3)):
        side = base * scale
        for gy in range(rows):
            cy = (gy + 0.5) * height / rows
            for gx in range(cols):
                cx = (gx + 0.5) * width / cols
                if abs(cx - u) <= 0.75 * side and abs(cy - v) <= 0.75 * side:
                    continue
                negatives.append(square_crop(image, cx, cy, side))
    return positives, negatives


class ImageEncoder:
    def __init__(self, checkpoint: Path, device: str) -> None:
        import open_clip

        selected = device
        if selected == "auto":
            selected = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(selected)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=str(checkpoint)
        )
        self.model = self.model.to(self.device).eval()

    def encode(self, images: Sequence[Image.Image], batch_size: int = 128) -> np.ndarray:
        chunks = []
        for start in range(0, len(images), batch_size):
            batch = torch.stack([self.preprocess(im) for im in images[start:start + batch_size]])
            batch = batch.to(self.device)
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        features = self.model.encode_image(batch)
                else:
                    features = self.model.encode_image(batch)
                features = torch.nn.functional.normalize(features.float(), dim=-1)
            chunks.append(features.cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 512), dtype=np.float32)


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"frames": 0}
    positives = np.asarray([r["positive_score"] for r in rows], dtype=float)
    negatives = np.asarray([s for r in rows for s in r["negative_scores"]], dtype=float)
    wins = [
        float(np.mean(float(r["positive_score"]) > np.asarray(r["negative_scores"], dtype=float)))
        for r in rows if r["negative_scores"]
    ]
    return {
        "frames": len(rows),
        "episodes": len({r["trial_id"] for r in rows}),
        "positive_mean": float(np.mean(positives)),
        "positive_median": float(np.median(positives)),
        "negative_mean": float(np.mean(negatives)) if len(negatives) else None,
        "negative_median": float(np.median(negatives)) if len(negatives) else None,
        "pairwise_auc": float(np.mean(wins)) if wins else None,
        "top1": float(np.mean([r["rank"] <= 1 for r in rows])),
        "top5": float(np.mean([r["rank"] <= 5 for r in rows])),
        "median_rank": float(np.median([r["rank"] for r in rows])),
        "mean_candidates": float(np.mean([1 + len(r["negative_scores"]) for r in rows])),
        "chance_top1": float(np.mean([1.0 / (1 + len(r["negative_scores"])) for r in rows])),
    }


def group_summaries(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {name: summarize(items) for name, items in sorted(grouped.items())}


def pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def markdown(result: Dict[str, Any], out: Path) -> str:
    overall = result["metrics"]["overall"]
    control = result["metrics"]["wrong_reference_control"]
    lines = [
        "# Offline appearance-retrieval gate",
        "",
        f"Frozen input: `{result['input_run']}`. Encoder: `{result['encoder']}`. "
        f"This is an oracle-region diagnostic, **not an SR result**.",
        "",
        f"Static appearance coverage was {result['coverage']['covered_trials']}/"
        f"{result['coverage']['dynamic_trials']} dynamic trials. The evaluation used "
        f"{overall.get('frames', 0)} frames from {overall.get('episodes', 0)} episodes.",
        "",
        "The dump contains only the ground-truth projected centre, not a target mask or "
        "box. Positive crops use that centre and a size estimated from known YCB mesh "
        "extent and range; distractors are deterministic multi-scale grid crops.",
        "",
        "| Slice | Frames | Episodes | Pairwise AUC | Top-1 | Top-5 | Median rank | Chance Top-1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    slices = {"overall": overall, **result["metrics"]["by_condition"]}
    slices["seen-never-named"] = result["metrics"]["seen_never_named"]
    for name, values in slices.items():
        lines.append(
            f"| {name} | {values.get('frames', 0)} | {values.get('episodes', 0)} | "
            f"{pct(values.get('pairwise_auc'))} | {pct(values.get('top1'))} | "
            f"{pct(values.get('top5'))} | {values.get('median_rank', 'n/a')} | "
            f"{pct(values.get('chance_top1'))} |"
        )
    lines += [
        "",
        "## Wrong-reference control",
        "",
        "The same oracle target crops are scored against other YCB objects' static "
        "references from the same scene. If target-centred crops ranked well merely "
        "because they are salient or share room background, this control would approach "
        "the matched-reference result.",
        "",
        "| Reference | Frames | Pairwise AUC | Top-1 | Top-5 | Median rank |",
        "|---|---:|---:|---:|---:|---:|",
        f"| correct target | {overall.get('frames', 0)} | {pct(overall.get('pairwise_auc'))} | "
        f"{pct(overall.get('top1'))} | {pct(overall.get('top5'))} | "
        f"{overall.get('median_rank', 'n/a')} |",
        f"| wrong target, same scene | {control.get('frames', 0)} | "
        f"{pct(control.get('pairwise_auc'))} | {pct(control.get('top1'))} | "
        f"{pct(control.get('top5'))} | {control.get('median_rank', 'n/a')} |",
        "",
        "## By target",
        "",
        "| Target | Frames | Episodes | Pairwise AUC | Top-1 | Top-5 | Median rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in result["metrics"]["by_target"].items():
        lines.append(
            f"| {name} | {values.get('frames', 0)} | {values.get('episodes', 0)} | "
            f"{pct(values.get('pairwise_auc'))} | {pct(values.get('top1'))} | "
            f"{pct(values.get('top5'))} | {values.get('median_rank', 'n/a')} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "A positive result establishes only that stored and live appearance are separable "
        "when the correct live region is supplied. Before runtime integration, the same "
        "test must be repeated with proposals generated without ground truth. Missing "
        "static references (notably classes the mapping detector never captured) remain "
        "unrecoverable by this mechanism alone.",
        "",
        f"Machine-readable details: [{out.with_suffix('.json').name}]({out.with_suffix('.json').name}).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/osg_dualmap_tightring"))
    parser.add_argument("--maps", type=Path, default=Path("outputs/maps_v5"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/clip/ViT-B-32.pt"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frames-per-episode", type=int, default=8)
    parser.add_argument("--reference-radius-m", type=float, default=0.75)
    parser.add_argument("--reference-limit", type=int, default=3)
    parser.add_argument("--hfov-deg", type=float, default=90.0)
    parser.add_argument("--crop-padding", type=float, default=1.8)
    args = parser.parse_args()

    out = args.out or args.run / "offline_improvements" / "APPEARANCE_RETRIEVAL.md"
    if not args.checkpoint.is_file():
        raise SystemExit(f"image checkpoint not found: {args.checkpoint}")
    records = load_records(args.run)
    if not records:
        raise SystemExit(f"no dynamic records under {args.run}")

    encoder = ImageEncoder(args.checkpoint, args.device)
    ref_cache: Dict[Tuple[str, str, str], Tuple[np.ndarray, List[Dict[str, Any]]]] = {}
    rows: List[Dict[str, Any]] = []
    control_rows: List[Dict[str, Any]] = []
    covered_trials = 0
    covered_pairs = set()
    missing_pairs = set()

    # Load each unique static reference first.  The second pass can then score
    # the exact same regions against every available wrong-object reference in
    # the same scene without another image-encoder call.
    for record in records.values():
        block = record["authored_layout"]["dualmap"]
        scene, query, label = str(record["scene"]), str(block["query"]), str(record["target"])
        key = (scene, query, label)
        if key not in ref_cache:
            refs, metadata = reference_crops(
                args.maps, scene, query, label, args.reference_radius_m, args.reference_limit
            )
            ref_cache[key] = (encoder.encode(refs), metadata)

    for trial_id, record in sorted(records.items()):
        block = record["authored_layout"]["dualmap"]
        scene, query, label = str(record["scene"]), str(block["query"]), str(record["target"])
        key = (scene, query, label)
        ref_features, ref_metadata = ref_cache[key]
        if not len(ref_features):
            missing_pairs.add((scene, query))
            continue
        covered_trials += 1
        covered_pairs.add((scene, query))
        opportunity = (
            not bool(block.get("success"))
            and int(record.get("gt_kf_in_view") or 0) > 0
            and int(record.get("gt_kf_detected") or 0) == 0
        )
        for frame in dump_frames(args.run / "gt_dump", trial_id, args.frames_per_episode):
            image = Image.open(frame["path"]).convert("RGB")
            positives, negatives = region_crops(
                image, frame["u"], frame["v"], float(frame["range_m"]), query,
                args.hfov_deg, args.crop_padding,
            )
            features = encoder.encode(positives + negatives)
            scores = features @ ref_features.T
            positive_score = float(np.max(scores[:len(positives)]))
            negative_scores = np.max(scores[len(positives):], axis=1).astype(float).tolist()
            rank = 1 + sum(score >= positive_score for score in negative_scores)
            rows.append({
                "trial_id": trial_id,
                "condition": block["condition"],
                "query": query,
                "scene": scene,
                "success": bool(block.get("success")),
                "seen_never_named": opportunity,
                "frame": frame["path"].name,
                "range_m": float(frame["range_m"]),
                "named": frame.get("named") == "1",
                "positive_score": positive_score,
                "negative_scores": negative_scores,
                "rank": rank,
                "reference_tracks": ref_metadata,
            })
            for wrong_key, (wrong_features, _) in ref_cache.items():
                if wrong_key[0] != scene or wrong_key[1] == query or not len(wrong_features):
                    continue
                wrong_scores = features @ wrong_features.T
                wrong_positive = float(np.max(wrong_scores[:len(positives)]))
                wrong_negatives = np.max(
                    wrong_scores[len(positives):], axis=1
                ).astype(float).tolist()
                control_rows.append({
                    "trial_id": trial_id,
                    "condition": block["condition"],
                    "query": query,
                    "reference_query": wrong_key[1],
                    "positive_score": wrong_positive,
                    "negative_scores": wrong_negatives,
                    "rank": 1 + sum(score >= wrong_positive for score in wrong_negatives),
                })

    result = {
        "input_run": str(args.run),
        "maps": str(args.maps),
        "encoder": f"OpenAI CLIP ViT-B/32 ({args.checkpoint})",
        "oracle_inputs": ["projected target centre", "recorded range", "known YCB mesh extent"],
        "parameters": {
            "frames_per_episode": args.frames_per_episode,
            "reference_radius_m": args.reference_radius_m,
            "reference_limit": args.reference_limit,
            "hfov_deg": args.hfov_deg,
            "crop_padding": args.crop_padding,
        },
        "coverage": {
            "dynamic_trials": len(records),
            "covered_trials": covered_trials,
            "covered_scene_targets": [list(x) for x in sorted(covered_pairs)],
            "missing_scene_targets": [list(x) for x in sorted(missing_pairs)],
        },
        "metrics": {
            "overall": summarize(rows),
            "by_condition": group_summaries(rows, "condition"),
            "by_target": group_summaries(rows, "query"),
            "seen_never_named": summarize([r for r in rows if r["seen_never_named"]]),
            "wrong_reference_control": summarize(control_rows),
        },
        "frames": rows,
        "wrong_reference_control_frames": control_rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    out.write_text(markdown(result, out), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
