#!/usr/bin/env python3
"""Would matching by appearance find what matching by label cannot?

Two questions, answered offline against frozen data, before any agent code.

**Part A -- the prior map.** Every track in a prior map carries a crop
(`best_crop_png`, 343/343 in 00848). Embed them all, score each against the
query text, and report where the true object ranks -- as a track, and as the
CONTAINER holding it, which is what the search prior would actually cut on. The
scissors is the case: the mapping pass saw it and called it `bleach bottle`,
0.09-0.10 m from the true position in all three scenes, so a label match scores
zero and a feature match should not. A text-only control (query text against
LABEL text) says whether a synonym table would have done the same job cheaper.

**Part B -- the live frames.** Over the keyframes where the object was in view
and the detector never named it, embed every region the live detector proposes
(any label) and every FastSAM region, and report where the true one ranks. Part
A decides whether the agent would GO to the right surface; part B decides
whether it would recognise the object once there.

Nothing here is an SR result, and ground truth is used only to score: the
projected target pixel in each dump filename says which region was the object,
after the regions were generated.

  HF_HUB_OFFLINE=1 python scripts/legacy/probe_feature_memory.py --part both --backbone both
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from osg.eval import dualmap_release as release  # noqa: E402
from osg.objects.feature_memory import container_feature_scores, feature_term  # noqa: E402

UV_RE = re.compile(r"_u(?P<u>\d+)_v(?P<v>\d+)\.png$")
# Longest mesh extent per query, for the evaluation-only size sanity check on
# class-agnostic proposals (scripts/legacy/offline_appearance_retrieval.py:36).
YCB_EXTENT_M = {
    "cracker box": 0.213421, "soup can": 0.101895, "banana": 0.178395,
    "pitcher": 0.242288, "bowl": 0.161225, "mug": 0.116941,
    "plate": 0.260670, "scissors": 0.201556,
}
RANGE_BANDS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, float("inf")))


def band_of(range_m: float) -> str:
    for lo, hi in RANGE_BANDS:
        if lo <= range_m < hi:
            return f"{lo:.0f}-{hi:.0f} m" if hi != float("inf") else f">={lo:.0f} m"
    return "?"


# --------------------------------------------------------------------- part A

def load_tracks(map_path: Path) -> List[Dict[str, Any]]:
    """Every track in a snapshot, with its decoded crop."""
    from osg.graph.map_store import _decode_crop

    blob = json.loads(map_path.read_text(encoding="utf-8"))
    out = []
    for rec in blob.get("tracks") or []:
        centre = rec.get("center")
        if not centre or len(centre) < 3:
            continue
        out.append({
            "id": int(rec["id"]),
            "label": str(rec.get("label", "")),
            "xz": (float(centre[0]), float(centre[2])),
            "best_score": float(rec.get("best_score") or 0.0),
            "crop": _decode_crop(rec.get("best_crop_png")),
        })
    return out


def truth_targets(scene: str, query: str):
    """The release's own (centre, half-extent) boxes for a static query, or None
    when the scene has no instance of it (00829 and 00880 have no mug)."""
    try:
        return release.static_target_positions(scene, query)
    except Exception:
        return None


def nearest_track(tracks: Sequence[Dict[str, Any]], targets):
    """The mapped track closest to any true instance, and how far off it is."""
    best, best_d = None, float("inf")
    for track in tracks:
        d = release.horizontal_distance_xz(track["xz"], targets)
        if d < best_d:
            best, best_d = track, d
    return best, best_d


def part_a(
    encoder, backbone: str, args, prompts: Sequence[str], queries: Sequence[str]
) -> List[Dict[str, Any]]:
    """Track rank and container rank of the true object, per scene and query."""
    from osg.core.config import SceneGraphConfig
    from osg.exploration.search_belief import InspectionLog, build_container_candidates
    from osg.sim.dualmap_env import agent_query
    from rank_search_surfaces import _footprint_distance, load_scene_graph

    rows: List[Dict[str, Any]] = []
    for scene in release.SCENES:
        map_path = args.maps / scene / f"{scene}.json"
        if not map_path.is_file():
            print(f"  ! no map for {scene}", flush=True)
            continue
        tracks = load_tracks(map_path)
        crops = [t["crop"] for t in tracks]
        have = [i for i, c in enumerate(crops) if c is not None and getattr(c, "size", 0) > 0]
        print(f"  {scene}: {len(tracks)} tracks, {len(have)} with crops", flush=True)
        feats = np.full((len(tracks), encoder.dim), np.nan, dtype=np.float32)
        encoded = encoder.encode_images([crops[i] for i in have])
        for slot, i in enumerate(have):
            feats[i] = encoded[slot]

        graph = load_scene_graph(map_path, SceneGraphConfig())
        layer = _TrackShim(tracks, feats)
        for query in queries:
            targets = truth_targets(scene, query)
            if not targets:
                continue
            points = [(float(c[0]), float(c[2])) for c, _ in targets]
            label = agent_query(query)
            true_track, true_d = nearest_track(tracks, targets)
            for prompt in prompts:
                text = prompt.format(q=query)
                text_ft = encoder.text_feature(text)
                sims = np.where(np.isnan(feats[:, 0]), -1.0, feats @ text_ft)
                order = np.argsort(-sims)
                rank = int(np.where(order == tracks.index(true_track))[0][0]) + 1 if true_track else None
                top = [{
                    "track_id": tracks[i]["id"], "label": tracks[i]["label"],
                    "sim": round(float(sims[i]), 4),
                    "dist_to_truth_m": round(float(
                        release.horizontal_distance_xz(tracks[i]["xz"], targets)), 3),
                } for i in order[:5]]
                best_wrong = next((t for t in top if t["track_id"] != (true_track or {}).get("id")), None)
                sim_true = round(float(sims[tracks.index(true_track)]), 4) if true_track else None

                # Text-only control: what a synonym table would have given.
                label_texts = sorted({t["label"] for t in tracks})
                label_ft = encoder.encode_text([prompt.format(q=l) for l in label_texts])
                lab_sims = label_ft @ text_ft
                lab_order = np.argsort(-lab_sims)
                true_label = true_track["label"] if true_track else None
                text_rank = (int(np.where(np.asarray(label_texts)[lab_order] == true_label)[0][0]) + 1
                             if true_label in label_texts else None)

                # Containers, matched by distance to the surface, never by id.
                # Two child rules: the support relation alone (DualMap parity)
                # and the support relation widened by proximity, because the
                # support test binds only 3-5% of mapped objects in these maps.
                true_cids = {int(cid) for cid, node in graph.containers.items()
                             if min(_footprint_distance(node, p) for p in points) <= args.container_near_m}
                base = build_container_candidates(graph, label, InspectionLog())
                base_ranked = sorted(base, key=lambda c: -(c.prior * c.detect_prob))
                base_rank = next((i + 1 for i, c in enumerate(base_ranked)
                                  if int(c.ref_id) in true_cids), None)

                variants: Dict[str, Any] = {}
                for name, radius in (("bound", 0.0), ("radius", args.child_radius_m)):
                    cscores = container_feature_scores(
                        graph, layer, text_ft, skip_refuted=False, radius_m=radius
                    )
                    ranked_c = sorted(cscores.items(), key=lambda kv: -kv[1])
                    c_rank = next((i + 1 for i, (cid, _) in enumerate(ranked_c)
                                   if cid in true_cids), None)
                    gap = (ranked_c[0][1] - ranked_c[1][1]) if len(ranked_c) > 1 else None
                    sweep = []
                    unit = math.log(2) / max(gap if gap else 0.02, 0.01)
                    for beta in (unit, 2 * unit, 4 * unit):
                        for floor in (0.1, 0.25, 0.5):
                            fused = build_container_candidates(
                                graph, label, InspectionLog(), feature_scores=cscores,
                                feature_beta=beta, feature_floor=floor,
                            )
                            fr = sorted(fused, key=lambda c: -(c.prior * c.detect_prob))
                            sweep.append({
                                "beta": round(beta, 2), "floor": floor,
                                "fused_rank": next((i + 1 for i, c in enumerate(fr)
                                                    if int(c.ref_id) in true_cids), None),
                            })
                    variants[name] = {
                        "n_scored": len(cscores),
                        "container_rank": c_rank,
                        "container_gap": round(float(gap), 4) if gap is not None else None,
                        "fused": sweep,
                    }
                c_rank = variants["radius"]["container_rank"]
                rows.append({
                    "backbone": backbone, "scene": scene, "query": query,
                    "agent_label": label, "prompt": prompt,
                    "n_tracks": len(tracks), "n_containers": len(graph.containers),
                    "true_track_id": true_track["id"] if true_track else None,
                    "true_track_label": true_label,
                    "true_track_dist_m": round(float(true_d), 3),
                    "within_near_m": bool(true_d <= args.near_m),
                    "track_rank": rank, "sim_true": sim_true,
                    "sim_best_wrong": best_wrong["sim"] if best_wrong else None,
                    "best_wrong_label": best_wrong["label"] if best_wrong else None,
                    "gap": (round(sim_true - best_wrong["sim"], 4)
                            if sim_true is not None and best_wrong else None),
                    "top5": top,
                    "text_only_rank": text_rank, "n_labels": len(label_texts),
                    "n_true_containers": len(true_cids),
                    "container_rank": c_rank,
                    "container_rank_bound": variants["bound"]["container_rank"],
                    "container_gap": variants["radius"]["container_gap"],
                    "baseline_container_rank": base_rank,
                    "variants": variants,
                    "fused": variants["radius"]["fused"],
                })
                if args.save_crops and top:
                    save_top_crops(args.out / "crops" / f"{scene}_{query}_{backbone}",
                                   [tracks[i] for i in order[:5]])
    return rows


class _TrackShim:
    """`object_layer.get(track_id)` over the probe's plain dicts."""

    def __init__(self, tracks: Sequence[Dict[str, Any]], feats: np.ndarray) -> None:
        self._by_id = {}
        for i, t in enumerate(tracks):
            ft = None if np.isnan(feats[i, 0]) else feats[i]
            self._by_id[int(t["id"])] = type("T", (), {
                "id": int(t["id"]), "label": t["label"], "clip_ft": ft,
                "absence_arrivals": 0, "failed_attempts": 0,
                "blacklisted": False, "disabled": False,
            })()

    def get(self, track_id: int):
        return self._by_id.get(int(track_id))


def save_top_crops(out_dir: Path, tracks: Sequence[Dict[str, Any]]) -> None:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, track in enumerate(tracks, start=1):
        crop = track.get("crop")
        if crop is None or getattr(crop, "size", 0) == 0:
            continue
        name = re.sub(r"[^a-z0-9]+", "_", str(track["label"]).lower())
        cv2.imwrite(str(out_dir / f"{rank}_{track['id']}_{name}.png"), crop[..., ::-1])


# --------------------------------------------------------------------- part B

def dump_episodes(dump: Path, query: str) -> List[Path]:
    return sorted(d for d in dump.glob("*") if d.is_dir() and d.name.endswith("__" + query))


def dump_rows(episode: Path) -> List[Dict[str, Any]]:
    index = episode / "index.tsv"
    if not index.is_file():
        return []
    by_kf = {int(r["kf"]): r for r in csv.DictReader(index.open(encoding="utf-8"), delimiter="\t")}
    out = []
    for raw in sorted((episode / "raw").glob("*.png")):
        found = UV_RE.search(raw.name)
        kf = int(raw.name[2:6]) if raw.name.startswith("kf") else None
        if found is None or kf is None:
            continue
        row = by_kf.get(kf, {})
        out.append({
            "path": raw, "u": int(found.group("u")), "v": int(found.group("v")),
            "range_m": float(row.get("range_m") or 0.0),
            "named": str(row.get("named") or "0") == "1",
            "episode": episode.name,
        })
    return out


def covers(box: Sequence[float], u: float, v: float, pad: float = 8.0) -> bool:
    return box[0] - pad <= u <= box[2] + pad and box[1] - pad <= v <= box[3] + pad


def yoloe_regions(detector, rgb: np.ndarray):
    dets = detector.detect(rgb)
    crops, boxes, labels, scores = [], [], [], []
    for det in dets:
        crop = det.crop if det.crop is not None else det.crop_from(rgb)
        if crop is None or getattr(crop, "size", 0) == 0:
            continue
        crops.append(crop)
        boxes.append([float(x) for x in det.bbox_xyxy])
        labels.append(str(det.label))
        scores.append(float(det.score))
    return crops, boxes, labels, scores


def fastsam_regions(model, image, args, device: str):
    from offline_appearance_proposals import proposal_crops

    crops, masks, boxes = proposal_crops(
        model, image, imgsz=args.proposal_imgsz, confidence=args.proposal_confidence,
        iou=args.proposal_iou, device=device, min_area_px=args.min_area_px,
        max_area_fraction=args.max_area_fraction, padding_fraction=args.padding_fraction,
    )
    return crops, masks, boxes


def part_b(encoder, backbone: str, args, prompts: Sequence[str], queries: Sequence[str]):
    """Rank of the true region among everything a live frame proposes."""
    from PIL import Image

    from offline_appearance_proposals import is_target_proposal
    from osg.sim.dualmap_env import agent_query

    dump = args.run / "gt_dump"
    sources = [s.strip() for s in str(args.regions).split(",") if s.strip()]
    detector = fastsam = None
    rows: List[Dict[str, Any]] = []

    for query in queries:
        episodes = dump_episodes(dump, query)
        frames = [f for ep in episodes for f in dump_rows(ep)]
        if not frames:
            print(f"  ! no dumped frames for {query}", flush=True)
            continue
        label = agent_query(query)
        print(f"  {query}: {len(frames)} frames in {len(episodes)} episodes "
              f"(detector name {label!r})", flush=True)

        if "yoloe" in sources and detector is None:
            detector = build_probe_detector(args)
        if "yoloe" in sources:
            from osg.perception.vocabulary import target_vocabulary
            detector.set_vocabulary(target_vocabulary(label, args._vocabulary))
        if "fastsam" in sources and fastsam is None:
            from ultralytics import FastSAM
            fastsam = FastSAM(str(args.proposal_model))

        text_fts = {p: encoder.text_feature(p.format(q=query)) for p in prompts}
        for n, frame in enumerate(frames):
            if args.max_frames and n >= args.max_frames:
                break
            image = Image.open(frame["path"]).convert("RGB")
            rgb = np.asarray(image)
            for source in sources:
                if source == "yoloe":
                    crops, boxes, labels, scores = yoloe_regions(detector, rgb)
                    true_idx = [i for i, b in enumerate(boxes) if covers(b, frame["u"], frame["v"])]
                else:
                    crops, masks, boxes = fastsam_regions(fastsam, image, args, args.device)
                    labels = [""] * len(crops)
                    scores = [0.0] * len(crops)
                    true_idx = [i for i, (m, b) in enumerate(zip(masks, boxes))
                                if is_target_proposal(m, b, frame["u"], frame["v"])]
                    # Evaluation-only: a mask covering the centre can still be the
                    # whole desk, so reject boxes far larger than the object can be.
                    focal = rgb.shape[1] / 2.0
                    side = focal * YCB_EXTENT_M[query] / max(frame["range_m"], 0.1)
                    cap = 16.0 * side * side
                    true_idx = [i for i in true_idx
                                if (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) <= cap]
                feats = encoder.encode_images(crops) if crops else np.empty((0, encoder.dim), np.float32)
                for prompt, text_ft in text_fts.items():
                    sims = (feats @ text_ft) if len(feats) else np.empty((0,), np.float32)
                    wrong = [float(sims[i]) for i in range(len(sims)) if i not in set(true_idx)]
                    best_true = max((float(sims[i]) for i in true_idx), default=None)
                    rank = (1 + sum(s >= best_true for s in wrong)) if best_true is not None else None
                    rows.append({
                        "backbone": backbone, "query": query, "prompt": prompt,
                        "source": source, "episode": frame["episode"],
                        "frame": frame["path"].name, "range_m": frame["range_m"],
                        "band": band_of(frame["range_m"]), "named": frame["named"],
                        "n_regions": len(crops), "covered": bool(true_idx),
                        "sim_true": None if best_true is None else round(best_true, 4),
                        "sim_best_wrong": round(max(wrong), 4) if wrong else None,
                        "margin": (round(best_true - max(wrong), 4)
                                   if best_true is not None and wrong else None),
                        "rank": rank,
                        "wrong_sims": [round(s, 4) for s in wrong],
                        "true_labels": [labels[i] for i in true_idx],
                    })
    return rows


def build_probe_detector(args):
    """The live detector, at the exact settings of the arm under test."""
    from hydra import compose, initialize_config_dir

    from osg.core.config import register_configs
    from osg.pipeline.components import build_detector

    register_configs()
    root = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(root), version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"+experiment={args.preset}"])
    args._vocabulary = [str(v) for v in cfg.detector.vocabulary]
    print(f"  detector: {cfg.detector.weights} imgsz={cfg.detector.imgsz} "
          f"conf={cfg.detector.conf}", flush=True)
    return build_detector(cfg)


# --------------------------------------------------------------------- report

def pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100.0 * float(x):.1f}%"


def summarise_b(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"frames": 0}
    covered = [r for r in rows if r["covered"]]
    return {
        "frames": len(rows),
        "covered": len(covered),
        "cover_rate": len(covered) / len(rows),
        "top1": sum(1 for r in covered if r["rank"] == 1) / len(rows),
        "top5": sum(1 for r in covered if r["rank"] and r["rank"] <= 5) / len(rows),
        "named_rate": sum(1 for r in rows if r["named"]) / len(rows),
        "median_regions": float(np.median([r["n_regions"] for r in rows])),
        "median_sim_true": (float(np.median([r["sim_true"] for r in covered]))
                            if covered else None),
    }


def threshold_curve(pos: Sequence[Dict[str, Any]], neg: Sequence[Dict[str, Any]]):
    """Recall of the true region against the rate at which a target-free frame
    would admit something, over a grid of thresholds."""
    out = []
    for tau in np.arange(0.15, 0.36, 0.01):
        hit = [r for r in pos if r["sim_true"] is not None and r["sim_true"] >= tau]
        fp = [r for r in neg if r["sim_best_wrong"] is not None and r["sim_best_wrong"] >= tau]
        out.append({
            "tau": round(float(tau), 3),
            "recall": len(hit) / len(pos) if pos else None,
            "false_admit": len(fp) / len(neg) if neg else None,
        })
    return out


def write_report(result: Dict[str, Any], out: Path) -> None:
    lines = [
        "# Feature memory probe: can appearance find what the label cannot?",
        "",
        "Offline over frozen prior maps and dumped keyframes. Ground truth scores "
        "the answer; it never generates the regions. **Not an SR result.**",
        "",
    ]
    for backbone, rows in sorted(result.get("part_a", {}).items()):
        lines += [f"## A. Prior map ({backbone})", "",
                  "| scene | query | true track | label | dist | track rank | sim | best wrong | gap | text-only rank | container rank (bound) | baseline rank |",
                  "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            if r["prompt"] != result["primary_prompt"]:
                continue
            lines.append(
                f"| {r['scene'][:5]} | {r['query']} | {r['true_track_id']} | {r['true_track_label']} | "
                f"{r['true_track_dist_m']:.2f} m | {r['track_rank']}/{r['n_tracks']} | "
                f"{r['sim_true']} | {r['sim_best_wrong']} ({r['best_wrong_label']}) | {r['gap']} | "
                f"{r['text_only_rank']}/{r['n_labels']} | "
                f"{r['container_rank']}/{r['n_containers']} ({r['container_rank_bound']}) | "
                f"{r['baseline_container_rank']} |")
        lines.append("")
    for backbone, rows in sorted(result.get("part_b", {}).items()):
        lines += [f"## B. Live frames ({backbone})", "",
                  "| query | source | band | frames | covered | top-1 | top-5 | named (label baseline) |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            if r["prompt"] != result["primary_prompt"]:
                continue
            grouped[(r["query"], r["source"], r["band"])].append(r)
            grouped[(r["query"], r["source"], "all")].append(r)
        for (query, source, band), items in sorted(grouped.items()):
            s = summarise_b(items)
            lines.append(
                f"| {query} | {source} | {band} | {s['frames']} | {pct(s['cover_rate'])} | "
                f"{pct(s['top1'])} | {pct(s['top5'])} | {pct(s['named_rate'])} |")
        lines.append("")
    if result.get("thresholds"):
        lines += ["## B3. Admission threshold", "",
                  "| backbone | query | tau | recall (<2 m) | false admit |", "|---|---|---:|---:|---:|"]
        for key, curve in sorted(result["thresholds"].items()):
            backbone, query = key.split("/")
            for point in curve:
                if point["false_admit"] is not None and point["false_admit"] <= 0.10:
                    lines.append(f"| {backbone} | {query} | {point['tau']} | "
                                 f"{pct(point['recall'])} | {pct(point['false_admit'])} |")
                    break
        lines.append("")
    lines += ["## Gates", ""] + [f"- **{k}**: {v}" for k, v in result.get("gates", {}).items()]
    lines += ["", f"Machine-readable: `{out.with_suffix('.json').name}`.", ""]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_gates(result: Dict[str, Any]) -> Dict[str, str]:
    gates: Dict[str, str] = {}
    prompt = result["primary_prompt"]
    for backbone, rows in result.get("part_a", {}).items():
        pairs = [r for r in rows if r["prompt"] == prompt and r["query"] in ("scissors", "mug")]
        counted = [r for r in pairs if r["within_near_m"]]
        top5 = sum(1 for r in counted if r["track_rank"] and r["track_rank"] <= 5)
        gapped = sum(1 for r in counted if (r["gap"] or -1) > 0)
        gates[f"A1 track rank ({backbone})"] = (
            f"{'PASS' if top5 >= 3 and gapped >= 3 else 'FAIL'} -- top-5 on {top5}/{len(counted)} "
            f"scoring pairs, positive gap on {gapped}")
        top3 = sum(1 for r in counted if r["container_rank"] and r["container_rank"] <= 3)
        never_worse = all(
            (min((f["fused_rank"] or 10 ** 6) for f in r["fused"]) <= (r["baseline_container_rank"] or 10 ** 6))
            for r in counted)
        gates[f"A2 container rank ({backbone})"] = (
            f"{'PASS' if top3 >= 3 and never_worse else 'FAIL'} -- true container top-3 on "
            f"{top3}/{len(counted)}, fused never worse than baseline: {never_worse}")
        ctrl = [r for r in rows if r["prompt"] == prompt and r["query"] not in ("scissors", "mug")
                and r["within_near_m"]]
        ctrl_top5 = sum(1 for r in ctrl if r["track_rank"] and r["track_rank"] <= 5)
        beats_text = sum(1 for r in counted
                         if r["track_rank"] and r["text_only_rank"]
                         and r["track_rank"] <= r["text_only_rank"])
        gates[f"A3 controls ({backbone})"] = (
            f"{'PASS' if ctrl and ctrl_top5 / len(ctrl) >= 0.75 else 'FAIL'} -- named targets top-5 on "
            f"{ctrl_top5}/{len(ctrl)}; image beats text-only on {beats_text}/{len(counted)} hard pairs")
    for backbone, rows in result.get("part_b", {}).items():
        for query in sorted({r["query"] for r in rows}):
            near = [r for r in rows if r["prompt"] == prompt and r["query"] == query
                    and r["source"] == "yoloe" and r["range_m"] < 2.0]
            if not near:
                continue
            top1 = sum(1 for r in near if r["covered"] and r["rank"] == 1) / len(near)
            gates[f"B1 live admission {query} ({backbone})"] = (
                f"{'PASS' if top1 >= 0.25 else 'FAIL'} -- top-1 {pct(top1)} on {len(near)} frames "
                f"within 2 m (label baseline {pct(sum(1 for r in near if r['named']) / len(near))})")
    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=("A", "B", "both"), default="both")
    parser.add_argument("--backbone", choices=("vitb32", "mobileclip_s2", "both"), default="both")
    parser.add_argument("--maps", type=Path, default=Path("outputs/maps_v5"))
    parser.add_argument("--run", type=Path, default=Path("outputs/osg_dualmap_tightring"))
    parser.add_argument("--out", type=Path, default=Path("outputs/feature_memory_probe"))
    parser.add_argument("--queries", nargs="+", default=["scissors", "mug"])
    parser.add_argument("--controls", nargs="+",
                        default=["cracker box", "bowl", "banana", "pitcher", "plate", "soup can"])
    parser.add_argument("--prompts", nargs="+", default=["{q}", "a photo of a {q}"])
    parser.add_argument("--regions", default="yoloe,fastsam")
    parser.add_argument("--preset", default="dualmap_protocol_osg_look_flat_anchor_v2_island_close")
    parser.add_argument("--near-m", type=float, default=0.3)
    parser.add_argument("--container-near-m", type=float, default=0.5)
    parser.add_argument("--child-radius-m", type=float, default=1.5,
                        help="widen a surface's child set to mapped objects within "
                             "this distance of its centre (0 = support relation only)")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = every dumped frame")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("data/clip/ViT-B-32.pt"))
    parser.add_argument("--proposal-model", type=Path,
                        default=Path("outputs/dualmap_comparison/vendor/DualMap/model/FastSAM-s.pt"))
    parser.add_argument("--proposal-imgsz", type=int, default=1024)
    parser.add_argument("--proposal-confidence", type=float, default=0.25)
    parser.add_argument("--proposal-iou", type=float, default=0.7)
    parser.add_argument("--min-area-px", type=float, default=144.0)
    parser.add_argument("--max-area-fraction", type=float, default=0.35)
    parser.add_argument("--padding-fraction", type=float, default=0.10)
    args = parser.parse_args()
    args._vocabulary = []
    args.out.mkdir(parents=True, exist_ok=True)

    from osg.perception.feature_encoder import FeatureEncoder

    backbones = ("vitb32", "mobileclip_s2") if args.backbone == "both" else (args.backbone,)
    result: Dict[str, Any] = {
        "maps": str(args.maps), "run": str(args.run), "preset": args.preset,
        "prompts": list(args.prompts), "primary_prompt": args.prompts[0],
        "part_a": {}, "part_b": {}, "thresholds": {},
    }
    for backbone in backbones:
        print(f"[{backbone}] loading", flush=True)
        encoder = FeatureEncoder(backbone=backbone, checkpoint=str(args.checkpoint),
                                 device=args.device)
        if args.part in ("A", "both"):
            print(f"[{backbone}] part A: prior maps", flush=True)
            result["part_a"][backbone] = part_a(
                encoder, backbone, args, args.prompts,
                list(args.queries) + list(args.controls))
        if args.part in ("B", "both"):
            print(f"[{backbone}] part B: dumped frames", flush=True)
            rows = part_b(encoder, backbone, args, args.prompts, args.queries)
            result["part_b"][backbone] = rows
            for query in sorted({r["query"] for r in rows}):
                pos = [r for r in rows if r["query"] == query and r["source"] == "yoloe"
                       and r["prompt"] == result["primary_prompt"] and r["range_m"] < 2.0
                       and r["covered"]]
                neg = [r for r in rows if r["query"] != query and r["source"] == "yoloe"
                       and r["prompt"] == result["primary_prompt"]]
                if pos:
                    result["thresholds"][f"{backbone}/{query}"] = threshold_curve(pos, neg)
        del encoder
        import torch
        torch.cuda.empty_cache()

    result["gates"] = evaluate_gates(result)
    out = args.out / "FEATURE_MEMORY_PROBE.md"
    out.with_suffix(".json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    write_report(result, out)
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
