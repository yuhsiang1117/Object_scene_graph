#!/usr/bin/env python3
"""Aggregate the DualMap released-benchmark reruns across seeds.

Reports SR and SPL for the static, in-anchor and cross-anchor splits.  DualMap
publishes SR only (Table II; Appendix Tables IX/X are binary success columns),
so the published column is SR and the SPL column has no published counterpart.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any, Dict, List

CONDITIONS = ("static", "in_anchor", "cross_anchor")
LABEL = {"static": "Static", "in_anchor": "In-anchor", "cross_anchor": "Cross-anchor"}
SCENES = ("00829-QaLdnwvtxbs", "00848-ziup5kvtCCR", "00880-Nfvxx8J5NCo")

# Table II of the paper: per-scene SR for each split.
PUBLISHED_SR = {
    ("00829-QaLdnwvtxbs", "static"): 0.731,
    ("00848-ziup5kvtCCR", "static"): 0.692,
    ("00880-Nfvxx8J5NCo", "static"): 0.692,
    ("00829-QaLdnwvtxbs", "in_anchor"): 12 / 18,
    ("00848-ziup5kvtCCR", "in_anchor"): 12 / 18,
    ("00880-Nfvxx8J5NCo", "in_anchor"): 11 / 18,
    ("00829-QaLdnwvtxbs", "cross_anchor"): 10 / 18,
    ("00848-ziup5kvtCCR", "cross_anchor"): 12 / 18,
    ("00880-Nfvxx8J5NCo", "cross_anchor"): 10 / 17,
}
PUBLISHED_OVERALL = {"static": 0.705, "in_anchor": 35 / 54, "cross_anchor": 32 / 53}


def load_seed(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted((root / "trials").glob("*/result.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def load_errors(root: Path) -> List[Dict[str, Any]]:
    """Trials DualMap itself crashed on.  Reported, never silently dropped."""
    rows = []
    for path in sorted((root / "trials").glob("*/error.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def spread(values: List[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def spread_pct(values: List[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return pct(values[0])
    return f"{100 * statistics.mean(values):.1f}% ± {100 * statistics.stdev(values):.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/dualmap_official_bench"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[12, 13, 14])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    per_seed: Dict[int, List[Dict[str, Any]]] = {}
    per_seed_errors: Dict[int, List[Dict[str, Any]]] = {}
    for seed in args.seeds:
        root = args.root / f"seed{seed}"
        if (root / "trials").is_dir():
            per_seed[seed] = load_seed(root)
            per_seed_errors[seed] = load_errors(root)
    if not per_seed:
        raise SystemExit("no seed results found")

    lines: List[str] = []
    lines.append("# DualMap on its own released HM3D benchmark\n")
    lines.append(
        "Official code (`Eku127/DualMap`) and the released `HM3D_collect` dataset, "
        "run unmodified.  SR is DualMap's own metric: the agent stops within 1 m of "
        "the queried object, within three attempts for the dynamic splits.  "
        "**DualMap does not publish SPL** and neither official repository computes "
        "it, so the SPL column is measured here and has no published counterpart.\n"
    )
    seeds_done = ", ".join(str(s) for s in sorted(per_seed))
    lines.append(f"Seeds (agent start pose): {seeds_done}\n")

    # ---- headline table ----
    lines.append("## Overall by split\n")
    lines.append("| Split | Trials/seed | SR (ours) | SPL (ours) | SR (published) |")
    lines.append("|---|---:|---:|---:|---:|")
    for condition in CONDITIONS:
        srs, spls, counts = [], [], []
        for rows in per_seed.values():
            values = [r for r in rows if r["condition"] == condition]
            if not values:
                continue
            counts.append(len(values))
            srs.append(sum(int(r["success"]) for r in values) / len(values))
            spls.append(sum(float(r["spl"]) for r in values) / len(values))
        n = counts[0] if counts else 0
        lines.append(
            f"| {LABEL[condition]} | {n} | {spread_pct(srs)} | {spread(spls)} "
            f"| {pct(PUBLISHED_OVERALL[condition])} |"
        )
    lines.append("")

    # ---- per scene ----
    lines.append("## By scene\n")
    lines.append("| Split | Scene | SR (ours) | SPL (ours) | SR (published) |")
    lines.append("|---|---|---:|---:|---:|")
    for condition in CONDITIONS:
        for scene in SCENES:
            srs, spls = [], []
            for rows in per_seed.values():
                values = [
                    r for r in rows
                    if r["condition"] == condition and r["scene"] == scene
                ]
                if not values:
                    continue
                srs.append(sum(int(r["success"]) for r in values) / len(values))
                spls.append(sum(float(r["spl"]) for r in values) / len(values))
            lines.append(
                f"| {LABEL[condition]} | {scene.split('-')[0]} | {spread_pct(srs)} "
                f"| {spread(spls)} | {pct(PUBLISHED_SR[(scene, condition)])} |"
            )
    lines.append("")

    # ---- per seed detail ----
    lines.append("## Per seed\n")
    lines.append("| Seed | Split | Completed | Successes | SR | SPL |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for seed in sorted(per_seed):
        for condition in CONDITIONS:
            values = [r for r in per_seed[seed] if r["condition"] == condition]
            if not values:
                continue
            successes = sum(int(r["success"]) for r in values)
            lines.append(
                f"| {seed} | {LABEL[condition]} | {len(values)} | {successes} "
                f"| {pct(successes / len(values))} "
                f"| {sum(float(r['spl']) for r in values) / len(values):.3f} |"
            )
    lines.append("")

    # ---- failure taxonomy ----
    lines.append("## Failure modes (dynamic splits)\n")
    lines.append("| Split | Terminal | Trials |")
    lines.append("|---|---|---:|")
    taxonomy: Dict[Any, int] = defaultdict(int)
    for rows in per_seed.values():
        for r in rows:
            if r["condition"] == "static":
                continue
            taxonomy[(r["condition"], r.get("terminal", "?"))] += 1
    for (condition, terminal), count in sorted(taxonomy.items()):
        lines.append(f"| {LABEL[condition]} | {terminal} | {count} |")
    lines.append("")

    # ---- static SPL caveat, with the evidence for it ----
    shortest = [
        float(r["shortest_success_path_m"])
        for rows in per_seed.values()
        for r in rows
        if r["condition"] == "static"
        and r.get("shortest_success_path_m") is not None
    ]
    if shortest:
        trivial = sum(1 for value in shortest if value <= 1.0)
        lines.append("## Note on static SPL\n")
        lines.append(
            "The static split queries HM3DSem classes and counts reaching any "
            "instance, so dense classes give a nearer optimum than the dynamic "
            "splits do.  The effect is real but modest: median shortest successful "
            f"path is {statistics.median(shortest):.2f} m, and "
            f"{trivial}/{len(shortest)} static trials ({100 * trivial / len(shortest):.0f}%) "
            "begin within 1 m of a valid target, where SPL divides by a near-zero "
            "optimum and is uninformative.  Static SPL should therefore be read as a "
            "floor rather than a clean path-efficiency figure.  Static SR is "
            "unaffected and remains DualMap's published metric.\n"
        )

    crashes = [row for rows in per_seed_errors.values() for row in rows]
    lines.append("## Trials DualMap crashed on\n")
    if not crashes:
        lines.append("None.\n")
    else:
        lines.append(
            f"{len(crashes)} trial(s) raised inside DualMap's own code and produced "
            "no result.  They are excluded from the SR/SPL denominators above and "
            "listed here rather than counted as either successes or failures.\n"
        )
        lines.append("| Trial | Error |")
        lines.append("|---|---|")
        for row in crashes:
            lines.append(f"| `{row['trial_id']}` | {row['error']}: {row['message']} |")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
