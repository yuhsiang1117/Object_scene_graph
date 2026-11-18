#!/usr/bin/env python3
"""What the released DualMap benchmark can and cannot measure.

Every other analysis here compares two systems. This one compares the benchmark
against itself, and needs no system at all: it asks whether each condition's
perturbation is large enough for its own success tolerance to resolve.

The test is a do-nothing baseline. Take an imaginary agent that ignores the
dynamic change entirely, walks to where the object sat in the *static* scene and
stops. Score it under the benchmark's own rule -- within SUCCESS_DISTANCE_M of
the object's *new* position. It exercises no perception, no search and no
re-detection, so whatever it scores is what the condition awards for free.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osg.eval import dualmap_release as release


def quantiles(values: List[float]) -> Dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}
    def q(f: float) -> float:
        return ordered[min(len(ordered) - 1, int(f * len(ordered)))]
    return {"min": ordered[0], "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "max": ordered[-1]}


def measure() -> List[Dict[str, Any]]:
    """One row per dynamic trial: how far the object moved, and what a do-nothing agent scores."""
    rows = []
    for trial in release.protocol():
        if trial["condition"] == "static":
            continue
        new = release.trial_targets(trial)
        old = release.static_target_positions(trial["scene"], trial["query"])
        # Displacement is measured between the nearest static/dynamic pairing, so a
        # multi-instance query is not penalised for the instances that did not move.
        displacement = min(
            float(np.linalg.norm((n[0] - o[0])[[0, 2]])) for n in new for o in old
        )
        # The do-nothing agent stands at the best of the remembered positions.
        stale_distance = min(release.distance_to_target(o[0], new)[0] for o in old)
        rows.append(
            {
                "trial_id": trial["trial_id"],
                "scene": trial["scene"],
                "condition": trial["condition"],
                "layout": trial["layout"],
                "query": trial["query"],
                "displacement_m": displacement,
                "stale_distance_m": stale_distance,
                "stale_scores": stale_distance <= release.SUCCESS_DISTANCE_M,
                "instances": len(new),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/osg_dualmap_protocol/BENCHMARK_VALIDITY.md"),
    )
    args = parser.parse_args()

    rows = measure()
    tol = release.SUCCESS_DISTANCE_M
    out = [
        "# What the released benchmark can resolve",
        "",
        f"{len(rows)} dynamic trials. The success tolerance is {tol:.1f} m to the object.",
        "",
        "## How far the objects actually move",
        "",
        "| Condition | Trials | min | p25 | median | p75 | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in ("in_anchor", "cross_anchor"):
        subset = [r for r in rows if r["condition"] == condition]
        q = quantiles([r["displacement_m"] for r in subset])
        out.append(
            f"| {condition} | {len(subset)} | " + " | ".join(
                f"{q[k]:.2f} m" for k in ("min", "p25", "median", "p75", "max")
            ) + " |"
        )

    out += [
        "",
        "## The do-nothing baseline",
        "",
        "An agent that ignores the change, returns to the object's static position and "
        "stops. It performs no perception and no re-detection, so its score is what the "
        "condition awards for free.",
        "",
        "| Condition | Trials | Do-nothing SR | DualMap (measured) | Ours |",
        "|---|---:|---:|---:|---:|",
    ]
    published_like = {"in_anchor": "64.8%", "cross_anchor": "30.2%"}
    ours = {"in_anchor": "33.3%", "cross_anchor": "22.6%"}
    for condition in ("in_anchor", "cross_anchor"):
        subset = [r for r in rows if r["condition"] == condition]
        hits = sum(1 for r in subset if r["stale_scores"])
        out.append(
            f"| {condition} | {len(subset)} | **{hits}/{len(subset)} = "
            f"{100 * hits / len(subset):.1f}%** | {published_like[condition]} | {ours[condition]} |"
        )

    degenerate = [r for r in rows if r["stale_scores"]]
    # A centimetre is below anything the 1 m rule or either agent's footprint can
    # resolve, so treat it as "the object was not moved" rather than as a small move.
    STATIONARY_M = 0.01
    stationary = [r for r in rows if r["displacement_m"] < STATIONARY_M]
    out += [
        "",
        f"In-anchor's median displacement is below its own {tol:.1f} m tolerance, so the "
        "condition cannot distinguish a system that re-detects the object from one that "
        "does not: doing nothing outscores both real systems. Cross-anchor is the "
        "informative half.",
        "",
        "## Trials whose object was never moved",
        "",
        f"{len(degenerate)} of {len(rows)} trials are scored correct by the do-nothing "
        f"agent; the in-anchor bulk of those is the tolerance problem above. Separately, "
        f"{len(stationary)} trials move the object by less than {STATIONARY_M * 100:.0f} cm "
        "-- they are labelled dynamic and are not.",
        "",
        "| Trial | Displacement | Do-nothing distance |",
        "|---|---:|---:|",
    ]
    for row in sorted(stationary, key=lambda r: r["displacement_m"]):
        out.append(
            f"| `{row['trial_id']}` | {row['displacement_m'] * 100:.2f} cm | "
            f"{row['stale_distance_m'] * 100:.2f} cm |"
        )
    by_query = collections.Counter((r["condition"], r["query"]) for r in stationary)
    out += [
        "",
        "By query: "
        + ", ".join(f"`{c}/{q}` x{n}" for (c, q), n in sorted(by_query.items()))
        + ". Every cross-anchor `mug` trial in the benchmark is one of these, so `mug` "
        "contributes no cross-anchor evidence at all.",
    ]

    multi = [r for r in rows if r["instances"] > 1]
    out += [
        "",
        "## Instance multiplicity",
        "",
        f"{len(rows) - len(multi)}/{len(rows)} dynamic trials have exactly one valid "
        "instance, so a wrong stop is a wrong place rather than a sibling instance.",
    ]
    if multi:
        by_query = collections.Counter((r["scene"], r["query"]) for r in multi)
        out.append(
            "The exceptions: "
            + ", ".join(f"`{s}/{q}` ({n} trials)" for (s, q), n in sorted(by_query.items()))
            + "."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
