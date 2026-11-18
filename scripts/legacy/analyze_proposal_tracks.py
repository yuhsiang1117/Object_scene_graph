#!/usr/bin/env python3
"""Set the proposal commit bar from tracks, not from frames.

Reads an observation run (`_sensor_v4_observe`: the stage on every keyframe,
commits off) and, for every track the proposal stage alone has seen, asks two
things: how far is it from the object the trial is scored on, and what are its
`n_proposal_obs` and `proposal_sim`. A track within `--true-m` of the target
is the object; the rest are phantoms. Then sweeps (commit_min_obs, commit_tau)
and reports, per operating point, the tracks and the EPISODES on each side --
an episode counts as "phantom passes" if any phantom track in it would clear
the bar, and as "true passes" if its true track would.

That second table is the one that matters: what lost 18 trials was one
phantom per episode clearing the bar before the detector found the object.
"""
from __future__ import annotations

import argparse, glob, json, math
from collections import defaultdict
from pathlib import Path


def load(root):
    out = {}
    for f in glob.glob(f"{root}/**/episodes.jsonl", recursive=True):
        for line in open(f):
            if line.strip():
                d = json.loads(line); out[d["episode_id"]] = d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="outputs/osg_v4_observe")
    ap.add_argument("--true-m", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("docs/history/PROPOSAL_COMMIT_BAR.md"))
    a = ap.parse_args()

    run = load(a.run)
    rows = []
    for k, d in run.items():
        # Ground truth is the authored layout's object position(s), horizontal
        # (x, z). NOT `target_obj_xy`, which is the agent's own committed
        # goal -- measuring "true" against that would grade phantoms by
        # whether the agent believed them.
        lay = d.get("authored_layout") or {}
        gts = lay.get("target_positions") or ([lay["target_position"]] if lay.get("target_position") else [])
        if not gts:
            continue
        for t in d.get("target_tracks") or []:
            n_p = int(t.get("n_proposal_obs", 0))
            if n_p <= 0 or n_p < int(t["n_obs"]):
                continue                       # named at least once: not this question
            c = t["center"]
            dist = min(math.hypot(c[0] - g[0], c[2] - g[2]) for g in gts)
            rows.append(dict(ep=k, q=k.rsplit("__", 1)[-1], n=n_p,
                             sim=float(t.get("proposal_sim", -1.0)), dist=dist,
                             true=dist <= a.true_m))
    eps = sorted(run)
    T = [r for r in rows if r["true"]]; P = [r for r in rows if not r["true"]]
    L = [f"# The proposal commit bar, set on tracks", "",
         f"run `{a.run}`: {len(eps)} episodes, {len(rows)} proposal-only target-label tracks "
         f"({len(T)} within {a.true_m} m of the object, {len(P)} phantoms)", ""]

    def q(xs, f):
        xs = sorted(xs); return xs[int(f * (len(xs) - 1))] if xs else float("nan")
    L += ["## Per track", "",
          "| population | n | n_proposal_obs p50/p90 | proposal_sim p10/p50/p90 |", "|---|---:|---:|---:|"]
    for name, S in (("true", T), ("phantom", P)):
        if S:
            L.append(f"| {name} | {len(S)} | {q([r['n'] for r in S], .5)}/{q([r['n'] for r in S], .9)} | "
                     f"{q([r['sim'] for r in S], .1):.3f}/{q([r['sim'] for r in S], .5):.3f}/{q([r['sim'] for r in S], .9):.3f} |")

    L += ["", "## Operating points", "",
          "episodes: 'phantom passes' = at least one phantom track clears the bar; "
          "'true passes' = the true track clears it (only episodes that have one)", "",
          "| min_obs | tau | phantom tracks pass | true tracks pass | episodes phantom passes | episodes true passes |",
          "|---:|---:|---:|---:|---:|---:|"]
    ep_true = {r["ep"] for r in T}
    best = []
    for mo in (1, 2, 3, 4, 5, 6, 8):
        for tau in (-1.0, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32):
            ok = lambda r: r["n"] >= mo and r["sim"] >= tau
            pt = sum(map(ok, P)); tt = sum(map(ok, T))
            pe = len({r["ep"] for r in P if ok(r)}); te = len({r["ep"] for r in T if ok(r)})
            L.append(f"| {mo} | {tau:.2f} | {pt}/{len(P)} | {tt}/{len(T)} | {pe}/{len(eps)} | {te}/{len(ep_true)} |")
            best.append((pe, -te, mo, tau))
    L += ["", "## Per query (phantom tracks, and the true track's best statistics)", "",
          "| query | phantoms | true tracks | true max n_obs | true max sim | phantom p90 sim |", "|---|---:|---:|---:|---:|---:|"]
    byq = defaultdict(lambda: {"T": [], "P": []})
    for r in rows: byq[r["q"]]["T" if r["true"] else "P"].append(r)
    for qname in sorted(byq):
        t, p = byq[qname]["T"], byq[qname]["P"]
        L.append(f"| {qname} | {len(p)} | {len(t)} | {max([r['n'] for r in t], default=0)} | "
                 f"{max([r['sim'] for r in t], default=float('nan')):.3f} | {q([r['sim'] for r in p], .9) if p else float('nan'):.3f} |")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
