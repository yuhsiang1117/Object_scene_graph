#!/usr/bin/env python3
"""Pack the revision's runs into one directory that can be released and re-analysed away from
this machine: every episode record and log (not the frames), the inputs a re-run needs, which
code each run ran on, and the tables re-derived from the bundle itself. `docs/RELEASE.md`
describes the result and becomes its README.

    python scripts/release/package.py --out outputs/release_2026-11-04 --tar
    python scripts/release/package.py --out outputs/release_x --dry-run    # what, and how big

What goes in, by source (all under --outputs unless stated):

  benchmark_v2/runs/        every `mfv2/<arm>_s<seed>`; the stale-twin subset from
                            `mfv2/_profile/arm_smoke`, renamed to the same `<arm>_s<seed>` layout
  benchmark_v2/checks/      the final arm re-run on the ablations' code (determinism) and the
                            speed-up's profiling / equivalence runs
  benchmark_v2/inputs/      pass-1 prior maps, authored layouts (JSON, not renders), affinity
                            caches, the map-coverage audit
  benchmark_v2/campaign/    the lane queue, chain scripts and logs, launcher done-lists
  benchmark_v2/report/      `report_2026-10-25`
  dualmap_release/          MD-SG's runs on DualMap's release, DualMap's two reruns, the four
                            static-map baselines' navigation runs and rankings, MD-SG's prior maps
  provenance/               code_provenance.csv, a patch for every uncommitted code snapshot, the
                            code at HEAD (git archive), the session's orchestration scripts, and
                            with --git-bundle the branch's history

Only text files are copied from run directories (JSON, JSONL, CSV, logs, YAML, scripts);
keyframes, viz and videos stay behind (1.6 GB of viz alone), as do the baselines' raw maps
(Clio's are 187 GB) -- `scripts/baselines/README.md` rebuilds those.

Secrets: the config a run records includes the VLM's API key (summary.json `verify_api_key`).
Every text file is redacted on the way in -- API-key-shaped strings, `*api_key` / `*token` /
`*secret` / `*password` fields, and every secret value of docker/.env and the environment --
and the finished bundle is scanned byte for byte for those values; a hit fails the build.
No secret value is ever printed.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from analyze_multifloor import scored  # noqa: E402

TEXT_EXT = {".json", ".jsonl", ".csv", ".log", ".out", ".txt", ".md", ".yaml", ".yml", ".sh", ".py"}
TEXT_NAMES = {".failed"}
SKIP_DIRS = {"keyframes", "viz", "debug", "verify_debug", "renders", "__pycache__"}
SCENES_V2 = ("00800-TEEsavR23oF", "00821-eF36g7L6Z9M", "00862-LT9Jq6dN3Ea", "00873-bxsVRursffK",
             "00878-XB4GS9ShBRE")

# ---------------------------------------------------------------- what each run is

FINAL = "osg_landing_vet_live"
LADDER = ["osg", "osg_gate2", "osg_storey", "osg_climb", "osg_descent", "osg_mouth", "osg_mouth_walk",
          "osg_landing", "osg_landing_vet", FINAL]


def role(arm: str) -> str:
    if arm == FINAL:
        return "final arm"
    if arm in ("ascent", "ascent_yoloe"):
        return "baseline"
    if arm == "osg_vet_live_oracle" or arm.startswith("osg_landing_vet_oracle") or arm.endswith("_oracle"):
        return "ceiling (privileged)"
    if arm.startswith("osg_vet_live_"):
        return "ablation of the final arm"
    if arm == "osg_mouth_walk":
        return "previous final arm"
    if arm.startswith("osg_mouth_walk_"):
        return "ablation of the previous final arm"
    if arm in LADDER:
        return "development ladder"
    if arm in ("ascent_1att", "osg_gate"):
        return "superseded (see docs/history)"
    if arm.endswith("_dump"):
        return "diagnostic"
    return "variant, not adopted"


# Launchers that ran one scene per call on a frozen snapshot record only a done-list; the
# snapshot each was started on (`outputs/campaign_trees/<name>`) is from its launch command.
DONE_LISTS = {"_dump_done.txt": ("wip20261028a", "osg_landing_vet_live_dump"),
              "_oracle_done.txt": ("wip20261028b", None),
              "_refute_done.txt": ("wip20261029b", None),
              "_c3_done.txt": ("wip20261029c", None)}
# Runs no lane log records (or records under a name since reused), by run directory:
# the snapshot, and where that is written down. These take precedence over the logs.
MANUAL_TREES = {
    "osg_s42": ("8be0651", "chain_step1.log: 'chain start, code .../8be0651'"),
    "ascent_1att_s42": ("8be0651", "chain_step1.log; ran as ascent_s42 with eval.attempts=1, renamed"),
    "osg_gate_s42": ("58a0b96", "chain_gate.log: 'gate chain start, code .../58a0b96'"),
    "ascent_s43": ("8ab564d", "queue/jobs.txt: 'ascent s43 runs outside the queue (started by chain_v2b.sh)', "
                              "whose ASCENT lane runs 8ab564d"),
    "osg_landing_vet_live_yesfinal_s42": ("53659ca", "inferred: started 2026-10-23 14:34:25, the minute that "
                                                     "snapshot was made; stopped 20:31 with s43/s44"),
    "osg_vet_live_no_twins_s42": ("worktree", "arm_smoke.sh from the worktree, 2026-10-29 10:50-12:08, "
                                              "before 190ad3f committed its preset"),
    "osg_vet_live_no_twins_s43": ("worktree", "as seed 42"),
    "osg_vet_live_no_twins_s44": ("worktree", "as seed 42"),
}
# Candidate commits a work-in-progress snapshot is diffed against (the nearest is used).
WIP_BASES = ("ee43aba", "29fb05e", "619be9a", "190ad3f")
CODE_DIRS = ("src", "configs", "scripts", "tests")


# ---------------------------------------------------------------- secrets

class Redactor:
    PATTERNS = [
        re.compile(rb"nvapi-[A-Za-z0-9_\-]{20,}"),
        re.compile(rb"\bsk-[A-Za-z0-9_\-]{20,}"),
        re.compile(rb"\bhf_[A-Za-z0-9]{30,}"),
        re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    ]
    FIELD = re.compile(rb'("[A-Za-z0-9_]*(?:api_key|apikey|access_token|auth_token|secret|password)"\s*:\s*)"[^"]*"',
                       re.IGNORECASE)
    YAML_FIELD = re.compile(rb"^(\s*[A-Za-z0-9_]*(?:api_key|apikey|access_token|auth_token|secret|password)\s*:\s*)(\S.*)$",
                            re.IGNORECASE | re.MULTILINE)
    MARK = b"<redacted>"

    def __init__(self, env_files: Iterable[Path]):
        values = set()
        name = re.compile(r"(KEY|TOKEN|SECRET|PASS)", re.IGNORECASE)
        for f in env_files:
            if not f.is_file():
                continue
            for line in f.read_text(errors="replace").splitlines():
                k, sep, v = line.strip().partition("=")
                if sep and not k.startswith("#") and name.search(k):
                    v = v.strip().strip("'\"")
                    if len(v) >= 8:
                        values.add(v.encode())
        for k, v in os.environ.items():
            if name.search(k) and len(v) >= 8 and not v.startswith("/"):
                values.add(v.encode())
        self.values = sorted(values, key=len, reverse=True)
        self.hits: Counter = Counter()

    def text(self, data: bytes) -> bytes:
        for v in self.values:
            if v in data:
                self.hits["secret value from env"] += data.count(v)
                data = data.replace(v, self.MARK)
        for p in self.PATTERNS:
            data, n = p.subn(self.MARK, data)
            self.hits[f"key-shaped string {p.pattern[:8].decode()}"] += n
        data, n = self.FIELD.subn(rb'\1"' + self.MARK + b'"', data)
        self.hits["secret-named JSON field"] += n
        data, n = self.YAML_FIELD.subn(rb"\1" + self.MARK, data)
        self.hits["secret-named YAML field"] += n
        return data

    def leaks(self, data: bytes) -> int:
        return sum(data.count(v) for v in self.values) + sum(len(p.findall(data)) for p in self.PATTERNS)


# ---------------------------------------------------------------- copying

class Bundle:
    def __init__(self, out: Path, redactor: Redactor, dry: bool):
        self.out, self.r, self.dry = out, redactor, dry
        self.bytes: Counter = Counter()
        self.files: Counter = Counter()

    def _section(self, rel: Path) -> str:
        return "/".join(rel.parts[:2])

    def file(self, src: Path, rel: str, text: Optional[bool] = None) -> None:
        dst = self.out / rel
        is_text = text if text is not None else (src.suffix in TEXT_EXT or src.name in TEXT_NAMES)
        key = self._section(Path(rel))
        if self.dry:
            self.bytes[key] += src.stat().st_size
            self.files[key] += 1
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if is_text:
            dst.write_bytes(self.r.text(src.read_bytes()))
        else:
            shutil.copy2(src, dst)
        self.bytes[key] += dst.stat().st_size
        self.files[key] += 1

    def tree(self, src: Path, rel: str, binary_ext: Tuple[str, ...] = (), exclude: Tuple[str, ...] = ()) -> None:
        """Every text file below src (and files with binary_ext), keyframes / viz left out."""
        if not src.exists():
            print(f"  missing, skipped: {src}")
            return
        for p in sorted(src.rglob("*")):
            sub = p.relative_to(src)
            if not p.is_file() or SKIP_DIRS & set(sub.parts[:-1]) or any(str(sub).startswith(x) for x in exclude):
                continue
            if p.suffix in TEXT_EXT or p.name in TEXT_NAMES:
                self.file(p, f"{rel}/{p.relative_to(src)}", text=True)
            elif p.suffix in binary_ext:
                self.file(p, f"{rel}/{p.relative_to(src)}", text=False)

    def write(self, rel: str, content: str) -> None:
        if self.dry:
            return
        dst = self.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(self.r.text(content.encode()))


# ---------------------------------------------------------------- provenance

def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def tree_blobs(tree: Path) -> Dict[str, str]:
    out = {}
    for d in CODE_DIRS:
        for p in (tree / d).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                out[str(p.relative_to(tree))] = blob_sha(p.read_bytes())
    return out


def commit_blobs(commit: str) -> Dict[str, str]:
    out = {}
    for line in git("ls-tree", "-r", commit, "--", *CODE_DIRS).splitlines():
        meta, path = line.split("\t", 1)
        out[path] = meta.split()[2]
    return out


def unified_patch(commit: str, tree: Path, paths: List[str]) -> str:
    """A `git apply`-able patch taking `commit` to the snapshot, for the text files that differ."""
    chunks = []
    for path in paths:
        new_p = tree / path
        try:
            old = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO, capture_output=True,
                                 check=True).stdout.decode()
        except subprocess.CalledProcessError:
            old = None
        new = new_p.read_text() if new_p.is_file() else None
        a = f"a/{path}" if old is not None else "/dev/null"
        b = f"b/{path}" if new is not None else "/dev/null"
        diff = list(difflib.unified_diff((old or "").splitlines(True), (new or "").splitlines(True), a, b))
        if diff:
            if diff[-1] and not diff[-1].endswith("\n"):
                diff[-1] += "\n\\ No newline at end of file\n"
            chunks.append(f"diff --git a/{path} b/{path}\n" + ("new file mode 100644\n" if old is None else "")
                          + ("deleted file mode 100644\n" if new is None else "") + "".join(diff))
    return "".join(chunks)


class Provenance:
    """Which code each (arm, seed) ran on, and whether that code is a commit of this repo."""

    def __init__(self, mfv2: Path, trees: Path):
        self.mfv2, self.trees = mfv2, trees
        self.runs: Dict[Tuple[str, int], List[Tuple[str, str]]] = {}
        self._checked: Dict[str, Tuple[str, str, Optional[str]]] = {}
        start = re.compile(r"start (\S+) s(\d+) \((?:\S*/campaign_trees/)?([^)\s]+)\)")
        chained = re.compile(r"\] (\S+) s(\d+) exit \d+ \((?:\S*/campaign_trees/)?([^)\s]+)\)")
        for log in sorted([*(mfv2 / "queue").glob("*.log"), *mfv2.glob("chain_*.log")]):
            for line in log.read_text(errors="replace").splitlines():
                m = start.search(line) or chained.search(line)
                if m:
                    self.add(m.group(1), int(m.group(2)), m.group(3), f"{log.relative_to(mfv2)}")
        # The queue's own record: "<date> <time> <lane> exit=N <snapshot> <arm> <seed>".
        queued = re.compile(r"exit=\d+ (?:\S*/campaign_trees/)?(\S+) (\S+) (\d+)\s*$")
        if (mfv2 / "queue/done.txt").is_file():
            for line in (mfv2 / "queue/done.txt").read_text(errors="replace").splitlines():
                m = queued.search(line)
                if m:
                    self.add(m.group(2), int(m.group(3)), m.group(1), "queue/done.txt")
        done = re.compile(r"exit=\d+ (?:(\S+) )?s(\d+) \S+")
        for name, (tree, arm) in DONE_LISTS.items():
            f = mfv2 / name
            if f.is_file():
                for line in f.read_text().splitlines():
                    m = done.search(line)
                    if m:
                        self.add(m.group(1) or arm, int(m.group(2)), tree, name)

    def for_run(self, run_dir: str, arm: str, seed: int) -> List[Tuple[str, str]]:
        if run_dir in MANUAL_TREES:
            return [MANUAL_TREES[run_dir]]
        return self.runs.get((arm, seed), [])

    def add(self, arm: str, seed: int, tree: str, source: str) -> None:
        lst = self.runs.setdefault((arm, seed), [])
        if tree not in [t for t, _ in lst]:
            lst.append((tree, source))

    def check(self, tree: str, patches: Bundle) -> Tuple[str, str, Optional[str]]:
        """(commit, how it matches, patch path) for a snapshot name."""
        if tree in self._checked:
            return self._checked[tree]
        path = self.trees / tree
        if tree == "worktree":
            res = ("190ad3f", "worktree before the commit; no snapshot kept", None)
        elif re.fullmatch(r"[0-9a-f]{7,40}", tree):
            full = git("rev-parse", "--verify", f"{tree}^{{commit}}").strip()
            inside = subprocess.run(["git", "merge-base", "--is-ancestor", full, "HEAD"], cwd=REPO).returncode == 0
            if path.is_dir():
                tb, cb = tree_blobs(path), commit_blobs(full)
                n = sum(tb.get(p) != cb.get(p) for p in set(tb) | set(cb))
                how = "snapshot == commit" if n == 0 else f"snapshot differs from commit in {n} files"
            else:
                how = "commit (snapshot not kept)"
            res = (full[:12], how + ("" if inside else ", NOT in this branch's history"), None)
        elif path.is_dir():
            tb = tree_blobs(path)
            best = None
            for base in WIP_BASES:
                cb = commit_blobs(base)
                diff = sorted(p for p in set(tb) | set(cb) if tb.get(p) != cb.get(p))
                if best is None or len(diff) < len(best[1]):
                    best = (base, diff)
            base, diff = best
            rel = f"provenance/patches/{tree}.patch"
            patches.write(rel, unified_patch(base, path, diff))
            full = git("rev-parse", "--verify", f"{base}^{{commit}}").strip()
            res = (full[:12], f"uncommitted snapshot = commit + {rel} ({len(diff)} files)", rel)
        else:
            res = ("", "snapshot missing", None)
        self._checked[tree] = res
        return res


# ---------------------------------------------------------------- sections

def out_header(out_file: Path) -> Dict[str, str]:
    """arm / preset / seed / original dir from run_mfv2.sh's first log line."""
    if out_file.is_file():
        for line in out_file.read_text(errors="replace").splitlines():
            m = re.search(r"arm=(\S+) preset=(\S+) seed=(\d+) -> (\S+)", line)
            if m:
                return {"arm": m.group(1), "preset": m.group(2), "seed": m.group(3), "dir": m.group(4)}
    return {}


def run_stats(d: Path) -> Tuple[int, int, float, int]:
    """(episodes, successes on the object's storey, wall hours, scene summaries)."""
    n = ok = 0
    wall = 0.0
    for f in d.rglob("episodes.jsonl"):
        if SKIP_DIRS & set(f.relative_to(d).parts):
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                n += 1
                ok += scored(e)["success"]
                wall += float(e.get("wall_time_s") or 0)
    return n, ok, wall / 3600, len(list(d.glob("*/summary.json")))


def benchmark_v2(b: Bundle, outputs: Path, layouts: Path, prov: Provenance, manifest: List[dict]) -> None:
    mfv2 = outputs / "mfv2"
    runs = sorted(d for d in mfv2.iterdir() if d.is_dir() and re.fullmatch(r"[a-z].*_s\d+", d.name))
    print(f"benchmark_v2: {len(runs)} runs")
    for d in runs:
        b.tree(d, f"benchmark_v2/runs/{d.name}")
        h = out_header(mfv2 / f"{d.name}.out")
        arm, seed = d.name.rsplit("_s", 1)
        stats = run_stats(d)
        notes = []
        if h and Path(h["dir"]).name != d.name:
            notes.append(f"ran as {Path(h['dir']).name}")
        if stats[3] not in (0, len(SCENES_V2)):
            notes.append(f"{stats[3]}/{len(SCENES_V2)} scenes")
        manifest.append(row("benchmark_v2", arm, seed, role(arm), h.get("preset", f"mfv2_{arm}"),
                            prov.for_run(d.name, h.get("arm", arm), int(seed)), prov, b, stats,
                            f"benchmark_v2/runs/{d.name}", f"mfv2/{d.name}", "; ".join(notes)))
    # Every lane's console log, the failed and superseded attempts' among them.
    for f in sorted(mfv2.glob("[a-z]*.out")):
        b.file(f, f"benchmark_v2/runs/{f.name}")
    # The stale-twin ablation ran only where the rule fires, one episode per call; renamed to the
    # campaign's `<arm>_s<seed>` layout so the analyzers pair it like any other arm.
    twins = mfv2 / "_profile/arm_smoke/mfv2_osg_vet_live_no_twins"
    for sd in sorted(twins.glob("s*")):
        name = f"osg_vet_live_no_twins_{sd.name}"
        b.tree(sd, f"benchmark_v2/runs/{name}")
        manifest.append(row("benchmark_v2", "osg_vet_live_no_twins", sd.name[1:], "ablation of the final arm (subset)",
                            "mfv2_osg_vet_live_no_twins", prov.for_run(name, "", 0), prov, b, run_stats(sd),
                            f"benchmark_v2/runs/{name}", str(sd.relative_to(outputs)),
                            "only the episodes where the final arm retired a stale twin"))
    b.tree(mfv2 / "_rerun_ee43aba", "benchmark_v2/checks/rerun_ee43aba")
    b.tree(mfv2 / "_profile", "benchmark_v2/checks/profile", exclude=("arm_smoke/mfv2_osg_vet_live_no_twins",))
    # Inputs a pass-2 re-run reads.
    b.tree(outputs / "maps_v2_ascent", "benchmark_v2/inputs/prior_maps/maps_v2_ascent", binary_ext=(".npz", ".npy", ".png"))
    b.tree(outputs / "maps_v2_osg", "benchmark_v2/inputs/prior_maps/maps_v2_osg", binary_ext=(".npz", ".npy"))
    for f in ("affinity_cache.json", "affinity_cache_llm_only.json"):
        if (outputs / f).is_file():
            b.file(outputs / f, f"benchmark_v2/inputs/{f}")
    b.tree(layouts, "benchmark_v2/inputs/layouts/dualmap_multifloor")
    b.tree(outputs / "audit", "benchmark_v2/inputs/audit")
    if (mfv2 / "audit_v2.txt").is_file():
        b.file(mfv2 / "audit_v2.txt", "benchmark_v2/inputs/audit/audit_v2.txt")
    # How the campaign was driven.
    b.tree(mfv2 / "queue", "benchmark_v2/campaign/queue")
    for f in sorted([*mfv2.glob("chain_*"), *mfv2.glob("_*_done.txt"), *mfv2.glob("*.sh")]):
        b.file(f, f"benchmark_v2/campaign/{f.name}")
    b.tree(outputs / "report_2026-10-25", "benchmark_v2/report", binary_ext=(".png", ".jpg", ".mp4"))


DUALMAP_RUNS = [
    # (bundle name, source below outputs, what it is)
    ("mdsg_paper_artifact", "osg_v4_fuse_cls_full", "MD-SG, the run Table I's dynamic rows were taken from"),
    ("mdsg_rerun_original_maps", "RERUN_UPLOADMAP", "MD-SG re-run on the original prior maps (2026-09-27)"),
    ("mdsg_rerun_rebuilt_maps", "RERUN/dualmap186", "MD-SG re-run on rebuilt prior maps (2026-09-20)"),
    ("dualmap_rerun_oct", "dualmap_official_bench", "DualMap's released runner, Oct 2026 rerun"),
    ("dualmap_rerun_nov", "baselines/dualmap_rerun", "DualMap's released runner, Nov 2026 rerun"),
    ("khronos", "baselines/khronos/nav", "Khronos (static map), shared navigator"),
    ("clio_batch", "baselines/clio/nav", "Clio-batch (static map), shared navigator"),
    ("clio_batch_cd", "baselines/clio/nav_cd", "Clio-batch with Khronos change detection on (check)"),
    ("hovsg", "baselines/hovsg/nav", "HOV-SG (static map), shared navigator"),
    ("conceptgraphs", "baselines/conceptgraphs/nav", "ConceptGraphs (static map), shared navigator"),
]
RANKINGS = {
    "khronos": "baselines/khronos/map/{scene}/rankings.json",
    "clio_batch": "baselines/clio/map/{scene}/clio/rankings.json",
    "hovsg": "baselines/hovsg/graphs_skip25/hm3dsem/{scene}/rankings.json",
    "conceptgraphs": "baselines/conceptgraphs/data/{scene}/rankings.json",
}


def trial_stats(d: Path) -> Tuple[int, int]:
    n = ok = 0
    for f in d.rglob("result.json"):
        r = json.loads(f.read_text())
        if "trial_id" in r:
            n += 1
            ok += int(r.get("success", 0))
    if n:
        return n, ok
    eps, succ, _, _ = run_stats(d)
    return eps, succ


def dualmap_release(b: Bundle, outputs: Path, manifest: List[dict]) -> None:
    from osg.eval.dualmap_release import SCENES

    for name, src, what in DUALMAP_RUNS:
        s = outputs / src
        b.tree(s, f"dualmap_release/runs/{name}")
        if not s.exists():
            continue
        # One row per seed directory; a directory of scenes is one run; anything else holds
        # several runs (RERUN_UPLOADMAP: the unified arm, v4_fuse_cls, the lost-3 probes).
        kids = sorted(p for p in s.iterdir() if p.is_dir())
        if any(k.name.startswith("seed") for k in kids):
            units = [k for k in kids if k.name.startswith("seed")]
        elif not kids or all(k.name[:5].isdigit() for k in kids):
            units = [s]
        else:
            units = kids
        for u in units:
            n, ok = trial_stats(u)
            sub = "" if u == s else f"/{u.name}"
            manifest.append({"section": "dualmap_release", "name": name + ("" if u == s or u.name.startswith("seed") else sub),
                             "seed": u.name[4:] if u.name.startswith("seed") else "", "role": what,
                             "preset_or_method": name, "code": "", "code_check": "", "episodes": n, "successes": ok,
                             "wall_h": "", "path": f"dualmap_release/runs/{name}{sub}", "source": f"{src}{sub}", "note": ""})
    for name, pattern in RANKINGS.items():
        for scene in SCENES:
            p = outputs / pattern.format(scene=scene)
            if p.is_file():
                b.file(p, f"dualmap_release/rankings/{name}/{scene}.json")
    for name, src in (("original", "maps_released_upload/maps_released"), ("rebuilt", "maps_released")):
        b.tree(outputs / src, f"dualmap_release/prior_maps/{name}", binary_ext=(".npz",))
    b.tree(outputs / "collector_layouts", "dualmap_release/layouts/collector_layouts")
    for f in sorted((outputs / "baselines").glob("*.md")):
        b.file(f, f"dualmap_release/tables/as_reported/{f.name}")


def row(section, arm, seed, role_, preset, trees, prov, b, stats, path, source, note="") -> dict:
    codes, checks, detail = [], [], []
    for tree, src in trees:
        commit, how, patch = prov.check(tree, b)
        codes.append(f"{tree}" + (f"={commit}" if commit and not commit.startswith(tree) else ""))
        checks.append(how)
        detail.append((tree, commit, how, patch, src))
    n, ok, wall_h, _ = stats
    return {"section": section, "name": arm, "seed": seed, "role": role_, "preset_or_method": preset,
            "code": " ; ".join(codes) or "unrecorded", "code_check": " ; ".join(checks),
            "episodes": n, "successes": ok, "wall_h": round(wall_h, 2), "path": path, "source": source,
            "note": note, "_prov": detail}


# ---------------------------------------------------------------- tables re-derived from the bundle

def tables(out: Path) -> None:
    py = sys.executable
    runs = "benchmark_v2/runs"
    dm = "dualmap_release"
    t = "benchmark_v2/tables"
    cmds = [
        (f"{t}/arms.txt", [py, REPO / "scripts/analyze_mfv2.py", "arms", "--root", runs, "--csv", f"{t}/arms_by_seed.csv"]),
        (f"{t}/ablations_final_arm.txt", [py, REPO / "scripts/analyze_mfv2.py", "ablations", "--root", runs]),
        (f"{t}/ablations_previous_final_arm.txt",
         [py, REPO / "scripts/analyze_mfv2.py", "ablations", "--root", runs, "--set", "mouth_walk"]),
        (f"{t}/wall_time.txt", [py, REPO / "scripts/analyze_mfv2.py", "wall", "--root", runs, "--ref", FINAL,
                                "osg_landing_vet_live_follows", "osg_landing_vet_live_dump", "osg_vet_live_no_floor_llm",
                                "osg_landing_vet_oracle_sweep", "osg"]),
        (f"{t}/multifloor_final_vs_ascent.md",
         [py, REPO / "scripts/analyze_multifloor.py", *(f"FINAL={runs}/{FINAL}_s{s}" for s in (42, 43, 44)),
          *(f"ASCENT={runs}/ascent_s{s}" for s in (42, 43, 44))]),
        (f"{dm}/tables/table_I.txt",
         [py, REPO / "scripts/baselines/report_table.py",
          "--run", f"DualMap (rerun)={dm}/runs/dualmap_rerun_nov/seed*",
          "--run", f"DualMap (Oct rerun)={dm}/runs/dualmap_rerun_oct/seed*",
          "--run", f"Khronos={dm}/runs/khronos/seed*", "--run", f"Clio-batch={dm}/runs/clio_batch/seed*",
          "--run", f"HOV-SG={dm}/runs/hovsg/seed*", "--run", f"ConceptGraphs={dm}/runs/conceptgraphs/seed*",
          "--static-map", "Khronos", "--static-map", "Clio-batch", "--static-map", "HOV-SG",
          "--static-map", "ConceptGraphs", "--out", f"{dm}/tables/table_I.md"]),
    ]
    from osg.eval.dualmap_release import SCENES

    for name, src in (("original", f"{dm}/prior_maps/original"), ("rebuilt", f"{dm}/prior_maps/rebuilt")):
        for scene in SCENES:
            cmds.append((f"{dm}/rankings/mdsg_{name}/{scene}.log",
                         [py, REPO / "scripts/baselines/osg_rankings.py", "--maps", src, "--scene", scene,
                          "--out", f"{dm}/rankings/mdsg_{name}/{scene}.json"]))
    cmds.append((f"{dm}/tables/retrieval_quality.txt",
                 [py, REPO / "scripts/baselines/rank_quality.py",
                  *(a for m, label in (("khronos", "Khronos"), ("clio_batch", "Clio-batch"), ("hovsg", "HOV-SG"),
                                       ("conceptgraphs", "ConceptGraphs"), ("mdsg_original", "MD-SG (original map)"),
                                       ("mdsg_rebuilt", "MD-SG (rebuilt map)"))
                    for a in ("--rankings", f"{label}={dm}/rankings/{m}/{{scene}}.json"))]))
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}:{os.environ.get('PYTHONPATH', '')}")
    script = ["#!/usr/bin/env bash", "# Re-derive every table from this bundle. Run from the bundle root with the",
              "# repository checked out at REPO (any commit at or after the one in provenance/).",
              "set -eu", 'REPO=${REPO:?path to the Object_scene_graph checkout}',
              'export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"']
    for target, cmd in cmds:
        args = [str(c).replace(str(REPO), "$REPO") if isinstance(c, Path) else c for c in cmd]
        args[0] = "python"
        script.append(" ".join(a if re.fullmatch(r"[\w./=$+-]+", a) else "'" + a.replace("'", "'\\''") + "'"
                               for a in args) + f" > {target}")
        (out / target).parent.mkdir(parents=True, exist_ok=True)
        res = subprocess.run([str(c) for c in cmd], cwd=out, env=env, capture_output=True, text=True)
        (out / target).write_text(res.stdout + (f"\n[stderr]\n{res.stderr}" if res.returncode else ""))
        print(f"  {'ok ' if res.returncode == 0 else 'ERR'} {target}")
    (out / "REPRODUCE_TABLES.sh").write_text("\n".join(script) + "\n")


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--outputs", type=Path, help="the campaign's outputs/ (default: the main checkout's)")
    ap.add_argument("--layouts", type=Path, default=Path("/habitat-data-collector/outputs/dualmap_multifloor"),
                    help="Benchmark v2's authored layouts (JSON only is copied)")
    ap.add_argument("--env-file", type=Path, action="append",
                    help="secrets to redact (default: docker/.env of this checkout and of the main one)")
    ap.add_argument("--session-scripts", type=Path, help="orchestration scripts to keep under provenance/")
    ap.add_argument("--git-bundle", action="store_true",
                    help="also ship the branch's history (provenance/osg.bundle): every campaign commit "
                         "resolvable offline, but it carries the commits' author names and e-mails")
    ap.add_argument("--tar", action="store_true", help="also write <out>.tar.gz")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    common = Path(git("rev-parse", "--git-common-dir").strip())
    common = common if common.is_absolute() else (REPO / common).resolve()
    # A worktree's own outputs/ is empty; the runs live in the main checkout's.
    outputs = (args.outputs or common.parent / "outputs").resolve()
    env_files = args.env_file or [REPO / "docker/.env", common.parent / "docker/.env"]
    red = Redactor(env_files)
    print(f"redacting {len(red.values)} secret value(s) from {sum(f.is_file() for f in env_files)} env file(s) "
          f"and the environment")
    if args.out.exists() and any(args.out.iterdir()) and not args.dry_run:
        print(f"{args.out} exists and is not empty; refusing to mix two builds", file=sys.stderr)
        return 2
    b = Bundle(args.out, red, args.dry_run)
    prov = Provenance(outputs / "mfv2", outputs / "campaign_trees")
    manifest: List[dict] = []

    benchmark_v2(b, outputs, args.layouts, prov, manifest)
    dualmap_release(b, outputs, manifest)
    if args.session_scripts:
        b.tree(args.session_scripts, "provenance/session_scripts")

    if args.dry_run:
        for k in sorted(b.bytes):
            print(f"  {k:40s} {b.files[k]:6d} files {b.bytes[k] / 1e6:9.1f} MB")
        print(f"  total {sum(b.bytes.values()) / 1e6:.1f} MB")
        return 0

    # Provenance: one row per (run, snapshot), then the code itself.
    head = git("rev-parse", "HEAD").strip()
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    if git("status", "--porcelain", "--", *CODE_DIRS).strip():
        print("warning: uncommitted changes under src/ configs/ scripts/ tests/; code_HEAD is the commit, "
              "not the working tree", file=sys.stderr)
    prov_rows = []
    for m in manifest:
        for tree, commit, how, patch, src in m.pop("_prov", []):
            prov_rows.append({"run": Path(m["path"]).name, "snapshot": tree, "commit": commit, "check": how,
                              "patch": patch or "", "recorded_in": src})
    with open(args.out / "provenance/code_provenance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prov_rows[0]))
        w.writeheader()
        w.writerows(prov_rows)
    # The code at HEAD without history (no author metadata); the history only on request, and
    # only if no secret ever entered it.
    with open(args.out / f"provenance/code_{head[:12]}.tar.gz", "wb") as f:
        subprocess.run(["git", "archive", "--format=tar.gz", f"--prefix=Object_scene_graph-{head[:12]}/", head],
                       cwd=REPO, check=True, stdout=f)
    if args.git_bundle:
        dirty = any(subprocess.run(["git", "log", "--all", "--format=%h", f"-S{v.decode()}"], cwd=REPO,
                                   capture_output=True, text=True).stdout.strip() for v in red.values)
        if dirty:
            print("a secret value appears in the git history; no git bundle written", file=sys.stderr)
        else:
            subprocess.run(["git", "bundle", "create", str((args.out / "provenance/osg.bundle").resolve()), branch],
                           cwd=REPO, check=True, capture_output=True)
    with open(args.out / "MANIFEST.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    env_lines = [f"built from {branch} @ {head}", f"python {sys.version.split()[0]} ({sys.executable})"]
    try:
        env_lines.append(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip())
    except FileNotFoundError:
        pass
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    b.write("ENVIRONMENT.txt", "\n".join(env_lines) + "\n\n# pip freeze of the evaluation env\n" + freeze)

    print("tables, re-derived from the bundle:")
    tables(args.out)

    readme = (REPO / "docs/RELEASE.md").read_text()
    by_role = Counter((r["section"], r["role"]) for r in manifest)
    inv = ["", "## Inventory of this build", "", f"Built from `{branch}` at `{head[:12]}`.", "",
           "| section | what | runs |", "|---|---|---:|"]
    inv += [f"| {s} | {r} | {n} |" for (s, r), n in sorted(by_role.items())]
    inv += ["", "| section | files | MB |", "|---|---:|---:|"]
    inv += [f"| {k} | {b.files[k]} | {b.bytes[k] / 1e6:.1f} |" for k in sorted(b.bytes)]
    b.write("README.md", readme + "\n".join(inv) + "\n")

    # Leak scan, then checksums.
    leaked = []
    for p in sorted(args.out.rglob("*")):
        if p.is_file() and p.name != "osg.bundle" and red.leaks(p.read_bytes()):
            leaked.append(p.relative_to(args.out))
    print("redactions: " + ", ".join(f"{k} x{v}" for k, v in red.hits.items() if v) or "none")
    if leaked:
        print(f"LEAK: {len(leaked)} file(s) still hold a secret, e.g. {leaked[0]}; bundle NOT finished", file=sys.stderr)
        return 3
    with open(args.out / "SHA256SUMS", "w") as f:
        for p in sorted(args.out.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS":
                f.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(args.out)}\n")
    total = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"done: {args.out} ({total / 1e6:.0f} MB), {len(manifest)} runs in MANIFEST.csv")
    if args.tar:
        tgz = args.out.with_name(args.out.name + ".tar.gz")
        subprocess.run(["tar", "-czf", str(tgz), "-C", str(args.out.parent), args.out.name], check=True)
        print(f"     {tgz} ({tgz.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
