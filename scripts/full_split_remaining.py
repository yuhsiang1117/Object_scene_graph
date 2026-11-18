"""Which episodes of a split have not been scored yet.

The runner appends to `episodes.jsonl` (`runner.py:241`), so a run that dies
part-way leaves a valid partial record and the rest can be asked for by id.
This derives that set from one or more output directories and emits it as the
Hydra override the runner takes.

    python scripts/full_split_remaining.py outputs/full_final outputs/full_final_resume
    python scripts/full_split_remaining.py --write /tmp/arg.txt outputs/...   # prints the count

Deriving the set fresh on every attempt is what makes the supervisor safe to
restart: it can never re-run a scored episode or skip an unscored one, however
many times the process died.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
from pathlib import Path

DEFAULT_CONTENT = "data/datasets/objectnav/hm3d/v1/val/content"


def scored(dirs: list) -> set:
    done = set()
    for d in dirs:
        f = Path(d) / "episodes.jsonl"
        if not f.is_file():
            continue
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # A kill mid-write can leave one truncated line; the episode it
                # described is simply treated as unscored and run again.
                continue
            done.add((r["scene"], str(r["episode_id"])))
    return done


def every_episode(content_dir: str) -> list:
    out = []
    for f in sorted(glob.glob(f"{content_dir}/*.json.gz")):
        d = json.load(gzip.open(f))
        for i, e in enumerate(d["episodes"]):
            out.append((e["scene_id"].split("/")[-1], str(i)))
    if not out:
        raise SystemExit(f"no episodes found under {content_dir}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", help="output dirs already holding episodes.jsonl")
    p.add_argument("--content-dir", default=DEFAULT_CONTENT)
    p.add_argument("--write", default="", help="write the Hydra override here; print the count")
    a = p.parse_args()

    allep = every_episode(a.content_dir)
    done = scored(a.dirs)
    rem = [f"{s}:{i}" for s, i in allep if (s, i) not in done]
    arg = "eval.episode_ids=[" + ",".join(rem) + "]"
    if a.write:
        Path(a.write).write_text(arg)
        print(len(rem))
        print(f"{len(done)} of {len(allep)} scored, {len(rem)} left, "
              f"{len({r.split(':')[0] for r in rem})} scenes", file=sys.stderr)
    else:
        print(arg)


if __name__ == "__main__":
    main()
