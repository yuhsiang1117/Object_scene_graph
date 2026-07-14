"""Pre-fetch YOLOE weights + the mobileclip text encoder into data/weights so
eval runs are offline-safe (first set_classes() otherwise downloads it).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

WEIGHTS = {"small": "yoloe-11s-seg.pt", "medium": "yoloe-11m-seg.pt", "large": "yoloe-11l-seg.pt"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=list(WEIGHTS), default="small")
    args = p.parse_args()

    dest = Path("data/weights")
    dest.mkdir(parents=True, exist_ok=True)
    os.chdir(dest)  # ultralytics downloads into cwd

    from ultralytics import YOLOE

    name = WEIGHTS[args.profile]
    model = YOLOE(name)  # downloads if missing
    # Trigger the text-encoder (mobileclip) download with a dummy vocabulary
    names = ["chair", "bed"]
    model.set_classes(names, model.get_text_pe(names))
    print(f"weights ready: {dest / name}")


if __name__ == "__main__":
    main()
