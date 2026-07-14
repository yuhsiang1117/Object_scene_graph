"""HM3D scene + ObjectNav episode download helper (run inside the nav container).

Scenes are license-gated: request access at
https://matterport.com/habitat-matterport-3d-research-dataset then pass your
API token id/secret. Episodes (ObjectNav v2 = HM3D-semantics v0.2, 6
categories) are a public download.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

EPISODES_URL = (
    "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip"
)
DATA = Path("data")


def download_scenes(username: str, password: str, uids: list) -> None:
    cmd = [
        sys.executable, "-m", "habitat_sim.utils.datasets_download",
        "--uids", *uids,
        "--data-path", str(DATA),
        "--username", username,
        "--password", password,
    ]
    subprocess.check_call(cmd)


def download_episodes() -> None:
    dest = DATA / "datasets/objectnav/hm3d"
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "objectnav_hm3d_v2.zip"
    if not zip_path.exists():
        print(f"downloading {EPISODES_URL} ...")
        urllib.request.urlretrieve(EPISODES_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    print(f"episodes extracted under {dest}")
    # Provide a tiny val_mini split (first scene's episodes) for smoke evals
    _make_val_mini(dest)


def _make_val_mini(dest: Path) -> None:
    import gzip
    import json

    val = dest / "v2/val/val.json.gz"
    content_dir = dest / "v2/val/content"
    mini_dir = dest / "v2/val_mini"
    if not val.exists() or (mini_dir / "val_mini.json.gz").exists():
        return
    mini_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(val, "rt") as f:
        top = json.load(f)
    with gzip.open(mini_dir / "val_mini.json.gz", "wt") as f:
        json.dump(top, f)
    scenes = sorted(content_dir.glob("*.json.gz"))[:1]
    mini_content = mini_dir / "content"
    mini_content.mkdir(exist_ok=True)
    for s in scenes:
        (mini_content / s.name).write_bytes(s.read_bytes())
    print(f"val_mini split created from {scenes[0].name if scenes else 'nothing'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--username", help="Matterport API token id")
    p.add_argument("--password", help="Matterport API token secret")
    p.add_argument("--uids", nargs="+", default=["hm3d_minival_v0.2"])
    p.add_argument("--episodes-only", action="store_true")
    args = p.parse_args()

    if not args.episodes_only:
        if not (args.username and args.password):
            p.error("--username/--password required for scene download (see data/README.md)")
        download_scenes(args.username, args.password, args.uids)
    download_episodes()


if __name__ == "__main__":
    main()
