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

EPISODES_URL = {
    # ObjectNav v2 = HM3D-semantics v0.2, 6 categories (current default).
    "v2": "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip",
    # ObjectNav v1 = HM3D-semantics v0.1 (matches the OLD ROS system; pair with
    # `--uids hm3d_val_v0.1` for the v0.1 render scenes).
    "v1": "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip",
}
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


def download_episodes(version: str = "v2") -> None:
    url = EPISODES_URL[version]
    dest = DATA / "datasets/objectnav/hm3d"
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / f"objectnav_hm3d_{version}.zip"
    if not zip_path.exists():
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    # The zip's top-level dir is objectnav_hm3d_<version>/; configs expect <version>/
    link = dest / version
    extracted = dest / f"objectnav_hm3d_{version}"
    if not link.exists() and extracted.exists():
        link.symlink_to(extracted.name)
    print(f"episodes extracted under {dest} (splits: "
          f"{sorted(p.name for p in link.iterdir() if p.is_dir())})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--username", help="Matterport API token id")
    p.add_argument("--password", help="Matterport API token secret")
    p.add_argument("--uids", nargs="+", default=["hm3d_minival_v0.2"])
    p.add_argument("--episodes-only", action="store_true")
    p.add_argument("--episodes-version", choices=sorted(EPISODES_URL), default="v2",
                   help="ObjectNav episode version: v2 (default, HM3D-sem v0.2) "
                        "or v1 (HM3D-sem v0.1, matches the old ROS system)")
    args = p.parse_args()

    if not args.episodes_only:
        if not (args.username and args.password):
            p.error("--username/--password required for scene download (see data/README.md)")
        download_scenes(args.username, args.password, args.uids)
    download_episodes(args.episodes_version)


if __name__ == "__main__":
    main()
