"""Pre-fetch model weights so eval runs are offline-safe.

The nav container has NO outbound network, so anything a model would download
lazily has to be staged first. Two destinations, and the difference matters:

  data/weights   a docker NAMED VOLUME -- visible to the container, invisible
                 to the host. Written from inside the container.
  data/clip      an ordinary directory under the bind-mounted repo -- writable
                 from the host, readable by the container at /workspace/data.

Run the YOLOE part inside the container and the CLIP part on the host:

    python scripts/download_weights.py                 # YOLOE + mobileclip
    python scripts/download_weights.py --clip          # CLIP, host side
    python scripts/download_weights.py --pointnav      # PointNav mover (sensor-only nav)
    python scripts/download_weights.py --rednet        # RedNet stair segmentation
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

WEIGHTS = {"small": "yoloe-11s-seg.pt", "medium": "yoloe-11m-seg.pt", "large": "yoloe-11l-seg.pt"}


def download_clip(name: str = "ViT-B/32") -> None:
    """Stage the CLIP checkpoint for the semantic value map (exploration=value)."""
    import clip

    dest = Path("data/clip")
    dest.mkdir(parents=True, exist_ok=True)
    clip.load(name, device="cpu", download_root=str(dest))
    print(f"clip ready: {dest} ({name})")


# ---------------------------------------------------------------------- pointnav

# The frozen point-goal controller ASCENT uses as its mover. It ships in-tree
# with the reference checkout rather than being downloadable, so this is a
# convert-and-stage step, not a fetch -- but it still has to run inside the
# container, because data/weights is a docker NAMED VOLUME.
POINTNAV_SRC = "relative_work/ascent/third_party/vlfm/data/pointnav_weights.pth"


def _load_vlfm_checkpoint(src: Path) -> dict:
    """Unpickle VLFM's checkpoint without habitat_baselines installed.

    The file is a habitat-baselines checkpoint: {"config", "extra_state",
    "state_dict"}, where "config" pickles habitat_baselines and vlfm classes
    that do not exist in this environment. Stand in fabricated modules for the
    duration of the load so the config unpickles into inert placeholders; only
    "state_dict" is kept, and the converted file this writes needs none of it.
    """
    import importlib.abc
    import importlib.machinery
    import sys
    import types

    import torch

    class _Placeholder:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

    class _ShimModule(types.ModuleType):
        __path__: list = []

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            cls = type(name, (_Placeholder,), {})
            setattr(self, name, cls)
            return cls

    # Only stand in for packages that are genuinely absent -- snapshot that
    # BEFORE installing the finder, or the root package the finder itself
    # creates would look "already present" and its submodules would go
    # unshimmed.
    absent = tuple(n for n in ("habitat_baselines", "vlfm") if n not in sys.modules)

    class _ShimFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in absent:
                return importlib.machinery.ModuleSpec(name, self, is_package=True)
            return None

        def create_module(self, spec):
            return _ShimModule(spec.name)

        def exec_module(self, module):
            pass

    finder = _ShimFinder()
    sys.meta_path.insert(0, finder)
    try:
        return torch.load(src, map_location="cpu", weights_only=False)
    finally:
        sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if name.split(".")[0] in absent and isinstance(sys.modules[name], _ShimModule):
                del sys.modules[name]


def stage_pointnav(src: str = POINTNAV_SRC) -> None:
    """Convert VLFM's PointNav checkpoint into a weights-only file.

    Two things happen here rather than at eval time: the pickled config object
    is dropped (so the runtime load can use weights_only=True), and the one
    legacy key name is rewritten. Both are one-time and deterministic; doing
    them per-episode would mean shipping the shim above into the control loop.
    """
    import torch

    from osg.planning.pointnav import rename_checkpoint_keys

    source = Path(src)
    if not source.exists():
        raise SystemExit(
            f"{source} not found. It ships with the ASCENT reference checkout; "
            "point --pointnav-src at a copy of vlfm's pointnav_weights.pth."
        )
    ckpt = _load_vlfm_checkpoint(source)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    renamed = rename_checkpoint_keys({k: v for k, v in state_dict.items()})

    dest_dir = Path("data/weights")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "pointnav_weights.pth"
    torch.save(renamed, dest)

    # Prove the conversion by loading it the way the agent will -- a rename that
    # silently missed would otherwise surface as a randomly-initialised action
    # head, which looks like a merely bad policy rather than a broken one.
    from osg.planning.pointnav import load_pointnav_policy

    load_pointnav_policy(dest)
    print(f"pointnav ready: {dest} ({len(renamed)} tensors, loads strictly)")


# ------------------------------------------------------------------------ rednet

# ASCENT's up-stair signal: RedNet MPCAT40 RGB-D segmentation, whose stair class
# is the thing S14a's decision rule pointed at once camera pitch was ruled out.
# Google Drive, from ASCENT's own README table.
REDNET_DRIVE_ID = "1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv"


def download_rednet(file_id: str = REDNET_DRIVE_ID) -> None:
    """Fetch rednet_semmap_mp3d_40.pth (626 MB) into data/weights.

    Drive serves an interstitial HTML page for anything this large instead of
    the file, so the confirm token and uuid have to be read back out of it and
    replayed. `gdown` does the same dance; it is not installed here.
    """
    import http.cookiejar
    import re
    import urllib.request

    dest_dir = Path("data/weights")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "rednet_semmap_mp3d_40.pth"
    if dest.exists():
        print(f"rednet already staged: {dest} ({dest.stat().st_size} bytes)")
        return

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    base = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
    body = opener.open(base, timeout=60).read()
    if len(body) > 5_000_000:  # small enough to have been served directly
        dest.write_bytes(body)
    else:
        page = body.decode("utf-8", "ignore")
        confirm = re.search(r'name="confirm"\s+value="([^"]+)"', page)
        uuid = re.search(r'name="uuid"\s+value="([^"]+)"', page)
        if not confirm:
            raise SystemExit(
                "Drive did not return a confirm token. Download "
                f"rednet_semmap_mp3d_40.pth by hand into {dest_dir}."
            )
        url = f"{base}&confirm={confirm.group(1)}"
        if uuid:
            url += f"&uuid={uuid.group(1)}"
        resp = opener.open(url, timeout=180)
        with dest.open("wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)

    # Prove it, rather than trusting a byte count: a Drive error page saved to
    # disk is still a file, and would surface much later as a load failure.
    import torch

    ckpt = torch.load(dest, map_location="cpu", weights_only=False)
    n = len(ckpt["model_state"])
    out = ckpt["model_state"]["module.final_deconv_custom.bias"].shape[0]
    assert out == 40, f"expected 40 MPCAT40 classes, got {out}"
    print(f"rednet ready: {dest} ({n} tensors, {out} classes, epoch {ckpt['epoch']})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=list(WEIGHTS), default="small")
    p.add_argument("--clip", action="store_true",
                   help="fetch the CLIP checkpoint into data/clip (run on the HOST)")
    p.add_argument("--clip-name", default="ViT-B/32")
    p.add_argument("--pointnav", action="store_true",
                   help="stage the PointNav mover weights into data/weights "
                        "(agent.navigation=pointnav)")
    p.add_argument("--pointnav-src", default=POINTNAV_SRC)
    p.add_argument("--rednet", action="store_true",
                   help="fetch RedNet MPCAT40 stair segmentation weights (626 MB)")
    args = p.parse_args()

    if args.clip:
        download_clip(args.clip_name)
        return

    if args.pointnav:
        stage_pointnav(args.pointnav_src)
        return

    if args.rednet:
        download_rednet()
        return

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
