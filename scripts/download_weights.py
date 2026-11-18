"""Stage every optional model asset used by OSG and the ASCENT policies.

YOLOE/MobileCLIP, CLIP, PointNav and RedNet can be selected independently so
CPU-only development does not pull large checkpoints accidentally.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

WEIGHTS = {"small": "yoloe-11s-seg.pt", "medium": "yoloe-11m-seg.pt", "large": "yoloe-11l-seg.pt"}
POINTNAV_SOURCES = (
    "data/external/ascent/third_party/vlfm/data/pointnav_weights.pth",
    "relative_work/ascent/third_party/vlfm/data/pointnav_weights.pth",
)
REDNET_DRIVE_ID = "1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv"


def download_clip(name: str) -> None:
    import clip

    destination = Path("data/clip")
    destination.mkdir(parents=True, exist_ok=True)
    clip.load(name, device="cpu", download_root=str(destination))
    print(f"clip ready: {destination} ({name})")


def _load_vlfm_checkpoint(source: Path) -> dict:
    """Load the reference checkpoint while ignoring its pickled config."""
    import importlib.abc
    import importlib.machinery
    import sys
    import types

    import torch

    class _Placeholder:
        def __init__(self, *_args, **_kwargs):
            pass

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

    class _ShimModule(types.ModuleType):
        __path__: list = []

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            placeholder = type(name, (_Placeholder,), {})
            setattr(self, name, placeholder)
            return placeholder

    absent = tuple(
        package for package in ("habitat_baselines", "vlfm")
        if package not in sys.modules
    )

    class _ShimFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in absent:
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            return _ShimModule(spec.name)

        def exec_module(self, module):
            return None

    finder = _ShimFinder()
    sys.meta_path.insert(0, finder)
    try:
        return torch.load(source, map_location="cpu", weights_only=False)
    finally:
        sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if name.split(".")[0] in absent and isinstance(sys.modules[name], _ShimModule):
                del sys.modules[name]


def stage_pointnav(source_name: str = "") -> None:
    import torch

    from osg.planning.pointnav import load_pointnav_policy, rename_checkpoint_keys

    candidates = [Path(source_name)] if source_name else [Path(p) for p in POINTNAV_SOURCES]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise SystemExit("PointNav source not found; pass --pointnav-src explicitly")
    checkpoint = _load_vlfm_checkpoint(source)
    state = checkpoint.get("state_dict", checkpoint)
    state = rename_checkpoint_keys(dict(state))
    destination = Path("data/weights/pointnav_weights.pth")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)
    load_pointnav_policy(destination)
    print(f"pointnav ready: {destination} ({len(state)} tensors, strict load passed)")


def download_rednet() -> None:
    """Fetch and validate ASCENT's 40-class RedNet checkpoint."""
    import urllib.request

    import torch

    destination = Path("data/weights/rednet_semmap_mp3d_40.pth")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        url = (
            "https://drive.usercontent.google.com/download?export=download&confirm=t"
            f"&id={REDNET_DRIVE_ID}"
        )
        urllib.request.urlretrieve(url, destination)
    try:
        checkpoint = torch.load(destination, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state", checkpoint)
        output = state["module.final_deconv_custom.bias"].shape[0]
        if output != 40:
            raise ValueError(f"expected 40 output classes, got {output}")
    except Exception as exc:
        raise SystemExit(f"RedNet checkpoint validation failed: {destination}: {exc}") from exc
    print(f"rednet ready: {destination} ({destination.stat().st_size} bytes)")


def download_yoloe(profile: str) -> None:
    destination = Path("data/weights")
    destination.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(destination)  # ultralytics downloads into cwd
    try:
        from ultralytics import YOLOE

        name = WEIGHTS[profile]
        model = YOLOE(name)
        names = ["chair", "bed"]
        model.set_classes(names, model.get_text_pe(names))
    finally:
        os.chdir(previous)
    print(f"weights ready: {destination / WEIGHTS[profile]}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=list(WEIGHTS), default="small")
    p.add_argument("--clip", action="store_true")
    p.add_argument("--clip-name", default="ViT-B/32")
    p.add_argument("--pointnav", action="store_true")
    p.add_argument("--pointnav-src", default="")
    p.add_argument("--rednet", action="store_true")
    args = p.parse_args()
    requested = False
    if args.clip:
        download_clip(args.clip_name)
        requested = True
    if args.pointnav:
        stage_pointnav(args.pointnav_src)
        requested = True
    if args.rednet:
        download_rednet()
        requested = True
    if not requested:
        download_yoloe(args.profile)


if __name__ == "__main__":
    main()
