"""Shared pieces of the ICRA video build: frame geometry, fonts, text, episode logs.

`make_assets.py` renders every shot of `shotlist.json` into a 1280x720 / 24 fps segment;
`build_video.py` concatenates them and encodes under the 20 MB attachment limit. Both read
the same shot list, and `docs/VIDEO.md`'s storyboard table is printed from it, so the doc
and the cut cannot drift apart.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORKSPACE = pathlib.Path(__file__).resolve().parents[2]
SHOTLIST = pathlib.Path(__file__).resolve().parent / "shotlist.json"
OUT_ROOT = WORKSPACE / "outputs" / "video"
ASSETS = OUT_ROOT / "assets"
SEGMENTS = OUT_ROOT / "segments"

W, H, FPS = 1280, 720, 24

FONT_DIR = pathlib.Path("/usr/share/fonts/truetype/dejavu")
_FONTS: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(size: int, weight: str = "regular", mono: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    if weight == "bold":
        name += "-Bold"
    key = (name, size)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(str(FONT_DIR / f"{name}.ttf"), size)
    return _FONTS[key]


# Palette: one dark ground, one accent per role. Kept few so the cut reads as one thing.
INK = (24, 26, 30)
PAPER = (240, 238, 232)
MUTED = (160, 160, 158)
ACCENT = (232, 120, 48)      # ours / attention
GREEN = (52, 160, 96)        # success / belief high
RED = (204, 60, 52)          # failure / belief low
BLUE = (70, 120, 200)        # reference method


def load_shotlist() -> list[dict]:
    with SHOTLIST.open() as fh:
        data = json.load(fh)
    shots = data["shots"]
    t = 0.0
    for s in shots:
        s["t_start"] = t
        t += float(s["dur"])
    data["total"] = t
    return data


def load_episode(run: str, scene: str, episode_id: str) -> dict:
    path = WORKSPACE / run / scene / "episodes.jsonl"
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("episode_id") == episode_id:
                return rec
    raise FileNotFoundError(f"{episode_id} not in {path}")


def debug_clip_path(run: str, scene: str, episode_id: str) -> pathlib.Path:
    layout = episode_id.split("__")[1]
    return WORKSPACE / run / scene / "viz" / "debug" / f"{scene}_{layout}_ep{episode_id}.mp4"


# ----------------------------------------------------------------------------- drawing

def to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def to_bgr(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt) -> tuple[int, int]:
    l, t, r, b = draw.textbbox((0, 0), text, font=fnt)
    return r - l, b - t


class Canvas:
    """An RGB frame plus the overlay layer the labels are drawn on.

    Labels sit on translucent grounds, so they go onto one RGBA layer that is composited
    once in `finish()`; drawing them straight onto the frame would lose the alpha.
    """

    def __init__(self, base: Image.Image):
        self.base = base.convert("RGB")
        self.layer = Image.new("RGBA", self.base.size, (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.layer)

    def pill(self, xy: tuple[int, int], text: str, fnt, fg=PAPER, bg=(0, 0, 0),
             alpha: int = 170, pad: int = 10, anchor: str = "lt") -> tuple[int, int, int, int]:
        """A rounded label. `anchor` is lt / rt / lb / rb / cb (centre-bottom) / ct."""
        tw, th = text_size(self.draw, text, fnt)
        w, h = tw + 2 * pad, th + 2 * pad
        x, y = xy
        if "r" in anchor:
            x -= w
        if "b" in anchor:
            y -= h
        if anchor.startswith("c"):
            x -= w // 2
        box = (x, y, x + w, y + h)
        self.draw.rounded_rectangle(box, radius=8, fill=(*bg, alpha))
        self.draw.text((x + pad, y + pad - 2), text, font=fnt, fill=(*fg, 255))
        return box

    def paste(self, im: Image.Image, xy: tuple[int, int], border=PAPER, width: int = 2) -> None:
        self.base.paste(im.convert("RGB"), xy)
        x, y = xy
        self.draw.rectangle((x - width, y - width, x + im.width + width - 1, y + im.height + width - 1),
                            outline=(*border, 255), width=width)

    def finish(self) -> Image.Image:
        return Image.alpha_composite(self.base.convert("RGBA"), self.layer).convert("RGB")


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        trial = (cur + " " + w_).strip()
        if text_size(draw, trial, fnt)[0] <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines


def paragraph(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt, width: int,
              fill=PAPER, spacing: int = 8) -> int:
    x, y = xy
    for line in wrap(draw, text, fnt, width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += text_size(draw, "Ag", fnt)[1] + spacing
    return y


def letterbox(im: Image.Image, box: tuple[int, int] = (W, H), bg=INK,
              crop: tuple[float, float, float, float] | None = None) -> Image.Image:
    """Fit an image into `box` keeping aspect; `crop` is a relative (x0, y0, x1, y1) window."""
    if crop is not None:
        w, h = im.size
        im = im.crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
    bw, bh = box
    scale = min(bw / im.width, bh / im.height)
    im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", box, bg)
    canvas.paste(im, ((bw - im.width) // 2, (bh - im.height) // 2))
    return canvas


def fade_edges(frames: Iterable[np.ndarray], n_total: int, fade_s: float = 0.3):
    """Multiply the first and last `fade_s` seconds by a ramp: a cut from black and to black."""
    n_fade = max(1, int(fade_s * FPS))
    for i, f in enumerate(frames):
        k = 1.0
        if i < n_fade:
            k = i / n_fade
        elif i >= n_total - n_fade:
            k = max(0.0, (n_total - 1 - i) / n_fade)
        yield f if k >= 1.0 else (f.astype(np.float32) * k).astype(np.uint8)


# ----------------------------------------------------------------------------- ffmpeg

class SegmentWriter:
    """Pipe raw BGR frames into ffmpeg; one intermediate H.264 file per shot."""

    def __init__(self, path: pathlib.Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE)
        self.n = 0

    def write(self, frame: np.ndarray) -> None:
        assert frame.shape == (H, W, 3), frame.shape
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.n += 1

    def close(self) -> None:
        self.proc.stdin.close()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")


def probe(path: pathlib.Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_name",
         "-show_entries", "format=duration,size", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)
