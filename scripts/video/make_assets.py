#!/usr/bin/env python3
"""Render every shot of `shotlist.json` into a 1280x720 / 24 fps segment.

    python scripts/video/make_assets.py            # all shots -> outputs/video/segments/
    python scripts/video/make_assets.py --only hero problem_clip
    python scripts/video/make_assets.py --frame hero 12.0   # one PNG at t=12 s, for a look

Sources are the per-episode debug videos (`<run>/<scene>/viz/debug/*.mp4`, 8 fps,
RGB+detections beside a costmap panel), the paper figures under `docs/figures/` and
`outputs/figures/`, `episodes.jsonl`, which supplies every caption number (the floor log, the
FSM state, the presence events), and the real-robot recordings under
`assets/real_experiment/` (phone takes and the Thor's RViz screen captures, paired per step
by `shotlist.json`). Nothing here re-simulates anything.

Frame design for simulator footage: the first-person RGB fills the frame (the 1280x960
panel centre-cropped to 16:9 so the detection boxes stay large), the costmap crop sits as a
bottom-right inset, the badge at top-left reads `target · step · floor · state`, and the
bottom band carries the one-line explanation the narration is making at that moment.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Iterator

import cv2
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import (ACCENT, ASSETS, BLUE, FPS, GREEN, H, INK, MUTED, OUT_ROOT, PAPER, RED,  # noqa: E402
                    SEGMENTS, W, WORKSPACE, Canvas, SegmentWriter, debug_clip_path, fade_edges,
                    font, letterbox, load_episode, load_shotlist, paragraph, probe, to_bgr, to_pil, wrap)

STATE_LABEL = {
    "goto_frontier": "explore to a frontier",
    "explore": "explore",
    "approach": "approach a candidate",
    "close_look": "inspect the surface",
    "climb": "take the stairs",
    "done": "stop",
}

MAP_INSET = (240, 300)   # max size of the whole-map inset
MAP_ZOOM_PX = 320        # side of the agent-following inset


# ----------------------------------------------------------------------------- sources

def _map_bbox(path: pathlib.Path, first: int, last: int) -> tuple[int, int, int, int]:
    """Where the map actually is inside the 480-px costmap panel: the non-grey bbox,
    sampled over the step range so the inset does not jump as the map grows."""
    cap = cv2.VideoCapture(str(path))
    xs, ys = [], []
    for idx in range(first, last + 1, max(1, (last - first) // 12)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            break
        panel = f[:, 1280:].astype(int)
        grey = ((abs(panel[:, :, 0] - panel[:, :, 1]) < 6) & (abs(panel[:, :, 1] - panel[:, :, 2]) < 6)
                & (abs(panel[:, :, 0] - 128) < 12))
        nz = np.argwhere(~grey)
        if len(nz):
            (y0, x0), (y1, x1) = nz.min(0), nz.max(0)
            xs += [x0, x1]
            ys += [y0, y1]
    cap.release()
    if not xs:
        return 0, 0, 480, 960
    m = 8
    return max(0, min(xs) - m), max(0, min(ys) - m), min(480, max(xs) + m), min(960, max(ys) + m)


class ClipSource:
    """Sequential access to one debug video, re-laid out as (full RGB, map inset)."""

    def __init__(self, path: pathlib.Path, kind: str, first: int, last: int, zoom: int = 0):
        self.path, self.kind = path, kind
        self.first, self.last = first, last
        self.zoom = zoom            # panel px of costmap to show around the agent; 0 = whole map
        self.agent_px = None
        self.cap = cv2.VideoCapture(str(path))
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if last >= self.n_frames:
            raise ValueError(f"{path.name}: asked for frame {last}, has {self.n_frames}")
        self.bbox = _map_bbox(path, first, last) if kind == "our" else None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        self.idx = first - 1
        self.cur: tuple[np.ndarray, np.ndarray] | None = None

    def at(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        idx = min(idx, self.last)
        while self.idx < idx:
            ok, f = self.cap.read()
            if not ok:
                break
            self.idx += 1
            self.cur = self._relayout(f)
        assert self.cur is not None
        return self.cur

    def _relayout(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.kind == "our":
            rgb = f[120:840, :1280]
            panel = f[:, 1280:]
            if self.zoom:
                # follow the agent: the renderer draws it as the only saturated red dot
                red = (panel[:, :, 2] > 200) & (panel[:, :, 1] < 60) & (panel[:, :, 0] < 60)
                ys, xs = np.nonzero(red)
                if len(xs):
                    self.agent_px = (int(xs.mean()), int(ys.mean()))
                cx, cy = self.agent_px or ((self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2)
                h_ = self.zoom // 2
                # stay inside the *explored* map as it is now -- free (white) or occupied
                # (dark) cells, not the unknown grey (140) or the panel padding (128): a
                # map narrower than the window gives a narrower crop, scaled up in the inset
                known = ~((abs(panel[:, :, 0].astype(int) - panel[:, :, 1]) < 12)
                          & (abs(panel[:, :, 1].astype(int) - panel[:, :, 2]) < 12)
                          & (panel[:, :, 0] > 105) & (panel[:, :, 0] < 165))
                # a column or row counts as map only with a few known cells in it, so
                # codec noise at the panel's edge cannot stretch the window
                cols = np.nonzero(known.sum(axis=0) >= 6)[0]
                rows = np.nonzero(known.sum(axis=1) >= 6)[0]
                if len(cols) and len(rows):
                    bx0, bx1, by0, by1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
                else:
                    bx0, by0, bx1, by1 = self.bbox
                x0 = min(max(bx0, cx - h_), max(bx0, bx1 - self.zoom))
                y0 = min(max(by0, cy - h_), max(by0, by1 - self.zoom))
                m = panel[y0:min(y0 + self.zoom, by1), x0:min(x0 + self.zoom, bx1)]
            else:
                x0, y0, x1, y1 = self.bbox
                m = panel[y0:y1, x0:x1]
        else:  # ASCENT's own panel: RGB 640x480 | obstacle map 480x480 | value map
            rgb = cv2.resize(f[:, :640], (1280, 960), interpolation=cv2.INTER_CUBIC)[120:840]
            m = f[:, 640:1120]
        box = (MAP_ZOOM_PX, MAP_ZOOM_PX) if self.zoom else MAP_INSET
        s = min(box[0] / m.shape[1], box[1] / m.shape[0])
        m = cv2.resize(m, (max(1, int(m.shape[1] * s)), max(1, int(m.shape[0] * s))),
                       interpolation=cv2.INTER_NEAREST if s > 1 else cv2.INTER_AREA)
        return rgb, m

    def close(self) -> None:
        self.cap.release()


class SeqReader:
    """Frames of an ordinary video (the phone, the RViz capture) at the cut's own rate.

    ffmpeg seeks to `t0` and resamples to `FPS / speed` frames per source second, so each
    `next()` is exactly one output frame at 24 fps with the source running `speed` times
    faster. Decoding through a pipe sidesteps cv2's unreliable seeking in the HEVC phone
    files; nothing is resampled twice.
    """

    def __init__(self, path: pathlib.Path, t0: float, span: float, speed: float = 1.0):
        import subprocess
        info = probe(path)
        self.w, self.h = int(info["streams"][0]["width"]), int(info["streams"][0]["height"])
        self.proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-ss", f"{t0:.3f}", "-t", f"{span + 0.5:.3f}", "-i", str(path),
             "-vf", f"fps={FPS / speed:.6f}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "-"],
            stdout=subprocess.PIPE)
        self.cur = None

    def next(self) -> np.ndarray:
        buf = self.proc.stdout.read(self.w * self.h * 3)
        if len(buf) == self.w * self.h * 3:
            self.cur = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
        assert self.cur is not None, "video ended before the requested time"
        return self.cur

    def close(self) -> None:
        self.proc.kill()
        self.proc.stdout.close()
        self.proc.wait()


def video_frame(path: pathlib.Path, t: float) -> np.ndarray:
    """One frame at `t` seconds, decoded by ffmpeg."""
    import subprocess
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(path),
                          "-frames:v", "1", "-f", "image2", "-vcodec", "png", "-"],
                         capture_output=True, check=True).stdout
    return cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)


# Where the two windows sit in the 1920x1080 RViz screen captures. "A" is the layout of
# step0 and the first part of step1 (camera window without a toolbar); "B" has RViz's
# toolbar and is what step2 and the end of step1 recorded; "full" is the whole screen
# minus the panel, for the operator's map-loading window.
RVIZ_CROPS = {
    "A": {"map": (640, 135, 1900, 1050), "cam": (66, 66, 626, 1066)},
    "B": {"map": (690, 135, 1895, 1045), "cam": (80, 66, 650, 1080)},
    "full": {"map": (66, 40, 1920, 1080), "cam": None},
}


def _fit(bgr: np.ndarray, box: tuple[int, int]) -> np.ndarray:
    s_ = min(box[0] / bgr.shape[1], box[1] / bgr.shape[0])
    return cv2.resize(bgr, (max(1, int(bgr.shape[1] * s_)), max(1, int(bgr.shape[0] * s_))),
                      interpolation=cv2.INTER_AREA)


def _compute_floor_score(cache: pathlib.Path) -> None:
    """Run the paper figure's computation (same functions, same constants) and cache it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fsf", WORKSPACE / "scripts/render_floor_score_formula.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    from osg.exploration.search_belief import InspectionLog
    _, _, _, layer, graph, floors = m.load_graph(m.SCENE, m.MAPS)
    anchor_xy = m.anchor_of(layer, m.TARGET)[[0, 2]]
    heights = {int(f["key"]): float(f["height_y"]) for f in floors}
    start = m.per_floor(graph, m.TARGET, InspectionLog(), anchor_xy)
    here = max(start, key=lambda k: m.mean_of(start[k]))
    other = max((k for k in heights if k != here), key=lambda k: m.mean_of(start.get(k, [])))
    first, means, inspected = m.sweep(graph, m.TARGET, anchor_xy, here, other)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"scene": m.SCENE, "target": m.TARGET, "here": here, "other": other,
                                 "n_here": len(first[here]), "n_other": len(first[other]),
                                 "means": means.tolist(), "margin": m.MARGIN, "inspected": inspected}, indent=1))


def draw_scene_graph(w: int, h: int) -> Image.Image:
    """The hierarchy as a tree big enough to read on a phone: building -> floor -> room ->
    container -> object, two storeys joined by a stair edge. Node labels are the ones the
    paper's 00873 figure carries; the layout is schematic."""
    im = Image.new("RGB", (w, h), (30, 32, 38))
    cv = Canvas(im)
    d = cv.draw
    C_BLD, C_FLR, C_ROOM, C_CONT, C_OBJ = (86, 120, 200), (70, 170, 150), (110, 180, 90), ACCENT, (230, 200, 90)
    cols = [int(w * x) for x in (0.10, 0.30, 0.50, 0.70, 0.90)]
    heads = ["building", "floor", "room", "container", "object"]
    f_head, f_lab = font(17, "bold"), font(16)
    for x, t in zip(cols, heads):
        tw = d.textbbox((0, 0), t, font=f_head)[2]
        d.text((x - tw // 2, 10), t, font=f_head, fill=(*MUTED, 255))
    top, bot = 0.30, 0.74     # the two storeys' rows, as fractions of h

    def node(x, y, r, col, label, above=True):
        d.ellipse((x - r, y - r, x + r, y + r), fill=(*col, 255), outline=(*PAPER, 255), width=2)
        tw, th = d.textbbox((0, 0), label, font=f_lab)[2:]
        d.text((x - tw // 2, y - r - th - 4 if above else y + r + 4), label, font=f_lab, fill=(*PAPER, 255))

    def edge(x0, y0, x1, y1, col=(150, 150, 150), width=3, dash=False):
        if not dash:
            d.line((x0, y0, x1, y1), fill=(*col, 255), width=width)
            return
        n = 14
        for i in range(0, n, 2):
            d.line((x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n,
                    x0 + (x1 - x0) * (i + 1) / n, y0 + (y1 - y0) * (i + 1) / n), fill=(*col, 255), width=width)

    yb = int(h * 0.52)
    rows = {0: {"floor": "floor 0 · y = 0.0 m", "room": "room 2", "cont": ["bed", "refrigerator"], "obj": [["picture"], ["cup"]]},
            1: {"floor": "floor 1 · y = −3.0 m", "room": "room 4", "cont": ["shelf", "cabinet"], "obj": [["box"], ["can"]]}}
    for k, yf in ((0, top), (1, bot)):
        y = int(h * yf)
        edge(cols[0], yb, cols[1], y)
        edge(cols[1], y, cols[2], y)
        spread = int(h * 0.11)
        for j, (c, objs) in enumerate(zip(rows[k]["cont"], rows[k]["obj"])):
            yc = y + (j * 2 - 1) * spread
            edge(cols[2], y, cols[3], yc)
            for o in objs:
                edge(cols[3], yc, cols[4], yc, col=(120, 120, 120), width=2)
                node(cols[4], yc, 7, C_OBJ, o, above=False)
            node(cols[3], yc, 11, C_CONT, c, above=(j == 0))
        node(cols[2], y, 12, C_ROOM, rows[k]["room"])
        node(cols[1], y, 14, C_FLR, rows[k]["floor"])
    # the stair edge between the storeys, written only after a completed traversal
    edge(cols[1], int(h * top) + 16, cols[1], int(h * bot) - 16, col=C_FLR, width=4, dash=True)
    d.text((cols[1] + 12, yb - 10), "stair edge", font=f_lab, fill=(*C_FLR, 255))
    node(cols[0], yb, 16, C_BLD, "building")
    return cv.finish()


def fit_font(cv: Canvas, text: str, width: int, sizes=(30, 28, 26, 24, 22, 20, 18), weight="bold"):
    """The largest of `sizes` at which `text` fits in `width` px."""
    for size in sizes:
        fnt = font(size, weight)
        if cv.draw.textbbox((0, 0), text, font=fnt)[2] <= width:
            return fnt
    return fnt


def caption_band(cv: Canvas, text: str, y: int = H - 14, alpha: int = 200) -> None:
    """A one-line bottom caption, shrunk until it fits the frame width."""
    for size in (26, 24, 22, 20, 18):
        fnt = font(size)
        if cv.draw.textbbox((0, 0), text, font=fnt)[2] + 24 <= W - 32:
            break
    cv.pill((W // 2, y), text, fnt, bg=INK, alpha=alpha, anchor="cb")


SUB_SIZE, SUB_WIDTH, SUB_HOLD = 26, 1100, 0.4    # must match make_narration.subtitle_lines
_CUES: list[dict] | None = None


def load_cues() -> list[dict]:
    """The narration's cues with timings measured from the synthesised audio."""
    global _CUES
    if _CUES is None:
        path = OUT_ROOT / "narration" / "cues.json"
        if not path.exists():
            sys.exit(f"{path} missing: run make_narration.py first (subtitles come from its timings)")
        _CUES = json.loads(path.read_text())
    return _CUES


def subtitle_at(t_abs: float) -> str | None:
    for c in load_cues():
        if c["t0"] <= t_abs < c["t1"] + SUB_HOLD:
            return c["text"]
    return None


def subtitle_band(cv: Canvas, text: str, y: int = H - 14) -> None:
    """The subtitle: up to two centred lines on one translucent ground."""
    fnt = font(SUB_SIZE)
    lines = wrap(cv.draw, text, fnt, SUB_WIDTH)
    lh = SUB_SIZE + 8
    widths = [cv.draw.textbbox((0, 0), ln, font=fnt)[2] for ln in lines]
    w, h = max(widths) + 28, lh * len(lines) + 14
    x0, y0 = W // 2 - w // 2, y - h
    cv.draw.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=8, fill=(*INK, 210))
    for i, ln in enumerate(lines):
        cv.draw.text((W // 2 - widths[i] // 2, y0 + 6 + i * lh), ln, font=fnt, fill=(*PAPER, 255))


def section_chip(cv: Canvas, spec: dict, y: int = 14) -> int:
    """The current section, top-right; returns the y below it for anything stacked there."""
    sec = spec.get("section")
    if not sec:
        return y
    box = cv.pill((W - 16, y), sec, font(20, "bold"), fg=INK, bg=PAPER, alpha=235, pad=8, anchor="rt")
    return box[3] + 8


def _lookup(table: list, step: int, default=None):
    """Last entry whose step <= `step`, for step-indexed logs like state_log / floor_log."""
    out = default
    for row in table:
        if row[0] <= step:
            out = row[1]
    return out


# ----------------------------------------------------------------------------- shots

class Renderer:
    def __init__(self, data: dict):
        self.data = data
        self.meta = data["meta"]
        self.runs = {"our": self.meta["our_run"], "ascent": self.meta["ascent_run"]}

    def episode(self, key: str, run: str) -> tuple[dict, dict]:
        ep = self.data["episodes"][key]
        rec = load_episode(self.runs[run], ep["scene"], ep["id"])
        return ep, rec

    def source(self, key: str, run: str, steps: list[int], zoom: int = 0) -> ClipSource:
        ep = self.data["episodes"][key]
        path = debug_clip_path(self.runs[run], ep["scene"], ep["id"])
        # frame i of the debug video is step i+1 (the caption counts the trajectory)
        return ClipSource(path, run, steps[0] - 1, steps[1] - 1, zoom=zoom)

    # -- what every frame gets: the section chip and the subtitle ----------------------

    def finish_frame(self, im: Image.Image, spec: dict, k: int, chip: bool = True) -> np.ndarray:
        cv = Canvas(im)
        if chip:
            section_chip(cv, spec)
        text = subtitle_at(spec["t_start"] + k / FPS)
        if text:
            subtitle_band(cv, text)
        return to_bgr(cv.finish())

    # -- simulator footage -----------------------------------------------------------

    def _decorate(self, spec: dict, rec: dict, step: int, rgb: np.ndarray, m: np.ndarray,
                  scale: float = 1.0, label: str | None = "MD-SG") -> Image.Image:
        """One output frame: RGB base, map inset, badge, captions, events, insets.

        Positions follow the base image's own size, so a composite can decorate a cropped
        panel; `scale` grows the type for panels that are shrunk afterwards.
        """
        cv = Canvas(to_pil(rgb))
        W, H = cv.base.size
        f_badge, f_cap, f_small = font(int(24 * scale), "bold"), font(int(26 * scale)), font(int(19 * scale))
        target = spec.get("target") or rec["target"]
        state = STATE_LABEL.get(_lookup(rec.get("state_log", []), step, "explore"), "")
        floor = _lookup(spec.get("floors", []), step, "")
        head = f"{label} · " if label else ""
        cv.pill((16, 14), f"{head}target: {target} · step {step}", f_badge, bg=INK, alpha=200)
        sub = " · ".join(x for x in (floor, state) if x)
        if sub:
            cv.pill((16, 14 + int(46 * scale)), sub, f_small, fg=PAPER, bg=INK, alpha=170)
        # map inset, bottom-right by default, above the caption band
        mi = to_pil(m)
        if scale != 1.0:
            mi = mi.resize((int(mi.width * scale), int(mi.height * scale)), Image.LANCZOS)
        mx = 16 if spec.get("map_pos") == "bl" else W - mi.width - 16
        my = H - mi.height - int(70 * scale)
        cv.paste(mi, (mx, my), border=MUTED, width=1)
        cv.pill((mx + (0 if spec.get("map_pos") == "bl" else mi.width), my - 6),
                "costmap \u00b7 around the agent" if spec.get("map_zoom") and label else "costmap", f_small,
                bg=INK, alpha=150, pad=5, anchor="lb" if spec.get("map_pos") == "bl" else "rb")
        # belief bar
        belief = spec.get("belief")
        if belief:
            p = _lookup(belief, step)
            if p is not None:  # a null entry hides the bar: no logged value, nothing shown
                col = GREEN if p >= 0.5 else (ACCENT if p >= 0.3 else RED)
                label_ = f"presence belief p = {p:.2f}"
                bw = max(int(220 * scale), cv.draw.textbbox((0, 0), label_, font=f_small)[2])
                x0, y0 = 26, 14 + int(46 * scale) + (int(40 * scale) if sub else 0)
                cv.draw.rounded_rectangle((x0 - 10, y0, x0 + bw + 10, y0 + int(52 * scale)), radius=8,
                                          fill=(*INK, 200))
                cv.draw.text((x0, y0 + 6), label_, font=f_small, fill=(*PAPER, 255))
                cv.draw.rectangle((x0, y0 + int(32 * scale), x0 + bw, y0 + int(44 * scale)),
                                  fill=(60, 60, 60, 255))
                cv.draw.rectangle((x0, y0 + int(32 * scale), x0 + int(bw * p), y0 + int(44 * scale)),
                                  fill=(*col, 255))
        # picture insets (verifier image, scene-graph tree)
        for ins in spec.get("insets", []):
            if ins["from"] <= step <= ins["to"]:
                im = self._inset_image(ins)
                x, y = ins["box"][0], ins["box"][1]
                cv.paste(im, (x, y), border=PAPER, width=2)
                if ins.get("label"):
                    cv.pill((x + im.width // 2, y + im.height + 8), ins["label"], f_small, bg=INK,
                            alpha=200, anchor="ct")
        # event flash: accent pill under the badge (and the belief bar) for `hold` steps
        ev_y = int(110 * scale) + (int(60 * scale) if belief and _lookup(belief, step) is not None else 0)
        for ev in spec.get("events", []):
            if ev[0] <= step < ev[0] + ev[2]:
                cv.pill((16, ev_y), ev[1], f_badge, fg=INK, bg=ACCENT, alpha=230)
        return cv.finish()

    _inset_cache: dict[str, Image.Image] = {}

    def _inset_image(self, ins: dict) -> Image.Image:
        key = json.dumps(ins, sort_keys=True)
        if key not in self._inset_cache:
            w, h = ins["box"][2], ins["box"][3]
            if ins.get("draw") == "scene_graph":
                self._inset_cache[key] = draw_scene_graph(w, h)
            else:
                im = Image.open(WORKSPACE / ins["image"])
                self._inset_cache[key] = letterbox(im, (w, h), bg=INK, crop=ins.get("crop"))
        return self._inset_cache[key]

    def render_clip(self, spec: dict) -> Iterator[np.ndarray]:
        ep, rec = self.episode(spec["episode"], spec["run"])
        src = self.source(spec["episode"], spec["run"], spec["steps"], zoom=int(spec.get("map_zoom", 0)))
        a, b = spec["steps"]
        n_out = int(round(spec["dur"] * FPS))
        n_hold = int(round(spec.get("hold", 0) * FPS))
        sps = (b - a + 1) / ((n_out - n_hold) / FPS)
        spec = {**spec, "target": ep["target"]}
        for k in range(n_out):
            step = min(b, a + int((k / FPS) * sps))
            rgb, m = src.at(step - 1)
            yield self.finish_frame(self._decorate(spec, rec, step, rgb, m), spec, k)
        src.close()

    def render_composite(self, spec: dict) -> Iterator[np.ndarray]:
        sides = []
        for side in (spec["left"], spec["right"]):
            ep, rec = self.episode(side["episode"], side["run"])
            src = self.source(side["episode"], side["run"], side["steps"], zoom=int(side.get("map_zoom", 0)))
            sides.append((side, {**side, "target": ep["target"]}, rec, src))
        n_out = int(round(spec["dur"] * FPS))
        f_cap, f_end = font(26), font(30, "bold")
        for k in range(n_out):
            canvas = Image.new("RGB", (W, H), INK)
            for i, (side, sspec, rec, src) in enumerate(sides):
                a, b = side["steps"]
                step = min(b, a + int((k / FPS) * side["sps"]))
                rgb, m = src.at(step - 1)
                # centre 4:3 of the 16:9 panel: two of them fill the width at 640x480
                im = self._decorate(sspec, rec, step, rgb[:, 160:1120], m, scale=1.5, label=None)
                if step >= b:  # frozen: this side has ended
                    cv = Canvas(im)
                    col = GREEN if side.get("end_color") == "green" else RED
                    cv.pill((im.width // 2, im.height // 2 - 60), side["end"],
                            fit_font(cv, side["end"], im.width - 100, sizes=(46, 42, 38, 34, 30)),
                            fg=PAPER, bg=col, alpha=225, pad=20, anchor="ct")
                    im = cv.finish()
                im = im.resize((W // 2, 480), Image.LANCZOS)
                canvas.paste(im, (i * (W // 2), 90))
            cv = Canvas(canvas)
            for i, (side, *_) in enumerate(sides):  # method labels in the band above each panel
                cv.pill((i * (W // 2) + 16, 42), side["label"], font(26, "bold"), fg=INK,
                        bg=ACCENT if i == 0 else BLUE, alpha=255, anchor="lt")
            yield self.finish_frame(cv.finish(), spec, k)
        for *_, src in sides:
            src.close()

    # -- the real robot: human camera beside RViz ----------------------------------------

    def render_dual(self, spec: dict) -> Iterator[np.ndarray]:
        """Phone footage on the left, the RViz map window on the right, the RViz camera
        panel (robot view + detections) as a portrait inset; both sources advance at the
        same speed from `phone_t0` / `rviz_t0`, so the two views stay in step."""
        layout = spec.get("rviz_layout", "A")   # one key, or [[t_rel, key], ...] for a take
        layouts = [[0, layout]] if isinstance(layout, str) else layout
        speed = float(spec.get("speed", 1.0))
        n_out = int(round(spec["dur"] * FPS))
        n_hold = int(round(spec.get("hold", 0) * FPS))
        span = spec["phone_t"][1] - spec["phone_t"][0]
        phone = SeqReader(WORKSPACE / spec["phone"], spec["phone_t"][0], span, speed)
        rviz = SeqReader(WORKSPACE / spec["rviz"], spec["rviz_t0"], span, speed)
        f_title, f_lab, f_cap, f_ev = font(26, "bold"), font(20), font(26), font(24, "bold")
        PH, PW = 439, 780
        MW = 460
        try:
            yield from self._dual_frames(spec, phone, rviz, layouts, speed, n_out, n_hold, span,
                                         f_title, f_lab, f_cap, f_ev, PH, PW, MW)
        finally:  # also on an early exit (the --frame preview): no dangling ffmpeg
            phone.close()
            rviz.close()

    def _dual_frames(self, spec, phone, rviz, layouts, speed, n_out, n_hold, span,
                     f_title, f_lab, f_cap, f_ev, PH, PW, MW) -> Iterator[np.ndarray]:
        for k in range(n_out):
            t_rel = min(span, (min(k, n_out - n_hold) / FPS) * speed)
            crops = RVIZ_CROPS[_lookup(layouts, t_rel, "A")]
            if k < n_out - n_hold:  # the hold freezes both views on their last frame
                pf, rf = phone.next(), rviz.next()
            canvas = Image.new("RGB", (W, H), INK)
            cv = Canvas(canvas)
            # phone, 16:9
            pim = _fit(pf, (PW, PH))
            cv.paste(to_pil(pim), (16, 100), border=MUTED, width=1)
            # RViz map window
            x0, y0, x1, y1 = crops["map"]
            mim = _fit(rf[y0:y1, x0:x1], (MW, PH))
            mx = 16 + PW + 8 + (MW - mim.shape[1]) // 2
            cv.paste(to_pil(mim), (mx, 100), border=MUTED, width=1)
            # RViz camera panel as a portrait inset over the phone's corner
            if crops["cam"] is not None and spec.get("cam_inset", True):
                x0, y0, x1, y1 = crops["cam"]
                cim = _fit(rf[y0:y1, x0:x1], (132, 236))
                cx, cy = 16 + PW - cim.shape[1] - 8, 100 + PH - cim.shape[0] - 8
                cv.paste(to_pil(cim), (cx, cy), border=PAPER, width=2)
                cv.pill((cx + cim.shape[1] // 2, cy - 4), "robot camera", font(15), bg=INK,
                        alpha=200, pad=4, anchor="cb")
            cv.pill((16, 14), spec["title"], f_title, fg=INK, bg=ACCENT, alpha=255)
            section_chip(cv, spec)
            if speed != 1.0:
                cv.pill((W - 16, 100 + PH + 10), f"\u25b6 {speed:g}\u00d7 speed", font(22, "bold"), fg=PAPER,
                        bg=INK, alpha=220, anchor="rt")
            cv.pill((16, 96), "human camera", f_lab, bg=INK, alpha=200, pad=6, anchor="lb")
            cv.pill((16 + PW + 8, 96), "RViz on the Thor \u00b7 graph, costmap, goal", f_lab,
                    bg=INK, alpha=200, pad=6, anchor="lb")
            for ev in spec.get("events", []):  # [t_rel_from, t_rel_to, text]
                if ev[0] <= t_rel <= ev[1]:
                    cv.pill((W // 2, 100 + PH + 22), ev[2], f_ev, fg=INK, bg=ACCENT, alpha=235, anchor="ct")
            yield self.finish_frame(cv.finish(), spec, k, chip=False)

    # -- stills and cards ----------------------------------------------------------------

    def render_still(self, spec: dict) -> Iterator[np.ndarray]:
        im = letterbox(Image.open(WORKSPACE / spec["src"]), (W, H - 96), bg=INK, crop=spec.get("crop"))
        canvas = Image.new("RGB", (W, H), INK)
        canvas.paste(im, (0, 0))
        cv = Canvas(canvas)
        for lab in spec.get("labels", []):  # [x_rel, y_rel, text, anchor] on the canvas
            cv.pill((int(lab[0] * W), int(lab[1] * H)), lab[2], font(22, "bold"), fg=INK, bg=ACCENT,
                    alpha=235, anchor=lab[3] if len(lab) > 3 else "lt")
        base = cv.finish()
        for k in range(int(round(spec["dur"] * FPS))):
            yield self.finish_frame(base, spec, k)

    def _bg_frame(self, bg: dict | None) -> Image.Image:
        if not bg:
            return Image.new("RGB", (W, H), INK)
        if "video" in bg:  # a frame of an ordinary video, e.g. the phone footage
            im = letterbox(to_pil(video_frame(WORKSPACE / bg["video"], bg["t"])), (W, H), bg=INK)
        else:
            src = self.source(bg["episode"], "our", [bg["step"], bg["step"]])
            rgb, _ = src.at(bg["step"] - 1)
            src.close()
            im = to_pil(rgb)
        if bg.get("blur"):
            im = im.filter(ImageFilter.GaussianBlur(bg["blur"]))
        return Image.blend(im, Image.new("RGB", (W, H), INK), 0.55)

    def render_slate(self, spec: dict) -> Iterator[np.ndarray]:
        card = getattr(self, f"card_{spec['card']}")
        base = card(spec)
        for k in range(int(round(spec["dur"] * FPS))):
            yield self.finish_frame(base, spec, k)

    def card_title(self, spec: dict) -> Image.Image:
        cv = Canvas(self._bg_frame(spec.get("bg")))
        d = cv.draw
        d.text((80, 200), self.meta["title"], font=font(110, "bold"), fill=(*PAPER, 255))
        y = paragraph(d, (84, 340), self.meta["subtitle"], font(34), 1100, fill=(*PAPER, 255))
        d.text((84, y + 30), self.meta["venue"], font=font(26), fill=(*MUTED, 255))
        if self.meta.get("authors"):
            d.text((84, y + 70), self.meta["authors"], font=font(26), fill=(*PAPER, 255))
        return cv.finish()

    SECTIONS = ["The problem", "Our system", "Three modules: scene graph \u00b7 presence belief \u00b7 belief-guided search",
                "Experiments and comparison", "A real robot", "Conclusion"]

    def card_outline(self, spec: dict) -> Image.Image:
        cv = Canvas(Image.new("RGB", (W, H), INK))
        d = cv.draw
        d.text((80, 50), "In this video", font=font(34, "bold"), fill=(*ACCENT, 255))
        y = 130
        for i, t in enumerate(self.SECTIONS, 1):
            d.ellipse((80, y + 2, 122, y + 44), fill=(*ACCENT, 255))
            n = str(i)
            tw = d.textbbox((0, 0), n, font=font(26, "bold"))[2]
            d.text((101 - tw // 2, y + 6), n, font=font(26, "bold"), fill=(*INK, 255))
            d.text((146, y + 4), t, font=font(30), fill=(*PAPER, 255))
            y += 74
        return cv.finish()

    def card_system(self, spec: dict) -> Image.Image:
        canvas = Image.new("RGB", (W, H), INK)
        fig = letterbox(Image.open(WORKSPACE / "docs/figures/full_pipeline.png"), (W - 80, 330), bg=INK)
        canvas.paste(fig, (40, 60))
        cv = Canvas(canvas)
        d = cv.draw
        d.text((40, 14), "One persistent, floor-aware scene graph  \u00b7  three mechanisms", font=font(28, "bold"),
               fill=(*ACCENT, 255))
        mods = [("Scene graph", "open-vocabulary detections \u2192 ellipsoid tracks \u2192 floor \u00b7 room \u00b7 container \u00b7 object; stair edges only after a traversal"),
                ("Presence-belief revision", "a visible miss lowers belief, an occluded view does not; a low ceiling lets history be overturned"),
                ("Belief-guided search", "every surface carries a posterior; failed looks spend it; the floor score decides when to switch storey")]
        colw = (W - 80 - 2 * 24) // 3
        for i, (head, body) in enumerate(mods):
            x = 40 + i * (colw + 24)
            d.rounded_rectangle((x, 410, x + colw, 610), radius=12, outline=(*MUTED, 255), width=2)
            d.ellipse((x + 16, 424, x + 52, 460), fill=(*ACCENT, 255))
            n = str(i + 1)
            tw = d.textbbox((0, 0), n, font=font(24, "bold"))[2]
            d.text((x + 34 - tw // 2, 429), n, font=font(24, "bold"), fill=(*INK, 255))
            d.text((x + 64, 428), head, font=font(24, "bold"), fill=(*PAPER, 255))
            paragraph(d, (x + 16, 476), body, font(19), colw - 32, fill=(*MUTED, 255), spacing=5)
        return cv.finish()

    def card_eq3(self, spec: dict) -> Image.Image:
        cv = Canvas(Image.new("RGB", (W, H), INK))
        d = cv.draw
        d.text((60, 50), "Presence belief  (Eq. 3)", font=font(38, "bold"), fill=(*ACCENT, 255))
        d.text((60, 105), "log-odds L per mapped object,  p = σ(L),  clipped to [−6, +3]",
               font=font(26), fill=(*MUTED, 255))
        rows = [
            ("detected, visible", "ΔL = log( r / q )", "positive", PAPER, False),
            ("missed, visible", "ΔL = log( (1−r) / (1−q) )", "negative — the channel a static map lacks", PAPER, True),
            ("not visible", "ΔL = 0", "unobserved is not observed-absent", MUTED, False),
        ]
        y = 190
        for a, b, c, col, hi in rows:
            if hi:
                d.rounded_rectangle((44, y - 14, W - 44, y + 66), radius=10, fill=(*ACCENT, 60))
            d.text((60, y), a, font=font(30, "bold"), fill=(*col, 255))
            d.text((420, y), b, font=font(30, mono=True), fill=(*col, 255))
            d.text((60, y + 36), c, font=font(22), fill=(*(ACCENT if hi else MUTED), 255))
            y += 110
        paragraph(d, (60, 540),
                  "visible = a large, readable projection in range with nothing nearer in the depth map. "
                  "Nearer depth means occlusion and no update; the low +3 ceiling is what lets a few honest misses overturn a long history.",
                  font(22), W - 120, fill=(*PAPER, 255), spacing=6)
        return cv.finish()

    def _table_card(self, spec: dict, title: str, rows: list[tuple], header: tuple,
                    note: str, y0: int = 150, ours_idx: int | None = None) -> Canvas:
        cv = Canvas(self._bg_frame(spec.get("bg")))
        d = cv.draw
        d.text((60, 50), title, font=fit_font(cv, title, W - 120), fill=(*ACCENT, 255))
        cols = [60, 520, 760, 1000]
        for x, h_ in zip(cols, header):
            d.text((x, y0), h_, font=font(22), fill=(*MUTED, 255))
        y = y0 + 44
        for i, row in enumerate(rows):
            ours = i == ours_idx
            fnt = font(34, "bold" if ours else "regular")
            col = ACCENT if ours else PAPER
            for x, cell in zip(cols, row):
                d.text((x, y), str(cell), font=fnt, fill=(*col, 255))
            y += 60
        if note:
            paragraph(d, (60, H - 80), note, font(20), W - 120, fill=(*MUTED, 255))
        return cv

    def card_table2(self, spec: dict) -> Image.Image:
        cv = self._table_card(
            spec, "Multi-floor cross-anchor relocation · HM3D v0.2 · 5 scenes · 25 episodes  (Table II)",
            [("ASCENT", "25", "0.0 %", "0.000"), ("MD-SG (ours)", "25", "40.0 %", "0.162")],
            ("method", "episodes", "success", "SPL"),
            "1000-step budget. Every episode starts from a map built on the static layout; the target has been moved to another storey.",
            y0=200, ours_idx=1)
        return cv.finish()

    def card_table1(self, spec: dict) -> Image.Image:
        cv = self._table_card(
            spec, "DualMap's benchmark · single floor · cross-anchor · 53 episodes  (Table I)",
            [("DualMap (re-evaluated)", "", "30.2 %", ""), ("MD-SG (ours)", "", "52.8 %", "")],
            ("method", "", "success", ""),
            "", y0=120, ours_idx=1)
        d = cv.draw
        d.text((60, 330), "Ablation, same 53 episodes  (Table III)", font=font(26, "bold"), fill=(*ACCENT, 255))
        bars = [("w/o presence-belief revision", 41.5), ("w/o belief-guided surface scoring", 35.8), ("full system", 52.8)]
        y = 380
        for name, v in bars:
            d.text((60, y), name, font=font(24), fill=(*PAPER, 255))
            d.rectangle((520, y + 4, 520 + int(v * 9), y + 30), fill=(*(ACCENT if v == 52.8 else MUTED), 255))
            d.text((530 + int(v * 9), y), f"{v:.1f} %", font=font(24, "bold"), fill=(*PAPER, 255))
            y += 56
        paragraph(d, (60, H - 80), "All seven episodes lost without presence revision commit to a stale prior-map track that never decays.",
                  font(20), W - 120, fill=(*MUTED, 255))
        return cv.finish()

    def card_floor_score(self, spec: dict) -> Image.Image:
        """The floor score as the agent inspects and fails: computed by the paper renderer's
        own functions (`scripts/render_floor_score_formula.py`) on the saved 00873 map,
        cached as JSON so the cut does not need the map to rebuild."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cache = ASSETS / "floor_score_curve.json"
        if not cache.exists():
            _compute_floor_score(cache)
        d = json.loads(cache.read_text())
        means = np.asarray(d["means"])
        here, other = means[:, 0], means[:, 1]
        margin = float(d["margin"])
        k = np.arange(len(means))
        accept = int(np.nonzero(other >= margin * here)[0][0])
        n = 15
        c_here, c_other = "#e87830", "#5cc08c"
        ink = "#%02x%02x%02x" % INK
        pap = "#%02x%02x%02x" % PAPER
        mut = "#%02x%02x%02x" % MUTED

        fig = plt.figure(figsize=(12.8, 6.3), dpi=100, facecolor=ink)
        ax = fig.add_axes([0.09, 0.15, 0.86, 0.66], facecolor=ink)
        ax.axvspan(accept, n, color=c_other, alpha=0.12, lw=0)
        ax.plot(k[:n + 1], here[:n + 1], color=c_here, lw=5)
        ax.plot(k[:n + 1], other[:n + 1], color=c_other, lw=5)
        ax.text(0.3, here[0] + 0.004, "floor 0 \u00b7 where the map said the object is", color=c_here, fontsize=18, fontweight="bold")
        ax.text(0.3, other[0] + 0.004, "floor 1", color=c_other, fontsize=18, fontweight="bold")
        ax.plot([accept], [here[accept]], "*", color=pap, ms=26, zorder=5)
        ax.annotate(f"after {accept} failed looks:\nswitch to floor 1", (accept, here[accept]),
                    xytext=(accept + 0.7, here[accept] + 0.028), color=pap, fontsize=19, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=pap, lw=1.5))
        ax.set_xlim(0, n)
        ax.set_ylim(0, 0.12)
        ax.set_yticks([])
        ax.set_xticks(range(0, n + 1, 2))
        ax.set_xlabel("failed looks on floor 0", color=pap, fontsize=19)
        ax.set_ylabel("floor score", color=pap, fontsize=19)
        ax.tick_params(colors=mut, labelsize=15)
        for sp in ax.spines.values():
            sp.set_color(mut)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.text(0.09, 0.91, "When does the agent leave a floor?", color="#%02x%02x%02x" % ACCENT,
                 fontsize=26, fontweight="bold")
        fig.text(0.09, 0.845, "every failed look lowers that floor\u2019s score; when the other floor is ahead by a margin, it switches",
                 color=pap, fontsize=16)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        canvas = Image.new("RGB", (W, H), INK)
        canvas.paste(Image.fromarray(buf.copy()), (0, 0))
        return canvas

    def card_topology(self, spec: dict) -> Image.Image:
        cv = Canvas(Image.new("RGB", (W, H), INK))
        d = cv.draw
        d.text((60, 40), "On the robot", font=font(34, "bold"), fill=(*ACCENT, 255))
        boxes = [
            ((60, 130, 560, 330), "osg-thor  (Jetson AGX Thor, L4T/CUDA)",
             "the pipeline, the scene graph, the perception models, ollama.  No ROS, no Habitat."),
            ((60, 380, 560, 520), "osg-bridge  (ros:humble)", "rclpy and nothing else; talks to osg-thor over a local socket."),
            ((760, 130, 1220, 520), "Stretch 3", "Nav2, stretch_driver, RealSense, TF.  Keeps its own low-level control."),
        ]
        for (x0, y0, x1, y1), head, body in boxes:
            d.rounded_rectangle((x0, y0, x1, y1), radius=14, outline=(*PAPER, 255), width=2)
            d.text((x0 + 20, y0 + 18), head, font=font(24, "bold"), fill=(*PAPER, 255))
            paragraph(d, (x0 + 20, y0 + 60), body, font(21), x1 - x0 - 40, fill=(*MUTED, 255))
        d.line((310, 330, 310, 380), fill=(*ACCENT, 255), width=4)
        d.text((325, 340), "socket", font=font(19), fill=(*ACCENT, 255))
        d.line((560, 450, 760, 450), fill=(*ACCENT, 255), width=4)
        d.text((620, 415), "DDS", font=font(19), fill=(*ACCENT, 255))
        d.text((580, 250), "goal pose → NavigateToPose", font=font(19), fill=(*PAPER, 255))
        d.text((580, 280), "← RGB-D, TF", font=font(19), fill=(*PAPER, 255))
        paragraph(d, (60, 570),
                  "The pipeline keeps doing all of the thinking and stops steering. Nav2's map is 2-D, so the storey is declared by the operator on /osg/floor.",
                  font(24), W - 120, fill=(*PAPER, 255))
        return cv.finish()

    def card_close(self, spec: dict) -> Image.Image:
        cv = Canvas(self._bg_frame(spec.get("bg")))
        d = cv.draw
        d.text((80, 60), "Conclusion", font=font(34, "bold"), fill=(*ACCENT, 255))
        lines = ["A map should hold beliefs, not facts: presence revision lets a stale map be wrong and recover.",
                 "A search should know when a floor is spent: belief-guided search decides when to change storey.",
                 "Both run on a real robot today, on a Stretch 3 with a Jetson Thor."]
        y = 130
        for i, t in enumerate(lines, 1):
            d.ellipse((80, y + 4, 118, y + 42), fill=(*ACCENT, 255))
            n = str(i)
            tw = d.textbbox((0, 0), n, font=font(24, "bold"))[2]
            d.text((99 - tw // 2, y + 9), n, font=font(24, "bold"), fill=(*INK, 255))
            y = paragraph(d, (140, y + 4), t, font(28, "bold"), W - 220, fill=(*PAPER, 255), spacing=6) + 28
        d.text((80, H - 130), f"{self.meta['title']} \u00b7 {self.meta['venue']}", font=font(26), fill=(*MUTED, 255))
        if self.meta.get("code_url"):
            d.text((80, H - 96), self.meta["code_url"], font=font(26), fill=(*ACCENT, 255))
        return cv.finish()

    # -- driver --------------------------------------------------------------------------

    def frames(self, spec: dict) -> Iterator[np.ndarray]:
        return getattr(self, f"render_{spec['kind']}")(spec)

    def render(self, spec: dict) -> pathlib.Path:
        out = SEGMENTS / f"{spec['id']}.mp4"
        n = int(round(spec["dur"] * FPS))
        w = SegmentWriter(out)
        for f in fade_edges(self.frames(spec), n):
            w.write(f)
        w.close()
        if w.n != n:
            raise RuntimeError(f"{spec['id']}: wrote {w.n} frames, expected {n}")
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="shot ids to render (default: all)")
    ap.add_argument("--frame", nargs=2, metavar=("SHOT", "T"), help="write one PNG at T seconds into SHOT")
    args = ap.parse_args()

    data = load_shotlist()
    r = Renderer(data)
    by_id = {s["id"]: s for s in data["shots"]}
    if args.frame:
        spec, t = by_id[args.frame[0]], float(args.frame[1])
        k = int(t * FPS)
        for i, f in enumerate(r.frames(spec)):
            if i == k:
                ASSETS.mkdir(parents=True, exist_ok=True)
                out = ASSETS / f"frame_{spec['id']}_{t:.1f}s.png"
                cv2.imwrite(str(out), f)
                print(out)
                break
        return

    SEGMENTS.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for spec in data["shots"]:
        if args.only and spec["id"] not in args.only:
            continue
        out = r.render(spec)
        manifest[spec["id"]] = {"segment": str(out.relative_to(WORKSPACE)), "dur": spec["dur"],
                                "kind": spec["kind"], "t_start": spec["t_start"]}
        print(f"{spec['t_start']:6.1f}s  {spec['id']:<16} {spec['dur']:>4}s  -> {out.name}")
    with (SEGMENTS / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"total {data['total']:.1f} s of {data['meta']['limits']['max_seconds']} s")


if __name__ == "__main__":
    main()
