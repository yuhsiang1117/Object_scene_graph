#!/usr/bin/env python3
"""Synthesise the voiceover from `narration.json` and write the timed transcript.

    /tmp/.conda/envs/tts/bin/python scripts/video/make_narration.py
        # -> outputs/video/narration/{voice.wav,cues.json,transcript.md}, outputs/video/icra2027.srt
    python scripts/video/make_assets.py               # every frame draws its cue as a subtitle
    python scripts/video/build_video.py --narration outputs/video/narration/voice.wav

Runs in its own interpreter (`conda create -n tts python=3.11`, then
`pip install kokoro soundfile pillow`): Kokoro cannot be installed into the habitat env, so
this file imports neither `common.py` nor cv2.

Each cue is placed at its shot's start (from `shotlist.json`) plus `at`. A cue that would run
into the next cue fails the script, and so does a cue whose text needs more than two
subtitle lines: the voice never drifts off the picture it describes, and the subtitle never
covers it. `cues.json` carries the timings measured from the synthesised audio; the frame
renderer reads it, so subtitles appear exactly when the words do. The WAV is exactly as long
as the cut, so `build_video.py`'s `-shortest` keeps every frame.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np
import soundfile as sf

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parents[1] / "outputs" / "video" / "narration"
SR = 24_000
GAP = 0.25          # minimum silence between two cues, s
SUB_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SUB_SIZE, SUB_WIDTH, SUB_LINES = 26, 1100, 2   # must match make_assets.subtitle_band

# Names the TTS mispronounces, spelled the way they are said. A cue may also carry a
# `display` string -- the subtitle -- when the spoken form differs (digits vs words).
SAY = {"MD-SG": "M D S G", "ASCENT": "Ascent", "DualMap's": "Dual Map's", "DualMap": "Dual Map"}


def spoken(text: str) -> str:
    for k, v in SAY.items():
        text = re.sub(rf"\b{re.escape(k)}(?=\W|$)", v, text)
    return text


def fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def trim_silence(audio: np.ndarray, thresh: float = 0.005, keep_s: float = 0.03) -> np.ndarray:
    """Cut the model's leading and trailing silence so a cue's audio starts at its `t0` --
    the subtitle is drawn from that same `t0`, so voice and text appear together."""
    loud = np.nonzero(np.abs(audio) > thresh)[0]
    if not len(loud):
        return audio
    a = max(0, int(loud[0] - keep_s * SR))
    b = min(len(audio), int(loud[-1] + keep_s * SR))
    return audio[a:b]


def subtitle_lines(text: str) -> int:
    """How many lines the renderer will need for this cue, with its font and width."""
    from PIL import ImageFont
    fnt = ImageFont.truetype(SUB_FONT, SUB_SIZE)
    lines, cur = 0, ""
    for w in text.split():
        trial = (cur + " " + w).strip()
        if fnt.getlength(trial) <= SUB_WIDTH or not cur:
            cur = trial
        else:
            lines += 1
            cur = w
    return lines + (1 if cur else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    shots = json.loads((HERE / "shotlist.json").read_text())["shots"]
    starts, t = {}, 0.0
    for s in shots:
        starts[s["id"]] = (t, s)
        t += float(s["dur"])
    total = t
    nar = json.loads((HERE / "narration.json").read_text())

    from kokoro import KPipeline  # heavy; after the cheap checks
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

    cues = nar["cues"]
    long = [c for c in cues if subtitle_lines(c.get("display", c["text"])) > SUB_LINES]
    if long:
        sys.exit("subtitles over two lines:\n  " + "\n  ".join(f"{c['shot']}: {c['text']!r}" for c in long))
    track = np.zeros(int(round(total * SR)), np.float32)
    rows, errors = [], []
    for i, c in enumerate(cues):
        t0 = starts[c["shot"]][0] + c["at"]
        limit = (starts[cues[i + 1]["shot"]][0] + cues[i + 1]["at"] - GAP) if i + 1 < len(cues) else total - 0.2
        audio = np.concatenate([np.asarray(r[2], np.float32)
                                for r in pipe(spoken(c["text"]), voice=nar["voice"], speed=nar["speed"])])
        audio = trim_silence(audio)               # Kokoro pads ~0.3 s of silence at both ends
        t1 = t0 + len(audio) / SR
        if t1 > limit:
            errors.append(f"{c['shot']} @{fmt(t0)}: {t1 - t0:.2f} s of speech, {limit - t0:.2f} s free "
                          f"(over by {t1 - limit:.2f} s): {c['text']!r}")
        a = int(round(t0 * SR))
        track[a:a + len(audio)] += audio[:len(track) - a]
        rows.append((t0, t1, c["shot"], c.get("display", c["text"])))
        print(f"{fmt(t0)}–{fmt(t1)}  {c['shot']:<16} {c['text']}")

    if errors:
        sys.exit("cues overrun the next one:\n  " + "\n  ".join(errors))

    peak = float(np.abs(track).max())
    track *= 0.89 / peak                     # ≈ −1 dBFS
    args.out.mkdir(parents=True, exist_ok=True)
    sf.write(args.out / "voice.wav", track, SR, subtype="PCM_16")

    with (args.out / "cues.json").open("w") as fh:
        json.dump([{"t0": round(a, 3), "t1": round(b, 3), "shot": s_, "text": txt} for a, b, s_, txt in rows], fh, indent=1)
    srt = []
    for i, (a, b, _, txt) in enumerate(rows, 1):
        srt += [str(i), f"{srt_time(a)} --> {srt_time(b + 0.4)}", txt, ""]
    (args.out.parent / "icra2027.srt").write_text("\n".join(srt))
    speech = sum(b - a for a, b, *_ in rows)
    words = sum(len(r[3].split()) for r in rows)
    md = ["# Voiceover transcript", "",
          f"Kokoro-82M, voice `{nar['voice']}`, speed {nar['speed']}. {words} words, "
          f"{speech:.0f} s of speech in a {total:.0f} s cut. Generated from `scripts/video/narration.json`.", "",
          "| time | shot | narration |", "|---|---|---|"]
    md += [f"| {fmt(a)}–{fmt(b)} | `{s}` | {txt} |" for a, b, s, txt in rows]
    (args.out / "transcript.md").write_text("\n".join(md) + "\n")
    print(f"{args.out / 'voice.wav'}: {total:.1f} s, {words} words, {speech:.0f} s of speech")


if __name__ == "__main__":
    main()
