#!/usr/bin/env python3
"""Assemble the ICRA video from the rendered segments, under the attachment limits.

    python scripts/video/build_video.py                         # -> outputs/video/icra2027_rough.mp4
    python scripts/video/build_video.py --narration voice.wav   # mux a recorded voiceover
    python scripts/video/build_video.py --print-storyboard      # the table for docs/VIDEO.md

Size: ICRA's limit is 20 MB for at most 3 minutes. The video bitrate is derived from the
total duration so a two-pass H.264 encode lands at ~18.5 MB with the audio, and the result
is checked against both limits before the script reports success.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import FPS, OUT_ROOT, SEGMENTS, load_shotlist, probe  # noqa: E402

TARGET_BYTES = 18_500_000     # headroom under the 20 MB limit for container overhead
AUDIO_KBPS = 64


def storyboard(data: dict) -> str:
    here = pathlib.Path(__file__).resolve().parent
    nar = json.loads((here / "narration.json").read_text())["cues"]
    rows = ["| t | section | shot | beat | on screen | narration |", "|---|---|---|---|---|---|"]
    for s in data["shots"]:
        t0 = s["t_start"]
        lines = " ".join(c.get("display", c["text"]) for c in nar if c["shot"] == s["id"])
        rows.append(f"| {int(t0 // 60)}:{int(t0 % 60):02d} (+{s['dur']} s) | {s.get('section', '')} | `{s['id']}` | "
                    f"{s['beat']} | {s.get('on_screen', '')} | {lines} |")
    rows.append(f"\nTotal: {data['total']:.0f} s of {data['meta']['limits']['max_seconds']} s.")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--narration", type=pathlib.Path, help="voiceover audio file to mux (wav/m4a/mp3)")
    ap.add_argument("--out", type=pathlib.Path, default=OUT_ROOT / "icra2027_rough.mp4")
    ap.add_argument("--print-storyboard", action="store_true")
    args = ap.parse_args()

    data = load_shotlist()
    if args.print_storyboard:
        print(storyboard(data))
        return

    limits = data["meta"]["limits"]
    total = data["total"]
    here = pathlib.Path(__file__).resolve().parent
    cues = OUT_ROOT / "narration" / "cues.json"
    if not cues.exists():
        sys.exit("no cues.json: run make_narration.py (the subtitles are drawn from its timings)")
    stale = [f.name for f in (here / "narration.json", here / "shotlist.json") if f.stat().st_mtime > cues.stat().st_mtime]
    if stale:
        sys.exit(f"{', '.join(stale)} changed after cues.json: re-run make_narration.py, then make_assets.py")
    rows = json.loads(cues.read_text())
    print(f"narration: {len(rows)} cues, {sum(len(r['text'].split()) for r in rows)} words, "
          f"{sum(r['t1'] - r['t0'] for r in rows):.0f} s of speech")
    if total > limits["max_seconds"]:
        sys.exit(f"shot list runs {total:.1f} s, over the {limits['max_seconds']} s limit")

    parts = []
    for s in data["shots"]:
        seg = SEGMENTS / f"{s['id']}.mp4"
        if not seg.exists():
            sys.exit(f"missing segment {seg}; run scripts/video/make_assets.py first")
        parts.append(seg)

    concat = SEGMENTS / "concat.txt"
    concat.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))

    v_kbps = min(1000, int((TARGET_BYTES * 8 / 1000 - AUDIO_KBPS * total) / total))
    audio_in = (["-i", str(args.narration)] if args.narration
                else ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono"])
    common = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
              *audio_in, "-map", "0:v:0", "-map", "1:a:0", "-shortest",
              "-c:v", "libx264", "-preset", "slow", "-profile:v", "high", "-pix_fmt", "yuv420p",
              "-r", str(FPS), "-b:v", f"{v_kbps}k", "-maxrate", f"{int(v_kbps * 1.4)}k",
              "-bufsize", f"{v_kbps * 2}k", "-movflags", "+faststart"]
    log = str(SEGMENTS / "x264pass")
    subprocess.run([*common, "-pass", "1", "-passlogfile", log, "-an", "-f", "mp4", "/dev/null"], check=True)
    subprocess.run([*common, "-pass", "2", "-passlogfile", log, "-c:a", "aac", "-b:a", f"{AUDIO_KBPS}k",
                    "-ac", "1", str(args.out)], check=True)

    info = probe(args.out)
    size, dur = int(info["format"]["size"]), float(info["format"]["duration"])
    st = info["streams"][0]
    print(f"{args.out}: {st['width']}x{st['height']} {st['codec_name']} {st['r_frame_rate']} fps, "
          f"{dur:.1f} s, {size / 1e6:.2f} MB  (video {v_kbps} kbps)")
    ok = size <= limits["max_bytes"] and dur <= limits["max_seconds"] + 0.5
    print("within limits" if ok else "OVER LIMIT")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
