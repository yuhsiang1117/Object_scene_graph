"""Stitch an episode's keyframe segmentation frames (from run_episode_report.py)
into a video, with a per-frame banner (keyframe #, step, target-seen), so the
exploration + perception process can be watched in time order.

Usage:
  python scripts/keyframes_to_video.py --episode 1:tv_monitor          # one episode
  python scripts/keyframes_to_video.py --dir outputs/report/ep1_tv_monitor
  python scripts/keyframes_to_video.py --all --fps 6                    # every ep in outputs/report
Output: <episode-dir>/exploration.mp4
"""
from __future__ import annotations
import argparse
import glob
import re
from pathlib import Path

import cv2


def _parse_summary(ep_dir: Path):
    """Return (header_line, {kf_int: 'seen' or 'none'})."""
    seen = {}
    header = ep_dir.name
    sf = ep_dir / "summary.txt"
    if sf.exists():
        lines = sf.read_text().splitlines()
        if lines:
            header = lines[0]
        for ln in lines[1:]:
            m = re.match(r"kf\s+(\d+)\s+step\s+(\d+).*target_seen\s+(\d+)", ln)
            if m:
                seen[int(m.group(1))] = int(m.group(3)) > 0
    return header, seen


def _banner(img, text, seen):
    h, w = img.shape[:2]
    bar_h = 30
    cv2.rectangle(img, (0, 0), (w, bar_h), (24, 24, 28), -1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    # target-seen indicator dot
    col = (60, 220, 60) if seen else (70, 70, 200)
    cv2.circle(img, (w - 16, 15), 7, col, -1)
    cv2.putText(img, "TGT" if seen else "", (w - 66, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (60, 220, 60), 1, cv2.LINE_AA)
    return img


def make_video(ep_dir: Path, fps: float) -> bool:
    frames = sorted(glob.glob(str(ep_dir / "keyframes" / "kf_*.jpg")))
    if not frames:
        print(f"  {ep_dir.name}: no keyframes, skipped")
        return False
    header, seen = _parse_summary(ep_dir)
    result = "OK" if "result=OK" in header else "fail"
    tag = header.split("summary")[0]  # cheap
    ep_label = ep_dir.name
    first = cv2.imread(frames[0])
    h, w = first.shape[:2]
    out_path = ep_dir / "exploration.mp4"
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        img = cv2.imread(f)
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        m = re.search(r"kf_(\d+)_step(\d+)", Path(f).name)
        kf = int(m.group(1)) if m else -1
        step = int(m.group(2)) if m else -1
        txt = f"{ep_label} [{result}]  kf {kf}  step {step}"
        _banner(img, txt, seen.get(kf, False))
        vw.write(img)
    vw.release()
    print(f"  {ep_dir.name}: {len(frames)} frames -> {out_path} ({len(frames)/fps:.1f}s @ {fps}fps)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="", help="id:target, resolved under outputs/report/")
    ap.add_argument("--dir", default="", help="explicit episode dir")
    ap.add_argument("--root", default="outputs/report", help="report root for --episode/--all")
    ap.add_argument("--all", action="store_true", help="every ep dir under --root")
    ap.add_argument("--fps", type=float, default=5.0)
    args = ap.parse_args()

    dirs = []
    if args.dir:
        dirs = [Path(args.dir)]
    elif args.episode:
        i, t = args.episode.split(":")
        dirs = [Path(args.root) / f"ep{i}_{t}"]
    elif args.all:
        dirs = [Path(p) for p in sorted(glob.glob(str(Path(args.root) / "ep*")))
                if (Path(p) / "keyframes").is_dir()]
    else:
        raise SystemExit("give --episode id:target, --dir <path>, or --all")

    made = 0
    for d in dirs:
        if not d.exists():
            print(f"  {d}: missing")
            continue
        made += make_video(d, args.fps)
    print(f"done: {made} video(s)")


if __name__ == "__main__":
    main()
