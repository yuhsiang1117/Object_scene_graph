#!/usr/bin/env python3
"""A video of the object layer turning depth points into ellipsoids, from a
pass-1 recording (eval.record_object_frames_dir), next to the unchanged layer.

Each keyframe of one recorded episode is fed to two layers in lockstep -- the
variant under `--variant` (default R2M4 of docs/history/AB_RESULTS.md S77) and
the layer as measured (V0) -- exactly as scripts/replay_object_layer.py does,
so what is drawn IS what the replay measured. Per keyframe, three panels:

  left    the camera frame (re-rendered at the recorded pose by the side
          simulator, osg/eval/gt_semantic.py: the recording keeps depth, not
          colour) with every detection's mask outlined in the colour of the
          variant track it joined; grey = no track (proposal / gated out)
  middle  the variant, around where the camera looks: each track's buffered
          voxel points in its colour, this keyframe's cleaned depth points in
          white, the ellipsoids (three principal sections; thick = updated
          now), and the ground-truth boxes of objects seen so far (grey)
  right   V0 over the same window: its ellipsoids, and the same white points
          for reference -- V0 never keeps them

    python scripts/render_ellipsoid_fusion.py --scene 00873-bxsVRursffK \\
        --out outputs/viz_ellipsoid/fusion_00873.mp4
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
sys.path.insert(0, str(WORKSPACE / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R2M4 = ("init=points reinit_truncated=true buffer=true refine_mode=points overlap_mode=voxel "
        "assoc_3d_overlap=0.3 normalize_labels=true merge_split_overlap=0.5")
BASE = ["+experiment=mf5_mapping_both", "eval.rgb_width=1280", "eval.rgb_height=960",
        "detector.imgsz=1280"]
PALETTE = plt.get_cmap("tab20").colors
GREY = (0.62, 0.62, 0.62)


def compose(overrides):
    from hydra import compose as hcompose, initialize_config_dir

    with initialize_config_dir(config_dir=str(WORKSPACE / "configs"), version_base="1.3"):
        return hcompose("config", overrides=list(overrides))


def colour(tid: int):
    return PALETTE[(int(tid) * 7) % len(PALETTE)]


def to_plot(p: np.ndarray) -> np.ndarray:
    """World (x, y up, z) -> plot (x, -z, y): a proper rotation, z up."""
    p = np.asarray(p, float).reshape(-1, 3)
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)


def ellipse_sections(e, n: int = 48):
    t = np.linspace(0.0, 2.0 * math.pi, n)
    c, s = np.cos(t), np.sin(t)
    out = []
    for a, b in ((0, 1), (1, 2), (0, 2)):
        local = np.zeros((n, 3))
        local[:, a] = c * e.axes[a]
        local[:, b] = s * e.axes[b]
        out.append(local @ e.R.T + e.center)
    return out


def box_edges(c, h):
    c, h = np.asarray(c, float), np.asarray(h, float)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    return [corners[[i, j]] for i, j in edges]


def assoc_of(layer, k: int) -> dict:
    """{detection index: track id} for the detections of keyframe k."""
    out = {}
    for t in layer._tracks.values():
        if not t.observations:
            continue
        ref = getattr(t.observations[-1], "det_ref", None)
        if ref is not None and int(ref[0]) == k:
            out[int(ref[1])] = t.id
    return out


def visible_tracks(layer, floor_key, centre, radius):
    out = []
    for t in layer._tracks.values():
        if t.proposal_only or t.blacklisted or int(t.floor_key) != int(floor_key):
            continue
        d = np.asarray(t.ellipsoid.center)[[0, 2]] - centre
        if float(np.hypot(*d)) <= radius:
            out.append(t)
    return out


def draw_3d(ax, title, tracks, updated, new_pts, gt_boxes, cam, frustum, window, floor_y,
            azim, show_buffer):
    ax.cla()
    (cx, cz), r = window
    lo_y, hi_y = floor_y - 0.1, floor_y + 2.6

    def inside(p):
        # mplot3d draws data outside the axis limits too: clip by hand.
        p = np.asarray(p, float).reshape(-1, 3)
        return ((np.abs(p[:, 0] - cx) <= r) & (np.abs(p[:, 2] - cz) <= r)
                & (p[:, 1] >= lo_y) & (p[:, 1] <= hi_y))

    def pts_in(p):
        return p[inside(p)]

    def line_in(p):
        p = np.asarray(p, float).copy()
        p[~inside(p)] = np.nan
        return p
    if show_buffer:
        for t in tracks:
            buf = getattr(t, "buffer", None)
            if buf is None or not len(buf):
                continue
            pts = pts_in(buf.points())
            if not len(pts):
                continue
            if len(pts) > 350:
                pts = pts[np.linspace(0, len(pts) - 1, 350).astype(int)]
            q = to_plot(pts)
            ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=4.0, color=colour(t.id), alpha=0.7,
                       depthshade=False, linewidths=0)
    for pts in new_pts:
        pts = pts_in(pts)
        if not len(pts):
            continue
        q = to_plot(pts)
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=3.0, color=(1.0, 1.0, 1.0) if show_buffer else GREY,
                   alpha=0.9 if show_buffer else 0.5, depthshade=False, linewidths=0,
                   edgecolors="none")
    for c, h in gt_boxes:
        for seg in box_edges(c, h):
            q = to_plot(line_in(seg))
            ax.plot(q[:, 0], q[:, 1], q[:, 2], color=GREY, lw=0.5, alpha=0.6)
    for t in tracks:
        lw = 2.2 if t.id in updated else 0.8
        for sec in ellipse_sections(t.ellipsoid):
            q = to_plot(line_in(sec))
            ax.plot(q[:, 0], q[:, 1], q[:, 2], color=colour(t.id), lw=lw, alpha=0.95)
    q = to_plot(np.vstack([cam] + list(frustum)))
    for i in range(1, 5):
        ax.plot(q[[0, i], 0], q[[0, i], 1], q[[0, i], 2], color="#ffd400", lw=1.2)
    ax.plot(q[[1, 2, 4, 3, 1], 0], q[[1, 2, 4, 3, 1], 1], q[[1, 2, 4, 3, 1], 2], color="#ffd400", lw=1.2)
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(-cz - r, -cz + r)
    ax.set_zlim(lo_y, hi_y)
    ax.set_box_aspect((2 * r, 2 * r, 2.7), zoom=1.2)
    ax.view_init(elev=32, azim=azim)
    ax.set_facecolor("#14161c")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.08, 0.09, 0.11, 1.0))
        axis.set_ticklabels([])
        axis.line.set_color((0.3, 0.3, 0.3))
    ax.grid(False)
    ax.set_title(title, color="white", fontsize=12, pad=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", default=str(WORKSPACE / "outputs/objrec_v2"))
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", type=int, default=0, help="index in the scene's recorded chain")
    ap.add_argument("--variant", default=R2M4, help="scene_graph.ellipsoid overrides, space-separated")
    ap.add_argument("--variant-name", default="R2M4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-kf", type=int, default=0)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--radius", type=float, default=2.6, help="half-width of the 3D window (m)")
    args = ap.parse_args()

    from osg.core.config import register_configs

    register_configs()
    from analyze_ellipsoids import STOP_CATEGORIES
    from osg.eval import gt_semantic as gts
    from osg.eval.object_recorder import iter_keyframes
    from osg.eval.object_replay import ref_layer_factory, scene_episodes, start_layer
    from osg.objects.point_buffer import clean_points

    import cv2
    import imageio.v2 as imageio

    rec = Path(args.rec)
    ep = scene_episodes(rec, args.scene)[args.episode]
    meta = json.loads((ep / "prior.json").read_text()).get("meta", {})
    cfg_v = compose(BASE + [f"scene_graph.ellipsoid.{x}" for x in args.variant.split()])
    cfg_0 = compose(BASE)
    layer_v = start_layer(cfg_v, ep, layer_factory=ref_layer_factory)
    layer_0 = start_layer(cfg_0, ep, layer_factory=ref_layer_factory)

    layout = gts.find_layouts(cfg_v, {args.scene}).get((args.scene, meta.get("layout_id")))
    if layout is None:
        raise SystemExit(f"no layout {(args.scene, meta.get('layout_id'))}")
    gt_labels = json.loads((ep / "gt.json").read_text())["kf"] if (ep / "gt.json").exists() else None
    inst_file = rec / "_instances" / f"{args.scene}__{meta.get('layout_id')}.json"
    instances = json.loads(inst_file.read_text()) if inst_file.exists() else {}

    kfs = sorted(ep.glob("kf_*.npz"))
    n_kf = len(kfs) if args.max_kf <= 0 else min(len(kfs), args.max_kf)
    sim = None
    fig = plt.figure(figsize=(19.2, 7.2), dpi=100, facecolor="#0d0f13")
    ax_img = fig.add_axes([0.005, 0.12, 0.33, 0.76])
    ax_v = fig.add_axes([0.335, 0.0, 0.33, 0.93], projection="3d")
    ax_0 = fig.add_axes([0.665, 0.0, 0.33, 0.93], projection="3d")
    header = fig.text(0.01, 0.965, "", color="white", fontsize=13, family="monospace")
    fwd_s = None
    seen = set()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=7, macro_block_size=8)
    try:
        for k, (frame, dets, fk) in enumerate(iter_keyframes(ep)):
            if k >= n_kf:
                break
            for i, d in enumerate(dets):
                d.ref = (k, i)
            dets_0 = copy.deepcopy(dets)
            layer_v.update(frame, dets, floor_key=fk)
            layer_0.update(frame, dets_0, floor_key=fk)
            if gt_labels is not None and k < len(gt_labels):
                seen.update(int(iid) for iid, _p in gt_labels[k] if iid is not None)
            if k % args.stride:
                continue
            if sim is None:
                intr = frame.intrinsics
                hfov = math.degrees(2.0 * math.atan(intr.width / (2.0 * intr.fx)))
                sim = gts.make_sim(layout, cfg_v, intr.width, intr.height, hfov)
            rgb = gts.render_rgb_at(sim, frame.T_wc).copy()

            assoc_v = assoc_of(layer_v, k)
            assoc_0 = assoc_of(layer_0, k)
            new_pts = []
            for i, d in enumerate(dets):
                tid = assoc_v.get(i)
                col = colour(tid) if tid is not None else GREY
                bgr = tuple(int(255 * c) for c in col[:3])
                m = np.asarray(d.mask, dtype=np.uint8)
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, contours, -1, bgr, 3 if tid is not None else 1)
                x1, y1 = (int(v) for v in d.bbox_xyxy[:2])
                tag = f"#{tid} {d.label}" if tid is not None else d.label
                cv2.putText(rgb, tag, (x1 + 3, max(y1 + 22, 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(rgb, tag, (x1 + 3, max(y1 + 22, 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            bgr, 2, cv2.LINE_AA)
                if tid is not None:
                    pts = clean_points(d, frame, layer_v.ecfg)
                    if pts is not None:
                        w = pts.world
                        new_pts.append(w[:: max(1, len(w) // 400)])

            cam = frame.camera_position
            fwd = frame.T_wc[:3, 2]
            f2 = np.array([fwd[0], fwd[2]])
            f2 /= max(float(np.linalg.norm(f2)), 1e-6)
            fwd_s = f2 if fwd_s is None else (0.85 * fwd_s + 0.15 * f2)
            fwd_s /= max(float(np.linalg.norm(fwd_s)), 1e-6)
            centre = np.array([cam[0], cam[2]]) + 1.6 * f2
            # Plot coords are (x, -z): the viewer stands behind the camera.
            azim = math.degrees(math.atan2(fwd_s[1], -fwd_s[0]))
            intr = frame.intrinsics
            corners = []
            for u, v in ((0, 0), (intr.width, 0), (0, intr.height), (intr.width, intr.height)):
                ray = np.array([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0]) * 0.45
                corners.append(frame.T_wc[:3, :3] @ ray + cam)
            floor_y = float(cam[1]) - 1.3

            gt_boxes = []
            for iid in seen:
                g = instances.get(str(iid))
                if g is None or str(g["category"]).lower() in STOP_CATEGORIES or max(g["half"]) > 2.5:
                    continue
                c = np.asarray(g["center"], float)
                if (abs(c[0] - centre[0]) <= args.radius and abs(c[2] - centre[1]) <= args.radius
                        and floor_y - 0.3 <= c[1] <= floor_y + 2.8):
                    gt_boxes.append((c, np.asarray(g["half"], float)))

            tv = visible_tracks(layer_v, fk, centre, args.radius)
            t0 = visible_tracks(layer_0, fk, centre, args.radius)
            draw_3d(ax_v, f"{args.variant_name}: buffered points -> ellipsoids", tv,
                    set(assoc_v.values()), new_pts, gt_boxes, cam, corners,
                    (centre, args.radius), floor_y, azim, show_buffer=True)
            draw_3d(ax_0, "V0: moments init + 2D refine (no points kept)", t0,
                    set(assoc_0.values()), new_pts, gt_boxes, cam, corners,
                    (centre, args.radius), floor_y, azim, show_buffer=False)
            ax_img.cla()
            ax_img.imshow(rgb)
            ax_img.set_axis_off()
            nv = sum(1 for t in layer_v._tracks.values() if not t.proposal_only)
            n0 = sum(1 for t in layer_0._tracks.values() if not t.proposal_only)
            header.set_text(
                f"{args.scene}  episode {ep.name.split('__')[-2]}  keyframe {k + 1}/{n_kf}   "
                f"tracks: {args.variant_name} {nv}  V0 {n0}   "
                f"3D-associated so far: {layer_v.funnel.get('assoc_3d', 0)}")
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(img)
            if k % 50 == 0:
                print(f"  kf {k + 1}/{n_kf}", flush=True)
        for _ in range(args.fps * 2):       # hold the last frame
            writer.append_data(img)
    finally:
        writer.close()
        if sim is not None:
            sim.close()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
