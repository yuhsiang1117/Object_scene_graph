#!/usr/bin/env python3
"""One object's ellipsoid, built keyframe by keyframe by two object layers.

scripts/render_ellipsoid_fusion.py shows a whole room; this follows ONE
ground-truth instance through a recorded episode and draws nothing else:

  left    the camera frame (re-rendered at the recorded pose), the object's
          true silhouette dashed, its detection (when there is one) solid
  middle  the variant (default R2M4, docs/history/AB_RESULTS.md S77): the
          object's tracks only -- their buffered depth points, this frame's
          cleaned points in white, their ellipsoids -- the true box dashed, the
          robot's field of view and path
  right   V0, the layer as measured, over the same view

A track is "the object's" when most of its labelled observations are of that
instance (the same rule as scripts/analyze_ellipsoids.py), re-evaluated every
keyframe. Keyframes that detect the object are held `--slow` times; spans
where it is out of sight are fast-forwarded by `--skip`. The video ends on an
orbit around the final state.

    python scripts/render_ellipsoid_focus.py --scene 00873-bxsVRursffK \\
        --instance 95 --out outputs/viz_ellipsoid/focus_00873_pillow95.mp4
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
sys.path.insert(0, str(WORKSPACE / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from render_ellipsoid_fusion import BASE, R2M4, box_edges, compose, ellipse_sections, to_plot

C_VAR = "#ff8c1a"
C_V0 = "#3fa7ff"
C_GT = "#e6e6e6"
C_FOV = "#ffd400"


def dashed_contours(img, mask, colour, thick=2, dash=14, gap=9):
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        pts = c[:, 0, :]
        if len(pts) < 2:
            continue
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        on = (seg % (dash + gap)) < dash
        for i in range(len(pts) - 1):
            if on[i] and on[i + 1]:
                cv2.line(img, tuple(int(v) for v in pts[i]), tuple(int(v) for v in pts[i + 1]),
                         colour, thick, cv2.LINE_AA)


def object_tracks(layer, gt_kf, iid):
    """Tracks most of whose labelled observations are of instance `iid`."""
    out = []
    for t in layer._tracks.values():
        if t.proposal_only or t.blacklisted:
            continue
        ids = []
        for o in t.observations:
            ref = getattr(o, "det_ref", None)
            if ref is None or ref[0] >= len(gt_kf) or ref[1] >= len(gt_kf[ref[0]]):
                continue
            v = gt_kf[ref[0]][ref[1]][0]
            if v is not None:
                ids.append(int(v))
        if ids and Counter(ids).most_common(1)[0][0] == iid:
            out.append(t)
    return out


def half_extents(e):
    return np.array([e.world_extent(np.eye(3)[i]) for i in range(3)])


def iou_aabb(c1, h1, c2, h2):
    lo = np.maximum(c1 - h1, c2 - h2)
    hi = np.minimum(c1 + h1, c2 + h2)
    inter = float(np.prod(np.clip(hi - lo, 0.0, None)))
    union = float(np.prod(2 * h1) + np.prod(2 * h2) - inter)
    return inter / union if union > 0 else 0.0


def stats(tracks, gc, gh, view_dir):
    if not tracks:
        return "no ellipsoid yet"
    t = max(tracks, key=lambda x: len(x.observations))
    e = t.ellipsoid
    err = np.asarray(e.center) - gc
    along = float(err @ view_dir) if view_dir is not None else float("nan")
    return (f"tracks {len(tracks)}   obs {len(t.observations)}   centre error {np.linalg.norm(err):.2f} m "
            f"(depth {along:+.2f})   IoU {iou_aabb(np.asarray(e.center), half_extents(e), gc, gh):.2f}")


def draw_panel(ax, title, colour, tracks, updated, new_pts, show_buffer, gc, gh, cam, corners,
               trail, window, azim, elev, note):
    ax.cla()
    (cx, cz), r, (lo_y, hi_y) = window

    def inside(p):
        p = np.asarray(p, float).reshape(-1, 3)
        return ((np.abs(p[:, 0] - cx) <= r) & (np.abs(p[:, 2] - cz) <= r)
                & (p[:, 1] >= lo_y) & (p[:, 1] <= hi_y))

    def line(p, **kw):
        p = np.asarray(p, float).copy()
        p[~inside(p)] = np.nan
        q = to_plot(p)
        ax.plot(q[:, 0], q[:, 1], q[:, 2], **kw)

    # floor grid for depth cues
    for v in np.linspace(-r, r, 9):
        line(np.array([[cx + v, lo_y, cz - r], [cx + v, lo_y, cz + r]]), color="#2a2e36", lw=0.6)
        line(np.array([[cx - r, lo_y, cz + v], [cx + r, lo_y, cz + v]]), color="#2a2e36", lw=0.6)
    # robot path and field of view
    if len(trail) > 1:
        line(np.asarray(trail), color="#9a8a3a", lw=1.0, alpha=0.8)
    if cam is not None:
        q = to_plot(np.vstack([cam] + corners))
        faces = [[q[0], q[i], q[j]] for i, j in ((1, 2), (2, 4), (4, 3), (3, 1))] + [[q[1], q[2], q[4], q[3]]]
        if inside(np.vstack([cam] + corners)).all():
            ax.add_collection3d(Poly3DCollection(faces, facecolor=C_FOV, alpha=0.07, edgecolor="none"))
        for i in range(1, 5):
            line(np.vstack([cam, corners[i - 1]]), color=C_FOV, lw=1.3)
        line(np.vstack([corners[0], corners[1], corners[3], corners[2], corners[0]]), color=C_FOV, lw=1.3)
        line(np.vstack([cam, cam]), marker="o", color=C_FOV, ms=5)
    # the true object
    for seg in box_edges(gc, gh):
        line(seg, color=C_GT, lw=1.6, ls=(0, (4, 3)))
    # depth points
    if show_buffer:
        for t in tracks:
            buf = getattr(t, "buffer", None)
            if buf is not None and len(buf):
                p = buf.points()
                p = p[inside(p)]
                if len(p):
                    q = to_plot(p)
                    ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=7, color=colour, alpha=0.55,
                               depthshade=False, linewidths=0)
    for p in new_pts:
        p = p[inside(p)]
        if len(p):
            q = to_plot(p)
            ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=6, color="white", alpha=0.85,
                       depthshade=False, linewidths=0)
    # ellipsoids
    for t in tracks:
        lw = 2.6 if t.id in updated else 1.6
        for sec in ellipse_sections(t.ellipsoid, n=72):
            line(sec, color=colour, lw=lw)
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(-cz - r, -cz + r)
    ax.set_zlim(lo_y, hi_y)
    ax.set_box_aspect((2 * r, 2 * r, hi_y - lo_y), zoom=1.35)
    ax.view_init(elev=elev, azim=azim)
    ax.set_facecolor("#14161c")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.08, 0.09, 0.11, 1.0))
        axis.set_ticklabels([])
        axis.line.set_color((0.25, 0.25, 0.25))
    ax.grid(False)
    ax.set_title(title, color=colour, fontsize=15, pad=4, fontweight="bold")
    ax.text2D(0.5, 0.02, note, transform=ax.transAxes, ha="center", color="white", fontsize=11.5,
              family="monospace")


def footprint(e, n=40):
    """The ellipsoid's outline seen from above: hull of its surface samples."""
    from scipy.spatial import ConvexHull

    u, v = np.meshgrid(np.linspace(0, 2 * math.pi, n), np.linspace(0, math.pi, n // 2))
    unit = np.stack([np.cos(u) * np.sin(v), np.cos(v), np.sin(u) * np.sin(v)], axis=-1).reshape(-1, 3)
    pts = (unit * e.axes) @ e.R.T + e.center
    xy = np.stack([pts[:, 0], -pts[:, 2]], axis=1)
    hull = ConvexHull(xy)
    return xy[np.r_[hull.vertices, hull.vertices[0]]]


def draw_map(ax, window, trail, cam, fwd, hfov, fov_m, gc, gh, tv, t0):
    ax.cla()
    (cx, cz), r = window
    ax.set_facecolor("#14161c")
    if len(trail) > 1:
        t = np.asarray(trail)
        ax.plot(t[:, 0], -t[:, 2], color="#9a8a3a", lw=1.0)
    if cam is not None:
        a = math.atan2(-fwd[2], fwd[0])
        ang = np.linspace(a - hfov / 2, a + hfov / 2, 24)
        wedge = np.vstack([[cam[0], -cam[2]],
                           np.stack([cam[0] + fov_m * np.cos(ang), -cam[2] + fov_m * np.sin(ang)], axis=1),
                           [cam[0], -cam[2]]])
        ax.fill(wedge[:, 0], wedge[:, 1], color=C_FOV, alpha=0.18, lw=0)
        ax.plot(wedge[:, 0], wedge[:, 1], color=C_FOV, lw=1.2)
        ax.plot([cam[0]], [-cam[2]], marker="o", color=C_FOV, ms=6)
    x0, x1, z0, z1 = gc[0] - gh[0], gc[0] + gh[0], gc[2] - gh[2], gc[2] + gh[2]
    ax.plot([x0, x1, x1, x0, x0], [-z0, -z0, -z1, -z1, -z0], color=C_GT, lw=1.6, ls=(0, (4, 3)))
    for tracks, c in ((t0, C_V0), (tv, C_VAR)):
        for tr in tracks:
            f = footprint(tr.ellipsoid)
            ax.plot(f[:, 0], f[:, 1], color=c, lw=2.0)
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(-cz - r, -cz + r)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#333842")
    ax.set_title("top view: robot field of view, true outline, both ellipsoids", color="white",
                 fontsize=11.5, pad=4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", default=str(WORKSPACE / "outputs/objrec_v2"))
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", type=int, default=0, help="index in the scene's recorded chain")
    ap.add_argument("--instance", type=int, required=True, help="ground-truth instance id")
    ap.add_argument("--variant", default=R2M4)
    ap.add_argument("--variant-name", default="R2M4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--slow", type=int, default=4, help="frames per keyframe that detects the object")
    ap.add_argument("--skip", type=int, default=2, help="keyframe stride while it is not detected")
    ap.add_argument("--pad", type=int, default=6, help="keyframes before the first / after the last detection")
    ap.add_argument("--fov-m", type=float, default=3.0, help="drawn length of the field of view")
    ap.add_argument("--radius", type=float, default=0.0,
                    help="half-width of the 3D window around the object (m); 0 = from its size")
    args = ap.parse_args()

    from osg.core.config import register_configs

    register_configs()
    import cv2
    import imageio.v2 as imageio

    from osg.eval import gt_semantic as gts
    from osg.eval.object_recorder import iter_keyframes
    from osg.eval.object_replay import ref_layer_factory, scene_episodes, start_layer
    from osg.objects.point_buffer import clean_points

    rec = Path(args.rec)
    ep = scene_episodes(rec, args.scene)[args.episode]
    meta = json.loads((ep / "prior.json").read_text()).get("meta", {})
    gt_kf = json.loads((ep / "gt.json").read_text())["kf"]
    inst = json.loads((rec / "_instances" / f"{args.scene}__{meta.get('layout_id')}.json").read_text())
    g = inst[str(args.instance)]
    gc, gh = np.asarray(g["center"], float), np.asarray(g["half"], float)
    hits = [k for k, kf in enumerate(gt_kf) if any(i == args.instance for i, _ in kf)]
    if not hits:
        raise SystemExit(f"instance {args.instance} is never detected in {ep.name}")
    k0, k1 = max(0, hits[0] - args.pad), min(len(gt_kf) - 1, hits[-1] + args.pad)

    cfg_v = compose(BASE + [f"scene_graph.ellipsoid.{x}" for x in args.variant.split()])
    cfg_0 = compose(BASE)
    layer_v = start_layer(cfg_v, ep, layer_factory=ref_layer_factory)
    layer_0 = start_layer(cfg_0, ep, layer_factory=ref_layer_factory)
    layout = gts.find_layouts(cfg_v, {args.scene})[(args.scene, meta.get("layout_id"))]

    # The map window: the object and every camera pose that saw it.
    cams_seen = []
    for k in hits:
        z = np.load(ep / f"kf_{k:05d}.npz")
        cams_seen.append(z["T_wc"][:3, 3])
    cams_seen = np.asarray(cams_seen)
    view_dir = gc - cams_seen.mean(axis=0)
    view_dir[1] = 0.0                       # the horizontal line of sight
    view_dir /= np.linalg.norm(view_dir)
    mx = 0.5 * (gc[0] + np.median(cams_seen[:, 0]))
    mz = 0.5 * (gc[2] + np.median(cams_seen[:, 2]))
    mr = float(np.clip(np.max(np.hypot(cams_seen[:, 0] - mx, cams_seen[:, 2] - mz)) + 0.8, 2.0, 5.0))
    map_window = ((mx, mz), mr)
    # The 3D window: the object alone.
    r = args.radius if args.radius > 0 else max(1.6, 3.0 * float(max(gh)) + 0.6)
    floor_y = float(gc[1] - gh[1])
    window = ((float(gc[0]), float(gc[2])), r, (floor_y - 0.05, float(gc[1]) + max(1.0, 3.0 * float(gh[1]))))
    # Side-on to the line of sight: a depth error reads as a sideways shift.
    azim0 = math.degrees(math.atan2(view_dir[0], view_dir[2])) + 20.0
    elev0 = 18.0

    fig = plt.figure(figsize=(19.2, 8.4), dpi=100, facecolor="#0d0f13")
    ax_img = fig.add_axes([0.005, 0.43, 0.33, 0.47])
    ax_map = fig.add_axes([0.055, 0.06, 0.23, 0.29])
    # The zoomed 3D panes spill past their axes; keep the 2D panels on top.
    ax_img.set_zorder(10)
    ax_map.set_zorder(10)
    ax_v = fig.add_axes([0.335, 0.06, 0.33, 0.84], projection="3d")
    ax_0 = fig.add_axes([0.665, 0.06, 0.33, 0.84], projection="3d")
    head = fig.text(0.01, 0.955, "", color="white", fontsize=15, family="monospace")
    sub = fig.text(0.01, 0.918, "", color="#bbbbbb", fontsize=12, family="monospace")
    fig.text(0.345, 0.025,
             "dashed = true object   white dots = this keyframe's depth points of it   "
             f"orange dots = points {args.variant_name} keeps\nyellow = robot field of view and path   "
             "depth = centre error along the robot's line of sight (+ = behind the object)",
             color="#cfcfcf", fontsize=11.5, family="monospace", linespacing=1.4)
    img_note = fig.text(0.17, 0.41, "", color="white", fontsize=12, ha="center", family="monospace")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    sim = None
    trail = []
    last = None
    n_det = 0
    try:
        for k, (frame, dets, fk) in enumerate(iter_keyframes(ep)):
            if k > k1:
                break
            for i, d in enumerate(dets):
                d.ref = (k, i)
            dets_0 = copy.deepcopy(dets)
            layer_v.update(frame, dets, floor_key=fk)
            layer_0.update(frame, dets_0, floor_key=fk)
            cam = frame.camera_position.copy()
            if k < k0:
                continue
            trail.append(cam)
            mine = [i for i, (iid, _p) in enumerate(gt_kf[k]) if iid == args.instance]
            if not mine and (k - k0) % args.skip:
                continue
            n_det += bool(mine)
            if sim is None:
                intr = frame.intrinsics
                hfov = math.degrees(2.0 * math.atan(intr.width / (2.0 * intr.fx)))
                sim = gts.make_sim(layout, cfg_v, intr.width, intr.height, hfov)
            rgb = gts.render_rgb_at(sim, frame.T_wc).copy()
            _depth, sem = gts.render_at(sim, frame.T_wc)
            visible = int((sem == args.instance).sum())
            dashed_contours(rgb, sem == args.instance, (235, 235, 235), thick=3)
            new_pts = []
            for i in mine:
                d = dets[i]
                m = np.asarray(d.mask, dtype=np.uint8)
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, contours, -1, (255, 140, 26), 3)
                x1, y1 = (int(v) for v in d.bbox_xyxy[:2])
                cv2.putText(rgb, f"detected: {d.label} {d.score:.2f}", (x1, max(y1 - 10, 24)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(rgb, f"detected: {d.label} {d.score:.2f}", (x1, max(y1 - 10, 24)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 140, 26), 2, cv2.LINE_AA)
                pts = clean_points(d, frame, layer_v.ecfg)
                if pts is not None:
                    new_pts.append(pts.world[:: max(1, len(pts.world) // 500)])

            intr = frame.intrinsics
            corners = []
            for u, v in ((0, 0), (intr.width, 0), (0, intr.height), (intr.width, intr.height)):
                ray = np.array([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0])
                corners.append(frame.T_wc[:3, :3] @ ray * args.fov_m + cam)

            tv = object_tracks(layer_v, gt_kf, args.instance)
            t0 = object_tracks(layer_0, gt_kf, args.instance)
            upd_v = {t.id for t in tv if t.observations and getattr(t.observations[-1], "det_ref", (None,))[0] == k}
            upd_0 = {t.id for t in t0 if t.observations and getattr(t.observations[-1], "det_ref", (None,))[0] == k}
            draw_panel(ax_v, f"{args.variant_name}: depth points -> ellipsoid", C_VAR, tv, upd_v, new_pts,
                       True, gc, gh, cam, corners, trail, window, azim0, elev0, stats(tv, gc, gh, view_dir))
            draw_panel(ax_0, "V0 (baseline): mask moments + 2D refine", C_V0, t0, upd_0, new_pts,
                       False, gc, gh, cam, corners, trail, window, azim0, elev0, stats(t0, gc, gh, view_dir))
            hfov_r = 2.0 * math.atan(intr.width / (2.0 * intr.fx))
            draw_map(ax_map, map_window, trail, cam, frame.T_wc[:3, 2], hfov_r, args.fov_m, gc, gh, tv, t0)
            ax_img.cla()
            ax_img.imshow(rgb)
            ax_img.set_axis_off()
            head.set_text(f"{g['category']} (true size {2*gh[0]:.2f} x {2*gh[1]:.2f} x {2*gh[2]:.2f} m)   "
                          f"{args.scene}  episode {ep.name.split('__')[-2]}   keyframe {k}")
            sub.set_text(f"detections of this object so far: {n_det}/{len(hits)}   "
                         + (f"slow motion (1/{args.slow})" if mine else f"fast-forward (x{args.skip}): not detected"))
            img_note.set_text("robot camera view" + ("" if visible else "  (object out of view)"))
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            for _ in range(args.slow if mine else 1):
                writer.append_data(img)
            last = (tv, t0, upd_v, upd_0)
            if k % 50 == 0:
                print(f"  kf {k}/{k1}", flush=True)
        # Orbit around the final state, the camera gone.
        tv, t0, _, _ = last
        sub.set_text("final ellipsoids, seen from around the object")
        img_note.set_text("")
        draw_map(ax_map, map_window, trail, None, None, 0.0, 0.0, gc, gh, tv, t0)
        n_orbit = args.fps * 6
        for j in range(n_orbit):
            a = azim0 + 140.0 * math.sin(math.pi * j / n_orbit)
            draw_panel(ax_v, f"{args.variant_name}: depth points -> ellipsoid", C_VAR, tv, set(), [], True,
                       gc, gh, None, [], trail, window, a, elev0, stats(tv, gc, gh, view_dir))
            draw_panel(ax_0, "V0 (baseline): mask moments + 2D refine", C_V0, t0, set(), [], False,
                       gc, gh, None, [], trail, window, a, elev0, stats(t0, gc, gh, view_dir))
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    finally:
        writer.close()
        if sim is not None:
            sim.close()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
