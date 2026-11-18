"""Everything `ObjectLayer.update` was given, written down, so it can be given again.

The ellipsoid work (docs/history/AB_RESULTS.md, "ellipsoid" sections) needs to
try a dozen object-layer variants on the SAME keyframes. Pass 1 is driven by
ASCENT, whose trajectory does not depend on the OSG object layer at all, so a
replay of a pass-1 recording through a different layer is not an approximation
of running that layer -- it IS that run, keyframe for keyframe.

Strictly one-way, like eval/gt_dump.py: this writes files and hands the layer
nothing back. One directory per episode:

    prior.json      the layer's tracks at its first update (what apply_map
                    restored), plus the layer state a replay must start from
    kf_00000.npz    one per update(): frame id, depth (float32, exact), pose,
                    intrinsics, floor key, and every detection -- label, score,
                    bbox (its own dtype: the size gate's arithmetic depends on
                    it), source, packed mask, crop as PNG
    final.json      the layer's tracks when the episode's last update is done,
                    which a replay of the unchanged layer must reproduce exactly

Read back with `load_keyframe` / `iter_keyframes`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from ..core.types import CameraIntrinsics, Detection, FrameData


def _layer_state(layer) -> dict:
    return {
        "target_label": str(layer.target_label),
        "keep_cloud_labels": sorted(str(x) for x in layer.keep_cloud_labels),
        "next_id": int(layer._next_id),
        "next_prop_id": int(layer._next_prop_id),
        "has_proposal_text": layer.proposal_text is not None,
        "disabled_pts": [[float(v) for v in xy] + [str(lbl)] for xy, lbl in layer._disabled_pts],
    }


def _tracks_blob(layer) -> dict:
    from ..graph.map_store import _track_record

    return {
        "state": _layer_state(layer),
        "funnel": dict(layer.funnel),
        "tracks": [
            dict(_track_record(t), proposal_only=bool(t.proposal_only),
                 from_prior=bool(getattr(t, "from_prior", False)))
            for t in layer._tracks.values()
        ],
    }


class ObjectFrameRecorder:
    """Attach as `layer.recorder`; `ObjectLayer.update` calls `write`."""

    def __init__(self, root: str, tag: str, meta: Optional[dict] = None) -> None:
        self.dir = Path(root) / tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta or {})
        self.n = 0
        self._closed = False

    def write(self, layer, frame: FrameData, dets: List[Detection], floor_key: int) -> None:
        import cv2

        if self.n == 0:
            blob = _tracks_blob(layer)
            blob["meta"] = self.meta
            (self.dir / "prior.json").write_text(json.dumps(blob))
        dets = list(dets or [])
        h, w = frame.depth.shape
        crops, offsets = [], [0]
        for d in dets:
            buf = b""
            if d.crop is not None and d.crop.size:
                ok, enc = cv2.imencode(".png", d.crop)
                buf = enc.tobytes() if ok else b""
            crops.append(buf)
            offsets.append(offsets[-1] + len(buf))
        intr = frame.intrinsics
        np.savez_compressed(
            self.dir / f"kf_{self.n:05d}.npz",
            frame_id=np.int64(frame.frame_id),
            depth=np.asarray(frame.depth),
            T_wc=np.asarray(frame.T_wc),
            intrinsics=np.array([intr.fx, intr.fy, intr.cx, intr.cy, intr.width, intr.height],
                                dtype=np.float64),
            rgb_shape=np.array(frame.rgb.shape, dtype=np.int64),
            floor_key=np.int64(floor_key),
            labels=np.array(json.dumps([str(d.label) for d in dets])),
            scores=np.array([float(d.score) for d in dets], dtype=np.float64),
            bboxes=(np.stack([np.asarray(d.bbox_xyxy) for d in dets])
                    if dets else np.zeros((0, 4), dtype=np.float32)),
            sources=np.array(json.dumps([str(getattr(d, "source", "detector")) for d in dets])),
            masks=(np.stack([np.packbits(np.asarray(d.mask, dtype=bool).reshape(-1)) for d in dets])
                   if dets else np.zeros((0, (h * w + 7) // 8), dtype=np.uint8)),
            mask_shape=np.array([h, w], dtype=np.int64),
            crops=np.frombuffer(b"".join(crops), dtype=np.uint8),
            crop_offsets=np.array(offsets, dtype=np.int64),
        )
        self.n += 1

    def close(self, layer) -> None:
        if self._closed:
            return
        self._closed = True
        blob = _tracks_blob(layer)
        blob["meta"] = self.meta
        blob["n_keyframes"] = self.n
        (self.dir / "final.json").write_text(json.dumps(blob))


# ------------------------------------------------------------------ reading


def load_keyframe(path: Path):
    """(FrameData, [Detection], floor_key) exactly as `update` received them,
    except `rgb`, which is a zero image of the recorded shape: nothing in the
    object layer reads its pixels (crops are recorded), only its shape."""
    import cv2

    z = np.load(path)
    fx, fy, cx, cy, w, h = (float(v) for v in z["intrinsics"])
    intr = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=int(w), height=int(h))
    rgb = np.zeros(tuple(int(v) for v in z["rgb_shape"]), dtype=np.uint8)
    frame = FrameData(frame_id=int(z["frame_id"]), rgb=rgb, depth=z["depth"],
                      T_wc=z["T_wc"], intrinsics=intr)
    labels = json.loads(str(z["labels"]))
    sources = json.loads(str(z["sources"]))
    mh, mw = (int(v) for v in z["mask_shape"])
    crops, offs = z["crops"], z["crop_offsets"]
    dets = []
    for i, label in enumerate(labels):
        mask = np.unpackbits(z["masks"][i], count=mh * mw).reshape(mh, mw).astype(bool)
        raw = crops[offs[i]:offs[i + 1]]
        crop = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED) if len(raw) else None
        det = Detection(label=label, score=float(z["scores"][i]),
                        bbox_xyxy=z["bboxes"][i], mask=mask, crop=crop)
        det.source = sources[i]
        dets.append(det)
    return frame, dets, int(z["floor_key"])


def iter_keyframes(episode_dir: Path) -> Iterator[tuple]:
    for path in sorted(Path(episode_dir).glob("kf_*.npz")):
        yield load_keyframe(path)
