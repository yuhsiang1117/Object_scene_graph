"""HOV-SG's sequential mask merge, faster, with the same result.

HOV-SG's `seq_merge` calls `merge_3d_masks` on every accumulated mask plus the new frame's, once
per frame. On the DualMap sequences (453-613 frames after HOV-SG's skip of 10) that is ~45 s per
frame by frame 76 of 00829 and growing -- about a day per scene. Two things are redone each frame
that need not be, and both are replaced by exact equivalents:

  * the pairwise box-IoU screen, a Python double loop over `compute_3d_bbox_iou`: computed with
    numpy one row at a time, the same float64 operations in the same order, so the same pairs pass
    (`check_equivalence`, bit for bit on random masks);
  * the two expensive calls, FAISS overlap ratios of unchanged pairs and `pcd_denoise_dbscan` of
    unchanged masks (every mask is re-denoised as a one-mask component): both are deterministic
    functions of the point and colour arrays, so they are cached by content (`MergeCache`).

Everything else -- thresholds, connected components, the order masks are concatenated in -- is
HOV-SG's. `verify_on_frames` runs HOV-SG's own `seq_merge` and this one on dumped real frames and
compares the masks point for point (00829, 40 frames: identical; 281 s -> 144 s).

    import fast_merge; fast_merge.install()   # before HOV-SG's graph module is imported
"""
from __future__ import annotations

import numpy as np


def _bounds(mask_list):
    boxes = [pcd.get_axis_aligned_bounding_box() for pcd in mask_list]
    mins = np.array([np.asarray(b.get_min_bound()) for b in boxes], dtype=np.float64).reshape(-1, 3)
    maxs = np.array([np.asarray(b.get_max_bound()) for b in boxes], dtype=np.float64).reshape(-1, 3)
    return mins, maxs


def _row_iou(mins, maxs, i):
    """compute_3d_bbox_iou(box i, box j) for every j > i, padding 0."""
    omin = np.maximum(mins[i], mins[i + 1:])
    omax = np.minimum(maxs[i], maxs[i + 1:])
    size = np.maximum(omax - omin, 0.0)
    # np.prod over a length-3 vector, written out in its left-to-right order
    overlap = size[:, 0] * size[:, 1] * size[:, 2]
    e1 = maxs[i] - mins[i]
    v1 = e1[0] * e1[1] * e1[2]
    e2 = maxs[i + 1:] - mins[i + 1:]
    v2 = e2[:, 0] * e2[:, 1] * e2[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return overlap / (v1 + v2 - overlap)


def overlap_matrix_fast(mask_list, radius, iou_thresh):
    from hovsg.utils.graph_utils import find_overlapping_ratio_faiss

    n = len(mask_list)
    matrix = np.zeros((n, n))
    if n == 0:
        return matrix
    mins, maxs = _bounds(mask_list)
    for i in range(n - 1):
        for k in np.nonzero(_row_iou(mins, maxs, i) > iou_thresh)[0]:
            j = i + 1 + int(k)
            matrix[i, j] = find_overlapping_ratio_faiss(mask_list[i], mask_list[j], radius=1.5 * radius)
    return matrix


def overlap_matrix_reference(mask_list, radius, iou_thresh):
    """The loop of hovsg.utils.graph_utils.merge_3d_masks, verbatim."""
    from hovsg.utils.graph_utils import compute_3d_bbox_iou, find_overlapping_ratio_faiss

    aa_bb = [pcd.get_axis_aligned_bounding_box() for pcd in mask_list]
    overlap_matrix = np.zeros((len(mask_list), len(mask_list)))
    for i in range(len(mask_list)):
        for j in range(i + 1, len(mask_list)):
            if compute_3d_bbox_iou(aa_bb[i], aa_bb[j]) > iou_thresh:
                overlap_matrix[i, j] = find_overlapping_ratio_faiss(mask_list[i], mask_list[j], radius=1.5 * radius)
    return overlap_matrix


def merge_3d_masks(mask_list, overlap_threshold=0.5, radius=0.02, iou_thresh=0.05):
    """hovsg.utils.graph_utils.merge_3d_masks with the vectorised screen."""
    from scipy.sparse.csgraph import connected_components
    from hovsg.utils.graph_utils import merge_point_clouds_list

    overlap_matrix = overlap_matrix_fast(mask_list, radius, iou_thresh)
    if overlap_matrix.size == 0:
        return mask_list
    graph = overlap_matrix > overlap_threshold
    n_components, component_labels = connected_components(graph)
    component_indices = [np.where(component_labels == i)[0] for i in range(n_components)]
    pcd_list_merged = []
    for indices in component_indices:
        pcd_list_merged.append(merge_point_clouds_list([mask_list[i] for i in indices], voxel_size=0.5 * radius))
    return pcd_list_merged


def check_equivalence(n=300, seed=0, spread=3.0):
    """Random masks; the IoU of every pair, and the two overlap matrices, must be identical."""
    import open3d as o3d
    from hovsg.utils.graph_utils import compute_3d_bbox_iou

    rng = np.random.default_rng(seed)
    masks = []
    for _ in range(n):
        centre = rng.uniform(-spread, spread, 3)
        extent = rng.uniform(0.02, 0.6, 3)
        if rng.random() < 0.05:
            extent[rng.integers(3)] = 0.0  # a flat box: the 0/0 case
        pts = centre + rng.uniform(-0.5, 0.5, (int(rng.integers(5, 200)), 3)) * extent
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        masks.append(pcd)
    boxes = [m.get_axis_aligned_bounding_box() for m in masks]
    mins, maxs = _bounds(masks)
    screened = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(n - 1):
            ref_iou = np.array([compute_3d_bbox_iou(boxes[i], boxes[j]) for j in range(i + 1, n)])
            fast_iou = _row_iou(mins, maxs, i)
            assert np.array_equal(ref_iou, fast_iou, equal_nan=True), f"row {i}: IoU differs"
            screened += int(np.sum(ref_iou > 0.05))
        ref = overlap_matrix_reference(masks, 0.02, 0.05)
    fast = overlap_matrix_fast(masks, 0.02, 0.05)
    assert np.array_equal(ref, fast), f"{np.sum(ref != fast)} entries differ"
    return screened, int(np.count_nonzero(ref))


class MergeCache:
    """Results of HOV-SG's two expensive pure functions, keyed by the content of their inputs.

    `seq_merge` re-runs `merge_3d_masks` on every accumulated mask at every frame: each unchanged
    pair's FAISS overlap ratio is recomputed, and every mask -- merged or not -- goes back through
    `pcd_denoise_dbscan` (merge_point_clouds_list of a one-mask component). Both are deterministic
    functions of the point and colour arrays, so a repeat of the same input returns the stored
    output. Entries not used in the latest call are dropped.
    """

    def __init__(self):
        self.overlap, self.denoise = {}, {}
        self._overlap_used, self._denoise_used = set(), set()
        self.hits = self.misses = 0

    @staticmethod
    def key(pcd):
        import hashlib

        h = hashlib.blake2b(digest_size=16)
        p = np.ascontiguousarray(np.asarray(pcd.points))
        c = np.ascontiguousarray(np.asarray(pcd.colors))
        h.update(np.int64(p.shape[0]).tobytes())
        h.update(p.tobytes())
        h.update(np.int64(c.shape[0]).tobytes())
        h.update(c.tobytes())
        return h.digest()

    def overlap_ratio(self, ki, kj, a, b, radius):
        from hovsg.utils.graph_utils import find_overlapping_ratio_faiss

        k = (ki, kj, radius)
        self._overlap_used.add(k)
        if k in self.overlap:
            self.hits += 1
            return self.overlap[k]
        self.misses += 1
        v = self.overlap[k] = find_overlapping_ratio_faiss(a, b, radius=radius)
        return v

    def denoised(self, pcd, eps, min_points):
        import open3d as o3d
        from hovsg.utils.graph_utils import pcd_denoise_dbscan

        k = (self.key(pcd), eps, min_points)
        self._denoise_used.add(k)
        if k not in self.denoise:
            self.misses += 1
            out = pcd_denoise_dbscan(pcd, eps=eps, min_points=min_points)
            self.denoise[k] = (np.asarray(out.points).copy(), np.asarray(out.colors).copy())
        else:
            self.hits += 1
        points, colors = self.denoise[k]
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(points)
        if len(colors):
            out.colors = o3d.utility.Vector3dVector(colors)
        return out

    def end_call(self):
        self.overlap = {k: v for k, v in self.overlap.items() if k in self._overlap_used}
        self.denoise = {k: v for k, v in self.denoise.items() if k in self._denoise_used}
        self._overlap_used, self._denoise_used = set(), set()


def merge_3d_masks_cached(mask_list, cache, overlap_threshold=0.5, radius=0.02, iou_thresh=0.05):
    """merge_3d_masks with the vectorised screen and the content cache; same masks out."""
    import open3d as o3d
    from scipy.sparse.csgraph import connected_components

    n = len(mask_list)
    overlap_matrix = np.zeros((n, n))
    if n:
        keys = [cache.key(m) for m in mask_list]
        mins, maxs = _bounds(mask_list)
        for i in range(n - 1):
            for k in np.nonzero(_row_iou(mins, maxs, i) > iou_thresh)[0]:
                j = i + 1 + int(k)
                overlap_matrix[i, j] = cache.overlap_ratio(keys[i], keys[j], mask_list[i], mask_list[j], 1.5 * radius)
    if overlap_matrix.size == 0:
        cache.end_call()
        return mask_list
    graph = overlap_matrix > overlap_threshold
    n_components, component_labels = connected_components(graph)
    merged = []
    for c in range(n_components):
        indices = np.where(component_labels == c)[0]
        # merge_point_clouds_list: the first mask += the others, in order, then denoised
        pcd = o3d.geometry.PointCloud(mask_list[indices[0]])
        for i in indices[1:]:
            pcd += mask_list[i]
        merged.append(cache.denoised(pcd, 0.1, 10))
    cache.end_call()
    return merged


def seq_merge_cached(frames_pcd, th, down_size, proxy_th, cache=None):
    """hovsg.utils.graph_utils.seq_merge over merge_3d_masks_cached."""
    from tqdm import tqdm

    cache = cache or MergeCache()
    global_masks = frames_pcd[0]
    for i in tqdm(range(1, len(frames_pcd))):
        global_masks = merge_3d_masks_cached(global_masks + frames_pcd[i], cache, overlap_threshold=th,
                                             radius=down_size, iou_thresh=proxy_th)
    return merge_3d_masks_cached(global_masks, cache, overlap_threshold=th, radius=down_size, iou_thresh=proxy_th)


def same_masks(a, b):
    if len(a) != len(b):
        return False
    return all(np.array_equal(np.asarray(x.points), np.asarray(y.points))
               and np.array_equal(np.asarray(x.colors), np.asarray(y.colors)) for x, y in zip(a, b))


def verify_on_frames(path, frames=40, th=0.75, down_size=0.02, proxy_th=0.05):
    """HOV-SG's seq_merge and seq_merge_cached on the first `frames` dumped frames: same masks?"""
    import time
    from hovsg.utils import graph_utils  # unpatched: install() is not called here

    data = load_frames(path)[:frames]
    t0 = time.time()
    # HOV-SG's merge mutates its input masks, so it gets its own copy
    ref = graph_utils.seq_merge(load_frames(path)[:frames], th, down_size, proxy_th)
    t1 = time.time()
    cache = MergeCache()
    got = seq_merge_cached(data, th, down_size, proxy_th, cache)
    t2 = time.time()
    print(f"{frames} frames: HOV-SG {t1 - t0:.1f} s, cached {t2 - t1:.1f} s "
          f"(cache hits {cache.hits}, misses {cache.misses}); identical: {same_masks(ref, got)}")
    return same_masks(ref, got)


def dump_frames(frames_pcd, path):
    """Each frame's 3D masks (points, colours) as HOV-SG hands them to its merge."""
    import pickle

    frames = [[(np.asarray(m.points).copy(), np.asarray(m.colors).copy()) for m in frame] for frame in frames_pcd]
    with open(path, "wb") as fh:
        pickle.dump(frames, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_frames(path):
    import pickle
    import open3d as o3d

    with open(path, "rb") as fh:
        frames = pickle.load(fh)
    out = []
    for frame in frames:
        masks = []
        for points, colors in frame:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if len(colors):
                pcd.colors = o3d.utility.Vector3dVector(colors)
            masks.append(pcd)
        out.append(masks)
    return out


def install():
    import os
    import hovsg.utils.graph_utils as graph_utils

    graph_utils.merge_3d_masks = merge_3d_masks  # hierarchical_merge looks it up here
    dump = os.environ.get("HOVSG_FRAMES_DUMP")

    def seq_merge(frames_pcd, th, down_size, proxy_th):
        if dump:
            dump_frames(frames_pcd, dump)
            print(f"[fast_merge] dumped {len(frames_pcd)} frames of masks to {dump}", flush=True)
            if os.environ.get("HOVSG_DUMP_ONLY") == "1":
                raise SystemExit(0)
        cache = MergeCache()
        out = seq_merge_cached(frames_pcd, th, down_size, proxy_th, cache)
        print(f"[fast_merge] cached sequential merge: {len(out)} masks, cache hits {cache.hits}, "
              f"misses {cache.misses}", flush=True)
        return out

    graph_utils.seq_merge = seq_merge  # graph.py imports the name after install()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:  # python fast_merge.py <dumped frames.pkl> [frames]
        ok = verify_on_frames(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
        raise SystemExit(0 if ok else 1)
    for seed, spread in [(0, 3.0), (1, 3.0), (2, 1.0), (3, 0.5), (4, 0.3)]:
        screened, overlapping = check_equivalence(seed=seed, spread=spread)
        print(f"seed {seed} spread {spread}: identical; {screened} pairs pass the IoU screen, "
              f"{overlapping} overlap")
