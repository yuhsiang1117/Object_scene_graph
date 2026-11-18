"""IP-Basic depth completion, vendored from ASCENT's `depth_camera_filtering`.

ASCENT runs `filter_depth(depth, blur_type=None)` on every frame before the
depth reaches its obstacle, object and value maps (`ascent_policy.py:230`).
The PointNav policy gets the raw depth. OSG's port fed the raw depth to the
maps as well, and the audit found the consequence: a habitat depth pixel of
exactly 0 (no hit, or nearer than `min_depth`) contributed NO obstacle in OSG
where ASCENT's dilation fill makes it a real one, and mask pixels closer than
0.5 m never entered the object cloud.

Because habitat's depth is normalised to [0, 1] and `recover_nonzero=True`
restores every non-zero pixel, the net effect on this input is exactly: fill
the zeros from their neighbours, touch nothing else. The `> 0.1` "valid"
tests inside only shape what the fill values are.

Source: relative_work/ascent/third_party/depth_camera_filtering/
depth_camera_filtering/filtering.py (numpy + cv2 only). Kept verbatim apart
from dropping the uint8 wrapper and the process-dict debug output.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)

CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=np.uint8,
)

DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=np.uint8,
)

CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    dtype=np.uint8,
)


def filter_depth(
    depth_img: np.ndarray,
    clip_far_thresh: Optional[float] = None,
    set_black_value: Optional[float] = None,
    use_multiscale: bool = True,
    recover_nonzero: bool = True,
    **kwargs,
) -> np.ndarray:
    """Filter a float depth image with a multiscale dilation."""
    assert np.issubdtype(depth_img.dtype, np.floating), "depth_img must be np.float32"
    assert depth_img.ndim == 2, "depth_img must be 2D"
    nonzero_mask = (depth_img != 0) if recover_nonzero else None
    if use_multiscale:
        filtered = fill_in_multiscale(depth_img, **kwargs)
    else:
        filtered = fill_in_fast(depth_img, **kwargs)
    if nonzero_mask is not None:
        # Recover pixels that weren't black before but were turned black by filtering
        filtered[nonzero_mask] = depth_img[nonzero_mask]
    if clip_far_thresh is not None:
        filtered = np.clip(filtered, 0, clip_far_thresh)
    if set_black_value is not None:
        filtered[filtered == 0] = set_black_value
    return filtered


def fill_in_fast(
    depth_map: np.ndarray,
    max_depth: float = 100.0,
    custom_kernel: np.ndarray = DIAMOND_KERNEL_5,
    extrapolate: bool = False,
    blur_type: Optional[str] = "bilateral",
) -> np.ndarray:
    """Fast, in-place depth completion."""
    valid_pixels = depth_map > 0.1
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    depth_map = cv2.dilate(depth_map, custom_kernel)
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    empty_pixels = depth_map < 0.1
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]
        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0: top_row_pixels[pixel_col_idx], pixel_col_idx] = top_pixel_values[pixel_col_idx]
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    depth_map = cv2.medianBlur(depth_map, 5)

    if blur_type == "bilateral":
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == "gaussian":
        valid_pixels = depth_map > 0.1
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    valid_pixels = depth_map > 0.1
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
    return depth_map


def fill_in_multiscale(
    depth_map: np.ndarray,
    max_depth: float = 100.0,
    dilation_kernel_far: np.ndarray = CROSS_KERNEL_3,
    dilation_kernel_med: np.ndarray = CROSS_KERNEL_5,
    dilation_kernel_near: np.ndarray = CROSS_KERNEL_7,
    extrapolate: bool = False,
    blur_type: Optional[str] = "bilateral",
) -> np.ndarray:
    """Slower, multi-scale dilation with additional noise removal."""
    depths_in = np.float32(depth_map)

    valid_pixels_near = (depths_in > 0.1) & (depths_in <= 15.0)
    valid_pixels_med = (depths_in > 15.0) & (depths_in <= 30.0)
    valid_pixels_far = depths_in > 30.0

    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = s1_inverted_depths > 0.1
    s1_inverted_depths[valid_pixels] = max_depth - s1_inverted_depths[valid_pixels]

    dilated_far = cv2.dilate(np.multiply(s1_inverted_depths, valid_pixels_far), dilation_kernel_far)
    dilated_med = cv2.dilate(np.multiply(s1_inverted_depths, valid_pixels_med), dilation_kernel_med)
    dilated_near = cv2.dilate(np.multiply(s1_inverted_depths, valid_pixels_near), dilation_kernel_near)

    valid_pixels_near = dilated_near > 0.1
    valid_pixels_med = dilated_med > 0.1
    valid_pixels_far = dilated_far > 0.1

    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    s3_closed_depths = cv2.morphologyEx(s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = s3_closed_depths > 0.1
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    top_mask = np.ones(depths_in.shape, dtype=bool)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.1)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    valid_pixels = s4_blurred_depths > 0.1
    empty_pixels = ~valid_pixels & top_mask

    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=bool)
    top_row_pixels = np.argmax(s5_dilated_depths > 0.1, axis=0)
    top_pixel_values = s5_dilated_depths[top_row_pixels, range(s5_dilated_depths.shape[1])]
    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[0: top_row_pixels[pixel_col_idx], pixel_col_idx] = top_pixel_values[pixel_col_idx]
        else:
            top_mask[0: top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    s7_blurred_depths = np.copy(s6_extended_depths)
    for _ in range(6):
        empty_pixels = (s7_blurred_depths < 0.1) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.1) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == "gaussian":
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.1) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == "bilateral":
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.1)
    s8_inverted_depths[valid_pixels] = max_depth - s8_inverted_depths[valid_pixels]
    return s8_inverted_depths
